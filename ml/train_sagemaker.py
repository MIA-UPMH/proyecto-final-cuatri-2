#!/usr/bin/env python3
"""
SageMaker-compatible training script for the Respiratory CNN.

SageMaker passes hyperparameters as CLI arguments (--epochs 12, etc.) and
provides data via environment variables:
  SM_CHANNEL_TRAINING  -> /opt/ml/input/data/training
  SM_CHANNEL_VALIDATION -> /opt/ml/input/data/validation
  SM_MODEL_DIR         -> /opt/ml/model
  SM_OUTPUT_DATA_DIR   -> /opt/ml/output/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report
import numpy as np


# ─────────────────────────────────────────────
# Model definition (mirrors train.py)
# ─────────────────────────────────────────────
class SimpleRespiratoryCNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        # After 3x MaxPool2d(2): 224->112->56->28
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def build_loaders(train_dir: Path, val_dir: Path, image_size: int, batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return train_loader, val_loader, train_ds.classes


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_sagemaker] Device: {device}")

    train_dir = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    val_dir = Path(os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    print(f"[train_sagemaker] Training data : {train_dir}")
    print(f"[train_sagemaker] Validation data: {val_dir}")
    print(f"[train_sagemaker] Model dir      : {model_dir}")
    print(f"[train_sagemaker] Output dir     : {output_dir}")
    print(f"[train_sagemaker] Hyperparams    : {vars(args)}")

    # Validate data dirs
    if not train_dir.exists():
        print(f"[train_sagemaker] ERROR: Training dir not found: {train_dir}")
        sys.exit(1)
    if not val_dir.exists():
        print(f"[train_sagemaker] WARNING: Validation dir not found: {val_dir}, using train dir for validation")
        val_dir = train_dir

    train_loader, val_loader, class_names = build_loaders(
        train_dir, val_dir, args.image_size, args.batch_size
    )
    print(f"[train_sagemaker] Classes: {class_names}")
    print(f"[train_sagemaker] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    num_classes = len(class_names)
    model = SimpleRespiratoryCNN(num_classes=num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch}/{args.epochs} batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={running_loss/total:.4f} acc={correct/total:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # ── Validate ────────────────────────────
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_acc = val_correct / val_total
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(macro_f1, 4),
        }
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs} — loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} macro_f1={macro_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "best_model.pth")
            print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

    # ── Save final model ────────────────────────
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "final_model.pth")

    # Classification report on val set with best model
    print(f"\nLoading best model (epoch {best_epoch}) for final report...")
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=device))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    final_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    final_val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report (best model):\n", report)

    # ── Save metadata ───────────────────────────
    metadata = {
        "class_names": class_names,
        "image_size": args.image_size,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 4),
        "final_val_accuracy": round(final_val_acc, 4),
        "final_macro_f1": round(final_macro_f1, 4),
        "epochs_completed": args.epochs,
        "history": history,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Output metrics for SageMaker (picked up by pipeline.py after download)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[train_sagemaker] Done. Best val_acc={best_val_acc:.4f} macro_f1={final_macro_f1:.4f}")
    print(f"[train_sagemaker] Model saved to {model_dir}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Respiratory CNN SageMaker training script")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)

    args = parser.parse_args()
    train(args)
