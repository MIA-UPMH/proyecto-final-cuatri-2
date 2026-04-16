#!/usr/bin/env python3
"""
experiment.py — Script standalone de entrenamiento y evaluación CNN
Clasificación de Enfermedades Respiratorias: COVID / NEUMONIA / NORMAL

Uso:
    python experiment.py --s3-bucket corte3-cnn-artifacts-123581233269
    python experiment.py --data-dir /ruta/local/dataset --epochs 20

Genera en --output-dir (default: ./outputs/):
    fig1_class_distribution.png
    fig2_training_curves.png
    fig3_confusion_matrix.png
    fig4_roc_curves.png
    fig5_sample_predictions.png
    fig6_metrics_table.png
    training_log.txt
    metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

# ─── Check dependencies ────────────────────────────────────────────────────────
MISSING = []
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError:
    MISSING.append("torch torchvision")

try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    MISSING.append("matplotlib")

try:
    import seaborn as sns
except ImportError:
    MISSING.append("seaborn")

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, f1_score
    )
    from sklearn.preprocessing import label_binarize
except ImportError:
    MISSING.append("scikit-learn")

try:
    from tqdm import tqdm
except ImportError:
    MISSING.append("tqdm")

if MISSING:
    print("Instalando dependencias faltantes...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + " ".join(MISSING).split())
    print("Dependencias instaladas. Reinicia el script.")
    sys.exit(0)


# ─── Modelo CNN ───────────────────────────────────────────────────────────────
class RespiratoryCNN(nn.Module):
    """
    CNN de 3 bloques convolucionales para clasificación de Rx de tórax.
    Arquitectura: Conv→BN→ReLU→Pool (x3) → FC(256) → Dropout → FC(num_classes)
    """
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 3→32, 224→112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 2: 32→64, 112→56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 3: 64→128, 56→28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Tras 3 MaxPool2d(2): 224 → 28
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Descarga desde S3 ───────────────────────────────────────────────────────
def download_from_s3(bucket: str, prefix: str, local_dir: Path) -> None:
    try:
        import boto3
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
        import boto3

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))

    print(f"  Descargando {len(objects)} archivos de s3://{bucket}/{prefix}")
    for obj in tqdm(objects, desc=f"  s3/{prefix}"):
        key = obj["Key"]
        relative = key[len(prefix):]
        if not relative or relative.endswith("/"):
            continue
        dest = local_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            s3.download_file(bucket, key, str(dest))


# ─── Data loaders ────────────────────────────────────────────────────────────
def build_loaders(data_dir: Path, image_size: int, batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(data_dir / "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(str(data_dir / "test"),  transform=eval_tf)

    # num_workers=0 es necesario dentro de contenedores Docker (shared memory limitado)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.classes


# ─── Entrenamiento ────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, epochs, lr, log_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_state = None

    log_lines = []
    def log(msg):
        ts = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_lines.append(line)

    log(f"Inicio entrenamiento | device={device} | epochs={epochs} | lr={lr} | params={model.count_parameters():,}")
    log(f"Train: {len(train_loader.dataset)} imgs | Val: {len(val_loader.dataset)} imgs")
    log("-" * 70)

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [train]", leave=False, ncols=90)
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            bar.set_postfix(loss=f"{running_loss/total:.3f}", acc=f"{correct/total:.3f}")

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

        val_loss /= val_total
        val_acc   = val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed   = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        msg = (f"Epoch {epoch:02d}/{epochs}  "
               f"loss={train_loss:.4f}  acc={train_acc:.4f}  "
               f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
               f"lr={current_lr:.6f}  t={elapsed:.1f}s")
        log(msg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log(f"  ★ Nuevo mejor modelo guardado (val_acc={val_acc:.4f})")

        scheduler.step()

    total_time = time.time() - total_start
    log(f"-" * 70)
    log(f"Entrenamiento completado en {total_time:.1f}s. Mejor val_acc={best_val_acc:.4f}")

    # Guardar log
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    return history, best_state, best_val_acc


# ─── Evaluación ──────────────────────────────────────────────────────────────
def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="  Evaluando test set", ncols=80):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = (all_preds == all_labels).mean()

    print("\n" + "=" * 70)
    print("REPORTE DE CLASIFICACIÓN (Test Set)")
    print("=" * 70)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return {
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
        "report": report,
        "cm": cm,
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
    }


# ─── Figuras ─────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "font.size": 11,
}
plt.rcParams.update(STYLE)
COLORS = {"COVID": "#e74c3c", "NEUMONIA": "#f39c12", "NORMALL": "#27ae60", "NORMAL": "#27ae60"}


def fig1_class_distribution(data_dir: Path, class_names: list, out: Path):
    """Distribución de clases en cada split."""
    counts = {}
    for split in ["train", "val", "test"]:
        split_counts = []
        for cls in class_names:
            cls_dir = data_dir / split / cls
            n = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
            split_counts.append(n)
        counts[split] = split_counts

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Distribución de Clases por Partición del Dataset", fontsize=14, fontweight="bold")

    colors = [COLORS.get(c, "#3498db") for c in class_names]
    for ax, (split, vals) in zip(axes, counts.items()):
        bars = ax.bar(class_names, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
        ax.set_title(f"{split.upper()} ({sum(vals)} imágenes)", fontweight="bold")
        ax.set_ylabel("Número de imágenes")
        ax.set_ylim(0, max(max(v) for v in counts.values()) * 1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig2_training_curves(history: dict, out: Path):
    """Curvas de pérdida y exactitud durante el entrenamiento."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Entrenamiento", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   "r-s", markersize=4, label="Val Loss",   linewidth=2)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida (Cross-Entropy)")
    ax1.set_title("Función de Pérdida")
    ax1.legend()
    ax1.set_xticks(list(epochs))

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", markersize=4, label="Train Acc", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   "r-s", markersize=4, label="Val Acc",   linewidth=2)
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Exactitud (%)")
    ax2.set_title("Exactitud por Época")
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.set_xticks(list(epochs))

    # Mark best val acc
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc   = max(history["val_acc"]) * 100
    ax2.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Mejor val: {best_acc:.1f}%")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig3_confusion_matrix(eval_results: dict, class_names: list, out: Path):
    """Matriz de confusión normalizada y con valores absolutos."""
    cm = eval_results["cm"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Matriz de Confusión — Test Set", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Valores Absolutos", "Normalizada (por clase real)"],
        [".0f", ".2%"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Clase Predicha", fontsize=11)
        ax.set_ylabel("Clase Real", fontsize=11)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig4_roc_curves(eval_results: dict, class_names: list, out: Path):
    """Curvas ROC por clase (one-vs-rest)."""
    labels_bin = label_binarize(eval_results["labels"], classes=list(range(len(class_names))))
    probs = eval_results["probs"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Curvas ROC por Clase (One-vs-Rest)", fontsize=14, fontweight="bold")

    colors_roc = [COLORS.get(c, "#3498db") for c in class_names]
    for i, (cls, color) in enumerate(zip(class_names, colors_roc)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], [p[i] for p in probs])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Aleatorio (AUC = 0.500)")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig5_sample_predictions(model, test_loader, device, class_names: list, out: Path, n=12):
    """Grid de imágenes del test set con predicción vs etiqueta real."""
    model.eval()
    images_shown, preds_list, labels_list, correct_list = [], [], [], []

    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_dev = inputs.to(device)
            outputs = model(inputs_dev)
            _, pred = outputs.max(1)
            for img, lbl, pr in zip(inputs, labels, pred):
                images_shown.append(inv_norm(img).clamp(0, 1).permute(1, 2, 0).numpy())
                preds_list.append(pr.item())
                labels_list.append(lbl.item())
                correct_list.append(pr.item() == lbl.item())
                if len(images_shown) >= n:
                    break
            if len(images_shown) >= n:
                break

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle("Predicciones en el Conjunto de Prueba", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < len(images_shown):
            ax.imshow(images_shown[i])
            pred_name = class_names[preds_list[i]]
            true_name = class_names[labels_list[i]]
            ok = correct_list[i]
            color = "green" if ok else "red"
            ax.set_title(f"Pred: {pred_name}\nReal: {true_name}",
                         color=color, fontsize=9, fontweight="bold")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig6_metrics_table(eval_results: dict, class_names: list, history: dict, out: Path):
    """Tabla visual de métricas por clase."""
    report = eval_results["report"]
    rows = []
    for cls in class_names:
        r = report.get(cls, {})
        rows.append([cls,
                     f"{r.get('precision', 0):.4f}",
                     f"{r.get('recall', 0):.4f}",
                     f"{r.get('f1-score', 0):.4f}",
                     str(r.get('support', 0))])
    # Macro avg
    m = report.get("macro avg", {})
    rows.append(["MACRO AVG",
                 f"{m.get('precision', 0):.4f}",
                 f"{m.get('recall', 0):.4f}",
                 f"{m.get('f1-score', 0):.4f}",
                 "—"])

    cols = ["Clase", "Precisión", "Recall", "F1-Score", "Support"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # Header color
    header_color = "#2c3e50"
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Row colors
    row_colors = ["#ecf0f1", "white"]
    for i, row in enumerate(rows, 1):
        for j in range(len(cols)):
            cell = table[i, j]
            if i == len(rows):  # Macro avg row
                cell.set_facecolor("#d5e8d4")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor(row_colors[i % 2])

    ax.set_title("Métricas de Clasificación por Clase — Test Set",
                 fontsize=13, fontweight="bold", pad=20)
    best_val = max(history["val_acc"])
    fig.text(0.5, 0.02,
             f"Exactitud total: {eval_results['accuracy']:.4f}  |  "
             f"Macro F1: {eval_results['macro_f1']:.4f}  |  "
             f"Mejor val_acc: {best_val:.4f}",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CNN Respiratory Disease Classifier — Experiment Script")
    parser.add_argument("--s3-bucket",   default="",                   help="S3 bucket con dataset (si no usas --data-dir)")
    parser.add_argument("--s3-prefix",   default="dataset",            help="Prefijo S3 del dataset (default: dataset)")
    parser.add_argument("--data-dir",    default="/tmp/dataset",        help="Directorio local del dataset")
    parser.add_argument("--output-dir",  default="./outputs",           help="Directorio de salida para figuras y logs")
    parser.add_argument("--epochs",      type=int,   default=20,        help="Número de épocas (default: 20)")
    parser.add_argument("--batch-size",  type=int,   default=32,        help="Tamaño de batch (default: 32)")
    parser.add_argument("--image-size",  type=int,   default=224,       help="Tamaño de imagen (default: 224)")
    parser.add_argument("--lr",          type=float, default=0.001,     help="Learning rate (default: 0.001)")
    parser.add_argument("--dropout",     type=float, default=0.3,       help="Dropout rate (default: 0.3)")
    parser.add_argument("--device",      default="auto",                help="auto | cpu | cuda")
    parser.add_argument("--model-path",  default="",                    help="Guardar/cargar modelo (.pth)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"  CNN Respiratory Disease Classifier")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"{'='*70}\n")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    data_dir = Path(args.data_dir)
    if args.s3_bucket and not (data_dir / "train").exists():
        print(f"Descargando dataset desde S3...")
        for split in ["train", "val", "test"]:
            download_from_s3(args.s3_bucket, f"{args.s3_prefix}/{split}/", data_dir / split)
        print("Dataset descargado.\n")

    if not (data_dir / "train").exists():
        print(f"ERROR: No se encontró el dataset en {data_dir}")
        print("Usa --s3-bucket BUCKET o --data-dir /ruta/dataset")
        sys.exit(1)

    # Loaders
    train_loader, val_loader, test_loader, class_names = build_loaders(
        data_dir, args.image_size, args.batch_size
    )
    print(f"Clases: {class_names}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}\n")

    # Modelo
    model = RespiratoryCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
    print(f"Parámetros entrenables: {model.count_parameters():,}\n")

    # Entrenar
    print("─── ENTRENAMIENTO ───────────────────────────────────────────────────")
    history, best_state, best_val_acc = train_model(
        model, train_loader, val_loader, device,
        args.epochs, args.lr,
        log_path=out_dir / "training_log.txt"
    )

    # Cargar mejor modelo
    model.load_state_dict(best_state)
    if args.model_path:
        torch.save(best_state, args.model_path)
        print(f"Modelo guardado: {args.model_path}")

    # Evaluar en test
    print("\n─── EVALUACIÓN TEST SET ─────────────────────────────────────────────")
    eval_results = evaluate_model(model, test_loader, device, class_names)

    # Guardar métricas
    metrics_out = {
        "accuracy": eval_results["accuracy"],
        "macro_f1": eval_results["macro_f1"],
        "best_val_accuracy": best_val_acc,
        "class_names": class_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "image_size": args.image_size,
        "device": str(device),
        "n_params": model.count_parameters(),
        "classification_report": eval_results["report"],
        "history": history,
        "confusion_matrix": eval_results["cm"].tolist(),
        "timestamp": datetime.utcnow().isoformat(),
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    # Figuras
    print("\n─── GENERANDO FIGURAS ───────────────────────────────────────────────")
    fig1_class_distribution(data_dir, class_names, out_dir / "fig1_class_distribution.png")
    fig2_training_curves(history, out_dir / "fig2_training_curves.png")
    fig3_confusion_matrix(eval_results, class_names, out_dir / "fig3_confusion_matrix.png")
    fig4_roc_curves(eval_results, class_names, out_dir / "fig4_roc_curves.png")
    fig5_sample_predictions(model, test_loader, device, class_names, out_dir / "fig5_sample_predictions.png")
    fig6_metrics_table(eval_results, class_names, history, out_dir / "fig6_metrics_table.png")

    # Resumen final
    print(f"\n{'='*70}")
    print(f"  RESULTADOS FINALES")
    print(f"{'='*70}")
    print(f"  Test Accuracy : {eval_results['accuracy']:.4f}  ({eval_results['accuracy']*100:.2f}%)")
    print(f"  Macro F1      : {eval_results['macro_f1']:.4f}")
    print(f"  Best Val Acc  : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
    print(f"\n  Archivos generados en: {out_dir.resolve()}")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
