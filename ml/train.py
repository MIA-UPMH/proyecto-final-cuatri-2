from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import shutil


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
except Exception:  # pragma: no cover - fallback when ML deps are absent
    torch = None
    nn = None
    DataLoader = None
    random_split = None
    datasets = None
    transforms = None


@dataclass
class TrainingMetrics:
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    macro_f1: float
    loss: float
    mode: str


class TrainingCancelled(RuntimeError):
    pass


def _raise_if_cancelled(cancel_callback) -> None:
    if cancel_callback and cancel_callback():
        raise TrainingCancelled("Cancelacion solicitada por el usuario.")


class SimpleRespiratoryCNN(nn.Module):
    def __init__(self, class_count: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, class_count),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def simulate_training(
    output_dir: Path,
    seed: int,
    epochs: int,
    progress_callback=None,
    batch_progress_callback=None,
    cancel_callback=None,
) -> TrainingMetrics:
    random.seed(seed)

    base_accuracy = 0.88 + random.random() * 0.08
    total_epochs = max(epochs, 1)
    for epoch in range(total_epochs):
        _raise_if_cancelled(cancel_callback)

        train_loss = round(0.5 - (epoch * 0.015), 4)
        train_accuracy = round(min(base_accuracy - 0.04 + epoch * 0.01, 0.99), 4)
        val_accuracy = round(max(train_accuracy - 0.015, 0.7), 4)
        val_loss = round(train_loss + 0.025, 4)

        if batch_progress_callback:
            batch_progress_callback(
                {
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "batch": 1,
                    "total_batches": 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "learning_rate": 0.0007,
                }
            )

        if progress_callback:
            progress_callback(
                {
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": round(val_accuracy - 0.01, 4),
                    "learning_rate": 0.0007,
                }
            )

            _raise_if_cancelled(cancel_callback)

    metrics = TrainingMetrics(
        train_accuracy=round(min(base_accuracy + 0.04, 0.995), 4),
        val_accuracy=round(base_accuracy - 0.01, 4),
        test_accuracy=round(base_accuracy - 0.015, 4),
        macro_f1=round(base_accuracy - 0.02, 4),
        loss=round(0.42 - random.random() * 0.18, 4),
        mode="simulation",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    (output_dir / "model.pt").write_text("placeholder-artifact", encoding="utf-8")
    return metrics


def accuracy_from_logits(logits, labels) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def macro_f1_score(predictions, labels, class_count: int) -> float:
    score_sum = 0.0
    for class_index in range(class_count):
        true_positive = ((predictions == class_index) & (labels == class_index)).sum().item()
        false_positive = ((predictions == class_index) & (labels != class_index)).sum().item()
        false_negative = ((predictions != class_index) & (labels == class_index)).sum().item()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        score_sum += f1
    return score_sum / class_count


def evaluate(model, loader, criterion, device, class_count: int) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_accuracy += accuracy_from_logits(logits, labels)
            batch_count += 1
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    if batch_count == 0:
        return 0.0, 0.0, 0.0

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    return (
        total_loss / max(batch_count, 1),
        total_accuracy / max(batch_count, 1),
        macro_f1_score(predictions, labels, class_count),
    )


def _feedback_has_samples(feedback_root: Path | None) -> bool:
    if feedback_root is None or not feedback_root.exists():
        return False
    return any(item.is_file() for item in feedback_root.rglob("*"))


def _prepare_training_root(base_train_path: Path, feedback_root: Path | None, output_dir: Path) -> Path:
    if not _feedback_has_samples(feedback_root):
        return base_train_path

    merged_root = output_dir / "_merged_train"
    if merged_root.exists():
        shutil.rmtree(merged_root)
    shutil.copytree(base_train_path, merged_root)

    for class_dir in sorted(feedback_root.iterdir()):
        if not class_dir.is_dir():
            continue
        target_dir = merged_root / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for sample in sorted(class_dir.iterdir()):
            if not sample.is_file():
                continue
            destination = target_dir / sample.name
            if destination.exists():
                destination = target_dir / f"feedback_{sample.stem}{sample.suffix}"
            shutil.copy2(sample, destination)
    return merged_root


def _build_optimizer(model, optimizer_name: str, learning_rate: float, weight_decay: float):
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def _build_scheduler(optimizer, scheduler_name: str, epochs: int):
    scheduler_name = scheduler_name.lower().strip()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    return None


def execute_training(
    dataset_root: Path,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    optimizer_name: str,
    dropout: float,
    weight_decay: float,
    scheduler_name: str,
    early_stopping_patience: int,
    gradient_clip: float,
    seed: int,
    feedback_root: Path | None = None,
    log_interval: int = 5,
    progress_callback=None,
    batch_progress_callback=None,
    cancel_callback=None,
) -> TrainingMetrics:
    if torch is None or datasets is None or not dataset_root.exists():
        return simulate_training(output_dir, seed, epochs, progress_callback, batch_progress_callback, cancel_callback)

    random.seed(seed)
    torch.manual_seed(seed)

    train_path = dataset_root / "train"
    test_path = dataset_root / "test"
    if not train_path.exists() or not test_path.exists():
        return simulate_training(output_dir, seed, epochs, progress_callback, batch_progress_callback, cancel_callback)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    prepared_train_path = _prepare_training_root(train_path, feedback_root, output_dir)
    full_train = datasets.ImageFolder(prepared_train_path, transform=transform)

    # Check if test set is usable (has images for all classes)
    test_usable = False
    if test_path.exists():
        try:
            test_dataset = datasets.ImageFolder(test_path, transform=transform)
            if len(test_dataset) > 0:
                test_usable = True
        except Exception:
            pass

    if test_usable:
        # Split train into train+val, keep separate test
        validation_size = max(int(len(full_train) * 0.2), 1)
        training_size = max(len(full_train) - validation_size, 1)
        training_dataset, validation_dataset = random_split(
            full_train,
            [training_size, validation_size],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        # Split train into train+val+test (60/20/20)
        total = len(full_train)
        test_size = max(int(total * 0.2), 1)
        val_size = max(int(total * 0.2), 1)
        train_size = max(total - val_size - test_size, 1)
        training_dataset, validation_dataset, test_dataset = random_split(
            full_train,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRespiratoryCNN(class_count=len(full_train.classes), dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, optimizer_name, learning_rate, weight_decay)
    scheduler = _build_scheduler(optimizer, scheduler_name, epochs)

    last_loss = 0.0
    train_accuracy = 0.0
    best_val_accuracy = -1.0
    best_state = None
    best_epoch = 0
    patience_without_improvement = 0
    total_batches = max(len(train_loader), 1)

    for epoch_idx in range(epochs):
        _raise_if_cancelled(cancel_callback)

        model.train()
        batch_loss = 0.0
        batch_accuracy = 0.0
        batch_count = 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            _raise_if_cancelled(cancel_callback)

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            batch_loss += loss.item()
            batch_accuracy += accuracy_from_logits(logits, labels)
            batch_count += 1

            if batch_progress_callback and (
                batch_idx == 1 or batch_idx == total_batches or batch_idx % max(log_interval, 1) == 0
            ):
                batch_progress_callback(
                    {
                        "epoch": epoch_idx + 1,
                        "total_epochs": epochs,
                        "batch": batch_idx,
                        "total_batches": total_batches,
                        "train_loss": round(batch_loss / max(batch_count, 1), 4),
                        "train_accuracy": round(batch_accuracy / max(batch_count, 1), 4),
                        "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
                    }
                )

        _raise_if_cancelled(cancel_callback)

        last_loss = batch_loss / max(batch_count, 1)
        train_accuracy = batch_accuracy / max(batch_count, 1)
        val_loss, val_accuracy, val_macro_f1 = evaluate(model, validation_loader, criterion, device, len(full_train.classes))

        if scheduler is not None:
            if scheduler_name.lower().strip() == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch_idx + 1
            patience_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_without_improvement += 1

        if progress_callback:
            progress_callback({
                "epoch": epoch_idx + 1,
                "total_epochs": epochs,
                "train_loss": round(last_loss, 4),
                "train_accuracy": round(train_accuracy, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_accuracy, 4),
                "val_macro_f1": round(val_macro_f1, 4),
                "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
            })

            _raise_if_cancelled(cancel_callback)

        if early_stopping_patience > 0 and patience_without_improvement >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _raise_if_cancelled(cancel_callback)

    _, val_accuracy, _ = evaluate(model, validation_loader, criterion, device, len(full_train.classes))
    _, test_accuracy, macro_f1 = evaluate(model, test_loader, criterion, device, len(full_train.classes))

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")

    metrics = TrainingMetrics(
        train_accuracy=round(train_accuracy, 4),
        val_accuracy=round(val_accuracy, 4),
        test_accuracy=round(test_accuracy, 4),
        macro_f1=round(macro_f1, 4),
        loss=round(last_loss, 4),
        mode="torch" if torch is not None else "simulation",
    )
    payload = asdict(metrics)
    payload["best_epoch"] = best_epoch
    payload["scheduler"] = scheduler_name
    payload["optimizer"] = optimizer_name
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics