"""Train the baseline CNN using the existing PlantVillage DataLoaders."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from baseline_cnn import create_baseline_cnn
from plantvillage_pytorch import create_dataloaders


DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_CLASSES = 38
DEFAULT_NUM_WORKERS = 2
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seeds used by Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use GPU automatically when it is available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch and return loss/accuracy."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += outputs.argmax(dim=1).eq(labels).sum().item()
        total_samples += batch_size

    return {
        "loss": running_loss / total_samples,
        "accuracy": running_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on validation data and return loss/accuracy."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += outputs.argmax(dim=1).eq(labels).sum().item()
        total_samples += batch_size

    return {
        "loss": running_loss / total_samples,
        "accuracy": running_correct / total_samples,
    }


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
) -> list[dict[str, float | int]]:
    """Train with train/validation splits and save best validation checkpoint."""
    if "train" not in dataloaders or "validation" not in dataloaders:
        raise ValueError("dataloaders deve conter os splits 'train' e 'validation'.")
    if epochs < 1:
        raise ValueError("epochs deve ser maior ou igual a 1.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_metrics = evaluate(
            model=model,
            dataloader=dataloaders["validation"],
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(row)

        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                dataloaders=dataloaders,
            )

        marker = " *" if improved else ""
        print(
            f"Epoca {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_acc={validation_metrics['accuracy']:.4f}{marker}"
        )

    return history


def train_baseline(
    metadata_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
) -> tuple[nn.Module, list[dict[str, float | int]], Path]:
    """Create DataLoaders, train the baseline CNN and save its artifacts."""
    set_seed(seed)
    device = get_device()
    metadata_csv = Path(metadata_csv)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "baseline_cnn_best.pth"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"CSV de split nao encontrado: {metadata_csv}")
    if not image_root.exists():
        raise FileNotFoundError(f"Diretorio de imagens nao encontrado: {image_root}")

    dataloaders = create_dataloaders(
        metadata_csv=metadata_csv,
        image_root=image_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    model = create_baseline_cnn(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Device: {device}")
    print(f"Parametros totais: {count_parameters(model):,}")
    print(f"Parametros treinaveis: {count_parameters(model, trainable_only=True):,}")

    history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
    )
    save_history(history, output_dir / "baseline_cnn_history.csv")
    return model, history, checkpoint_path


def save_history(history: list[dict[str, float | int]], output_path: str | Path) -> Path:
    """Save epoch metrics as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return output_path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_metadata_csv = project_root / "results" / "plantvillage_metadata_split.csv"
    local_image_root = project_root / "data" / "plantvillage_color"
    colab_image_root = Path("/content/plantvillage_color")
    default_image_root = colab_image_root if colab_image_root.exists() else local_image_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=default_metadata_csv)
    parser.add_argument("--image-root", type=Path, default=default_image_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "baseline_cnn",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_baseline(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    history: list[dict[str, float | int]],
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    dataloaders: dict[str, DataLoader],
) -> None:
    train_dataset = dataloaders["train"].dataset
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "history": [row.copy() for row in history],
        "class_to_idx": getattr(train_dataset, "class_to_idx", None),
        "idx_to_class": getattr(train_dataset, "idx_to_class", None),
    }
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()
