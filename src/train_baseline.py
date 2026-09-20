"""Train and evaluate the baseline CNN with reusable experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from baseline_cnn import create_baseline_cnn
from experiment_utils import (
    EarlyStopping,
    collect_experiment_metadata,
    compute_classification_metrics,
    measure_inference,
    save_architecture_summary,
    save_classification_metrics,
    save_confusion_matrix,
    save_experiment_metadata,
    save_timing_summary,
    summarize_training_times,
    synchronize_device,
)
from plantvillage_pytorch import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_dataloaders,
)


DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_CLASSES = 38
DEFAULT_NUM_WORKERS = 2
DEFAULT_SEED = 42
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4


@dataclass(frozen=True)
class TrainingResult:
    """Training history and the final early-stopping state."""

    history: list[dict[str, float | int]]
    epochs_executed: int
    best_epoch: int
    best_validation_loss: float
    stopped_epoch: int
    early_stopping: bool

    def summary(self) -> dict[str, float | int | bool]:
        """Return values that describe how training finished."""
        return {
            "epochs_executed": self.epochs_executed,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_epoch": self.stopped_epoch,
            "early_stopping": self.early_stopping,
        }


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
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
) -> TrainingResult:
    """Train with train/validation splits and save the best validation checkpoint.

    ``epoch_time_seconds`` covers the training and validation passes for each
    epoch. CUDA is synchronized immediately before and after that interval.
    Early stopping monitors validation loss in ``min`` mode.
    """
    if "train" not in dataloaders or "validation" not in dataloaders:
        raise ValueError("dataloaders deve conter os splits 'train' e 'validation'.")
    if epochs < 1:
        raise ValueError("epochs deve ser maior ou igual a 1.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    history: list[dict[str, float | int]] = []
    best_checkpoint_loss = float("inf")
    best_checkpoint_epoch: int | None = None
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    for epoch in range(1, epochs + 1):
        synchronize_device(device)
        epoch_start_time = time.perf_counter()
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
        synchronize_device(device)
        epoch_time_seconds = time.perf_counter() - epoch_start_time

        if not np.isfinite(train_metrics["loss"]):
            raise FloatingPointError(
                f"Loss de treino nao finita na epoca {epoch}: "
                f"{train_metrics['loss']}"
            )
        if not np.isfinite(validation_metrics["loss"]):
            raise FloatingPointError(
                f"Loss de validacao nao finita na epoca {epoch}: "
                f"{validation_metrics['loss']}"
            )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "epoch_time_seconds": epoch_time_seconds,
        }
        history.append(row)

        early_stopper.step(
            validation_metrics["loss"],
            epoch,
        )
        checkpoint_improved = validation_metrics["loss"] < best_checkpoint_loss
        if checkpoint_improved:
            best_checkpoint_loss = validation_metrics["loss"]
            best_checkpoint_epoch = epoch
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                dataloaders=dataloaders,
                early_stopping_state=early_stopper.summary(),
            )

        marker = " *" if checkpoint_improved else ""
        print(
            f"Epoca {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_acc={validation_metrics['accuracy']:.4f} | "
            f"tempo={epoch_time_seconds:.2f}s | "
            f"sem_melhoria={early_stopper.bad_epochs}/{patience}{marker}"
        )

        if early_stopper.should_stop:
            print(
                "Early stopping acionado na epoca "
                f"{epoch}: validation_loss sem melhora significativa por "
                f"{patience} epocas consecutivas."
            )
            break

    training_timing = summarize_training_times(history)
    print(
        "Tempo total de treinamento: "
        f"{training_timing['total_time_seconds']:.2f}s"
    )
    print(
        "Tempo medio por epoca: "
        f"{training_timing['average_time_per_epoch_seconds']:.2f}s"
    )
    if best_checkpoint_epoch is None or not np.isfinite(best_checkpoint_loss):
        raise RuntimeError("Nenhum checkpoint valido foi salvo durante o treinamento.")

    return TrainingResult(
        history=history,
        epochs_executed=len(history),
        best_epoch=best_checkpoint_epoch,
        best_validation_loss=best_checkpoint_loss,
        stopped_epoch=early_stopper.stopped_epoch or early_stopper.last_epoch or len(history),
        early_stopping=early_stopper.should_stop,
    )


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
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
) -> tuple[nn.Module, list[dict[str, float | int]], Path]:
    """Train and evaluate the baseline CNN, saving all experiment artifacts."""
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
    train_validation_loaders = {
        split: dataloaders[split] for split in ("train", "validation")
    }
    model = create_baseline_cnn(num_classes=num_classes)
    class_names = list(dataloaders["validation"].dataset.classes)
    if len(class_names) != num_classes:
        raise ValueError(
            f"O modelo foi configurado para {num_classes} classes, "
            f"mas os dados possuem {len(class_names)}."
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    experiment_metadata = collect_experiment_metadata(
        device=device,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss_function=criterion,
        seed=seed,
        num_workers=num_workers,
        input_image_size=(3, IMAGE_SIZE, IMAGE_SIZE),
        num_classes=num_classes,
        train_samples=len(dataloaders["train"].dataset),
        validation_samples=len(dataloaders["validation"].dataset),
    )
    experiment_metadata.update(
        {
            "model": "CNN baseline",
            "architecture": "BaselineCNN",
            "weights": None,
            "pretrained_dataset": None,
            "parameters_frozen": False,
            "all_parameters_trainable": True,
            "fine_tuning_strategy": "trained_from_scratch",
            "loss": "CrossEntropyLoss",
            "early_stopping_monitor": "validation_loss",
            "early_stopping_mode": "min",
            "evaluation_split": "validation",
            "test_split_used": False,
            "preprocessing": {
                "image_size": [IMAGE_SIZE, IMAGE_SIZE],
                "normalization_source": "ImageNet statistics",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "augmentation_policy": "baseline_shared_pipeline",
            },
        }
    )
    experiment_metadata_path = save_experiment_metadata(
        experiment_metadata,
        output_dir / "experiment_metadata.json",
    )

    print(f"Device: {device}")
    print(model)
    print(f"Parametros totais: {count_parameters(model):,}")
    print(f"Parametros treinaveis: {count_parameters(model, trainable_only=True):,}")
    architecture_summary, architecture_path = save_architecture_summary(
        model,
        output_dir / "architecture_summary.json",
    )
    print(f"Camadas convolucionais: {architecture_summary['num_convolutional_layers']}")
    for layer in architecture_summary["convolutional_layers"]:
        print(
            f"  {layer['name']}: {layer['in_channels']} -> {layer['out_channels']}, "
            f"kernel={layer['kernel_size']}, stride={layer['stride']}, "
            f"padding={layer['padding']}"
        )
    print(f"Pooling: {architecture_summary['pooling_layers']}")
    print(f"Dropout: {architecture_summary['dropout_layers']}")
    print(f"Classificador final: {architecture_summary['classification_layer']}")
    print(f"Resumo da arquitetura salvo em: {architecture_path}")

    training_result = train_model(
        model=model,
        dataloaders=train_validation_loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        patience=patience,
        min_delta=min_delta,
    )
    history = training_result.history
    history_path = save_history(history, output_dir / "baseline_cnn_history.csv")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    inference_result = measure_inference(
        model=model,
        dataloader=dataloaders["validation"],
        device=device,
        warmup_batches=1,
    )
    overall_metrics, per_class_metrics = compute_classification_metrics(
        targets=inference_result.targets,
        predictions=inference_result.predictions,
        class_names=class_names,
    )
    overall_metrics_path, per_class_metrics_path = save_classification_metrics(
        overall_metrics=overall_metrics,
        per_class_metrics=per_class_metrics,
        overall_path=output_dir / "validation_metrics.csv",
        per_class_path=output_dir / "validation_metrics_by_class.csv",
    )
    confusion_matrix_path = save_confusion_matrix(
        targets=inference_result.targets,
        predictions=inference_result.predictions,
        class_names=class_names,
        output_path=output_dir / "validation_confusion_matrix.png",
    )

    training_timing = summarize_training_times(history)
    training_timing.update(training_result.summary())
    timing_summary_path = save_timing_summary(
        training_timing=training_timing,
        inference_timing=inference_result.timing_metrics(),
        output_path=output_dir / "timing_summary.json",
    )

    experiment_metadata["training"] = training_result.summary()
    experiment_metadata["best_checkpoint"] = str(checkpoint_path)
    experiment_metadata["best_checkpoint_restored_before_validation"] = True
    experiment_metadata["best_model_validation_accuracy"] = checkpoint[
        "validation_accuracy"
    ]
    save_experiment_metadata(experiment_metadata, experiment_metadata_path)

    print("Metricas de validacao:")
    for metric_name, metric_value in overall_metrics.items():
        if isinstance(metric_value, float):
            print(f"  {metric_name}: {metric_value:.6f}")
        else:
            print(f"  {metric_name}: {metric_value}")
    print(
        "Inferencia na validacao: "
        f"total={inference_result.total_time_seconds:.4f}s | "
        f"media={inference_result.average_time_per_image_seconds:.8f}s/imagem | "
        f"throughput={inference_result.throughput_images_per_second:.2f} imagens/s"
    )
    print("\nBaseline CNN concluida")
    print(f"Epocas executadas: {training_result.epochs_executed} / {epochs}")
    print(f"Early stopping: {'sim' if training_result.early_stopping else 'nao'}")
    print(f"Melhor epoca: {training_result.best_epoch}")
    print(f"Melhor validation loss: {training_result.best_validation_loss:.6f}")
    print(
        "Validation accuracy do melhor modelo: "
        f"{checkpoint['validation_accuracy']:.6f}"
    )
    print(
        "Tempo total de treinamento: "
        f"{training_timing['total_time_seconds']:.2f}s"
    )
    print(
        "Tempo medio por epoca: "
        f"{training_timing['average_time_per_epoch_seconds']:.2f}s"
    )
    print(
        "Tempo medio de inferencia por imagem: "
        f"{inference_result.average_time_per_image_seconds:.8f}s"
    )
    print(
        "Throughput: "
        f"{inference_result.throughput_images_per_second:.2f} imagens/s"
    )
    print(f"Parametros treinaveis: {count_parameters(model, trainable_only=True)}")
    print("Artefatos salvos:")
    for artifact_path in (
        checkpoint_path,
        history_path,
        overall_metrics_path,
        per_class_metrics_path,
        confusion_matrix_path,
        architecture_path,
        timing_summary_path,
        experiment_metadata_path,
    ):
        print(f"  {artifact_path}")
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
        "epoch_time_seconds",
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Numero maximo de epocas antes do early stopping.",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Epocas consecutivas sem melhora antes de interromper.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help="Melhora minima em validation_loss para reiniciar a patience.",
    )
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
        patience=args.patience,
        min_delta=args.min_delta,
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
    early_stopping_state: dict[str, float | int | str | bool | None],
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
        "early_stopping": early_stopping_state,
        "history": [row.copy() for row in history],
        "class_to_idx": getattr(train_dataset, "class_to_idx", None),
        "idx_to_class": getattr(train_dataset, "idx_to_class", None),
    }
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()
