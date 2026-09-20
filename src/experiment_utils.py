"""Reusable experiment utilities for image-classification models."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class InferenceResult:
    """Predictions and timing measurements from one complete DataLoader pass."""

    targets: np.ndarray
    predictions: np.ndarray
    total_time_seconds: float
    average_time_per_image_seconds: float
    throughput_images_per_second: float
    num_images: int
    warmup_batches: int

    def timing_metrics(self) -> dict[str, float | int | str]:
        """Return the timing fields in a serialization-friendly dictionary."""
        return {
            "total_time_seconds": self.total_time_seconds,
            "average_time_per_image_seconds": self.average_time_per_image_seconds,
            "throughput_images_per_second": self.throughput_images_per_second,
            "num_images": self.num_images,
            "warmup_batches": self.warmup_batches,
            "measurement_scope": "end_to_end_dataloader",
        }


@dataclass
class EarlyStopping:
    """Track a monitored metric and stop after consecutive non-improvements."""

    patience: int = 5
    min_delta: float = 1e-4
    mode: str = "min"
    best_value: float | None = field(default=None, init=False)
    best_epoch: int | None = field(default=None, init=False)
    bad_epochs: int = field(default=0, init=False)
    stopped_epoch: int | None = field(default=None, init=False)
    last_epoch: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience deve ser maior ou igual a 1.")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta deve ser finito e maior ou igual a zero.")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode deve ser 'min' ou 'max'.")

    def step(self, value: float, epoch: int) -> bool:
        """Update state and return whether ``value`` is a meaningful improvement."""
        if not np.isfinite(value):
            raise FloatingPointError(f"Metrica monitorada nao finita na epoca {epoch}: {value}")

        self.last_epoch = epoch
        if self._is_improvement(value):
            self.best_value = float(value)
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stopped_epoch = epoch
        return False

    @property
    def should_stop(self) -> bool:
        """Whether the configured patience has been exhausted."""
        return self.stopped_epoch is not None

    def summary(self) -> dict[str, float | int | str | bool | None]:
        """Return the stopping state in a JSON-serializable structure."""
        return {
            "monitor": "validation_loss",
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_value,
            "epochs_without_improvement": self.bad_epochs,
            "stopped_epoch": self.stopped_epoch or self.last_epoch,
            "early_stopping": self.should_stop,
        }

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta


def synchronize_device(device: torch.device) -> None:
    """Wait for queued CUDA work so wall-clock measurements are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup_batches: int = 1,
) -> InferenceResult:
    """Measure one full inference pass and collect predictions.

    On CUDA, warm-up batches run before the timed pass and are not included in
    the reported values. The timed pass always covers the complete DataLoader,
    including loading, device transfer, model forward pass and prediction
    collection. This end-to-end scope is recorded with the timing metrics.
    """
    if warmup_batches < 0:
        raise ValueError("warmup_batches deve ser maior ou igual a zero.")

    was_training = model.training
    effective_warmup_batches = warmup_batches if device.type == "cuda" else 0

    try:
        model.eval()

        completed_warmup_batches = 0
        if effective_warmup_batches:
            for batch_index, batch in enumerate(dataloader):
                images, _ = _unpack_batch(batch)
                images = images.to(device, non_blocking=True)
                model(images)
                completed_warmup_batches = batch_index + 1
                if completed_warmup_batches >= effective_warmup_batches:
                    break
            synchronize_device(device)

        target_batches: list[torch.Tensor] = []
        prediction_batches: list[torch.Tensor] = []
        num_images = 0

        synchronize_device(device)
        start_time = time.perf_counter()
        for batch in dataloader:
            images, labels = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            target_batches.append(labels.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())
            num_images += int(labels.size(0))

        synchronize_device(device)
        total_time = time.perf_counter() - start_time
    finally:
        model.train(was_training)

    if num_images == 0:
        raise ValueError("O DataLoader de inferencia esta vazio.")
    if total_time <= 0:
        raise RuntimeError("Nao foi possivel obter um tempo de inferencia valido.")

    targets = torch.cat(target_batches).numpy()
    predictions = torch.cat(prediction_batches).numpy()
    return InferenceResult(
        targets=targets,
        predictions=predictions,
        total_time_seconds=total_time,
        average_time_per_image_seconds=total_time / num_images,
        throughput_images_per_second=num_images / total_time,
        num_images=num_images,
        warmup_batches=completed_warmup_batches,
    )


def compute_classification_metrics(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    """Compute overall, macro, weighted and per-class classification metrics."""
    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    class_names = [str(class_name) for class_name in class_names]

    if targets_array.ndim != 1 or predictions_array.ndim != 1:
        raise ValueError("targets e predictions devem ser vetores unidimensionais.")
    if targets_array.size == 0:
        raise ValueError("Nao ha amostras para calcular as metricas.")
    if targets_array.size != predictions_array.size:
        raise ValueError("targets e predictions devem ter o mesmo tamanho.")
    if not class_names:
        raise ValueError("class_names nao pode ser vazio.")

    labels = np.arange(len(class_names))
    if np.any(targets_array < 0) or np.any(targets_array >= len(class_names)):
        raise ValueError("targets contem indices fora do intervalo de class_names.")
    if np.any(predictions_array < 0) or np.any(predictions_array >= len(class_names)):
        raise ValueError("predictions contem indices fora do intervalo de class_names.")

    precision, recall, f1_score, support = precision_recall_fscore_support(
        targets_array,
        predictions_array,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        targets_array,
        predictions_array,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        targets_array,
        predictions_array,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    overall_metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(targets_array, predictions_array)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "num_samples": int(targets_array.size),
        "num_classes": len(class_names),
    }
    per_class_metrics = [
        {
            "class_index": int(class_index),
            "class_name": class_name,
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1_score": float(f1_score[class_index]),
            "support": int(support[class_index]),
        }
        for class_index, class_name in enumerate(class_names)
    ]
    return overall_metrics, per_class_metrics


def save_classification_metrics(
    overall_metrics: dict[str, float | int],
    per_class_metrics: list[dict[str, float | int | str]],
    overall_path: str | Path,
    per_class_path: str | Path,
) -> tuple[Path, Path]:
    """Save overall and per-class metrics as CSV files."""
    overall_path = Path(overall_path)
    per_class_path = Path(per_class_path)
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    per_class_path.parent.mkdir(parents=True, exist_ok=True)

    with overall_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(overall_metrics))
        writer.writeheader()
        writer.writerow(overall_metrics)

    per_class_fieldnames = [
        "class_index",
        "class_name",
        "precision",
        "recall",
        "f1_score",
        "support",
    ]
    with per_class_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=per_class_fieldnames)
        writer.writeheader()
        writer.writerows(per_class_metrics)

    return overall_path, per_class_path


def save_confusion_matrix(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    figsize: tuple[float, float] = (24.0, 22.0),
    dpi: int = 200,
) -> Path:
    """Render and save a legible confusion matrix with all class names."""
    class_names = [str(class_name) for class_name in class_names]
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(targets, predictions, labels=labels)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Matriz de confusao - validacao",
        xlabel="Classe predita",
        ylabel="Classe real",
        xticks=labels,
        yticks=labels,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.tick_params(axis="both", labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def describe_model_architecture(model: nn.Module) -> dict[str, Any]:
    """Create a JSON-serializable structural summary of a PyTorch model."""
    convolutional_layers: list[dict[str, Any]] = []
    pooling_layers: list[dict[str, Any]] = []
    dropout_layers: list[dict[str, Any]] = []
    linear_layers: list[dict[str, Any]] = []

    pooling_types = (
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
    )
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Conv2d):
            convolutional_layers.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": _json_value(module.kernel_size),
                    "stride": _json_value(module.stride),
                    "padding": _json_value(module.padding),
                }
            )
        elif isinstance(module, pooling_types):
            pooling_layer: dict[str, Any] = {
                "name": name,
                "type": type(module).__name__,
            }
            for attribute in ("kernel_size", "stride", "padding", "output_size"):
                if hasattr(module, attribute):
                    pooling_layer[attribute] = _json_value(getattr(module, attribute))
            pooling_layers.append(pooling_layer)
        elif isinstance(module, dropout_types):
            dropout_layers.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "p": float(module.p),
                }
            )
        elif isinstance(module, nn.Linear):
            linear_layers.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
            )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    classification_layer = linear_layers[-1] if linear_layers else None
    return {
        "model_class": type(model).__name__,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "num_convolutional_layers": len(convolutional_layers),
        "convolutional_layers": convolutional_layers,
        "pooling_layers": pooling_layers,
        "dropout_layers": dropout_layers,
        "classification_layer": classification_layer,
        "num_output_classes": (
            classification_layer["out_features"] if classification_layer else None
        ),
        "model_repr": str(model),
    }


def save_architecture_summary(
    model: nn.Module,
    output_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Describe a model and save its architecture summary as readable JSON."""
    summary = describe_model_architecture(model)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return summary, output_path


def summarize_training_times(
    history: Sequence[dict[str, float | int]],
) -> dict[str, float | int | str]:
    """Summarize per-epoch timing values from a training history."""
    epoch_times = [float(row["epoch_time_seconds"]) for row in history]
    if not epoch_times:
        raise ValueError("O historico de treinamento esta vazio.")
    return {
        "epochs": len(epoch_times),
        "total_time_seconds": float(sum(epoch_times)),
        "average_time_per_epoch_seconds": float(sum(epoch_times) / len(epoch_times)),
        "measurement_scope": "sum_of_epoch_train_and_validation_passes",
    }


def save_timing_summary(
    training_timing: dict[str, Any],
    inference_timing: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save training and validation-inference timing measurements as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "training": training_timing,
        "validation_inference": inference_timing,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return output_path


def collect_experiment_metadata(
    *,
    device: torch.device | None,
    batch_size: int | None,
    max_epochs: int | None,
    patience: int | None,
    min_delta: float | None,
    learning_rate: float | None,
    optimizer: Any | None,
    loss_function: Any | None,
    seed: int | None,
    num_workers: int | None,
    input_image_size: Sequence[int] | None,
    num_classes: int | None,
    train_samples: int | None,
    validation_samples: int | None,
) -> dict[str, Any]:
    """Collect portable environment and run configuration metadata.

    Optional system values are recorded as ``None`` if they cannot be queried,
    so recording metadata never blocks an experiment.
    """
    metadata: dict[str, Any] = {
        "execution_timestamp": datetime.now().astimezone().isoformat(),
        "operating_system": _safe_value(platform.system),
        "platform": _safe_value(platform.platform),
        "python_version": sys.version,
        "pytorch_version": getattr(torch, "__version__", None),
        "torchvision_version": _get_torchvision_version(),
        "cuda_available": _safe_cuda_is_available(),
        "pytorch_cuda_version": getattr(torch.version, "cuda", None),
        "selected_device": str(device) if device is not None else None,
        "gpu_name": None,
        "gpu_total_memory_bytes": None,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "min_delta": min_delta,
        "learning_rate": learning_rate,
        "optimizer": type(optimizer).__name__ if optimizer is not None else None,
        "loss_function": (
            type(loss_function).__name__ if loss_function is not None else None
        ),
        "seed": seed,
        "num_workers": num_workers,
        "input_image_size": (
            list(input_image_size) if input_image_size is not None else None
        ),
        "num_classes": num_classes,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
    }

    if metadata["cuda_available"]:
        gpu_index = _resolve_gpu_index(device)
        metadata["gpu_name"] = _safe_value(torch.cuda.get_device_name, gpu_index)
        properties = _safe_value(torch.cuda.get_device_properties, gpu_index)
        metadata["gpu_total_memory_bytes"] = getattr(properties, "total_memory", None)

    return metadata


def save_experiment_metadata(metadata: dict[str, Any], output_path: str | Path) -> Path:
    """Save experiment metadata as formatted JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return output_path


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("Cada batch deve conter, no minimo, imagens e rotulos.")
    images, labels = batch[0], batch[1]
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Imagens e rotulos do batch devem ser tensores PyTorch.")
    return images, labels


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_value(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except Exception:
        return None


def _safe_cuda_is_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_torchvision_version() -> str | None:
    try:
        import torchvision

        return getattr(torchvision, "__version__", None)
    except Exception:
        return None


def _resolve_gpu_index(device: torch.device | None) -> int:
    if device is not None and device.type == "cuda" and device.index is not None:
        return device.index
    current_device = _safe_value(torch.cuda.current_device)
    return int(current_device) if current_device is not None else 0


__all__ = [
    "EarlyStopping",
    "InferenceResult",
    "collect_experiment_metadata",
    "compute_classification_metrics",
    "describe_model_architecture",
    "measure_inference",
    "save_architecture_summary",
    "save_classification_metrics",
    "save_confusion_matrix",
    "save_experiment_metadata",
    "save_timing_summary",
    "summarize_training_times",
    "synchronize_device",
]
