"""Compare saved validation artifacts from the four PlantVillage models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_TRAIN_SAMPLES = 38_008
EXPECTED_VALIDATION_SAMPLES = 8_172
EXPECTED_NUM_CLASSES = 38
EXPECTED_INPUT_SIZE = [3, 224, 224]


@dataclass(frozen=True)
class ModelSpec:
    """Names and model-specific artifacts for one experiment."""

    key: str
    display_name: str
    directory_name: str
    checkpoint_name: str
    history_name: str

    def artifact_names(self) -> tuple[str, ...]:
        """Return every artifact expected from a completed experiment."""
        return (
            self.checkpoint_name,
            self.history_name,
            "validation_metrics.csv",
            "validation_metrics_by_class.csv",
            "validation_confusion_matrix.png",
            "architecture_summary.json",
            "timing_summary.json",
            "experiment_metadata.json",
        )


MODEL_SPECS = (
    ModelSpec(
        key="baseline_cnn",
        display_name="CNN baseline",
        directory_name="baseline_cnn",
        checkpoint_name="baseline_cnn_best.pth",
        history_name="baseline_cnn_history.csv",
    ),
    ModelSpec(
        key="resnet50",
        display_name="ResNet50",
        directory_name="resnet50",
        checkpoint_name="resnet50_best.pth",
        history_name="resnet50_history.csv",
    ),
    ModelSpec(
        key="densenet201",
        display_name="DenseNet201",
        directory_name="densenet201",
        checkpoint_name="densenet201_best.pth",
        history_name="densenet201_history.csv",
    ),
    ModelSpec(
        key="efficientnet_b0",
        display_name="EfficientNet-B0",
        directory_name="efficientnet_b0",
        checkpoint_name="efficientnet_b0_best.pth",
        history_name="efficientnet_b0_history.csv",
    ),
)

SUMMARY_COLUMNS = [
    "model_key",
    "model_name",
    "status",
    "missing_artifacts",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "total_parameters",
    "trainable_parameters",
    "parameters_millions",
    "epochs_executed",
    "best_epoch",
    "best_validation_loss",
    "training_time_seconds",
    "training_time_minutes",
    "average_time_per_epoch_seconds",
    "inference_time_seconds",
    "inference_seconds_per_image",
    "inference_milliseconds_per_image",
    "throughput_images_per_second",
    "gpu",
    "gpu_total_memory_gb",
    "selected_device",
    "pytorch_version",
    "cuda_version",
    "batch_size",
    "num_workers",
    "input_size",
]

PER_CLASS_COLUMNS = [
    "model_key",
    "model_name",
    "class_index",
    "class_name",
    "precision",
    "recall",
    "f1_score",
    "support",
]

HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "epoch_time_seconds",
]


@dataclass
class ComparisonResult:
    """Loaded tables, histories, warnings and saved comparison paths."""

    summary: pd.DataFrame
    per_class: pd.DataFrame
    histories: dict[str, pd.DataFrame]
    warnings: list[str]
    summary_path: Path | None = None
    per_class_path: Path | None = None
    warnings_path: Path | None = None


def compare_validation_results(
    results_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    save: bool = True,
) -> ComparisonResult:
    """Load validation-only artifacts and build objective comparison tables.

    Missing experiments are represented as ``pending`` rows. Partially written
    or malformed experiments are marked ``incomplete`` or ``invalid`` and emit
    clear warnings instead of producing an ambiguous exception.
    """
    results_root = Path(results_root)
    output_dir = (
        results_root / "model_comparison" if output_dir is None else Path(output_dir)
    )

    rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    histories: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    metadata_by_model: dict[str, dict[str, Any]] = {}

    for spec in MODEL_SPECS:
        row, per_class, history, metadata, model_warnings = _load_model_artifacts(
            results_root,
            spec,
        )
        rows.append(row)
        warnings.extend(model_warnings)
        if per_class is not None and row["status"] != "invalid":
            per_class_frames.append(per_class)
        if history is not None and row["status"] != "invalid":
            histories[spec.display_name] = history
        if metadata is not None and row["status"] == "complete":
            metadata_by_model[spec.display_name] = metadata

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    for column in (
        "total_parameters",
        "trainable_parameters",
        "epochs_executed",
        "best_epoch",
        "batch_size",
        "num_workers",
    ):
        numeric = pd.to_numeric(
            summary[column],
            errors="coerce",
        )
        numeric = numeric.where(numeric.isna() | numeric.mod(1).eq(0))
        summary[column] = numeric.astype("Int64")
    per_class = (
        pd.concat(per_class_frames, ignore_index=True)[PER_CLASS_COLUMNS]
        if per_class_frames
        else pd.DataFrame(columns=PER_CLASS_COLUMNS)
    )
    if not per_class.empty:
        model_order = {spec.key: index for index, spec in enumerate(MODEL_SPECS)}
        per_class["_model_order"] = per_class["model_key"].map(model_order)
        per_class = (
            per_class.sort_values(["class_index", "_model_order"])
            .drop(columns="_model_order")
            .reset_index(drop=True)
        )

    warnings.extend(_validate_class_consistency(per_class))
    warnings.extend(_validate_hardware_consistency(metadata_by_model))
    warnings = _deduplicate(warnings)

    result = ComparisonResult(
        summary=summary,
        per_class=per_class,
        histories=histories,
        warnings=warnings,
    )
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.summary_path = output_dir / "validation_summary.csv"
        result.per_class_path = (
            output_dir / "validation_metrics_by_class_comparison.csv"
        )
        result.warnings_path = output_dir / "comparison_warnings.txt"
        summary.to_csv(result.summary_path, index=False)
        per_class.to_csv(result.per_class_path, index=False)
        _save_warnings(warnings, result.warnings_path)

    return result


def save_comparison_plots(
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save six simple bar charts without truncated axes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available = summary.loc[summary["status"].eq("complete")].copy()
    if available.empty:
        return []

    plot_specs = (
        ("accuracy", "Validation accuracy", "Accuracy", "validation_accuracy.png", 1.0),
        ("f1_macro", "F1 macro", "F1 macro", "validation_f1_macro.png", 1.0),
        (
            "parameters_millions",
            "Quantidade de parâmetros",
            "Milhões de parâmetros",
            "parameters_millions.png",
            None,
        ),
        (
            "training_time_minutes",
            "Tempo total de treinamento",
            "Minutos",
            "training_time_minutes.png",
            None,
        ),
        (
            "inference_milliseconds_per_image",
            "Tempo médio de inferência por imagem",
            "Milissegundos por imagem",
            "inference_milliseconds_per_image.png",
            None,
        ),
        (
            "throughput_images_per_second",
            "Throughput de validação",
            "Imagens por segundo",
            "throughput_images_per_second.png",
            None,
        ),
    )

    saved_paths: list[Path] = []
    for column, title, ylabel, filename, upper_limit in plot_specs:
        plot_data = available[["model_name", column]].dropna()
        if plot_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(plot_data["model_name"], plot_data[column], color="#4472C4")
        ax.set(title=title, xlabel="Modelo", ylabel=ylabel)
        ax.set_ylim(bottom=0, top=upper_limit)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def save_training_curve_plots(
    histories: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[Path]:
    """Plot real recorded epochs for loss and accuracy without interpolation."""
    if not histories:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = (
        (
            "train_loss",
            "validation_loss",
            "Loss de treino e validação",
            "Loss",
            "training_loss_curves.png",
        ),
        (
            "train_accuracy",
            "validation_accuracy",
            "Accuracy de treino e validação",
            "Accuracy",
            "training_accuracy_curves.png",
        ),
    )

    saved_paths: list[Path] = []
    for train_column, validation_column, title, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for model_name, history in histories.items():
            if not {"epoch", train_column, validation_column}.issubset(history.columns):
                continue
            ax.plot(
                history["epoch"],
                history[train_column],
                label=f"{model_name} - train",
            )
            ax.plot(
                history["epoch"],
                history[validation_column],
                linestyle="--",
                label=f"{model_name} - validation",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set(title=title, xlabel="Época", ylabel=ylabel)
        ax.set_xlim(left=1)
        if ylabel == "Accuracy":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def _load_model_artifacts(
    results_root: Path,
    spec: ModelSpec,
) -> tuple[
    dict[str, Any],
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict[str, Any] | None,
    list[str],
]:
    model_dir = results_root / spec.directory_name
    paths = {name: model_dir / name for name in spec.artifact_names()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    empty_artifacts = [
        name for name, path in paths.items() if path.is_file() and path.stat().st_size == 0
    ]
    row = {column: None for column in SUMMARY_COLUMNS}
    row.update(
        {
            "model_key": spec.key,
            "model_name": spec.display_name,
            "status": "pending" if not model_dir.is_dir() else "incomplete",
            "missing_artifacts": "; ".join(missing),
        }
    )
    warnings: list[str] = []
    if missing:
        warnings.append(
            f"{spec.display_name}: pendente/incompleto; faltam: {', '.join(missing)}."
        )

    parse_errors: list[str] = []
    if empty_artifacts:
        parse_errors.append(f"artefatos vazios: {', '.join(empty_artifacts)}")
    metadata: dict[str, Any] | None = None
    per_class: pd.DataFrame | None = None
    history: pd.DataFrame | None = None
    recorded_warmup: int | None = None

    metrics_path = model_dir / "validation_metrics.csv"
    if metrics_path.is_file():
        try:
            metrics = _read_single_row_csv(metrics_path)
            for column in (
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ):
                value = _required_number(metrics, column, metrics_path)
                if not 0 <= value <= 1:
                    raise ValueError(f"{column} fora do intervalo [0, 1]")
                row[column] = value
            if _required_integer(metrics, "num_samples", metrics_path, minimum=1) != EXPECTED_VALIDATION_SAMPLES:
                raise ValueError(
                    f"num_samples deve ser {EXPECTED_VALIDATION_SAMPLES} (validation)"
                )
            if _required_integer(metrics, "num_classes", metrics_path, minimum=1) != EXPECTED_NUM_CLASSES:
                raise ValueError(f"num_classes deve ser {EXPECTED_NUM_CLASSES}")
        except Exception as exc:
            parse_errors.append(f"validation_metrics.csv: {exc}")

    architecture_path = model_dir / "architecture_summary.json"
    if architecture_path.is_file():
        try:
            architecture = _read_json(architecture_path)
            row["total_parameters"] = _required_integer(
                architecture,
                "total_parameters",
                architecture_path,
                minimum=1,
            )
            row["trainable_parameters"] = _required_integer(
                architecture,
                "trainable_parameters",
                architecture_path,
                minimum=0,
            )
            if row["trainable_parameters"] > row["total_parameters"]:
                raise ValueError("trainable_parameters excede total_parameters")
            row["parameters_millions"] = row["total_parameters"] / 1_000_000
        except Exception as exc:
            parse_errors.append(f"architecture_summary.json: {exc}")

    timing_path = model_dir / "timing_summary.json"
    if timing_path.is_file():
        try:
            timing = _read_json(timing_path)
            training = _required_mapping(timing, "training", timing_path)
            inference = _required_mapping(
                timing,
                "validation_inference",
                timing_path,
            )
            if training.get("measurement_scope") != (
                "sum_of_epoch_train_and_validation_passes"
            ):
                raise ValueError(
                    "training.measurement_scope deve cobrir treino e validação por época"
                )
            if inference.get("measurement_scope") != "end_to_end_dataloader":
                raise ValueError(
                    "validation_inference.measurement_scope deve ser end_to_end_dataloader"
                )
            row["epochs_executed"] = _first_integer(
                training,
                ("epochs_executed", "epochs"),
                timing_path,
                minimum=1,
            )
            row["best_epoch"] = _required_integer(
                training,
                "best_epoch",
                timing_path,
                minimum=1,
            )
            if row["best_epoch"] > row["epochs_executed"]:
                raise ValueError("best_epoch excede epochs_executed")
            row["best_validation_loss"] = _required_number(
                training,
                "best_validation_loss",
                timing_path,
            )
            if row["best_validation_loss"] < 0:
                raise ValueError("best_validation_loss não pode ser negativo")
            row["training_time_seconds"] = _required_number(
                training,
                "total_time_seconds",
                timing_path,
            )
            if row["training_time_seconds"] <= 0:
                raise ValueError("total_time_seconds deve ser positivo")
            row["training_time_minutes"] = row["training_time_seconds"] / 60
            row["average_time_per_epoch_seconds"] = _required_number(
                training,
                "average_time_per_epoch_seconds",
                timing_path,
            )
            if row["average_time_per_epoch_seconds"] <= 0:
                raise ValueError("average_time_per_epoch_seconds deve ser positivo")
            row["inference_time_seconds"] = _required_number(
                inference,
                "total_time_seconds",
                timing_path,
            )
            if row["inference_time_seconds"] <= 0:
                raise ValueError("tempo total de inferência deve ser positivo")
            row["inference_seconds_per_image"] = _required_number(
                inference,
                "average_time_per_image_seconds",
                timing_path,
            )
            if row["inference_seconds_per_image"] <= 0:
                raise ValueError("tempo de inferência por imagem deve ser positivo")
            row["inference_milliseconds_per_image"] = (
                row["inference_seconds_per_image"] * 1000
            )
            row["throughput_images_per_second"] = _required_number(
                inference,
                "throughput_images_per_second",
                timing_path,
            )
            if row["throughput_images_per_second"] <= 0:
                raise ValueError("throughput deve ser positivo")
            if _required_integer(inference, "num_images", timing_path, minimum=1) != EXPECTED_VALIDATION_SAMPLES:
                raise ValueError(
                    "timing de inferência não cobre exatamente "
                    f"{EXPECTED_VALIDATION_SAMPLES} imagens de validação"
                )
            recorded_warmup = _required_integer(
                inference,
                "warmup_batches",
                timing_path,
                minimum=0,
            )
        except Exception as exc:
            parse_errors.append(f"timing_summary.json: {exc}")

    metadata_path = model_dir / "experiment_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = _read_json(metadata_path)
            _populate_hardware_fields(row, metadata)
            metadata_warnings, metadata_errors = _validate_metadata(
                spec.display_name,
                metadata,
            )
            warnings.extend(metadata_warnings)
            parse_errors.extend(metadata_errors)
            if recorded_warmup is not None:
                selected_device = str(metadata.get("selected_device", ""))
                expected_warmup = (
                    None
                    if not selected_device
                    else int(selected_device.startswith("cuda"))
                )
                if expected_warmup is not None and recorded_warmup != expected_warmup:
                    warnings.append(
                        f"{spec.display_name}: warm-up registrado={recorded_warmup}, "
                        f"esperado={expected_warmup} para {selected_device}."
                    )
        except Exception as exc:
            parse_errors.append(f"experiment_metadata.json: {exc}")

    per_class_path = model_dir / "validation_metrics_by_class.csv"
    if per_class_path.is_file():
        try:
            per_class = pd.read_csv(per_class_path)
            required = set(PER_CLASS_COLUMNS[2:])
            missing_columns = sorted(required - set(per_class.columns))
            if missing_columns:
                raise ValueError(f"colunas ausentes: {', '.join(missing_columns)}")
            per_class = per_class[PER_CLASS_COLUMNS[2:]].copy()
            _validate_per_class_frame(per_class, per_class_path)
            per_class.insert(0, "model_name", spec.display_name)
            per_class.insert(0, "model_key", spec.key)
            per_class = per_class.sort_values("class_index").reset_index(drop=True)
            if len(per_class) != EXPECTED_NUM_CLASSES:
                warnings.append(
                    f"{spec.display_name}: métricas por classe possuem "
                    f"{len(per_class)} linhas, esperado={EXPECTED_NUM_CLASSES}."
                )
        except Exception as exc:
            parse_errors.append(f"validation_metrics_by_class.csv: {exc}")
            per_class = None

    history_path = model_dir / spec.history_name
    if history_path.is_file():
        try:
            history = pd.read_csv(history_path)
            missing_columns = sorted(set(HISTORY_COLUMNS) - set(history.columns))
            if missing_columns:
                raise ValueError(f"colunas ausentes: {', '.join(missing_columns)}")
            history = history[HISTORY_COLUMNS].copy()
            _validate_history_frame(history, history_path)
        except Exception as exc:
            parse_errors.append(f"{spec.history_name}: {exc}")
            history = None

    parse_errors.extend(
        _validate_model_consistency(
            row=row,
            per_class=per_class,
            history=history,
            metadata=metadata,
        )
    )
    if parse_errors:
        row["status"] = "invalid"
        warnings.append(f"{spec.display_name}: " + " | ".join(parse_errors))
    elif not missing:
        row["status"] = "complete"

    return row, per_class, history, metadata, warnings


def _validate_metadata(
    model_name: str,
    metadata: dict[str, Any],
) -> tuple[list[str], list[str]]:
    expected = {
        "batch_size": 32,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "min_delta": 1e-4,
        "seed": 42,
        "input_image_size": EXPECTED_INPUT_SIZE,
        "num_classes": EXPECTED_NUM_CLASSES,
        "train_samples": EXPECTED_TRAIN_SAMPLES,
        "validation_samples": EXPECTED_VALIDATION_SAMPLES,
    }
    warnings: list[str] = []
    errors: list[str] = []
    for field, expected_value in expected.items():
        if field not in metadata:
            warnings.append(
                f"{model_name}: metadata sem {field}; valor não verificável."
            )
            continue
        actual = metadata.get(field)
        if field == "min_delta" and actual is not None:
            matches = bool(np.isclose(float(actual), float(expected_value)))
        else:
            matches = actual == expected_value
        if not matches:
            errors.append(
                f"{model_name}: metadata {field}={actual!r}, esperado={expected_value!r}."
            )

    protocol_expected = {
        "early_stopping_monitor": "validation_loss",
        "early_stopping_mode": "min",
        "evaluation_split": "validation",
        "test_split_used": False,
        "best_checkpoint_restored_before_validation": True,
    }
    for field, expected_value in protocol_expected.items():
        if field not in metadata:
            warnings.append(
                f"{model_name}: metadata legado sem {field}; protocolo não verificável "
                "explicitamente no artefato."
            )
        elif (
            metadata[field] is not expected_value
            if isinstance(expected_value, bool)
            else metadata[field] != expected_value
        ):
            errors.append(
                f"{model_name}: metadata {field}={metadata[field]!r}, "
                f"esperado={expected_value!r}."
            )

    for field in ("selected_device", "pytorch_version", "num_workers"):
        if metadata.get(field) is None:
            warnings.append(
                f"{model_name}: metadata de hardware/execução sem {field}."
            )

    selected_device = str(metadata.get("selected_device", ""))
    if selected_device.startswith("cuda"):
        for field in (
            "gpu_name",
            "gpu_total_memory_bytes",
            "pytorch_cuda_version",
        ):
            if metadata.get(field) is None:
                warnings.append(
                    f"{model_name}: execução CUDA sem metadata {field}."
                )

    preprocessing = metadata.get("preprocessing")
    if preprocessing is None:
        warnings.append(
            f"{model_name}: metadata legado sem descrição de preprocessing."
        )
    elif not isinstance(preprocessing, dict):
        errors.append(f"{model_name}: preprocessing deve ser um objeto JSON.")
    else:
        if preprocessing.get("image_size") != [224, 224]:
            errors.append(
                f"{model_name}: preprocessing.image_size deve ser [224, 224]."
            )
        if preprocessing.get("mean") != [0.485, 0.456, 0.406]:
            errors.append(f"{model_name}: média de normalização não é a do ImageNet.")
        if preprocessing.get("std") != [0.229, 0.224, 0.225]:
            errors.append(f"{model_name}: desvio de normalização não é o do ImageNet.")

        if model_name == "EfficientNet-B0":
            if preprocessing.get("compatible_with_shared_pipeline") is not True:
                errors.append(
                    "EfficientNet-B0: compatibilidade com o pipeline compartilhado "
                    "não está registrada."
                )
            if preprocessing.get("applied_resize") != [224, 224]:
                errors.append(
                    "EfficientNet-B0: applied_resize deve registrar [224, 224]."
                )
            if preprocessing.get("official_weights_preset_geometry_applied") is not False:
                errors.append(
                    "EfficientNet-B0: a não aplicação da geometria do preset oficial "
                    "não está registrada."
                )
            if not preprocessing.get("geometry_decision_rationale"):
                errors.append(
                    "EfficientNet-B0: justificativa da geometria compartilhada ausente."
                )
            if not isinstance(preprocessing.get("weights_preset_reference"), dict):
                errors.append(
                    "EfficientNet-B0: referência ao preset oficial ausente."
                )

    return warnings, errors


def _validate_per_class_frame(dataframe: pd.DataFrame, source: Path) -> None:
    numeric_columns = ["class_index", "precision", "recall", "f1_score", "support"]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
    numeric_values = dataframe[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"{source.name}: valores numéricos não finitos")

    indices = dataframe["class_index"].to_numpy(dtype=float)
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError(f"{source.name}: class_index deve ser inteiro")
    if dataframe["class_index"].astype(int).tolist() != list(
        range(EXPECTED_NUM_CLASSES)
    ):
        raise ValueError(
            f"{source.name}: class_index deve conter exatamente 0..{EXPECTED_NUM_CLASSES - 1}"
        )
    if dataframe["class_name"].isna().any() or not dataframe["class_name"].is_unique:
        raise ValueError(f"{source.name}: nomes de classe ausentes ou duplicados")

    for column in ("precision", "recall", "f1_score"):
        if not dataframe[column].between(0, 1, inclusive="both").all():
            raise ValueError(f"{source.name}: {column} fora do intervalo [0, 1]")
    supports = dataframe["support"].to_numpy(dtype=float)
    if not np.equal(supports, np.floor(supports)).all() or (supports < 0).any():
        raise ValueError(f"{source.name}: support deve ser inteiro não negativo")
    dataframe["class_index"] = dataframe["class_index"].astype(int)
    dataframe["support"] = dataframe["support"].astype(int)
    if int(dataframe["support"].sum()) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(
            f"{source.name}: soma de support deve ser {EXPECTED_VALIDATION_SAMPLES}"
        )


def _validate_history_frame(dataframe: pd.DataFrame, source: Path) -> None:
    for column in HISTORY_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
    if dataframe.empty:
        raise ValueError(f"{source.name}: histórico vazio")
    if not np.isfinite(dataframe[HISTORY_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"{source.name}: valores numéricos não finitos")

    epochs = dataframe["epoch"].to_numpy(dtype=float)
    if not np.equal(epochs, np.floor(epochs)).all():
        raise ValueError(f"{source.name}: epoch deve ser inteiro")
    dataframe["epoch"] = dataframe["epoch"].astype(int)
    if dataframe["epoch"].tolist() != list(range(1, len(dataframe) + 1)):
        raise ValueError(f"{source.name}: épocas devem ser sequenciais a partir de 1")
    for column in ("train_accuracy", "validation_accuracy"):
        if not dataframe[column].between(0, 1, inclusive="both").all():
            raise ValueError(f"{source.name}: {column} fora do intervalo [0, 1]")
    for column in ("train_loss", "validation_loss"):
        if (dataframe[column] < 0).any():
            raise ValueError(f"{source.name}: {column} contém valor negativo")
    if (dataframe["epoch_time_seconds"] <= 0).any():
        raise ValueError(f"{source.name}: epoch_time_seconds deve ser positivo")


def _validate_model_consistency(
    *,
    row: dict[str, Any],
    per_class: pd.DataFrame | None,
    history: pd.DataFrame | None,
    metadata: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    history_timing_fields = (
        "epochs_executed",
        "best_epoch",
        "best_validation_loss",
        "training_time_seconds",
        "average_time_per_epoch_seconds",
    )
    if history is not None and all(
        row.get(field) is not None for field in history_timing_fields
    ):
        epochs_executed = int(row["epochs_executed"])
        if len(history) != epochs_executed:
            errors.append("history: quantidade de épocas diverge do timing_summary")

        best_epoch = int(row["best_epoch"])
        best_rows = history.loc[history["epoch"].eq(best_epoch)]
        if best_rows.empty:
            errors.append("history: best_epoch do timing_summary não existe")
        else:
            recorded_loss = float(best_rows.iloc[0]["validation_loss"])
            if not np.isclose(
                recorded_loss,
                float(row["best_validation_loss"]),
                rtol=1e-7,
                atol=1e-10,
            ):
                errors.append("history: loss da best_epoch diverge do timing_summary")
        if not np.isclose(
            float(history["validation_loss"].min()),
            float(row["best_validation_loss"]),
            rtol=1e-7,
            atol=1e-10,
        ):
            errors.append("history: checkpoint não corresponde ao menor validation_loss")
        if not np.isclose(
            float(history["epoch_time_seconds"].sum()),
            float(row["training_time_seconds"]),
            rtol=1e-7,
            atol=1e-6,
        ):
            errors.append("history: soma dos tempos diverge do timing_summary")
        if not np.isclose(
            float(history["epoch_time_seconds"].mean()),
            float(row["average_time_per_epoch_seconds"]),
            rtol=1e-7,
            atol=1e-6,
        ):
            errors.append("history: tempo médio diverge do timing_summary")

    if per_class is not None:
        supports = per_class["support"].to_numpy(dtype=float)
        for overall_name, class_name in (
            ("precision_macro", "precision"),
            ("recall_macro", "recall"),
            ("f1_macro", "f1_score"),
        ):
            if row.get(overall_name) is not None and not np.isclose(
                float(per_class[class_name].mean()),
                float(row[overall_name]),
                rtol=1e-6,
                atol=1e-8,
            ):
                errors.append(f"per-class: {overall_name} diverge da média por classe")
        for overall_name, class_name in (
            ("precision_weighted", "precision"),
            ("recall_weighted", "recall"),
            ("f1_weighted", "f1_score"),
        ):
            if row.get(overall_name) is not None and not np.isclose(
                float(np.average(per_class[class_name], weights=supports)),
                float(row[overall_name]),
                rtol=1e-6,
                atol=1e-8,
            ):
                errors.append(f"per-class: {overall_name} diverge da média ponderada")

    inference_seconds = row.get("inference_seconds_per_image")
    inference_total = row.get("inference_time_seconds")
    throughput = row.get("throughput_images_per_second")
    if inference_seconds is not None and inference_total is not None:
        if not np.isclose(
            float(inference_seconds) * EXPECTED_VALIDATION_SAMPLES,
            float(inference_total),
            rtol=1e-6,
            atol=1e-8,
        ):
            errors.append("timing: total de inferência diverge do tempo por imagem")
    if inference_seconds is not None and throughput is not None:
        if not np.isclose(
            1 / float(inference_seconds),
            float(throughput),
            rtol=1e-6,
            atol=1e-8,
        ):
            errors.append("timing: throughput diverge do tempo por imagem")

    if metadata is not None:
        checkpoint_accuracy = metadata.get("best_model_validation_accuracy")
        if checkpoint_accuracy is not None and row.get("accuracy") is not None:
            try:
                matches = np.isclose(
                    float(checkpoint_accuracy),
                    float(row["accuracy"]),
                    rtol=1e-7,
                    atol=1e-10,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                errors.append(
                    "metadata: accuracy do checkpoint diverge de validation_metrics"
                )
    return errors


def _validate_class_consistency(per_class: pd.DataFrame) -> list[str]:
    if per_class.empty:
        return []
    warnings: list[str] = []
    reference_name: str | None = None
    reference: pd.DataFrame | None = None
    for model_name, group in per_class.groupby("model_name", sort=False):
        ordered = group.sort_values("class_index").reset_index(drop=True)
        if reference is None:
            reference = ordered[["class_index", "class_name", "support"]]
            reference_name = str(model_name)
            continue
        indices_match = ordered["class_index"].tolist() == reference[
            "class_index"
        ].tolist()
        names_match = ordered["class_name"].tolist() == reference["class_name"].tolist()
        support_match = ordered["support"].tolist() == reference["support"].tolist()
        if not indices_match or not names_match:
            warnings.append(
                f"Ordem/nomes de classes divergem entre {reference_name} e {model_name}."
            )
        if not support_match:
            warnings.append(
                f"Support por classe diverge entre {reference_name} e {model_name}."
            )
    return warnings


def _validate_hardware_consistency(
    metadata_by_model: dict[str, dict[str, Any]],
) -> list[str]:
    if len(metadata_by_model) < 2:
        return []
    warnings: list[str] = []
    hardware = {
        model: (
            metadata.get("selected_device"),
            metadata.get("gpu_name"),
            metadata.get("gpu_total_memory_bytes"),
        )
        for model, metadata in metadata_by_model.items()
    }
    if len(set(hardware.values())) > 1:
        details = "; ".join(f"{model}={value}" for model, value in hardware.items())
        warnings.append(
            "Hardware diferente entre modelos; tempos não são diretamente comparáveis. "
            + details
        )

    workers = {
        model: metadata.get("num_workers")
        for model, metadata in metadata_by_model.items()
    }
    if len(set(workers.values())) > 1:
        warnings.append(
            "num_workers diferente entre modelos pode afetar a comparação de tempos: "
            + "; ".join(f"{model}={value}" for model, value in workers.items())
        )
    return warnings


def _populate_hardware_fields(row: dict[str, Any], metadata: dict[str, Any]) -> None:
    row["gpu"] = metadata.get("gpu_name")
    memory_bytes = metadata.get("gpu_total_memory_bytes")
    row["gpu_total_memory_gb"] = (
        float(memory_bytes) / 1_000_000_000 if memory_bytes is not None else None
    )
    row["selected_device"] = metadata.get("selected_device")
    row["pytorch_version"] = metadata.get("pytorch_version")
    row["cuda_version"] = metadata.get("pytorch_cuda_version")
    row["batch_size"] = metadata.get("batch_size")
    row["num_workers"] = metadata.get("num_workers")
    input_size = metadata.get("input_image_size")
    row["input_size"] = (
        "x".join(str(value) for value in input_size)
        if isinstance(input_size, list)
        else input_size
    )


def _read_single_row_csv(path: Path) -> dict[str, Any]:
    dataframe = pd.read_csv(path)
    if len(dataframe) != 1:
        raise ValueError(f"esperada uma linha, encontradas {len(dataframe)}")
    return dataframe.iloc[0].to_dict()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise TypeError("o JSON raiz deve ser um objeto")
    return value


def _required_mapping(
    mapping: dict[str, Any],
    key: str,
    source: Path,
) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"{source.name}: objeto obrigatório ausente: {key}")
    return value


def _required_number(mapping: dict[str, Any], key: str, source: Path) -> float:
    if key not in mapping or pd.isna(mapping[key]):
        raise KeyError(f"{source.name}: campo obrigatório ausente: {key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{source.name}: {key} não é numérico")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{source.name}: {key} não é finito")
    return numeric_value


def _required_integer(
    mapping: dict[str, Any],
    key: str,
    source: Path,
    *,
    minimum: int | None = None,
) -> int:
    value = _required_number(mapping, key, source)
    if value != float(int(value)):
        raise ValueError(f"{source.name}: {key} deve ser inteiro")
    integer_value = int(value)
    if minimum is not None and integer_value < minimum:
        raise ValueError(f"{source.name}: {key} deve ser >= {minimum}")
    return integer_value


def _first_integer(
    mapping: dict[str, Any],
    keys: Iterable[str],
    source: Path,
    *,
    minimum: int | None = None,
) -> int:
    for key in keys:
        if key in mapping and not pd.isna(mapping[key]):
            return _required_integer(
                mapping,
                key,
                source,
                minimum=minimum,
            )
    raise KeyError(f"{source.name}: nenhum dos campos foi encontrado: {tuple(keys)}")


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _save_warnings(warnings: list[str], path: Path) -> None:
    text = (
        "\n".join(f"- {warning}" for warning in warnings)
        if warnings
        else "Nenhum aviso de comparabilidade."
    )
    path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "results",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Não salvar os gráficos comparativos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_root / "model_comparison"
    result = compare_validation_results(
        results_root=args.results_root,
        output_dir=output_dir,
        save=True,
    )
    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths.extend(save_comparison_plots(result.summary, output_dir))
        plot_paths.extend(save_training_curve_plots(result.histories, output_dir))

    print(result.summary.to_string(index=False))
    if result.warnings:
        print("\nAvisos:")
        for warning in result.warnings:
            print(f"- {warning}")
    print("\nArquivos salvos:")
    for path in (
        result.summary_path,
        result.per_class_path,
        result.warnings_path,
        *plot_paths,
    ):
        if path is not None:
            print(f"- {path}")


if __name__ == "__main__":
    main()
