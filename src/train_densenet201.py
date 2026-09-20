"""Fine-tune and evaluate ImageNet-pretrained DenseNet201 on PlantVillage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn, optim

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from densenet201_model import (
    DEFAULT_WEIGHTS,
    create_densenet201,
    densenet201_weights_name,
    describe_densenet201,
    validate_densenet201_preprocessing,
)
from experiment_utils import (
    collect_experiment_metadata,
    compute_classification_metrics,
    measure_inference,
    save_classification_metrics,
    save_confusion_matrix,
    save_experiment_metadata,
    save_timing_summary,
    summarize_training_times,
)
from plantvillage_pytorch import IMAGE_SIZE, build_image_transforms, create_dataloaders
from train_baseline import (
    count_parameters,
    get_device,
    save_history,
    set_seed,
    train_model,
)
from train_resnet50 import restore_best_checkpoint


DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_CLASSES = 38
DEFAULT_NUM_WORKERS = 2
DEFAULT_SEED = 42
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4


def train_densenet201(
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
    """Fine-tune DenseNet201 and evaluate only the best model on validation."""
    set_seed(seed)
    device = get_device()
    metadata_csv = Path(metadata_csv)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "densenet201_best.pth"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"CSV de split nao encontrado: {metadata_csv}")
    if not image_root.exists():
        raise FileNotFoundError(f"Diretorio de imagens nao encontrado: {image_root}")

    preprocessing = validate_densenet201_preprocessing(DEFAULT_WEIGHTS)
    dataloaders = create_dataloaders(
        metadata_csv=metadata_csv,
        image_root=image_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
        transform_factory=build_image_transforms,
    )
    # Keep the reserved test loader outside every routine that performs work.
    train_validation_loaders = {
        split: dataloaders[split] for split in ("train", "validation")
    }

    model = create_densenet201(num_classes=num_classes, weights=DEFAULT_WEIGHTS)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Todos os parametros da DenseNet201 devem estar treinaveis.")

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
            "model": "DenseNet201",
            "architecture": "torchvision.models.densenet201",
            "weights": "ImageNet",
            "torchvision_weights_api": "DenseNet201_Weights.DEFAULT",
            "torchvision_weights_resolved": densenet201_weights_name(DEFAULT_WEIGHTS),
            "pretrained_dataset": "ImageNet-1K",
            "parameters_frozen": False,
            "all_parameters_trainable": True,
            "fine_tuning_strategy": "full_network",
            "loss": "CrossEntropyLoss",
            "early_stopping_monitor": "validation_loss",
            "early_stopping_mode": "min",
            "evaluation_split": "validation",
            "test_split_used": False,
            "preprocessing": {
                "image_size": [IMAGE_SIZE, IMAGE_SIZE],
                **preprocessing,
            },
        }
    )
    experiment_metadata_path = save_experiment_metadata(
        experiment_metadata,
        output_dir / "experiment_metadata.json",
    )

    architecture_summary = describe_densenet201(model, DEFAULT_WEIGHTS)
    architecture_path = save_experiment_metadata(
        architecture_summary,
        output_dir / "architecture_summary.json",
    )

    print(f"Device: {device}")
    print(model)
    print(f"Pesos: {densenet201_weights_name(DEFAULT_WEIGHTS)}")
    print("Estrategia: fine-tuning de toda a rede")
    print(f"Parametros totais: {count_parameters(model):,}")
    print(f"Parametros treinaveis: {count_parameters(model, trainable_only=True):,}")
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
    history_path = save_history(history, output_dir / "densenet201_history.csv")

    checkpoint = restore_best_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        expected_best_epoch=training_result.best_epoch,
    )

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
    print("\nDenseNet201 concluida")
    print(f"Epocas executadas: {training_result.epochs_executed} / {epochs}")
    print(f"Early stopping: {'sim' if training_result.early_stopping else 'nao'}")
    print(f"Melhor epoca: {training_result.best_epoch}")
    print(f"Melhor validation loss: {training_result.best_validation_loss:.6f}")
    print(
        "Validation accuracy do melhor modelo: "
        f"{checkpoint['validation_accuracy']:.6f}"
    )
    print(f"Tempo total de treinamento: {training_timing['total_time_seconds']:.2f}s")
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local or Colab execution."""
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
        default=project_root / "results" / "densenet201",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument(
        "--min-delta", type=float, default=DEFAULT_EARLY_STOPPING_MIN_DELTA
    )
    return parser.parse_args()


def main() -> None:
    """Run the full train/validation experiment from the command line."""
    args = parse_args()
    train_densenet201(
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


if __name__ == "__main__":
    main()
