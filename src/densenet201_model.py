"""Torchvision DenseNet201 configured for PlantVillage transfer learning."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import DenseNet201_Weights, densenet201
from torchvision.models.densenet import DenseNet

from plantvillage_pytorch import IMAGENET_MEAN, IMAGENET_STD


DEFAULT_NUM_CLASSES = 38
DEFAULT_WEIGHTS = DenseNet201_Weights.DEFAULT
DENSENET201_BLOCK_CONFIG = (6, 12, 48, 32)
DENSENET201_GROWTH_RATE = 32


def create_densenet201(
    num_classes: int = DEFAULT_NUM_CLASSES,
    weights: DenseNet201_Weights | None = DEFAULT_WEIGHTS,
) -> DenseNet:
    """Load DenseNet201 weights, replace only ``classifier`` and fine-tune all."""
    if num_classes < 1:
        raise ValueError("num_classes deve ser maior ou igual a 1.")

    model = densenet201(weights=weights)
    classifier_in_features = model.classifier.in_features
    model.classifier = nn.Linear(classifier_in_features, num_classes)

    # Torchvision leaves parameters trainable by default. Keeping this explicit
    # documents and enforces the full-network fine-tuning strategy.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model


def densenet201_weights_name(weights: DenseNet201_Weights | None) -> str:
    """Return a stable, human-readable torchvision weights identifier."""
    if weights is None:
        return "None"
    return f"{type(weights).__name__}.{weights.name}"


def validate_densenet201_preprocessing(
    weights: DenseNet201_Weights = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Confirm the shared pipeline normalization matches the weights preset."""
    weights_transform = weights.transforms()
    weights_mean = tuple(float(value) for value in weights_transform.mean)
    weights_std = tuple(float(value) for value in weights_transform.std)

    if weights_mean != IMAGENET_MEAN or weights_std != IMAGENET_STD:
        raise ValueError(
            "A normalizacao global nao e compativel com os pesos da DenseNet201: "
            f"pipeline=({IMAGENET_MEAN}, {IMAGENET_STD}), "
            f"pesos=({weights_mean}, {weights_std})."
        )

    return {
        "normalization_source": densenet201_weights_name(weights),
        "mean": list(weights_mean),
        "std": list(weights_std),
        "compatible_with_shared_pipeline": True,
        "augmentation_policy": "same_as_baseline_and_resnet50",
    }


def describe_densenet201(
    model: DenseNet,
    weights: DenseNet201_Weights | None = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Create a compact architecture summary for later model comparisons."""
    if not isinstance(model.classifier, nn.Linear):
        raise TypeError("A camada final model.classifier deve ser nn.Linear.")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    classifier_parameters = sum(
        parameter.numel() for parameter in model.classifier.parameters()
    )

    return {
        "model_class": type(model).__name__,
        "architecture": "DenseNet201",
        "torchvision_constructor": "torchvision.models.densenet201",
        "pretrained_weights_api": "DenseNet201_Weights.DEFAULT",
        "pretrained_weights": densenet201_weights_name(weights),
        "pretrained_dataset": "ImageNet-1K" if weights is not None else None,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "fine_tuning_strategy": "full_network",
        "parameters_frozen": False,
        "num_convolutional_layers": sum(
            1 for module in model.modules() if isinstance(module, nn.Conv2d)
        ),
        "num_batch_norm_layers": sum(
            1 for module in model.modules() if isinstance(module, nn.BatchNorm2d)
        ),
        "classification_layer": {
            "name": "classifier",
            "type": type(model.classifier).__name__,
            "in_features": model.classifier.in_features,
            "out_features": model.classifier.out_features,
            "parameter_count": classifier_parameters,
        },
        "final_layer_replaced": "classifier",
        "original_final_layer_out_features": 1000,
        "final_layer_in_features": model.classifier.in_features,
        "final_layer_out_features": model.classifier.out_features,
        "final_layer_parameter_count": classifier_parameters,
        "num_output_classes": model.classifier.out_features,
        "main_blocks": {
            "stem": ["conv0", "norm0", "relu0", "pool0"],
            "growth_rate": DENSENET201_GROWTH_RATE,
            "dense_blocks": [
                {
                    "name": f"denseblock{block_index}",
                    "num_dense_layers": num_layers,
                }
                for block_index, num_layers in enumerate(
                    DENSENET201_BLOCK_CONFIG,
                    start=1,
                )
            ],
            "transition_layers": ["transition1", "transition2", "transition3"],
            "final_normalization": "norm5",
            "global_pooling": "adaptive average pooling in forward pass",
            "classifier": "classifier (Linear)",
        },
    }


__all__ = [
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WEIGHTS",
    "DENSENET201_BLOCK_CONFIG",
    "DENSENET201_GROWTH_RATE",
    "create_densenet201",
    "densenet201_weights_name",
    "describe_densenet201",
    "validate_densenet201_preprocessing",
]
