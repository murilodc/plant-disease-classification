"""Torchvision EfficientNet-B0 configured for PlantVillage transfer learning."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.models.efficientnet import EfficientNet

from plantvillage_pytorch import IMAGENET_MEAN, IMAGENET_STD


DEFAULT_NUM_CLASSES = 38
DEFAULT_WEIGHTS = EfficientNet_B0_Weights.DEFAULT
EFFICIENTNET_B0_STAGE_REPEATS = (1, 2, 2, 3, 3, 4, 1)
EFFICIENTNET_B0_STAGE_CHANNELS = (16, 24, 40, 80, 112, 192, 320)
EFFICIENTNET_B0_STAGE_KERNELS = (3, 3, 5, 3, 5, 5, 3)
EFFICIENTNET_B0_STAGE_STRIDES = (1, 2, 2, 2, 1, 2, 1)
EFFICIENTNET_B0_EXPANSION_RATIOS = (1, 6, 6, 6, 6, 6, 6)


def create_efficientnet_b0(
    num_classes: int = DEFAULT_NUM_CLASSES,
    weights: EfficientNet_B0_Weights | None = DEFAULT_WEIGHTS,
) -> EfficientNet:
    """Load EfficientNet-B0, replace only its final Linear and fine-tune all."""
    if num_classes < 1:
        raise ValueError("num_classes deve ser maior ou igual a 1.")

    model = efficientnet_b0(weights=weights)
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("model.classifier deve ser nn.Sequential.")
    if len(model.classifier) != 2:
        raise ValueError("O classifier esperado deve conter Dropout e Linear.")
    if not isinstance(model.classifier[0], nn.Dropout):
        raise TypeError("model.classifier[0] deve preservar o Dropout original.")
    if not isinstance(model.classifier[1], nn.Linear):
        raise TypeError("model.classifier[1] deve ser nn.Linear.")

    classifier_in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(classifier_in_features, num_classes)

    # Explicitly enforce the full-network fine-tuning strategy.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model


def efficientnet_b0_weights_name(
    weights: EfficientNet_B0_Weights | None,
) -> str:
    """Return a stable, human-readable torchvision weights identifier."""
    if weights is None:
        return "None"
    return f"{type(weights).__name__}.{weights.name}"


def validate_efficientnet_b0_preprocessing(
    weights: EfficientNet_B0_Weights = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Confirm shared normalization compatibility and document the preset."""
    weights_transform = weights.transforms()
    weights_mean = tuple(float(value) for value in weights_transform.mean)
    weights_std = tuple(float(value) for value in weights_transform.std)

    if weights_mean != IMAGENET_MEAN or weights_std != IMAGENET_STD:
        raise ValueError(
            "A normalizacao global nao e compativel com os pesos da "
            "EfficientNet-B0: "
            f"pipeline=({IMAGENET_MEAN}, {IMAGENET_STD}), "
            f"pesos=({weights_mean}, {weights_std})."
        )

    return {
        "normalization_source": efficientnet_b0_weights_name(weights),
        "mean": list(weights_mean),
        "std": list(weights_std),
        "compatible_with_shared_pipeline": True,
        "augmentation_policy": "same_as_baseline_resnet50_and_densenet201",
        "applied_resize": [224, 224],
        "official_weights_preset_geometry_applied": False,
        "geometry_decision_rationale": (
            "shared_224x224_pipeline_preserved_for_experimental_consistency"
        ),
        "weights_preset_reference": {
            "crop_size": list(weights_transform.crop_size),
            "resize_size": list(weights_transform.resize_size),
            "interpolation": str(weights_transform.interpolation),
        },
    }


def describe_efficientnet_b0(
    model: EfficientNet,
    weights: EfficientNet_B0_Weights | None = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Create a compact architecture summary for model comparisons."""
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("model.classifier deve ser nn.Sequential.")
    if len(model.classifier) != 2:
        raise ValueError("O classifier esperado deve conter dois modulos.")

    dropout = model.classifier[0]
    final_layer = model.classifier[1]
    if not isinstance(dropout, nn.Dropout) or not isinstance(final_layer, nn.Linear):
        raise TypeError("O classifier deve conter Dropout seguido de Linear.")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    final_layer_parameters = sum(
        parameter.numel() for parameter in final_layer.parameters()
    )

    stages = [
        {
            "name": f"features.{stage_index}",
            "block_type": "MBConv",
            "num_blocks": repeats,
            "output_channels": channels,
            "kernel_size": kernel_size,
            "first_block_stride": stride,
            "expansion_ratio": expansion_ratio,
        }
        for stage_index, (
            repeats,
            channels,
            kernel_size,
            stride,
            expansion_ratio,
        ) in enumerate(
            zip(
                EFFICIENTNET_B0_STAGE_REPEATS,
                EFFICIENTNET_B0_STAGE_CHANNELS,
                EFFICIENTNET_B0_STAGE_KERNELS,
                EFFICIENTNET_B0_STAGE_STRIDES,
                EFFICIENTNET_B0_EXPANSION_RATIOS,
            ),
            start=1,
        )
    ]

    return {
        "model_class": type(model).__name__,
        "architecture": "EfficientNet-B0",
        "torchvision_constructor": "torchvision.models.efficientnet_b0",
        "pretrained_weights_api": "EfficientNet_B0_Weights.DEFAULT",
        "pretrained_weights": efficientnet_b0_weights_name(weights),
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
            "name": "classifier.1",
            "type": type(final_layer).__name__,
            "in_features": final_layer.in_features,
            "out_features": final_layer.out_features,
            "parameter_count": final_layer_parameters,
        },
        "classifier_dropout": {
            "name": "classifier.0",
            "type": type(dropout).__name__,
            "p": float(dropout.p),
            "inplace": bool(dropout.inplace),
            "preserved_from_original": True,
        },
        "final_layer_replaced": "classifier.1",
        "original_final_layer_out_features": 1000,
        "final_layer_in_features": final_layer.in_features,
        "final_layer_out_features": final_layer.out_features,
        "final_layer_parameter_count": final_layer_parameters,
        "num_output_classes": final_layer.out_features,
        "main_blocks": {
            "stem": "features.0 (Conv2dNormActivation, 3 -> 32)",
            "mbconv_stages": stages,
            "total_mbconv_blocks": sum(EFFICIENTNET_B0_STAGE_REPEATS),
            "head": "features.8 (Conv2dNormActivation, 320 -> 1280)",
            "global_pooling": "avgpool (AdaptiveAvgPool2d)",
            "classifier": "Dropout(p=0.2) followed by Linear",
        },
    }


__all__ = [
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WEIGHTS",
    "EFFICIENTNET_B0_EXPANSION_RATIOS",
    "EFFICIENTNET_B0_STAGE_CHANNELS",
    "EFFICIENTNET_B0_STAGE_KERNELS",
    "EFFICIENTNET_B0_STAGE_REPEATS",
    "EFFICIENTNET_B0_STAGE_STRIDES",
    "create_efficientnet_b0",
    "describe_efficientnet_b0",
    "efficientnet_b0_weights_name",
    "validate_efficientnet_b0_preprocessing",
]
