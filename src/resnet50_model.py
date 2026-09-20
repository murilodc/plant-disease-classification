"""Torchvision ResNet50 configured for PlantVillage transfer learning."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import ResNet

from plantvillage_pytorch import IMAGE_SIZE, SPLITS


DEFAULT_NUM_CLASSES = 38
DEFAULT_WEIGHTS = ResNet50_Weights.DEFAULT
RESNET50_STAGE_DEPTHS = (3, 4, 6, 3)


def create_resnet50(
    num_classes: int = DEFAULT_NUM_CLASSES,
    weights: ResNet50_Weights | None = DEFAULT_WEIGHTS,
) -> ResNet:
    """Load ResNet50 weights, replace only ``fc`` and enable full fine-tuning."""
    if num_classes < 1:
        raise ValueError("num_classes deve ser maior ou igual a 1.")

    model = resnet50(weights=weights)
    classifier_in_features = model.fc.in_features
    model.fc = nn.Linear(classifier_in_features, num_classes)

    # Explicit even though torchvision parameters are trainable by default: this
    # experiment fine-tunes the complete network, including the replaced head.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model


def build_resnet50_transforms(
    split: str,
    weights: ResNet50_Weights = DEFAULT_WEIGHTS,
) -> transforms.Compose:
    """Build 224x224 transforms normalized for the selected ImageNet weights.

    The geometric and augmentation policy intentionally matches the baseline.
    Mean and standard deviation come from the torchvision weights preset rather
    than from global mutable state, making the pretrained-model contract clear.
    """
    if split not in SPLITS:
        raise ValueError(f"Split invalido: {split}. Use um de {SPLITS}.")

    weights_transform = weights.transforms()
    normalize = transforms.Normalize(
        mean=tuple(weights_transform.mean),
        std=tuple(weights_transform.std),
    )

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def resnet50_weights_name(weights: ResNet50_Weights | None) -> str:
    """Return a stable, human-readable torchvision weights identifier."""
    if weights is None:
        return "None"
    return f"{type(weights).__name__}.{weights.name}"


def describe_resnet50(
    model: ResNet,
    weights: ResNet50_Weights | None = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Describe the transfer-learning choices and ResNet50's main blocks."""
    if not isinstance(model.fc, nn.Linear):
        raise TypeError("A camada final model.fc deve ser nn.Linear.")

    return {
        "architecture": "torchvision.models.resnet50",
        "pretrained_weights_api": "ResNet50_Weights.DEFAULT",
        "pretrained_weights": resnet50_weights_name(weights),
        "pretrained_dataset": "ImageNet-1K" if weights is not None else None,
        "fine_tuning_strategy": "full_network",
        "parameters_frozen": False,
        "final_layer_replaced": "fc",
        "original_final_layer_out_features": 1000,
        "final_layer_in_features": model.fc.in_features,
        "num_output_classes": model.fc.out_features,
        "main_blocks": {
            "stem": ["conv1", "bn1", "relu", "maxpool"],
            "residual_stages": [
                {
                    "name": f"layer{stage_index}",
                    "block_type": "Bottleneck",
                    "num_blocks": num_blocks,
                }
                for stage_index, num_blocks in enumerate(RESNET50_STAGE_DEPTHS, start=1)
            ],
            "global_pooling": "avgpool (AdaptiveAvgPool2d)",
            "classifier": "fc (Linear)",
        },
    }


__all__ = [
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WEIGHTS",
    "RESNET50_STAGE_DEPTHS",
    "build_resnet50_transforms",
    "create_resnet50",
    "describe_resnet50",
    "resnet50_weights_name",
]
