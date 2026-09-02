"""Simple baseline CNN trained from scratch for PlantVillage classes."""

from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Compact convolutional neural network for 224x224 RGB images."""

    def __init__(self, num_classes: int = 38) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.20),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)


def create_baseline_cnn(num_classes: int = 38) -> BaselineCNN:
    """Create the baseline CNN for the requested number of classes."""
    return BaselineCNN(num_classes=num_classes)


__all__ = ["BaselineCNN", "create_baseline_cnn"]
