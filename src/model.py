"""
Custom CNN architecture for CIFAR-10 image classification.

Architecture:
- 3 convolutional blocks with batch normalization and max pooling
- Fully connected classifier with dropout
- Designed for 32x32 RGB images (10 classes)
"""

import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """
    Custom CNN for CIFAR-10 classification.

    Architecture:
        Conv Block 1: Conv2d(3, 32) -> BN -> ReLU -> Conv2d(32, 32) -> BN -> ReLU -> MaxPool -> Dropout
        Conv Block 2: Conv2d(32, 64) -> BN -> ReLU -> Conv2d(64, 64) -> BN -> ReLU -> MaxPool -> Dropout
        Conv Block 3: Conv2d(64, 128) -> BN -> ReLU -> Conv2d(128, 128) -> BN -> ReLU -> MaxPool -> Dropout
        FC: Linear(128*4*4 -> 256) -> ReLU -> Dropout -> Linear(256 -> 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CIFAR10CNN()
    print(f"Architecture:\n{model}")
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    dummy = torch.randn(4, 3, 32, 32)
    output = model(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
