"""
Model architectures for Breast Cancer Ultrasound Classification.
Includes:
1. BreastNet: A custom modern CNN with residual connections and channel attention.
2. Transfer learning models (ResNet-18, ResNet-50, EfficientNet-B0) with custom diagnostic classification heads.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for adaptive channel feature recalibration."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """Convolutional building block with BatchNorm, ReLU, and optional SE-attention."""

    def __init__(self, in_channels: int, out_channels: int, use_se: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.se(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class BreastNet(nn.Module):
    """
    Experimental CNN for image classification research on B-mode ultrasound data.
    Features progressive channel expansion (3 -> 32 -> 64 -> 128 -> 256),
    channel attention mechanisms, global average pooling, and regularized dense head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 3):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 32, use_se=False)
        self.block2 = ConvBlock(32, 64, use_se=True)
        self.block3 = ConvBlock(64, 128, use_se=True)
        self.block4 = ConvBlock(128, 256, use_se=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        out = self.classifier(x)
        return out

    def get_cam_target_layer(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM visualization."""
        return self.block4.conv[3]  # The last Conv2d in block4


class TransferBreastModel(nn.Module):
    """
    Transfer learning wrapper supporting ResNet18, ResNet50, and EfficientNet backbones.
    Replaces the default ImageNet head with a multi-stage dropout classification head.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.features = base_model
            self.target_layer = self.features.layer4[-1].conv2

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.features = base_model
            self.target_layer = self.features.layer4[-1].conv3

        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.features = base_model
            self.target_layer = self.features.features[-1]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from ['resnet18', 'resnet50', 'efficientnet_b0']")

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.classifier(feat)
        return out

    def get_cam_target_layer(self) -> nn.Module:
        """Returns the appropriate target layer for Grad-CAM."""
        return self.target_layer


def build_model(
    model_name: str = "breastnet",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory function for instantiating models.
    Options: 'breastnet', 'resnet18', 'resnet50', 'efficientnet_b0'
    """
    model_name = model_name.lower()
    if model_name == "breastnet":
        return BreastNet(num_classes=num_classes)
    elif model_name in ["resnet18", "resnet50", "efficientnet_b0"]:
        return TransferBreastModel(
            backbone=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model name '{model_name}'.")
