"""Model and preprocessing shared by training and evaluation."""

from typing import Any

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


CLASS_NAMES = ("benign", "malignant", "normal")
IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int = len(CLASS_NAMES), pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(model.fc.in_features, num_classes))
    return model


def load_checkpoint(path: str, device: Any) -> tuple[nn.Module, dict[str, Any]]:
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    class_order = checkpoint.get("class_order")
    if not isinstance(class_order, list) or len(class_order) < 2:
        raise ValueError("Checkpoint must contain at least two supported class labels.")
    if any(not isinstance(label, str) for label in class_order) or len(
        set(class_order)
    ) != len(class_order) or any(label not in CLASS_NAMES for label in class_order):
        raise ValueError("Checkpoint contains a class label outside the project class mapping.")
    if checkpoint.get("image_size") != IMAGE_SIZE:
        raise ValueError("Checkpoint preprocessing image_size is unsupported.")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Checkpoint preprocessing normalization is missing.")
    if (
        tuple(normalization.get("mean", ())) != IMAGE_MEAN
        or tuple(normalization.get("std", ())) != IMAGE_STD
    ):
        raise ValueError("Checkpoint preprocessing normalization is unsupported.")
    model = build_model(num_classes=len(class_order), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
