"""
Dataset and DataLoader definitions for Breast Cancer Ultrasound Classification.
Includes medical-specific image augmentations and class imbalance weighting.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    NUM_WORKERS,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
)


def get_train_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """
    Data augmentations tailored for medical ultrasound images:
    - Random rotations (+/- 15 deg)
    - Random horizontal & subtle vertical flips
    - Color jitter (brightness/contrast) to simulate scanner transducer gain variations
    - Affine scaling / translation
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_eval_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic transforms for validation, testing, and inference."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for loading breast imaging files organized by category folders.
    Structure expected:
        root_dir/
            benign/
                img1.png ...
            malignant/
                img2.png ...
            normal/
                img3.png ...
    """

    def __init__(self, root_dir: Path, transform=None, class_names: List[str] = CLASS_NAMES):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for cls_name in self.class_names:
            cls_folder = self.root_dir / cls_name
            if not cls_folder.is_dir():
                continue
            cls_idx = self.class_to_idx[cls_name]
            for file_path in cls_folder.iterdir():
                if file_path.suffix.lower() in valid_extensions and "_mask" not in file_path.name.lower():
                    self.samples.append((file_path, cls_idx))

        if not self.samples:
            raise RuntimeError(f"Found 0 images in {self.root_dir} for classes: {self.class_names}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)

    def get_class_counts(self) -> Dict[str, int]:
        """Returns the distribution of samples per class."""
        counts = {cls: 0 for cls in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts


def calculate_class_weights(dataset: BreastCancerDataset, device: torch.device = None) -> torch.Tensor:
    """
    Computes inverse frequency weights to counteract class imbalance during cross-entropy training:
    weight_c = total_samples / (num_classes * count_c)
    """
    counts = dataset.get_class_counts()
    total = len(dataset)
    num_classes = len(counts)
    weights = [
        total / (num_classes * max(counts[cls], 1)) for cls in dataset.class_names
    ]
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if device:
        weight_tensor = weight_tensor.to(device)
    return weight_tensor


def get_dataloaders(
    train_dir: Path = TRAIN_DATA_DIR,
    val_dir: Path = VAL_DATA_DIR,
    test_dir: Path = TEST_DATA_DIR,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Instantiates training, validation, and testing DataLoaders along with loss class weights.
    """
    train_transform = get_train_transforms(image_size)
    eval_transform = get_eval_transforms(image_size)

    train_dataset = BreastCancerDataset(train_dir, transform=train_transform)
    val_dataset = BreastCancerDataset(val_dir, transform=eval_transform)
    test_dataset = BreastCancerDataset(test_dir, transform=eval_transform)

    class_weights = calculate_class_weights(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_weights
