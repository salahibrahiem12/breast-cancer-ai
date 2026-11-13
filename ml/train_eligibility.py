"""Train an ultrasound-like eligibility gate from a prepared grouped manifest."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.model import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, build_model
from ml.prepare_eligibility import validate_manifest

CLASS_ORDER = ("not_breast_ultrasound", "breast_ultrasound")


class EligibilityDataset(Dataset):
    def __init__(self, source: Path, manifest: Path, split: str):
        self.source = source.resolve()
        with manifest.open(newline="", encoding="utf-8") as file:
            all_rows = list(csv.DictReader(file))
        validate_manifest(manifest)
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} rows in {manifest}.")
        for row in self.rows:
            path = (self.source / row["path"]).resolve()
            try:
                path.relative_to(self.source)
            except ValueError as exc:
                raise ValueError(f"Image path escapes source: {row['path']}") from exc
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Image content changed after preparation: {row['path']}.")
        base = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
        if split == "train":
            base = [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        self.transform = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.source / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        label = 1 if row["label"] == "breast_ultrasound" else 0
        return tensor, label


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _collect_scores(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    scores, labels = [], []
    for images, targets in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1)[:, 1]
        scores.extend(probabilities.cpu().tolist())
        labels.extend(targets.tolist())
    return np.asarray(scores), np.asarray(labels)


def train(
    source: Path,
    manifest: Path,
    output: Path,
    epochs: int = 25,
    batch_size: int = 16,
    patience: int = 6,
    seed: int = 42,
    pretrained: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint exists: {output}; pass --overwrite to replace.")
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = EligibilityDataset(source, manifest, "train")
    validation_data = EligibilityDataset(source, manifest, "validation")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=0
    )
    labels = [
        1 if row["label"] == "breast_ultrasound" else 0 for row in train_data.rows
    ]
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training split must contain positives and negatives.")
    class_weights = counts.sum() / (2 * counts)
    model = build_model(num_classes=2, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, min_lr=1e-6
    )
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * targets.size(0)
        validation_scores, validation_labels = _collect_scores(
            model, validation_loader, device
        )
        validation_loss = float(
            nn.functional.binary_cross_entropy(
                torch.tensor(validation_scores, dtype=torch.float32),
                torch.tensor(validation_labels, dtype=torch.float32),
            ).item()
        )
        validation_accuracy = float(
            ((validation_scores >= 0.5) == validation_labels).mean()
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_data),
                "validation_loss": validation_loss,
                "validation_accuracy_at_0_5": validation_accuracy,
            }
        )
        print(
            f"epoch={epoch} train_loss={total_loss / len(train_data):.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_accuracy_at_0_5={validation_accuracy:.4f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No validation-selected eligibility checkpoint was produced.")
    with manifest.open(newline="", encoding="utf-8") as file:
        manifest_rows = list(csv.DictReader(file))
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_digest:
        raise ValueError("Eligibility manifest changed during training; no checkpoint was saved.")
    checkpoint = {
        "format_version": 1,
        "model_version": "breastcare-ultrasound-eligibility-v1",
        "architecture": "torchvision_resnet18",
        "model_state": best_state,
        "class_order": list(CLASS_ORDER),
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": IMAGE_MEAN, "std": IMAGE_STD},
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "training_source": str(source.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_digest,
        "evaluation_scope": (
            "patient_grouped" if manifest_rows and all(row["patient_id"] for row in manifest_rows)
            else "duplicate_grouped_image_level"
        ),
        "history": history,
        "public_inference_approved": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    report = {
        "checkpoint": str(output.resolve()),
        "manifest_sha256": manifest_digest,
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "evaluation_scope": checkpoint["evaluation_scope"],
        "threshold_selection_split": "threshold",
        "final_test_split": "test",
        "public_inference_approved": False,
    }
    output.with_suffix(".training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch-size, and patience must be positive.")
    train(
        args.source, args.manifest, args.output, args.epochs, args.batch_size,
        args.patience, args.seed, not args.no_pretrained, args.overwrite,
    )


if __name__ == "__main__":
    main()
