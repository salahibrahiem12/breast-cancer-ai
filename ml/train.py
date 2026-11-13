"""Train a versioned ResNet-18 with grouped splits and validation-only selection."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.model import CLASS_NAMES, IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, build_model


class ManifestDataset(Dataset):
    def __init__(
        self,
        source: Path,
        manifest: Path,
        split: str,
        class_names: tuple[str, ...] = CLASS_NAMES,
        augment: bool = True,
    ):
        self.class_names = class_names
        with manifest.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.samples = [
            row for row in rows
            if row["split"] == split and row["label"] in self.class_names
        ]
        if not self.samples:
            raise ValueError(f"No {split} samples in manifest {manifest}.")
        self.source = source.resolve()
        for row in self.samples:
            sample_path = (self.source / row["path"]).resolve()
            try:
                sample_path.relative_to(self.source)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest image path escapes the source directory: {row['path']}"
                ) from exc
        if split == "train" and augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=8),
                    transforms.ColorJitter(brightness=0.12, contrast=0.12),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.samples[index]
        with Image.open(self.source / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_names.index(row["label"]), row["path"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    logits, labels, paths = [], [], []
    for images, targets, sample_paths in loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(targets)
        paths.extend(sample_paths)
    return torch.cat(logits), torch.cat(labels), paths


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperatures = torch.logspace(-1, 1, steps=301, dtype=torch.float64)
    logits = logits.to(torch.float64)
    labels = labels.to(torch.long)
    losses = torch.stack(
        [nn.functional.cross_entropy(logits / temperature, labels) for temperature in temperatures]
    )
    return float(temperatures[int(losses.argmin())])


def class_weights(dataset: ManifestDataset) -> torch.Tensor:
    counts = np.bincount(
        [dataset.class_names.index(row["label"]) for row in dataset.samples],
        minlength=len(dataset.class_names),
    )
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing a class: {counts.tolist()}.")
    weights = counts.sum() / (len(dataset.class_names) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def validation_selection_key(
    malignant_recall: float,
    macro_f1: float,
    validation_loss: float,
    policy: str,
) -> tuple[float, float, float]:
    if policy == "malignant_recall_then_macro_f1":
        return malignant_recall, macro_f1, -validation_loss
    if policy == "macro_f1_then_malignant_recall":
        return macro_f1, malignant_recall, -validation_loss
    raise ValueError(f"Unsupported validation selection policy: {policy}")


def train(
    source: Path,
    manifest: Path,
    output: Path,
    epochs: int = 25,
    batch_size: int = 16,
    patience: int = 6,
    learning_rate: float = 1e-4,
    seed: int = 42,
    pretrained: bool = True,
    overwrite: bool = False,
    use_class_weights: bool = True,
    augment: bool = True,
    freeze_backbone: bool = False,
    selection_policy: str = "malignant_recall_then_macro_f1",
) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint exists: {output}. Use --overwrite to replace it.")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preparation_report_path = manifest.parent / "preparation_report.json"
    if not preparation_report_path.is_file():
        raise FileNotFoundError(
            f"Missing preparation report {preparation_report_path}; create splits with ml.prepare_dataset."
        )
    preparation = json.loads(preparation_report_path.read_text(encoding="utf-8"))
    class_names = tuple(preparation.get("class_order", ()))
    if len(class_names) < 2 or any(label not in CLASS_NAMES for label in class_names):
        raise ValueError("Preparation report does not contain a valid class_order.")
    if freeze_backbone and not pretrained:
        raise ValueError("A frozen backbone requires pretrained weights.")
    if selection_policy not in {
        "malignant_recall_then_macro_f1",
        "macro_f1_then_malignant_recall",
    }:
        raise ValueError(f"Unsupported validation selection policy: {selection_policy}")
    train_dataset = ManifestDataset(source, manifest, "train", class_names, augment)
    validation_dataset = ManifestDataset(source, manifest, "validation", class_names)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes=len(class_names), pretrained=pretrained).to(device)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_dataset).to(device) if use_class_weights else None,
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_backbone:
            model.eval()
            model.fc.train()
        total_loss = 0.0
        targets_seen, predictions_seen = [], []
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            targets_seen.extend(labels.detach().cpu().tolist())
            predictions_seen.extend(logits.detach().argmax(dim=1).cpu().tolist())

        validation_logits, validation_labels, _ = collect_logits(
            model, validation_loader, device
        )
        validation_loss = float(
            nn.functional.cross_entropy(validation_logits, validation_labels).item()
        )
        validation_predictions = validation_logits.argmax(dim=1).numpy()
        validation_targets = validation_labels.numpy()
        macro_f1 = float(
            f1_score(validation_targets, validation_predictions, average="macro", zero_division=0)
        )
        malignant_recall = (
            float(
                recall_score(
                    validation_targets,
                    validation_predictions,
                    labels=[class_names.index("malignant")],
                    average=None,
                    zero_division=0,
                )[0]
            )
            if "malignant" in class_names else 0.0
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_dataset),
                "train_macro_f1": float(
                    f1_score(targets_seen, predictions_seen, average="macro", zero_division=0)
                ),
                "validation_loss": validation_loss,
                "validation_macro_f1": macro_f1,
                "validation_malignant_recall": malignant_recall,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch={epoch} train_loss={total_loss / len(train_dataset):.4f} "
            f"val_loss={validation_loss:.4f} val_macro_f1={macro_f1:.4f} "
            f"val_malignant_recall={malignant_recall:.4f}"
        )

        selection_key = validation_selection_key(
            malignant_recall, macro_f1, validation_loss, selection_policy
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early_stopping=patience_{patience}")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a selected validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    validation_logits, validation_labels, _ = collect_logits(
        model, validation_loader, device
    )
    temperature = fit_temperature(validation_logits, validation_labels)

    with manifest.open(newline="", encoding="utf-8") as file:
        manifest_rows = list(csv.DictReader(file))
    patient_grouped = bool(manifest_rows) and all(row.get("patient_id") for row in manifest_rows)
    checkpoint = {
        "format_version": 1,
        "model_version": "breastcare-resnet18-v1",
        "architecture": "torchvision_resnet18",
        "model_state": best_state,
        "class_order": list(class_names),
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": IMAGE_MEAN, "std": IMAGE_STD},
        "temperature": temperature,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_selection": f"{selection_policy}_then_loss",
        "training_config": {
            "class_weighting": "balanced" if use_class_weights else "none",
            "augmentation": augment,
            "backbone": "frozen_pretrained" if freeze_backbone else (
                "fine_tuned_pretrained" if pretrained else "fine_tuned_random_init"
            ),
            "selection_policy": selection_policy,
        },
        "seed": seed,
        "training_source": str(source.resolve()),
        "manifest": str(manifest.resolve()),
        "evaluation_scope": (
            "patient_grouped_internal_holdout" if patient_grouped else "image_level_internal_holdout"
        ),
        "history": history,
        "inference_enabled": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary_output)
    temporary_output.replace(output)
    report = {
        "checkpoint": str(output),
        "model_version": checkpoint["model_version"],
        "selected_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_temperature": temperature,
        "evaluation_scope": checkpoint["evaluation_scope"],
        "training_config": checkpoint["training_config"],
        "inference_enabled": False,
        "history": history,
    }
    report_path = output.with_suffix(".training.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("models_saved/versioned/breastcare_resnet18_v1.pth"),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting for a controlled ablation.",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable train-only image augmentation for a controlled ablation.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the classifier head; requires pretrained weights.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=(
            "malignant_recall_then_macro_f1",
            "macro_f1_then_malignant_recall",
        ),
        default="malignant_recall_then_macro_f1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive.")
    train(
        args.source, args.manifest, args.output, args.epochs, args.batch_size,
        args.patience, args.learning_rate, args.seed, not args.no_pretrained,
        args.overwrite, not args.no_class_weights, not args.no_augmentation,
        args.freeze_backbone, args.selection_policy,
    )


if __name__ == "__main__":
    main()
