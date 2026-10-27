"""
Training engine for Breast Cancer Detection models.
Features:
- Weighted Cross-Entropy Loss to address medical class imbalance
- AdamW optimizer with Cosine Annealing / Plateau LR scheduling
- Strict validation tracking and early stopping
- Automatic checkpointing and learning curve generation
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure project root in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DEVICE,
    EPOCHS,
    FIGURES_DIR,
    FINAL_MODEL_PATH,
    LEARNING_RATE,
    PATIENCE,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from src.dataset import get_dataloaders
from src.models import build_model
from src.utils import plot_training_curves, save_checkpoint, set_seed


class Trainer:
    """Trainer orchestrator managing training epochs, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        class_weights: torch.Tensor = None,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        device: torch.device = DEVICE,
        checkpoint_path: Path = BEST_MODEL_PATH,
        patience: int = PATIENCE,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience

        # Loss function with medical class imbalance weighting
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Runs one full training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Runs validation pass on hold-out validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs: int = EPOCHS) -> Dict[str, List[float]]:
        """Executes complete training process with early stopping."""
        best_val_loss = float("inf")
        epochs_no_improve = 0

        print(f"\nTraining on Device: {self.device}")
        print(f"Total Epochs: {epochs} | Early Stopping Patience: {self.patience}\n" + "-" * 70)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, self.checkpoint_path)
                status_mark = "[BEST]"
            else:
                epochs_no_improve += 1
                status_mark = ""

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:6.2f}% || "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:6.2f}% | "
                f"LR: {current_lr:.1e} {status_mark}"
            )

            # Early stopping check
            if epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered: No validation loss improvement for {self.patience} epochs.")
                break

        print("-" * 70)
        print(f"Training Complete! Best Validation Loss: {best_val_loss:.4f}")
        print(f"Saved Best Model to: {self.checkpoint_path}")
        return self.history


def train_pipeline(
    model_name: str = "breastnet",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    seed: int = RANDOM_SEED,
):
    """Convenience function to run the full training pipeline."""
    set_seed(seed)
    train_loader, val_loader, _, class_weights = get_dataloaders(batch_size=batch_size)

    model = build_model(model_name=model_name)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        lr=lr,
        patience=patience,
    )

    history = trainer.fit(epochs=epochs)

    # Plot and save learning curves
    curve_path = FIGURES_DIR / f"learning_curves_{model_name}.png"
    plot_training_curves(history, save_path=curve_path)
    print(f"Saved learning curves to: {curve_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch Breast Cancer Detection Model")
    parser.add_argument(
        "--model",
        type=str,
        default="breastnet",
        choices=["breastnet", "resnet18", "resnet50", "efficientnet_b0"],
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    args = parser.parse_args()

    train_pipeline(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
