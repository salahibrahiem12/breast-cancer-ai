"""
Configuration module for PyTorch Breast Cancer Detection.
Defines paths, hyperparameters, device settings, and class labels.
"""

from pathlib import Path
import torch

# Prevent Windows pagefile exhaustion by capping OpenMP thread pool
torch.set_num_threads(2)

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
VAL_DATA_DIR = PROCESSED_DATA_DIR / "val"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"
BUSI_PROCESSED_DATA_DIR = DATA_DIR / "processed_busi"
BUSI_TRAIN_DATA_DIR = BUSI_PROCESSED_DATA_DIR / "train"
BUSI_VAL_DATA_DIR = BUSI_PROCESSED_DATA_DIR / "val"
BUSI_TEST_DATA_DIR = BUSI_PROCESSED_DATA_DIR / "test"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models_saved"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure output directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Diagnostic Target Classes
# Standard Medical Breast Ultrasound Classes
CLASS_NAMES = ["benign", "malignant", "normal"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASS_NAMES)}

# Model & Training Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_WORKERS = 0  # 0 is safe for Windows multiprocessing
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15
PATIENCE = 5  # Early stopping patience
RANDOM_SEED = 42

# Device selection (Auto CUDA / CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
BEST_MODEL_PATH = MODELS_DIR / "best_breast_cancer_model.pth"
FINAL_MODEL_PATH = MODELS_DIR / "final_breast_cancer_model.pth"
BUSI_MODEL_PATH = MODELS_DIR / "best_breast_cancer_model_busi.pth"
BUSI_RESNET18_MODEL_PATH = MODELS_DIR / "best_breast_cancer_resnet18_busi.pth"

# ImageNet statistics for normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
