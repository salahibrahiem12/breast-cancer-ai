"""
PyTorch Breast Cancer Detection Package
"""

from src.config import CLASS_NAMES, DEVICE, NUM_CLASSES
from src.dataset import BreastCancerDataset, get_dataloaders
from src.models import BreastNet, build_model
from src.predict import predict_single_image

__version__ = "1.0.0"
