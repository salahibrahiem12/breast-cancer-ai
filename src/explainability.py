"""Experimental Grad-CAM utilities; inference is blocked by the safety gate."""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import CLASS_NAMES, DEVICE, FIGURES_DIR, IMAGE_SIZE, NORM_MEAN, NORM_STD
from src.dataset import get_eval_transforms
from src.eligibility import (
    UNSUPPORTED_IMAGE_MESSAGE,
    UnsupportedImageError,
    decode_rgb_image,
    require_verified_ultrasound,
)


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    Hooks into activations and gradients of the target convolutional layer.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes Grad-CAM heatmap for a given input image tensor (1, C, H, W).
        Returns:
            heatmap: (H, W) float array normalized in [0, 1]
            pred_class: predicted class index
            confidence: uncalibrated softmax score (not a probability estimate)
        """
        raise UnsupportedImageError(UNSUPPORTED_IMAGE_MESSAGE)
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        confidence = probabilities[0, target_class].item()

        # Backward pass on the score of the target class
        score = output[0, target_class]
        score.backward()

        # Pool the gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Pass through ReLU to focus only on features that positively contribute
        cam = F.relu(cam)

        # Upsample to match input tensor spatial size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence


def overlay_cam_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Overlays the Grad-CAM heatmap onto the original image."""
    image_rgb = image.convert("RGB")
    w, h = image_rgb.size

    # Colorize heatmap using matplotlib colormap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Discard alpha
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored_heatmap).resize((w, h), Image.Resampling.BILINEAR)

    # Blend original image and heatmap
    blended = Image.blend(image_rgb, heatmap_img, alpha=alpha)
    return blended


def explain_prediction(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device = DEVICE,
    save_path: Optional[Path] = None,
) -> Tuple[str, float, Image.Image]:
    """
    Runs experimental Grad-CAM only after ultrasound eligibility is verified.
    The current safety gate rejects every image; Grad-CAM is not lesion localization.
    """
    image_path = Path(image_path)
    raw_image = decode_rgb_image(image_path)
    require_verified_ultrasound(raw_image)

    eval_transform = get_eval_transforms()
    input_tensor = eval_transform(raw_image).unsqueeze(0).to(device)

    # Determine target layer for Grad-CAM
    if hasattr(model, "get_cam_target_layer"):
        target_layer = model.get_cam_target_layer()
    else:
        # Fallback to the last convolutional layer found
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        target_layer = conv_layers[-1]

    cam_engine = GradCAM(model, target_layer)
    heatmap, pred_idx, confidence = cam_engine.generate_cam(input_tensor)

    pred_class = CLASS_NAMES[pred_idx]
    cam_overlay = overlay_cam_on_image(raw_image, heatmap, alpha=0.5)

    # Create diagnostic visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_image)
    axes[0].set_title(f"Original Ultrasound\nFile: {image_path.name}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap\n(Suspicion Activation)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(cam_overlay)
    axes[2].set_title(
        f"Experimental Activation Overlay\nClass: {pred_class.upper()} "
        f"(uncalibrated score {confidence*100:.1f}%)",
        fontsize=12,
        fontweight="bold",
        color="darkred" if pred_class == "malignant" else "darkgreen",
    )
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pred_class, confidence, cam_overlay
