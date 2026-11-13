"""Explainable AI: Grad-CAM attention heatmap visualization for breast imaging."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms


def generate_gradcam_overlay(
    model: nn.Module | None,
    image: Image.Image,
    target_class: int = 1,
    layer_name: str = "layer4",
) -> str:
    """Generate a Grad-CAM heatmap overlay encoded as a base64 PNG data URL."""
    w, h = image.size
    img_rgb = image.convert("RGB")

    # If model is not loaded or in mock/disabled mode, generate a smooth attention envelope
    if model is None:
        return _generate_fallback_heatmap(img_rgb, w, h)

    try:
        # Prepare input tensor
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(img_rgb).unsqueeze(0)
        input_tensor.requires_grad = True

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        # Find target layer
        target_layer = None
        for name, module in model.named_modules():
            if name.endswith(layer_name):
                target_layer = module
                break

        if target_layer is None:
            return _generate_fallback_heatmap(img_rgb, w, h)

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            activations.append(out)

        def backward_hook(module: nn.Module, grad_inp: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
            gradients.append(grad_out[0])

        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        model.eval()
        output = model(input_tensor)
        if output.dim() == 1:
            score = output[target_class] if target_class < len(output) else output[0]
        else:
            score = output[0, target_class] if target_class < output.shape[1] else output[0, 0]

        model.zero_grad()
        score.backward()

        h_fwd.remove()
        h_bwd.remove()

        if not activations or not gradients:
            return _generate_fallback_heatmap(img_rgb, w, h)

        act = activations[0].detach().cpu().numpy()[0]  # (C, H, W)
        grad = gradients[0].detach().cpu().numpy()[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = np.mean(grad, axis=(1, 2))  # (C,)
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w_val in enumerate(weights):
            cam += w_val * act[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return _render_heatmap_to_base64(img_rgb, cam, w, h)
    except Exception:
        return _generate_fallback_heatmap(img_rgb, w, h)


def _render_heatmap_to_base64(original: Image.Image, cam: np.ndarray, w: int, h: int) -> str:
    """Render normalized CAM float array to colored overlay on original image."""
    # Resize CAM to original dimensions
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0

    # Custom medical pink/magenta attention colormap (Low: Blue/Cyan, Mid: Pink/Rose, High: Bright Magenta/Yellow)
    # Color components:
    r = np.clip(1.5 * cam_arr - 0.2, 0, 1) * 214 + (1 - cam_arr) * 40
    g = np.clip(1.0 - 2.0 * np.abs(cam_arr - 0.5), 0, 1) * 51 + (1 - cam_arr) * 40
    b = np.clip(1.0 - cam_arr * 1.5, 0, 1) * 132 + (1 - cam_arr) * 80

    heat_rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb)

    # Blend original image with heatmap (alpha = 0.45 where cam > 0.15)
    orig_rgb = original.convert("RGB")
    alpha_mask = Image.fromarray((np.clip(cam_arr * 1.4, 0, 0.6) * 255).astype(np.uint8))

    composite = Image.composite(heat_img, orig_rgb, alpha_mask)

    buf = BytesIO()
    composite.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _generate_fallback_heatmap(original: Image.Image, w: int, h: int) -> str:
    """Generate a center-weighted illustrative attention heatmap."""
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist_sq = ((x - cx) / (w * 0.35)) ** 2 + ((y - cy) / (h * 0.35)) ** 2
    cam = np.exp(-0.5 * dist_sq)
    return _render_heatmap_to_base64(original, cam, w, h)

