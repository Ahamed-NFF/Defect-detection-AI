"""
Grad-CAM heatmaps: show WHERE the classifier thinks the defect is.

Uses `pytorch-grad-cam`. Produces an overlay heatmap for the inference UI and
for qualitative figures in the report.

Usage (as a function called by the backend):
    overlay = gradcam_overlay(model, image_tensor, target_layer)   # HxWx3 uint8

target_layer defaults to the last conv block of the backbone (ResNet-50's
layer4, EfficientNet's features) when omitted. The image_tensor is the same
ImageNet-normalised tensor the classifier consumes; it is de-normalised
internally so the heatmap is blended over the real colours.

Owner: Member 3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Must match the normalisation used in src/data (ImageNet stats).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_target_layer(model):
    """Best-guess final conv block for the backbones we use."""
    if hasattr(model, "layer4"):           # ResNet family
        return model.layer4[-1]
    if hasattr(model, "features"):          # EfficientNet / VGG-style
        return model.features[-1]
    raise ValueError(
        "could not infer a target layer for this model; pass target_layer explicitly."
    )


def _denormalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """(C,H,W) normalised tensor -> (H,W,C) float RGB in [0,1] for blending."""
    t = image_tensor.detach().cpu().clone()
    for c in range(t.shape[0]):
        t[c] = t[c] * std[c] + mean[c]
    return t.permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)


def gradcam_overlay(model, image_tensor, target_layer=None, target_class=None,
                    save_path=None):
    """Return a Grad-CAM overlay (HxWx3 uint8 RGB) for one image.

    Args:
        model: the trained classifier (eval mode is set internally).
        image_tensor: (C,H,W) or (1,C,H,W) ImageNet-normalised tensor.
        target_layer: conv layer to attribute against; defaults to the backbone's
            last conv block.
        target_class: class index to explain (0=good, 1=defect). Defaults to the
            model's top prediction.
        save_path: optional path to also write the overlay as a PNG.

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
        raise ValueError(f"expected a single image (1,C,H,W), got {tuple(image_tensor.shape)}")

    device = next(model.parameters()).device
    # The backbone is frozen (requires_grad=False), so gradients would otherwise
    # stop at the trainable FC head and never reach the conv activations Grad-CAM
    # attributes against. Requiring grad on the INPUT puts the conv feature maps
    # back in the backward graph (frozen weights still aren't updated).
    input_tensor = image_tensor.to(device).clone().requires_grad_(True)

    layer = target_layer if target_layer is not None else default_target_layer(model)
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    was_training = model.training
    model.eval()
    try:
        with GradCAM(model=model, target_layers=[layer]) as cam:
            grayscale = cam(input_tensor=input_tensor, targets=targets)[0]  # (H,W) in [0,1]
    finally:
        if was_training:
            model.train()

    rgb = _denormalize(input_tensor[0])               # (H,W,3) float [0,1]
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)  # (H,W,3) uint8

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(save_path)

    return overlay
