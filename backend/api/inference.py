"""
Inference engine behind the FastAPI service.

Loads the trained classifier once, then for each uploaded image runs:
    preprocess -> classify -> Grad-CAM overlay -> (optional) LLaVA description.

Kept separate from main.py so it can be unit-tested without HTTP. The engine
degrades gracefully before a checkpoint exists: in dev mode it runs on the
ImageNet-pretrained backbone with an untrained head so the UI can be built
before GPU training has happened (predictions are meaningless but well-formed).

Owner: Member 4
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import torch
from PIL import Image

from src.classifier.model import build_classifier
from src.data.dataset import DEFAULT_IMG_SIZE, LABEL_NAMES, _default_transform
from src.explain.gradcam import gradcam_overlay


def _resolve_device(choice="auto") -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


class InferenceEngine:
    """Holds the model + preprocessing and serves single-image predictions."""

    def __init__(self, checkpoint=None, backbone="resnet50",
                 img_size=DEFAULT_IMG_SIZE, device="auto", allow_random=False,
                 enable_vlm=False):
        self.backbone = backbone
        self.img_size = img_size
        self.device = _resolve_device(device)
        self.enable_vlm = enable_vlm
        self.transform = _default_transform(img_size)

        self.checkpoint = str(checkpoint) if checkpoint else None
        self.random_weights = True  # flips to False once real weights load

        # freeze_backbone is irrelevant for inference; keep grad enabled so
        # Grad-CAM can attribute through the conv layers.
        self.model = build_classifier(backbone, num_classes=2,
                                      freeze_backbone=False).to(self.device)

        if self.checkpoint and Path(self.checkpoint).exists():
            state = torch.load(self.checkpoint, map_location=self.device)
            # accept both a raw state_dict and a {'state_dict': ...} wrapper
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            self.model.load_state_dict(state)
            self.random_weights = False
        elif not allow_random:
            raise FileNotFoundError(
                f"checkpoint not found: {self.checkpoint!r}. Train a model first "
                f"(python -m src.classifier.train --config ...) or start the API "
                f"with DEFECT_API_ALLOW_RANDOM=1 for an untrained dev server."
            )

        self.model.eval()

    @property
    def ready(self) -> bool:
        """True when real trained weights are loaded (not the dev fallback)."""
        return not self.random_weights

    def info(self) -> dict:
        return {
            "backbone": self.backbone,
            "checkpoint": self.checkpoint,
            "device": str(self.device),
            "img_size": self.img_size,
            "random_weights": self.random_weights,
            "vlm_enabled": self.enable_vlm,
        }

    def predict(self, image: Image.Image) -> dict:
        """Classify one PIL image and return the API payload dict."""
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        defect_prob = float(probs[1].item())

        overlay = gradcam_overlay(self.model, tensor, target_class=pred)
        heatmap_b64 = self._png_b64(overlay)

        description = self._describe(image, LABEL_NAMES[pred]) if self.enable_vlm else None

        return {
            "label": LABEL_NAMES[pred],
            "confidence": float(probs[pred].item()),
            "defect_probability": defect_prob,
            "heatmap_png_b64": heatmap_b64,
            "description": description,
            "model": self.info(),
        }

    @staticmethod
    def _png_b64(rgb_array) -> str:
        buf = io.BytesIO()
        Image.fromarray(rgb_array).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _describe(self, image: Image.Image, label: str):
        """Optional LLaVA description; never breaks /predict if it's unavailable."""
        try:
            from src.vlm.describe import describe_defect
            return describe_defect(image, label)
        except Exception:
            return None


def engine_from_env() -> InferenceEngine:
    """Build an InferenceEngine from environment variables (used by main.py).

    DEFECT_API_CHECKPOINT     path to a .pt checkpoint
    DEFECT_API_BACKBONE       resnet50 (default) | efficientnet_b0
    DEFECT_API_IMG_SIZE       default 256
    DEFECT_API_DEVICE         auto (default) | cpu | cuda
    DEFECT_API_ALLOW_RANDOM   1 -> run untrained if no checkpoint (dev mode)
    DEFECT_API_ENABLE_VLM     1 -> attempt LLaVA descriptions
    """
    return InferenceEngine(
        checkpoint=os.environ.get("DEFECT_API_CHECKPOINT"),
        backbone=os.environ.get("DEFECT_API_BACKBONE", "resnet50"),
        img_size=int(os.environ.get("DEFECT_API_IMG_SIZE", DEFAULT_IMG_SIZE)),
        device=os.environ.get("DEFECT_API_DEVICE", "auto"),
        allow_random=os.environ.get("DEFECT_API_ALLOW_RANDOM", "0") == "1",
        enable_vlm=os.environ.get("DEFECT_API_ENABLE_VLM", "0") == "1",
    )
