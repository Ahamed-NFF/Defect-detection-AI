"""
Generate Grad-CAM figures for the report.

For each category, loads a trained checkpoint and saves side-by-side
(original | Grad-CAM overlay) PNGs for a few real test-defect images, so the
report can show the classifier is attending to the actual defect region.

Run (on the machine that has the checkpoints + data, i.e. the GPU box):
    python scripts/make_gradcam_figs.py --categories bottle hazelnut carpet \
        --run-suffix ours --per-category 4

Outputs to reports/figures/gradcam/<category>_<i>.png (tracked in git so they
sync to the report machine).

Owner: Member 3 / Member 4 (report figures)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/make_gradcam_figs.py): put the
# repo root on sys.path so `import src...` resolves like the `python -m` modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src.classifier.model import build_classifier
from src.data.dataset import DEFECT, DefectDataset, _default_transform
from src.explain.gradcam import gradcam_overlay


def _resolve_device(choice="auto") -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def _side_by_side(orig: Image.Image, overlay_rgb, img_size: int) -> Image.Image:
    """Compose [resized original | Grad-CAM overlay] into one image."""
    left = orig.convert("RGB").resize((img_size, img_size))
    right = Image.fromarray(overlay_rgb)
    canvas = Image.new("RGB", (img_size * 2 + 8, img_size), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (img_size + 8, 0))
    return canvas


def main(argv=None):
    p = argparse.ArgumentParser(description="Make Grad-CAM report figures.")
    p.add_argument("--categories", nargs="+", default=["bottle", "hazelnut", "carpet"])
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--ckpt-dir", default="experiments/checkpoints")
    p.add_argument("--run-suffix", default="ours",
                   help="checkpoint = <ckpt-dir>/<category>_<suffix>.pt")
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--per-category", type=int, default=4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--out", default="reports/figures/gradcam")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)

    dev = _resolve_device(args.device)
    transform = _default_transform(args.img_size)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in args.categories:
        ckpt = Path(args.ckpt_dir) / f"{cat}_{args.run_suffix}.pt"
        if not ckpt.exists():
            print(f"[{cat}] skip — checkpoint not found: {ckpt}")
            continue

        net = build_classifier(args.backbone, num_classes=2, freeze_backbone=False).to(dev)
        net.load_state_dict(torch.load(ckpt, map_location=dev))
        net.eval()

        ds = DefectDataset(Path(args.data_root) / cat, "test", transform)
        defects = [path for path, label in ds.samples if label == DEFECT]
        if not defects:
            print(f"[{cat}] skip — no test defect images found")
            continue

        n = min(args.per_category, len(defects))
        print(f"[{cat}] {n} figure(s) from {ckpt.name}")
        for i, path in enumerate(defects[:n]):
            with Image.open(path) as im:
                orig = im.convert("RGB")
                tensor = transform(orig).unsqueeze(0)
            overlay = gradcam_overlay(net, tensor, target_class=DEFECT)
            fig = _side_by_side(orig, overlay, args.img_size)
            dest = out_dir / f"{cat}_{i}.png"
            fig.save(dest)
            print(f"    wrote {dest}  (src: {Path(path).name})")

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == "__main__":
    main()
