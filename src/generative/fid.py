"""
Compute FID (Frechet Inception Distance) between real and synthetic defects.

FID is the standard quality metric for generative models in this literature.
Report per-category FID in the results table. Lower = synthetic distribution
closer to real. Uses `clean-fid` (more reproducible than pytorch-fid).

Usage:
    python -m src.generative.fid --real data/raw/bottle/test \
        --fake data/synthetic/bottle

Note: clean-fid recurses subfolders, so pointing --real at .../test includes the
good images too. For a defect-vs-defect comparison, point --real at a specific
defect folder (e.g. data/raw/bottle/test/broken_large) or a merged defect dir.

Owner: Member 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def count_images(folder) -> int:
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*")
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def compute_fid(real_dir, fake_dir, mode="clean", num_workers=0) -> float:
    """Return FID between two image folders (recurses subfolders).

    Raises a clear error if either folder is missing or has too few images
    (FID needs a reasonable sample to estimate the feature covariance).
    """
    real_dir, fake_dir = Path(real_dir), Path(fake_dir)
    n_real, n_fake = count_images(real_dir), count_images(fake_dir)
    if n_real < 2 or n_fake < 2:
        raise SystemExit(
            f"need >=2 images in each dir; got real={n_real} ({real_dir}), "
            f"fake={n_fake} ({fake_dir}). Generate synthetic images first."
        )
    if min(n_real, n_fake) < 50:
        print(f"  warning: small sample (real={n_real}, fake={n_fake}); "
              f"FID is noisy below ~50 images per set.")

    try:
        from cleanfid import fid
    except ImportError:
        raise SystemExit("clean-fid is required: pip install clean-fid")

    print(f"computing FID (mode={mode}) | real={n_real} imgs, fake={n_fake} imgs ...")
    score = float(fid.compute_fid(str(real_dir), str(fake_dir), mode=mode,
                                  num_workers=num_workers))
    print(f"FID = {score:.4f}  (lower is better)")
    return score


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FID between real and synthetic defect images.")
    p.add_argument("--real", required=True, help="real images dir (e.g. data/raw/bottle/test)")
    p.add_argument("--fake", required=True, help="synthetic images dir (e.g. data/synthetic/bottle)")
    p.add_argument("--mode", default="clean", choices=["clean", "legacy_pytorch", "legacy_tensorflow"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default=None, help="optional JSON path to record the score")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    score = compute_fid(args.real, args.fake, mode=args.mode, num_workers=args.num_workers)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"real": args.real, "fake": args.fake, "mode": args.mode, "fid": score}, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
