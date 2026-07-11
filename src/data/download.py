"""
Download a 6-category subset of MVTec AD via Hugging Face.

The official MVTec AD release is a ~5 GB tarball gated behind a registration
form, which is impractical on a bandwidth-constrained machine. The
``Voxel51/mvtec-ad`` dataset on Hugging Face mirrors the same data and lets us
fetch only the categories we actually need. We use six:

    bottle, hazelnut, metal_nut, screw     (4 objects)
    carpet, tile                           (2 textures)

The Voxel51 mirror is a *FiftyOne export*, not the canonical MVTec layout:
images live flat under ``data/data_N/*.png`` and the per-image labels
(category / split / defect type / mask path) come from a top-level
``samples.json`` manifest. This module therefore:

    1. Downloads ``samples.json`` (~1.6 MB).
    2. Filters samples to the requested categories.
    3. Downloads ONLY the image + mask files those samples reference,
       via ``snapshot_download(allow_patterns=<exact paths>)``.
    4. Reorganizes the downloaded files into the canonical structure that
       ``src/data/dataset.py`` expects::

           data/raw/<category>/train/good/*.png
           data/raw/<category>/test/good/*.png
           data/raw/<category>/test/<defect_type>/*.png
           data/raw/<category>/ground_truth/<defect_type>/*.png   (defects only)

    5. Prints a per-category summary table so a failed/empty download is loud.

Run as::

    python -m src.data.download --categories bottle hazelnut metal_nut screw carpet tile --out data/raw

Idempotent: a category whose canonical layout already verifies is skipped
unless ``--force`` is passed.

Owner: Member 1 (Data & Baselines Lead)
License: MVTec AD is CC BY-NC-SA 4.0 — academic use only, cite the paper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CATEGORIES = ["bottle", "hazelnut", "metal_nut", "screw", "carpet", "tile"]
HF_REPO_ID = "Voxel51/mvtec-ad"
MANIFEST_FILENAME = "samples.json"
STAGING_DIRNAME = "_hf_staging"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Dependency check
# --------------------------------------------------------------------------- #
def _import_hf():
    """Lazy-import huggingface_hub; print install hint and exit cleanly if absent."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
        return hf_hub_download, snapshot_download, HfHubHTTPError
    except ImportError:
        print(
            "ERROR: huggingface_hub is required for this script.\n"
            "Install it with:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _category_is_complete(category_root: Path) -> bool:
    """True if ``category_root`` already holds a canonical, populated MVTec layout."""
    train_good = category_root / "train" / "good"
    test_dir = category_root / "test"
    if not train_good.is_dir() or not test_dir.is_dir():
        return False
    if _count_images(train_good) == 0:
        return False
    defect_dirs = [
        d for d in test_dir.iterdir()
        if d.is_dir() and d.name != "good"
    ]
    if not defect_dirs:
        return False
    return any(_count_images(d) > 0 for d in defect_dirs)


# --------------------------------------------------------------------------- #
# Manifest (samples.json) handling
# --------------------------------------------------------------------------- #
def _download_manifest(staging_dir: Path) -> Path:
    """Fetch ``samples.json`` from the HF repo into ``staging_dir``."""
    hf_hub_download, _, HfHubHTTPError = _import_hf()
    staging_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching manifest '{MANIFEST_FILENAME}' from '{HF_REPO_ID}'...")
    try:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=MANIFEST_FILENAME,
            local_dir=str(staging_dir),
        )
    except HfHubHTTPError as exc:
        print(
            f"\nERROR: failed to download manifest ({exc.__class__.__name__}): {exc}\n"
            "  Check network / HF availability.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return Path(path)


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Parse the FiftyOne ``samples.json`` and return its list of sample records."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise RuntimeError(
            f"unexpected manifest schema in {manifest_path}: 'samples' key missing"
        )
    return samples


def _filter_samples(samples: list[dict], categories) -> list[dict]:
    """Keep only samples whose category label is in ``categories``."""
    wanted = set(categories)
    return [s for s in samples if s.get("category", {}).get("label") in wanted]


def _required_repo_paths(samples: list[dict]) -> list[str]:
    """All repo-relative file paths referenced by the given samples (images + masks)."""
    paths: set[str] = set()
    for s in samples:
        fp = s.get("filepath")
        if fp:
            paths.add(fp)
        mask = s.get("defect_mask") or {}
        mp = mask.get("mask_path")
        if mp:
            paths.add(mp)
    return sorted(paths)


# --------------------------------------------------------------------------- #
# Hugging Face download
# --------------------------------------------------------------------------- #
def _hf_download_files(repo_paths: list[str], staging_dir: Path) -> Path:
    """Fetch the exact repo paths into ``staging_dir`` via ``snapshot_download``."""
    _, snapshot_download, HfHubHTTPError = _import_hf()
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(repo_paths)} files from '{HF_REPO_ID}'...")
    print(f"  staging dir: {staging_dir}")

    try:
        local_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(staging_dir),
            allow_patterns=repo_paths,
        )
    except HfHubHTTPError as exc:
        print(
            f"\nERROR: Hugging Face download failed ({exc.__class__.__name__}): {exc}\n"
            "  Check network connectivity and that 'Voxel51/mvtec-ad' is reachable.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except OSError as exc:
        print(
            f"\nERROR: filesystem error during download: {exc}\n"
            f"  Make sure '{staging_dir}' is writable and has enough free space.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as exc:  # last-resort net/auth failure
        print(
            f"\nERROR: unexpected failure during download: {exc}\n"
            "  Check network / HF availability and try again.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return Path(local_path)


# --------------------------------------------------------------------------- #
# Reorganization into canonical MVTec layout
# --------------------------------------------------------------------------- #
def _canonical_image_dest(out_dir: Path, sample: dict) -> Path:
    """Where ``sample``'s image should live in the canonical layout."""
    category = sample["category"]["label"]
    split = sample["split"]  # "train" or "test"
    defect = sample["defect"]["label"]  # "good" or a defect type
    basename = Path(sample["filepath"]).name
    return out_dir / category / split / defect / basename


def _canonical_mask_dest(out_dir: Path, sample: dict) -> Path | None:
    """Where ``sample``'s mask should live, if it has one."""
    mask = sample.get("defect_mask") or {}
    mp = mask.get("mask_path")
    if not mp:
        return None
    category = sample["category"]["label"]
    defect = sample["defect"]["label"]
    if defect == "good":
        return None  # canonical MVTec has masks only for defects
    return out_dir / category / "ground_truth" / defect / Path(mp).name


def _place(src: Path, dest: Path) -> None:
    """Move ``src`` to ``dest``, creating parents. Skip silently if dest exists."""
    if not src.exists():
        return
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def _reorganize(samples: list[dict], staging: Path, out_dir: Path) -> dict:
    """Move every staged file into its canonical slot. Returns placement stats."""
    placed_images = 0
    placed_masks = 0
    missing: list[str] = []

    for s in samples:
        src_img = staging / s["filepath"]
        dest_img = _canonical_image_dest(out_dir, s)
        if src_img.exists():
            _place(src_img, dest_img)
            placed_images += 1
        else:
            missing.append(s["filepath"])

        mask = s.get("defect_mask") or {}
        mp = mask.get("mask_path")
        if mp:
            src_mask = staging / mp
            dest_mask = _canonical_mask_dest(out_dir, s)
            if dest_mask is not None and src_mask.exists():
                _place(src_mask, dest_mask)
                placed_masks += 1

    print(f"  placed {placed_images} image(s) and {placed_masks} mask(s) "
          f"into canonical layout.")
    if missing:
        print(f"  ! {len(missing)} expected file(s) were not in the download "
              f"(first 5): {missing[:5]}", file=sys.stderr)
    return {"images": placed_images, "masks": placed_masks, "missing": missing}


# --------------------------------------------------------------------------- #
# Verification + summary table
# --------------------------------------------------------------------------- #
def _summarize_category(category_root: Path) -> dict:
    """Image counts + mask presence for one canonical category folder."""
    summary = {
        "exists": category_root.is_dir(),
        "train_good": _count_images(category_root / "train" / "good"),
        "test_good": _count_images(category_root / "test" / "good"),
        "test_defects": {},
        "has_masks": False,
    }
    test_dir = category_root / "test"
    if test_dir.is_dir():
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir() and sub.name != "good":
                summary["test_defects"][sub.name] = _count_images(sub)
    gt_dir = category_root / "ground_truth"
    if gt_dir.is_dir():
        summary["has_masks"] = any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTS
            for p in gt_dir.rglob("*")
        )
    return summary


def _print_summary(summaries: dict) -> None:
    print()
    print("=" * 72)
    print(f"{'MVTec AD download summary':^72}")
    print("=" * 72)
    any_empty = False
    for cat, s in summaries.items():
        total_defect = sum(s["test_defects"].values())
        ok = s["exists"] and s["train_good"] > 0 and total_defect > 0
        any_empty = any_empty or not ok
        status = "OK" if ok else "EMPTY -- download failed?"
        masks = "yes" if s["has_masks"] else "no"

        print(f"\n[{cat}]  {status}")
        print(f"  train/good ............... {s['train_good']}")
        print(f"  test/good ................ {s['test_good']}")
        print(f"  test/<defect_type> "
              f"({len(s['test_defects'])} types):")
        if s["test_defects"]:
            for d, n in sorted(s["test_defects"].items()):
                print(f"    {d:<22} {n}")
        else:
            print("    (none)")
        print(f"  ground_truth masks ....... {masks}")

    print("\n" + "=" * 72)
    if any_empty:
        print(
            "WARNING: one or more categories are missing images. Re-run with --force\n"
            "or check the staging output above.",
            file=sys.stderr,
        )
    print()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def download(categories, out_dir, force: bool = False, **_ignored):
    """End-to-end pipeline: manifest -> filtered HF download -> reorganize -> verify.

    Args:
        categories: iterable of MVTec category names.
        out_dir: destination root. Canonical layout lands at ``<out_dir>/<cat>/``.
        force: re-download even if a category already verifies as complete.

    Returns:
        dict mapping category -> summary (see ``_summarize_category``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = list(categories)

    todo, skipped = [], []
    for cat in categories:
        if not force and _category_is_complete(out_dir / cat):
            skipped.append(cat)
        else:
            todo.append(cat)

    if skipped:
        print(f"Already extracted (skipping): {skipped}")

    if todo:
        staging = out_dir / STAGING_DIRNAME
        if staging.exists():
            shutil.rmtree(staging)

        manifest_path = _download_manifest(staging)
        samples = _load_manifest(manifest_path)
        target_samples = _filter_samples(samples, todo)
        if not target_samples:
            print(
                f"\nERROR: manifest contains no samples for {todo}. "
                f"Available categories may have changed.",
                file=sys.stderr,
            )
            raise SystemExit(3)

        repo_paths = _required_repo_paths(target_samples)
        print(f"Filtered manifest: {len(target_samples)} samples "
              f"-> {len(repo_paths)} files to fetch")

        _hf_download_files(repo_paths, staging)
        _reorganize(target_samples, staging, out_dir)

        # Best-effort staging cleanup; ignore if HF left a .cache lock behind.
        shutil.rmtree(staging, ignore_errors=True)

    summaries = {cat: _summarize_category(out_dir / cat) for cat in categories}
    _print_summary(summaries)
    return summaries


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download a 6-category subset of MVTec AD from Hugging Face "
            "(Voxel51/mvtec-ad) and normalize it into the canonical layout."
        ),
    )
    p.add_argument(
        "--categories", nargs="+", default=CATEGORIES,
        help=f"categories to fetch (default: {' '.join(CATEGORIES)})",
    )
    p.add_argument(
        "--out", default="data/raw",
        help="destination root for canonical category folders",
    )
    p.add_argument(
        "--force", action="store_true",
        help="re-download even if a category already verifies",
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    download(args.categories, args.out, force=args.force)


if __name__ == "__main__":
    main()
