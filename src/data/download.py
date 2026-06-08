"""
Download a 3-category subset of MVTec AD via Hugging Face.

The official MVTec AD release is a ~5 GB tarball gated behind a registration
form, which is impractical on a bandwidth-constrained machine. The
``Voxel51/mvtec-ad`` dataset on Hugging Face mirrors the same data and lets us
fetch only the categories we actually need via ``allow_patterns``. We use
exactly three:

    bottle, hazelnut, carpet     (2 objects + 1 texture)

The Voxel51 mirror is *not guaranteed* to ship the canonical MVTec layout. This
module therefore does two things after the filtered download:

    1. Inspect the resulting directory tree to figure out which layout we got.
    2. Reorganize (if needed) into the canonical structure that
       ``src/data/dataset.py`` expects::

           data/raw/<category>/train/good/*.png
           data/raw/<category>/test/good/*.png
           data/raw/<category>/test/<defect_type>/*.png
           data/raw/<category>/ground_truth/<defect_type>/*.png   (if present)

It then prints a per-category summary table so a failed/empty download is loud.

Run as::

    python -m src.data.download --categories bottle hazelnut carpet --out data/raw

Idempotent: a category whose canonical layout already verifies is skipped
unless ``--force`` is passed.

Owner: Member 1 (Data & Baselines Lead)
License: MVTec AD is CC BY-NC-SA 4.0 — academic use only, cite the paper.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CATEGORIES = ["bottle", "hazelnut", "carpet"]
HF_REPO_ID = "Voxel51/mvtec-ad"
STAGING_DIRNAME = "_hf_staging"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Dependency check
# --------------------------------------------------------------------------- #
def _import_hf():
    """Lazy-import huggingface_hub; print install hint and exit cleanly if absent."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
        return snapshot_download, HfHubHTTPError
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
# Hugging Face download
# --------------------------------------------------------------------------- #
def _allow_patterns(categories) -> list[str]:
    """Build fnmatch patterns covering the canonical category folder at common depths.

    huggingface_hub matches with stdlib ``fnmatch``; ``*`` matches across path
    separators, so ``bottle/*`` already covers ``bottle/train/good/000.png``.
    The nested variants exist in case the mirror wraps everything under a
    repo-level prefix (e.g. ``mvtec_anomaly_detection/bottle/...``).
    """
    patterns: list[str] = []
    for cat in categories:
        patterns.extend([f"{cat}/*", f"*/{cat}/*", f"*/*/{cat}/*"])
    return patterns


def _hf_filtered_download(categories, staging_dir: Path) -> Path:
    """Fetch only the requested categories from ``HF_REPO_ID`` into ``staging_dir``."""
    snapshot_download, HfHubHTTPError = _import_hf()

    staging_dir.mkdir(parents=True, exist_ok=True)
    patterns = _allow_patterns(categories)

    print(f"Downloading {list(categories)} from '{HF_REPO_ID}' (filtered)...")
    print(f"  staging dir   : {staging_dir}")
    print(f"  allow_patterns: {patterns}")

    try:
        local_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(staging_dir),
            allow_patterns=patterns,
        )
    except HfHubHTTPError as exc:
        print(
            f"\nERROR: Hugging Face download failed ({exc.__class__.__name__}): {exc}\n"
            "  Check network connectivity, the repo id, and that "
            "'Voxel51/mvtec-ad' is reachable from this machine.",
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
# Layout detection + reorganization
# --------------------------------------------------------------------------- #
def _looks_like_canonical_category(path: Path) -> bool:
    """Heuristic: a canonical MVTec category folder has train/ and test/ children."""
    return (
        path.is_dir()
        and (path / "train").is_dir()
        and (path / "test").is_dir()
    )


def _find_canonical_root(staging: Path, category: str) -> Path | None:
    """Locate the canonical-layout root for ``category`` somewhere under ``staging``.

    Returns the directory whose basename equals ``category`` and which contains
    both ``train/`` and ``test/`` subfolders. Searches the obvious top-level
    location first, then falls back to a recursive walk.
    """
    direct = staging / category
    if _looks_like_canonical_category(direct):
        return direct

    for candidate in staging.rglob(category):
        if candidate.name != category:
            continue  # rglob can return prefix-matches in odd FS layouts
        if _looks_like_canonical_category(candidate):
            return candidate

    return None


def _move_into_place(src: Path, dest: Path) -> None:
    """Move the canonical MVTec subtree from ``src`` to ``dest``.

    Only the three known top-level dirs (train/test/ground_truth) are moved so
    that any FiftyOne sidecar files in the same parent don't get dragged along.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return

    for sub in ("train", "test", "ground_truth"):
        sub_src = src / sub
        if not sub_src.exists():
            continue
        sub_dest = dest / sub
        if sub_dest.exists():
            # Idempotent: leave whatever's already in place untouched.
            continue
        shutil.move(str(sub_src), str(sub_dest))


def _describe_unknown_layout(staging: Path, category: str) -> None:
    """Print a snapshot of what we got so the user can debug."""
    print(
        f"  ! could not locate a canonical MVTec subtree for '{category}' under {staging}.",
        file=sys.stderr,
    )
    sample = []
    for p in staging.rglob("*"):
        try:
            rel = p.relative_to(staging)
        except ValueError:
            continue
        sample.append(str(rel))
        if len(sample) >= 15:
            break
    if sample:
        print("    first entries downloaded:", file=sys.stderr)
        for s in sample:
            print(f"      {s}", file=sys.stderr)
    print(
        "    The Voxel51 mirror may have changed its layout. Inspect the staging\n"
        "    directory and adapt _find_canonical_root() if so.",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Verification + summary table
# --------------------------------------------------------------------------- #
def _summarize_category(category_root: Path) -> dict:
    """Image counts + mask presence for one category folder."""
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
    """End-to-end pipeline: filtered HF download -> reorganize -> verify.

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
        local = _hf_filtered_download(todo, staging)

        for cat in todo:
            canonical = _find_canonical_root(local, cat)
            if canonical is None:
                _describe_unknown_layout(local, cat)
                continue
            dest = out_dir / cat
            print(f"  reorganizing '{cat}': {canonical} -> {dest}")
            _move_into_place(canonical, dest)

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
            "Download a 3-category subset of MVTec AD from Hugging Face "
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
