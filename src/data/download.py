"""
Download and verify the MVTec AD dataset.

MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
License: CC BY-NC-SA 4.0 (academic use OK, cite the paper).

MVTec is distributed as archives (the full ``mvtec_anomaly_detection.tar.xz`` or
per-category tarballs such as ``bottle.tar.xz``). The expected workflow is:

    1. Download the archive manually once and drop it into ``data/raw/``.
    2. Run this script to extract + verify the chosen categories:

        python -m src.data.download --categories bottle hazelnut carpet --out data/raw

After extraction each category folder follows the standard MVTec layout::

    data/raw/bottle/
        train/good/*.png            # good images only (MVTec convention)
        test/good/*.png             # good test images
        test/<defect_type>/*.png    # one folder per defect type
        ground_truth/<defect_type>/ # segmentation masks (defects only)
        license.txt  readme.txt

Owner: Member 1 (Data & Baselines Lead)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

CATEGORIES = ["bottle", "hazelnut", "carpet"]  # chosen subset; 2 objects + 1 texture

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".zip")

_CHUNK = 1 << 20  # 1 MiB, for streaming hashes


# --------------------------------------------------------------------------- #
# Archive discovery + inspection
# --------------------------------------------------------------------------- #
def _find_archives(out_dir: Path) -> list[Path]:
    """Return archive files sitting directly in out_dir, sorted by name."""
    archives = [
        p for p in sorted(out_dir.iterdir())
        if p.is_file() and p.name.lower().endswith(ARCHIVE_SUFFIXES)
    ]
    return archives


def _is_zip(path: Path) -> bool:
    return path.name.lower().endswith(".zip")


def _list_members(archive: Path) -> list[str]:
    """Return all member names inside an archive (tar or zip)."""
    if _is_zip(archive):
        with zipfile.ZipFile(archive) as zf:
            return zf.namelist()
    with tarfile.open(archive) as tf:
        return tf.getnames()


def _top_level_dirs(members: list[str]) -> set[str]:
    """First path component of every member (the category folders)."""
    tops: set[str] = set()
    for name in members:
        name = name.strip("/").lstrip("./")
        if not name:
            continue
        tops.add(name.split("/", 1)[0])
    return tops


def _archive_for_category(archives: list[Path], category: str) -> Path | None:
    """Find the archive that contains <category>/ at its top level."""
    for archive in archives:
        try:
            if category in _top_level_dirs(_list_members(archive)):
                return archive
        except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
            print(f"  ! could not read {archive.name}: {exc}", file=sys.stderr)
    return None


# --------------------------------------------------------------------------- #
# Safe extraction (guards against path-traversal / absolute-path members)
# --------------------------------------------------------------------------- #
def _is_within(base: Path, target: Path) -> bool:
    """True if target resolves to a path inside base."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract(archive: Path, category: str, out_dir: Path) -> int:
    """Extract only members under ``<category>/`` into out_dir. Returns count."""
    prefix = category.strip("/") + "/"
    extracted = 0

    if _is_zip(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                norm = name.strip("/").lstrip("./")
                if not (norm == category or norm.startswith(prefix)):
                    continue
                dest = out_dir / norm
                if not _is_within(out_dir, dest):
                    raise RuntimeError(f"unsafe member path in archive: {name!r}")
                zf.extract(name, out_dir)
                extracted += 1
        return extracted

    with tarfile.open(archive) as tf:
        members = []
        for m in tf.getmembers():
            norm = m.name.strip("/").lstrip("./")
            if not (norm == category or norm.startswith(prefix)):
                continue
            dest = out_dir / norm
            if not _is_within(out_dir, dest):
                raise RuntimeError(f"unsafe member path in archive: {m.name!r}")
            members.append(m)
        # Use the 'data' filter where available (py3.12+) for extra safety.
        try:
            tf.extractall(out_dir, members=members, filter="data")
        except TypeError:
            tf.extractall(out_dir, members=members)
        extracted = sum(1 for m in members if m.isfile())
    return extracted


# --------------------------------------------------------------------------- #
# Checksums + structural verification
# --------------------------------------------------------------------------- #
def sha256sum(path: Path) -> str:
    """Streaming SHA-256 of a file (archives are large)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def verify_category(out_dir: Path, category: str) -> dict:
    """Check the extracted layout for one category and return image counts.

    Raises FileNotFoundError if the mandatory train/good or test folders are
    missing — that means extraction failed or the wrong archive was supplied.
    """
    root = out_dir / category
    train_good = root / "train" / "good"
    test_dir = root / "test"

    if not train_good.is_dir():
        raise FileNotFoundError(f"missing {train_good} (extraction incomplete?)")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"missing {test_dir} (extraction incomplete?)")

    defect_types = sorted(
        d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good"
    )
    if not defect_types:
        raise FileNotFoundError(f"no defect folders under {test_dir}")

    counts = {
        "train_good": _count_images(train_good),
        "test_good": _count_images(test_dir / "good"),
        "test_defect": sum(_count_images(test_dir / d) for d in defect_types),
        "defect_types": defect_types,
    }
    if counts["train_good"] == 0 or counts["test_defect"] == 0:
        raise FileNotFoundError(f"{category}: extracted folders contain no images")
    return counts


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def download(categories, out_dir, force=False, expected_sha256=None):
    """Extract + verify the listed MVTec categories found in out_dir.

    Args:
        categories: iterable of category names (e.g. ["bottle", "hazelnut"]).
        out_dir: directory holding the downloaded archive(s); also the
            extraction target (the MVTec layout is created in place).
        force: re-extract even if the category folder already verifies.
        expected_sha256: optional hex digest; if given, every archive that is
            actually used for extraction is checked against it.

    Returns:
        dict mapping category -> image-count summary (see verify_category).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = _find_archives(out_dir)
    if not archives:
        raise FileNotFoundError(
            f"no archives found in {out_dir}. Download MVTec AD and place the "
            f"tarball (e.g. mvtec_anomaly_detection.tar.xz or bottle.tar.xz) here."
        )
    print(f"Found {len(archives)} archive(s) in {out_dir}: "
          f"{', '.join(a.name for a in archives)}")

    summary: dict[str, dict] = {}
    for category in categories:
        print(f"\n[{category}]")

        # Skip work if already extracted and valid (unless --force).
        if not force:
            try:
                counts = verify_category(out_dir, category)
                print(f"  already extracted: {counts['train_good']} good / "
                      f"{counts['test_defect']} defect images. Skipping "
                      f"(use --force to re-extract).")
                summary[category] = counts
                continue
            except FileNotFoundError:
                pass  # not extracted yet (or incomplete) -> proceed

        archive = _archive_for_category(archives, category)
        if archive is None:
            print(f"  ! no archive in {out_dir} contains '{category}/'. Skipping.",
                  file=sys.stderr)
            continue

        if expected_sha256 is not None:
            print(f"  verifying SHA-256 of {archive.name} ...")
            digest = sha256sum(archive)
            if digest.lower() != expected_sha256.lower():
                raise ValueError(
                    f"checksum mismatch for {archive.name}:\n"
                    f"    expected {expected_sha256}\n    got      {digest}"
                )
            print("  checksum OK")

        print(f"  extracting from {archive.name} ...")
        n = _safe_extract(archive, category, out_dir)
        counts = verify_category(out_dir, category)
        print(f"  extracted {n} member(s) -> {counts['train_good']} good / "
              f"{counts['test_defect']} defect images "
              f"({len(counts['defect_types'])} defect types: "
              f"{', '.join(counts['defect_types'])})")
        summary[category] = counts

    if not summary:
        raise RuntimeError("nothing was extracted or verified - check archive contents.")

    print("\nDone. Verified categories: " + ", ".join(sorted(summary)))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract + verify MVTec AD categories.")
    p.add_argument("--categories", nargs="+", default=CATEGORIES,
                   help=f"categories to extract (default: {' '.join(CATEGORIES)})")
    p.add_argument("--out", default="data/raw",
                   help="dir holding the archive(s); also the extraction target")
    p.add_argument("--force", action="store_true",
                   help="re-extract even if already present")
    p.add_argument("--sha256", default=None,
                   help="optional expected SHA-256 of the archive(s) used")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    download(args.categories, args.out, force=args.force,
             expected_sha256=args.sha256)


if __name__ == "__main__":
    main()
