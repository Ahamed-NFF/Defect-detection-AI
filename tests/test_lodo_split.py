"""
Leak-freeness verification for the LODO split (src.data.dataset.build_loaders_lodo).

For every category currently downloaded under data/raw/ and every one of its
defect-type folds, asserts:
    - the held-out type's images are entirely in test, absent from train/val,
    - train/val/test are pairwise disjoint at the file level,
    - (printed, not asserted) per-fold, per-split class counts, and a flag on
      any fold whose test defect set is smaller than MIN_TEST_DEFECTS.

Categories not yet downloaded (metal_nut/screw/tile until fetched) are simply
absent from data/raw/ and are not iterated -- nothing here hardcodes which or
how many categories exist.

Run with prints visible:
    pytest -s tests/test_lodo_split.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.dataset import _list_images, build_loaders_lodo, defect_types

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
MIN_TEST_DEFECTS = 8


def _available_categories() -> list[str]:
    """Categories actually downloaded (populated test/<type>/ dirs) -- not hardcoded."""
    if not DATA_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in DATA_ROOT.iterdir()
        if p.is_dir() and defect_types(p.name, data_root=str(DATA_ROOT))
    )


CATEGORIES = _available_categories()


@pytest.mark.skipif(not CATEGORIES, reason=f"no downloaded categories under {DATA_ROOT}")
def test_lodo_no_leakage():
    print(f"\nLODO verification over categories: {CATEGORIES}\n")

    small_folds = []
    total_folds = 0

    for category in CATEGORIES:
        types = defect_types(category, data_root=str(DATA_ROOT))
        print(f"=== {category} ({len(types)} defect types: {types}) ===")

        for holdout in types:
            total_folds += 1
            train_loader, val_loader, test_loader = build_loaders_lodo(
                category, holdout, batch_size=8, data_root=str(DATA_ROOT), num_workers=0,
            )
            train_ds = train_loader.dataset
            val_ds = val_loader.dataset
            test_ds = test_loader.dataset

            train_paths = {p for p, _ in train_ds.samples}
            val_paths = {p for p, _ in val_ds.samples}
            test_paths = {p for p, _ in test_ds.samples}

            # 1. held-out type entirely in test, absent from train/val.
            holdout_paths = set(_list_images(DATA_ROOT / category / "test" / holdout))
            assert holdout_paths, f"{category}/{holdout}: no images found on disk"
            missing_from_test = holdout_paths - test_paths
            assert not missing_from_test, (
                f"{category}/{holdout}: {len(missing_from_test)} held-out image(s) "
                f"missing from test: {sorted(p.name for p in missing_from_test)[:5]}"
            )
            leaked_train = holdout_paths & train_paths
            assert not leaked_train, (
                f"{category}/{holdout}: held-out images leaked into TRAIN: "
                f"{sorted(p.name for p in leaked_train)[:5]}"
            )
            leaked_val = holdout_paths & val_paths
            assert not leaked_val, (
                f"{category}/{holdout}: held-out images leaked into VAL: "
                f"{sorted(p.name for p in leaked_val)[:5]}"
            )

            # 2. train/val/test disjoint at the file level (any label).
            assert not (train_paths & val_paths), f"{category}/{holdout}: train/val overlap"
            assert not (train_paths & test_paths), f"{category}/{holdout}: train/test overlap"
            assert not (val_paths & test_paths), f"{category}/{holdout}: val/test overlap"

            train_counts = train_ds.class_counts()
            val_counts = val_ds.class_counts()
            test_counts = test_ds.class_counts()
            print(f"  holdout={holdout:<22} "
                  f"train(good={train_counts['good']:>3}, defect={train_counts['defect']:>3})  "
                  f"val(good={val_counts['good']:>3}, defect={val_counts['defect']:>3})  "
                  f"test(good={test_counts['good']:>3}, defect={test_counts['defect']:>3})")

            if test_counts["defect"] < MIN_TEST_DEFECTS:
                small_folds.append((category, holdout, test_counts["defect"]))

    print(f"\n{total_folds} fold(s) verified leak-free across "
          f"{len(CATEGORIES)} categor{'y' if len(CATEGORIES) == 1 else 'ies'}.")
    if small_folds:
        print(f"\nFolds with <{MIN_TEST_DEFECTS} test defects "
              f"(too coarse to report alone -- pool these):")
        for cat, holdout, n in small_folds:
            print(f"  {cat}/{holdout}: {n} test defects")
    else:
        print(f"\nNo folds below the {MIN_TEST_DEFECTS}-test-defect threshold.")


if __name__ == "__main__":
    if not CATEGORIES:
        raise SystemExit(
            f"No categories downloaded under {DATA_ROOT}. Run:\n"
            f"  python -m src.data.download --categories bottle hazelnut carpet --out data/raw"
        )
    test_lodo_no_leakage()
    print("\nOK")
