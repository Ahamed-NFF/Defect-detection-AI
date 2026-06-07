"""
Quick data-layer checks for the MVTec defect loaders.

Two kinds of tests:
  * Pure logic (no dataset needed): the deterministic re-split is disjoint and
    reproducible.
  * Integration (needs an extracted category): build loaders, pull a batch,
    print shapes/labels, assert basic invariants. These SKIP cleanly until the
    dataset is downloaded so CI stays green pre-data.

Run just this file with prints visible:
    pytest tests/test_data.py -s
Or run it directly to eyeball a batch:
    python tests/test_data.py
"""

from pathlib import Path

import pytest

from src.data.dataset import (
    DEFECT,
    GOOD,
    DefectDataset,
    _partition,
    build_loaders,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
CATEGORY = "bottle"


def _category_available(category=CATEGORY) -> bool:
    return (DATA_ROOT / category / "train" / "good").is_dir()


# --------------------------------------------------------------------------- #
# Pure-logic tests (no dataset required)
# --------------------------------------------------------------------------- #
def test_partition_is_disjoint_and_reproducible():
    paths = [Path(f"img_{i}.png") for i in range(100)]
    a = _partition(paths, seed=42)
    b = _partition(paths, seed=42)

    # Reproducible: same seed -> identical partition.
    assert a == b

    # Disjoint + covers everything exactly once.
    train, val, test = set(a["train"]), set(a["val"]), set(a["test"])
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(paths)

    # Roughly the configured 70/15/15 split.
    assert len(a["train"]) == 70 and len(a["val"]) == 15 and len(a["test"]) == 15


# --------------------------------------------------------------------------- #
# Integration tests (need an extracted category — skip otherwise)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _category_available(),
                    reason=f"{CATEGORY} not extracted under {DATA_ROOT}")
def test_loads_a_batch_and_reports_shapes():
    train_loader, val_loader, test_loader = build_loaders(
        CATEGORY, batch_size=4, data_root=str(DATA_ROOT), num_workers=0,
    )

    print(f"\n[{CATEGORY}] split sizes (images):")
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        ds = loader.dataset
        print(f"  {name:5s}: {len(ds):4d}  classes={ds.class_counts()}")

    images, labels = next(iter(train_loader))
    print(f"  batch images: tuple{tuple(images.shape)} dtype={images.dtype}")
    print(f"  batch labels: {labels.tolist()}")

    # Shape: (B, 3, H, W); a 4-D float tensor.
    assert images.ndim == 4 and images.shape[1] == 3
    assert images.shape[0] == labels.shape[0]
    # Labels are binary good/defect.
    assert set(labels.tolist()).issubset({GOOD, DEFECT})


@pytest.mark.skipif(not _category_available(),
                    reason=f"{CATEGORY} not extracted under {DATA_ROOT}")
def test_few_shot_caps_real_defects():
    n = 5
    full = DefectDataset(DATA_ROOT / CATEGORY, "train")
    capped = DefectDataset(DATA_ROOT / CATEGORY, "train", few_shot_n=n)

    full_defects = full.class_counts()["defect"]
    capped_defects = capped.class_counts()["defect"]
    print(f"\n[{CATEGORY}] train defects: full={full_defects}, "
          f"few_shot_n={n} -> {capped_defects}")

    assert capped_defects == min(n, full_defects)
    # Good images are untouched by the cap.
    assert capped.class_counts()["good"] == full.class_counts()["good"]


if __name__ == "__main__":
    if not _category_available():
        raise SystemExit(
            f"No data under {DATA_ROOT / CATEGORY}. Download MVTec AD, place the "
            f"archive in data/raw/, then run:\n"
            f"  python -m src.data.download --categories {CATEGORY} --out data/raw"
        )
    test_loads_a_batch_and_reports_shapes()
    test_few_shot_caps_real_defects()
    print("\nOK")
