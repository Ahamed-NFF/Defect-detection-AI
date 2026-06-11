"""
PyTorch Dataset + DataLoader factories for MVTec categories.

Provides:
    - DefectDataset: yields (image_tensor, label) where label in {0: good, 1: defect}
    - build_loaders(category, batch_size, few_shot_n=None): train/val/test loaders
      few_shot_n limits the number of REAL defect images in the training set,
      which is the core knob for the few-shot ablation experiments.

Why a re-split?
    MVTec AD is an *anomaly-detection* benchmark: its native ``train/`` folder
    holds GOOD images only and every defect lives under ``test/``. A binary
    good/defect classifier needs defects in its training set, so we pool all
    images (good = train/good + test/good, defect = test/<type>) and carve a
    deterministic, seeded, class-stratified train/val/test split. Because the
    split is a pure function of the (sorted) file list + seed, the three loaders
    built below see disjoint, reproducible partitions.

The training knobs (apply to the TRAIN split only):
    - few_shot_n   : cap the number of real defect images (the ablation lever).
    - synthetic_dir: fold in diffusion-generated defect images (the "ours" runs).

Owner: Member 1
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from .augment import train_transforms

GOOD, DEFECT = 0, 1
LABEL_NAMES = {GOOD: "good", DEFECT: "defect"}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Split fractions for the re-split described in the module docstring.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42  # fixed so every DefectDataset instance agrees on the partition

# ImageNet stats (ResNet-50 / EfficientNet are ImageNet-pretrained).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = 256


def _list_images(folder: Path) -> list[Path]:
    """All image files directly relevant under folder, sorted for determinism."""
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _index_category(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (good_paths, defect_paths) pooled across MVTec's train/ and test/."""
    good = _list_images(root / "train" / "good") + _list_images(root / "test" / "good")
    defect: list[Path] = []
    test_dir = root / "test"
    if test_dir.is_dir():
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir() and sub.name != "good":
                defect += _list_images(sub)
    return sorted(set(good)), sorted(set(defect))


def _partition(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    """Deterministically shuffle and slice a path list into train/val/test."""
    items = list(paths)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(round(n * SPLIT_FRACTIONS["train"]))
    n_val = int(round(n * SPLIT_FRACTIONS["val"]))
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def _default_transform(img_size: int = DEFAULT_IMG_SIZE) -> T.Compose:
    """Plain resize + normalise (used when no transform is supplied)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DefectDataset(Dataset):
    """Binary good/defect dataset over one MVTec category.

    Args:
        root: path to the extracted category folder (e.g. data/raw/bottle).
        split: 'train' | 'val' | 'test'. (The docstring spec lists train/test;
            'val' is the carved validation partition used by build_loaders.)
        transform: torchvision transform pipeline. Defaults to resize+normalise.
        few_shot_n: if set, cap the number of real defect images (train split
            only) — the few-shot ablation lever.
        synthetic_dir: optional path to synthetic defect images mixed into the
            train split only (the diffusion-augmented "ours" runs).
        synthetic_n: cap on how many synthetic images to mix in (train only).
            int -> use at most that many; "balance" -> add just enough to bring
            the defect count up to the good count (avoids flipping the imbalance
            the other way); None -> use all synthetic images (legacy behaviour).
    """

    def __init__(self, root, split="train", transform=None,
                 few_shot_n=None, synthetic_dir=None, synthetic_n=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform if transform is not None else _default_transform()
        self.few_shot_n = few_shot_n
        self.synthetic_dir = synthetic_dir
        self.synthetic_n = synthetic_n
        self.label_names = LABEL_NAMES

        if split not in SPLIT_FRACTIONS:
            raise ValueError(f"split must be one of {sorted(SPLIT_FRACTIONS)}, got {split!r}")
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"category folder not found: {self.root}. Run "
                f"`python -m src.data.download --categories <cat> --out data/raw` first."
            )

        self.samples = self._build_samples()  # list of (path, label)

    def _build_samples(self) -> list[tuple[Path, int]]:
        good_all, defect_all = _index_category(self.root)
        if not good_all and not defect_all:
            raise FileNotFoundError(f"no images found under {self.root}")

        # Stratify: partition each class independently (distinct seeds so the
        # two shuffles don't correlate), then take this split's slice.
        good = _partition(good_all, SPLIT_SEED)[self.split]
        defect = _partition(defect_all, SPLIT_SEED + 1)[self.split]

        samples = [(p, GOOD) for p in good] + [(p, DEFECT) for p in defect]

        # few_shot_n + synthetic mixing apply to the TRAIN split only.
        if self.split == "train":
            if self.few_shot_n is not None:
                real_good = [(p, lab) for p, lab in samples if lab == GOOD]
                real_defect = [(p, lab) for p, lab in samples if lab == DEFECT]
                # `defect` is already a deterministic shuffle -> first-n is a
                # reproducible random subset.
                samples = real_good + real_defect[: self.few_shot_n]
            if self.synthetic_dir is not None:
                syn = _list_images(Path(self.synthetic_dir))
                cap = self.synthetic_n
                if cap == "balance":
                    n_good = sum(1 for _, lab in samples if lab == GOOD)
                    n_defect = sum(1 for _, lab in samples if lab == DEFECT)
                    cap = max(0, n_good - n_defect)  # fill defects up to good count
                if cap is not None:
                    syn = syn[:cap]
                samples += [(p, DEFECT) for p in syn]

        return samples

    def class_counts(self) -> dict[str, int]:
        """{'good': n0, 'defect': n1} for the current split — handy for sanity checks."""
        counts = {GOOD: 0, DEFECT: 0}
        for _, label in self.samples:
            counts[label] += 1
        return {LABEL_NAMES[k]: v for k, v in counts.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return img, label


def build_loaders(category, batch_size=32, few_shot_n=None, synthetic_dir=None,
                  synthetic_n=None, data_root="data/raw", img_size=DEFAULT_IMG_SIZE,
                  traditional_aug=True, num_workers=0, seed=SPLIT_SEED):
    """Return (train_loader, val_loader, test_loader) for a category.

    Args:
        category: MVTec category name (e.g. "bottle").
        batch_size: loader batch size.
        few_shot_n: cap on real defect images in the train split (ablation).
        synthetic_dir: dir of synthetic defects to fold into the train split.
        synthetic_n: cap on synthetic images (int) or "balance" to match the
            good count; None uses all of them.
        data_root: where extracted categories live (default "data/raw").
        img_size: square resize edge length.
        traditional_aug: enable Baseline-2 flips/rotations/jitter on the train
            split (val/test always use the plain eval transform).
        num_workers: DataLoader workers (default 0 — safest on Windows).
        seed: reserved for reproducibility; the split itself uses SPLIT_SEED.
    """
    root = Path(data_root) / category
    train_tf = train_transforms(img_size=img_size, traditional_aug=traditional_aug)
    eval_tf = _default_transform(img_size=img_size)

    train_ds = DefectDataset(root, "train", train_tf,
                             few_shot_n=few_shot_n, synthetic_dir=synthetic_dir,
                             synthetic_n=synthetic_n)
    val_ds = DefectDataset(root, "val", eval_tf)
    test_ds = DefectDataset(root, "test", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader
