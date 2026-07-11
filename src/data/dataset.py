"""
PyTorch Dataset + DataLoader factories for MVTec categories.

Two split protocols live here:

    1. LEAVE-ONE-DEFECT-TYPE-OUT (LODO) -- current protocol, use this for all
       new experiments. LodoDefectDataset / build_loaders_lodo(). For a given
       category and a held-out defect type, train sees every OTHER defect type
       plus MVTec's own train/good; test is the held-out type entirely (plus a
       slice of MVTec's test/good); val is a seeded slice of the non-held-out
       defect pool plus a seeded slice of test/good. The held-out type never
       appears in train or val, so the split is leak-free by construction: the
       partition boundary is defect-TYPE identity (a folder name), not a
       re-shuffled file list. See build_loaders_lodo()'s docstring for the full
       protocol, and defect_types() to enumerate the folds for a category.

    2. POOLED SPLIT (DEPRECATED) -- DefectDataset / build_loaders() /
       build_loaders_multi(). Pools train/good + test/good into one "good"
       bucket and all test/<type> folders into one "defect" bucket, then does
       a single deterministic seeded shuffle+slice (70/15/15) over each pooled
       bucket. This ERASES MVTec's own train/test boundary and defect-type
       identity, which inflated early metrics (AUROC=1.000 on almost every
       row). Kept ONLY so results already in experiments/results/ (produced
       before the LODO rework) stay reproducible -- do not use it for new
       experiments and do not change its behaviour.

The training knobs (apply to the TRAIN split only, in both protocols):
    - few_shot_n   : cap the number of real defect images (the ablation lever).
    - synthetic_dir: fold in diffusion-generated defect images (the "ours" runs).

Owner: Member 1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as T

from .augment import train_transforms

GOOD, DEFECT = 0, 1
LABEL_NAMES = {GOOD: "good", DEFECT: "defect"}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Split fractions for the DEPRECATED pooled re-split (see module docstring).
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42  # deprecated pooled split only -- every DefectDataset instance agrees on it

# --- LODO split constants -------------------------------------------------
# Fraction of MVTec's test/good pool assigned to val (the rest goes to test).
# Symmetric (50/50) mirrors the deprecated path's val:test = 15:15 ratio.
LODO_GOOD_VAL_FRACTION = 0.5
# Fraction of the non-held-out defect pool (all defect types EXCEPT the one
# being held out) assigned to val, per fold. Deliberately generous (not a
# smaller value) so val has enough defects for stable best-epoch selection /
# threshold calibration even on thin categories -- e.g. bottle's 3-type case
# leaves only ~41-43 non-held-out defects per fold; 0.25 of that is ~10-11 val
# images instead of ~6-7 at 0.15, at the cost of a bit less training data.
LODO_VAL_DEFECT_FRACTION = 0.25

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
    """DEPRECATED (pooled/re-split protocol -- see module docstring).

    Kept unmodified so results already in experiments/results/ stay
    reproducible. Use LodoDefectDataset for new experiments.

    Binary good/defect dataset over one MVTec category.

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
    """DEPRECATED (pooled/re-split protocol -- see module docstring).

    Kept unmodified so results already in experiments/results/ stay
    reproducible. Use build_loaders_lodo() for new experiments.

    Return (train_loader, val_loader, test_loader) for a category.

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
            NOTE: this param is NOT actually honoured (pre-existing bug, kept
            as-is deliberately since fixing it could silently change already-
            reproduced results) -- build_loaders_lodo()'s seed IS genuinely
            honoured; use that for new experiments.
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


class _ConcatDefects(ConcatDataset):
    """ConcatDataset that keeps a class_counts() summed across its children."""

    def class_counts(self) -> dict[str, int]:
        total = {"good": 0, "defect": 0}
        for d in self.datasets:
            for k, v in d.class_counts().items():
                total[k] += v
        return total


def build_loaders_multi(categories, batch_size=32, few_shot_n=None,
                        use_synthetic=False, synthetic_root="data/synthetic",
                        synthetic_n="balance", data_root="data/raw",
                        img_size=DEFAULT_IMG_SIZE, traditional_aug=True, num_workers=0):
    """DEPRECATED (pooled/re-split protocol -- see module docstring).

    Kept unmodified so results already in experiments/results/ (e.g. the
    combined all-category "all_ours" run) stay reproducible. There is no LODO
    equivalent yet for the combined multi-category model.

    Train/val/test loaders pooled across several categories (one combined model).

    Each category is split independently (its own deterministic 70/15/15) and the
    matching splits are concatenated, so every category is represented in train,
    val, and test. For the augmented runs, each category's own synthetic dir
    (synthetic_root/<category>) is mixed into its train split and balanced
    separately via synthetic_n.
    """
    train_tf = train_transforms(img_size=img_size, traditional_aug=traditional_aug)
    eval_tf = _default_transform(img_size=img_size)

    train_sets, val_sets, test_sets = [], [], []
    for cat in categories:
        root = Path(data_root) / cat
        syn_dir = str(Path(synthetic_root) / cat) if use_synthetic else None
        train_sets.append(DefectDataset(root, "train", train_tf, few_shot_n=few_shot_n,
                                        synthetic_dir=syn_dir, synthetic_n=synthetic_n))
        val_sets.append(DefectDataset(root, "val", eval_tf))
        test_sets.append(DefectDataset(root, "test", eval_tf))

    train_loader = DataLoader(_ConcatDefects(train_sets), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(_ConcatDefects(val_sets), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(_ConcatDefects(test_sets), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# LEAVE-ONE-DEFECT-TYPE-OUT (LODO) split -- current protocol
# --------------------------------------------------------------------------- #
# Self-identifying protocol tag recorded in every per-fold generator manifest:
# "train_defects_only" = the generator trains on exactly the classifier's
# TRAIN-split defects (the val slice of the non-held-out pool is excluded),
# per lodo_train_defect_paths(). Bump this string if the protocol changes.
LODO_PROTOCOL_VERSION = "lodo_v1_train_defects_only"


def defect_types(category, data_root="data/raw") -> list[str]:
    """Sorted defect-type directory names under <data_root>/<category>/test/.

    Excludes "good". Enumerates the available LODO folds for a category (one
    fold per type, that type held out). Returns [] if the category hasn't
    been downloaded yet (its test/ dir doesn't exist) -- callers that need to
    distinguish "not downloaded" from "downloaded but no defects" should
    additionally check Path(data_root, category, "test").is_dir().
    """
    test_dir = Path(data_root) / category / "test"
    if not test_dir.is_dir():
        return []
    return sorted(d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good")


def _seeded_shuffle(paths: list[Path], seed: int) -> list[Path]:
    """Deterministically shuffle a COPY of paths given seed."""
    items = list(paths)
    random.Random(seed).shuffle(items)
    return items


def _mix_train_extras(samples, few_shot_n, synthetic_dir, synthetic_n):
    """Apply the few-shot cap + synthetic mixing to a TRAIN split's sample list.

    Same semantics as the inline logic in DefectDataset._build_samples, but
    factored out for LodoDefectDataset. DefectDataset keeps its own inline
    copy (not this helper) so the deprecated pooled path's behaviour can't
    drift if this helper is ever touched.
    """
    if few_shot_n is not None:
        real_good = [(p, lab) for p, lab in samples if lab == GOOD]
        real_defect = [(p, lab) for p, lab in samples if lab == DEFECT]
        # `defect` entries are already a deterministic shuffle -> first-n is a
        # reproducible random subset.
        samples = real_good + real_defect[:few_shot_n]
    if synthetic_dir is not None:
        syn = _list_images(Path(synthetic_dir))
        cap = synthetic_n
        if cap == "balance":
            n_good = sum(1 for _, lab in samples if lab == GOOD)
            n_defect = sum(1 for _, lab in samples if lab == DEFECT)
            cap = max(0, n_good - n_defect)  # fill defects up to good count
        if cap is not None:
            syn = syn[:cap]
        samples = samples + [(p, DEFECT) for p in syn]
    return samples


class LodoDefectDataset(Dataset):
    """Binary good/defect dataset for one leave-one-defect-type-out (LODO) fold.

    The partition boundary is defect-TYPE identity (a test/<type> folder
    name), not a re-shuffled file list, so it's leak-free by construction:
    the held-out type's images can only ever end up in the 'test' split.

    Protocol (category root = data_root/category):
        good (train)    = ALL of root/train/good (MVTec's own boundary, held
            fixed across every fold of a category).
        good (val/test) = root/test/good, seeded-shuffled and split
            LODO_GOOD_VAL_FRACTION / (1 - LODO_GOOD_VAL_FRACTION) into
            val/test. The same split is reused by every fold of a category
            (it doesn't depend on which defect type is held out).
        defect (test)     = ALL images of root/test/<holdout_defect_type>.
        defect (train/val) = images of every OTHER test/<type> folder,
            pooled, seeded-shuffled, and split LODO_VAL_DEFECT_FRACTION into
            val (the rest into train). Only non-held-out-type defects ever
            reach val, so model selection / threshold calibration never sees
            the held-out type's distribution.

    Args:
        root: path to the extracted category folder (e.g. data/raw/bottle).
        holdout_defect_type: the test/<type> folder name to hold out for this
            fold (must be one of defect_types(category)).
        split: 'train' | 'val' | 'test'.
        transform: torchvision transform pipeline. Defaults to resize+normalise.
        few_shot_n: if set, cap the number of real defect images in the TRAIN
            split (the few-shot ablation lever; same semantics as DefectDataset).
        synthetic_dir: optional path to synthetic defect images mixed into the
            train split only (the diffusion-augmented "ours" runs).
        synthetic_n: cap on synthetic images mixed in (train only); same
            semantics as DefectDataset (int, "balance", or None).
        seed: drives BOTH the good val/test split and the defect train/val
            split (offset by +1 for the latter so the two shuffles don't
            correlate, mirroring the deprecated path's SPLIT_SEED /
            SPLIT_SEED+1 convention). Genuinely honoured -- unlike
            build_loaders()'s unused seed param, passing a different value
            here produces a different, still fully deterministic, partition.
    """

    def __init__(self, root, holdout_defect_type, split="train", transform=None,
                 few_shot_n=None, synthetic_dir=None, synthetic_n=None, seed=SPLIT_SEED):
        self.root = Path(root)
        self.holdout_defect_type = holdout_defect_type
        self.split = split
        self.transform = transform if transform is not None else _default_transform()
        self.few_shot_n = few_shot_n
        self.synthetic_dir = synthetic_dir
        self.synthetic_n = synthetic_n
        self.seed = seed
        self.label_names = LABEL_NAMES

        if split not in SPLIT_FRACTIONS:
            raise ValueError(f"split must be one of {sorted(SPLIT_FRACTIONS)}, got {split!r}")
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"category folder not found: {self.root}. Run "
                f"`python -m src.data.download --categories <cat> --out data/raw` first."
            )
        types = defect_types(self.root.name, data_root=str(self.root.parent))
        if holdout_defect_type not in types:
            raise ValueError(
                f"holdout_defect_type={holdout_defect_type!r} not found under "
                f"{self.root / 'test'} (available: {types})"
            )

        self.samples = self._build_samples()  # list of (path, label)

    def _build_samples(self) -> list[tuple[Path, int]]:
        types = defect_types(self.root.name, data_root=str(self.root.parent))
        nonheld_types = [t for t in types if t != self.holdout_defect_type]

        # --- good: MVTec's own train/test boundary, seeded split of test/good ---
        train_good = _list_images(self.root / "train" / "good")
        test_good_all = sorted(_list_images(self.root / "test" / "good"))
        good_shuffled = _seeded_shuffle(test_good_all, self.seed)
        n_val_good = int(round(len(good_shuffled) * LODO_GOOD_VAL_FRACTION))
        val_good, test_good = good_shuffled[:n_val_good], good_shuffled[n_val_good:]

        # --- defect: held-out type -> test; every other type -> seeded train/val ---
        test_defect = sorted(_list_images(self.root / "test" / self.holdout_defect_type))

        nonheld_pool: list[Path] = []
        for t in nonheld_types:
            nonheld_pool += _list_images(self.root / "test" / t)
        nonheld_shuffled = _seeded_shuffle(sorted(nonheld_pool), self.seed + 1)
        n_val_defect = int(round(len(nonheld_shuffled) * LODO_VAL_DEFECT_FRACTION))
        val_defect, train_defect = nonheld_shuffled[:n_val_defect], nonheld_shuffled[n_val_defect:]

        if self.split == "train":
            good, defect = train_good, train_defect
        elif self.split == "val":
            good, defect = val_good, val_defect
        else:  # "test"
            good, defect = test_good, test_defect

        samples = [(p, GOOD) for p in good] + [(p, DEFECT) for p in defect]

        # few_shot_n + synthetic mixing apply to the TRAIN split only.
        if self.split == "train":
            samples = _mix_train_extras(samples, self.few_shot_n,
                                        self.synthetic_dir, self.synthetic_n)

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


def build_loaders_lodo(category, holdout_defect_type, batch_size=32, few_shot_n=None,
                       synthetic_dir=None, synthetic_n=None, data_root="data/raw",
                       img_size=DEFAULT_IMG_SIZE, traditional_aug=True, num_workers=0,
                       seed=SPLIT_SEED):
    """Return (train_loader, val_loader, test_loader) for ONE LODO fold.

    See LodoDefectDataset's docstring for the exact protocol. `seed` is
    genuinely honoured here (drives the good val/test split and the defect
    train/val split) -- pass a different value to get a different, still
    fully deterministic, partition.

    Args:
        category: MVTec category name (e.g. "bottle").
        holdout_defect_type: the test/<type> folder to hold out (the fold).
            Use defect_types(category, data_root) to enumerate valid values.
        batch_size, few_shot_n, synthetic_dir, synthetic_n, data_root,
            img_size, traditional_aug, num_workers: same semantics as
            build_loaders().
        seed: seeds the good val/test split and the defect train/val split
            (see LodoDefectDataset). Default SPLIT_SEED=42.
    """
    root = Path(data_root) / category
    train_tf = train_transforms(img_size=img_size, traditional_aug=traditional_aug)
    eval_tf = _default_transform(img_size=img_size)

    train_ds = LodoDefectDataset(root, holdout_defect_type, "train", train_tf,
                                 few_shot_n=few_shot_n, synthetic_dir=synthetic_dir,
                                 synthetic_n=synthetic_n, seed=seed)
    val_ds = LodoDefectDataset(root, holdout_defect_type, "val", eval_tf, seed=seed)
    test_ds = LodoDefectDataset(root, holdout_defect_type, "test", eval_tf, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# LODO single source of truth for the generative pipeline
#
# The per-fold LoRA generator must train on EXACTLY the real defect images the
# classifier trains on for that fold -- no more (no held-out type, no val
# slice), no less. These helpers are that single source of truth, plus the one
# place the per-fold checkpoint/synthetic path conventions live. They are
# imported by src/generative/train_lora.py, src/generative/generate.py,
# src/classifier/train.py, and scripts/run_lodo_sweep.sh -- path agreement
# between producer and consumer is enforced by these shared functions, never
# by coincidentally-equal strings. Do NOT reimplement any of this in the
# generative code.
# --------------------------------------------------------------------------- #
def lodo_train_defect_paths(category, holdout_defect_type, data_root="data/raw",
                            seed=SPLIT_SEED) -> list[Path]:
    """The real defect image files the classifier TRAINS on for one LODO fold.

    Implemented by instantiating LodoDefectDataset's train split itself (no
    few-shot cap, no synthetic mixing) and filtering out the good images, so
    this is definitionally the classifier's training defect set -- if the
    split logic ever changes, this changes with it and the two cannot drift.
    Excludes the held-out type AND the val slice of the non-held-out pool
    (LODO_PROTOCOL_VERSION = "train_defects_only").

    Returned sorted for stable manifests/hashes; training order is up to the
    caller's shuffling.
    """
    root = Path(data_root) / category
    ds = LodoDefectDataset(root, holdout_defect_type, split="train", seed=seed)
    return sorted(p for p, label in ds.samples if label == DEFECT)


def lodo_lora_checkpoint_dir(category, holdout_defect_type,
                             checkpoints_root="experiments/checkpoints") -> Path:
    """Where the per-fold LoRA generator checkpoints + manifest.json live."""
    return Path(checkpoints_root) / f"{category}_holdout_{holdout_defect_type}_lora"


def lodo_synthetic_dir(category, holdout_defect_type,
                       synthetic_root="data/synthetic_lodo") -> Path:
    """Where a fold's synthetic images are written and read from.

    Deliberately a SIBLING of data/synthetic/ (not nested inside it): the
    deprecated pooled "ours" configs point synthetic_dir at
    data/synthetic/<category>, which _list_images scans RECURSIVELY -- nesting
    fold dirs under it would let a re-run of an old pooled config silently
    ingest fold images. The sibling root makes that structurally impossible.
    """
    return Path(synthetic_root) / category / f"holdout_{holdout_defect_type}"


# --------------------------------------------------------------------------- #
# CLI: list the LODO folds available for a category (used by sweep scripts)
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="List LODO defect-type folds for a category (filesystem-only, no network)."
    )
    p.add_argument("category")
    p.add_argument("--data-root", default="data/raw")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    types = defect_types(args.category, args.data_root)
    if not types:
        raise SystemExit(
            f"no defect types found for category={args.category!r} under "
            f"{args.data_root!r} -- has it been downloaded? "
            f"(python -m src.data.download --categories {args.category})"
        )
    print(" ".join(types))


if __name__ == "__main__":
    main()
