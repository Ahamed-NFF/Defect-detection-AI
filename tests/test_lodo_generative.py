"""
Generator/classifier file-set equality for the LODO generative pipeline.

The paper's claim requires that each fold's LoRA generator trains on EXACTLY
the real defect images the classifier trains on for that fold — the held-out
type and the val slice excluded. For every downloaded category and every fold:

    * the file set the generative path WOULD train on
      (src.data.dataset.lodo_train_defect_paths — what train_lora consumes)
      equals the defect files in the classifier's actual train split
      (build_loaders_lodo — what src.classifier.train consumes),
    * that set contains NONE of the held-out type's files,
    * the fold manifest (train_lora.build_fold_manifest) self-documents the
      equality and refuses leaky inputs,
    * generate.py's manifest guard refuses missing/mismatched/leaky manifests,
    * the per-fold synthetic path is a sibling of data/synthetic/ (never
      nested inside the deprecated shared pool, which is scanned recursively).

CPU/filesystem only — no GPU, no LoRA training, no generation.

Run with prints visible:
    pytest -s tests/test_lodo_generative.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.dataset import (
    DEFECT,
    build_loaders_lodo,
    defect_types,
    lodo_lora_checkpoint_dir,
    lodo_synthetic_dir,
    lodo_train_defect_paths,
)
from src.generative.generate import check_fold_manifest
from src.generative.train_lora import build_fold_manifest, format_config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
DIFFUSION_CONFIG = REPO_ROOT / "configs" / "diffusion_lora.yaml"


def _available_categories() -> list[str]:
    if not DATA_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in DATA_ROOT.iterdir()
        if p.is_dir() and defect_types(p.name, data_root=str(DATA_ROOT))
    )


CATEGORIES = _available_categories()


@pytest.mark.skipif(not CATEGORIES, reason=f"no downloaded categories under {DATA_ROOT}")
def test_generator_trains_on_exactly_the_classifier_train_defects():
    print(f"\ngenerator/classifier file-set equality over: {CATEGORIES}\n")
    for category in CATEGORIES:
        for holdout in defect_types(category, data_root=str(DATA_ROOT)):
            gen_files = lodo_train_defect_paths(category, holdout, data_root=str(DATA_ROOT))

            train_loader, _, _ = build_loaders_lodo(category, holdout,
                                                    data_root=str(DATA_ROOT), num_workers=0)
            clf_files = sorted(p for p, label in train_loader.dataset.samples
                               if label == DEFECT)

            assert gen_files == clf_files, (
                f"{category}/holdout={holdout}: generator and classifier train-defect "
                f"file sets differ (generator={len(gen_files)}, classifier={len(clf_files)})"
            )
            heldout_in_set = [p for p in gen_files if p.parent.name == holdout]
            assert not heldout_in_set, (
                f"{category}/holdout={holdout}: held-out type leaked into the generator "
                f"training set: {[p.name for p in heldout_in_set][:5]}"
            )
            types = sorted({p.parent.name for p in gen_files})
            print(f"  {category}/holdout={holdout:<22} {len(gen_files):>3} files, "
                  f"types={types}  == classifier train defects, no held-out files")


@pytest.mark.skipif(not CATEGORIES, reason=f"no downloaded categories under {DATA_ROOT}")
def test_fold_manifest_documents_equality():
    cfg_raw = load_config(DIFFUSION_CONFIG)
    for category in CATEGORIES:
        cfg = format_config(cfg_raw, category)
        for holdout in defect_types(category, data_root=str(DATA_ROOT)):
            m = build_fold_manifest(category, holdout, cfg, data_root=str(DATA_ROOT))
            assert m["file_set_equal"] is True
            assert m["held_out_type"] == holdout
            assert holdout not in m["trained_on_types"]
            assert holdout in m["excluded_types"]
            assert m["n_train_files"] == m["classifier_train_defect_count"]
            assert m["train_files_sha256"] == m["classifier_train_defect_sha256"]
            assert m["protocol_version"] == "lodo_v1_train_defects_only"


@pytest.mark.skipif(not CATEGORIES, reason=f"no downloaded categories under {DATA_ROOT}")
def test_generate_guard_refuses_bad_manifests(tmp_path):
    category = CATEGORIES[0]
    types = defect_types(category, data_root=str(DATA_ROOT))
    holdout, other = types[0], types[1]
    cfg = format_config(load_config(DIFFUSION_CONFIG), category)
    manifest = build_fold_manifest(category, holdout, cfg, data_root=str(DATA_ROOT))
    manifest["status"] = "complete"
    manifest_path = tmp_path / "manifest.json"

    # correct fold passes
    manifest_path.write_text(json.dumps(manifest))
    assert check_fold_manifest(tmp_path, category, holdout)["held_out_type"] == holdout

    # missing manifest refuses
    with pytest.raises(SystemExit):
        check_fold_manifest(tmp_path / "nonexistent", category, holdout)

    # wrong fold refuses
    with pytest.raises(SystemExit):
        check_fold_manifest(tmp_path, category, other)

    # leaky manifest (held-out type among training types) refuses
    leaky = dict(manifest, trained_on_types=manifest["trained_on_types"] + [holdout])
    manifest_path.write_text(json.dumps(leaky))
    with pytest.raises(SystemExit):
        check_fold_manifest(tmp_path, category, holdout)

    # unfinished training refuses
    unfinished = dict(manifest, status="training")
    manifest_path.write_text(json.dumps(unfinished))
    with pytest.raises(SystemExit):
        check_fold_manifest(tmp_path, category, holdout)


def test_fold_paths_are_outside_the_deprecated_shared_pool():
    syn = lodo_synthetic_dir("bottle", "broken_large")
    deprecated_pool = Path("data/synthetic")
    assert deprecated_pool not in syn.parents, (
        f"{syn} nests under {deprecated_pool}, which the deprecated pooled configs "
        f"scan recursively -- fold images would leak into pooled re-runs"
    )
    assert syn == Path("data/synthetic_lodo/bottle/holdout_broken_large")
    assert lodo_lora_checkpoint_dir("bottle", "broken_large") == Path(
        "experiments/checkpoints/bottle_holdout_broken_large_lora")


if __name__ == "__main__":
    if not CATEGORIES:
        raise SystemExit(
            f"No categories downloaded under {DATA_ROOT}. Run:\n"
            f"  python -m src.data.download --categories bottle hazelnut carpet --out data/raw"
        )
    test_generator_trains_on_exactly_the_classifier_train_defects()
    test_fold_manifest_documents_equality()
    test_fold_paths_are_outside_the_deprecated_shared_pool()
    print("\nOK")
