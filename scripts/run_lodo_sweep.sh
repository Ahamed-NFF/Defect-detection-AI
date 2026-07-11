#!/usr/bin/env bash
# LODO sweep: run baseline1/baseline2/ours for EVERY defect-type fold of one
# category. Usage: bash scripts/run_lodo_sweep.sh <category>
#
# Folds are discovered from the filesystem (src.data.dataset's CLI, backed by
# defect_types()) -- nothing here hardcodes which categories exist or how
# many folds a category has. A category with no data/raw/<category>/test/
# yet (e.g. metal_nut/screw/tile until they're downloaded and verified) is
# skipped with a clear message instead of erroring or silently doing nothing;
# once its data lands this script picks up its folds with no changes.
#
# Reuses the SAME three exp_<category>_{baseline1,baseline2,ours}.yaml configs
# as scripts/run_all_experiments.sh (unchanged) -- no per-fold YAML files are
# generated. The fold is selected purely via --holdout-defect-type, and
# src.classifier.train appends "__lodo_<type>" to run_name so each fold's
# checkpoint/result JSON gets its own filename.
#
# CAVEAT ("ours" rows): this reuses the SAME per-category synthetic_dir /
# diffusion LoRA checkpoint as the non-LODO experiments (data/synthetic/<cat>,
# trained on ALL real defects of that category). That means the generative
# model has seen the held-out type's real images during its own training,
# even though the classifier's discriminative training never does -- a
# caveat for the "does synthetic augmentation generalise to unseen types"
# claim. Retraining LoRA per fold (excluding the held-out type) would close
# this gap but multiplies the generative budget by the fold count; flagged
# here as an open decision, not implemented.
set -e
CAT=${1:?"usage: bash scripts/run_lodo_sweep.sh <category>"}

TYPES=$(python -m src.data.dataset "${CAT}" 2>&1) || {
  echo "skipping ${CAT}: ${TYPES}"
  exit 0
}

echo "LODO folds for ${CAT}: ${TYPES}"
for TYPE in ${TYPES}; do
  echo "=== ${CAT} | holdout=${TYPE} ==="
  python -m src.classifier.train --config configs/exp_${CAT}_baseline1.yaml \
    --holdout-defect-type "${TYPE}"
  python -m src.classifier.train --config configs/exp_${CAT}_baseline2.yaml \
    --holdout-defect-type "${TYPE}"
  python -m src.classifier.train --config configs/exp_${CAT}_ours.yaml \
    --holdout-defect-type "${TYPE}"
done
python -m src.eval.metrics  # assemble comparison + FID tables
