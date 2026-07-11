#!/usr/bin/env bash
# LODO sweep: for EVERY defect-type fold of one category, run the full
# per-fold pipeline:
#
#   1. train a PER-FOLD LoRA generator on exactly the classifier's train
#      defects for that fold (held-out type + val slice excluded; the file
#      list comes from src.data.dataset.lodo_train_defect_paths, the same
#      source of truth the classifier uses) -> writes checkpoint + manifest
#      to experiments/checkpoints/<cat>_holdout_<type>_lora/
#   2. generate that fold's synthetic images (manifest-guarded: refuses a
#      checkpoint whose manifest doesn't match the fold)
#      -> data/synthetic_lodo/<cat>/holdout_<type>/
#   3. run baseline1/baseline2/ours classifiers for the fold; the "ours" run
#      derives its synthetic_dir from the SAME shared path helper the
#      generator wrote to (src.classifier.train overrides the config's
#      shared-pool synthetic_dir under LODO).
#
# Usage:
#   bash scripts/run_lodo_sweep.sh <category>            # run the sweep
#   bash scripts/run_lodo_sweep.sh <category> --dry-run  # print the per-fold
#       plan (training types, checkpoint dir, synthetic dirs) without running
#       anything -- no GPU needed
#
# Folds are discovered from the filesystem (src.data.dataset's CLI) -- nothing
# here hardcodes which categories exist or how many folds a category has. A
# category with no data/raw/<category>/test/ yet (e.g. metal_nut/screw/tile
# until they're downloaded and verified) is skipped with a clear message.
#
# Resumable (shared GPU queue): a fold whose generator checkpoint + manifest
# already exist and verify (fold_lora_complete) skips LoRA retraining; a fold
# whose synthetic dir already holds >= N_SYNTH images skips generation
# (generate.py is also internally resumable if partially done). Classifier
# runs are re-run unconditionally -- they're the cheap part.
#
# Reuses the SAME three exp_<category>_{baseline1,baseline2,ours}.yaml configs
# for every fold -- no per-fold YAML files. The fold is selected purely via
# --holdout-defect-type.
set -e
CAT=${1:?"usage: bash scripts/run_lodo_sweep.sh <category> [--dry-run]"}
MODE=${2:-}
N_SYNTH=800

TYPES=$(python -m src.data.dataset "${CAT}" 2>&1) || {
  echo "skipping ${CAT}: ${TYPES}"
  exit 0
}
echo "LODO folds for ${CAT}: ${TYPES}"

if [ "${MODE}" == "--dry-run" ]; then
  for TYPE in ${TYPES}; do
    echo "=== ${CAT} | holdout=${TYPE} (dry run) ==="
    python - "${CAT}" "${TYPE}" <<'EOF'
import sys
from src.data.dataset import (lodo_lora_checkpoint_dir, lodo_synthetic_dir,
                              lodo_train_defect_paths)
from src.generative.train_lora import fold_lora_complete

cat, holdout = sys.argv[1], sys.argv[2]
files = lodo_train_defect_paths(cat, holdout)
types = sorted({p.parent.name for p in files})
ckpt = lodo_lora_checkpoint_dir(cat, holdout)
syn = lodo_synthetic_dir(cat, holdout)
print(f"  generator trains on types : {types}  ({len(files)} real defect files)")
print(f"  generator checkpoint dir  : {ckpt}")
print(f"  synthetic output dir      : {syn}")
print(f"  'ours' classifier reads   : {syn}  (same shared helper, derived in src.classifier.train)")
print(f"  lora complete already?    : {fold_lora_complete(cat, holdout)}")
EOF
  done
  exit 0
fi

for TYPE in ${TYPES}; do
  echo "=== ${CAT} | holdout=${TYPE} ==="

  # -- 1. per-fold LoRA generator (skip if already complete + manifest verifies)
  if python -c "import sys; from src.generative.train_lora import fold_lora_complete; \
sys.exit(0 if fold_lora_complete('${CAT}', '${TYPE}') else 1)"; then
    echo "  fold LoRA already complete (manifest verified) -- skipping training"
  else
    python -m src.generative.train_lora --category "${CAT}" \
      --holdout-defect-type "${TYPE}" --config configs/diffusion_lora.yaml
  fi

  # -- 2. per-fold synthetic images (skip if target count already present)
  SYN_DIR=$(python -c "from src.data.dataset import lodo_synthetic_dir; \
print(lodo_synthetic_dir('${CAT}', '${TYPE}'))")
  HAVE=$(find "${SYN_DIR}" -maxdepth 1 -type f 2>/dev/null | wc -l)
  if [ "${HAVE}" -ge "${N_SYNTH}" ]; then
    echo "  ${SYN_DIR} already has ${HAVE} >= ${N_SYNTH} images -- skipping generation"
  else
    python -m src.generative.generate --category "${CAT}" \
      --holdout-defect-type "${TYPE}" --n "${N_SYNTH}"
  fi

  # -- 3. classifier runs for the fold
  python -m src.classifier.train --config configs/exp_${CAT}_baseline1.yaml \
    --holdout-defect-type "${TYPE}"
  python -m src.classifier.train --config configs/exp_${CAT}_baseline2.yaml \
    --holdout-defect-type "${TYPE}"
  python -m src.classifier.train --config configs/exp_${CAT}_ours.yaml \
    --holdout-defect-type "${TYPE}"
done
python -m src.eval.metrics  # assemble comparison + FID tables
