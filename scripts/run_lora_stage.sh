#!/usr/bin/env bash
# LoRA-only stage of the LODO sweep: train every per-fold generator for the
# given categories, sequentially (single shared GPU), WITHOUT proceeding to
# generation or classifier runs (use scripts/run_lodo_sweep.sh for the full
# per-fold pipeline).
#
# Usage: bash scripts/run_lora_stage.sh [category ...]   (default: bottle hazelnut carpet)
#
# Resumable: a fold whose checkpoint manifest verifies complete
# (fold_lora_complete) is skipped; an interrupted fold resumes from its last
# checkpoint-<step>. Logs per fold to experiments/lora_logs/<cat>_<type>.log
# with a summary.csv of exit codes and wall-clock seconds.
set -u
CATS=${@:-"bottle hazelnut carpet"}
LOGDIR=experiments/lora_logs
SUMMARY=${LOGDIR}/summary.csv
mkdir -p "${LOGDIR}"
[ -f "${SUMMARY}" ] || echo "category,holdout,exit_code,seconds,skipped" > "${SUMMARY}"

for CAT in ${CATS}; do
  TYPES=$(python -m src.data.dataset "${CAT}") || { echo "!! ${CAT}: fold discovery failed -- downloaded?"; continue; }
  for TYPE in ${TYPES}; do
    if python -c "import sys; from src.generative.train_lora import fold_lora_complete; \
sys.exit(0 if fold_lora_complete('${CAT}', '${TYPE}') else 1)"; then
      echo "== ${CAT}/${TYPE}: already complete (manifest verified), skipping =="
      echo "${CAT},${TYPE},0,0,yes" >> "${SUMMARY}"
      continue
    fi
    echo "== ${CAT}/${TYPE}: training =="
    T0=$(date +%s)
    python -m src.generative.train_lora --category "${CAT}" \
      --holdout-defect-type "${TYPE}" --config configs/diffusion_lora.yaml \
      > "${LOGDIR}/${CAT}_${TYPE}.log" 2>&1
    RC=$?
    T1=$(date +%s)
    echo "${CAT},${TYPE},${RC},$((T1-T0)),no" >> "${SUMMARY}"
    echo "== ${CAT}/${TYPE}: exit=${RC} elapsed=$((T1-T0))s =="
    if [ ${RC} -ne 0 ]; then
      echo "!! ${CAT}/${TYPE} FAILED (rc=${RC}) -- see ${LOGDIR}/${CAT}_${TYPE}.log"
    fi
  done
done
# Final status: one line per fold, read from the manifests on disk.
echo "ALL FOLDS PROCESSED -- final status:"
python - ${CATS} <<'EOF'
import sys
from src.data.dataset import defect_types
from src.generative.train_lora import fold_lora_complete

for cat in sys.argv[1:]:
    for t in defect_types(cat):
        state = "complete" if fold_lora_complete(cat, t) else "INCOMPLETE"
        print(f"  {cat}/{t}: {state}")
EOF
