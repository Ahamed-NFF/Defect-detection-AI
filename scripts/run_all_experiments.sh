#!/usr/bin/env bash
# Convenience: run the full experiment sweep for one category.
# Usage: bash scripts/run_all_experiments.sh <category>
#   category is any of: bottle hazelnut metal_nut screw carpet tile
#   (any category with configs/exp_<category>_{baseline1,baseline2,ours}.yaml)
set -e
CAT=${1:-bottle}
python -m src.classifier.train --config configs/exp_${CAT}_baseline1.yaml
python -m src.classifier.train --config configs/exp_${CAT}_baseline2.yaml
python -m src.generative.train_lora --category ${CAT} --config configs/diffusion_lora.yaml
python -m src.generative.generate --category ${CAT} --n 800 \
  --lora experiments/checkpoints/${CAT}_lora --out data/synthetic/${CAT}
python -m src.generative.fid --real data/raw/${CAT}/test --fake data/synthetic/${CAT} \
  --out experiments/results/fid_${CAT}.json
python -m src.classifier.train --config configs/exp_${CAT}_ours.yaml
python -m src.eval.metrics  # assemble comparison + FID tables
