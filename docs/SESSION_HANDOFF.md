# Session handoff — GPU machine migration (2026-07-11)

State snapshot for resuming the LODO experiment pipeline on a new GPU machine.
Everything below was verified from disk (manifests/checkpoints), not session
memory.

## Where we are in the pipeline

The LODO (leave-one-defect-type-out) rebuild is **complete and committed**:
split protocol, per-fold LoRA generative pipeline with manifests + leak
guards, sweep scripts, and CPU verification tests (12/12 folds leak-free,
generator/classifier file-set equality proven — see `tests/test_lodo_*.py`).

Execution progress:

| Stage | Status |
|---|---|
| Stage 0: CPU sanity run of classifier LODO path | done (artifacts cleaned up) |
| Stage 1: 12 per-fold LoRA generator trainings | **10/12 complete** (see below) |
| Stage 2: per-fold synthetic generation (N_SYNTH=500) | **not started** |
| Stage 3: 36 classifier runs (baseline1/2/ours × 12 folds) | **not started** |
| Stage 4: metrics assembly + FID | not started |

## Per-fold LoRA status (verified from manifests on disk)

| Fold | Status | Train imgs | Wall-clock |
|---|---|---|---|
| bottle/broken_large | complete | 32 | 356s |
| bottle/broken_small | complete | 31 | 345s |
| bottle/contamination | complete | 32 | 346s |
| hazelnut/crack | complete | 39 | 343s |
| hazelnut/cut | complete | 40 | 354s |
| hazelnut/hole | complete | 39 | 378s |
| hazelnut/print | complete | 40 | 377s |
| carpet/color | complete | 52 | 342s |
| carpet/cut | complete | 54 | 354s |
| carpet/hole | complete | 54 | ~350s |
| **carpet/metal_contamination** | **interrupted** (pre-first-checkpoint; manifest correctly says "training", will restart from scratch, ~6 min) | 54 | — |
| **carpet/thread** | **not started** | — | — |

No pathological losses observed (no NaN/divergence; final single-batch losses
0.14–0.37, which is normal diffusion-loss noise). ~350 s/fold at 1500 steps on
an RTX 5090.

## What's in git vs what must transfer out-of-band

**In git (origin/main):** all code, configs, sweep scripts, tests, and the 10
completed folds' `manifest.json` files (force-added past the
`experiments/checkpoints/` ignore rule — they document what each generator
trained on).

**Transfer out-of-band (scp/drive):** the 10 completed LoRA weight files —
`experiments/checkpoints/<fold>_lora/pytorch_lora_weights.safetensors`
(~12.3 MB each, **123 MB total**). Place them next to their committed
manifests at the same paths. The ~4.8 GB of `checkpoint-<step>/` dirs are
only mid-training resume states — completed folds never need them; do NOT
bother transferring them.

`fold_lora_complete()` requires manifest (status=complete) + weights together
before the sweep will skip a fold, so: weights transferred → those 10 folds
skip; weights missing → they harmlessly retrain (~6 min each).

The stale `experiments/checkpoints/carpet_holdout_metal_contamination_lora/`
dir (manifest status="training", no weights) can be deleted or left — the
sweep treats it as incomplete either way and rewrites it.

## Resuming on the new machine

```bash
# 0. clone, set up venv, install CUDA torch build (see environment notes),
#    then fetch the data (only bottle/hazelnut/carpet needed for this stage):
python -m src.data.download --categories bottle hazelnut carpet --out data/raw

# 1. drop the 10 transferred .safetensors files into their checkpoint dirs,
#    then finish the remaining LoRA folds (completed ones auto-skip):
bash scripts/run_lora_stage.sh                 # trains carpet metal_contamination + thread

# 2. after reviewing losses/manifests: generation + classifiers, per category.
#    run_lodo_sweep.sh does LoRA(skip)->generate(500 imgs)->baseline1/2/ours per fold:
bash scripts/run_lodo_sweep.sh bottle
bash scripts/run_lodo_sweep.sh hazelnut
bash scripts/run_lodo_sweep.sh carpet
#    (use "bash scripts/run_lodo_sweep.sh <cat> --dry-run" to preview wiring)

# 3. tables: python -m src.eval.metrics   (also run automatically by the sweep)
```

Do NOT use `scripts/run_all_experiments.sh` or the `data/synthetic/<cat>`
pools — deprecated pooled-split protocol, kept only for reproducing the old
module results.

## Environment notes for the new machine

- **CUDA/torch:** old machine ran torch 2.11.0+cu128 (CUDA 12.8) on an RTX
  5090; Blackwell-generation cards need a cu128+ build. diffusers 0.38.0,
  peft 0.19.1, transformers 5.10.2. `pip install -r requirements.txt` then a
  matching CUDA torch per docs/SETUP_WINDOWS.md.
- **Model caches (avoids network on first run):** ResNet-50 IMAGENET1K_V2 at
  `~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth` (~103 MB);
  Stable Diffusion base `runwayml/stable-diffusion-v1-5` in
  `~/.cache/huggingface/hub/` (several GB). Both auto-download if absent.
- **LLaVA stage (later):** prompt-eval runs used a local ollama serve —
  install ollama + pull the llava model before re-running
  `scripts/run_prompt_eval.py`.
- **Frontend (demo only):** `npm install --legacy-peer-deps` in `frontend/`.
- Seeds: everything derives from `SPLIT_SEED=42` in `src/data/dataset.py`
  (LODO split + generator file selection) — don't override per-machine.
