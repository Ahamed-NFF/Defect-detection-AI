# GAN/Diffusion-Augmented Manufacturing Defect Detection

Advanced AI module project. Synthesize realistic defect images with a fine-tuned
generative model to overcome data imbalance, then train a defect classifier that
beats the no-augmentation baseline — wrapped in a working inspector tool with
defect localization and natural-language descriptions.

## Problem

Real manufacturing defects are rare. A line might produce thousands of good units
before a handful of defective ones. A classifier trained on that imbalance learns
to just predict "good". We use **Stable Diffusion + LoRA** to synthesize defect
images, augment training, and demonstrate measurable gains.

## Techniques (maps to the brief — 4 of the required techniques)

| Brief technique        | Our implementation                                   |
|------------------------|------------------------------------------------------|
| Generative AI (Diffusion) | Stable Diffusion + LoRA fine-tuned on defect images |
| Transfer Learning      | ResNet-50 / EfficientNet pretrained on ImageNet      |
| Few-shot Learning      | Ablation: 10 / 30 / 50 real defects + synthetic aug  |
| Prompt Engineering     | LLaVA defect descriptions (CoT + few-shot prompting) |

## Dataset

**MVTec AD** — https://www.mvtec.com/company/research/datasets/mvtec-ad
(CC BY-NC-SA 4.0). Subset: `bottle`, `hazelnut`, `carpet` (2 objects + 1 texture).
Place the downloaded archive in `data/raw/` and run the download/extract script.

## The core experiment

| Run         | Training data                               | Goal                  |
|-------------|---------------------------------------------|-----------------------|
| Baseline 1  | real defects only                           | the number to beat    |
| Baseline 2  | real + traditional aug (flip/rotate/crop)   | fair comparison       |
| **Ours**    | real + traditional + diffusion-synthesized  | target +5–10% F1      |
| Few-shot A  | 10 real defects (+/- synthetic)             | GAN helps most here   |
| Few-shot B  | 50 real defects (+/- synthetic)             | benefit shrinks       |

Metrics: per-class Precision/Recall/F1, AUROC (classifier) + FID (generative quality).

## Repo layout

```
src/
  data/        download, dataset, traditional augmentation   [Member 1]
  generative/  SD+LoRA training, generation, FID             [Member 2]
  classifier/  transfer-learning model + train loop          [Member 3]
  explain/     Grad-CAM localization                          [Member 3]
  vlm/         LLaVA descriptions + prompt engineering        [Member 4]
  eval/        metrics + results-table assembly               [Member 3/4]
backend/api/   FastAPI inference service                      [Member 4]
frontend/      Next.js demo UI                                [Member 4]
configs/       YAML configs (one per experiment row)
experiments/   checkpoints + results JSON (gitignored)
notebooks/     exploration + figure generation
reports/       report draft + figures
docs/          setup notes (incl. Windows/PowerShell)
```

## Quick start

```bash
# 1. Environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   mac/linux: source .venv/bin/activate
pip install -r requirements.txt
# install matching CUDA torch build — see docs/SETUP_WINDOWS.md

# 2. Data
#    download MVTec, drop archive in data/raw/, then:
python -m src.data.download --categories bottle hazelnut carpet --out data/raw

# 3. Baselines (gives the numbers to beat)
python -m src.classifier.train --config configs/exp_bottle_baseline1.yaml
python -m src.classifier.train --config configs/exp_bottle_baseline2.yaml

# 4. Generative augmentation
python -m src.generative.train_lora --category bottle --config configs/diffusion_lora.yaml
python -m src.generative.generate   --category bottle --n 800 \
    --lora experiments/checkpoints/bottle_lora --out data/synthetic/bottle
python -m src.generative.fid        --real data/raw/bottle/test --fake data/synthetic/bottle

# 5. Our method + comparison
python -m src.classifier.train --config configs/exp_bottle_ours.yaml

# 6. Demo (two terminals)
uvicorn backend.api.main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

## Team roles

- **Member 1 — Data & Baselines:** dataset prep, loaders, traditional aug, baseline runs
- **Member 2 — Generative:** SD+LoRA training, synthetic generation, FID
- **Member 3 — Classifier & Eval:** transfer-learning classifier, ablations, Grad-CAM, metrics
- **Member 4 — Product & Integration:** FastAPI + Next.js UI, LLaVA integration, demo video

## Important constraints

- **Shared GPU queue:** generative runs are LoRA-based and checkpoint every 250 steps
  (`configs/diffusion_lora.yaml`) so an evicted job resumes instead of restarting.
- **LLaVA is time-boxed:** if not integrated by end of Week 4, ship with 3 techniques.
- **Commits:** keep feature / bugfix / experiment commits separate (no squashing).

## Deliverables (per the brief)

1. Final report (≤20 pages excl. refs/appendices)
2. Demonstrable output — this functional tool + the results study
3. This documented, reproducible code
4. ≤5-minute video presentation

## Key references

- Bergmann et al., *MVTec AD* (CVPR 2019)
- Karras et al., *Training GANs with Limited Data* (StyleGAN2-ADA, NeurIPS 2020)
- *AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion* (AAAI 2024)
- Hu et al., *LoRA: Low-Rank Adaptation* (2021)
- Roth et al., *PatchCore* (CVPR 2022) — modern anomaly-detection baseline
