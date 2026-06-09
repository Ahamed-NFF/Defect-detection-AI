"""
Generate synthetic defect images from a trained LoRA checkpoint.

Produces N images per category into data/synthetic/<category>/.
These feed the augmented classifier (your method) in the experiment table —
src/data/dataset.py mixes them into the TRAIN split via synthetic_dir.

Usage:
    python -m src.generative.generate --category bottle --n 800 \
        --lora experiments/checkpoints/bottle_lora --out data/synthetic/bottle

Resumable: counts images already in --out and only generates the remainder, so
an evicted job can be re-run. Requires a GPU and diffusers>=0.27.

Owner: Member 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FILENAME_TMPL = "defect_{:05d}.png"


# --------------------------------------------------------------------------- #
# Pure helpers (no torch/diffusers needed — unit-testable)
# --------------------------------------------------------------------------- #
def load_config(path):
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def existing_count(out_dir) -> int:
    out = Path(out_dir)
    if not out.is_dir():
        return 0
    return sum(1 for p in out.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def resolve_prompt(config: dict, category: str, override=None) -> str:
    if override:
        return override
    tmpl = config.get("instance_prompt", "a photo of a sks {category} with a defect")
    return tmpl.format(category=category)


# --------------------------------------------------------------------------- #
# Generation (GPU path — heavy imports live here)
# --------------------------------------------------------------------------- #
def generate(category, n, lora_path, out_dir, config_path="configs/diffusion_lora.yaml",
             steps=30, guidance=7.5, batch_size=4, seed=0, prompt=None):
    """Sample `n` synthetic defect images into out_dir (skips already-present)."""
    # GPU check before the heavy diffusers import so a CPU box fails fast.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("generate needs a CUDA GPU. Run this on the GPU machine.")

    from diffusers import StableDiffusionPipeline

    config = load_config(config_path)
    base_model = config.get("base_model", "runwayml/stable-diffusion-v1-5")
    full_prompt = resolve_prompt(config, category, prompt)

    device = torch.device("cuda")
    dtype = torch.float16

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    already = existing_count(out)
    remaining = max(0, n - already)
    print(f"[{category}] target {n}, have {already}, generating {remaining} "
          f"| base={base_model} lora={lora_path}")
    print(f"  prompt: {full_prompt!r}")
    if remaining == 0:
        return str(out)

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.load_lora_weights(lora_path)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(seed)
    made = 0
    idx = already  # continue numbering after existing files
    while made < remaining:
        bs = min(batch_size, remaining - made)
        result = pipe(
            [full_prompt] * bs,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        for img in result.images:
            img.save(out / FILENAME_TMPL.format(idx))
            idx += 1
            made += 1
        print(f"  {made}/{remaining} generated")

    print(f"[{category}] wrote {made} images -> {out}")
    return str(out)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic defect images from a trained LoRA.")
    p.add_argument("--category", required=True)
    p.add_argument("--n", type=int, required=True, help="target total images in --out")
    p.add_argument("--lora", required=True, help="path to trained LoRA weights dir")
    p.add_argument("--out", required=True, help="output dir (e.g. data/synthetic/bottle)")
    p.add_argument("--config", default="configs/diffusion_lora.yaml")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default=None, help="override the generation prompt")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    generate(args.category, args.n, args.lora, args.out, config_path=args.config,
             steps=args.steps, guidance=args.guidance, batch_size=args.batch_size,
             seed=args.seed, prompt=args.prompt)


if __name__ == "__main__":
    main()
