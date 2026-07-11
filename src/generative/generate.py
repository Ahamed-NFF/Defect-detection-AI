"""
Generate synthetic defect images from a trained LoRA checkpoint.

LODO fold mode (current protocol): pass --holdout-defect-type and the fold's
checkpoint dir + output dir are derived from the shared path helpers in
src.data.dataset (experiments/checkpoints/<cat>_holdout_<type>_lora/ ->
data/synthetic_lodo/<cat>/holdout_<type>/), and the checkpoint's manifest.json
is verified first: generation REFUSES to run if the manifest is missing, is
for a different category/held-out type, or lists the requested held-out type
among its training types — belt-and-suspenders against pointing a fold's
classifier at the wrong generator.

    python -m src.generative.generate --category bottle \
        --holdout-defect-type broken_large --n 800

All-defects mode (DEPRECATED for LODO — generator saw every defect type; kept
only for the old pooled-split experiments):

    python -m src.generative.generate --category bottle --n 800 \
        --lora experiments/checkpoints/bottle_lora --out data/synthetic/bottle

Resumable: counts images already in the output dir and only generates the
remainder, so an evicted job can be re-run. Requires a GPU and diffusers>=0.27.

Owner: Member 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FILENAME_TMPL = "defect_{:05d}.png"
MANIFEST_FILENAME = "manifest.json"  # written by train_lora's LODO fold path


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


def check_fold_manifest(lora_dir, category, holdout_defect_type) -> dict:
    """Verify a LoRA checkpoint's manifest matches the requested LODO fold.

    Refuses (SystemExit) if the manifest is missing/unreadable, belongs to a
    different category or held-out type, lists the requested held-out type
    among its training types, or was written under an unknown protocol
    version. Returns the parsed manifest on success. CPU-only, so a
    mis-wired fold fails fast anywhere — no GPU needed to hit the guard.
    """
    from src.data.dataset import LODO_PROTOCOL_VERSION  # lazy: pulls in torchvision

    manifest_path = Path(lora_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise SystemExit(
            f"REFUSING to generate: no {MANIFEST_FILENAME} in {lora_dir} — this "
            f"checkpoint was not trained by the LODO fold path (or training never "
            f"started). Train it with:\n"
            f"  python -m src.generative.train_lora --category {category} "
            f"--holdout-defect-type {holdout_defect_type}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"REFUSING to generate: unreadable {manifest_path}: {exc}")

    problems = []
    if manifest.get("protocol_version") != LODO_PROTOCOL_VERSION:
        problems.append(f"protocol_version={manifest.get('protocol_version')!r} "
                        f"(expected {LODO_PROTOCOL_VERSION!r})")
    if manifest.get("category") != category:
        problems.append(f"category={manifest.get('category')!r} (expected {category!r})")
    if manifest.get("held_out_type") != holdout_defect_type:
        problems.append(f"held_out_type={manifest.get('held_out_type')!r} "
                        f"(expected {holdout_defect_type!r})")
    if holdout_defect_type in manifest.get("trained_on_types", []):
        problems.append(f"held-out type {holdout_defect_type!r} appears in "
                        f"trained_on_types — this generator SAW the held-out type")
    if manifest.get("status") != "complete":
        problems.append(f"status={manifest.get('status')!r} (training unfinished?)")
    if problems:
        raise SystemExit(
            f"REFUSING to generate from {lora_dir} for fold "
            f"{category}/holdout={holdout_defect_type}:\n  - "
            + "\n  - ".join(problems)
        )
    return manifest


# --------------------------------------------------------------------------- #
# Generation (GPU path — heavy imports live here)
# --------------------------------------------------------------------------- #
def generate(category, n, lora_path, out_dir, config_path="configs/diffusion_lora.yaml",
             steps=30, guidance=7.5, batch_size=4, seed=0, prompt=None,
             holdout_defect_type=None):
    """Sample `n` synthetic defect images into out_dir (skips already-present).

    holdout_defect_type: if set, verifies lora_path's manifest matches this
    LODO fold before anything else (see check_fold_manifest) — a wrong or
    leaky checkpoint refuses to generate.
    """
    # Fold guard first (CPU-only) so a mis-wired fold fails fast even off-GPU.
    if holdout_defect_type is not None:
        manifest = check_fold_manifest(lora_path, category, holdout_defect_type)
        print(f"[{category}] fold manifest OK: holdout={holdout_defect_type}, "
              f"generator trained on {manifest['trained_on_types']} "
              f"({manifest['n_train_files']} files, protocol={manifest['protocol_version']})")

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
    p.add_argument("--n", type=int, required=True, help="target total images in the output dir")
    p.add_argument("--lora", default=None,
                   help="path to trained LoRA weights dir (required unless "
                        "--holdout-defect-type derives it)")
    p.add_argument("--out", default=None,
                   help="output dir (required unless --holdout-defect-type derives it)")
    p.add_argument("--config", default="configs/diffusion_lora.yaml")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default=None, help="override the generation prompt")
    p.add_argument("--holdout-defect-type", default=None,
                   help="LODO fold: derive --lora/--out from the shared path helpers "
                        "(explicit --lora/--out still win) and verify the checkpoint's "
                        "manifest matches this fold before generating")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    lora, out = args.lora, args.out
    if args.holdout_defect_type:
        from src.data.dataset import lodo_lora_checkpoint_dir, lodo_synthetic_dir

        lora = lora or str(lodo_lora_checkpoint_dir(args.category, args.holdout_defect_type))
        out = out or str(lodo_synthetic_dir(args.category, args.holdout_defect_type))
    if not lora or not out:
        raise SystemExit("--lora and --out are required unless --holdout-defect-type is given")
    generate(args.category, args.n, lora, out, config_path=args.config,
             steps=args.steps, guidance=args.guidance, batch_size=args.batch_size,
             seed=args.seed, prompt=args.prompt,
             holdout_defect_type=args.holdout_defect_type)


if __name__ == "__main__":
    main()
