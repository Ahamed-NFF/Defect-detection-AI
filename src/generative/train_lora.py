"""
Fine-tune Stable Diffusion with LoRA on real defect images of ONE category.

Approach:
    - Base model: runwayml/stable-diffusion-v1-5 (or SDXL if GPU allows)
    - LoRA via Hugging Face `diffusers` + `peft`
    - Train on the few real defect images for a category, with a text prompt
      like "a photo of a sks bottle with a defect"
    - LoRA keeps the trainable params tiny -> fits short, interruptible GPU slots
      (important: lab GPU is a shared queue, runs must be checkpointable)

Why this over StyleGAN2-ADA: shorter runs (1-3h vs 24-48h), checkpointable,
more current literature (AnomalyDiffusion AAAI'24, blended latent diffusion '24).

Usage:
    # LODO fold generator (current protocol): trains on EXACTLY the real
    # defect images the classifier trains on for that fold (held-out type and
    # val slice excluded -- src.data.dataset.lodo_train_defect_paths is the
    # single source of truth), checkpoints to
    # experiments/checkpoints/<cat>_holdout_<type>_lora/, and writes a
    # manifest.json recording exactly what it saw:
    python -m src.generative.train_lora --category bottle \
        --holdout-defect-type broken_large --config configs/diffusion_lora.yaml

    # All-defects generator (DEPRECATED for LODO -- the generator sees every
    # defect type, so its synthetic images are invalid for any LODO "ours"
    # row; kept only for the old pooled-split experiments):
    python -m src.generative.train_lora --category bottle \
        --config configs/diffusion_lora.yaml

Checkpointing / resume (shared-queue safe):
    every `checkpoint_every` optimiser steps a training_state.pt (LoRA weights +
    optimiser + step) is written to <output_dir>/checkpoint-<step>/. On restart
    the latest one is loaded automatically (disable with --no-resume). The final
    diffusers-format LoRA weights are written to <output_dir>/ for generate.py.

Requires a GPU and: diffusers>=0.27, peft>=0.10, transformers, accelerate.

Owner: Member 2 (Generative Lead)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import yaml

# NOTE: torch / diffusers / peft are imported lazily inside train() so this
# module (and its pure helpers below) can be imported/tested without them.
# src.data.dataset is likewise imported lazily (it pulls in torchvision).

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MANIFEST_FILENAME = "manifest.json"
FINAL_WEIGHTS_FILENAME = "pytorch_lora_weights.safetensors"  # written by save_final()


# --------------------------------------------------------------------------- #
# Pure helpers (no torch/diffusers needed — unit-testable)
# --------------------------------------------------------------------------- #
def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_config(config: dict, category: str) -> dict:
    """Fill ``{category}`` placeholders in string config values."""
    out = dict(config)
    for key, val in out.items():
        if isinstance(val, str) and "{category}" in val:
            out[key] = val.format(category=category)
    return out


def find_defect_images(data_root, category) -> list[Path]:
    """Real defect images for a category: data/raw/<cat>/test/<defect>/* (not good).

    DEPRECATED for LODO: this returns EVERY defect type, including whatever a
    fold holds out, so a generator trained on it leaks the held-out type.
    Used only by the all-defects path (old pooled-split experiments). LODO
    fold training uses src.data.dataset.lodo_train_defect_paths instead.
    """
    test_dir = Path(data_root) / category / "test"
    if not test_dir.is_dir():
        return []
    images: list[Path] = []
    for sub in sorted(test_dir.iterdir()):
        if sub.is_dir() and sub.name != "good":
            images += sorted(
                p for p in sub.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
    return images


def latest_checkpoint(output_dir):
    """Return (path, step) of the highest-numbered checkpoint-<step> dir, else (None, 0)."""
    out = Path(output_dir)
    if not out.is_dir():
        return None, 0
    best, best_step = None, -1
    for d in out.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if d.is_dir() and m and int(m.group(1)) > best_step:
            best, best_step = d, int(m.group(1))
    return (best, best_step) if best is not None else (None, 0)


# --------------------------------------------------------------------------- #
# LODO fold manifest (pure/CPU -- no torch needed, unit-testable)
# --------------------------------------------------------------------------- #
def _sha256_of_paths(paths) -> str:
    """Stable digest of a (pre-sorted) list of paths, for manifest evidence."""
    return hashlib.sha256("\n".join(str(p) for p in paths).encode()).hexdigest()


def build_fold_manifest(category, holdout_defect_type, cfg, data_root="data/raw",
                        split_seed=None, torch_seed=42) -> dict:
    """Manifest for one fold's generator: exactly what it trains on, and proof.

    The training file list comes from src.data.dataset.lodo_train_defect_paths
    (the classifier's own LODO partition code -- the single source of truth).
    As independent evidence, the classifier's train-split defect files are
    ALSO collected through the classifier's real entry point
    (build_loaders_lodo) and their count/hash + an explicit equality result
    are recorded, so the manifest itself documents the generator/classifier
    file-set equality without cross-referencing other artifacts. Raises if
    they somehow differ, or if the held-out type shows up in the file set.

    ``cfg`` is the (already category-formatted) diffusion config dict; the
    hyperparameters recorded here are informational for the paper trail.
    """
    from src.data.dataset import (
        DEFECT,
        SPLIT_SEED,
        LODO_PROTOCOL_VERSION,
        build_loaders_lodo,
        defect_types,
        lodo_train_defect_paths,
    )

    if split_seed is None:
        split_seed = SPLIT_SEED

    train_files = lodo_train_defect_paths(category, holdout_defect_type,
                                          data_root=data_root, seed=split_seed)
    if not train_files:
        raise SystemExit(
            f"no LODO train defect images for {category!r} holdout={holdout_defect_type!r} "
            f"under {data_root!r} -- has the category been downloaded?"
        )
    trained_on_types = sorted({p.parent.name for p in train_files})
    all_types = defect_types(category, data_root=data_root)
    excluded_types = sorted(set(all_types) - set(trained_on_types))

    if holdout_defect_type in trained_on_types:
        raise RuntimeError(
            f"LODO invariant violated: held-out type {holdout_defect_type!r} present "
            f"in the generator training set for {category!r}"
        )

    # Independent pass through the classifier's actual entry point.
    train_loader, _, _ = build_loaders_lodo(category, holdout_defect_type,
                                            data_root=data_root, seed=split_seed)
    clf_files = sorted(p for p, label in train_loader.dataset.samples if label == DEFECT)
    file_set_equal = [str(p) for p in clf_files] == [str(p) for p in train_files]
    if not file_set_equal:
        raise RuntimeError(
            f"generator/classifier train-defect file sets differ for {category!r} "
            f"holdout={holdout_defect_type!r} (generator={len(train_files)}, "
            f"classifier={len(clf_files)}) -- the source of truth has drifted"
        )

    return {
        "protocol_version": LODO_PROTOCOL_VERSION,
        "category": category,
        "held_out_type": holdout_defect_type,
        "trained_on_types": trained_on_types,
        "excluded_types": excluded_types,
        "train_files": [str(p) for p in train_files],
        "n_train_files": len(train_files),
        "train_files_sha256": _sha256_of_paths(train_files),
        "classifier_train_defect_count": len(clf_files),
        "classifier_train_defect_sha256": _sha256_of_paths(clf_files),
        "file_set_equal": file_set_equal,
        "split_seed": split_seed,
        "torch_seed": torch_seed,
        "base_model": cfg.get("base_model", "runwayml/stable-diffusion-v1-5"),
        "lora_rank": int(cfg.get("lora_rank", 16)),
        "resolution": int(cfg.get("resolution", 512)),
        "max_train_steps": int(cfg.get("max_train_steps", 1500)),
        "instance_prompt": cfg.get("instance_prompt",
                                   f"a photo of a sks {category} with a defect"),
        "status": "training",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def fold_lora_complete(category, holdout_defect_type,
                       checkpoints_root="experiments/checkpoints") -> bool:
    """True iff a fold's generator finished AND its manifest matches expectation.

    Used by the sweep's resume logic (shared GPU queue): requires final
    diffusers weights + a manifest whose category/held-out type/protocol all
    agree and whose status is "complete". A stale/mismatched/mid-training
    fold returns False and gets (re)trained.
    """
    from src.data.dataset import LODO_PROTOCOL_VERSION, lodo_lora_checkpoint_dir

    out = lodo_lora_checkpoint_dir(category, holdout_defect_type,
                                   checkpoints_root=checkpoints_root)
    manifest_path = out / MANIFEST_FILENAME
    if not (manifest_path.is_file() and (out / FINAL_WEIGHTS_FILENAME).is_file()):
        return False
    try:
        m = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        m.get("protocol_version") == LODO_PROTOCOL_VERSION
        and m.get("category") == category
        and m.get("held_out_type") == holdout_defect_type
        and holdout_defect_type not in m.get("trained_on_types", [])
        and m.get("status") == "complete"
    )


# --------------------------------------------------------------------------- #
# Training (GPU path — heavy imports live here)
# --------------------------------------------------------------------------- #
def train(category, config, data_root="data/raw", max_steps_override=None,
          resume=True, num_workers=2, seed=42, holdout_defect_type=None,
          split_seed=None):
    """Run LoRA fine-tuning. Checkpoints every N steps to survive queue eviction.

    holdout_defect_type: if set, trains a LODO fold generator: images come
    from src.data.dataset.lodo_train_defect_paths (exactly the classifier's
    train-split defects for that fold), output goes to
    lodo_lora_checkpoint_dir(), and a manifest.json documenting the training
    set is written there (status "training" -> "complete"). If None, the
    DEPRECATED-for-LODO all-defects path runs (every test/<type> image,
    config-template output_dir) for the old pooled-split experiments.
    split_seed: LODO split seed for the fold's file selection; defaults to
    the dataset module's SPLIT_SEED so generator and classifier stay in
    lockstep by default.
    """
    # Check the GPU first so a CPU box fails fast with a clear message instead of
    # paying the (slow) cost of importing diffusers/transformers.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "train_lora needs a CUDA GPU (LoRA SD fine-tuning is impractical on CPU). "
            "Run this on the GPU machine."
        )

    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    from PIL import Image

    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    cfg = format_config(config, category)
    base_model = cfg.get("base_model", "runwayml/stable-diffusion-v1-5")
    resolution = int(cfg.get("resolution", 512))
    batch_size = int(cfg.get("train_batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    lr = float(cfg.get("learning_rate", 1e-4))
    rank = int(cfg.get("lora_rank", 16))
    max_steps = int(max_steps_override or cfg.get("max_train_steps", 1500))
    ckpt_every = int(cfg.get("checkpoint_every", 250))
    prompt = cfg.get("instance_prompt", f"a photo of a sks {category} with a defect")
    fp16 = cfg.get("mixed_precision", "fp16") == "fp16"

    device = torch.device("cuda")
    weight_dtype = torch.float16 if fp16 else torch.float32
    torch.manual_seed(seed)

    manifest = None
    if holdout_defect_type is not None:
        from src.data.dataset import lodo_lora_checkpoint_dir

        manifest = build_fold_manifest(category, holdout_defect_type, cfg,
                                       data_root=data_root, split_seed=split_seed,
                                       torch_seed=seed)
        output_dir = Path(lodo_lora_checkpoint_dir(category, holdout_defect_type))
        images = [Path(f) for f in manifest["train_files"]]
        print(f"[{category}] LODO fold: holdout={holdout_defect_type} | training on "
              f"{len(images)} real defect images of types {manifest['trained_on_types']} "
              f"(= the classifier's train defects; val slice + held-out type excluded)")
    else:
        output_dir = Path(cfg.get("output_dir", f"experiments/checkpoints/{category}_lora"))
        images = find_defect_images(data_root, category)
        print(f"[{category}] all-defects generator (DEPRECATED for LODO -- sees every "
              f"defect type; valid only for the old pooled-split experiments)")

    if not images:
        raise SystemExit(
            f"no defect images under {Path(data_root)/category/'test'} — "
            f"run `python -m src.data.download --categories {category}` first."
        )
    print(f"[{category}] training LoRA on {len(images)} real defect images "
          f"| base={base_model} res={resolution} rank={rank} steps={max_steps}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
        print(f"  wrote {output_dir / MANIFEST_FILENAME} (status=training)")

    # --- dataset: image -> [-1, 1] tensor; the prompt is fixed for all images ---
    tform = T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    class DefectImages(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            with Image.open(self.paths[i]) as im:
                return tform(im.convert("RGB"))

    loader = DataLoader(DefectImages(images), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True)

    # --- load frozen SD components ---
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # --- attach LoRA adapters to the UNet attention projections ---
    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    # keep trainable LoRA params in fp32 for stable optimisation
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    # precompute the (fixed) text embedding once
    ids = tokenizer(prompt, padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    with torch.no_grad():
        text_embed = text_encoder(ids.to(device))[0]  # (1, seq, dim)

    # --- resume from the latest checkpoint if present ---
    global_step = 0
    if resume:
        ckpt_dir, step = latest_checkpoint(output_dir)
        state_file = ckpt_dir / "training_state.pt" if ckpt_dir else None
        if state_file and state_file.exists():
            state = torch.load(state_file, map_location=device)
            set_peft_model_state_dict(unet, state["lora"])
            optimizer.load_state_dict(state["optimizer"])
            global_step = state["step"]
            print(f"  resumed from {ckpt_dir} at step {global_step}")

    def save_checkpoint(step):
        ckpt = output_dir / f"checkpoint-{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"lora": get_peft_model_state_dict(unet),
             "optimizer": optimizer.state_dict(), "step": step},
            ckpt / "training_state.pt",
        )
        print(f"  checkpoint @ step {step} -> {ckpt}")

    def save_final():
        lora_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(output_dir), unet_lora_layers=lora_sd, safe_serialization=True,
        )
        print(f"  saved final LoRA weights -> {output_dir} (for generate.py)")
        if manifest is not None:
            manifest["status"] = "complete"
            manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
            print(f"  updated {output_dir / MANIFEST_FILENAME} (status=complete)")

    # --- training loop ---
    unet.train()
    micro = 0
    optimizer.zero_grad()
    while global_step < max_steps:
        for pixel_values in loader:
            pixel_values = pixel_values.to(device, dtype=weight_dtype)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            ehs = text_embed.expand(bsz, -1, -1)

            with torch.autocast("cuda", enabled=fp16):
                model_pred = unet(noisy, timesteps, ehs).sample
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = F.mse_loss(model_pred.float(), target.float()) / grad_accum

            scaler.scale(loss).backward()
            micro += 1
            if micro % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 50 == 0:
                    print(f"  step {global_step}/{max_steps}  loss={loss.item()*grad_accum:.4f}")
                if global_step % ckpt_every == 0:
                    save_checkpoint(global_step)
                if global_step >= max_steps:
                    break

    save_checkpoint(global_step)
    save_final()
    print(f"[{category}] done at step {global_step}.")
    return str(output_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tune Stable Diffusion on one category's defects.")
    p.add_argument("--category", required=True)
    p.add_argument("--config", default="configs/diffusion_lora.yaml")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--max-steps", type=int, default=None, help="override config max_train_steps")
    p.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-defect-type", default=None,
                   help="LODO fold: train the generator on exactly the classifier's "
                        "train defects for this fold (held-out type + val slice "
                        "excluded); writes checkpoint + manifest to "
                        "experiments/checkpoints/<cat>_holdout_<type>_lora/")
    p.add_argument("--split-seed", type=int, default=None,
                   help="LODO split seed for fold file selection (default: the "
                        "dataset module's SPLIT_SEED, keeping generator and "
                        "classifier in lockstep)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    train(args.category, config, data_root=args.data_root,
          max_steps_override=args.max_steps, resume=not args.no_resume,
          num_workers=args.num_workers, seed=args.seed,
          holdout_defect_type=args.holdout_defect_type, split_seed=args.split_seed)


if __name__ == "__main__":
    main()
