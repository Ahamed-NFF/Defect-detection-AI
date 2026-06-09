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
import re
from pathlib import Path

import yaml

# NOTE: torch / diffusers / peft are imported lazily inside train() so this
# module (and its pure helpers below) can be imported/tested without them.

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
    """Real defect images for a category: data/raw/<cat>/test/<defect>/* (not good)."""
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
# Training (GPU path — heavy imports live here)
# --------------------------------------------------------------------------- #
def train(category, config, data_root="data/raw", max_steps_override=None,
          resume=True, num_workers=2, seed=42):
    """Run LoRA fine-tuning. Checkpoints every N steps to survive queue eviction."""
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
    output_dir = Path(cfg.get("output_dir", f"experiments/checkpoints/{category}_lora"))
    fp16 = cfg.get("mixed_precision", "fp16") == "fp16"

    device = torch.device("cuda")
    weight_dtype = torch.float16 if fp16 else torch.float32
    torch.manual_seed(seed)

    images = find_defect_images(data_root, category)
    if not images:
        raise SystemExit(
            f"no defect images under {Path(data_root)/category/'test'} — "
            f"run `python -m src.data.download --categories {category}` first."
        )
    print(f"[{category}] training LoRA on {len(images)} real defect images "
          f"| base={base_model} res={resolution} rank={rank} steps={max_steps}")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    train(args.category, config, data_root=args.data_root,
          max_steps_override=args.max_steps, resume=not args.no_resume,
          num_workers=args.num_workers, seed=args.seed)


if __name__ == "__main__":
    main()
