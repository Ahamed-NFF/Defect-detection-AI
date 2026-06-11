"""
Train the defect classifier under one of the experiment configurations.

The SAME script runs every row of the experiment table; the config decides which:
    - Baseline 1 : traditional_aug=False, use_synthetic=False
    - Baseline 2 : traditional_aug=True,  use_synthetic=False
    - Our method : traditional_aug=True,  use_synthetic=True (synthetic_dir set)
    - Few-shot   : few_shot_n in {10,30,50}, synthetic on/off

Logs metrics to experiments/results/<run_name>.json so the eval module can
assemble the comparison table automatically.

Usage:
    python -m src.classifier.train --config configs/exp_bottle_ours.yaml

Useful overrides (no GPU needed for a sanity check):
    python -m src.classifier.train --config configs/exp_bottle_baseline1.yaml \
        --epochs 1 --limit-batches 2 --device cpu      # fast end-to-end smoke test

Owner: Member 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.classifier.model import build_classifier
from src.data.dataset import build_loaders
from src.eval.metrics import classification_metrics

SEED = 42
CKPT_DIR = Path("experiments/checkpoints")
RESULTS_DIR = Path("experiments/results")


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(choice="auto") -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


@torch.no_grad()
def evaluate(net, loader, device, limit_batches=None):
    """Run the model over a loader; return (y_true, y_pred, y_score=P(defect))."""
    net.eval()
    y_true, y_pred, y_score = [], [], []
    for i, (images, labels) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device)
        logits = net(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(defect)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        y_score.extend(probs.cpu().tolist())
    return y_true, y_pred, y_score


def train_one_epoch(net, loader, criterion, optimizer, device, limit_batches=None):
    net.train()
    running, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for i, (images, labels) in enumerate(pbar):
        if limit_batches is not None and i >= limit_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = net(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
        n += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / max(n, 1)


def train(config, device="auto", limit_batches=None, num_workers=0,
          data_root="data/raw", epochs_override=None):
    """Run one experiment row end to end and write its results JSON."""
    torch.manual_seed(SEED)
    dev = resolve_device(device)

    run_name = config["run_name"]
    category = config["category"]
    backbone = config.get("backbone", "resnet50")
    traditional_aug = config.get("traditional_aug", True)
    use_synthetic = config.get("use_synthetic", False)
    few_shot_n = config.get("few_shot_n")
    batch_size = config.get("batch_size", 32)
    lr = float(config.get("lr", 1e-4))
    epochs = epochs_override if epochs_override is not None else config.get("epochs", 30)

    synthetic_dir = config.get("synthetic_dir") if use_synthetic else None
    synthetic_n = config.get("synthetic_n") if use_synthetic else None

    print(f"=== {run_name} | category={category} | device={dev} ===")
    print(f"    traditional_aug={traditional_aug} use_synthetic={use_synthetic} "
          f"synthetic_dir={synthetic_dir} synthetic_n={synthetic_n} "
          f"few_shot_n={few_shot_n} epochs={epochs} batch_size={batch_size} lr={lr}")

    train_loader, val_loader, test_loader = build_loaders(
        category, batch_size=batch_size, few_shot_n=few_shot_n,
        synthetic_dir=synthetic_dir, synthetic_n=synthetic_n, data_root=data_root,
        traditional_aug=traditional_aug, num_workers=num_workers,
    )
    print(f"    train={len(train_loader.dataset)} "
          f"({train_loader.dataset.class_counts()}) | "
          f"val={len(val_loader.dataset)} | test={len(test_loader.dataset)}")

    net = build_classifier(backbone, num_classes=2, freeze_backbone=True).to(dev)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{run_name}.pt"

    best_val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(net, train_loader, criterion, optimizer, dev,
                               limit_batches=limit_batches)
        yt, yp, ys = evaluate(net, val_loader, dev, limit_batches=limit_batches)
        val = classification_metrics(yt, yp, ys)
        val_f1 = val["macro"]["f1"]
        print(f"  epoch {epoch:>2}/{epochs}  train_loss={loss:.4f}  "
              f"val_macro_f1={val_f1:.4f}  val_defect_f1={val['per_class']['defect']['f1']:.4f}")
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(net.state_dict(), ckpt_path)

    # Final test evaluation using the best checkpoint.
    if ckpt_path.exists():
        net.load_state_dict(torch.load(ckpt_path, map_location=dev))
    yt, yp, ys = evaluate(net, test_loader, dev, limit_batches=limit_batches)
    test_metrics = classification_metrics(yt, yp, ys)

    print(f"  TEST  macro_f1={test_metrics['macro']['f1']:.4f}  "
          f"defect_f1={test_metrics['per_class']['defect']['f1']:.4f}  "
          f"auroc={test_metrics['auroc']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": run_name,
        "category": category,
        "config": config,
        "device": str(dev),
        "epochs_run": epochs,
        "best_val_macro_f1": best_val_f1,
        "checkpoint": str(ckpt_path),
        "metrics": test_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = RESULTS_DIR / f"{run_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out_path}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the defect classifier for one experiment row.")
    p.add_argument("--config", required=True, help="path to an experiment YAML")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="cap batches per train/eval loop (for fast smoke tests)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-root", default="data/raw")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    train(config, device=args.device, limit_batches=args.limit_batches,
          num_workers=args.num_workers, data_root=args.data_root,
          epochs_override=args.epochs)


if __name__ == "__main__":
    main()
