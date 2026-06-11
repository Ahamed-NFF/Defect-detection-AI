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
from src.eval.metrics import best_threshold, classification_metrics

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
    # Classifier-quality levers (uniform across all experiment rows so only the
    # DATA differs between them). Defaults reflect proper transfer learning.
    freeze_backbone = config.get("freeze_backbone", False)   # fine-tune the whole net
    class_weighted = config.get("class_weighted", True)      # weight loss by inverse freq
    tune_threshold = config.get("tune_threshold", True)      # calibrate operating point on val
    epochs = epochs_override if epochs_override is not None else config.get("epochs", 30)

    synthetic_dir = config.get("synthetic_dir") if use_synthetic else None
    synthetic_n = config.get("synthetic_n") if use_synthetic else None

    print(f"=== {run_name} | category={category} | device={dev} ===")
    print(f"    traditional_aug={traditional_aug} use_synthetic={use_synthetic} "
          f"synthetic_dir={synthetic_dir} synthetic_n={synthetic_n} "
          f"few_shot_n={few_shot_n} epochs={epochs} batch_size={batch_size} lr={lr}")
    print(f"    freeze_backbone={freeze_backbone} class_weighted={class_weighted} "
          f"tune_threshold={tune_threshold}")

    train_loader, val_loader, test_loader = build_loaders(
        category, batch_size=batch_size, few_shot_n=few_shot_n,
        synthetic_dir=synthetic_dir, synthetic_n=synthetic_n, data_root=data_root,
        traditional_aug=traditional_aug, num_workers=num_workers,
    )
    print(f"    train={len(train_loader.dataset)} "
          f"({train_loader.dataset.class_counts()}) | "
          f"val={len(val_loader.dataset)} | test={len(test_loader.dataset)}")

    net = build_classifier(backbone, num_classes=2, freeze_backbone=freeze_backbone).to(dev)

    # Weight the loss by inverse class frequency so the rare defect class drives
    # gradients (handles imbalance without flipping it).
    if class_weighted:
        counts = train_loader.dataset.class_counts()
        n_good, n_defect = max(counts["good"], 1), max(counts["defect"], 1)
        total = n_good + n_defect
        weight = torch.tensor([total / (2 * n_good), total / (2 * n_defect)],
                              dtype=torch.float32, device=dev)
        print(f"    class weights: good={weight[0]:.3f} defect={weight[1]:.3f}")
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{run_name}.pt"

    # Select the best epoch by val AUROC: it's threshold-independent, so it
    # doesn't get stuck at 0 when 0.5 collapses to all-"good".
    best_val_auroc = -1.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(net, train_loader, criterion, optimizer, dev,
                               limit_batches=limit_batches)
        yt, yp, ys = evaluate(net, val_loader, dev, limit_batches=limit_batches)
        val = classification_metrics(yt, yp, ys)
        val_auroc = val["auroc"] if val["auroc"] is not None else val["macro"]["f1"]
        print(f"  epoch {epoch:>2}/{epochs}  train_loss={loss:.4f}  "
              f"val_auroc={val_auroc:.4f}  val_defect_f1={val['per_class']['defect']['f1']:.4f}")
        if val_auroc >= best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(net.state_dict(), ckpt_path)

    # Reload best checkpoint, calibrate the threshold on val, evaluate on test.
    if ckpt_path.exists():
        net.load_state_dict(torch.load(ckpt_path, map_location=dev))

    threshold = 0.5
    if tune_threshold:
        v_true, _, v_score = evaluate(net, val_loader, dev, limit_batches=limit_batches)
        threshold = best_threshold(v_true, v_score)

    t_true, _, t_score = evaluate(net, test_loader, dev, limit_batches=limit_batches)
    t_pred = [1 if s >= threshold else 0 for s in t_score]
    test_metrics = classification_metrics(t_true, t_pred, t_score)
    # keep the naive-0.5 numbers too, for an honest before/after in the report
    metrics_at_0p5 = classification_metrics(
        t_true, [1 if s >= 0.5 else 0 for s in t_score], t_score)

    print(f"  threshold={threshold:.3f}  TEST  macro_f1={test_metrics['macro']['f1']:.4f}  "
          f"defect_f1={test_metrics['per_class']['defect']['f1']:.4f}  "
          f"auroc={test_metrics['auroc']}  (defect_f1@0.5={metrics_at_0p5['per_class']['defect']['f1']:.4f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": run_name,
        "category": category,
        "config": config,
        "device": str(dev),
        "epochs_run": epochs,
        "best_val_auroc": best_val_auroc,
        "threshold": threshold,
        "checkpoint": str(ckpt_path),
        "metrics": test_metrics,
        "metrics_at_0p5": metrics_at_0p5,
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
