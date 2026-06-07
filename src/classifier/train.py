"""
Train the defect classifier under one of the experiment configurations.

The SAME script runs every row of the experiment table; the config decides which:
    - Baseline 1 : traditional_aug=False, synthetic=False
    - Baseline 2 : traditional_aug=True,  synthetic=False
    - Our method : traditional_aug=True,  synthetic=True
    - Few-shot   : few_shot_n in {10,30,50}, synthetic on/off

Logs metrics to experiments/results/<run_name>.json so the eval module can
assemble the comparison table automatically.

Usage:
    python -m src.classifier.train --config configs/exp_bottle_ours.yaml

Owner: Member 3
"""


def train(config):
    raise NotImplementedError("Member 3: standard PyTorch train loop + metric logging")
