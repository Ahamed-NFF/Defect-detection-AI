"""
Generate synthetic defect images from a trained LoRA checkpoint.

Produces N images per category into data/synthetic/<category>/.
These feed the augmented classifier (your method) in the experiment table.

Usage:
    python -m src.generative.generate --category bottle --n 800 \
        --lora experiments/checkpoints/bottle_lora --out data/synthetic/bottle

Owner: Member 2
"""


def generate(category, n, lora_path, out_dir):
    raise NotImplementedError("Member 2: load pipeline + LoRA, sample n images, save")
