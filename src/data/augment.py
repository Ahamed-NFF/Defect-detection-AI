"""
Traditional (non-generative) augmentation pipelines.

This is BASELINE 2 in the experiment table: rotate / flip / crop / colour jitter.
Keep this separate from synthetic (diffusion) augmentation so the ablation is clean.

Owner: Member 1
"""

import torchvision.transforms as T


def train_transforms(img_size=256, traditional_aug=True):
    """Return a torchvision transform pipeline.

    traditional_aug=False  -> only resize + normalise (Baseline 1)
    traditional_aug=True   -> + flips/rotations/jitter (Baseline 2)
    """
    base = [T.Resize((img_size, img_size))]
    if traditional_aug:
        base += [
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.1, 0.1, 0.1),
        ]
    base += [T.ToTensor(),
             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(base)
