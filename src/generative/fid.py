"""
Compute FID (Frechet Inception Distance) between real and synthetic defects.

FID is the standard quality metric for generative models in this literature.
Report per-category FID in the results table. Lower = synthetic distribution
closer to real. Use `pytorch-fid` or `clean-fid` (clean-fid is more reproducible).

Usage:
    python -m src.generative.fid --real data/raw/bottle/test \
        --fake data/synthetic/bottle

Owner: Member 2
"""


def compute_fid(real_dir, fake_dir):
    raise NotImplementedError("Member 2: wrap clean-fid / pytorch-fid")
