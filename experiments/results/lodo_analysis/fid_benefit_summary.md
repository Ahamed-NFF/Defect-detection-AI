# FID -> benefit correlation (LODO, 26 folds) — numbers only

Directional hypothesis: LOWER FID (better synthetic fidelity) -> HIGHER
delta(ours - baseline2) held-out defect-F1, i.e. expected NEGATIVE correlation.

## B. Fold-level correlations (n=26)
- FID vs delta(ours-b2) defF1: Spearman rho = -0.1046 (p=0.6112); Pearson r = -0.1624 (p=0.4279).
  Observed sign: negative -> matches the hypothesis.
- Secondary, FID vs delta(ours-b1) defF1: Spearman rho = +0.1451 (p=0.4795); Pearson r = -0.0594 (p=0.7731).
- Caveat: per-fold FID real-side samples are 31-72 images (clean-fid flags <50
  as noisy), so individual FID points carry small-sample noise; the trend is
  across 26 points.

## C. Category level (n=6, descriptive only)
| category | folds | mean FID | mean delta(ours-b2) |
|---|---|---|---|
| bottle | 3 | 155.2 | +0.000 |
| hazelnut | 4 | 117.6 | +0.053 |
| carpet | 5 | 134.3 | +0.192 |
| metal_nut | 4 | 164.5 | +0.054 |
| screw | 5 | 102.0 | -0.116 |
| tile | 5 | 219.8 | -0.182 |
- Category-level correlation (mean FID vs mean delta): Spearman rho = -0.143 (p=0.787), Pearson r = -0.398 (p=0.435) — n=6, underpowered, descriptive only.
- carpet: mean FID 134.3, mean delta +0.192 | tile: mean FID 219.8, mean delta -0.182.

## D. Threshold-collapse cross-check (folds with delta <= -0.05)
| fold | delta_f1 | ours_auroc | b2_auroc | ours_recall | b2_recall | calibration-dominant |
|---|---|---|---|---|---|---|
| tile/rough | -0.654 | 0.988 | 1.000 | 0.133 | 0.800 | yes |
| screw/thread_side | -0.307 | 0.944 | 0.969 | 0.391 | 0.870 | yes |
| screw/manipulated_front | -0.284 | 0.956 | 0.923 | 0.250 | 0.542 | yes |
| tile/gray_stroke | -0.198 | 0.908 | 0.978 | 0.062 | 0.188 | no |
| metal_nut/color | -0.196 | 0.893 | 0.909 | 0.273 | 0.455 | yes |
| hazelnut/hole | -0.096 | 1.000 | 1.000 | 0.778 | 0.944 | yes |
| metal_nut/scratch | -0.071 | 0.968 | 0.992 | 0.826 | 0.913 | yes |
| tile/glue_strip | -0.059 | 1.000 | 1.000 | 0.889 | 1.000 | yes |

Of the 8 strongly negative folds, 7/8 are calibration-dominant (ours AUROC within 0.05 of b2 -- discrimination intact -- while calibrated recall drops).

Files: fid_benefit.csv, fig_fid_vs_benefit.png (Spearman in caption).
