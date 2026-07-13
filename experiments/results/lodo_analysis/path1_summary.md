# Path-1 exploratory: candidate predictors of delta(ours-b2) defect-F1

n=26 folds; 11 candidates tested -> at alpha=0.05 expect ~0.6 false positives by chance. Judged on strength AND a-priori mechanism, not p-value alone.

## B. Candidates ranked (plausible-mechanism first, then |Spearman|)
| candidate | Spearman rho | p | Pearson r | p | mechanism? | screw-consistent? |
|---|---|---|---|---|---|---|
| edge_gap | +0.511 | 0.0077 | +0.495 | 0.0102 | plausible | yes |
| edge_ratio | +0.487 | 0.0117 | +0.455 | 0.0194 | plausible | NO |
| std_gap | +0.425 | 0.0302 | +0.204 | 0.3180 | plausible | NO |
| b2_auroc | -0.385 | 0.0524 | -0.432 | 0.0277 | plausible | yes |
| b2_defF1 | -0.315 | 0.1166 | -0.386 | 0.0514 | plausible | yes |
| fid | -0.105 | 0.6112 | -0.162 | 0.4279 | plausible | NO |
| n_real_train | -0.082 | 0.6911 | -0.190 | 0.3520 | plausible | yes |
| n_train_types | -0.013 | 0.9480 | -0.121 | 0.5554 | plausible | yes |
| texture | +0.170 | 0.4073 | +0.031 | 0.8797 | weak/none -> likely spurious if it correlates | yes |
| mean_gap | -0.085 | 0.6799 | +0.005 | 0.9792 | weak/none -> likely spurious if it correlates | NO |
| n_test_defect | -0.076 | 0.7118 | +0.074 | 0.7197 | weak/none -> likely spurious if it correlates | yes |

Mechanism notes:

- **edge_gap**: Syn high-frequency content bias vs real (syn - real Sobel density): synthetic too smooth (<0) or too busy (>0) teaches the classifier wrong texture statistics. Clear mechanism, esp. for high-frequency texture categories.
- **edge_ratio**: Same mechanism as edge_gap, scale-relative.
- **std_gap**: Syn global-contrast bias vs real (syn - real).
- **b2_auroc**: Baseline discrimination difficulty; related to headroom but not the same term as in the delta, so less mechanically coupled.
- **b2_defF1**: Headroom: delta is bounded above by 1-b2_defF1; saturated folds cannot gain. STRONG prior mechanism, but PARTLY MECHANICAL COUPLING: b2 enters delta with a negative sign, so noise in b2 alone produces a negative correlation (regression to the mean). Treat direction as expected by construction; the informative part is the size.
- **fid**: Inception-feature fidelity (already tested: rho=-0.10, p=0.61).
- **n_real_train**: More real defect data -> smaller marginal value of synthetic. Plausible either direction; category-confounded.
- **n_train_types**: Generator/classifier saw more defect-type diversity -> synthetic defects may generalize better to an unseen type. Plausible but confounded with category (only 4 distinct values).
- **texture**: Texture-vs-object control. Known insufficient a priori: carpet and tile are both textures with opposite outcomes.
- **mean_gap**: Syn brightness bias vs real. Weak mechanism on its own (classifier normalizes); would need to be large to matter.
- **n_test_defect**: Test-set size affects measurement noise, not the true benefit. NO benefit mechanism -> any correlation likely spurious.

## C. Carpet vs tile vs screw (category means per candidate)
| candidate | carpet (all-win) | tile (all-lose) | screw (best-FID, loses) | all-26 mean |
|---|---|---|---|---|
| edge_gap | 0.508 | -0.028 | 0.043 | 0.123 |
| edge_ratio | 1.764 | 0.922 | 1.632 | 1.461 |
| std_gap | 0.092 | -0.010 | 0.027 | 0.007 |
| b2_auroc | 0.585 | 0.996 | 0.973 | 0.906 |
| b2_defF1 | 0.549 | 0.835 | 0.863 | 0.774 |
| fid | 134.342 | 219.756 | 102.032 | 149.021 |
| n_real_train | 53.200 | 50.600 | 71.200 | 51.385 |
| n_train_types | 4.000 | 4.000 | 4.000 | 3.462 |
| texture | 1.000 | 1.000 | 0.000 | 0.385 |
| mean_gap | -0.063 | -0.008 | -0.078 | -0.032 |
| n_test_defect | 17.800 | 16.800 | 23.800 | 19.923 |

## D. Post-hoc checks (crucial)

- **|edge_gap| (the actual 'deviation hurts' test): Spearman rho = +0.409 (p=0.038)** — POSITIVE,
  i.e. larger absolute deviation associates with MORE benefit. The pre-stated
  smoothness-mismatch-harms mechanism is REJECTED in both signed and absolute form;
  the signed edge_gap correlation means 'synthetic busier than real -> more benefit',
  which was not the hypothesized mechanism.
- **Leverage check (leave carpet out, n=21): edge_gap rho = +0.134 (p=0.563);
  b2_auroc rho = +0.127 (p=0.583, sign flips).** Both top candidates collapse:
  their 26-fold correlations are driven by carpet, not by a fold-level relationship.
- Screw check: moot at fold level after the leverage check; at category level both
  top candidates are screw-consistent (screw b2_auroc 0.973 -> predicts no benefit;
  screw edge_gap +0.043, near tile's) but this is 6-category description, not evidence.

## E. Bottom line (facts, not forced)

**No candidate survives as a defensible fold-level predictor.** The honest
conclusion is: strong per-category consistency with no single identified
fold-level predictor. What CAN be said descriptively: carpet — the only
all-win category — is uniquely extreme on two mechanistically sensible axes at
once: (1) by far the weakest baseline (b2 held-out AUROC 0.585, near chance,
vs >=0.9 for every other category — headroom) and (2) by far the largest
synthetic high-frequency surplus (edge_gap +0.508 vs |gap|<=0.09 elsewhere).
These are category-level observations (n=6), not validated predictors, and the
two are confounded with each other on this data.
