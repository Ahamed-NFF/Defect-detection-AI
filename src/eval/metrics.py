"""
Classification metrics + results-table builder.

Per-class precision/recall/F1 matters more than overall accuracy here, because
the defect class is the minority and the whole point. Also compute AUROC.

assemble_table() reads every classifier-run experiments/results/*.json (any
file with a "metrics" key) and emits the comparison table (markdown + LaTeX).
assemble_fid_table() reads every experiments/results/fid_<category>.json and
emits the FID table. Both discover categories from whatever result files
exist, so adding a category is just a matter of dropping its JSON there —
nothing here hardcodes bottle/hazelnut/carpet/etc.

Usage:
    python -m src.eval.metrics            # assemble comparison + FID tables

Owner: Member 3 (metrics), Member 4 (table -> report figures)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

LABEL_NAMES = {0: "good", 1: "defect"}


def best_threshold(y_true, y_score) -> float:
    """Threshold on P(defect) that maximises the defect-class F1 on this set.

    Used to calibrate the operating point on the validation set instead of the
    naive 0.5 — important here because the classes are imbalanced and AUROC
    shows the scores separate well even when 0.5 collapses to all-"good".
    Returns 0.5 if y_true has only one class.
    """
    y_true = list(y_true)
    y_score = list(y_score)
    if len(set(y_true)) < 2:
        return 0.5

    # Candidates = midpoints between consecutive observed scores (plus the two
    # edges). Midpoints generalise better than landing exactly on a score.
    uniq = sorted(set(y_score))
    cands = [uniq[0] - 1e-6]
    cands += [(a + b) / 2 for a, b in zip(uniq, uniq[1:])]
    cands += [uniq[-1] + 1e-6]

    def f1_at(t):
        tp = sum(1 for yt, s in zip(y_true, y_score) if s >= t and yt == 1)
        fp = sum(1 for yt, s in zip(y_true, y_score) if s >= t and yt == 0)
        fn = sum(1 for yt, s in zip(y_true, y_score) if s < t and yt == 1)
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom > 0 else 0.0

    scored = [(t, f1_at(t)) for t in cands]
    best_f1 = max(f1 for _, f1 in scored)
    # When the data is perfectly separable many thresholds tie at best_f1; pick
    # the MEDIAN of the winners (most central -> best generalisation) instead of
    # an extreme one that overfits a tiny validation set.
    winners = sorted(t for t, f1 in scored if f1 >= best_f1 - 1e-9)
    return float(winners[len(winners) // 2])


def classification_metrics(y_true, y_pred, y_score=None):
    """Return a metrics dict for binary good(0)/defect(1) predictions.

    Args:
        y_true: iterable of ground-truth labels in {0, 1}.
        y_pred: iterable of predicted labels in {0, 1}.
        y_score: optional iterable of P(defect) scores for AUROC. If omitted
            (or only one class is present in y_true) AUROC is reported as None.

    Returns:
        dict with accuracy; per-class precision/recall/f1/support for good and
        defect; macro precision/recall/f1; and auroc.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    labels = [0, 1]
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    auroc = None
    if y_score is not None and len(set(y_true)) == 2:
        try:
            auroc = float(roc_auc_score(y_true, list(y_score)))
        except ValueError:
            auroc = None

    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[LABEL_NAMES[lbl]] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "auroc": auroc,
    }


# --------------------------------------------------------------------------- #
# Results-table assembly
# --------------------------------------------------------------------------- #
def _flatten_run(run: dict) -> dict:
    """Pull the comparison-table columns out of one results JSON."""
    metrics = run.get("metrics", {})
    defect = metrics.get("per_class", {}).get("defect", {})
    macro = metrics.get("macro", {})
    return {
        "run_name": run.get("run_name", "?"),
        "category": run.get("category", "?"),
        "few_shot_n": run.get("config", {}).get("few_shot_n"),
        "synthetic": run.get("config", {}).get("use_synthetic", False),
        "accuracy": metrics.get("accuracy"),
        "defect_precision": defect.get("precision"),
        "defect_recall": defect.get("recall"),
        "defect_f1": defect.get("f1"),
        "macro_f1": macro.get("f1"),
        "auroc": metrics.get("auroc"),
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def assemble_table(results_dir="experiments/results", out_dir=None):
    """Collect all classifier-run JSONs into one comparison table (markdown + latex).

    Reads every ``*.json`` under results_dir that has a "metrics" key (i.e. a
    classifier run, not an FID result — see assemble_fid_table for those),
    extracts the key metrics, and writes ``comparison.md`` and ``comparison.tex``
    next to them (or to out_dir). Categories are whatever appears in the run
    JSONs — nothing here is hardcoded, so new categories show up automatically.
    Returns the list of per-run row dicts.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(results_dir.glob("*.json"))
    if not files:
        print(f"no results JSON found in {results_dir} — run some experiments first.")
        return []

    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping {f.name}: {exc}")
            continue
        if "metrics" not in data:
            continue  # not a classifier run (e.g. an fid_<category>.json)
        rows.append(_flatten_run(data))

    if not rows:
        print(f"no classifier-run JSON found in {results_dir} — run some experiments first.")
        return []

    # Sort for a stable, readable table: by category then run name.
    rows.sort(key=lambda r: (str(r["category"]), str(r["run_name"])))

    cols = [
        ("run_name", "Run"), ("category", "Cat"), ("few_shot_n", "FewShot"),
        ("synthetic", "Synth"), ("defect_precision", "Def-P"),
        ("defect_recall", "Def-R"), ("defect_f1", "Def-F1"),
        ("macro_f1", "Macro-F1"), ("auroc", "AUROC"), ("accuracy", "Acc"),
    ]

    # Markdown
    header = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(_fmt(r[k]) for k, _ in cols) + " |" for r in rows
    ]
    md = "\n".join([header, sep, *body]) + "\n"
    (out_dir / "comparison.md").write_text(md)

    # LaTeX
    latex_lines = [
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        " & ".join(h for _, h in cols) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        latex_lines.append(" & ".join(_fmt(r[k]) for k, _ in cols) + " \\\\")
    latex_lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "comparison.tex").write_text("\n".join(latex_lines) + "\n")

    print(md)
    print(f"wrote {out_dir/'comparison.md'} and {out_dir/'comparison.tex'} "
          f"({len(rows)} run(s))")
    return rows


def assemble_fid_table(results_dir="experiments/results", out_dir=None):
    """Collect all ``fid_<category>.json`` files into an FID table (markdown + latex).

    Each file is written by ``python -m src.generative.fid ... --out
    experiments/results/fid_<category>.json``. The category is taken from the
    filename (nothing here hardcodes which categories exist), so any category
    with an fid_<category>.json present is picked up automatically. Writes
    ``fid_comparison.md`` and ``fid_comparison.tex`` next to the run tables (or
    to out_dir). Returns the list of per-category row dicts.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(results_dir.glob("fid_*.json"))
    if not files:
        print(f"no fid_<category>.json found in {results_dir} — "
              f"run src.generative.fid with --out first.")
        return []

    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping {f.name}: {exc}")
            continue
        category = f.stem[len("fid_"):]
        rows.append({"category": category, "fid": data.get("fid")})

    rows.sort(key=lambda r: str(r["category"]))

    cols = [("category", "Category"), ("fid", "FID")]

    header = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_fmt(r[k]) for k, _ in cols) + " |" for r in rows]
    md = "\n".join([header, sep, *body]) + "\n"
    (out_dir / "fid_comparison.md").write_text(md)

    latex_lines = [
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        " & ".join(h for _, h in cols) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        latex_lines.append(" & ".join(_fmt(r[k]) for k, _ in cols) + " \\\\")
    latex_lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "fid_comparison.tex").write_text("\n".join(latex_lines) + "\n")

    print(md)
    print(f"wrote {out_dir/'fid_comparison.md'} and {out_dir/'fid_comparison.tex'} "
          f"({len(rows)} categor{'y' if len(rows) == 1 else 'ies'})")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Assemble the experiment comparison + FID tables.")
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--out-dir", default=None,
                   help="where to write the table files (default: results dir)")
    args = p.parse_args(argv)
    assemble_table(args.results_dir, args.out_dir)
    assemble_fid_table(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
