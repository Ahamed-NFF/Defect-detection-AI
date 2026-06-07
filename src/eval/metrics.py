"""
Classification metrics + results-table builder.

Per-class precision/recall/F1 matters more than overall accuracy here, because
the defect class is the minority and the whole point. Also compute AUROC.

assemble_table() reads every experiments/results/*.json and emits the comparison
table (markdown + LaTeX) for the report.

Owner: Member 3 (metrics), Member 4 (table -> report figures)
"""


def classification_metrics(y_true, y_pred, y_score):
    """Return dict: accuracy, precision/recall/f1 (per class + macro), auroc."""
    raise NotImplementedError("Member 3: wrap sklearn.metrics")


def assemble_table(results_dir="experiments/results"):
    """Collect all run JSONs into one comparison table (md + latex)."""
    raise NotImplementedError("Member 3/4: glob results, build dataframe, export")
