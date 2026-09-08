"""Ranking and capacity metrics. One implementation, used by every model.

Four numbers: PR-AUC over the whole ranking, and precision, recall and lift
within the top k share of it. A retention team can only contact so many
customers, so what matters is the quality of the top of the list, not accuracy
at an arbitrary threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def k_label(k: float) -> str:
    """0.10 -> '10'."""
    return f"{int(round(k * 100)):02d}"


def top_k_metrics(y_true, y_prob, k: float) -> dict:
    """Precision, recall and lift among the top k share of the ranked list."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n_targeted = max(1, int(round(k * len(y_true))))
    captured = y_true[np.argsort(-y_prob, kind="stable")[:n_targeted]].sum()

    precision = captured / n_targeted
    base_rate = y_true.mean()
    return {
        "k": k,
        "n_targeted": n_targeted,
        "precision": precision,
        "recall": captured / y_true.sum() if y_true.sum() else np.nan,
        "lift": precision / base_rate if base_rate else np.nan,
    }


def curve_at_k(y_true, y_prob, k_values) -> pd.DataFrame:
    return pd.DataFrame([top_k_metrics(y_true, y_prob, k) for k in k_values])


def summary_metrics(y_true, y_prob, k_values) -> dict:
    metrics = {
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
    for k in k_values:
        scores = top_k_metrics(y_true, y_prob, k)
        for name in ("precision", "recall", "lift"):
            metrics[f"{name}_at_{k_label(k)}"] = float(scores[name])
    return metrics
