"""Model comparison table and figures, rebuilt from the saved predictions.

Every metric is recomputed here from `outputs/predictions/*.csv`, so the table
cannot drift from what the models actually produced.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from ..models.train import experiments, run_name
from .metrics import curve_at_k, k_label, summary_metrics
from .plots import plot_at_k, plot_precision_recall

logger = logging.getLogger(__name__)


def evaluate_all(cfg: dict) -> pd.DataFrame:
    k_values = cfg["evaluation"]["k_values"]
    headline = k_label(cfg["evaluation"]["headline_k"])
    rows, curves, predictions, base_rate = [], {}, {}, None

    for family, feature_set in experiments(cfg):
        name = run_name(family, feature_set)
        path = cfg["paths"]["predictions"] / f"{name}.csv"
        if not path.exists():
            logger.warning("no predictions for %s - skipping", name)
            continue

        test = pd.read_csv(path).query("split == 'test'")
        y_true, y_prob = test["y_true"], test["predicted_probability"]
        metrics = summary_metrics(y_true, y_prob, k_values)
        base_rate = metrics["base_rate"]

        curves[name] = curve_at_k(y_true, y_prob, k_values)
        predictions[name] = (y_true.to_numpy(), y_prob.to_numpy())
        meta_path = cfg["paths"]["models"] / f"{name}_meta.json"
        n_features = json.loads(meta_path.read_text())["n_features"] if meta_path.exists() else None

        rows.append({
            "model": name,
            "n_features": n_features,
            "pr_auc": metrics["pr_auc"],
            f"precision_at_{headline}": metrics[f"precision_at_{headline}"],
            f"recall_at_{headline}": metrics[f"recall_at_{headline}"],
            f"lift_at_{headline}": metrics[f"lift_at_{headline}"],
        })
        curves[name].assign(model=name).to_csv(
            cfg["paths"]["metrics"] / f"{name}_curve.csv", index=False
        )

    if not rows:
        raise RuntimeError("no predictions found - run `python -m src.pipeline train` first")

    comparison = pd.DataFrame(rows)  # rows follow config.experiments order
    comparison.to_csv(cfg["paths"]["metrics"] / "model_comparison.csv", index=False)
    logger.info("model comparison (test set, k=%s%%):\n%s",
                headline, comparison.round(4).to_string(index=False))

    plot_precision_recall(predictions, base_rate, cfg["paths"]["figures"] / "precision_recall.png")
    plot_at_k(curves, base_rate, cfg["paths"]["figures"] / "metrics_at_k.png")
    return comparison
