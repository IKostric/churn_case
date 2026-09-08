"""Two figures: the precision-recall curve, and the at-k curves.

Colour is fixed per model and never recycled; line style doubles the encoding
by model family so the figures survive greyscale and colour-blind viewing.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

MODEL_COLORS = {
    "logistic_main": "#0072B2",
    "catboost_main": "#D55E00",
    "catboost_main_text": "#117733",
    "catboost_main_clusters": "#CC79A7",
}
FAMILY_STYLE = {"logistic": "--", "catboost": "-"}
REFERENCE = "#888888"

logger = logging.getLogger(__name__)


def _style(model: str) -> dict:
    return {
        "color": MODEL_COLORS.get(model, "#666666"),
        "linestyle": FAMILY_STYLE.get(model.split("_", 1)[0], "-"),
        "linewidth": 2,
    }


def _save(fig, path) -> None:
    for ax in fig.axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path.name)


def plot_precision_recall(predictions: dict, base_rate: float, path) -> None:
    """The curve whose area is PR-AUC, drawn from every threshold."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model, (y_true, y_prob) in predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall * 100, precision * 100, **_style(model),
                label=f"{model} (PR-AUC={average_precision_score(y_true, y_prob):.3f})")
    ax.axhline(base_rate * 100, color=REFERENCE, linestyle=":", linewidth=1)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title(f"Precision-recall (random classifier = {base_rate:.1%})")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_at_k(curves: dict, base_rate: float, path) -> None:
    """Precision and recall against the share of customers targeted."""
    panels = [
        ("precision", "Precision (%)", base_rate * 100),
        ("recall", "Recall (%)", None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (column, ylabel, reference) in zip(axes, panels):
        for model, curve in curves.items():
            ax.plot(curve["k"] * 100, curve[column] * 100, label=model,
                    marker="o", markersize=4, **_style(model))
        if reference is not None:
            ax.axhline(reference, color=REFERENCE, linestyle=":", linewidth=1)
        ax.set_xlabel("Customers targeted (%)")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Metrics by targeting capacity", fontsize=12)
    _save(fig, path)
