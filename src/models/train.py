"""Configuration-driven experiment loop. Every run uses the same split and code."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from ..data.load import CUSTOMER_ID
from ..evaluation.metrics import summary_metrics
from ..features.build import feature_spec, load_feature_set
from . import catboost as catboost_model
from . import logistic as logistic_model
from .split import load_split

MODEL_FAMILIES = {"logistic": logistic_model, "catboost": catboost_model}

logger = logging.getLogger(__name__)


def run_name(family: str, feature_set: str) -> str:
    return f"{family}_{feature_set}"


def experiments(cfg: dict) -> list[tuple[str, str]]:
    return [(family, feature_set) for family, feature_set in cfg["experiments"]]


def _meta_path(name: str, cfg: dict):
    return cfg["paths"]["models"] / f"{name}_meta.json"


def _prediction_frame(ids: pd.Series, y: pd.Series, probability: np.ndarray, split: str) -> pd.DataFrame:
    frame = pd.DataFrame({
        CUSTOMER_ID: ids.to_numpy(),
        "y_true": y.to_numpy(),
        "predicted_probability": probability,
        "split": split,
    })
    frame["risk_rank"] = frame["predicted_probability"].rank(ascending=False, method="first").astype(int)
    frame["risk_percentile"] = frame["predicted_probability"].rank(pct=True, ascending=False)
    return frame.sort_values("risk_rank").reset_index(drop=True)


def run_experiment(family: str, feature_set: str, cfg: dict, table: pd.DataFrame | None = None) -> dict:
    name = run_name(family, feature_set)
    module = MODEL_FAMILIES[family]
    df = load_feature_set(feature_set, cfg) if table is None else table
    split = load_split(cfg)

    merged = df.merge(split, on=CUSTOMER_ID, how="inner")
    spec = feature_spec(df, feature_set)
    is_train = merged["split"] == "train"
    X, y = merged[spec["all"]], merged["target"]

    logger.info("=== %s: %d train / %d test rows, %d features ===",
                name, int(is_train.sum()), int((~is_train).sum()), len(spec["all"]))
    started = time.perf_counter()
    bundle = module.fit(X[is_train], y[is_train], merged.loc[is_train, "group_id"], spec, cfg)
    runtime = time.perf_counter() - started

    predictions = pd.concat([
        _prediction_frame(merged.loc[is_train, CUSTOMER_ID], y[is_train],
                          module.predict_proba(bundle, X[is_train]), "train"),
        _prediction_frame(merged.loc[~is_train, CUSTOMER_ID], y[~is_train],
                          module.predict_proba(bundle, X[~is_train]), "test"),
    ], ignore_index=True)
    predictions.to_csv(cfg["paths"]["predictions"] / f"{name}.csv", index=False)

    module.interpretation(bundle).to_csv(cfg["paths"]["models"] / f"{name}_interpretation.csv", index=False)

    k_values = cfg["evaluation"]["k_values"]
    meta = {
        "model": name,
        "family": family,
        "feature_set": feature_set,
        "params": bundle["params"],
        "cv_score": bundle["cv_score"],
        "runtime_seconds": round(runtime, 2),
        "n_features": len(spec["all"]),
        "n_train": int(is_train.sum()),
        "n_test": int((~is_train).sum()),
        "train_metrics": summary_metrics(
            predictions.query("split == 'train'")["y_true"],
            predictions.query("split == 'train'")["predicted_probability"], k_values),
        "test_metrics": summary_metrics(
            predictions.query("split == 'test'")["y_true"],
            predictions.query("split == 'test'")["predicted_probability"], k_values),
    }
    with open(_meta_path(name, cfg), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("%s done in %.1fs | test PR-AUC=%.4f lift@10=%.2f",
                name, runtime, meta["test_metrics"]["pr_auc"],
                meta["test_metrics"]["lift_at_10"])
    return meta
