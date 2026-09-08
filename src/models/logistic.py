"""Regularised logistic regression on structured features (+ sparse TF-IDF).

class_weight is left at None on purpose: the deliverable is a calibrated
probability ranking for a capacity-limited outreach list, not a balanced
0/1 decision at threshold 0.5.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features.text import build_vectorizer
from .split import cv_splitter

logger = logging.getLogger(__name__)


def build_pipeline(spec: dict, cfg: dict, C: float = 1.0) -> Pipeline:
    transformers = [
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), spec["numeric"]),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]), spec["categorical"]),
    ]
    if spec["text"]:
        # column passed as a string (not a list) so the vectorizer receives a 1-D series
        transformers.append(("text", build_vectorizer(cfg), spec["text"]))

    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)),
        ("classifier", LogisticRegression(
            l1_ratio=0,  # pure L2; `penalty="l2"` is deprecated as of sklearn 1.8
            C=C,
            class_weight=None,
            max_iter=cfg["logistic"]["max_iter"],
            random_state=cfg["seed"],
        )),
    ])


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    spec: dict,
    cfg: dict,
    params: dict | None = None,
) -> dict:
    """Tune C by group-aware CV (or use ``params`` as given) and fit on all of X."""
    if params is not None:
        pipeline = build_pipeline(spec, cfg, C=params["C"]).fit(X, y)
        return {"model": pipeline, "params": dict(params), "cv_score": None, "spec": spec}

    search = GridSearchCV(
        build_pipeline(spec, cfg),
        param_grid={"classifier__C": cfg["logistic"]["c_grid"]},
        scoring="average_precision",
        cv=cv_splitter(cfg["logistic"]["cv_folds"], cfg),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    best_C = search.best_params_["classifier__C"]
    logger.info("logistic: best C=%s, cv average_precision=%.4f", best_C, search.best_score_)
    return {
        "model": search.best_estimator_,
        "params": {"C": best_C},
        "cv_score": float(search.best_score_),
        "spec": spec,
    }


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(X)[:, 1]


def interpretation(bundle: dict) -> pd.DataFrame:
    pipeline = bundle["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_.ravel()
    return (
        pd.DataFrame({
            "feature": names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        })
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
