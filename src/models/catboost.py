"""CatBoost on mixed structured features, with text reduced to SVD components.

Thousands of raw TF-IDF dimensions would swamp a tree model, so text arrives as
a few dozen dense components (fitted on training rows only). Tuning is a small
grid scored on one group-aware validation fold with early stopping; the chosen
configuration is then refitted on the whole training set.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from ..features.text import build_text_reducer, svd_column_names
from .split import cv_splitter

logger = logging.getLogger(__name__)


def _prepare(X: pd.DataFrame, spec: dict, reducer=None) -> pd.DataFrame:
    frame = X[spec["numeric"]].astype(float).copy()
    for column in spec["categorical"]:
        frame[column] = X[column].astype("string").fillna("missing").astype(str)
    if spec["text"]:
        matrix = reducer.transform(X[spec["text"]])
        components = pd.DataFrame(
            matrix, columns=svd_column_names(matrix.shape[1]), index=X.index
        )
        frame = pd.concat([frame, components], axis=1)
    return frame


def _classifier(cfg: dict, depth: int, learning_rate: float, iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=cfg["catboost"]["l2_leaf_reg"],
        loss_function="Logloss",
        # Logloss for early stopping: PRAUC on a few thousand validation rows
        # with ~300 positives is noisy per-iteration and can stop at the first
        # tree, leaving an uncalibrated model. Candidates are still *selected*
        # by validation average precision below.
        eval_metric="Logloss",
        random_seed=cfg["seed"],
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _param_combinations(cfg: dict) -> list[dict]:
    grid = cfg["catboost"]["param_grid"]
    return [
        {"depth": depth, "learning_rate": learning_rate}
        for depth in grid["depth"]
        for learning_rate in grid["learning_rate"]
    ]


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    spec: dict,
    cfg: dict,
    params: dict | None = None,
) -> dict:
    reducer = None
    if params is not None:
        if spec["text"]:
            reducer = build_text_reducer(cfg).fit(X[spec["text"]])
        model = _classifier(cfg, params["depth"], params["learning_rate"], params["iterations"])
        model.fit(_prepare(X, spec, reducer), y, cat_features=spec["categorical"])
        return {"model": model, "reducer": reducer, "params": dict(params),
                "cv_score": None, "spec": spec}

    # one group-aware fold for early stopping and grid scoring
    inner = cv_splitter(cfg["logistic"]["cv_folds"], cfg)
    train_index, valid_index = next(iter(inner.split(X, y, groups=groups)))
    X_inner, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_inner, y_valid = y.iloc[train_index], y.iloc[valid_index]

    inner_reducer = build_text_reducer(cfg).fit(X_inner[spec["text"]]) if spec["text"] else None
    prepared_inner = _prepare(X_inner, spec, inner_reducer)
    prepared_valid = _prepare(X_valid, spec, inner_reducer)

    best = None
    for candidate in _param_combinations(cfg):
        model = _classifier(cfg, candidate["depth"], candidate["learning_rate"],
                            cfg["catboost"]["iterations"])
        model.fit(
            prepared_inner, y_inner,
            eval_set=(prepared_valid, y_valid),
            cat_features=spec["categorical"],
            early_stopping_rounds=cfg["catboost"]["early_stopping_rounds"],
        )
        score = average_precision_score(y_valid, model.predict_proba(prepared_valid)[:, 1])
        # get_best_iteration() can legitimately be 0 (the first tree was best);
        # `or 1` would silently turn that into 1, so test for None explicitly.
        best_iteration = model.get_best_iteration()
        rounds = (0 if best_iteration is None else int(best_iteration)) + 1
        logger.info("catboost candidate %s: valid AP=%.4f at %d rounds", candidate, score, rounds)
        if best is None or score > best["cv_score"]:
            best = {**candidate, "iterations": rounds, "cv_score": float(score)}

    # A floor matters: if early stopping fires almost immediately the model has
    # too few trees to produce a usable spread of scores, and the ranking the
    # at-k metrics depend on collapses.
    chosen = {"depth": best["depth"], "learning_rate": best["learning_rate"],
              "iterations": max(int(best["iterations"] * 1.1), cfg["catboost"]["min_iterations"])}
    logger.info("catboost: best %s (valid AP=%.4f)", chosen, best["cv_score"])

    if spec["text"]:
        reducer = build_text_reducer(cfg).fit(X[spec["text"]])
    model = _classifier(cfg, chosen["depth"], chosen["learning_rate"], chosen["iterations"])
    model.fit(_prepare(X, spec, reducer), y, cat_features=spec["categorical"])
    return {"model": model, "reducer": reducer, "params": chosen,
            "cv_score": best["cv_score"], "spec": spec}


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    prepared = _prepare(X, bundle["spec"], bundle["reducer"])
    return bundle["model"].predict_proba(prepared)[:, 1]


def interpretation(bundle: dict) -> pd.DataFrame:
    model = bundle["model"]
    return (
        pd.DataFrame({
            "feature": model.feature_names_,
            "importance": model.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
