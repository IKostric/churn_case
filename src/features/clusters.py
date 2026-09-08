"""Cluster inquiry embeddings into topics, then aggregate to customer features.

Leakage discipline: KMeans is fitted on inquiries belonging to *training*
customers only, and test inquiries are assigned to the nearest training
centroid. Nothing is refitted on test data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..data.load import CUSTOMER_ID
from .embeddings import INQUIRY_ID, embedding_columns

CLUSTER_COLUMN = "cluster"

logger = logging.getLogger(__name__)


def _matrix(inquiries: pd.DataFrame, n_dimensions: int) -> np.ndarray:
    return inquiries[embedding_columns(n_dimensions)].to_numpy(dtype=np.float32)


def select_k(train_matrix: np.ndarray, cfg: dict) -> tuple[int, pd.DataFrame]:
    """Pick k by silhouette over a small candidate list - a coherence check,
    not an optimisation sweep."""
    rng = np.random.default_rng(cfg["seed"])
    sample_size = min(cfg["clusters"]["silhouette_sample"], len(train_matrix))
    sample_index = rng.choice(len(train_matrix), size=sample_size, replace=False)

    rows = []
    for k in cfg["clusters"]["k_candidates"]:
        model = KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"]).fit(train_matrix)
        score = silhouette_score(train_matrix[sample_index], model.labels_[sample_index])
        rows.append({"k": k, "silhouette": float(score), "inertia": float(model.inertia_)})
        logger.info("k=%d: silhouette=%.4f inertia=%.1f", k, score, model.inertia_)

    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    logger.info("selected k=%d", best_k)
    return best_k, scores


def fit_clusters(inquiries: pd.DataFrame, train_customers: set[str], cfg: dict):
    """Fit KMeans on training-customer inquiries; assign every inquiry."""
    n_dimensions = cfg["clusters"]["embedding_dim"]
    is_train = inquiries[CUSTOMER_ID].isin(train_customers)
    logger.info(
        "clustering on %d training inquiries (%d test inquiries assigned afterwards)",
        int(is_train.sum()), int((~is_train).sum()),
    )

    train_matrix = _matrix(inquiries[is_train], n_dimensions)
    k, scores = select_k(train_matrix, cfg)
    model = KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"]).fit(train_matrix)

    assigned = inquiries.copy()
    assigned[CLUSTER_COLUMN] = model.predict(_matrix(inquiries, n_dimensions))
    assigned["distance_to_centroid"] = np.linalg.norm(
        _matrix(assigned, n_dimensions) - model.cluster_centers_[assigned[CLUSTER_COLUMN]], axis=1
    )
    return model, assigned, scores


def cluster_name(cluster: int) -> str:
    return f"cluster_{cluster:02d}"


def build_cluster_features(
    assigned: pd.DataFrame,
    customer_ids: pd.Series,
    n_clusters: int,
    cfg: dict,
) -> pd.DataFrame:
    """One share per topic: what fraction of the customer's inquiries it was.

    A share is scale-free, so a customer with one inquiry and a customer with
    seven are described on the same footing, and it already carries which
    topics someone raises most - recency and volume are covered by the inquiry
    metadata in `main`.
    """
    totals = assigned.groupby(CUSTOMER_ID).size()
    features = pd.DataFrame(index=pd.Index(customer_ids, name=CUSTOMER_ID))

    for cluster in range(n_clusters):
        per_topic = assigned[assigned[CLUSTER_COLUMN] == cluster].groupby(CUSTOMER_ID).size()
        features[f"share_inquiries_{cluster_name(cluster)}"] = (per_topic / totals).astype(float)

    # no inquiries at all -> zero share of every topic, not missing data
    features = features.fillna(0.0)
    logger.info("cluster features: %d columns for %d customers", features.shape[1], len(features))
    return features.reset_index()


def representative_inquiries(assigned: pd.DataFrame, n_per_cluster: int) -> pd.DataFrame:
    """The inquiries closest to each centroid - the input to LLM annotation."""
    return (
        assigned.sort_values([CLUSTER_COLUMN, "distance_to_centroid"])
        .groupby(CLUSTER_COLUMN)
        .head(n_per_cluster)
        .loc[:, [CLUSTER_COLUMN, INQUIRY_ID, "kanal", "distance_to_centroid", "text"]]
        .reset_index(drop=True)
    )
