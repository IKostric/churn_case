"""Free-text inquiry features.

The per-customer text is built here; the *vectorizer* is deliberately not fitted
here. It is a pipeline step so it can be fitted on training folds only - fitting
TF-IDF on the full table would leak test-set vocabulary and document frequencies.

TF-IDF rather than embeddings for the first implementation: transparent, fast,
reproducible, and free of external model dependencies.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from ..data.load import CUSTOMER_ID
from ..data.validate import assert_within_snapshot

TEXT_COLUMN = "inquiry_text"

logger = logging.getLogger(__name__)


def build_text_frame(
    henvendelser: pd.DataFrame,
    customer_ids: pd.Series,
    prepend_channel_token: bool = True,
) -> pd.DataFrame:
    """Concatenate each customer's inquiry history into a single document."""
    assert_within_snapshot(henvendelser["dato"], "henvendelser")

    chunks = henvendelser["tekst"].astype("string").fillna("")
    if prepend_channel_token:
        tokens = "kanal_" + henvendelser["kanal"].astype("string").str.replace("-", "_", regex=False)
        chunks = tokens + " " + chunks

    documents = (
        henvendelser.assign(chunk=chunks)
        .sort_values(["dato", "henvendelse_id"])
        .groupby(CUSTOMER_ID)["chunk"]
        .agg(" ".join)
    )
    text = pd.Series("", index=pd.Index(customer_ids, name=CUSTOMER_ID), dtype="object")
    text.update(documents)

    non_empty = int((text.str.len() > 0).sum())
    logger.info("text frame: %d of %d customers have inquiry text", non_empty, len(text))
    return text.rename(TEXT_COLUMN).reset_index()


def build_vectorizer(cfg: dict) -> TfidfVectorizer:
    """Sparse TF-IDF, used directly by the logistic pipeline."""
    text_cfg = cfg["text"]
    return TfidfVectorizer(
        min_df=text_cfg["min_df"],
        max_df=text_cfg["max_df"],
        ngram_range=tuple(text_cfg["ngram_range"]),
        max_features=text_cfg["max_features"],
        strip_accents="unicode",
        lowercase=True,
    )


def build_text_reducer(cfg: dict) -> Pipeline:
    """TF-IDF followed by SVD, so CatBoost gets a few dense components
    instead of thousands of sparse ones."""
    return Pipeline(
        [
            ("tfidf", build_vectorizer(cfg)),
            ("svd", TruncatedSVD(
                n_components=cfg["text"]["svd_components"],
                random_state=cfg["seed"],
            )),
        ]
    )


def svd_column_names(n_components: int) -> list[str]:
    return [f"text_svd_{i:03d}" for i in range(1, n_components + 1)]
