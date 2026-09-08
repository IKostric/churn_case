"""Sentence embeddings for individual inquiries.

The embedding model is frozen and label-free, so embedding every inquiry in one
pass leaks nothing - it never sees the target. What *is* fitted (the clustering)
is fitted on training inquiries only; see ``clusters.py``.

Embeddings are cached on disk keyed by model, dimension and corpus size, so
re-running the pipeline does not re-embed 20k inquiries.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from ..data.load import CUSTOMER_ID
from ..data.validate import assert_within_snapshot

INQUIRY_ID = "henvendelse_id"
META_COLUMNS = (INQUIRY_ID, CUSTOMER_ID, "dato", "kanal")
_WHITESPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

logger = logging.getLogger(__name__)


def clean_inquiry_text(series: pd.Series) -> pd.Series:
    """Fix formatting only - casing, punctuation and wording are left alone
    because the embedding model is what interprets them."""
    text = series.astype("string").fillna("")
    text = text.str.replace(_CONTROL, " ", regex=True)
    return text.str.replace(_WHITESPACE, " ", regex=True).str.strip()


def embedding_columns(n_dimensions: int) -> list[str]:
    return [f"e_{i:03d}" for i in range(n_dimensions)]


def _cache_paths(cfg: dict):
    directory = cfg["paths"]["features"]
    return directory / "inquiry_embeddings.parquet", directory / "inquiry_embeddings.meta.json"


def _cache_signature(cfg: dict, n_inquiries: int) -> dict:
    cluster_cfg = cfg["clusters"]
    return {
        "model": cluster_cfg["embedding_model"],
        "dimensions": cluster_cfg["embedding_dim"],
        "n_inquiries": int(n_inquiries),
    }


def _load_model(cfg: dict):
    from sentence_transformers import SentenceTransformer

    device = cfg["clusters"].get("device") or _default_device()
    model_name = cfg["clusters"]["embedding_model"]
    logger.info("loading embedding model %s on %s", model_name, device)
    return SentenceTransformer(
        model_name,
        device=device,
        truncate_dim=cfg["clusters"]["embedding_dim"],
    )


def _default_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_inquiries(henvendelser: pd.DataFrame, cfg: dict, use_cache: bool = True) -> pd.DataFrame:
    """One row per inquiry: metadata plus a unit-norm embedding vector."""
    assert_within_snapshot(henvendelser["dato"], "henvendelser")

    frame = henvendelser[list(META_COLUMNS)].copy()
    frame["text"] = clean_inquiry_text(henvendelser["tekst"])
    signature = _cache_signature(cfg, len(frame))
    cache_path, meta_path = _cache_paths(cfg)

    if use_cache and cache_path.exists() and meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached == signature:
            logger.info("reusing cached embeddings from %s", cache_path.name)
            return pd.read_parquet(cache_path)
        logger.info("embedding cache stale (%s != %s) - recomputing", cached, signature)

    model = _load_model(cfg)
    logger.info("embedding %d inquiries at %d dimensions", len(frame), signature["dimensions"])
    vectors = model.encode(
        frame["text"].tolist(),
        batch_size=cfg["clusters"]["batch_size"],
        normalize_embeddings=True,  # unit norm: cosine distance == euclidean ordering
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    embeddings = pd.DataFrame(vectors, columns=embedding_columns(vectors.shape[1]), index=frame.index)
    result = pd.concat([frame, embeddings], axis=1)
    result.to_parquet(cache_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(signature, fh, indent=2)
    logger.info("wrote %s (%d rows, %d dims)", cache_path.name, len(result), vectors.shape[1])
    return result
