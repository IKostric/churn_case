"""Assemble the modelling tables, one row per customer, one file per feature set."""

from __future__ import annotations

import logging

import pandas as pd

from ..data.load import CUSTOMER_ID
from ..data.validate import validate_modelling_table
from . import customers as customer_features
from . import engagement as engagement_features
from . import inquiries as inquiry_features
from .text import TEXT_COLUMN, build_text_frame

META_COLUMNS = (CUSTOMER_ID, "target")
CATEGORICAL_COLUMNS = ("fylke",)

logger = logging.getLogger(__name__)


def feature_path(feature_set: str, cfg: dict):
    return cfg["paths"]["features"] / f"{feature_set}.parquet"


def save_feature_set(df: pd.DataFrame, feature_set: str, cfg: dict) -> None:
    path = feature_path(feature_set, cfg)
    df.to_parquet(path, index=False)
    logger.info("wrote %s (%d rows, %d columns)", path.name, len(df), df.shape[1])


def load_feature_set(feature_set: str, cfg: dict) -> pd.DataFrame:
    path = feature_path(feature_set, cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m src.pipeline build-features` first")
    return pd.read_parquet(path)


def build_main(clean: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    kunder = clean["kunder"]
    target = customer_features.build_target(kunder)
    df = target.merge(
        customer_features.build_customer_features(kunder, cfg["leakage_columns"]),
        on=CUSTOMER_ID,
        how="left",
    )
    df = df.merge(
        engagement_features.build_engagement_features(
            clean["engasjement"],
            period_start=cfg["engagement"]["period_start"],
            period_end=cfg["engagement"]["period_end"],
            epsilon=cfg["engagement"]["epsilon"],
        ),
        on=CUSTOMER_ID,
        how="left",
    )
    if cfg["features"]["include_inquiry_metadata_in_main"]:
        df = df.merge(
            inquiry_features.build_inquiry_features(clean["henvendelser"], df[CUSTOMER_ID]),
            on=CUSTOMER_ID,
            how="left",
        )

    df = df[[*META_COLUMNS, *[c for c in df.columns if c not in META_COLUMNS]]]
    validate_modelling_table(df, cfg["leakage_columns"])
    return df


def add_text(base: pd.DataFrame, henvendelser: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Append the raw per-customer inquiry document to any base table."""
    text = build_text_frame(
        henvendelser,
        base[CUSTOMER_ID],
        prepend_channel_token=cfg["text"]["prepend_channel_token"],
    )
    df = base.merge(text, on=CUSTOMER_ID, how="left")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
    validate_modelling_table(df, cfg["leakage_columns"])
    return df


def add_clusters(base: pd.DataFrame, cluster_features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Append per-customer inquiry-topic features to any base table."""
    df = base.merge(cluster_features, on=CUSTOMER_ID, how="left")
    validate_modelling_table(df, cfg["leakage_columns"])
    return df


def feature_spec(df: pd.DataFrame, feature_set: str) -> dict:
    """Split the table's columns into numeric / categorical / text roles."""
    columns = [c for c in df.columns if c not in META_COLUMNS and c != "split"]
    text = TEXT_COLUMN if TEXT_COLUMN in columns else None
    categorical = [c for c in columns if c in CATEGORICAL_COLUMNS]
    numeric = [c for c in columns if c not in categorical and c != text]
    return {
        "feature_set": feature_set,
        "numeric": numeric,
        "categorical": categorical,
        "text": text,
        "all": numeric + categorical + ([text] if text else []),
    }
