"""Raw file loading. Parsing quirks live here and nowhere else."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

CUSTOMER_ID = "kunde_id"
TARGET = "avgang_6mnd"

logger = logging.getLogger(__name__)


def load_kunder(data_dir: Path) -> pd.DataFrame:
    """kunder.csv uses ';' separators and ',' as the decimal mark."""
    df = pd.read_csv(
        data_dir / "kunder.csv",
        sep=";",
        decimal=",",
        dtype={CUSTOMER_ID: "string"},
    )
    logger.info("loaded kunder: %d rows, %d columns", len(df), df.shape[1])
    return df


def load_engasjement(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "engasjement.csv", dtype={CUSTOMER_ID: "string"})
    df["maaned"] = pd.to_datetime(df["maaned"], format="%Y-%m", errors="coerce")
    logger.info("loaded engasjement: %d rows", len(df))
    return df


def load_henvendelser(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        data_dir / "henvendelser.csv",
        dtype={CUSTOMER_ID: "string", "henvendelse_id": "string", "tekst": "string"},
    )
    df["dato"] = pd.to_datetime(df["dato"], errors="coerce")
    logger.info("loaded henvendelser: %d rows", len(df))
    return df


def load_relasjoner(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "relasjoner.csv", dtype="string")
    logger.info("loaded relasjoner: %d rows", len(df))
    return df


def load_raw(cfg: dict) -> dict[str, pd.DataFrame]:
    data_dir = cfg["paths"]["data_dir"]
    return {
        "kunder": load_kunder(data_dir),
        "engasjement": load_engasjement(data_dir),
        "henvendelser": load_henvendelser(data_dir),
        "relasjoner": load_relasjoner(data_dir),
    }
