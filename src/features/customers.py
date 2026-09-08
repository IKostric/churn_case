"""Static customer features from kunder.csv."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.load import CUSTOMER_ID, TARGET

# Snapshot-dated columns only. Anything listed in config.leakage_columns is
# excluded here and asserted against in tests/test_leakage.py.
BASE_COLUMNS = (
    "alder",
    "fylke",
    "kundeforhold_aar",
    "antall_produkter",
    "har_kundeprogram",
    "laanebalanse",
    "rente_boliglaan",
    "innskudd",
    "raadgiversamtaler_12mnd",
)

CATEGORICAL_COLUMNS = ("fylke",)

logger = logging.getLogger(__name__)


def build_customer_features(kunder: pd.DataFrame, leakage_columns: list[str]) -> pd.DataFrame:
    present_leakage = sorted(set(leakage_columns) & set(BASE_COLUMNS))
    assert not present_leakage, f"leakage columns in BASE_COLUMNS: {present_leakage}"

    df = kunder[[CUSTOMER_ID, *BASE_COLUMNS]].copy()

    # Both balances are heavily right-skewed, so only the log versions are kept
    # - raw and log are the same information, and the log is what a linear
    # model can use.
    df["log_laanebalanse"] = np.log1p(df.pop("laanebalanse"))
    df["log_innskudd"] = np.log1p(df["innskudd"])

    df["har_laan"] = (df["log_laanebalanse"] > 0).astype(float)
    df["har_innskudd"] = (df.pop("innskudd") > 0).astype(float)

    # rente_boliglaan is missing in exactly the rows where har_laan is 0, so the
    # absence is structural, not unknown data: a customer without a mortgage
    # pays no mortgage interest. Filling 0 states that, and har_laan already
    # separates them from borrowers, so the model can still price the two groups
    # apart. Leaving it NaN would instead have it imputed as though a rate
    # existed and had merely gone unrecorded.
    structural = df["rente_boliglaan"].isna()
    assert structural.equals(df["har_laan"].eq(0)), \
        "rente_boliglaan is missing outside the no-loan rows - check the extract"
    df["rente_boliglaan"] = df["rente_boliglaan"].fillna(0.0)

    logger.info("customer features: %d rows, %d columns", len(df), df.shape[1] - 1)
    return df


def build_target(kunder: pd.DataFrame) -> pd.DataFrame:
    return kunder[[CUSTOMER_ID, TARGET]].rename(columns={TARGET: "target"})
