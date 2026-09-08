"""Deterministic cleaning. Every function is pure: same input, same output."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .load import CUSTOMER_ID, TARGET
from .validate import PLAUSIBLE_RANGES, SNAPSHOT_DATE

YES_VALUES = {"ja", "yes", "1", "true", "j", "y"}
NO_VALUES = {"nei", "no", "0", "false", "n"}

# Channel spellings seen in the data, plus obvious variants.
CHANNEL_ALIASES = {"epost": "e-post", "e_post": "e-post", "mail": "e-post", "email": "e-post"}

# The 400 duplicated kunde_id differ only in raadgiversamtaler_12mnd; keep the
# higher count (the more complete of the two records) and the first value of
# every other column, which are identical within a duplicate group anyway.
DEDUPE_MAX_COLUMNS = ("raadgiversamtaler_12mnd",)

logger = logging.getLogger(__name__)


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Map free-text yes/no spellings to 1.0/0.0, anything else to NaN."""
    normalized = normalize_text(series)
    mapped = normalized.map(lambda v: 1.0 if v in YES_VALUES else (0.0 if v in NO_VALUES else np.nan))
    return pd.to_numeric(mapped, errors="coerce")


def normalize_category(series: pd.Series) -> pd.Series:
    """Lower-case, whitespace-trimmed category labels."""
    return normalize_text(series)


def normalize_channel(series: pd.Series) -> pd.Series:
    normalized = normalize_category(series)
    return normalized.replace(CHANNEL_ALIASES)


def clip_to_plausible(df: pd.DataFrame) -> pd.DataFrame:
    """Values outside a plausible range are data errors, not signal.

    Balances are clipped to their bound (a negative deposit is a booking
    error, not a real position); everything else becomes NaN so the model
    treats it as missing rather than as a fabricated value.
    """
    df = df.copy()
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if column in ("laanebalanse", "innskudd"):
            df[column] = values.clip(lower=low, upper=high)
        else:
            df[column] = values.where((values >= low) & (values <= high))
    return df


def dedupe_kunder(kunder: pd.DataFrame) -> pd.DataFrame:
    duplicated = kunder[CUSTOMER_ID].duplicated().sum()
    if not duplicated:
        return kunder.reset_index(drop=True)
    agg = {c: "first" for c in kunder.columns if c != CUSTOMER_ID}
    for column in DEDUPE_MAX_COLUMNS:
        if column in agg:
            agg[column] = "max"
    deduped = kunder.groupby(CUSTOMER_ID, as_index=False, sort=True).agg(agg)
    logger.info("deduplicated kunder: %d duplicate rows collapsed", duplicated)
    return deduped


def clean_kunder(kunder: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_kunder(kunder)
    df["har_kundeprogram"] = normalize_boolean(df["har_kundeprogram"])
    df["fylke"] = normalize_category(df["fylke"])
    df = clip_to_plausible(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    return df


def clean_engasjement(engasjement: pd.DataFrame) -> pd.DataFrame:
    df = engasjement.dropna(subset=["maaned"]).copy()
    before = len(df)
    df = df.sort_values([CUSTOMER_ID, "maaned"]).drop_duplicates([CUSTOMER_ID, "maaned"], keep="first")
    if len(df) != before:
        logger.info("dropped %d duplicate customer-month engagement rows", before - len(df))
    for column in ("innlogginger", "korttransaksjoner"):
        df[column] = pd.to_numeric(df[column], errors="coerce").clip(lower=0)
    return df.reset_index(drop=True)


def clean_henvendelser(henvendelser: pd.DataFrame) -> pd.DataFrame:
    """Drop undated inquiries and anything after the snapshot date."""
    df = henvendelser.dropna(subset=["dato"]).copy()
    post_snapshot = (df["dato"] > SNAPSHOT_DATE).sum()
    if post_snapshot:
        logger.warning("dropping %d inquiries dated after the snapshot", post_snapshot)
    df = df[df["dato"] <= SNAPSHOT_DATE]
    df["kanal"] = normalize_channel(df["kanal"])
    df["tekst"] = df["tekst"].astype("string").fillna("").str.strip()
    return df.sort_values(["dato", "henvendelse_id"]).reset_index(drop=True)


def clean_relasjoner(relasjoner: pd.DataFrame, valid_ids: set[str]) -> pd.DataFrame:
    """Keep well-formed, de-duplicated, undirected pairs between known customers."""
    df = relasjoner.copy()
    df["relasjon"] = normalize_category(df["relasjon"])
    known = df["kunde_id_1"].isin(valid_ids) & df["kunde_id_2"].isin(valid_ids)
    self_pair = df["kunde_id_1"] == df["kunde_id_2"]
    dropped = (~known | self_pair).sum()
    if dropped:
        logger.warning("dropping %d unusable relation rows (unknown id or self-pair)", dropped)
    df = df[known & ~self_pair].copy()

    # canonical ordering so A-B and B-A collapse to one row
    left = np.minimum(df["kunde_id_1"], df["kunde_id_2"])
    right = np.maximum(df["kunde_id_1"], df["kunde_id_2"])
    df["kunde_id_1"], df["kunde_id_2"] = left, right
    return df.drop_duplicates(["kunde_id_1", "kunde_id_2"]).sort_values(
        ["kunde_id_1", "kunde_id_2"]
    ).reset_index(drop=True)


def clean_all(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    kunder = clean_kunder(raw["kunder"])
    valid_ids = set(kunder[CUSTOMER_ID])
    return {
        "kunder": kunder,
        "engasjement": clean_engasjement(raw["engasjement"]),
        "henvendelser": clean_henvendelser(raw["henvendelser"]),
        "relasjoner": clean_relasjoner(raw["relasjoner"], valid_ids),
    }
