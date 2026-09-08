"""Explicit data checks, run before any transformation.

Two layers, deliberately:

* ``validate_raw`` inspects the files as delivered. It reports rather than
  aborts for anything cleaning is designed to repair (the delivered
  kunder.csv genuinely contains 400 duplicated kunde_id, so a hard
  uniqueness assert here would only ever fail).
* ``validate_modelling_table`` hard-asserts the invariants that must hold
  once cleaning and feature building are done - uniqueness, a binary
  target, no leakage columns.
"""

from __future__ import annotations

import logging

import pandas as pd

from .load import CUSTOMER_ID, TARGET

SNAPSHOT_DATE = pd.Timestamp("2025-12-31")

# Nothing outside these bounds is a plausible value for a personal customer.
PLAUSIBLE_RANGES = {
    "alder": (18.0, 100.0),
    "kundeforhold_aar": (0.0, 100.0),
    "antall_produkter": (0.0, 30.0),
    "laanebalanse": (0.0, 1e9),
    "innskudd": (0.0, 1e9),
    "rente_boliglaan": (0.0, 15.0),
    "raadgiversamtaler_12mnd": (0.0, 50.0),
}

logger = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    """Raised for problems that cleaning cannot legitimately repair."""


def _issue(check: str, level: str, count: int, detail: str) -> dict:
    return {"check": check, "level": level, "count": int(count), "detail": detail}


def validate_raw(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a report of data problems; raise on fatal ones."""
    kunder, engasjement = raw["kunder"], raw["engasjement"]
    henvendelser, relasjoner = raw["henvendelser"], raw["relasjoner"]
    known_ids = set(kunder[CUSTOMER_ID].dropna())
    issues: list[dict] = []

    # --- kunder ---------------------------------------------------------
    issues.append(_issue("kunder.missing_id", "fatal",
                         kunder[CUSTOMER_ID].isna().sum(), "customer rows without kunde_id"))
    bad_target = ~kunder[TARGET].isin([0, 1])
    issues.append(_issue("kunder.target_values", "fatal",
                         bad_target.sum(), f"{TARGET} outside {{0, 1}}"))
    issues.append(_issue("kunder.duplicate_id", "warn",
                         kunder[CUSTOMER_ID].duplicated().sum(),
                         "duplicated kunde_id (resolved in clean.dedupe_kunder)"))
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in kunder:
            continue
        values = pd.to_numeric(kunder[column], errors="coerce")
        outside = ((values < low) | (values > high)) & values.notna()
        issues.append(_issue(f"kunder.range.{column}", "warn", outside.sum(),
                             f"outside plausible range [{low}, {high}]"))

    # --- engasjement ----------------------------------------------------
    issues.append(_issue("engasjement.duplicate_customer_month", "warn",
                         engasjement.duplicated([CUSTOMER_ID, "maaned"]).sum(),
                         "duplicated customer-month rows"))
    issues.append(_issue("engasjement.invalid_month", "fatal",
                         engasjement["maaned"].isna().sum(), "unparseable maaned"))
    issues.append(_issue("engasjement.post_snapshot", "fatal",
                         (engasjement["maaned"] > SNAPSHOT_DATE).sum(),
                         f"activity months after {SNAPSHOT_DATE.date()}"))
    issues.append(_issue("engasjement.unknown_customer", "warn",
                         (~engasjement[CUSTOMER_ID].isin(known_ids)).sum(),
                         "engagement rows for unknown kunde_id"))

    # --- henvendelser ---------------------------------------------------
    issues.append(_issue("henvendelser.invalid_date", "warn",
                         henvendelser["dato"].isna().sum(), "unparseable dato"))
    issues.append(_issue("henvendelser.post_snapshot", "warn",
                         (henvendelser["dato"] > SNAPSHOT_DATE).sum(),
                         f"inquiries after {SNAPSHOT_DATE.date()} (dropped in cleaning)"))
    issues.append(_issue("henvendelser.unknown_customer", "warn",
                         (~henvendelser[CUSTOMER_ID].isin(known_ids)).sum(),
                         "inquiries for unknown kunde_id"))
    issues.append(_issue("henvendelser.empty_text", "warn",
                         henvendelser["tekst"].fillna("").str.strip().eq("").sum(),
                         "inquiries with empty tekst"))

    # --- relasjoner -----------------------------------------------------
    unknown_side = (~relasjoner["kunde_id_1"].isin(known_ids)) | (~relasjoner["kunde_id_2"].isin(known_ids))
    issues.append(_issue("relasjoner.unknown_customer", "warn", unknown_side.sum(),
                         "relations referencing unknown kunde_id"))
    issues.append(_issue("relasjoner.self_pair", "warn",
                         (relasjoner["kunde_id_1"] == relasjoner["kunde_id_2"]).sum(),
                         "relations pointing at the same customer twice"))

    report = pd.DataFrame(issues)
    fatal = report[(report["level"] == "fatal") & (report["count"] > 0)]
    for row in report[report["count"] > 0].itertuples():
        logger.log(logging.ERROR if row.level == "fatal" else logging.WARNING,
                   "%s: %d (%s)", row.check, row.count, row.detail)
    if not fatal.empty:
        raise ValidationError(f"fatal data problems: {fatal['check'].tolist()}")
    return report


def validate_modelling_table(df: pd.DataFrame, leakage_columns: list[str]) -> None:
    """Invariants for the assembled one-row-per-customer modelling table."""
    assert df[CUSTOMER_ID].is_unique, "modelling table must be one row per customer"
    assert df[CUSTOMER_ID].notna().all(), "modelling table has missing customer ids"
    assert df["target"].isin([0, 1]).all(), "target must be binary"
    assert df["target"].notna().all(), "target must not be missing"
    present = sorted(set(leakage_columns) & set(df.columns))
    assert not present, f"leakage columns present in modelling table: {present}"


def assert_within_snapshot(dates: pd.Series, name: str) -> None:
    latest = dates.max()
    assert pd.isna(latest) or latest <= SNAPSHOT_DATE, (
        f"{name} contains data after the snapshot date {SNAPSHOT_DATE.date()}: {latest}"
    )
