"""Engagement features from the monthly panel, built to survive seasonality.

Both metrics have a strong, repeating yearly cycle (population mean logins peak
in April and trough in Oct-Nov, near-identically in 2024 and 2025). A naive
``last_3m - previous_3m`` trend therefore compares a seasonal trough to a
seasonal peak and mostly measures the calendar.

Five features spanning two independent mechanisms, neither of which replaces
the other:

* **level contrast** - ``last_month`` against ``mean_last_12m``: is the latest
  month below this customer's own normal volume. One month wide, so it reacts
  fast, but December is a seasonal trough for everybody, so it is
  season-contaminated in absolute terms.
* **season-free change** - ``q4_yoy_pct``: the last quarter against the *same*
  quarter a year earlier, so the annual cycle cancels. Three months wide, so
  slower, but it cannot mistake winter for churn.

Logins and cards duplicate each other on levels (r=0.98) but not on change
(r=0.19), which is why both trends are kept and a single volume baseline
serves both.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..data.load import CUSTOMER_ID

METRIC_PREFIX = {"innlogginger": "login", "korttransaksjoner": "card"}

logger = logging.getLogger(__name__)


def _check_period(months: list[pd.Timestamp], period_start: str, period_end: str) -> None:
    expected_start, expected_end = pd.Period(period_start, "M"), pd.Period(period_end, "M")
    actual_start, actual_end = pd.Period(months[0], "M"), pd.Period(months[-1], "M")
    if (actual_start, actual_end) != (expected_start, expected_end):
        logger.warning(
            "engagement period %s..%s differs from configured %s..%s",
            actual_start, actual_end, expected_start, expected_end,
        )


def build_engagement_features(
    engasjement: pd.DataFrame,
    period_start: str,
    period_end: str,
    epsilon: float = 1.0,
) -> pd.DataFrame:
    wide = {
        metric: engasjement.pivot(index=CUSTOMER_ID, columns="maaned", values=metric)
        for metric in METRIC_PREFIX
    }
    months = sorted(next(iter(wide.values())).columns)
    _check_period(months, period_start, period_end)

    last_month = months[-1]
    last_3m = months[-3:]
    last_12m = months[-12:]
    prior_3m = [m - pd.DateOffset(years=1) for m in last_3m]
    has_prior_year = set(prior_3m).issubset(set(months))
    if not has_prior_year:
        logger.warning("fewer than 24 months of history - year-over-year features skipped")

    features = pd.DataFrame(index=next(iter(wide.values())).index)
    for metric, prefix in METRIC_PREFIX.items():
        panel = wide[metric]
        features[f"{prefix}_last_month"] = panel[last_month]
        if not has_prior_year:
            continue
        recent, prior = panel[last_3m].mean(axis=1), panel[prior_3m].mean(axis=1)
        features[f"{prefix}_q4_yoy_pct"] = (recent - prior) / (prior.abs() + epsilon)

    # One normal-volume baseline is enough: the two metrics' 12-month means are
    # r=0.98, so the second carries no additional information.
    features["card_mean_last_12m"] = wide["korttransaksjoner"][last_12m].mean(axis=1)

    logger.info("engagement features: %d rows, %d columns", len(features), features.shape[1])
    return features.reset_index()
