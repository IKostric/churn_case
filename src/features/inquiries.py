"""Inquiry *metadata* features (how often, how recently, which channel).

Kept separate from ``text.py`` on purpose: this module covers what is knowable
from the inquiry log without reading a single message, so the text experiment
measures the added value of message *content* rather than re-discovering that
customers who contact the bank churn more.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..data.load import CUSTOMER_ID
from ..data.validate import SNAPSHOT_DATE, assert_within_snapshot

CHANNELS = ("chat", "telefon", "e-post")

logger = logging.getLogger(__name__)


def build_inquiry_features(henvendelser: pd.DataFrame, customer_ids: pd.Series) -> pd.DataFrame:
    assert_within_snapshot(henvendelser["dato"], "henvendelser")

    grouped = henvendelser.groupby(CUSTOMER_ID)
    features = pd.DataFrame(index=pd.Index(customer_ids, name=CUSTOMER_ID))
    features["n_inquiries"] = grouped.size()
    features["days_since_last_inquiry"] = (SNAPSHOT_DATE - grouped["dato"].max()).dt.days

    for window in (90, 365):
        cutoff = SNAPSHOT_DATE - pd.Timedelta(days=window)
        recent = henvendelser[henvendelser["dato"] >= cutoff].groupby(CUSTOMER_ID).size()
        features[f"n_inquiries_last_{window}d"] = recent

    per_channel = (
        henvendelser.pivot_table(index=CUSTOMER_ID, columns="kanal", values="dato", aggfunc="count")
        .reindex(columns=list(CHANNELS))
    )
    for channel in CHANNELS:
        features[f"n_inquiries_{channel.replace('-', '_')}"] = per_channel[channel]

    # Absence of a record means no contact, not missing data. days_since stays
    # NaN, which the imputer flags - "never contacted" is not "contacted long ago".
    count_columns = [c for c in features.columns if c.startswith("n_inquiries")]
    features[count_columns] = features[count_columns].fillna(0.0)

    logger.info("inquiry metadata features: %d columns", features.shape[1])
    return features.reset_index()
