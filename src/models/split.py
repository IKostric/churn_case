"""One grouped train/test split, created once and reused by every model.

Related customers must never straddle a split boundary. If a partner sits on
the other side, the partner-risk feature would either need the locked-away test
set to be scored early, or - inside cross-validation - would carry the
customer's own label back into its own feature: a partner's out-of-fold score
is produced by a model trained on every other fold, which includes the
customer's own fold. Grouping the pair removes both problems.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..data.load import CUSTOMER_ID

logger = logging.getLogger(__name__)


def build_groups(customer_ids: pd.Series, relasjoner: pd.DataFrame) -> pd.Series:
    """One group per relation pair; every other customer is their own group.

    The relations in this dataset are disjoint pairs - no customer appears in
    more than one - so a pair's group is just its lower id (cleaning already
    orders each pair). The assertion is what makes that safe: if a future
    extract links one customer into two pairs, this raises instead of quietly
    letting a pair straddle the split.
    """
    members = pd.concat([relasjoner["kunde_id_1"], relasjoner["kunde_id_2"]])
    assert members.is_unique, "relations are not disjoint pairs - grouping needs components"

    left = relasjoner["kunde_id_1"]
    pair_group = pd.concat([
        pd.Series(left.to_numpy(), index=left),
        pd.Series(left.to_numpy(), index=relasjoner["kunde_id_2"]),
    ])
    groups = pd.Series(customer_ids.to_numpy(),
                       index=pd.Index(customer_ids, name=CUSTOMER_ID), name="group_id")
    groups.update(pair_group)
    logger.info("groups: %d for %d customers", groups.nunique(), len(groups))
    return groups


def make_split(target: pd.DataFrame, groups: pd.Series, cfg: dict) -> pd.DataFrame:
    """Hold out one stratified, group-respecting fold as the test set."""
    splitter = StratifiedGroupKFold(
        n_splits=cfg["split"]["n_splits"], shuffle=True, random_state=cfg["seed"]
    )
    aligned_groups = groups.reindex(target[CUSTOMER_ID]).to_numpy()
    folds = list(splitter.split(target, target["target"], groups=aligned_groups))
    _, test_index = folds[cfg["split"]["test_fold"]]

    split = pd.DataFrame({CUSTOMER_ID: target[CUSTOMER_ID], "split": "train"})
    split.loc[split.index[test_index], "split"] = "test"
    split["group_id"] = aligned_groups

    rates = target.assign(split=split["split"]).groupby("split")["target"].agg(["mean", "size"])
    logger.info("split churn rate / size:\n%s", rates.round(4).to_string())
    return split


def split_path(cfg: dict):
    return cfg["paths"]["output_dir"] / "split.csv"


def save_split(split: pd.DataFrame, cfg: dict) -> None:
    split.to_csv(split_path(cfg), index=False)
    logger.info("wrote %s", split_path(cfg).name)


def load_split(cfg: dict) -> pd.DataFrame:
    path = split_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m src.pipeline build-features` first")
    return pd.read_csv(path, dtype={CUSTOMER_ID: "string", "group_id": "string"})


def cv_splitter(n_splits: int, cfg: dict) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
