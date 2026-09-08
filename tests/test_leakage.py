"""The checks whose failure would silently invalidate every result."""

import pandas as pd
import pytest

from src.data.validate import SNAPSHOT_DATE
from src.features.build import build_main, feature_spec
from src.features.customers import BASE_COLUMNS
from src.features.text import build_text_frame


@pytest.fixture(scope="module")
def main_table(clean, cfg):
    return build_main(clean, cfg)


def test_leakage_columns_never_reach_the_model(main_table, cfg):
    spec = feature_spec(main_table, "main")
    for column in cfg["leakage_columns"]:
        assert column not in main_table.columns
        assert column not in spec["all"]


def test_customer_feature_list_excludes_leakage(cfg):
    assert not set(cfg["leakage_columns"]) & set(BASE_COLUMNS)
    assert "antall_produkter_naa" not in BASE_COLUMNS
    assert "avslutningsgebyr" not in BASE_COLUMNS


def test_inquiries_used_are_within_the_snapshot(clean):
    assert clean["henvendelser"]["dato"].max() <= SNAPSHOT_DATE


def test_engagement_months_are_within_the_snapshot(clean):
    assert clean["engasjement"]["maaned"].max() <= SNAPSHOT_DATE


def test_text_frame_is_built_only_from_pre_snapshot_inquiries(clean):
    henvendelser = clean["henvendelser"]
    customer_ids = pd.Series(sorted(set(henvendelser["kunde_id"]))[:50])

    # a message dated after the snapshot must not survive into the documents
    future = henvendelser.iloc[[0]].copy()
    future["dato"] = SNAPSHOT_DATE + pd.Timedelta(days=30)
    future["tekst"] = "SENTINEL_FUTURE_TOKEN"
    with pytest.raises(AssertionError):
        build_text_frame(pd.concat([henvendelser, future]), customer_ids)

    text = build_text_frame(henvendelser, customer_ids)
    assert not text["inquiry_text"].str.contains("SENTINEL_FUTURE_TOKEN").any()


def test_modelling_table_is_one_row_per_customer_with_binary_target(main_table):
    assert main_table["kunde_id"].is_unique
    assert main_table["target"].isin([0, 1]).all()
    assert main_table["target"].notna().all()


def test_feature_spec_covers_every_non_meta_column(main_table):
    spec = feature_spec(main_table, "main")
    covered = set(spec["all"]) | {"kunde_id", "target"}
    assert covered == set(main_table.columns)
    assert not set(spec["numeric"]) & set(spec["categorical"])
