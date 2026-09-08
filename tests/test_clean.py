import numpy as np
import pandas as pd

from src.data.clean import (
    clean_relasjoner,
    clip_to_plausible,
    dedupe_kunder,
    normalize_boolean,
    normalize_category,
    normalize_channel,
)


def test_yes_spellings_all_map_to_one():
    values = pd.Series(["Ja", "JA", "ja", " ja ", "1", "true", "J"])
    assert normalize_boolean(values).tolist() == [1.0] * len(values)


def test_no_spellings_all_map_to_zero():
    values = pd.Series(["Nei", "NEI", "nei", "0", "false", "N"])
    assert normalize_boolean(values).tolist() == [0.0] * len(values)


def test_unknown_boolean_becomes_missing():
    result = normalize_boolean(pd.Series(["kanskje", "", None]))
    assert result.isna().all()


def test_category_and_channel_normalization():
    assert normalize_category(pd.Series([" Rogaland ", "ROGALAND"])).tolist() == ["rogaland"] * 2
    assert normalize_channel(pd.Series(["E-post", "epost", "EMAIL"])).tolist() == ["e-post"] * 3


def test_dedupe_keeps_max_advisor_calls_and_one_row_per_customer():
    kunder = pd.DataFrame({
        "kunde_id": ["K1", "K1", "K2"],
        "alder": [40.0, 40.0, 55.0],
        "raadgiversamtaler_12mnd": [0, 2, 1],
    })
    deduped = dedupe_kunder(kunder)
    assert deduped["kunde_id"].is_unique
    assert deduped.set_index("kunde_id").loc["K1", "raadgiversamtaler_12mnd"] == 2


def test_implausible_values_are_repaired_by_kind():
    df = pd.DataFrame({
        "alder": [45.0, 141.0, 10.0],
        "innskudd": [100.0, -5000.0, 0.0],
        "laanebalanse": [0.0, 0.0, 0.0],
    })
    cleaned = clip_to_plausible(df)
    # impossible ages become missing rather than a made-up number
    assert cleaned["alder"].tolist()[0] == 45.0
    assert np.isnan(cleaned["alder"].tolist()[1]) and np.isnan(cleaned["alder"].tolist()[2])
    # a negative deposit is a booking error, clipped to the bound
    assert cleaned["innskudd"].tolist() == [100.0, 0.0, 0.0]


def test_relations_are_undirected_and_deduplicated():
    relasjoner = pd.DataFrame({
        "kunde_id_1": ["K1", "K2", "K3", "K9"],
        "kunde_id_2": ["K2", "K1", "K3", "K1"],
        "relasjon": ["Husstand", "husstand", "husstand", "medlaantaker"],
    })
    cleaned = clean_relasjoner(relasjoner, valid_ids={"K1", "K2", "K3"})
    assert len(cleaned) == 1  # K1-K2 kept once; self-pair and unknown id dropped
    assert cleaned.iloc[0].tolist() == ["K1", "K2", "husstand"]
