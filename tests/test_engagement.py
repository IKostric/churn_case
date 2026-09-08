import pandas as pd
import pytest

from src.features.engagement import build_engagement_features
from tests.conftest import monthly_panel

PERIOD = {"period_start": "2024-01", "period_end": "2025-12", "epsilon": 1.0}
EXPECTED_FEATURES = {
    "card_last_month",
    "card_mean_last_12m",
    "card_q4_yoy_pct",
    "login_last_month",
    "login_q4_yoy_pct",
}


def _features(values: dict[str, float], metric: str = "innlogginger") -> pd.Series:
    return build_engagement_features(monthly_panel("K1", values, metric), **PERIOD).iloc[0]


def _months(year: int) -> list[str]:
    return [f"{year}-{month:02d}" for month in range(1, 13)]


def test_feature_set_is_exactly_the_five_chosen_features():
    values = dict.fromkeys(_months(2024) + _months(2025), 1.0)
    features = _features(values)
    assert set(features.index) - {"kunde_id"} == EXPECTED_FEATURES


def test_features_match_hand_computed_values():
    values = dict.fromkeys(_months(2024) + _months(2025), 10.0)
    values.update({"2024-10": 20.0, "2024-11": 20.0, "2024-12": 20.0})
    values.update({"2025-10": 10.0, "2025-11": 0.0, "2025-12": 5.0})

    features = _features(values)
    assert features["login_last_month"] == 5.0
    # last quarter mean (10+0+5)/3 = 5 vs the same quarter a year earlier (20)
    assert features["login_q4_yoy_pct"] == pytest.approx((5.0 - 20.0) / (20.0 + 1.0))


def test_pure_seasonality_produces_no_yoy_change():
    """A customer repeating the same seasonal shape has not disengaged.

    Comparing the window's start to its end would report a large decline here
    purely because Q1 is a seasonal peak and Q4 a seasonal trough; the
    year-over-year comparison cancels it.
    """
    seasonal = {1: 20.0, 2: 20.0, 3: 20.0, 4: 18.0, 5: 15.0, 6: 14.0,
                7: 13.0, 8: 12.0, 9: 11.0, 10: 10.0, 11: 10.0, 12: 10.0}
    values = {f"{year}-{month:02d}": value
              for year in (2024, 2025) for month, value in seasonal.items()}

    features = _features(values)
    assert features["login_q4_yoy_pct"] == pytest.approx(0.0)
    # a within-window comparison would have read -10 for this same customer
    assert features["login_last_month"] - 20.0 == pytest.approx(-10.0)


def test_level_contrast_is_available_from_the_kept_features():
    """`last_month` vs `mean_last_12m` is the fast signal; both must be present
    and on the same scale for the contrast to be learnable."""
    values = {m: 10.0 for m in _months(2024) + _months(2025)}
    values["2025-12"] = 2.0
    features = _features(values, metric="korttransaksjoner")
    assert features["card_last_month"] == 2.0
    assert features["card_mean_last_12m"] == pytest.approx((10.0 * 11 + 2.0) / 12)
    assert features["card_last_month"] < features["card_mean_last_12m"]


def test_percentage_change_is_safe_when_prior_year_is_zero():
    values = dict.fromkeys(_months(2024), 0.0) | dict.fromkeys(_months(2025), 5.0)
    features = _features(values)
    assert features["login_q4_yoy_pct"] == pytest.approx(5.0 / 1.0)
    assert features.notna().all()


def test_change_features_cover_both_metrics_but_level_baseline_is_single():
    values = dict.fromkeys(_months(2024), 1.0) | dict.fromkeys(_months(2025), 2.0)
    features = _features(values, metric="korttransaksjoner")
    assert {"card_q4_yoy_pct", "login_q4_yoy_pct"} <= set(features.index)
    assert {"card_last_month", "login_last_month"} <= set(features.index)
    # the two metrics' 12-month means are r=0.98, so only one baseline is kept
    assert "login_mean_last_12m" not in features.index
