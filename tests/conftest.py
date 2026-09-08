import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw(cfg):
    from src.data.load import load_raw

    return load_raw(cfg)


@pytest.fixture(scope="session")
def clean(raw):
    from src.data.clean import clean_all

    return clean_all(raw)


def monthly_panel(customer_id: str, values: dict[str, float], metric: str) -> pd.DataFrame:
    """Build an engagement panel for one customer from {'YYYY-MM': value}."""
    other = "korttransaksjoner" if metric == "innlogginger" else "innlogginger"
    return pd.DataFrame({
        "kunde_id": customer_id,
        "maaned": pd.to_datetime(list(values), format="%Y-%m"),
        metric: list(values.values()),
        other: 0.0,
    })
