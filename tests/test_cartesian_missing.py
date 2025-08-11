# data_engineering_assignment/tests/test_cartesian_missing.py
import pandas as pd
import numpy as np
import pytest

# Adjust import according to actual filename/location
from src.cartesian_missing import (
    normalize_keys,
    build_cartesian_dates_as_pairs,
    find_missing_flexible_per_ad,
)

# ---------- Fixtures ----------

@pytest.fixture
def sample_core_single_period():
    """Single ad_id with one period and fixed categorical values."""
    return pd.DataFrame([
        {
            "ad_id": 1,
            "objective": "sales",
            "age": "18-24",
            "gender": "male",
            "date_start": "2024-01-01",
            "date_stop": "2024-01-02",
        }
    ])

@pytest.fixture
def sample_event_single_match(sample_core_single_period):
    """Event DataFrame identical to core – ensures no missing combinations."""
    return sample_core_single_period.copy()

@pytest.fixture
def sample_core_multi_period_multi_dims():
    """
    Single ad_id with:
    - Two distinct time periods
    - Two different (age, gender) combinations
    """
    rows = [
        {"ad_id": 10, "objective": "sales", "age": "18-24", "gender": "male",
         "date_start": "2024-01-01", "date_stop": "2024-01-02"},
        {"ad_id": 10, "objective": "sales", "age": "25-34", "gender": "female",
         "date_start": "2024-01-03", "date_stop": "2024-01-04"},
    ]
    return pd.DataFrame(rows)

# ---------- Tests: normalize_keys ----------

def test_normalize_keys_types_and_dates():
    """
    Ensures normalize_keys:
    - Converts categorical fields to StringDtype
    - Replaces empty/'None'/'nan' with pd.NA
    - Formats valid dates as YYYY-MM-DD
    - Sets invalid dates to pd.NA
    """
    df = pd.DataFrame([
        {"objective": "SALES", "age": "18-24", "gender": "male",
         "date_start": "2024/01/01", "date_stop": "2024-01-02"},
        {"objective": "", "age": "None", "gender": "nan",
         "date_start": "not-a-date", "date_stop": None},
    ])

    out = normalize_keys(df)

    # Check categorical conversions and NA replacement
    for c in ["objective", "age", "gender"]:
        assert str(out[c].dtype) == "string"
        assert out[c].isna().sum() >= 1

    # Check date formatting and NA for invalid dates
    assert out.loc[0, "date_start"] == "2024-01-01"
    assert out.loc[0, "date_stop"] == "2024-01-02"
    assert pd.isna(out.loc[1, "date_start"])
    assert pd.isna(out.loc[1, "date_stop"])

# ---------- Tests: cartesian ----------

def test_build_cartesian_dates_as_pairs_basic(sample_core_single_period, sample_event_single_match):
    """
    Basic case:
    - One ad_id
    - One period
    - One value per categorical dimension
    Should return at least one row with all required columns.
    """
    cart = build_cartesian_dates_as_pairs(
        df_core=sample_core_single_period,
        event_tables=[sample_event_single_match],
        dims=["objective", "age", "gender"],
    )
    assert not cart.empty
    assert set(["ad_id", "date_start", "date_stop", "objective", "age", "gender"]).issubset(cart.columns)
    assert len(cart) >= 1

def test_build_cartesian_handles_missing_dims():
    """
    Case where core table has no age/gender columns:
    - Function should insert NA as default values for those dimensions.
    """
    df_core = pd.DataFrame([
        {"ad_id": 2, "objective": "reach", "date_start": "2024-01-01", "date_stop": "2024-01-02"}
    ])
    cart = build_cartesian_dates_as_pairs(df_core, event_tables=None, dims=["objective", "age", "gender"])
    row = cart.iloc[0]
    assert pd.isna(row["age"]) and pd.isna(row["gender"])
    assert row["objective"] == "reach"

def test_build_cartesian_multiple_periods(sample_core_multi_period_multi_dims):
    """
    Case with multiple time periods:
    - Ensure cartesian output contains at least as many unique periods as in core.
    """
    cart = build_cartesian_dates_as_pairs(
        sample_core_multi_period_multi_dims,
        event_tables=None,
        dims=["objective", "age", "gender"],
    )
    periods = cart[["date_start", "date_stop"]].drop_duplicates()
    assert len(periods) >= 2

# ---------- Tests: missing-combo detection ----------

def test_find_missing_flexible_per_ad_no_missing(sample_core_single_period, sample_event_single_match):
    """If events match cartesian combinations exactly – result should be empty."""
    cart = build_cartesian_dates_as_pairs(sample_core_single_period, [sample_event_single_match], ["objective","age","gender"])
    missing = find_missing_flexible_per_ad(
        cart,
        sample_event_single_match,
        ["objective","age","gender","date_start","date_stop"]
    )
    assert missing.empty

def test_find_missing_flexible_per_ad_all_missing(sample_core_single_period):
    """If no events exist – all cartesian combinations should be missing."""
    empty_events = pd.DataFrame(columns=sample_core_single_period.columns)
    cart = build_cartesian_dates_as_pairs(sample_core_single_period, [], ["objective","age","gender"])
    missing = find_missing_flexible_per_ad(
        cart, empty_events, ["objective","age","gender","date_start","date_stop"]
    )
    assert not missing.empty
    assert len(missing) == len(cart.drop_duplicates(subset=["ad_id","objective","age","gender","date_start","date_stop"]))

def test_find_missing_wildcard_age(sample_core_multi_period_multi_dims):
    """
    If event rows have NA for age/gender:
    - Should be treated as wildcard and cover all combinations for those dimensions.
    """
    ev = sample_core_multi_period_multi_dims.copy()
    ev["age"] = pd.NA
    ev["gender"] = pd.NA

    cart = build_cartesian_dates_as_pairs(
        sample_core_multi_period_multi_dims, [ev], ["objective","age","gender"]
    )

    missing = find_missing_flexible_per_ad(
        cart, ev, ["objective","age","gender","date_start","date_stop"]
    )

    assert missing.empty

def test_find_missing_partial_coverage(sample_core_multi_period_multi_dims):
    """
    If events cover only one period:
    - Missing combinations should correspond to the uncovered period.
    """
    df_core = sample_core_multi_period_multi_dims
    ev = df_core[df_core["date_start"] == "2024-01-01"].copy()

    cart = build_cartesian_dates_as_pairs(df_core, [ev], ["objective","age","gender"])
    missing = find_missing_flexible_per_ad(
        cart, ev, ["objective","age","gender","date_start","date_stop"]
    )

    second_period = df_core[df_core["date_start"] == "2024-01-03"][["objective","age","gender","date_start","date_stop"]]
    merged = missing.merge(second_period, on=["objective","age","gender","date_start","date_stop"], how="inner")
    assert not merged.empty
