#test/test_value_estimatio.py
# Tests for Step 3
import pandas as pd
import numpy as np
import pytest

from src.value_estimatio import (
    _safe_norm,
    _weights_age_gender,
    estimate_core_metrics,
    estimate_event_table,
    drop_duplicate_columns,
    build_neighbors_index,   
)

# ---------- Tests for _safe_norm ----------

def test_safe_norm_basic():
    w = np.array([1, 1, 2], dtype=float)
    normed = _safe_norm(w)
    assert np.isclose(normed.sum(), 1.0)
    assert np.all(normed >= 0)

def test_safe_norm_zero_sum():
    w = np.array([0, 0, 0], dtype=float)
    normed = _safe_norm(w)
    assert np.allclose(normed, [1/3, 1/3, 1/3])

def test_safe_norm_invalid_sum():
    w = np.array([np.nan, np.nan])
    normed = _safe_norm(w)
    assert np.allclose(normed, [0.5, 0.5])

# ---------- Tests for _weights_age_gender ----------

@pytest.fixture
def cart_g_sample():
    return pd.DataFrame({"age": ["18-24", "25-34"], "gender": ["male", "female"]})

def test_weights_from_present_g(cart_g_sample):
    present_g = pd.DataFrame({
        "age": ["18-24", "25-34"],
        "gender": ["male", "female"],
        "impressions": [30, 70]
    })
    w = _weights_age_gender(cart_g_sample, present_g=present_g)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0]

def test_weights_from_fallback(cart_g_sample):
    fallback_g = pd.DataFrame({
        "age": ["18-24", "25-34"],
        "gender": ["male", "female"],
        "impressions": [10, 90]
    })
    w = _weights_age_gender(cart_g_sample, present_g=None, fallback_g=fallback_g)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0]

def test_weights_uniform(cart_g_sample):
    w = _weights_age_gender(cart_g_sample, present_g=None, fallback_g=None)
    assert np.allclose(w, [0.5, 0.5])

# ---------- Fixtures for core/events ----------

@pytest.fixture
def df_core_base():
    return pd.DataFrame([{
        "ad_id": 1, "objective": "sales", "age": np.nan, "gender": np.nan,
        "date_start": "2024-01-01", "date_stop": "2024-01-02",
        "impressions": 100, "clicks": 10, "spend": 50
    }])

@pytest.fixture
def cart_base():
    return pd.DataFrame([
        {"ad_id": 1, "objective": "sales", "age": "18-24", "gender": "male",
         "date_start": "2024-01-01", "date_stop": "2024-01-02"},
        {"ad_id": 1, "objective": "sales", "age": "25-34", "gender": "female",
         "date_start": "2024-01-01", "date_stop": "2024-01-02"},
    ])

# ---------- Tests for estimate_core_metrics ----------

def test_estimate_core_metrics_fill(df_core_base, cart_base):
    result = estimate_core_metrics(df_core_base, cart_base)
    assert "impressions" in result.columns
# Amount should be equal to the platform's Target (100)
    assert np.isclose(result["impressions"].sum(), 100)

def test_estimate_core_metrics_no_missing(df_core_base, cart_base):
# Add an Age×Gender split that adds up to 100
    df_full = pd.concat([
        df_core_base,
        pd.DataFrame([
            {"ad_id": 1, "objective": "sales", "age": "18-24", "gender": "male",
             "date_start": "2024-01-01", "date_stop": "2024-01-02",
             "impressions": 60, "clicks": 5, "spend": 25},
            {"ad_id": 1, "objective": "sales", "age": "25-34", "gender": "female",
             "date_start": "2024-01-01", "date_stop": "2024-01-02",
             "impressions": 40, "clicks": 5, "spend": 25}
        ])
    ], ignore_index=True)
    result = estimate_core_metrics(df_full, cart_base)
# Output only contains the Age×Gender level, so the sum should remain 100 (not 200)
    assert np.isclose(result["impressions"].sum(), 100)

def test_estimate_core_metrics_neighbors_fallback(df_core_base, cart_base):
# (ad_id=2) with 80/20 distribution
    core_ag_train = pd.DataFrame([
        {"ad_id": 2, "objective": "sales", "age": "18-24", "gender": "male",
         "date_start": "2024-01-01", "date_stop": "2024-01-02",
         "impressions": 80, "clicks": 0, "spend": 0},
        {"ad_id": 2, "objective": "sales", "age": "25-34", "gender": "female",
         "date_start": "2024-01-01", "date_stop": "2024-01-02",
         "impressions": 20, "clicks": 0, "spend": 0},
    ])
    neighbors_index = build_neighbors_index(core_ag_train, feature_cols=("impressions","clicks","spend"), dims=("objective",))
    result = estimate_core_metrics(
        df_core_base, cart_base,
        weighting_strategy="neighbors",
        neighbors_index=neighbors_index,
        neighbors_k=3
    )
    #  80/20
    by_age = result.set_index(["age","gender"])["impressions"].to_dict()
    assert by_age[("18-24","male")] > by_age[("25-34","female")]
    assert np.isclose(result["impressions"].sum(), 100)

# ---------- Tests for estimate_event_table ----------

@pytest.fixture
def df_event_base():
    return pd.DataFrame([{
        "ad_id": 1, "objective": "sales", "age": np.nan, "gender": np.nan,
        "date_start": "2024-01-01", "date_stop": "2024-01-02",
        "action_type": "view", "stat": "7d", "amount": 200
    }])

@pytest.fixture
def df_core_est_base(cart_base):
    return pd.DataFrame([
        {"ad_id": 1, "objective": "sales", "age": "18-24", "gender": "male",
         "date_start": "2024-01-01", "date_stop": "2024-01-02", "impressions": 50},
        {"ad_id": 1, "objective": "sales", "age": "25-34", "gender": "female",
         "date_start": "2024-01-01", "date_stop": "2024-01-02", "impressions": 50}
    ])

def test_estimate_event_table_fill(df_event_base, cart_base, df_core_est_base):
    result = estimate_event_table(df_event_base, cart_base, df_core_est_base)
    assert "amount" in result.columns
    assert np.isclose(result["amount"].sum(), 200)

def test_estimate_event_table_no_target(df_event_base, cart_base, df_core_est_base):
    df_event_empty = df_event_base.iloc[0:0]
    result = estimate_event_table(df_event_empty, cart_base, df_core_est_base)
    assert np.isclose(result["amount"].sum(), 0.0)
