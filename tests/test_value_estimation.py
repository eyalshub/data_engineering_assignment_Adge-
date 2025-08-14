import pandas as pd
import numpy as np
import pytest
from src.value_estimatio import estimate_values_sum_preserving 


@pytest.fixture
def df_original_sample():
    return pd.DataFrame([
        # ad 1 – wildcard row with missing age
        {"ad_id": 1, "objective": "reach", "publisher_platform": "facebook", "age": np.nan, "gender": "male",
         "impressions": 120, "clicks": 12, "spend": 6.0},

        # ad 1 – specific row (to bias weights)
        {"ad_id": 1, "objective": "reach", "publisher_platform": "facebook", "age": "18-24", "gender": "male",
         "impressions": 30, "clicks": 3, "spend": 1.0},

        # ad 2 – fully specified
        {"ad_id": 2, "objective": "traffic", "publisher_platform": "instagram", "age": "25-34", "gender": "female",
         "impressions": 200, "clicks": 20, "spend": 8.0},
    ])

@pytest.fixture
def cartesian_sample():
    return pd.DataFrame([
        {"ad_id": 1, "objective": "reach", "publisher_platform": "facebook", "age": "18-24", "gender": "male"},
        {"ad_id": 1, "objective": "reach", "publisher_platform": "facebook", "age": "25-34", "gender": "male"},
        {"ad_id": 2, "objective": "traffic", "publisher_platform": "instagram", "age": "25-34", "gender": "female"},
    ])

def test_estimation_returns_all_combinations(df_original_sample, cartesian_sample):
    df_est = estimate_values_sum_preserving(df_original_sample, cartesian_sample)
    assert len(df_est) == len(cartesian_sample)
    assert df_est[["age", "gender", "objective", "publisher_platform"]].isnull().sum().sum() == 0

def test_totals_preserved_per_ad(df_original_sample, cartesian_sample):
    df_est = estimate_values_sum_preserving(df_original_sample, cartesian_sample)
    metrics = ["impressions", "clicks", "spend"]
    for ad_id in df_original_sample["ad_id"].unique():
        orig = df_original_sample[df_original_sample["ad_id"] == ad_id][metrics].sum()
        est = df_est[df_est["ad_id"] == ad_id][metrics].sum()
        for m in metrics:
            assert np.isclose(orig[m], est[m]), f"Mismatch in {m} for ad_id={ad_id}"

def test_distribution_bias_from_existing_row(df_original_sample, cartesian_sample):
    df_est = estimate_values_sum_preserving(df_original_sample, cartesian_sample)
    ad1 = df_est[df_est["ad_id"] == 1].sort_values("age")
    imp_18 = ad1[ad1["age"] == "18-24"]["impressions"].values[0]
    imp_25 = ad1[ad1["age"] == "25-34"]["impressions"].values[0]
    assert imp_18 > imp_25  # should be biased toward the already-present row

def test_single_ad_without_wildcards():
    df = pd.DataFrame([
        {"ad_id": 3, "objective": "video", "publisher_platform": "facebook", "age": "35-44", "gender": "female",
         "impressions": 300, "clicks": 15, "spend": 12.0}
    ])
    cart = pd.DataFrame([
        {"ad_id": 3, "objective": "video", "publisher_platform": "facebook", "age": "35-44", "gender": "female"}
    ])
    df_est = estimate_values_sum_preserving(df, cart)
    assert df_est.iloc[0]["impressions"] == 300
    assert df_est.iloc[0]["clicks"] == 15
    assert df_est.iloc[0]["spend"] == 12.0
