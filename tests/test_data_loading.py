# tests/test_data_loading.py
import pandas as pd
import pytest
import zipfile
from src.data_loading import (
    build_central_ad_dict,
    load_and_prepare_data,
    build_event_tables,
    _add_metrics_to_central_ads
)
from src.mappings import ACTION_MAPPING, ACTION_VALUES_MAPPING


# ------------------------
# Fixtures
# ------------------------
@pytest.fixture
def sample_dataframes():
    """Return minimal test DataFrames for age_gender and platform"""
    df_age_gender = pd.DataFrame([{
        "ad_id": 1,
        "age": "18-24",
        "gender": "male",
        "date_start": "2024-01-01",
        "date_stop": "2024-01-02",
        "impressions": 100,
        "clicks": 10,
        "spend": 5.0,
        "frequency": 1.5,
        "actions": '[{"action_type": "view_content", "value": 3}]',
        "action_values": '[{"action_type": "purchase", "value": 2}]',
        "video_play_actions": '[{"action_type": "video_view", "value": 4, "7d_click": 1}]',
        "video_thruplay_watched_actions": None,
        "video_avg_time_watched_actions": None
    }])

    df_platform = pd.DataFrame([{
        "ad_id": 1,
        "publisher_platform": "facebook",
        "objective": "conversions",
        "date_start": "2024-01-01",
        "date_stop": "2024-01-02",
        "impressions": 200,
        "clicks": 20,
        "spend": 10.0,
        "frequency": 2.0,
        "actions": '[{"action_type": "add_to_cart", "value": 5}]',
        "action_values": '[{"action_type": "purchase", "value": 1}]',
        "video_play_actions": None,
        "video_thruplay_watched_actions": None,
        "video_avg_time_watched_actions": None
    }])

    return df_age_gender, df_platform


# ------------------------
# Helper functions
# ------------------------
def create_test_zip(tmp_path, df_age_gender, df_platform):
    """Create a temporary ZIP file containing age_gender.csv and platform.csv"""
    age_gender_csv = tmp_path / "age_gender.csv"
    platform_csv = tmp_path / "platform.csv"
    df_age_gender.to_csv(age_gender_csv, index=False)
    df_platform.to_csv(platform_csv, index=False)

    zip_path = tmp_path / "test_data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(age_gender_csv, arcname="age_gender.csv")
        zf.write(platform_csv, arcname="platform.csv")

    return zip_path


# ------------------------
# Tests
# ------------------------
def test_build_central_ad_dict(sample_dataframes):
    df_age_gender, df_platform = sample_dataframes
    central_dict = build_central_ad_dict(df_age_gender, df_platform)
    assert 1 in central_dict
    assert central_dict[1]["flags"]["in_both"] is True


def test_add_metrics_to_central_ads(sample_dataframes):
    df_age_gender, _ = sample_dataframes
    central_ads = {1: {}}
    action_lookup = {v.strip().lower(): k for k, vals in ACTION_MAPPING.items() for v in vals}

    _add_metrics_to_central_ads(
        central_ads,
        df_age_gender,
        "ad_id",
        "actions",
        ["age", "gender", "date_start", "date_stop"],
        "age_gender_actions",
        mapping_lookup=action_lookup
    )

    assert "age_gender_actions" in central_ads[1]
    assert len(central_ads[1]["age_gender_actions"]) > 0


def test_load_and_prepare_data_from_csv(tmp_path, sample_dataframes):
    df_age_gender, df_platform = sample_dataframes

    age_gender_path = tmp_path / "age_gender.csv"
    platform_path = tmp_path / "platform.csv"
    df_age_gender.to_csv(age_gender_path, index=False)
    df_platform.to_csv(platform_path, index=False)

    central_ads, df_core, event_tables = load_and_prepare_data(age_gender_path, platform_path)

    # Core table correctness
    assert df_core["impressions"].sum() == 300
    # Event tables integrity
    for df in event_tables:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


def test_load_and_prepare_data_from_zip(tmp_path, sample_dataframes):
    zip_path = create_test_zip(tmp_path, *sample_dataframes)
    central_ads, df_core, event_tables = load_and_prepare_data(zip_path, zip_path)

    assert df_core["impressions"].sum() == 300
    for df in event_tables:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


def test_build_event_tables_from_zip(tmp_path, sample_dataframes):
    zip_path = create_test_zip(tmp_path, *sample_dataframes)
    central_ads, _, _ = load_and_prepare_data(zip_path, zip_path)
    event_tables = build_event_tables(central_ads)

    assert len(event_tables) == 5
    for df in event_tables:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
