# tests/test_data_loading.py
import pandas as pd
import pytest
import zipfile
from pathlib import Path

from src.data_loading import (
    build_central_ad_dict,
    build_core_table,
    build_event_tables,
    load_and_prepare_data,
    _add_metrics_to_central_ads,
)
from src.mappings import ACTION_MAPPING, ACTION_VALUES_MAPPING


# ------------------------
# Fixtures
# ------------------------
@pytest.fixture
def sample_dataframes():
    """
    Minimal test DataFrames for age_gender and platform sources.
    They include numeric metrics and JSON-like columns for actions.
    """
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
        "video_avg_time_watched_actions": None,
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
        "video_avg_time_watched_actions": None,
    }])

    return df_age_gender, df_platform


# ------------------------
# Helpers
# ------------------------
def create_test_zip(tmp_path: Path, df_age_gender: pd.DataFrame, df_platform: pd.DataFrame) -> Path:
    """
    Create a temporary ZIP file containing age_gender.csv and platform.csv.
    """
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
def test_build_central_ad_dict_flags_and_lists(sample_dataframes):
    """
    Verifies flags and that we preserve *all* rows per ad_id in lists (no overwrites).
    """
    df_age_gender, df_platform = sample_dataframes
    central = build_central_ad_dict(df_age_gender, df_platform)

    assert 1 in central
    assert central[1]["flags"]["in_both"] is True
    assert isinstance(central[1]["age_gender_rows"], list)
    assert isinstance(central[1]["platform_rows"], list)
    assert len(central[1]["age_gender_rows"]) == 1
    assert len(central[1]["platform_rows"]) == 1


def test_central_dict_preserves_all_rows():
    """
    If there are multiple rows per ad in a source, they should all be preserved (appended).
    """
    df_age_gender = pd.DataFrame([
        {"ad_id": 1, "age": "18-24", "gender": "male",   "impressions": 10, "clicks": 1, "spend": 1.0, "frequency": 1.2},
        {"ad_id": 1, "age": "25-34", "gender": "female", "impressions": 20, "clicks": 2, "spend": 2.0, "frequency": 1.3},
    ])
    df_platform = pd.DataFrame([
        {"ad_id": 1, "publisher_platform": "facebook",  "objective": "reach", "impressions": 30, "clicks": 3, "spend": 3.0, "frequency": 1.1},
        {"ad_id": 1, "publisher_platform": "instagram", "objective": "reach", "impressions": 40, "clicks": 4, "spend": 4.0, "frequency": 1.0},
    ])
    central = build_central_ad_dict(df_age_gender, df_platform)
    assert len(central[1]["age_gender_rows"]) == 2
    assert len(central[1]["platform_rows"]) == 2


def test_add_metrics_to_central_ads_categories_only(sample_dataframes):
    """
    Only categorical dims should be used as categories; dates belong to meta.
    """
    df_age_gender, _ = sample_dataframes
    central_ads = {1: {}}
    action_lookup = {v.strip().lower(): k for k, vals in ACTION_MAPPING.items() for v in vals}

    _add_metrics_to_central_ads(
        central_ads,
        df_age_gender,
        id_col="ad_id",
        json_col="actions",
        category_cols=["age", "gender"],       # categories only
        meta_cols=["date_start", "date_stop"], # dates go to meta
        target_key="age_gender_actions",
        mapping_lookup=action_lookup,
    )

    recs = central_ads[1]["age_gender_actions"]
    assert len(recs) > 0
    assert "categories" in recs[0] and "meta" in recs[0]
    assert "date_start" not in recs[0]["categories"]
    assert "date_start" in recs[0]["meta"]


def test_core_schema_and_sums(sample_dataframes):
    """
    Core DF must be flat (no dates/JSON), with expected columns; sums should match input sums.
    """
    df_age_gender, df_platform = sample_dataframes
    central = build_central_ad_dict(df_age_gender, df_platform)
    df_core = build_core_table(central)

    forbidden = {"date_start","date_stop","actions","action_values",
                 "video_play_actions","video_thruplay_watched_actions","video_avg_time_watched_actions"}
    assert forbidden.isdisjoint(df_core.columns)

    expected_cols = {"ad_id","objective","publisher_platform","age","gender",
                     "impressions","clicks","spend","frequency","source"}
    assert expected_cols.issubset(df_core.columns)

    assert df_core["impressions"].sum() == 300
    assert df_core["clicks"].sum() == 30
    assert df_core["spend"].sum() == 15.0


def test_core_grouping_by_dims_not_dates():
    """
    Grouping must be by categorical dims only (no dates in the key).
    """
    df_age_gender = pd.DataFrame([
        {"ad_id": 2, "age": "18-24", "gender": "male",   "impressions": 10, "clicks": 1, "spend": 1.0, "frequency": 1.2,
         "date_start":"2024-01-01","date_stop":"2024-01-02"},
        {"ad_id": 2, "age": "25-34", "gender": "female", "impressions": 20, "clicks": 2, "spend": 2.0, "frequency": 1.3,
         "date_start":"2024-01-05","date_stop":"2024-01-06"},
    ])
    df_platform = pd.DataFrame([
        {"ad_id": 2, "publisher_platform":"facebook", "objective":"reach", "impressions": 30, "clicks": 3, "spend": 3.0, "frequency": 1.1,
         "date_start":"2024-01-01","date_stop":"2024-01-02"},
    ])

    central = build_central_ad_dict(df_age_gender, df_platform)
    df_core = build_core_table(central)

    male = df_core[(df_core.ad_id==2) & (df_core.source=="age_gender") & (df_core.age=="18-24") & (df_core.gender=="male")]
    female = df_core[(df_core.ad_id==2) & (df_core.source=="age_gender") & (df_core.age=="25-34") & (df_core.gender=="female")]
    assert float(male["impressions"].sum()) == 10.0
    assert float(female["impressions"].sum()) == 20.0

    plat = df_core[(df_core.ad_id==2) & (df_core.source=="platform") & (df_core.publisher_platform=="facebook")]
    assert float(plat["impressions"].sum()) == 30.0


def test_load_and_prepare_data_from_csv(tmp_path, sample_dataframes):
    """
    Full pipeline from CSV files: core sums and event tables schema.
    """
    df_age_gender, df_platform = sample_dataframes

    age_gender_path = tmp_path / "age_gender.csv"
    platform_path   = tmp_path / "platform.csv"
    df_age_gender.to_csv(age_gender_path, index=False)
    df_platform.to_csv(platform_path, index=False)

    central_ads, df_core, event_tables = load_and_prepare_data(age_gender_path, platform_path)

    assert df_core["impressions"].sum() == 300

    # Event tables should exist and have the required schema, even if some are empty.
    required_cols = {"ad_id","objective","publisher_platform","age","gender","action_type","stat","amount"}
    for df in event_tables:
        assert isinstance(df, pd.DataFrame)
        assert required_cols.issubset(df.columns)


def test_load_and_prepare_data_from_zip(tmp_path, sample_dataframes):
    """
    Full pipeline from a ZIP file (age_gender.csv + platform.csv inside).
    """
    zip_path = create_test_zip(tmp_path, *sample_dataframes)
    central_ads, df_core, event_tables = load_and_prepare_data(zip_path, zip_path)

    assert df_core["impressions"].sum() == 300

    required_cols = {"ad_id","objective","publisher_platform","age","gender","action_type","stat","amount"}
    for df in event_tables:
        assert isinstance(df, pd.DataFrame)
        assert required_cols.issubset(df.columns)


def test_event_tables_schema_and_aggregation(sample_dataframes):
    """
    Event tables must be long-format with the right keys, and aggregate JSON values correctly.
    """
    df_age_gender, df_platform = sample_dataframes

    # Build central dict and attach actions (categories only; dates in meta)
    central = build_central_ad_dict(df_age_gender, df_platform)

    action_lookup = {v.strip().lower(): k for k, vals in ACTION_MAPPING.items() for v in vals}
    _add_metrics_to_central_ads(
        central, df_age_gender, "ad_id", "actions",
        category_cols=["age","gender"], meta_cols=["date_start","date_stop"],
        target_key="age_gender_actions", mapping_lookup=action_lookup
    )
    _add_metrics_to_central_ads(
        central, df_platform, "ad_id", "actions",
        category_cols=["publisher_platform","objective"], meta_cols=["date_start","date_stop"],
        target_key="platform_actions", mapping_lookup=action_lookup
    )

    events = build_event_tables(central)
    actions_df = events[0]  # actions

    # Helper to fetch aggregated amounts
    def amt(df, action_type, stat):
        m = (df["action_type"] == action_type) & (df["stat"] == stat)
        return float(df.loc[m, "amount"].sum())

    # From fixtures: view_content(value=3) and add_to_cart(value=5)
    assert amt(actions_df, "view_content", "value") == 3.0
    assert amt(actions_df, "add_to_cart", "value") == 5.0

    # Dates must not be part of the schema
    for col in ["date_start","date_stop"]:
        assert col not in actions_df.columns
