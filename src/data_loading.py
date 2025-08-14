# src/data_loading.py
import os
import zipfile
import json
import ast
import logging
import pandas as pd
from .mappings import ACTION_MAPPING, ACTION_VALUES_MAPPING
import tempfile
from pathlib import Path
# ----------------------
# Logging Configuration
# ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# =========================================
#  Utilities
# =========================================
def extract_zip(zip_path, extract_to="data"):
    """Extracts a zip file to a target folder."""
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f"Extracted {zip_path} to {extract_to}")

def ensure_columns_exist(df, cols):
    """Ensure required columns exist in the DataFrame."""
    for c in cols:
        if c not in df.columns:
            logging.warning(f"Column '{c}' missing, adding as empty.")
            df[c] = pd.NA
    return df

def safe_parse_actions(raw_actions):
    """Safely parse JSON-like action strings into a Python list."""
    if pd.isna(raw_actions):
        return []
    if isinstance(raw_actions, list):
        return raw_actions
    if isinstance(raw_actions, str):
        s = raw_actions.strip()
        if not s or s in ("[]", "{}", "nan", "None", "null"):
            return []
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(s.replace("'", '"'))
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception:
            return []
    return []

def dict_or_empty(x):
    return x if isinstance(x, dict) else {}

# =========================================
#  Step 1: Build Central Ads Dictionary
# =========================================
def build_central_ad_dict(df_age_gender, df_platform, id_col="ad_id"):
    """Build central ads dictionary with flags for each ad_id."""
    ids_age_gender = set(df_age_gender[id_col].unique())
    ids_platform = set(df_platform[id_col].unique())
    all_ids = ids_age_gender.union(ids_platform)

    central_dict = {
        ad_id: {
            "age_gender_rows": [],
            "platform_rows": [],
            "flags": {
                "has_age_gender": ad_id in ids_age_gender,
                "has_platform": ad_id in ids_platform,
                "in_both": ad_id in ids_age_gender and ad_id in ids_platform
            }
        } for ad_id in all_ids
    }

    for _, row in df_age_gender.iterrows():
        central_dict[row[id_col]]["age_gender_rows"].append(row.to_dict())

    for _, row in df_platform.iterrows():
        central_dict[row[id_col]]["platform_rows"].append(row.to_dict())

    logging.info(f"Central dict built with {len(central_dict)} ads  (rows preserved).")
    return central_dict

# =========================================
#  Step 2: Generic add_* function
# =========================================

def _add_metrics_to_central_ads(
    central_ads, df, id_col, json_col, category_cols, target_key,
    mapping_lookup=None, default_values=None, meta_cols=None
):
    df = ensure_columns_exist(df, [id_col, json_col] + category_cols + (meta_cols or []))

    for _, row in df.iterrows():
        parsed_list = safe_parse_actions(row[json_col])
        if not parsed_list and default_values:
            parsed_list = [default_values.copy()]

        # normalize numbers in JSON
        for act in parsed_list:
            for key in act:
                if key != "action_type":
                    try:
                        act[key] = float(act[key])
                    except (ValueError, TypeError):
                        act[key] = 0.0

        ad_id = row[id_col]
        entry = central_ads.setdefault(ad_id, {})  # <-- don't overwrite the whole entry
        entry.setdefault(target_key, [])           # <-- ensure list exists

        categories = {col: row[col] for col in category_cols if col in df.columns}
        meta = {col: row[col] for col in (meta_cols or []) if col in df.columns}

        actions_dict = {}
        for act in parsed_list:
            raw_action_type = str(act.get("action_type", "")).strip()
            if not raw_action_type:
                continue
            action_type = mapping_lookup.get(raw_action_type.lower(), raw_action_type) if mapping_lookup else raw_action_type
            actions_dict.setdefault(action_type, {})
            for key, val in act.items():
                if key != "action_type":
                    actions_dict[action_type][key] = actions_dict[action_type].get(key, 0.0) + val

        if actions_dict:
            entry[target_key].append({
                "categories": categories,  # ONLY categorical dims
                "meta": meta,              # dates etc. (not used as dims)
                "actions": actions_dict
            })


# =========================================
#  Step 3: Build Final Tables
# =========================================
def build_core_table(master_dict):
    """
    Create a flat table from preserved rows, with categorical dims + numeric metrics only.
    Dates are intentionally excluded from the grouping keys (not dims).
    """
    rows = []

    def _append_row(src_row, source_label):
        if not src_row:
            return
        rows.append({
            "ad_id": src_row.get("ad_id"),
            "objective": src_row.get("objective"),
            "publisher_platform": src_row.get("publisher_platform"),
            "age": src_row.get("age"),
            "gender": src_row.get("gender"),
            "impressions": float(src_row.get("impressions") or 0),
            "clicks": float(src_row.get("clicks") or 0),
            "spend": float(src_row.get("spend") or 0),
            "frequency": float(src_row.get("frequency") or 0),
            "source": source_label,
        })

    for ad_id_key, rec in master_dict.items():
        for r in rec.get("platform_rows", []):
            _append_row(r, "platform")
        for r in rec.get("age_gender_rows", []):
            _append_row(r, "age_gender")

    df_core = pd.DataFrame(rows)

    # Normalize categorical types & strip whitespace
    for c in ["objective", "publisher_platform", "age", "gender", "source"]:
        if c in df_core.columns:
            df_core[c] = df_core[c].astype("string").str.strip()
            df_core.loc[df_core[c].isin(["", "nan", "None"]), c] = pd.NA
            df_core[c] = df_core[c].astype("category")

    # Group ONLY by categorical dims (no dates)
    key_cols = ["ad_id", "objective", "publisher_platform", "age", "gender", "source"]
    df_core = df_core.groupby(key_cols, dropna=False, as_index=False)[
        ["impressions", "clicks", "spend", "frequency"]
    ].sum()

    logging.info(f"Core table built (flat) with {len(df_core)} rows, cols={list(df_core.columns)}")
    return df_core

def build_event_tables(master_dict):
    """
    Normalize JSON-derived metrics into long-format event tables.
    Keys: ad_id + categorical dims + action_type + stat
    NOTE: dates exist only under 'meta' and are NOT grouping keys.
    """
    event_names = [
        "actions",
        "action_values",
        "video_play_actions",
        "video_thruplay_watched_actions",
        "video_avg_time_watched_actions",
    ]
    buckets = {name: [] for name in event_names}
    key_cols = ["ad_id", "objective", "publisher_platform", "age", "gender", "action_type", "stat"]

    def _emit(scope_name, name, ev_list, ad_id):
        if not ev_list:
            return
        for ev in ev_list:
            cats = ev.get("categories", {}) or {}
            acts = ev.get("actions", {}) or {}
            # meta = ev.get("meta", {})  # available if you want to carry dates externally
            for action_type, stats in acts.items():
                for stat_name, val in stats.items():
                    buckets[name].append({
                        "ad_id": ad_id,
                        "objective": cats.get("objective"),
                        "publisher_platform": cats.get("publisher_platform"),
                        "age": cats.get("age"),
                        "gender": cats.get("gender"),
                        "action_type": action_type,
                        "stat": stat_name,
                        "amount": float(val or 0.0),
                        "source_scope": scope_name,  # optional, useful for QA
                    })

    for ad_id_key, rec in master_dict.items():
        # for each scope ("platform", "age_gender") gather respective event arrays if exist
        for scope in ["platform", "age_gender"]:
            for name in event_names:
                key = f"{scope}_{name}"
                if key in rec:
                    _emit(scope, name, rec.get(key, []), ad_id_key)

    dfs = {}
    for name, rows in buckets.items():
        df = pd.DataFrame(rows)
        if not df.empty:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
            df = df.groupby(key_cols, dropna=False, as_index=False)["amount"].sum()
        else:
            # Ensure consistent empty frame schema
            df = pd.DataFrame(columns=key_cols + ["amount"])
        dfs[name] = df

    logging.info("Event tables built (normalized, no dates as dims).")
    # Return in a stable order
    return tuple(dfs[name] for name in event_names)
# =========================================
#  Main wrapper function
# =========================================
def load_and_prepare_data(age_gender_path, platform_path):
    """
    Main function to run Step 1–5: 
    Loads CSV/ZIP data, builds central dictionary, and returns core & event tables.
    """

    def prepare_path(path, contains=None):
        """Return CSV path, extracting from ZIP if needed."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.suffix.lower() == ".zip":
            tmp_dir = Path(tempfile.mkdtemp())
            extract_zip(path, tmp_dir)
            csv_files = list(tmp_dir.glob("*.csv"))
            
            if contains:
                csv_files = [f for f in csv_files if contains.lower() in f.name.lower()]
            
            if len(csv_files) == 1:
                return csv_files[0]
            elif not csv_files:
                raise ValueError(f"No matching CSV found in {path} with filter '{contains}'")
            else:
                raise ValueError(f"Multiple CSVs found in {path}: {csv_files}")

        if path.suffix.lower() != ".csv":
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return path

    # Prepare paths (supports CSV or ZIP)
    age_gender_path = prepare_path(age_gender_path, contains="age_gender")
    platform_path   = prepare_path(platform_path, contains="platform")

    # Load CSVs
    logger.info(f"Loading age_gender CSV: {age_gender_path}")
    df_age_gender = pd.read_csv(age_gender_path)

    logger.info(f"Loading platform CSV: {platform_path}")
    df_platform = pd.read_csv(platform_path)

    # Build initial central ads dict
    central_ads = build_central_ad_dict(df_age_gender, df_platform)
    logger.info(f"Central ads dictionary initialized with {len(central_ads)} ads")

    # Lookup mappings
    ACTION_LOOKUP = {v.strip().lower(): k for k, vals in ACTION_MAPPING.items() for v in vals}
    ACTION_VALUES_LOOKUP = {v.strip().lower(): k for k, vals in ACTION_VALUES_MAPPING.items() for v in vals}

    # Add actions
    # --- actions ---
    _add_metrics_to_central_ads(central_ads, df_age_gender, "ad_id", "actions",
                                category_cols=["age", "gender"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="age_gender_actions",
                                mapping_lookup=ACTION_LOOKUP)

    _add_metrics_to_central_ads(central_ads, df_platform, "ad_id", "actions",
                                category_cols=["publisher_platform", "objective"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="platform_actions",
                                mapping_lookup=ACTION_LOOKUP)

    # --- action_values ---
    _add_metrics_to_central_ads(central_ads, df_age_gender, "ad_id", "action_values",
                                category_cols=["age", "gender"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="age_gender_action_values",
                                mapping_lookup=ACTION_VALUES_LOOKUP)

    _add_metrics_to_central_ads(central_ads, df_platform, "ad_id", "action_values",
                                category_cols=["publisher_platform", "objective"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="platform_action_values",
                                mapping_lookup=ACTION_VALUES_LOOKUP)

    # --- video_* (עם defaults כמו אצלך) ---
    video_defaults = {"action_type": "video_view", "value": 0.0, "7d_click": 0.0}

    _add_metrics_to_central_ads(central_ads, df_age_gender, "ad_id", "video_play_actions",
                                category_cols=["age", "gender"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="age_gender_video_play_actions",
                                default_values=video_defaults)

    _add_metrics_to_central_ads(central_ads, df_platform, "ad_id", "video_play_actions",
                                category_cols=["publisher_platform", "objective"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="platform_video_play_actions",
                                default_values=video_defaults)

    _add_metrics_to_central_ads(central_ads, df_age_gender, "ad_id", "video_thruplay_watched_actions",
                                category_cols=["age", "gender"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="age_gender_video_thruplay_watched_actions",
                                default_values=video_defaults)

    _add_metrics_to_central_ads(central_ads, df_platform, "ad_id", "video_thruplay_watched_actions",
                                category_cols=["publisher_platform", "objective"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="platform_video_thruplay_watched_actions",
                                default_values=video_defaults)

    _add_metrics_to_central_ads(central_ads, df_age_gender, "ad_id", "video_avg_time_watched_actions",
                                category_cols=["age", "gender"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="age_gender_video_avg_time_watched_actions",
                                default_values=video_defaults)

    _add_metrics_to_central_ads(central_ads, df_platform, "ad_id", "video_avg_time_watched_actions",
                                category_cols=["publisher_platform", "objective"],
                                meta_cols=["date_start", "date_stop"],
                                target_key="platform_video_avg_time_watched_actions",
                                default_values=video_defaults)

    # Final validation
    if not central_ads:
        logger.error("Central ads dictionary is empty after data loading!")
        raise ValueError("No ads found after processing.")

    # Build tables
    df_core = build_core_table(central_ads)
    event_tables = build_event_tables(central_ads)

    logger.info("Data preparation completed successfully.")
    return central_ads, df_core, event_tables


def load_and_unify_from_csv(core_csv, second_csv):
    _, df_core, event_tables = load_and_prepare_data(core_csv, second_csv)
    return {"core": df_core, "events": event_tables[0] if event_tables else None}



# src/data_loading.py
from pathlib import Path

EVENT_NAMES = [
    "actions",
    "action_values",
    "video_play_actions",
    "video_thruplay_watched_actions",
    "video_avg_time_watched_actions",
]

def save_stage1_outputs(df_core: pd.DataFrame, event_tables, out_dir="artifacts"):
    """
    Save Stage 1 outputs (core and event tables) as CSVs for external analysis (e.g., Colab).
    Returns a dict of file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    core_path = out_dir / "stage1_core.csv"
    df_core.to_csv(core_path, index=False)
    paths["core"] = str(core_path)

    for name, df in zip(EVENT_NAMES, event_tables):
        p = out_dir / f"stage1_events_{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = str(p)

    logger.info(f"Saved Stage 1 outputs: {paths}")
    return paths
