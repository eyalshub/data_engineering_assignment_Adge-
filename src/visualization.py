# Step 5: Visualization & Analytics
# ---------------------------------------------------------------------
import os
from typing import Sequence, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If drop_duplicate_columns is defined in Step 3 module, import it.
# Fallback to a local safe version if the import path differs.
try:
    from src.value_estimatio import drop_duplicate_columns  # adjust if your path is different
except Exception:
    def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Safe fallback: remove duplicate-named columns created by merges."""
        return df.loc[:, ~df.columns.duplicated()].copy()


# -----------------------------
# Small helpers
# -----------------------------
def _age_bucket_sort_key(series: pd.Series) -> pd.Series:
    """
    Map age bucket strings to a numeric sort key.
    Examples: "18-24" -> 18, "25-34" -> 25, "65+" -> 65.
    Non-parsable values are pushed to the end (1e9).
    """
    s = series.fillna("").astype(str)
    # "65+" -> "65" ; "18-24" -> "18" ; else -> ""
    base = (
        s.str.replace("+", "", regex=False)
         .str.extract(r"(\d+)")
         .astype(float)
         .fillna(1e9)[0]
    )
    return base


def _align_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent dtypes on join/group keys to avoid object/float mismatches.
    - ad_id -> Int64 (nullable)
    - objective/age/gender/date_start/date_stop -> pandas StringDtype
    """
    df = df.copy()
    if "ad_id" in df.columns:
        df["ad_id"] = pd.to_numeric(df["ad_id"], errors="coerce").astype("Int64")
    for c in ["objective", "age", "gender", "date_start", "date_stop"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


# -----------------------------
# Core comparison (original vs estimated)
# -----------------------------
def make_core_comparison_table(
    df_core_orig: pd.DataFrame,
    df_core_est: pd.DataFrame,
    metric: str = "impressions",
    keys: Sequence[str] = ("ad_id", "objective", "date_start", "date_stop", "age", "gender"),
) -> pd.DataFrame:
    """
    Build a comparison table (original vs estimated) at age×gender level.

    - original: value from age×gender rows in df_core_orig (missing -> 0)
    - estimated: value from df_core_est (already at age×gender granularity)
    - delta: estimated - original
    - original_present: whether original > 0
    """
    # 1) Align dtypes on both sides (critical for merges)
    df_core_orig = _align_key_dtypes(drop_duplicate_columns(df_core_orig))
    df_core_est  = _align_key_dtypes(drop_duplicate_columns(df_core_est))

    # 2) Keep only age×gender rows from original (platform rows have NA age/gender)
    orig_ag = df_core_orig.copy()
    if {"age", "gender"}.issubset(orig_ag.columns):
        orig_ag = orig_ag[orig_ag["age"].notna() | orig_ag["gender"].notna()].copy()

    # 3) Ensure metric is numeric
    for df in (orig_ag, df_core_est):
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)

    # 4) Aggregate to unique keys (avoid duplicate multiplication)
    keys_list = list(keys)
    orig_g = (
        orig_ag[keys_list + [metric]]
        .groupby(keys_list, dropna=False)[metric].sum().reset_index()
    )
    est_g = (
        df_core_est[keys_list + [metric]]
        .groupby(keys_list, dropna=False)[metric].sum().reset_index()
    )

    # 5) Merge & compute deltas
    comp = (
        est_g.rename(columns={metric: "estimated"})
        .merge(orig_g.rename(columns={metric: "original"}), on=keys_list, how="left")
    )
    comp["original"] = comp["original"].fillna(0.0)
    comp["delta"] = comp["estimated"] - comp["original"]
    comp["original_present"] = comp["original"] > 0

    # 6) Stable ordering (group keys first, then age/gender readability)
    comp = comp.sort_values(by=["ad_id", "objective", "date_start", "date_stop"], kind="stable").copy()
    try:
        comp = comp.sort_values(
            by=["age", "gender"],
            key=lambda s: _age_bucket_sort_key(s) if s.name == "age" else s.astype(str),
            kind="stable",
        )
    except Exception:
        pass

    return comp


def plot_core_bars(
    comp: pd.DataFrame,
    ad_id: int,
    metric: str = "impressions",
    filename: str = "core_impressions_bars.png",
) -> str:
    """
    Bar chart: Original vs Estimated per (age|gender) for a given ad_id.
    Returns the saved filename (or empty string if no rows for that ad_id).
    """
    sub = comp[comp["ad_id"] == ad_id].copy()
    if sub.empty:
        return ""

    # Order by age/gender for readability
    try:
        sub = sub.sort_values(
            by=["age", "gender"],
            key=lambda s: _age_bucket_sort_key(s) if s.name == "age" else s.astype(str),
            kind="stable",
        )
    except Exception:
        pass

    sub["label"] = sub["age"].astype(str) + " | " + sub["gender"].astype(str)
    x = np.arange(len(sub))
    width = 0.40

    fig = plt.figure()
    plt.bar(x - width / 2, sub["original"].values, width, label="Original")
    plt.bar(x + width / 2, sub["estimated"].values, width, label="Estimated")
    plt.xticks(x, sub["label"], rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} by Age|Gender — ad_id={ad_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    return filename


# -----------------------------
# Optional: Reach estimation from frequency
# -----------------------------
def estimate_reach_from_frequency(
    df_core_est: pd.DataFrame,
    df_core_plat: pd.DataFrame,
    group_keys: Sequence[str] = ("ad_id", "objective", "date_start", "date_stop"),
) -> pd.DataFrame:
    """
    Estimate age×gender reach using platform frequency per group.

    Assumption: frequency is uniform across age×gender within each (ad_id, period, objective).
    Steps:
      1) Compute platform reach_target = impressions_plat / max(frequency_plat, eps).
      2) For each group, compute weights w = impressions_est / sum(impressions_est).
      3) estimated_reach_age_gender = w * reach_target.
    """
    eps = 1e-12
    plat = _align_key_dtypes(drop_duplicate_columns(df_core_plat.copy()))

    # Keep platform rows (age/gender NA) if present
    if "age" in plat.columns and "gender" in plat.columns:
        plat = plat[plat["age"].isna() & plat["gender"].isna()].copy()

    # Ensure numeric columns exist
    for c in ("impressions", "frequency"):
        if c in plat.columns:
            plat[c] = pd.to_numeric(plat[c], errors="coerce").astype(float)
        else:
            plat[c] = np.nan

    plat["reach_target"] = plat["impressions"] / np.clip(plat["frequency"], eps, None)
    plat = plat[list(group_keys) + ["reach_target", "impressions"]].rename(
        columns={"impressions": "impressions_target"}
    )

    out = _align_key_dtypes(drop_duplicate_columns(df_core_est.copy()))
    out = out.merge(plat, on=list(group_keys), how="left")

    # Group-normalized weights from estimated impressions
    out["impressions_est"] = pd.to_numeric(out.get("impressions"), errors="coerce").fillna(0.0)
    gsum = out.groupby(list(group_keys))["impressions_est"].transform("sum").replace({0: np.nan})
    out["w_impr"] = out["impressions_est"] / gsum
    out["estimated_reach"] = out["w_impr"] * out["reach_target"]
    return out


# -----------------------------
# Actions summary & plot
# -----------------------------
def make_actions_summary(
    df_event_orig: pd.DataFrame,
    df_event_est: pd.DataFrame,
    df_core_est: pd.DataFrame,
    ad_id: int,
    group_keys: Sequence[str] = ("ad_id", "objective", "date_start", "date_stop", "age", "gender"),
) -> pd.DataFrame:
    """
    Summarize actions per (age, gender, action_type, stat) with rates.

    Returns a table with:
      amount_original, amount_estimated, delta_amount, impressions, action_rate
    """
    # Align dtypes across inputs (important for merges/joins)
    df_event_orig = _align_key_dtypes(drop_duplicate_columns(df_event_orig))
    df_event_est  = _align_key_dtypes(drop_duplicate_columns(df_event_est))
    df_core_est   = _align_key_dtypes(drop_duplicate_columns(df_core_est))

    keys = list(group_keys) + ["action_type", "stat"]
    orig = df_event_orig[df_event_orig["ad_id"] == ad_id].copy()
    est  = df_event_est[df_event_est["ad_id"] == ad_id].copy()

    # Ensure amount is numeric
    for df in (orig, est):
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    g_orig = orig.groupby(keys, dropna=False)["amount"].sum().reset_index(name="amount_original")
    g_est  = est.groupby(keys, dropna=False)["amount"].sum().reset_index(name="amount_estimated")

    # Outer merge so we don't miss categories that exist only in original or only in estimated
    comp = g_est.merge(g_orig, on=keys, how="outer")
    comp["amount_estimated"] = comp["amount_estimated"].fillna(0.0)
    comp["amount_original"]  = comp["amount_original"].fillna(0.0)
    comp["delta_amount"]     = comp["amount_estimated"] - comp["amount_original"]

    # Join impressions for action rates
    imps = df_core_est[df_core_est["ad_id"] == ad_id][list(group_keys) + ["impressions"]].copy()
    imps["impressions"] = pd.to_numeric(imps["impressions"], errors="coerce").fillna(0.0)

    comp = comp.merge(imps, on=list(group_keys), how="left")
    denom = comp["impressions"].replace({0: np.nan})
    comp["action_rate"] = comp["amount_estimated"] / denom

    # Order for readability
    try:
        comp = comp.sort_values(
            by=["age", "gender", "action_type", "stat"],
            key=lambda s: _age_bucket_sort_key(s) if s.name == "age" else s.astype(str),
            kind="stable",
        )
    except Exception:
        pass
    return comp


def plot_actions_bar(
    actions_comp: pd.DataFrame,
    ad_id: int,
    stat: Optional[str] = None,
    filename: str = "actions_bars_ad.png",
) -> str:
    """
    Bar chart of estimated actions by action_type (optionally filter by stat).
    Returns the saved filename (or empty string if no rows for that ad_id/stat).
    """
    sub = actions_comp[actions_comp["ad_id"] == ad_id].copy()
    if stat is not None and "stat" in sub.columns:
        sub = sub[sub["stat"] == stat]
    if sub.empty:
        return ""

    agg = sub.groupby(["action_type"], dropna=False)["amount_estimated"].sum().reset_index()

    fig = plt.figure()
    plt.bar(agg["action_type"].astype(str), agg["amount_estimated"].values)
    plt.ylabel("estimated amount")
    ttl = f"Estimated actions by type — ad_id={ad_id}"
    if stat is not None:
        ttl += f" (stat={stat})"
    plt.title(ttl)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    return filename


# -----------------------------
# Driver to produce artifacts (CSV + PNG)
# -----------------------------
def _pick_ad_for_viz(df_core_est: pd.DataFrame) -> Optional[int]:
    """Pick an ad_id with the largest number of rows (good chance of nice plots)."""
    g = df_core_est.groupby("ad_id").size().reset_index(name="n")
    if g.empty:
        return None
    return int(g.sort_values("n", ascending=False).iloc[0]["ad_id"])


def run_viz_pipeline(
    df_core_orig: pd.DataFrame,
    df_core_est: pd.DataFrame,
    df_event_orig: pd.DataFrame,
    df_event_est: pd.DataFrame,
    out_dir: str = "artifacts",
    ad_id: Optional[int] = None,
    metric: str = "impressions",
    stat: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Create comparison tables and plots for a selected ad_id.
    Saves CSVs/PNGs into `out_dir` and returns their paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    if ad_id is None:
        ad_id = _pick_ad_for_viz(df_core_est)
    if ad_id is None:
        raise ValueError("No ad_id found for visualization.")

    # 1) Core comparison (CSV)
    comp = make_core_comparison_table(df_core_orig, df_core_est, metric=metric)
    comp_path = os.path.join(out_dir, f"core_comparison_ad_{ad_id}.csv")
    comp.to_csv(comp_path, index=False)

    # 2) Core bars (PNG)
    png_core = os.path.join(out_dir, f"{metric}_bars_ad_{ad_id}.png")
    plot_core_bars(comp, ad_id=ad_id, metric=metric, filename=png_core)

    # 3) Actions summary (CSV) + bars (PNG), if events exist
    actions_path = None
    png_actions = None
    if df_event_est is not None and not df_event_est.empty:
        actions_comp = make_actions_summary(df_event_orig, df_event_est, df_core_est, ad_id=ad_id)
        actions_path = os.path.join(out_dir, f"actions_summary_ad_{ad_id}.csv")
        actions_comp.to_csv(actions_path, index=False)

        png_actions = os.path.join(out_dir, f"actions_bars_ad_{ad_id}.png")
        plot_actions_bar(actions_comp, ad_id=ad_id, stat=stat, filename=png_actions)

    return {
        "ad_id": ad_id,
        "core_table": comp_path,
        "core_plot": png_core,
        "actions_table": actions_path,
        "actions_plot": png_actions,
    }
