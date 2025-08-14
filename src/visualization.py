# src/visualization.py
import argparse
import os
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.data_loading import load_and_prepare_data
from src.cartesian_missing import build_cartesian_for_ad, find_missing_for_ad
from src.value_estimatio import estimate_values_sum_preserving, drop_duplicate_columns
logger = logging.getLogger(__name__)

# ----------------------------
# 📊 Analysis + Visualization
# ----------------------------
def _align_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ad_id" in df.columns:
        df["ad_id"] = pd.to_numeric(df["ad_id"], errors="coerce").astype("Int64")
    for c in ["objective", "age", "gender", "date_start", "date_stop", "publisher_platform"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df

def _age_bucket_sort_key(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    base = (
        s.str.replace("+", "", regex=False)
         .str.extract(r"(\d+)")
         .astype(float)
         .fillna(1e9)[0]
    )
    return base

def make_multimetric_comparison(
    df_core_orig: pd.DataFrame,
    df_core_est: pd.DataFrame,
    metrics: Sequence[str],
    group_keys: Sequence[str] = ["ad_id", "objective", "publisher_platform", "age", "gender"]
) -> pd.DataFrame:
    df_core_orig = _align_key_dtypes(drop_duplicate_columns(df_core_orig))
    df_core_est  = _align_key_dtypes(drop_duplicate_columns(df_core_est))

    df_core_orig = df_core_orig[df_core_orig["age"].notna() | df_core_orig["gender"].notna()].copy()

    for m in metrics:
        if m not in df_core_orig.columns:
            df_core_orig[m] = 0.0
        if m not in df_core_est.columns:
            df_core_est[m] = 0.0

    orig_g = (
        df_core_orig[group_keys + list(metrics)]
        .groupby(group_keys, dropna=False)
        .sum()
        .reset_index()
        .melt(id_vars=group_keys, var_name="metric", value_name="original")
    )

    est_g = (
        df_core_est[group_keys + list(metrics)]
        .groupby(group_keys, dropna=False)
        .sum()
        .reset_index()
        .melt(id_vars=group_keys, var_name="metric", value_name="estimated")
    )

    comp = est_g.merge(orig_g, on=group_keys + ["metric"], how="outer")
    comp["original"] = comp["original"].fillna(0.0)
    comp["estimated"] = comp["estimated"].fillna(0.0)
    comp["delta"] = comp["estimated"] - comp["original"]
    comp["pct_change"] = np.where(
        comp["original"] == 0,
        np.nan,
        100 * comp["delta"] / comp["original"]
    )

    return comp

def plot_metric_comparison_bar_improved(
    comp_df,
    ad_id: int,
    metric: str,
    filename: str,
    top_n: int = 15,
    log_scale: bool = False,
    show_values: bool = True
):
    df = comp_df[(comp_df["ad_id"] == ad_id) & (comp_df["metric"] == metric)].copy()
    if df.empty:
        print(f"⚠️ No data to plot for metric '{metric}' and ad_id={ad_id}")
        return ""

    # Clean and sort labels
    df["age_sort"] = _age_bucket_sort_key(df["age"])
    df["gender"] = df["gender"].fillna("Unknown")
    df["age"] = df["age"].fillna("Unknown")
    df["label"] = df["age"] + " | " + df["gender"]

    # Calculate filled portion (estimated - original)
    df["filled"] = df["estimated"] - df["original"]
    df["filled"] = df["filled"].apply(lambda x: max(x, 0))

    # Total height = for sorting and filtering
    df["total"] = df["original"] + df["filled"]
    df = df[df["total"] > 0].sort_values("total", ascending=False).head(top_n)

    x = np.arange(len(df))
    width = 0.5

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_orig = ax.bar(x, df["original"], width, label="Original", color="#4C72B0")
    bars_fill = ax.bar(x, df["filled"], width, bottom=df["original"], label="Filled", color="#FFA07A")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Age | Gender — ad_id={ad_id}", fontsize=13)
    ax.legend()

    # Value labels
    if show_values:
        for bar, val in zip(bars_orig, df["original"]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{int(val)}",
                        ha='center', va='center', fontsize=8, color='white')
        for bar, val, base in zip(bars_fill, df["filled"], df["original"]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, base + val / 2, f"+{int(val)}",
                        ha='center', va='center', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    return filename


def summarize_by_category(comp_df: pd.DataFrame, metric: str):
    df = comp_df[comp_df["metric"] == metric].copy()
    summary = (
        df.groupby(["objective", "publisher_platform", "age", "gender"], dropna=False)
        .agg(
            total_original=("original", "sum"),
            total_estimated=("estimated", "sum"),
            delta=("delta", "sum"),
            pct_change=("pct_change", "mean")
        )
        .reset_index()
    )
    return summary.sort_values("delta", key=abs, ascending=False)

# ----------------------------
# 🚀 Visualization Only Pipeline
# ----------------------------
def run_ad_pipeline(*, df_core_ad: pd.DataFrame, df_estimated: pd.DataFrame, ad_id: int, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    all_possible_metrics = [
    "impressions", "clicks", "spend", "frequency",
    "video_play_actions", "video_thruplay_watched_actions"
]

    existing_metrics = [
        m for m in all_possible_metrics
        if m in df_core_ad.columns and m in df_estimated.columns
    ]

    if not existing_metrics:
        raise ValueError("❌ None of the expected metric columns exist in the data.")

    print(f"📊 Using metrics: {existing_metrics}")

    comp_df = make_multimetric_comparison(df_core_ad, df_estimated, metrics=existing_metrics)

    csv_path = os.path.join(out_dir, f"multi_metric_comparison_ad_{ad_id}.csv")
    comp_df.to_csv(csv_path, index=False)

    plots = []
    for m in existing_metrics:
        out_png = os.path.join(out_dir, f"{m}_bars_ad_{ad_id}.png")
        plot_metric_comparison_bar_improved(
            comp_df, ad_id=ad_id, metric=m, filename=out_png, top_n=15, log_scale=True
        )
        plots.append(out_png)

    summary_csv = os.path.join(out_dir, f"impressions_summary_ad_{ad_id}.csv")
    summarize_by_category(comp_df, "impressions").to_csv(summary_csv, index=False)

    print("\n✅ Analysis completed for ad_id:", ad_id)
    print("📁 Outputs saved to:", out_dir)
    print("📈 Comparison CSV:", csv_path)
    print("📈 Category summary:", summary_csv)
    print("🖼️ Plots:")
    for p in plots:
        print(f"   - {p}")

    return {
        "comparison_csv": csv_path,
        "summary_csv": summary_csv,
        "plots": plots,
        "comparison_df": comp_df
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full pipeline for a single ad_id")
    parser.add_argument("--ad-id", type=int, required=True, help="The ad_id to analyze")
    parser.add_argument("--csv-age-gender", type=str, required=True, help="Path to age-gender CSV or ZIP")
    parser.add_argument("--csv-platform", type=str, required=True, help="Path to platform CSV or ZIP")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Directory to save outputs")
    args = parser.parse_args()

    from src.data_loading import load_and_prepare_data
    from src.cartesian_missing import build_cartesian_for_ad
    from src.value_estimatio import estimate_values_sum_preserving

    _, df_core, _ = load_and_prepare_data(args.csv_age_gender, args.csv_platform)
    df_core_ad = df_core[df_core["ad_id"] == args.ad_id].copy()

    if df_core_ad.empty:
        raise ValueError(f"No rows found for ad_id={args.ad_id}")

    cartesian = build_cartesian_for_ad(df_core, args.ad_id)
    df_estimated = estimate_values_sum_preserving(df_core_ad, cartesian)

    run_ad_pipeline(
        df_core_ad=df_core_ad,
        df_estimated=df_estimated,
        ad_id=args.ad_id,
        out_dir=args.out_dir
    )
