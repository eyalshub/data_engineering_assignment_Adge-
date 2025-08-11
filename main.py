#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline:
1) Load & unify (src/data_loading.py)
2) Build cartesian product (src/cartesian_missing.py)
3) Estimate values (src/value_estimatio.py)
5) Visualization (src/visualization.py)
"""

import os
import argparse
import logging
import inspect
from typing import Dict, Sequence, Optional

import pandas as pd
import numpy as np

# ---- Your modules ----
# Step 1
import src.data_loading as dl
# Step 2
import src.cartesian_missing as cm
# Step 3+
import src.value_estimatio as ve
# Step 5
import src.visualization as vz

logger = logging.getLogger("pipeline")

# -----------------------------
# helpers: safe call / resolver
# -----------------------------
def _safe_call(func, **kwargs):
    """Call func with only supported kwargs (ignore unknowns)."""
    sig = inspect.signature(func)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**allowed)

def _resolve(module, candidates):
    """Return first attribute that exists in module from candidates list."""
    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)
    return None

def _has_events_schema(df: pd.DataFrame) -> bool:
    req = {"action_type", "stat", "amount"}
    return req.issubset(set(df.columns))

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Ad value estimation pipeline (steps 1,2,3,5)."
    )

    # אפשר לבחור ZIP או CSV
    ap.add_argument("--data-zip", default="", help="Path to data.zip containing the CSVs")
    ap.add_argument("--core-name", default="", help="Age/Gender CSV name inside ZIP (e.g. 'test_raw_data_age_gender (1).csv')")
    ap.add_argument("--extra-core-name", default="", help="Platform-level CSV name inside ZIP (e.g. 'test_raw_data_publisher_platform.csv')")

    ap.add_argument("--core-csv", default="", help="Path to core (age-gender) CSV if not using ZIP")
    ap.add_argument("--second-csv", default="", help="Second CSV: events OR extra core (platform)")

    ap.add_argument("--out-dir", default="artifacts", help="Output directory")
    ap.add_argument("--strategy", choices=["proportional", "neighbors"], default="neighbors", help="Weighting strategy for estimation")
    ap.add_argument("--neighbors-k", type=int, default=5, help="K for neighbors strategy")
    ap.add_argument("--viz-metric", default="impressions", help="Metric to plot in core bars")
    ap.add_argument("--viz-ad-id", type=str, default="", help="Specific ad_id to visualize (optional)")
    ap.add_argument("--viz-stat", type=str, default="", help="Filter actions plot by stat (optional)")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    return ap.parse_args()

# -----------------------------
# main
# -----------------------------
def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- Step 1: Load & unify -----
    load_zip = _resolve(dl, ["load_and_unify_from_zip", "load_from_zip", "load_zip"])
    load_csv = _resolve(dl, ["load_and_unify_from_csv", "load_from_csv", "load"])

    if args.data_zip:
        if load_zip is None:
            raise SystemExit("data_loading.py is missing a ZIP loader (e.g. load_and_unify_from_zip).")
        dfs = _safe_call(
            load_zip,
            zip_path=args.data_zip,
            core_name=args.core_name or None,
            extra_core_name=args.extra_core_name or None,
        )
    else:
        if not args.core_csv:
            raise SystemExit("Provide either --data-zip or --core-csv.")
        if load_csv is None:
            raise SystemExit("data_loading.py is missing a CSV loader (e.g. load_and_unify_from_csv).")
        dfs = _safe_call(
            load_csv,
            core_csv=args.core_csv,
            second_csv=args.second_csv or None,
        )

    df_core: pd.DataFrame = dfs["core"]
    df_events: pd.DataFrame = dfs.get("events", pd.DataFrame())
    logger.info("Loaded core rows: %d; events rows: %d", len(df_core), len(df_events))

    # ----- Step 2: Cartesian product -----
    build_cart = _resolve(cm, ["build_cart", "build_cartesian_product", "build_cartesian"])
    if build_cart is None:
        raise SystemExit("cartesian_missing.py must expose build_cart (or build_cartesian_product).")

    cart = _safe_call(build_cart, df_core=df_core, df_events=df_events)
    if cart.empty:
        logger.error("Cart is empty. Check inputs.")
        return
    logger.info("Cart rows: %d", len(cart))

    # ----- Step 3: Value estimation -----
    # Optional neighbors index (if present in value_estimatio.py)
    build_neighbors_index = _resolve(ve, ["build_neighbors_index"])
    neighbors_index = None
    if args.strategy == "neighbors" and build_neighbors_index is not None:
        core_ag_only = df_core[(~df_core["age"].isna()) | (~df_core["gender"].isna())].copy()
        if not core_ag_only.empty:
            neighbors_index = _safe_call(
                build_neighbors_index,
                df=core_ag_only,
                feature_cols=("impressions", "clicks", "spend"),
                dims=("objective",),
            )
            logger.info("Neighbors index built.")
        else:
            logger.info("No age×gender rows for neighbors; will fallback to proportional/historical.")

    estimate_core_metrics = _resolve(ve, ["estimate_core_metrics"])
    if estimate_core_metrics is None:
        raise SystemExit("value_estimatio.py must expose estimate_core_metrics.")

    df_core_est = _safe_call(
        estimate_core_metrics,
        df_core=df_core,
        cart=cart,
        weighting_strategy=args.strategy,
        neighbors_index=neighbors_index,
        neighbors_k=args.neighbors_k,
    )
    logger.info("Core estimated rows: %d", len(df_core_est))
    df_core_est.to_csv(os.path.join(args.out_dir, "core_estimated.csv"), index=False)

    df_events_est = pd.DataFrame()
    if not df_events.empty and _has_events_schema(df_events):
        estimate_event_table = _resolve(ve, ["estimate_event_table"])
        if estimate_event_table is None:
            logger.warning("Events file present but estimate_event_table not found; skipping events.")
        else:
            df_events_est = _safe_call(
                estimate_event_table,
                df_event=df_events,
                cart=cart,
                df_core_est=df_core_est,
                weighting_strategy=args.strategy,
                neighbors_index=neighbors_index,
                neighbors_k=args.neighbors_k,
            )
            logger.info("Events estimated rows: %d", len(df_events_est))
            df_events_est.to_csv(os.path.join(args.out_dir, "events_estimated.csv"), index=False)

    # ----- Sanity checks (optional, אם יש לך פונקציות בשלב 1/3/5 שעושות את זה – אפשר להשאיר כאן) -----
    # נשמור בדיקות סכומים בסיסיות כדי לוודא שהאמידה סגורה לטוטלים בפלטפורמה.
    def _check_core_totals(df_core_orig, df_core_est, metrics=("impressions", "clicks", "spend")):
        keys = ["ad_id", "date_start", "date_stop", "objective"]
        plat = df_core_orig[df_core_orig["age"].isna() & df_core_orig["gender"].isna()].copy()
        out = []
        for m in metrics:
            if m not in plat.columns or m not in df_core_est.columns:
                continue
            p = plat.groupby(keys, dropna=False)[m].sum().reset_index().rename(columns={m: "target"})
            a = df_core_est.groupby(keys, dropna=False)[m].sum().reset_index().rename(columns={m: "agg"})
            cmp = p.merge(a, on=keys, how="outer").fillna(0.0)
            cmp["delta"] = cmp["agg"] - cmp["target"]
            cmp["metric"] = m
            out.append(cmp)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=keys + ["target", "agg", "delta", "metric"])

    core_chk = _check_core_totals(df_core, df_core_est)
    core_chk.to_csv(os.path.join(args.out_dir, "sanity_core_totals.csv"), index=False)
    logger.info("Core totals check saved (max |delta|=%s).", None if core_chk.empty else np.abs(core_chk["delta"]).max())

    # ----- Step 5: Visualization -----
    make_core_comparison_table = _resolve(vz, ["make_core_comparison_table"])
    plot_core_bars = _resolve(vz, ["plot_core_bars"])
    make_actions_summary = _resolve(vz, ["make_actions_summary"])
    plot_actions_bar = _resolve(vz, ["plot_actions_bar"])

# Select ad_id for visualization
    if args.viz_ad_id:
        try:
            ad_id_for_viz = int(args.viz_ad_id)
        except ValueError:
            ad_id_for_viz = args.viz_ad_id
    else:
        g = df_core_est.groupby("ad_id").size().reset_index(name="n")
        ad_id_for_viz = None if g.empty else g.sort_values("n", ascending=False).iloc[0]["ad_id"]

    if ad_id_for_viz is not None and make_core_comparison_table and plot_core_bars:
        comp = make_core_comparison_table(df_core, df_core_est, metric=args.viz_metric)
        comp_path = os.path.join(args.out_dir, f"core_comparison_ad_{ad_id_for_viz}.csv")
        comp.to_csv(comp_path, index=False)
        png_core = os.path.join(args.out_dir, f"{args.viz_metric}_bars_ad_{ad_id_for_viz}.png")
        plot_core_bars(comp, ad_id=ad_id_for_viz, metric=args.viz_metric, filename=png_core)
        logger.info("Saved core viz to %s ; %s", comp_path, png_core)

        if not df_events_est.empty and make_actions_summary and plot_actions_bar:
            actions_comp = make_actions_summary(df_events, df_events_est, df_core_est, ad_id=ad_id_for_viz)
            actions_path = os.path.join(args.out_dir, f"actions_summary_ad_{ad_id_for_viz}.csv")
            actions_comp.to_csv(actions_path, index=False)
            png_actions = os.path.join(args.out_dir, f"actions_bars_ad_{ad_id_for_viz}.png")
            plot_actions_bar(actions_comp, ad_id=ad_id_for_viz, stat=(args.viz_stat or None), filename=png_actions)
            logger.info("Saved actions viz to %s ; %s", actions_path, png_actions)
    else:
        logger.warning("No ad_id available for visualization or viz functions not found.")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
