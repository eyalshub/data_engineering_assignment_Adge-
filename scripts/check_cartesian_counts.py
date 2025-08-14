# scripts/check_cartesian_counts.py
import argparse
import os
import json
import numpy as np
import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cartesian_missing import (
    domains_per_ad, build_cartesian, DIMS,
    build_cartesian_for_ad, find_missing_for_ad
)

def load_events(artifacts_dir: str):
    names = [
        "stage1_events_actions.csv",
        "stage1_events_action_values.csv",
        # "stage1_events_video_play_actions.csv",
        # "stage1_events_video_thruplay_watched_actions.csv",
        # "stage1_events_video_avg_time_watched_actions.csv",
    ]
    events = []
    for nm in names:
        p = os.path.join(artifacts_dir, nm)
        if os.path.exists(p):
            events.append(pd.read_csv(p))
    return events

def domain_product(dom: dict) -> int:
    return int(np.prod([len(dom[d]) for d in DIMS], dtype=np.int64))

def main():
    ap = argparse.ArgumentParser(description="Verify Cartesian grid sizes vs per-ad domain products + export reports.")
    ap.add_argument("--core", default="artifacts/stage1_core.csv", help="Path to stage1_core.csv")
    ap.add_argument("--artifacts", default="artifacts", help="Directory containing event CSVs")
    ap.add_argument("--with-events", action="store_true", help="Include event tables as extra sources")

    # new: export options
    ap.add_argument("--save-domain-report", default="", help="CSV path for per-ad domain counts and cart size")
    ap.add_argument("--ad-id", type=int, default=None, help="Focus on a single ad_id for exports")
    ap.add_argument("--save-ad-cart", default="", help="CSV path to save the cartesian grid for --ad-id")
    ap.add_argument("--save-ad-missing", default="", help="CSV path to save missing combos for --ad-id")

    # dangerous (can be huge): export full cart
    ap.add_argument("--save-full-cart", default="", help="CSV path to save the FULL cartesian grid (all ads) — can be huge!")
    ap.add_argument("--i-know-this-is-large", action="store_true", help="Confirm you understand full cart may be huge")

    # summary options
    ap.add_argument("--top", type=int, default=10, help="Show top-N ads by cartesian size")
    ap.add_argument("--save-summary", default="", help="Optional path to save a JSON summary")
    args = ap.parse_args()

    if not os.path.exists(args.core):
        raise FileNotFoundError(f"Core file not found: {args.core}")

    core = pd.read_csv(args.core)
    extra = load_events(args.artifacts) if args.with_events else None

    # --- Expected total = sum of per-ad domain products
    doms = domains_per_ad(core, extra_sources=extra)
    expected_total = int(sum(domain_product(dom) for dom in doms.values()))

    # --- Build cartesian (matching the same sources choice)
    cart = build_cartesian(core, extra_sources=extra)
    got_total = len(cart)

    mode = "WITH EVENTS" if args.with_events else "CORE ONLY"
    print(f"{mode}: {got_total} vs {expected_total}")
    assert got_total == expected_total, "Cartesian row count != sum of per-ad domain products"

    # --- Domain report (per ad)
    if args.save_domain_report:
        rows = []
        for ad, dom in doms.items():
            rows.append({
                "ad_id": ad,
                "objective_count": len(dom["objective"]),
                "publisher_platform_count": len(dom["publisher_platform"]),
                "age_count": len(dom["age"]),
                "gender_count": len(dom["gender"]),
                "cartesian_size": domain_product(dom),
            })
        rep = pd.DataFrame(rows).sort_values("cartesian_size", ascending=False)
        Path(args.save_domain_report).parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(args.save_domain_report, index=False)
        print(f"Saved domain report -> {args.save_domain_report}")

    # --- Per-ad exports
    if args.ad_id is not None:
        if args.save_ad_cart:
            df_ad_cart = build_cartesian_for_ad(core, args.ad_id, extra_sources=extra)
            Path(args.save_ad_cart).parent.mkdir(parents=True, exist_ok=True)
            df_ad_cart.to_csv(args.save_ad_cart, index=False)
            print(f"Saved cartesian for ad_id={args.ad_id} -> {args.save_ad_cart}")

        if args.save_ad_missing:
            df_missing = find_missing_for_ad(core, args.ad_id, extra_sources=extra)
            Path(args.save_ad_missing).parent.mkdir(parents=True, exist_ok=True)
            df_missing.to_csv(args.save_ad_missing, index=False)
            print(f"Saved missing combos for ad_id={args.ad_id} -> {args.save_ad_missing}")

    # --- (Optional, heavy) full-cart export
    if args.save_full_cart:
        if not args.i_know_this_is_large:
            raise SystemExit("Refusing to write full cart without --i-know-this-is-large flag.")
        Path(args.save_full_cart).parent.mkdir(parents=True, exist_ok=True)
        cart.to_csv(args.save_full_cart, index=False)
        print(f"Saved FULL cart -> {args.save_full_cart}")

    # --- Top-N ads by cartesian size
    sizes = {ad: domain_product(dom) for ad, dom in doms.items()}
    top_items = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[: args.top]
    print(f"\nTop {args.top} ads by cartesian size:")
    for ad, sz in top_items:
        dom = doms[ad]
        print(f" ad_id={ad}: size={sz}, domains={{"
              f"objective:{len(dom['objective'])}, "
              f"publisher_platform:{len(dom['publisher_platform'])}, "
              f"age:{len(dom['age'])}, "
              f"gender:{len(dom['gender'])}}}")

    # --- Optional JSON summary
    if args.save_summary:
        summary = {
            "mode": mode,
            "total_rows": got_total,
            "expected_rows": expected_total,
            "top": [{"ad_id": int(ad), "size": int(sz)} for ad, sz in top_items],
        }
        Path(os.path.dirname(args.save_summary) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.save_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary -> {args.save_summary}")


if __name__ == "__main__":
    main()
