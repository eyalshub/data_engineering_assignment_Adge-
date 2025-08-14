# scripts/run_ad_pipeline.py

import os
import sys
import argparse
import pandas as pd

# ✅ Add project root to Python path for imports to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Project modules
from src.data_loading import load_and_prepare_data
from src.cartesian_missing import build_cartesian_for_ad
from src.value_estimatio import estimate_values_sum_preserving
from src.visualization import run_ad_pipeline


def run_ad_pipeline_from_files(ad_id: int, csv_age_gender: str, csv_platform: str, out_dir: str = "artifacts") -> dict:
    """
    Runs the full analytics pipeline for a specific ad_id:
    1. Loads and unifies the input CSVs.
    2. Builds the full Cartesian product of relevant dimensions.
    3. Estimates missing values per ad_id (sum-preserving).
    4. Generates visualizations and CSV reports.

    Parameters:
        ad_id (int): The ad ID to analyze.
        csv_age_gender (str): Path to the age-gender CSV file.
        csv_platform (str): Path to the publisher platform CSV file.
        out_dir (str): Output directory to save artifacts.

    Returns:
        dict: Core input, estimated data, and output metadata.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"📥 Loading and merging data for ad_id={ad_id} ...")
    _, df_core, _ = load_and_prepare_data(csv_age_gender, csv_platform)

    print("🔍 Filtering data by ad_id...")
    df_core_ad = df_core[df_core["ad_id"] == ad_id].copy()
    if df_core_ad.empty:
        raise ValueError(f"No rows found for ad_id={ad_id} in the dataset.")

    print("🧩 Building Cartesian grid...")
    cartesian = build_cartesian_for_ad(df_core, ad_id)

    print("🔄 Estimating missing values (sum-preserving)...")
    df_estimated = estimate_values_sum_preserving(df_core_ad, cartesian)

    print("📊 Running visualization pipeline...")
    results = run_ad_pipeline(df_core_ad=df_core_ad, df_estimated=df_estimated, ad_id=ad_id, out_dir=out_dir)


    print("\n✅ Pipeline completed successfully!")
    print("📁 Output directory:", out_dir)
    print("📊 Comparison CSV:", results['comparison_csv'])
    print("📈 Summary CSV:", results['summary_csv'])
    print("🖼️ Plots:")
    for plot in results['plots']:
        print(f"   - {plot}")

    return {
        "core_input": df_core_ad,
        "cartesian": cartesian,
        "estimated": df_estimated,
        "outputs": results,
    }


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for pipeline execution.
    """
    parser = argparse.ArgumentParser(
        description="Run full analytics pipeline on a specific ad_id",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ad-id", type=int, required=True, help="The ad_id to analyze")
    parser.add_argument("--csv-age-gender", type=str, required=True, help="Path to age-gender CSV or ZIP file")
    parser.add_argument("--csv-platform", type=str, required=True, help="Path to platform CSV or ZIP file")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Directory to save output files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        run_ad_pipeline_from_files(
            ad_id=args.ad_id,
            csv_age_gender=args.csv_age_gender,
            csv_platform=args.csv_platform,
            out_dir=args.out_dir
        )
    except Exception as e:
        print("\n❌ Pipeline execution failed.")
        print("💥 Error:", str(e))
        import traceback
        traceback.print_exc()
