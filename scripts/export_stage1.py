# scripts/export_stage1.py
import argparse
import sys
from pathlib import Path
# Add project root to sys.path so "src" is importable at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_loading import load_and_prepare_data, save_stage1_outputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age-gender", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    _, df_core, events = load_and_prepare_data(args.age_gender, args.platform)
    paths = save_stage1_outputs(df_core, events, out_dir=args.out_dir)
    print(paths)

if __name__ == "__main__":
    main()
