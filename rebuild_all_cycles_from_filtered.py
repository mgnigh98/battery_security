#!/usr/bin/env python3
import os, glob, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Folder with filtered *_labeled_3class.xlsx files")
    ap.add_argument("--out_csv", required=True, help="Output ALL_cycles_3class.csv path")
    ap.add_argument("--pattern", default="*_labeled_3class.xlsx")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not files:
        raise SystemExit(f"No files found: {os.path.join(args.indir, args.pattern)}")

    all_df = []
    for f in files:
        df = pd.read_excel(f, engine="openpyxl")
        df["file"] = os.path.basename(f)  # traceability, required later by early-window extractor
        all_df.append(df)

    ALL = pd.concat(all_df, ignore_index=True)

    # Sanity: ensure Cycle exists
    if "Cycle" not in ALL.columns and "cycle" in ALL.columns:
        ALL = ALL.rename(columns={"cycle": "Cycle"})

    ALL.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {args.out_csv} with {len(ALL)} rows from {len(files)} files")

    # quick counts
    if "cycle_label_3name" in ALL.columns:
        print("\nLabel counts:")
        print(ALL["cycle_label_3name"].astype(str).str.strip().str.lower().value_counts())

if __name__ == "__main__":
    main()
