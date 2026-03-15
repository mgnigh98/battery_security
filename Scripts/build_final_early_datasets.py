#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_LABEL_COL = "cycle_label_3name"
GROUP_COL = "file"
META_COLS = ["file", "Cycle", "Label", "cycle_label_3class", "cycle_label_3name"]


def detect_label_col(df: pd.DataFrame) -> str:
    for c in ["cycle_label_3name", "battery_label_3name", "Label"]:
        if c in df.columns:
            return c
    raise ValueError("No suitable label column found.")


def normalize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    label_map = {
        "0": "BAD",
        "1": "GOOD_not_drone",
        "2": "GOOD_drone",
        "bad": "BAD",
        "good_not_drone": "GOOD_not_drone",
        "good_drone": "GOOD_drone",
    }
    df[label_col] = df[label_col].replace(label_map)
    return df


def add_cross_window_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    window_pairs = [
        (1, 2),
        (2, 5),
        (5, 10),
        (10, 20),
        (20, 30),
        (30, 50),
        (50, 60),
        (1, 10),
        (1, 30),
    ]

    base_feats = [
        "IR_early",
        "V_sag",
        "V_sag_ratio",
        "dvdt_std",
        "power_std",
        "energy_ws",
        "energy_per_I",
    ]

    for w1, w2 in window_pairs:
        for feat in base_feats:
            c1 = f"early{w1}_{feat}"
            c2 = f"early{w2}_{feat}"
            if c1 in df.columns and c2 in df.columns:
                df[f"{feat}_growth_{w1}_to_{w2}"] = (
                    pd.to_numeric(df[c2], errors="coerce") -
                    pd.to_numeric(df[c1], errors="coerce")
                )

    return df


def feature_cols_for_window(df: pd.DataFrame, window_sec: int) -> list[str]:
    """
    Cumulative early features:
      2s  -> early1_* + early2_* + growth
      10s -> early1_* + early2_* + early5_* + early10_* + growth
      30s -> ... + early20_* + early30_* + growth
      60s -> all early* + growth
    """
    allowed = [w for w in [1, 2, 5, 10, 20, 30, 50, 60] if w <= window_sec]
    cols = []

    for w in allowed:
        prefix = f"early{w}_"
        cols.extend([c for c in df.columns if c.startswith(prefix)])

    growth_cols = [c for c in df.columns if "_growth_" in c]
    cols.extend(growth_cols)

    return sorted(set(cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_csv",
        type=Path,
        default=Path("all_csv_for_training") / "ALL_cycles_3class_early.csv",
        help="Input early-feature CSV.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("all_csv_for_training") / "final_early_model_data",
        help="Output directory for final model-ready datasets.",
    )
    ap.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[2, 10, 30, 60],
        help="Final windows to materialize.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    label_col = detect_label_col(df)

    df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce")
    df = df.dropna(subset=["Cycle", GROUP_COL, label_col]).copy()
    df["Cycle"] = df["Cycle"].astype(int)

    df = normalize_labels(df, label_col)
    df = df[df[label_col].isin(["BAD", "GOOD_drone", "GOOD_not_drone"])].copy()

    # Make sure final canonical label column exists
    if label_col != "cycle_label_3name":
        df["cycle_label_3name"] = df[label_col]

    df = add_cross_window_growth_features(df)

    print("Final label counts:")
    print(df["cycle_label_3name"].value_counts(dropna=False))

    summary_rows = []

    for w in args.windows:
        feat_cols = feature_cols_for_window(df, w)
        if not feat_cols:
            print(f"Skipping {w}s: no usable features found.")
            continue

        # Build final dataset for this window
        keep_cols = [c for c in META_COLS if c in df.columns] + feat_cols
        out_df = df[keep_cols].copy()
        out_df = out_df.sort_values(["file", "Cycle"]).reset_index(drop=True)

        out_csv = args.out_dir / f"final_early_{w}s.csv"
        out_df.to_csv(out_csv, index=False)

        # Save plain-text feature list
        feat_txt = args.out_dir / f"feature_list_{w}s.txt"
        with open(feat_txt, "w") as f:
            for col in feat_cols:
                f.write(col + "\n")

        summary_rows.append({
            "window_sec": w,
            "rows": len(out_df),
            "n_features": len(feat_cols),
            "output_csv": str(out_csv),
            "feature_list_txt": str(feat_txt),
        })

        print(f"Saved {out_csv} | rows={len(out_df)} | features={len(feat_cols)}")

    summary_df = pd.DataFrame(summary_rows).sort_values("window_sec").reset_index(drop=True)
    summary_csv = args.out_dir / "feature_counts.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nSaved feature summary: {summary_csv}")


if __name__ == "__main__":
    main()
