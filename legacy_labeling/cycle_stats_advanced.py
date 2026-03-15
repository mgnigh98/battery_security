#!/usr/bin/env python3
"""
cycle_stats_advanced.py

Aggregates labeled cycle workbooks and produces:
1) Failure-by-cycle-index stats (hard/soft/both)
2) Dataset-level medians/means for quantitative metrics from GOOD cycles
3) Quantified checks for every cycle vs those medians (four-criterion "AllPass")
4) Per-file summary counts

INPUT: a folder containing *_labeled.xlsx files with a 'cycle_labels' sheet
OUTPUT: stats_advanced/  (CSV files listed below)

Quantitative metrics per cycle:
- Chg_Spec_mAhg       (H1 context; lower is safer)
- DChg_Spec_mAhg      (H3 context; higher is better)
- Missing_Count       (H4 context; # of essential NaNs)
- IR_proxy            (S1; lower is better)
- CE_dev_abs = |CE - 1| (S2; lower is better)

Hard/Soft recognition:
- Hard flags: any columns starting with 'HF_' (we expect HF_CHG_SPEC_HIGH, HF_DCHG_SPEC_LOW, HF_MISSING)
- Soft flags: any columns starting with 'SP_' (we expect e.g., SP_IR_OUTLIER, SP_CE_SOFT)

Usage:
  python cycle_stats_advanced.py results/  -o stats_advanced/
"""

from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd

ESSENTIALS = ["Chg_Cap_Ah", "DChg_Cap_Ah", "Chg_Energy_Wh", "DChg_Energy_Wh", "CE"]

def load_cycle_labels(path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = None
    for cand in ["cycle_labels", "labels"]:
        if cand in xl.sheet_names:
            sheet = cand
            break
    if sheet is None:
        raise RuntimeError(f"No cycle_labels/labels sheet in {os.path.basename(path)}")
    df = pd.read_excel(xl, sheet_name=sheet)

    # Ensure expected columns exist even if missing
    for col in ["Chg_Spec_mAhg","DChg_Spec_mAhg","IR_proxy","CE","Chg_Cap_Ah","DChg_Cap_Ah","Chg_Energy_Wh","DChg_Energy_Wh","Label","Cycle"]:
        if col not in df.columns:
            df[col] = np.nan
    for hf in ["HF_CHG_SPEC_HIGH","HF_DCHG_SPEC_LOW","HF_MISSING"]:
        if hf not in df.columns:
            df[hf] = False

    # Any SP_ columns are soft flags; if none, create empties later.
    return df

def add_quant_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Missing count across essentials (float NaNs)
    miss = df[ESSENTIALS].isna().sum(axis=1)
    df["Missing_Count"] = miss.astype(int)
    # CE deviation
    df["CE_dev_abs"] = (df["CE"] - 1.0).abs()
    # Count of hard/soft flags per cycle (if present)
    hard_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("HF_")]
    soft_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("SP_")]
    df["Hard_Count"] = df[hard_cols].sum(axis=1) if hard_cols else 0
    df["Soft_Count"] = df[soft_cols].sum(axis=1) if soft_cols else 0
    return df

def main():
    ap = argparse.ArgumentParser(description="Advanced stats on labeled battery cycles.")
    ap.add_argument("input_dir", help="Folder containing *_labeled.xlsx files")
    ap.add_argument("-o","--outdir", default="stats_advanced", help="Output directory for CSVs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*_labeled.xlsx")))
    if not files:
        print(f"No *_labeled.xlsx files in {args.input_dir}")
        return

    # Load & concatenate all cycles
    all_rows = []
    for f in files:
        try:
            df = load_cycle_labels(f)
            df = add_quant_columns(df)
            df.insert(0, "file", os.path.basename(f))
            all_rows.append(df)
            print(f"[OK] {os.path.basename(f)}: cycles={len(df)}")
        except Exception as e:
            print(f"[SKIP] {os.path.basename(f)}: {e}")

    if not all_rows:
        print("No valid labeled files loaded.")
        return

    allc = pd.concat(all_rows, ignore_index=True)

    # --- 1) Failure by cycle index ---
    # Define hard/soft fail booleans at cycle level
    hard_fail = (allc.filter(like="HF_").sum(axis=1) > 0)
    soft_fail = (allc.filter(like="SP_").sum(axis=1) > 0) if any(c.startswith("SP_") for c in allc.columns) else pd.Series(False, index=allc.index)

    fail_by_idx = (
        allc
        .assign(hard_fail=hard_fail, soft_fail=soft_fail)
        .groupby("Cycle", dropna=True)
        .agg(
            cycles=("Cycle","size"),
            hard_fail_cnt=("hard_fail","sum"),
            soft_fail_cnt=("soft_fail","sum"),
        )
        .reset_index()
    )
    fail_by_idx["both_fail_cnt"] = 0
    if soft_fail.any():
        both = (
            allc.assign(hard_fail=hard_fail, soft_fail=soft_fail)
                .groupby("Cycle", dropna=True)
                .apply(lambda g: int(((g["hard_fail"]) & (g["soft_fail"])).sum()))
        )
        fail_by_idx["both_fail_cnt"] = fail_by_idx["Cycle"].map(both.to_dict()).fillna(0).astype(int)
    # percents
    fail_by_idx["pct_hard_fail"] = (100.0 * fail_by_idx["hard_fail_cnt"] / fail_by_idx["cycles"]).round(2)
    fail_by_idx["pct_soft_fail"] = (100.0 * fail_by_idx["soft_fail_cnt"] / fail_by_idx["cycles"]).round(2)
    fail_by_idx["pct_both_fail"] = (100.0 * fail_by_idx["both_fail_cnt"] / fail_by_idx["cycles"]).round(2)
    fail_by_idx.sort_values("Cycle").to_csv(os.path.join(args.outdir, "fail_by_cycle_index.csv"), index=False)

    # --- 2) Medians/means from GOOD cycles (dataset-level) ---
    good_mask = (allc["Label"].astype(str).str.upper() == "GOOD")
    good = allc[good_mask].copy()
    # metrics to summarize
    metrics = ["Chg_Spec_mAhg","DChg_Spec_mAhg","Missing_Count","IR_proxy","CE_dev_abs"]
    med = {m: np.nanmedian(good[m]) if m in good.columns else np.nan for m in metrics}
    mean = {m: np.nanmean(good[m]) if m in good.columns else np.nan for m in metrics}
    stat_rows = []
    for m in metrics:
        stat_rows.append({"metric": m, "median_GOOD": med[m], "mean_GOOD": mean[m]})
    pd.DataFrame(stat_rows).to_csv(os.path.join(args.outdir, "good_medians_means.csv"), index=False)

    # --- 3) Quantified checks vs medians (4 criteria) for every cycle ---
    # Direction rules:
    # - Chg_Spec_mAhg: pass if <= median_GOOD (lower is safer)
    # - DChg_Spec_mAhg: pass if >= median_GOOD (higher is better)
    # - IR_proxy: pass if <= median_GOOD (lower is better)
    # - CE_dev_abs: pass if <= median_GOOD (lower is better)
    allc["Q_pass_ChgSpec"]  = allc["Chg_Spec_mAhg"] <= med["Chg_Spec_mAhg"]
    allc["Q_pass_DChgSpec"] = allc["DChg_Spec_mAhg"] >= med["DChg_Spec_mAhg"]
    allc["Q_pass_IRproxy"]  = allc["IR_proxy"] <= med["IR_proxy"]
    allc["Q_pass_CEdev"]    = allc["CE_dev_abs"] <= med["CE_dev_abs"]
    allc["Quantified_AllPass"] = allc[["Q_pass_ChgSpec","Q_pass_DChgSpec","Q_pass_IRproxy","Q_pass_CEdev"]].all(axis=1)

    # Save per-cycle table with passes
    keep_cols = [
        "file","Cycle","Label",
        "Chg_Spec_mAhg","DChg_Spec_mAhg","Missing_Count","IR_proxy","CE","CE_dev_abs",
        "Q_pass_ChgSpec","Q_pass_DChgSpec","Q_pass_IRproxy","Q_pass_CEdev","Quantified_AllPass"
    ]
    allc[keep_cols].to_csv(os.path.join(args.outdir, "all_cycles_quantified.csv"), index=False)

    # --- 4) Per-file summary of quantified passes ---
    per_file_quant = (
        allc.groupby("file", dropna=False)
            .agg(
                cycles=("Cycle","size"),
                pass_ChgSpec=("Q_pass_ChgSpec","sum"),
                pass_DChgSpec=("Q_pass_DChgSpec","sum"),
                pass_IRproxy=("Q_pass_IRproxy","sum"),
                pass_CEdev=("Q_pass_CEdev","sum"),
                pass_All=("Quantified_AllPass","sum"),
            )
            .reset_index()
    )
    # add percentages
    for c in ["pass_ChgSpec","pass_DChgSpec","pass_IRproxy","pass_CEdev","pass_All"]:
        per_file_quant[f"pct_{c}"] = (100.0 * per_file_quant[c] / per_file_quant["cycles"]).round(2)
    per_file_quant.to_csv(os.path.join(args.outdir, "per_file_quantified_summary.csv"), index=False)

    print(f"Done. Wrote CSVs to {args.outdir}")

if __name__ == "__main__":
    main()
