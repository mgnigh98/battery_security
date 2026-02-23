#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_early_windows_into_main.py

Merges early-window columns (early1_*, early2_*, ...) from one or more CSVs into a base CSV.

Join keys (in order):
  - file + Cycle
If Cycle is not present, tries: Cycle_Index

Example:
  python merge_early_windows_into_main.py \
    --base drone_labels_out/ALL_cycles_3class_early.csv \
    --early drone_labels_out/ALL_cycles_3class_early_1s2s_trunc200.csv \
    --out  drone_labels_out/ALL_cycles_3class_early_merged.csv
"""

import argparse
import pandas as pd
import re

EARLY_RE = re.compile(r"^early(\d+)_", re.IGNORECASE)

def find_cycle_col(df):
    for c in ["Cycle", "Cycle_Index", "cycle", "cycle_index"]:
        if c in df.columns:
            return c
    return None

def early_cols(df):
    return [c for c in df.columns if EARLY_RE.match(str(c))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base CSV (kept rows).")
    ap.add_argument("--early", required=True, nargs="+", help="One or more CSVs containing earlyX_* columns.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    file_col = "file" if "file" in base.columns else None
    cyc_col = find_cycle_col(base)

    if file_col is None or cyc_col is None:
        raise ValueError("Base CSV must contain 'file' and a cycle column (Cycle or Cycle_Index).")

    key = [file_col, cyc_col]

    merged = base.copy()

    for p in args.early:
        e = pd.read_csv(p)

        if "file" not in e.columns:
            raise ValueError(f"{p} missing 'file' column")
        e_cyc = find_cycle_col(e)
        if e_cyc is None:
            raise ValueError(f"{p} missing cycle column (Cycle or Cycle_Index)")

        e_key = ["file", e_cyc]
        cols = early_cols(e)
        if not cols:
            print(f"[WARN] {p} has no earlyX_* columns. Skipping.")
            continue

        # keep only keys + early columns, drop duplicate rows by averaging if needed
        e_small = e[e_key + cols].copy()
        e_small = e_small.groupby(e_key, as_index=False).mean(numeric_only=True)

        # rename cycle col to match base if needed
        if e_cyc != cyc_col:
            e_small = e_small.rename(columns={e_cyc: cyc_col})

        # merge in
        merged = merged.merge(e_small, on=key, how="left", suffixes=("", "_dup"))

        # if any duplicates created (same col name), resolve by taking non-null from dup then drop dup
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        for dc in dup_cols:
            orig = dc[:-4]
            if orig in merged.columns:
                merged[orig] = merged[orig].fillna(merged[dc])
            merged = merged.drop(columns=[dc])

        print(f"[OK] Merged {p}: added {len(cols)} early columns")

    merged.to_csv(args.out, index=False)
    print(f"[DONE] Wrote {args.out} (rows={len(merged)}, cols={len(merged.columns)})")

if __name__ == "__main__":
    main()
