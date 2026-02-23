#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step_end_voltage_all_cycles_v1.py

Extract end-of-step voltages (and standby end time) for ALL cycles (GOOD + BAD)
using only the STEP sheet, following your mission pattern:

- Skip formation cycles: cycle_index < 4
- For each cycle: pick 1 charge step, then the next 5 discharge steps AFTER charge
- Name the 5 discharge steps as: take_off, hover, cruise, landing, standby
- Output one row per (file, cycle)
- Never impute with 0 (keep NaN if truly missing)

Optionally merges 3-class cycle labels (good_drone / good_no_drone / bad) from a
consolidated labels CSV (file, Cycle, cycle_label_3name), if provided.

Usage:
  # basic (no labels merge)
  python step_end_voltage_all_cycles_v1.py --originals path/to/original_excels -o step_stats_all

  # with labels merge (recommended)
  python step_end_voltage_all_cycles_v1.py --originals path/to/original_excels \
      --labels_csv drone_labels_out/all_cycle_labels_3class.csv \
      -o step_stats_all
"""

import argparse, glob, os, re, warnings
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
    module="openpyxl.styles.stylesheet",
)

# ------------------ helpers ------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    cur = {_norm(c): c for c in df.columns}
    ren = {}
    for canon, cand_list in mapping.items():
        for cand in cand_list:
            key = _norm(cand)
            if key in cur:
                ren[cur[key]] = canon
                break
    return df.rename(columns=ren)

def load_step_sheet(xls: pd.ExcelFile) -> Optional[pd.DataFrame]:
    name = None
    for s in xls.sheet_names:
        if s.lower() == "step":
            name = s
            break
    if name is None:
        return None

    df = pd.read_excel(xls, sheet_name=name)

    mapping = {
        "cycle_index":     ["Cycle Index","Cycle","cycle index","CycleIndex"],
        "step_index":      ["Step Index","StepIndex","Idx"],
        "step_number":     ["Step Number","StepNumber","Step No.","StepNo"],
        "step_type":       ["Step Type","StepType","Type","Step"],
        "step_time_h":     ["Step Time(h)","Step Time (h)","Time(h)","Time (h)"],
        "onset":           ["Oneset Date","Onset Date","Start Date","Oneset"],
        "end":             ["End Date","EndDate","End time","End"],
        "start_voltage_v": ["Oneset Volt.(V)","Onset Voltage(V)","Start Voltage(V)","Oneset Volt (V)"],
        "end_voltage_v":   ["End Voltage(V)","End Voltage (V)","End Volt.(V)","EndVolt(V)"],
        "chg_cap_ah":      ["Chg. Cap.(Ah)","Charge Capacity(Ah)","Chg Cap (Ah)","Chg.Cap.(Ah)"],
        "dchg_cap_ah":     ["DChg. Cap.(Ah)","Discharge Capacity(Ah)","DChg Cap (Ah)","DChg.Cap.(Ah)"],
    }
    df = remap_columns(df, mapping)

    # numeric columns
    for c in ["cycle_index","step_index","step_number","step_time_h",
              "start_voltage_v","end_voltage_v","chg_cap_ah","dchg_cap_ah"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # datetime columns
    for c in ["onset","end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df

def is_charge_row(cdf: pd.DataFrame) -> pd.Series:
    """
    Prefer capacity-based detection; fallback to text.
    """
    if "chg_cap_ah" in cdf.columns:
        s = pd.to_numeric(cdf["chg_cap_ah"], errors="coerce").fillna(0)
        return s > 0
    txt = cdf.get("step_type", pd.Series([""]*len(cdf))).astype(str).str.lower()
    return txt.str.contains("chg") & (~txt.str.contains("dchg")) & (~txt.str.contains("discharge"))

def is_discharge_row(cdf: pd.DataFrame) -> pd.Series:
    """
    Prefer capacity-based detection; fallback to text.
    """
    if "dchg_cap_ah" in cdf.columns:
        s = pd.to_numeric(cdf["dchg_cap_ah"], errors="coerce").fillna(0)
        return s > 0
    txt = cdf.get("step_type", pd.Series([""]*len(cdf))).astype(str).str.lower()
    return txt.str.contains("dchg") | txt.str.contains("discharge")

def pick_steps_from_step_sheet(step_df: pd.DataFrame, cycle: int) -> Optional[List[dict]]:
    """
    Pick 1 charge + 5 discharge AFTER charge.
    Returns list of dicts: [{'name': 'charge', ...}, {'name': 'take_off', ...}, ...]
    """
    cdf = step_df[step_df["cycle_index"] == cycle].copy()
    if cdf.empty:
        return None

    # sort to preserve temporal order
    sort_cols = []
    if "onset" in cdf.columns: sort_cols.append("onset")
    if "end" in cdf.columns: sort_cols.append("end")
    if "step_number" in cdf.columns: sort_cols.append("step_number")
    if sort_cols:
        cdf = cdf.sort_values(sort_cols, na_position="last")

    chg_rows = cdf[is_charge_row(cdf)]
    if chg_rows.empty:
        return None

    chg = chg_rows.iloc[0]
    chg_pos = chg.name

    # rows after that charge row
    after = cdf.loc[chg_pos+1:] if chg_pos in cdf.index else cdf.iloc[0:0]
    d_after = after[is_discharge_row(after)]
    if len(d_after) < 5:
        return None

    d5 = d_after.iloc[:5]

    out = [{"name": "charge", **chg.to_dict()}]
    for nm, (_, r) in zip(["take_off","hover","cruise","landing","standby"], d5.iterrows()):
        x = r.to_dict()
        x["name"] = nm
        out.append(x)
    return out

def load_labels_csv(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    need = {"file", "Cycle", "cycle_label_3name"}
    if not need.issubset(df.columns):
        raise ValueError(f"labels_csv must contain columns {sorted(need)}")
    df = df.copy()
    df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce")
    df = df.dropna(subset=["Cycle"])
    df["Cycle"] = df["Cycle"].astype(int)
    return df

# ------------------ processing ------------------

def process_one(original_path: str, labels_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    base = os.path.basename(original_path)
    try:
        xls = pd.ExcelFile(original_path)
        step_df = load_step_sheet(xls)
        if step_df is None or step_df.empty:
            print(f"[WARN] {base}: missing/empty 'step' sheet; skipping")
            return None

        # cycles >= 4
        step_df = step_df[step_df["cycle_index"].notna()].copy()
        step_df["cycle_index"] = step_df["cycle_index"].astype(int)
        cycles = sorted([c for c in step_df["cycle_index"].unique().tolist() if c >= 4])
        if not cycles:
            print(f"[INFO] {base}: no cycles >= 4; skipping")
            return None

        rows = []
        for c in cycles:
            steps = pick_steps_from_step_sheet(step_df, c)
            if steps is None:
                # As requested earlier: some files have only 4 discharge steps → you may skip those cycles
                # If you want to skip the whole file instead, we can enforce that elsewhere.
                continue

            row = {"file": base, "Cycle": int(c)}

            # attach label if provided
            if labels_df is not None:
                m = labels_df[(labels_df["file"] == base) & (labels_df["Cycle"] == int(c))]
                if not m.empty:
                    row["cycle_label_3name"] = m["cycle_label_3name"].iloc[0]

            for st in steps:
                nm = st["name"]
                ev = st.get("end_voltage_v", np.nan)
                row[f"{nm}_V_end"] = float(ev) if pd.notna(ev) else np.nan

                if nm == "standby":
                    th = st.get("step_time_h", np.nan)
                    row["standby_t_end_h"] = float(th) if pd.notna(th) else np.nan

            # optional: landing_V_rel_global (define relative to charge end voltage)
            # Adjust definition if you used something else.
            if pd.notna(row.get("landing_V_end", np.nan)) and pd.notna(row.get("charge_V_end", np.nan)):
                row["landing_V_rel_global"] = float(row["landing_V_end"] - row["charge_V_end"])
            else:
                row["landing_V_rel_global"] = np.nan

            rows.append(row)

        if not rows:
            print(f"[INFO] {base}: no cycles with complete 1+5 pattern")
            return None

        df = pd.DataFrame(rows).sort_values(["file", "Cycle"]).reset_index(drop=True)
        return df

    except Exception as e:
        print(f"[FAIL] {base}: {e}")
        return None

def main():
    ap = argparse.ArgumentParser(description="Compute end-of-step voltages for ALL cycles using STEP sheet.")
    ap.add_argument("--originals", required=True, help="Folder with original .xlsx files")
    ap.add_argument("--labels_csv", default=None, help="Optional consolidated labels CSV (file, Cycle, cycle_label_3name)")
    ap.add_argument("-o", "--outdir", default="step_stats_all", help="Output folder")
    args = ap.parse_args()

    labels_df = None
    if args.labels_csv:
        labels_df = load_labels_csv(args.labels_csv)

    files = sorted(glob.glob(os.path.join(args.originals, "*.xlsx")))
    if not files:
        print(f"No .xlsx files found in {args.originals}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    all_rows = []
    for of in files:
        df = process_one(of, labels_df)
        if df is not None and not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No data collected.")
        return

    ALL = pd.concat(all_rows, ignore_index=True)

    # write combined CSV
    out_csv = os.path.join(args.outdir, "ALL_step_ends_all_cycles.csv")
    ALL.to_csv(out_csv, index=False)

    # global summary
    step_cols = [c for c in ALL.columns if c.endswith("_V_end")]
    add_cols = step_cols + (["standby_t_end_h"] if "standby_t_end_h" in ALL.columns else []) + (["landing_V_rel_global"] if "landing_V_rel_global" in ALL.columns else [])
    gsum = [{"metric": col, "mean": float(np.nanmean(ALL[col])), "median": float(np.nanmedian(ALL[col]))} for col in add_cols]
    pd.DataFrame(gsum).to_csv(os.path.join(args.outdir, "GLOBAL_mean_median_all_cycles.csv"), index=False)

    print(f"[DONE] Wrote:\n - {out_csv}\n - {os.path.join(args.outdir, 'GLOBAL_mean_median_all_cycles.csv')}")

if __name__ == "__main__":
    main()
