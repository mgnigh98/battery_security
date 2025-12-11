#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_early_window_features.py

Goal:
  For each (file, Cycle) in ALL_cycles_3class.csv, open the corresponding
  original .xlsx battery file, read the 'record' sheet, and compute
  early-window features from the DISCHARGE part of that cycle.

  Windows: T = 20s, 30s, 50s, 60s  (configurable).
  Output: ALL_cycles_3class_early.csv with early features merged in.

Assumptions:
  - Original .xlsx files are the same ones used in cycle_label_v4.py.
  - 'record' sheet exists (case-insensitive) and contains:
      * Cycle Index (or similar)
      * Test_Time(s) or Step_Time(s)
      * Current(A)
      * Voltage(V)
  - Discharge rows have negative current (Current(A) < 0) OR
    step type contains 'dchg' / 'discharge' (if available).

Usage:
  python extract_early_window_features.py \
      --originals path/to/original_excels \
      --cycles_csv path/to/ALL_cycles_3class.csv \
      -o path/to/outdir
"""

import os
import re
import glob
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------- CONFIG (tweak here) ---------------- #

CONFIG: Dict[str, any] = {
    # Early windows in seconds:
    "WINDOWS_SEC": [5.0, 10.0, 20.0, 30.0, 50.0, 60.0],

    # Thresholds:
    "IR_PROXY_DRONE_MAX": 0.442,      # for early_IR_flag
    "V_SAG_COLLAPSE_THRESH": 0.15,    # V; sag > this => collapse flag

    # Discharge detection:
    "DISCHARGE_I_THRESHOLD": -1e-3,   # Current(A) < this => discharge

    # Sheet name finding:
    "RECORD_SHEET_CANDIDATES": ["record", "Record", "RECORD"],
}

# -------------- Helpers -------------- #

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Remap potentially varying column names to canonical ones using a
    list-of-candidates approach, similar to other scripts.
    """
    cur = { _norm(c): c for c in df.columns }
    ren: Dict[str, str] = {}
    for canon, cand_list in mapping.items():
        for cand in cand_list:
            key = _norm(cand)
            if key in cur:
                ren[cur[key]] = canon
                break
    return df.rename(columns=ren)

def find_record_sheet(xls: pd.ExcelFile) -> Optional[str]:
    """
    Find a sheet whose name matches one of RECORD_SHEET_CANDIDATES.
    """
    lower_map = {s.lower(): s for s in xls.sheet_names}
    for cand in CONFIG["RECORD_SHEET_CANDIDATES"]:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # fallback: try something that contains 'record'
    for s in xls.sheet_names:
        if "record" in s.lower():
            return s
    return None

def match_original_from_labeled_name(labeled_or_3class_name: str,
                                     originals_dir: str) -> Optional[str]:
    """
    Given a labeled or 3class filename (e.g., ABC.xlsx_labeled.xlsx
    or ABC.xlsx_labeled_3class.xlsx), try to find the corresponding
    original .xlsx in originals_dir.

    Strategy:
      - Strip suffixes "_labeled_3class.xlsx", "_labeled.xlsx", ".xlsx" in order.
      - Then match by normalized stem against any *.xlsx in originals_dir.
    """
    base = os.path.basename(labeled_or_3class_name)

    # strip known suffixes
    stem = base
    for suf in ["_labeled_3class.xlsx", "_labeled.xlsx"]:
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    if stem.endswith(".xlsx"):
        stem = stem[:-5]

    target_norm = _norm(stem)

    cands = glob.glob(os.path.join(originals_dir, "*.xlsx"))
    for f in cands:
        s = os.path.splitext(os.path.basename(f))[0]
        if _norm(s) == target_norm:
            return f
    return None

def compute_dvdt(voltage: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """
    Compute dv/dt using finite differences. Output length = len(voltage).
    """
    if len(voltage) < 2:
        return np.zeros_like(voltage)
    dt = np.diff(time_s)
    dt[dt == 0] = np.nan
    dv = np.diff(voltage)
    dvdt = np.empty_like(voltage)
    dvdt[0] = np.nan
    dvdt[1:] = dv / dt
    return dvdt

def integrate_energy_wattsec(voltage: np.ndarray, current: np.ndarray, time_s: np.ndarray) -> float:
    """
    Numerical integration of power = V*I over time; returns energy in watt-seconds (J).
    Use trapezoidal rule.
    """
    if len(voltage) < 2:
        return 0.0
    power = voltage * current
    return float(np.trapz(power, time_s))

# -------------- Core feature extraction -------------- #

def extract_early_features_for_cycle(
    rec_df_cycle: pd.DataFrame,
    windows: List[float]
) -> Dict[str, float]:
    """
    Given 'record' rows for a single cycle, compute early-window features
    from the DISCHARGE portion. Returns a flat dict with columns like:
      early20_V_mean, early20_V_sag, ... etc.
    """

    # Identify discharge rows
    # Priority 1: negative current
    dmask = rec_df_cycle["current_a"] < CONFIG["DISCHARGE_I_THRESHOLD"]

    if "step_type" in rec_df_cycle.columns:
        st = rec_df_cycle["step_type"].astype(str).str.lower()
        dmask = dmask | st.str.contains("dchg") | st.str.contains("discharge")

    dch = rec_df_cycle[dmask].copy()
    if dch.empty:
        # No discharge rows found; return NaNs
        return {f"early{int(T)}_missing" : 1 for T in windows}

    # Re-base time
    t0 = float(dch["test_time_s"].iloc[0])
    dch["t_rel"] = dch["test_time_s"] - t0

    # For dv/dt and energy integration, we want numpy arrays
    t_all = dch["t_rel"].to_numpy(dtype=float)
    v_all = dch["voltage_v"].to_numpy(dtype=float)
    i_all = dch["current_a"].to_numpy(dtype=float)

    dvdt_all = compute_dvdt(v_all, t_all)

    out: Dict[str, float] = {}

    for T in windows:
        wmask = t_all <= T
        if not np.any(wmask):
            # No points in this window – mark as missing
            prefix = f"early{int(T)}"
            out[f"{prefix}_missing"] = 1
            # also fill numeric fields with NaNs
            for name in [
                "V_mean", "V_min", "V_sag", "V_sag_ratio",
                "dvdt_mean", "dvdt_std",
                "I_mean", "I_std",
                "IR_early",
                "power_mean", "power_std",
                "energy_ws", "energy_per_I",
                "voltage_collapse_flag",
                "IR_high_flag",
                "dvdt_fluct_flag"
            ]:
                out[f"{prefix}_{name}"] = np.nan
            continue

        prefix = f"early{int(T)}"
        t = t_all[wmask]
        v = v_all[wmask]
        i = i_all[wmask]
        dvdt = dvdt_all[wmask]

        # Basic stats
        V0 = float(v[0])
        V_mean = float(np.nanmean(v))
        V_min = float(np.nanmin(v))
        V_sag = V0 - V_min
        V_sag_ratio = V_sag / V0 if V0 != 0 else np.nan

        dvdt_mean = float(np.nanmean(dvdt))
        dvdt_std = float(np.nanstd(dvdt))

        I_mean = float(np.nanmean(i))
        I_std = float(np.nanstd(i))

        # IR_early ~ V_sag / |I_mean|
        if not np.isnan(I_mean) and abs(I_mean) > 1e-6:
            IR_early = V_sag / abs(I_mean)
        else:
            IR_early = np.nan

        # Power, energy
        power = v * i
        power_mean = float(np.nanmean(power))
        power_std = float(np.nanstd(power))
        energy_ws = integrate_energy_wattsec(v, i, t)

        # Energy normalized by |I_mean| for some scale (optional)
        if not np.isnan(I_mean) and abs(I_mean) > 1e-6:
            energy_per_I = energy_ws / abs(I_mean)
        else:
            energy_per_I = np.nan

        # Flags
        voltage_collapse_flag = int(V_sag > CONFIG["V_SAG_COLLAPSE_THRESH"])
        IR_high_flag = int(not np.isnan(IR_early) and IR_early > CONFIG["IR_PROXY_DRONE_MAX"])
        # dv/dt fluctuation: large std relative to mean, or absolute std over threshold
        dvdt_fluct_flag = int(
            (not np.isnan(dvdt_std) and dvdt_std > abs(dvdt_mean) * 2.0) or
            (not np.isnan(dvdt_std) and dvdt_std > 0.01)
        )

        # Pack output
        out[f"{prefix}_missing"] = 0
        out[f"{prefix}_V_mean"] = V_mean
        out[f"{prefix}_V_min"] = V_min
        out[f"{prefix}_V_sag"] = V_sag
        out[f"{prefix}_V_sag_ratio"] = V_sag_ratio
        out[f"{prefix}_dvdt_mean"] = dvdt_mean
        out[f"{prefix}_dvdt_std"] = dvdt_std
        out[f"{prefix}_I_mean"] = I_mean
        out[f"{prefix}_I_std"] = I_std
        out[f"{prefix}_IR_early"] = IR_early
        out[f"{prefix}_power_mean"] = power_mean
        out[f"{prefix}_power_std"] = power_std
        out[f"{prefix}_energy_ws"] = energy_ws
        out[f"{prefix}_energy_per_I"] = energy_per_I
        out[f"{prefix}_voltage_collapse_flag"] = voltage_collapse_flag
        out[f"{prefix}_IR_high_flag"] = IR_high_flag
        out[f"{prefix}_dvdt_fluct_flag"] = dvdt_fluct_flag

    return out

# -------------- Main -------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Extract early-window features from 'record' sheets for each "
                    "(file, Cycle) in ALL_cycles_3class.csv"
    )
    ap.add_argument("--originals", required=True,
                    help="Folder with original .xlsx battery files")
    ap.add_argument("--cycles_csv", required=True,
                    help="Path to ALL_cycles_3class.csv from 3-class labeling")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Output folder (default: directory of cycles_csv)")
    args = ap.parse_args()

    originals_dir = args.originals
    cycles_csv = args.cycles_csv
    outdir = args.outdir or os.path.dirname(os.path.abspath(cycles_csv)) or "."
    os.makedirs(outdir, exist_ok=True)

    # Load cycles table
    ALL = pd.read_csv(cycles_csv)
    if "file" not in ALL.columns or "Cycle" not in ALL.columns:
        raise ValueError("ALL_cycles_3class.csv must have 'file' and 'Cycle' columns.")

    # We'll build a feature dict keyed by (file, Cycle)
    feat_rows = []

    # Process per file for efficiency
    for file_name, df_file in ALL.groupby("file"):
        print(f"\n[PROCESS] file group: {file_name} (cycles={len(df_file['Cycle'].unique())})")

        orig_path = match_original_from_labeled_name(file_name, originals_dir)
        if not orig_path:
            print(f"[WARN] No original .xlsx match found for {file_name}; "
                  f"early features will be missing for these cycles.")
            for cyc in df_file["Cycle"].unique():
                feat_rows.append({"file": file_name, "Cycle": cyc})
            continue

        print(f"[INFO] Matched to original: {os.path.basename(orig_path)}")

        try:
            xls = pd.ExcelFile(orig_path)
        except Exception as e:
            print(f"[WARN] Failed to open {orig_path}: {e}")
            for cyc in df_file["Cycle"].unique():
                feat_rows.append({"file": file_name, "Cycle": cyc})
            continue

        rec_sheet = find_record_sheet(xls)
        if rec_sheet is None:
            print(f"[WARN] No 'record' sheet found in {orig_path}; "
                  f"early features missing for these cycles.")
            for cyc in df_file["Cycle"].unique():
                feat_rows.append({"file": file_name, "Cycle": cyc})
            continue

        rec_df = pd.read_excel(xls, sheet_name=rec_sheet)

        # Remap columns
        # mapping = {
        #     "cycle_index": ["Cycle Index", "Cycle", "cycle_index"],
        #     "test_time_s": ["Test_Time(s)", "Test Time(s)", "Step_Time(s)", "Step Time(s)", "TestTime(s)"],
        #     "voltage_v": ["Voltage(V)", "Voltage (V)", "Voltage"],
        #     "current_a": ["Current(A)", "Current (A)", "Current"],
        #     "step_type": ["Step Type", "Step", "Mode"],
        # }
        # rec_df = remap_columns(rec_df, mapping)
        mapping = {
            "cycle_index": ["Cycle Index", "Cycle", "cycle_index"],
            # time in HOURS; we'll convert to seconds later
            "time_h": ["Time(h)", "Time (h)", "Step Time(h)", "Step Time (h)"],
            # keep this in case some files actually have seconds
            "test_time_s": ["Test_Time(s)", "Test Time(s)", "Test_Time (s)", "Test Time (s)",
                            "Step_Time(s)", "Step Time(s)", "Step_Time (s)", "Step Time (s)",
                            "Time(s)", "Time (s)", "TestTime(s)", "TestTime (s)",
                            "Elapsed Time(s)", "Elapsed_Time(s)", "Elapsed Time (s)"],
            "voltage_v": ["Voltage(V)", "Voltage (V)", "Voltage"],
            "current_a": ["Current(A)", "Current (A)", "Current"],
            "step_type": ["Step Type", "Step", "Mode"],
        }
        rec_df = remap_columns(rec_df, mapping)

        # --- Convert time from hours to seconds if needed ---
        if "test_time_s" not in rec_df.columns and "time_h" in rec_df.columns:
            # ensure numeric
            rec_df["time_h"] = pd.to_numeric(rec_df["time_h"], errors="coerce")
            rec_df["test_time_s"] = rec_df["time_h"] * 3600.0

        # Ensure needed columns exist
        needed = ["cycle_index", "test_time_s", "voltage_v", "current_a"]
        missing = [c for c in needed if c not in rec_df.columns]
        if missing:
            print(f"[WARN] Missing columns {missing} in record sheet of {orig_path}; "
                  f"early features missing for these cycles.")
            for cyc in df_file["Cycle"].unique():
                feat_rows.append({"file": file_name, "Cycle": cyc})
            continue

        # Force numeric
        for c in ["cycle_index", "test_time_s", "voltage_v", "current_a"]:
            rec_df[c] = pd.to_numeric(rec_df[c], errors="coerce")

        # For each cycle in this file, extract features
        for cyc in sorted(df_file["Cycle"].unique()):
            cyc_mask = rec_df["cycle_index"] == cyc
            rec_df_cycle = rec_df[cyc_mask].dropna(subset=["test_time_s", "voltage_v", "current_a"])
            row = {"file": file_name, "Cycle": cyc}

            if rec_df_cycle.empty:
                print(f"[WARN] file={file_name}, Cycle={cyc}: no record rows; skipping early features.")
            else:
                feats = extract_early_features_for_cycle(rec_df_cycle, CONFIG["WINDOWS_SEC"])
                row.update(feats)

            feat_rows.append(row)

    # Build features dataframe
    FEAT = pd.DataFrame(feat_rows)

    # Merge back into ALL
    merged = ALL.merge(FEAT, on=["file", "Cycle"], how="left")

    out_csv = os.path.join(outdir, "ALL_cycles_3class_early.csv")
    merged.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote {out_csv} with {len(merged)} rows")


if __name__ == "__main__":
    main()
