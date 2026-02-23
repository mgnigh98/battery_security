#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_early_window_features_1s2s_trunc200.py

What it does:
  - Reads your cycle-level labeled CSV (e.g., ALL_cycles_3class.csv)
  - Drops any rows with Cycle > 200
  - For each (file, Cycle), opens the original .xlsx and reads the 'record' sheet
  - Extracts early-window discharge features for windows: 1s and 2s
  - Saves merged output CSV:
      ALL_cycles_3class_early_1s2s_trunc200.csv

Run:
  python extract_early_window_features_1s2s_trunc200.py \
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


# ---------------- CONFIG ---------------- #
CONFIG: Dict[str, any] = {
    # Only 1s, 2s windows (as you requested)
    "WINDOWS_SEC": [1.0, 2.0],

    # Drop any cycle index > this threshold (as you requested)
    "MAX_CYCLE": 200,

    # Thresholds/flags
    "IR_PROXY_DRONE_MAX": 0.442,
    "V_SAG_COLLAPSE_THRESH": 0.15,   # volts
    "DISCHARGE_I_THRESHOLD": -1e-3,  # Current(A) < this => discharge

    "RECORD_SHEET_CANDIDATES": ["record", "Record", "RECORD"],
}


# -------------- Helpers -------------- #
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    cur = {_norm(c): c for c in df.columns}
    ren: Dict[str, str] = {}
    for canon, cand_list in mapping.items():
        for cand in cand_list:
            key = _norm(cand)
            if key in cur:
                ren[cur[key]] = canon
                break
    return df.rename(columns=ren)


def find_record_sheet(xls: pd.ExcelFile) -> Optional[str]:
    lower_map = {s.lower(): s for s in xls.sheet_names}
    for cand in CONFIG["RECORD_SHEET_CANDIDATES"]:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for s in xls.sheet_names:
        if "record" in s.lower():
            return s
    return None


def match_original_from_labeled_name(labeled_or_3class_name: str, originals_dir: str) -> Optional[str]:
    base = os.path.basename(labeled_or_3class_name)

    stem = base
    for suf in ["_labeled_3class.xlsx", "_labeled.xlsx"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    if stem.endswith(".xlsx"):
        stem = stem[:-5]

    target_norm = _norm(stem)
    for f in glob.glob(os.path.join(originals_dir, "*.xlsx")):
        s = os.path.splitext(os.path.basename(f))[0]
        if _norm(s) == target_norm:
            return f
    return None


def compute_dvdt(voltage: np.ndarray, time_s: np.ndarray) -> np.ndarray:
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
    if len(voltage) < 2:
        return 0.0
    power = voltage * current
    return float(np.trapz(power, time_s))


def extract_early_features_for_cycle(rec_df_cycle: pd.DataFrame, windows: List[float]) -> Dict[str, float]:
    # Discharge detection
    dmask = rec_df_cycle["current_a"] < CONFIG["DISCHARGE_I_THRESHOLD"]
    if "step_type" in rec_df_cycle.columns:
        st = rec_df_cycle["step_type"].astype(str).str.lower()
        dmask = dmask | st.str.contains("dchg") | st.str.contains("discharge")

    dch = rec_df_cycle[dmask].copy()
    if dch.empty:
        return {f"early{int(T)}_missing": 1 for T in windows}

    # Re-base time
    t0 = float(dch["test_time_s"].iloc[0])
    dch["t_rel"] = dch["test_time_s"] - t0

    t_all = dch["t_rel"].to_numpy(dtype=float)
    v_all = dch["voltage_v"].to_numpy(dtype=float)
    i_all = dch["current_a"].to_numpy(dtype=float)

    dvdt_all = compute_dvdt(v_all, t_all)

    out: Dict[str, float] = {}

    for T in windows:
        prefix = f"early{int(T)}"
        wmask = t_all <= T

        if not np.any(wmask):
            out[f"{prefix}_missing"] = 1
            for name in [
                "V_mean", "V_min", "V_sag", "V_sag_ratio",
                "dvdt_mean", "dvdt_std",
                "I_mean", "I_std",
                "IR_early",
                "power_mean", "power_std",
                "energy_ws", "energy_per_I",
                "voltage_collapse_flag",
                "IR_high_flag",
                "dvdt_fluct_flag",
            ]:
                out[f"{prefix}_{name}"] = np.nan
            continue

        t = t_all[wmask]
        v = v_all[wmask]
        i = i_all[wmask]
        dvdt = dvdt_all[wmask]

        V0 = float(v[0])
        V_mean = float(np.nanmean(v))
        V_min = float(np.nanmin(v))
        V_sag = V0 - V_min
        V_sag_ratio = V_sag / V0 if V0 != 0 else np.nan

        dvdt_mean = float(np.nanmean(dvdt))
        dvdt_std = float(np.nanstd(dvdt))

        I_mean = float(np.nanmean(i))
        I_std = float(np.nanstd(i))

        IR_early = (V_sag / abs(I_mean)) if (not np.isnan(I_mean) and abs(I_mean) > 1e-6) else np.nan

        power = v * i
        power_mean = float(np.nanmean(power))
        power_std = float(np.nanstd(power))
        energy_ws = integrate_energy_wattsec(v, i, t)
        energy_per_I = (energy_ws / abs(I_mean)) if (not np.isnan(I_mean) and abs(I_mean) > 1e-6) else np.nan

        voltage_collapse_flag = int(V_sag > CONFIG["V_SAG_COLLAPSE_THRESH"])
        IR_high_flag = int(not np.isnan(IR_early) and IR_early > CONFIG["IR_PROXY_DRONE_MAX"])
        dvdt_fluct_flag = int(
            (not np.isnan(dvdt_std) and dvdt_std > abs(dvdt_mean) * 2.0) or
            (not np.isnan(dvdt_std) and dvdt_std > 0.01)
        )

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", required=True, help="Folder with original .xlsx battery files")
    ap.add_argument("--cycles_csv", required=True, help="Path to ALL_cycles_3class.csv")
    ap.add_argument("-o", "--outdir", default=None, help="Output folder (default: directory of cycles_csv)")
    args = ap.parse_args()

    originals_dir = args.originals
    cycles_csv = args.cycles_csv
    outdir = args.outdir or os.path.dirname(os.path.abspath(cycles_csv)) or "."
    os.makedirs(outdir, exist_ok=True)

    ALL = pd.read_csv(cycles_csv)
    if "file" not in ALL.columns or "Cycle" not in ALL.columns:
        raise ValueError("cycles_csv must have 'file' and 'Cycle' columns.")

    # --------- TRUNCATE CYCLES HERE ---------
    ALL["Cycle"] = pd.to_numeric(ALL["Cycle"], errors="coerce")
    before = len(ALL)
    ALL = ALL[ALL["Cycle"].notna()].copy()
    ALL = ALL[ALL["Cycle"] <= CONFIG["MAX_CYCLE"]].copy()
    after = len(ALL)
    print(f"[INFO] Truncate: kept Cycle <= {CONFIG['MAX_CYCLE']} | rows {before} -> {after}")

    feat_rows = []

    for file_name, df_file in ALL.groupby("file"):
        cycles = sorted(df_file["Cycle"].unique().tolist())
        print(f"\n[PROCESS] {file_name} | cycles={len(cycles)} | min={cycles[0]} max={cycles[-1]}")

        orig_path = match_original_from_labeled_name(file_name, originals_dir)
        if not orig_path:
            print(f"[WARN] No original .xlsx match for {file_name}; early features missing for these cycles.")
            for cyc in cycles:
                feat_rows.append({"file": file_name, "Cycle": cyc})
            continue

        try:
            xls = pd.ExcelFile(orig_path)
        except Exception as e:
            print(f"[WARN] Failed to open {orig_path}: {e}")
            for cyc in cycles:
                feat_rows.append({"file": file_name, "Cycle": cyc})
            continue

        rec_sheet = find_record_sheet(xls)
        if rec_sheet is None:
            print(f"[WARN] No 'record' sheet in {orig_path}; early features missing for these cycles.")
            for cyc in cycles:
                feat_rows.append({"file": file_name, "Cycle": cyc})
            continue

        rec_df = pd.read_excel(xls, sheet_name=rec_sheet)

        mapping = {
            "cycle_index": ["Cycle Index", "Cycle", "cycle_index"],
            "time_h": ["Time(h)", "Time (h)", "Step Time(h)", "Step Time (h)"],
            "test_time_s": [
                "Test_Time(s)", "Test Time(s)", "Test_Time (s)", "Test Time (s)",
                "Step_Time(s)", "Step Time(s)", "Step_Time (s)", "Step Time (s)",
                "Time(s)", "Time (s)", "TestTime(s)", "TestTime (s)",
                "Elapsed Time(s)", "Elapsed_Time(s)", "Elapsed Time (s)"
            ],
            "voltage_v": ["Voltage(V)", "Voltage (V)", "Voltage"],
            "current_a": ["Current(A)", "Current (A)", "Current"],
            "step_type": ["Step Type", "Step", "Mode"],
        }
        rec_df = remap_columns(rec_df, mapping)

        # Convert hours -> seconds if needed
        if "test_time_s" not in rec_df.columns and "time_h" in rec_df.columns:
            rec_df["time_h"] = pd.to_numeric(rec_df["time_h"], errors="coerce")
            rec_df["test_time_s"] = rec_df["time_h"] * 3600.0

        needed = ["cycle_index", "test_time_s", "voltage_v", "current_a"]
        missing = [c for c in needed if c not in rec_df.columns]
        if missing:
            print(f"[WARN] Missing columns {missing} in {orig_path}; early features missing for these cycles.")
            for cyc in cycles:
                feat_rows.append({"file": file_name, "Cycle": cyc})
            continue

        for c in ["cycle_index", "test_time_s", "voltage_v", "current_a"]:
            rec_df[c] = pd.to_numeric(rec_df[c], errors="coerce")

        for cyc in cycles:
            row = {"file": file_name, "Cycle": cyc}
            rec_df_cycle = rec_df[rec_df["cycle_index"] == cyc].dropna(
                subset=["test_time_s", "voltage_v", "current_a"]
            )

            if rec_df_cycle.empty:
                print(f"[WARN] file={file_name}, Cycle={cyc}: no record rows; skipping early features.")
            else:
                feats = extract_early_features_for_cycle(rec_df_cycle, CONFIG["WINDOWS_SEC"])
                row.update(feats)

            feat_rows.append(row)

    FEAT = pd.DataFrame(feat_rows)
    merged = ALL.merge(FEAT, on=["file", "Cycle"], how="left")

    out_csv = os.path.join(outdir, "ALL_cycles_3class_early_1s2s_trunc200.csv")
    merged.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote: {out_csv} | rows={len(merged)}")


if __name__ == "__main__":
    main()
