#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_3class_drone_v1.py

Takes:
  - cycle-level labels from cycle_label_v4.py  ( *_labeled.xlsx )
  - step-level end-voltages from good_step_end_voltage_from_step_v3.py
    ( *_good_steps_v3.xlsx + GLOBAL_mean_median_v3.csv )

Produces:
  - *_labeled_3class.xlsx per battery (cycle-level 3-class labels + features)
  - battery_level_3class_summary.csv (one row per battery)
  - ALL_cycles_3class.csv (all cycles across all batteries, ready for ML)

Classes:
  cycle_label_3class:
    0 = BAD (unsuitable)
    1 = good but NOT drone-ready
    2 = GOOD for drones

  battery_label_3class:
    0 = BAD battery
    1 = good but NOT drone-ready
    2 = GOOD for drones
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import re
from typing import Dict, Any, Optional

# ----------------- CONFIG (tweak here) -----------------
CONFIG: Dict[str, Any] = {
    # Drone fitness at cycle level
    "IR_PROXY_DRONE_MAX": 0.442,   # <= this is "low IR"
    "DCHG_SPEC_DRONE_MIN": 140.0,  # mAh/g
    "CE_DRONE_LOW": 0.97,
    "CE_DRONE_HIGH": 1.03,
    "ENDV_MARGIN": 0.05,           # allowed below global median (V)

    # Battery-level aggregation (FLEXIBLE)
    # BAD if > BAD_RATIO_MAX of cycles are class 0
    "BAD_RATIO_MAX": 0.20,   # you can change to 0.10, 0.30, etc.

    # Drone-ready if >= DRONE_RATIO_MIN of cycles are class 2
    "DRONE_RATIO_MIN": 0.80, # you can change to 0.70, 0.90, etc.

    # Filenames / sheet names
    "GLOBAL_STEP_STATS_FILE": "GLOBAL_mean_median_v3.csv",
    "STEP_SHEET_NAME": "good_step_ends",
    "CYCLE_SHEET_NAME": "cycle_labels",
}


# --------------- Helpers ---------------

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def load_global_medians(step_stats_dir: str) -> Dict[str, float]:
    """
    Load GLOBAL_mean_median_v3.csv and return {metric: median}.
    """
    path = os.path.join(step_stats_dir, CONFIG["GLOBAL_STEP_STATS_FILE"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Global step stats file not found: {path}")
    g = pd.read_csv(path)
    if not {"metric", "median"}.issubset(g.columns):
        raise ValueError(f"{path} must contain 'metric' and 'median' columns")
    return dict(zip(g["metric"], g["median"]))


def match_step_file(labeled_file: str, step_stats_dir: str) -> Optional[str]:
    """
    Given something like AAA.xlsx_labeled.xlsx, find AAA_good_steps_v3.xlsx
    in step_stats_dir.
    """
    base = os.path.basename(labeled_file)
    stem = re.sub(r'_labeled$', '', os.path.splitext(base)[0])  # drop _labeled
    cand = os.path.join(step_stats_dir, f"{stem}_good_steps_v3.xlsx")
    if os.path.isfile(cand):
        return cand

    # Fuzzy fallback
    target_norm = _norm(stem)
    for f in glob.glob(os.path.join(step_stats_dir, "*_good_steps_v3.xlsx")):
        st = re.sub(r'_good_steps_v3$', '', os.path.splitext(os.path.basename(f))[0])
        if _norm(st) == target_norm:
            return f
    return None


def label_cycle_row_3class(row: pd.Series,
                           landing_V_ref: float,
                           cfg: Dict[str, Any]) -> int:
    """
    Given a row from the merged cycle+step dataframe, return 3-class label.
    0 = BAD, 1 = good-not-drone, 2 = drone-ready
    """

    # Hard fail flag from cycle_label_v4.py
    hard_fail = bool(row.get("hard_fail", False))

    if hard_fail:
        return 0  # BAD

    # ----- Drone fitness checks -----
    ir_proxy = float(row.get("IR_proxy", np.nan))
    dchg_spec = float(row.get("DChg_Spec_mAhg", np.nan))
    CE = float(row.get("CE", np.nan))

    landing_V = float(row.get("landing_V_end", np.nan))
    endV_ok = False

    if not np.isnan(landing_V) and not np.isnan(landing_V_ref):
        endV_ok = landing_V >= (landing_V_ref - cfg["ENDV_MARGIN"])

    is_low_IR = (not np.isnan(ir_proxy)) and (ir_proxy <= cfg["IR_PROXY_DRONE_MAX"])
    is_high_capacity = (not np.isnan(dchg_spec)) and (dchg_spec >= cfg["DCHG_SPEC_DRONE_MIN"])
    ce_ideal = (not np.isnan(CE)) and (cfg["CE_DRONE_LOW"] <= CE <= cfg["CE_DRONE_HIGH"])

    # Drone-ready = must pass all drone criteria
    if is_low_IR and is_high_capacity and ce_ideal and endV_ok:
        return 2  # GOOD for drones

    # Passed hard rules but not drone criteria
    return 1  # good but not drone-ready


def label_battery_3class(cycle_labels: np.ndarray, cfg: Dict[str, Any]) -> int:
    """
    Aggregate cycle-level 3-class labels into a battery-level label.
    """
    if len(cycle_labels) == 0:
        return 0  # degenerate: treat as BAD

    cycle_labels = np.asarray(cycle_labels, dtype=int)
    total = len(cycle_labels)

    bad_ratio = np.mean(cycle_labels == 0)
    drone_ratio = np.mean(cycle_labels == 2)

    if bad_ratio > cfg["BAD_RATIO_MAX"]:
        return 0  # BAD battery

    if drone_ratio >= cfg["DRONE_RATIO_MIN"]:
        return 2  # drone-ready

    return 1  # good but not drone-ready


# --------------- Main pipeline ---------------

def main():
    ap = argparse.ArgumentParser(
        description="Create 3-class (BAD / good-not-drone / drone-ready) labels "
                    "by combining cycle_label_v4 outputs with step end-voltage stats."
    )
    ap.add_argument("--labeled_dir", required=True,
                    help="Folder with *_labeled.xlsx files from cycle_label_v4.py")
    ap.add_argument("--step_stats_dir", required=True,
                    help="Folder with *_good_steps_v3.xlsx and GLOBAL_mean_median_v3.csv")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Output folder (default: labeled_dir)")
    args = ap.parse_args()

    labeled_dir = args.labeled_dir
    step_stats_dir = args.step_stats_dir
    outdir = args.outdir or labeled_dir
    os.makedirs(outdir, exist_ok=True)

    # Load global medians
    global_meds = load_global_medians(step_stats_dir)
    landing_V_ref = float(global_meds.get("landing_V_end", np.nan))
    if np.isnan(landing_V_ref):
        print("[WARN] landing_V_end median not found in GLOBAL_mean_median_v3.csv; "
              "drone endV check will be skipped for all cycles.")
    else:
        print(f"[INFO] Global landing_V_end median = {landing_V_ref:.4f} V")

    # Enumerate labeled files
    labeled_files = sorted(glob.glob(os.path.join(labeled_dir, "*_labeled.xlsx")))
    if not labeled_files:
        print(f"[ERROR] No *_labeled.xlsx files found in {labeled_dir}")
        return

    all_cycles_rows = []
    battery_summary_rows = []

    for lf in labeled_files:
        base = os.path.basename(lf)
        print(f"\n[PROCESS] {base}")

        try:
            cyc_df = pd.read_excel(lf, sheet_name=CONFIG["CYCLE_SHEET_NAME"])
        except Exception as e:
            print(f"[FAIL] Could not read cycle sheet '{CONFIG['CYCLE_SHEET_NAME']}' "
                  f"in {base}: {e}")
            continue

        if "Cycle" not in cyc_df.columns:
            print(f"[WARN] No 'Cycle' column in {base}; skipping.")
            continue

        # Match & merge step stats (landing_V_end etc.) for good cycles
        step_file = match_step_file(lf, step_stats_dir)
        if step_file:
            try:
                step_df = pd.read_excel(step_file, sheet_name=CONFIG["STEP_SHEET_NAME"])
                # step_df has columns: file, cycle, <step>_V_end, standby_t_end_h
                # Align 'cycle' -> 'Cycle'
                if "cycle" in step_df.columns:
                    step_df = step_df.rename(columns={"cycle": "Cycle"})
                # Merge on Cycle (left join: we keep all cycles even if no step stats)
                merge_cols = [c for c in step_df.columns if c != "file"]
                cyc_df = cyc_df.merge(step_df[merge_cols], on="Cycle", how="left")
                print(f"[INFO] Merged step stats from {os.path.basename(step_file)}")
            except Exception as e:
                print(f"[WARN] Could not merge step stats for {base}: {e}")
        else:
            print(f"[WARN] No *_good_steps_v3.xlsx match for {base}; "
                  f"step-based drone checks will be NaN.")

        # Compute some helper features
        if not np.isnan(landing_V_ref) and "landing_V_end" in cyc_df.columns:
            cyc_df["landing_V_rel_global"] = cyc_df["landing_V_end"] - landing_V_ref
        else:
            cyc_df["landing_V_rel_global"] = np.nan

        # 3-class cycle labels
        labels_3 = []
        for _, row in cyc_df.iterrows():
            lbl = label_cycle_row_3class(row, landing_V_ref, CONFIG)
            labels_3.append(lbl)
        cyc_df["cycle_label_3class"] = labels_3

        # Optional human-readable names
        name_map = {0: "BAD", 1: "GOOD_not_drone", 2: "GOOD_drone"}
        cyc_df["cycle_label_3name"] = cyc_df["cycle_label_3class"].map(name_map)

        # Battery-level label
        battery_label = label_battery_3class(cyc_df["cycle_label_3class"].values, CONFIG)
        cyc_df["battery_label_3class"] = battery_label
        cyc_df["battery_label_3name"] = name_map[battery_label]

        # Save per-battery Excel
        out_xlsx = os.path.join(outdir, base.replace("_labeled.xlsx", "_labeled_3class.xlsx"))
        cyc_df.to_excel(out_xlsx, sheet_name="cycle_labels_3class", index=False)
        print(f"[OK] Wrote {os.path.basename(out_xlsx)}")

        # Add to global tables
        cyc_df["file"] = base  # for traceability
        all_cycles_rows.append(cyc_df)

        # Summary row
        num_total = len(cyc_df)
        num_bad = int((cyc_df["cycle_label_3class"] == 0).sum())
        num_good_not_drone = int((cyc_df["cycle_label_3class"] == 1).sum())
        num_drone = int((cyc_df["cycle_label_3class"] == 2).sum())
        battery_summary_rows.append({
            "file": base,
            "n_cycles": num_total,
            "n_bad": num_bad,
            "n_good_not_drone": num_good_not_drone,
            "n_drone_ready": num_drone,
            "bad_ratio": num_bad / num_total if num_total else np.nan,
            "drone_ratio": num_drone / num_total if num_total else np.nan,
            "battery_label_3class": battery_label,
            "battery_label_3name": name_map[battery_label],
        })

    # Global outputs
    if all_cycles_rows:
        ALL = pd.concat(all_cycles_rows, ignore_index=True)
        all_csv = os.path.join(outdir, "ALL_cycles_3class.csv")
        ALL.to_csv(all_csv, index=False)
        print(f"\n[OK] Wrote ALL_cycles_3class.csv with {len(ALL)} rows")

    if battery_summary_rows:
        S = pd.DataFrame(battery_summary_rows)
        sum_csv = os.path.join(outdir, "battery_level_3class_summary.csv")
        S.to_csv(sum_csv, index=False)
        print(f"[OK] Wrote battery_level_3class_summary.csv with {len(S)} batteries")


if __name__ == "__main__":
    main()
