#!/usr/bin/env python3
"""
build_early_features_from_prefix.py

Build physics-informed early-window features from prefix time-series CSVs.

Input:
    all_csv_for_training/prefix_windows/cycle_timeseries_prefix_*s.csv

Output:
    all_csv_for_training/early_feature_windows/early_features_*s.csv

Each output row is one cycle, with:
    - metadata columns
    - aggregated early-window features inspired by prior successful pipeline
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


META_COLS = ["file", "Cycle", "Label", "cycle_label_3class", "cycle_label_3name"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def infer_window_from_name(path: Path) -> int:
    stem = path.stem  # cycle_timeseries_prefix_10s
    token = stem.split("_")[-1]
    return int(token.replace("s", ""))


def get_timepoints_from_columns(df: pd.DataFrame) -> List[int]:
    tvals = []
    for c in df.columns:
        m = re.match(r".+_t(\d+)$", c)
        if m:
            tvals.append(int(m.group(1)))
    return sorted(set(tvals))


def extract_series_matrix(df: pd.DataFrame, base_name: str, timepoints: List[int]) -> Optional[np.ndarray]:
    cols = [f"{base_name}_t{t}" for t in timepoints]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy(dtype=float)


def safe_mean(arr: np.ndarray) -> np.ndarray:
    return np.nanmean(arr, axis=1)


def safe_std(arr: np.ndarray) -> np.ndarray:
    return np.nanstd(arr, axis=1)


def safe_min(arr: np.ndarray) -> np.ndarray:
    return np.nanmin(arr, axis=1)


def safe_max(arr: np.ndarray) -> np.ndarray:
    return np.nanmax(arr, axis=1)


# ------------------------------------------------------------
# Feature computation
# ------------------------------------------------------------

def compute_early_features(
    df: pd.DataFrame,
    collapse_sag_ratio_thresh: float,
    ir_high_thresh: float,
    dvdt_fluct_thresh: float,
) -> pd.DataFrame:
    """
    Compute early-window aggregated features from prefix raw time-series columns.
    """
    timepoints = get_timepoints_from_columns(df)
    if len(timepoints) < 2:
        raise ValueError("Need at least 2 timepoints in prefix dataset.")

    dt = np.diff(timepoints)
    mean_dt = float(np.mean(dt)) if len(dt) > 0 else 1.0
    eps = 1e-8

    feat: Dict[str, np.ndarray] = {}

    # -------- core matrices --------
    vmat = extract_series_matrix(df, "voltage_v", timepoints)
    imat = extract_series_matrix(df, "current_a", timepoints)
    pmat = extract_series_matrix(df, "power_w", timepoints)
    emat = extract_series_matrix(df, "energy_wh", timepoints)
    dchg_cap_mat = extract_series_matrix(df, "dchg_cap", timepoints)
    dchg_spec_mat = extract_series_matrix(df, "dchg_spec", timepoints)

    if vmat is None:
        raise ValueError("voltage_v_t* columns are required.")
    if imat is None:
        raise ValueError("current_a_t* columns are required.")
    if pmat is None:
        raise ValueError("power_w_t* columns are required.")

    # -------- voltage features --------
    v0 = vmat[:, 0]
    vend = vmat[:, -1]
    vmin = safe_min(vmat)

    feat["V_mean"] = safe_mean(vmat)
    feat["V_min"] = vmin
    feat["V_max"] = safe_max(vmat)
    feat["V_std"] = safe_std(vmat)

    feat["V_end"] = vend
    feat["V_drop_end"] = v0 - vend
    feat["V_sag"] = v0 - vmin
    feat["V_sag_ratio"] = (v0 - vmin) / (np.abs(v0) + eps)

    # -------- dv/dt --------
    dv = np.diff(vmat, axis=1) / mean_dt
    feat["dvdt_mean"] = safe_mean(dv)
    feat["dvdt_std"] = safe_std(dv)
    feat["dvdt_min"] = safe_min(dv)
    feat["dvdt_max"] = safe_max(dv)

    # second derivative / curvature
    if dv.shape[1] >= 2:
        d2v = np.diff(dv, axis=1) / mean_dt
        feat["d2vdt2_mean"] = safe_mean(d2v)
        feat["d2vdt2_std"] = safe_std(d2v)
        feat["d2vdt2_min"] = safe_min(d2v)
        feat["d2vdt2_max"] = safe_max(d2v)

    # local voltage collapse / instability
    for w in [3, 5]:
        if vmat.shape[1] >= w:
            local_drop = []
            local_range = []
            for start in range(0, vmat.shape[1] - w + 1):
                block = vmat[:, start:start + w]
                local_drop.append(block[:, 0] - np.nanmin(block, axis=1))
                local_range.append(np.nanmax(block, axis=1) - np.nanmin(block, axis=1))
            local_drop = np.stack(local_drop, axis=1)
            local_range = np.stack(local_range, axis=1)
            feat[f"V_roll{w}_drop_max"] = np.nanmax(local_drop, axis=1)
            feat[f"V_roll{w}_range_max"] = np.nanmax(local_range, axis=1)

    # -------- current features --------
    feat["I_mean"] = safe_mean(imat)
    feat["I_std"] = safe_std(imat)
    feat["I_min"] = safe_min(imat)
    feat["I_max"] = safe_max(imat)
    feat["I_abs_mean"] = np.nanmean(np.abs(imat), axis=1)
    feat["I_abs_std"] = np.nanstd(np.abs(imat), axis=1)

    di = np.diff(imat, axis=1) / mean_dt
    feat["didt_mean"] = safe_mean(di)
    feat["didt_std"] = safe_std(di)
    feat["didt_min"] = safe_min(di)
    feat["didt_max"] = safe_max(di)

    # -------- power features --------
    feat["power_mean"] = safe_mean(pmat)
    feat["power_std"] = safe_std(pmat)
    feat["power_min"] = safe_min(pmat)
    feat["power_max"] = safe_max(pmat)
    feat["power_abs_mean"] = np.nanmean(np.abs(pmat), axis=1)
    feat["power_abs_std"] = np.nanstd(np.abs(pmat), axis=1)

    dp = np.diff(pmat, axis=1) / mean_dt
    feat["dpdt_mean"] = safe_mean(dp)
    feat["dpdt_std"] = safe_std(dp)
    feat["dpdt_min"] = safe_min(dp)
    feat["dpdt_max"] = safe_max(dp)

    for w in [3, 5]:
        if pmat.shape[1] >= w:
            local_range = []
            for start in range(0, pmat.shape[1] - w + 1):
                block = pmat[:, start:start + w]
                local_range.append(np.nanmax(block, axis=1) - np.nanmin(block, axis=1))
            local_range = np.stack(local_range, axis=1)
            feat[f"power_roll{w}_range_max"] = np.nanmax(local_range, axis=1)

    # -------- energy / cumulative discharge features --------
    # energy_ws from power integration over the window
    # if power is watts and dt is seconds, result is watt-seconds
    feat["energy_ws"] = np.nansum(pmat[:, :-1] * mean_dt, axis=1) if pmat.shape[1] >= 2 else pmat[:, 0]
    feat["energy_per_I"] = feat["energy_ws"] / (feat["I_abs_mean"] + eps)

    if emat is not None:
        feat["energy_end"] = emat[:, -1]
        feat["energy_mean"] = safe_mean(emat)
        feat["energy_std"] = safe_std(emat)
        feat["energy_slope_global"] = (emat[:, -1] - emat[:, 0]) / max(timepoints[-1] - timepoints[0], 1)

    if dchg_cap_mat is not None:
        feat["dchg_cap_end"] = dchg_cap_mat[:, -1]
        feat["dchg_cap_mean"] = safe_mean(dchg_cap_mat)
        feat["dchg_cap_std"] = safe_std(dchg_cap_mat)
        feat["dchg_cap_slope_global"] = (dchg_cap_mat[:, -1] - dchg_cap_mat[:, 0]) / max(timepoints[-1] - timepoints[0], 1)

    if dchg_spec_mat is not None:
        feat["dchg_spec_end"] = dchg_spec_mat[:, -1]
        feat["dchg_spec_mean"] = safe_mean(dchg_spec_mat)
        feat["dchg_spec_std"] = safe_std(dchg_spec_mat)
        feat["dchg_spec_slope_global"] = (dchg_spec_mat[:, -1] - dchg_spec_mat[:, 0]) / max(timepoints[-1] - timepoints[0], 1)

    # -------- early IR proxy --------
    # simple early proxy from initial voltage drop over mean absolute current
    feat["IR_early"] = (v0 - vend) / (feat["I_abs_mean"] + eps)

    # early voltage drops
    feat["V_drop_1s"] = vmat[:, 0] - vmat[:, min(1, vmat.shape[1] - 1)]
    feat["V_drop_2s"] = vmat[:, 0] - vmat[:, min(2, vmat.shape[1] - 1)]

    # global slope
    window_time = max(timepoints[-1] - timepoints[0], 1)
    feat["V_slope_global"] = (vmat[:, -1] - vmat[:, 0]) / window_time

    # normalized sag (IR indicator)
    feat["V_sag_norm"] = (vmat[:, 0] - np.nanmin(vmat, axis=1)) / (feat["I_abs_mean"] + 1e-8)

    # voltage recovery
    feat["voltage_recovery"] = vmat[:, -1] - np.nanmin(vmat, axis=1)

    # power stability
    feat["power_cv"] = feat["power_std"] / (np.abs(feat["power_mean"]) + 1e-8)

    # time index of minimum voltage
    feat["V_min_idx"] = np.nanargmin(vmat, axis=1).astype(float)

    # -------- binary flags --------
    feat["voltage_collapse_flag"] = (feat["V_sag_ratio"] > collapse_sag_ratio_thresh).astype(int)
    feat["IR_high_flag"] = (feat["IR_early"] > ir_high_thresh).astype(int)
    feat["dvdt_fluct_flag"] = (feat["dvdt_std"] > dvdt_fluct_thresh).astype(int)

    # -------- trajectory-shape features (8 strong features) --------
    window_time = max(timepoints[-1] - timepoints[0], 1)

    # 1) early voltage drop after 1 sample
    idx1 = min(1, vmat.shape[1] - 1)
    feat["traj_V_drop_1"] = vmat[:, 0] - vmat[:, idx1]

    # 2) early voltage drop after 2 samples
    idx2 = min(2, vmat.shape[1] - 1)
    feat["traj_V_drop_2"] = vmat[:, 0] - vmat[:, idx2]

    # 3) global voltage slope
    feat["traj_V_slope_global"] = (vmat[:, -1] - vmat[:, 0]) / window_time

    # 4) normalized voltage sag (very strong IR-like indicator)
    feat["traj_V_sag_norm"] = (vmat[:, 0] - np.nanmin(vmat, axis=1)) / (feat["I_abs_mean"] + eps)

    # 5) voltage recovery from minimum to end
    feat["traj_V_recovery"] = vmat[:, -1] - np.nanmin(vmat, axis=1)

    # 6) power coefficient of variation
    feat["traj_power_cv"] = feat["power_std"] / (np.abs(feat["power_mean"]) + eps)

    # 7) signed area under relative voltage-drop curve
    vrel = vmat - vmat[:, [0]]
    feat["traj_V_rel_area"] = np.nansum(vrel[:, :-1] * mean_dt, axis=1)

    # 8) maximum local 3-point voltage drop
    if vmat.shape[1] >= 3:
        local_drop3 = []
        for start in range(0, vmat.shape[1] - 3 + 1):
            block = vmat[:, start:start + 3]
            local_drop3.append(block[:, 0] - np.nanmin(block, axis=1))
        local_drop3 = np.stack(local_drop3, axis=1)
        feat["traj_V_roll3_drop_max"] = np.nanmax(local_drop3, axis=1)
    else:
        feat["traj_V_roll3_drop_max"] = np.zeros(vmat.shape[0])


    return pd.DataFrame(feat, index=df.index)


# ------------------------------------------------------------
# Main builder
# ------------------------------------------------------------

def process_one_prefix_file(
    in_csv: Path,
    out_csv: Path,
    collapse_sag_ratio_thresh: float,
    ir_high_thresh: float,
    dvdt_fluct_thresh: float,
) -> None:
    df = pd.read_csv(in_csv)

    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise ValueError(f"{in_csv} missing metadata columns: {missing_meta}")

    meta_df = df[META_COLS].copy()
    feat_df = compute_early_features(
        df=df,
        collapse_sag_ratio_thresh=collapse_sag_ratio_thresh,
        ir_high_thresh=ir_high_thresh,
        dvdt_fluct_thresh=dvdt_fluct_thresh,
    )

    out_df = pd.concat([meta_df, feat_df], axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv} | rows={len(out_df)} | features={feat_df.shape[1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build early aggregated battery features from prefix time-series datasets."
    )
    parser.add_argument(
        "--prefix_dir",
        type=Path,
        default=Path("all_csv_for_training") / "prefix_windows",
        help="Directory containing cycle_timeseries_prefix_*s.csv files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("all_csv_for_training") / "early_feature_windows",
        help="Directory to save early_features_*s.csv files.",
    )
    parser.add_argument(
        "--collapse_sag_ratio_thresh",
        type=float,
        default=0.12,
        help="Threshold for voltage_collapse_flag based on V_sag_ratio.",
    )
    parser.add_argument(
        "--ir_high_thresh",
        type=float,
        default=0.35,
        help="Threshold for IR_high_flag based on IR_early.",
    )
    parser.add_argument(
        "--dvdt_fluct_thresh",
        type=float,
        default=0.02,
        help="Threshold for dvdt_fluct_flag based on dvdt_std.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prefix_paths = sorted(args.prefix_dir.glob("cycle_timeseries_prefix_*s.csv"))
    if not prefix_paths:
        raise FileNotFoundError(f"No prefix CSVs found in {args.prefix_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for path in prefix_paths:
        w = infer_window_from_name(path)
        out_csv = args.out_dir / f"early_features_{w}s.csv"

        process_one_prefix_file(
            in_csv=path,
            out_csv=out_csv,
            collapse_sag_ratio_thresh=args.collapse_sag_ratio_thresh,
            ir_high_thresh=args.ir_high_thresh,
            dvdt_fluct_thresh=args.dvdt_fluct_thresh,
        )

        tmp = pd.read_csv(out_csv, nrows=5)
        summary_rows.append({
            "window_sec": w,
            "output_csv": str(out_csv),
            "feature_count": len([c for c in tmp.columns if c not in META_COLS]),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("window_sec")
    summary_path = args.out_dir / "early_feature_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
