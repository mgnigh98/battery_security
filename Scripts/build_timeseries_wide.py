#!/usr/bin/env python3
"""
build_timeseries_wide.py

Build a fixed-grid cycle-wise multivariate time-series dataset from raw battery Excel files.

Output:
- one wide CSV with one row per cycle
- prefix-window CSVs for 1s, 2s, 5s, 10s, 20s, 30s, 50s, 60s

Uses:
- master cycle CSV from all_csv_for_training/ALL_cycles_master_from_raw.csv
- raw Excel files from data/

Current alignment:
- take_off + hover + cruise 

- one row per cycle
- no datapoint-level repeated rows
- same cycle cohort across all windows if keep_only_full_window is enabled
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
    module="openpyxl"
)

PHASE_NAMES = ["take_off", "hover", "cruise", "landing", "standby"]
ALIGN_PHASES = ["take_off", "hover", "cruise"]

DEFAULT_WINDOWS = [1, 2, 5, 10, 20, 30, 50, 60]
DEFAULT_FEATURES = [
    "voltage_v",
    "current_a",
    "power_w",
    "capacity_ah",
    "spec_cap",
    "chg_cap",
    "chg_spec",
    "dchg_cap",
    "dchg_spec",
    "energy_wh",
]


# ============================================================
# Helpers
# ============================================================

def norm_text(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    x = re.sub(r"[^a-z0-9_\.]+", "", x)
    x = re.sub(r"_+", "_", x)
    return x


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_file_stem(name: str) -> str:
    stem = Path(str(name)).stem
    stem = stem.replace("_labeled_3class", "")
    stem = stem.replace("_labeled", "")
    return stem


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lookup = {norm_text(c): c for c in df.columns}
    for cand in candidates:
        key = norm_text(cand)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Could not find any of columns: {candidates}")
    return None


def choose_sheet_name(xl: pd.ExcelFile, target: str) -> str:
    for s in xl.sheet_names:
        if norm_text(s) == norm_text(target):
            return s
    raise KeyError(f"Sheet '{target}' not found. Available: {xl.sheet_names}")


# ============================================================
# Parse master / step / record
# ============================================================

def load_master_csv(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)

    required = ["file", "Cycle", "Label", "cycle_label_3class", "cycle_label_3name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    df["file_key"] = df["file"].map(normalize_file_stem)
    df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce")
    df = df.dropna(subset=["Cycle"]).copy()
    df["Cycle"] = df["Cycle"].astype(int)

    return df


def parse_step_sheet(step_df: pd.DataFrame) -> pd.DataFrame:
    col_cycle = find_col(step_df, ["Cycle Index"])
    col_step_idx = find_col(step_df, ["Step Index"])
    col_step_type = find_col(step_df, ["Step Type"])
    col_onset_date = find_col(step_df, ["Oneset Date"], required=False)
    col_end_date = find_col(step_df, ["End Date"], required=False)
    col_step_time_h = find_col(step_df, ["Step Time(h)"], required=False)

    out = pd.DataFrame({
        "Cycle": safe_num(step_df[col_cycle]),
        "Step_Index": safe_num(step_df[col_step_idx]),
        "Step_Type": step_df[col_step_type].astype(str).str.strip(),
        "Oneset_Date": pd.to_datetime(step_df[col_onset_date], errors="coerce") if col_onset_date else pd.NaT,
        "End_Date": pd.to_datetime(step_df[col_end_date], errors="coerce") if col_end_date else pd.NaT,
        "Step_Time_h": safe_num(step_df[col_step_time_h]) if col_step_time_h else np.nan,
    })

    out = out.dropna(subset=["Cycle", "Step_Index"]).copy()
    out["Cycle"] = out["Cycle"].astype(int)
    out["Step_Index"] = out["Step_Index"].astype(int)
    out["Step_Type_norm"] = out["Step_Type"].map(norm_text)

    return out.sort_values(["Cycle", "Step_Index"]).reset_index(drop=True)


def parse_record_sheet(record_df: pd.DataFrame) -> pd.DataFrame:
    col_cycle = find_col(record_df, ["Cycle Index"])
    col_step_type = find_col(record_df, ["Step Type"])
    col_date = find_col(record_df, ["Date"], required=False)
    col_time_h = find_col(record_df, ["Time(h)"], required=False)
    col_total_time_h = find_col(record_df, ["Total Time(h)"], required=False)
    col_current = find_col(record_df, ["Current(A)"], required=False)
    col_voltage = find_col(record_df, ["Voltage(V)"], required=False)
    col_capacity = find_col(record_df, ["Capacity(Ah)"], required=False)
    col_spec_cap = find_col(record_df, ["Spec. Cap.(mAh/g)"], required=False)
    col_chg_cap = find_col(record_df, ["Chg. Cap.(Ah)"], required=False)
    col_chg_spec = find_col(record_df, ["Chg. Spec. Cap.(mAh/g)"], required=False)
    col_dchg_cap = find_col(record_df, ["DChg. Cap.(Ah)"], required=False)
    col_dchg_spec = find_col(record_df, ["DChg. Spec. Cap.(mAh/g)"], required=False)
    col_energy = find_col(record_df, ["Energy(Wh)"], required=False)
    col_power = find_col(record_df, ["Power(W)"], required=False)

    out = pd.DataFrame({
        "Cycle": safe_num(record_df[col_cycle]),
        "Step_Type": record_df[col_step_type].astype(str).str.strip(),
        "record_date": pd.to_datetime(record_df[col_date], errors="coerce") if col_date else pd.NaT,
        "time_h": safe_num(record_df[col_time_h]) if col_time_h else np.nan,
        "total_time_h": safe_num(record_df[col_total_time_h]) if col_total_time_h else np.nan,
        "current_a": safe_num(record_df[col_current]) if col_current else np.nan,
        "voltage_v": safe_num(record_df[col_voltage]) if col_voltage else np.nan,
        "capacity_ah": safe_num(record_df[col_capacity]) if col_capacity else np.nan,
        "spec_cap": safe_num(record_df[col_spec_cap]) if col_spec_cap else np.nan,
        "chg_cap": safe_num(record_df[col_chg_cap]) if col_chg_cap else np.nan,
        "chg_spec": safe_num(record_df[col_chg_spec]) if col_chg_spec else np.nan,
        "dchg_cap": safe_num(record_df[col_dchg_cap]) if col_dchg_cap else np.nan,
        "dchg_spec": safe_num(record_df[col_dchg_spec]) if col_dchg_spec else np.nan,
        "energy_wh": safe_num(record_df[col_energy]) if col_energy else np.nan,
        "power_w": safe_num(record_df[col_power]) if col_power else np.nan,
    })

    out = out.dropna(subset=["Cycle"]).copy()
    out["Cycle"] = out["Cycle"].astype(int)
    out["Step_Type_norm"] = out["Step_Type"].map(norm_text)

    if out["power_w"].isna().all() and {"voltage_v", "current_a"}.issubset(out.columns):
        out["power_w"] = out["voltage_v"] * out["current_a"]

    # keep raw time too for fallback/debugging
    if out["time_h"].notna().any():
        out["time_sec_raw"] = out["time_h"] * 3600.0
    elif out["total_time_h"].notna().any():
        out["time_sec_raw"] = out["total_time_h"] * 3600.0
    else:
        out["time_sec_raw"] = np.nan

    return out.reset_index(drop=True)

def is_charge_step(step_type_norm: str) -> bool:
    return ("chg" in step_type_norm) and ("dchg" not in step_type_norm)


def is_discharge_step(step_type_norm: str) -> bool:
    return "dchg" in step_type_norm


def get_cycle_phase_steps(step_df: pd.DataFrame, cycle_num: int) -> Optional[pd.DataFrame]:
    """
    Return exactly 5 ordered discharge steps for one cycle with assigned phase names.
    """
    cdf = step_df.loc[step_df["Cycle"] == cycle_num].sort_values("Step_Index").copy()
    if cdf.empty:
        return None

    charge_steps = cdf.loc[cdf["Step_Type_norm"].map(is_charge_step)].copy()
    dchg_steps = cdf.loc[cdf["Step_Type_norm"].map(is_discharge_step)].sort_values("Step_Index").copy()

    if len(charge_steps) != 1 or len(dchg_steps) != 5:
        return None

    dchg_steps = dchg_steps.copy()
    dchg_steps["phase_name"] = PHASE_NAMES
    return dchg_steps.reset_index(drop=True)

# ============================================================
# Time-axis construction
# ============================================================

def build_aligned_cycle_long(
    record_df: pd.DataFrame,
    step_df: pd.DataFrame,
    cycle_num: int,
    align_phases: List[str],
) -> pd.DataFrame:
    """
    Build aligned cycle time-series using step-sheet timestamps:
      - Oneset_Date / End_Date from step sheet
      - record_date from record sheet
    """
    phase_steps = get_cycle_phase_steps(step_df, cycle_num)
    if phase_steps is None or phase_steps.empty:
        return pd.DataFrame()

    cdf = record_df.loc[record_df["Cycle"] == cycle_num].copy()
    if cdf.empty:
        return pd.DataFrame()

    if "record_date" not in cdf.columns or cdf["record_date"].isna().all():
        return pd.DataFrame()

    parts = []
    offset_sec = 0.0

    for phase in align_phases:
        srow = phase_steps.loc[phase_steps["phase_name"] == phase]
        if srow.empty:
            return pd.DataFrame()

        srow = srow.iloc[0]
        start_dt = srow["Oneset_Date"]
        end_dt = srow["End_Date"]

        if pd.isna(start_dt) or pd.isna(end_dt):
            return pd.DataFrame()

        pdf = cdf.loc[
            (cdf["record_date"] >= start_dt) &
            (cdf["record_date"] <= end_dt)
        ].copy()

        if pdf.empty:
            return pd.DataFrame()

        pdf = pdf.sort_values("record_date").copy()
        pdf["phase_time_sec"] = (pdf["record_date"] - pdf["record_date"].iloc[0]).dt.total_seconds()
        pdf["aligned_time_sec"] = pdf["phase_time_sec"] + offset_sec
        pdf["phase_name"] = phase

        finite_t = pdf["phase_time_sec"].dropna().values
        if len(finite_t) == 0:
            return pd.DataFrame()

        offset_sec += float(np.max(finite_t))
        parts.append(pdf)

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values("aligned_time_sec").reset_index(drop=True)
    return out

# ============================================================
# Interpolation / flattening
# ============================================================

def interp_one_feature(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x_src) & np.isfinite(y_src)
    x_src = x_src[mask]
    y_src = y_src[mask]

    if len(x_src) == 0:
        return np.full(len(x_tgt), np.nan)

    order = np.argsort(x_src)
    x_src = x_src[order]
    y_src = y_src[order]

    uniq_x, uniq_idx = np.unique(x_src, return_index=True)
    uniq_y = y_src[uniq_idx]

    if len(uniq_x) == 1:
        out = np.full(len(x_tgt), np.nan)
        out[np.isclose(x_tgt, uniq_x[0])] = uniq_y[0]
        return out

    out = np.full(len(x_tgt), np.nan)
    valid = (x_tgt >= uniq_x.min()) & (x_tgt <= uniq_x.max())
    out[valid] = np.interp(x_tgt[valid], uniq_x, uniq_y)
    return out


def interpolate_cycle_to_grid(
    cycle_long_df: pd.DataFrame,
    grid_sec: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    out = pd.DataFrame({"t_sec": grid_sec})
    x_src = cycle_long_df["aligned_time_sec"].astype(float).values

    for feat in feature_names:
        if feat in cycle_long_df.columns:
            y_src = pd.to_numeric(cycle_long_df[feat], errors="coerce").values
            out[feat] = interp_one_feature(x_src, y_src, grid_sec)
        else:
            out[feat] = np.nan

    return out


def flatten_cycle_grid(
    meta_row: pd.Series,
    cycle_grid_df: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, object]:
    row = {
        "file": meta_row["file"],
        "Cycle": int(meta_row["Cycle"]),
        "Label": meta_row["Label"],
        "cycle_label_3class": meta_row["cycle_label_3class"],
        "cycle_label_3name": meta_row["cycle_label_3name"],
    }

    for _, r in cycle_grid_df.iterrows():
        t = int(r["t_sec"])
        for feat in feature_names:
            row[f"{feat}_t{t}"] = r.get(feat, np.nan)

    return row


# ============================================================
# Main builder
# ============================================================

def build_timeseries_dataset(
    data_dir: Path,
    master_csv: Path,
    out_csv: Path,
    issues_csv: Path,
    prefix_dir: Path,
    max_time_sec: int,
    dt_sec: int,
    keep_only_full_window: bool,
    windows: List[int],
    feature_names: List[str],
) -> None:
    master_df = load_master_csv(master_csv)

    grid_sec = np.arange(0, max_time_sec + dt_sec, dt_sec)
    all_rows = []
    issues = []

    excel_files = sorted(list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls")))
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {data_dir}")

    file_map = {normalize_file_stem(fp.name): fp for fp in excel_files}

    grouped = master_df.groupby("file_key", sort=True)

    for file_key, group_df in grouped:
        fp = file_map.get(file_key, None)
        if fp is None:
            for _, mrow in group_df.iterrows():
                issues.append({
                    "file": mrow["file"],
                    "Cycle": mrow["Cycle"],
                    "reason": "raw_excel_not_found",
                })
            continue

        try:
            xl = pd.ExcelFile(fp)
            step_sheet = choose_sheet_name(xl, "step")
            record_sheet = choose_sheet_name(xl, "record")

            step_raw = pd.read_excel(xl, sheet_name=step_sheet)
            record_raw = pd.read_excel(xl, sheet_name=record_sheet)

            step_df = parse_step_sheet(step_raw)
            record_df = parse_record_sheet(record_raw)

        except Exception as e:
            for _, mrow in group_df.iterrows():
                issues.append({
                    "file": mrow["file"],
                    "Cycle": mrow["Cycle"],
                    "reason": f"file_read_error:{type(e).__name__}:{e}",
                })
            continue

        for _, mrow in group_df.iterrows():
            cyc = int(mrow["Cycle"])
            try:
                phase_steps = get_cycle_phase_steps(step_df, cyc)
                if phase_steps is None:
                    issues.append({
                        "file": mrow["file"],
                        "Cycle": cyc,
                        "reason": "invalid_step_structure",
                    })
                    continue

                cycle_long = build_aligned_cycle_long(
                    record_df=record_df,
                    step_df=step_df,
                    cycle_num=cyc,
                    align_phases=ALIGN_PHASES,

                )
                if cycle_long.empty:
                    issues.append({
                        "file": mrow["file"],
                        "Cycle": cyc,
                        "reason": "empty_aligned_cycle",
                    })
                    continue

                cycle_grid = interpolate_cycle_to_grid(
                    cycle_long_df=cycle_long,
                    grid_sec=grid_sec,
                    feature_names=feature_names,
                )

                if keep_only_full_window:
                    required_cols = feature_names
                    valid_mask = cycle_grid[required_cols].notna().all(axis=1)
                    # need all grid points valid
                    if not valid_mask.all():
                        issues.append({
                            "file": mrow["file"],
                            "Cycle": cyc,
                            "reason": "missing_values_in_full_window",
                        })
                        continue

                flat_row = flatten_cycle_grid(
                    meta_row=mrow,
                    cycle_grid_df=cycle_grid,
                    feature_names=feature_names,
                )
                all_rows.append(flat_row)

            except Exception as e:
                issues.append({
                    "file": mrow["file"],
                    "Cycle": cyc,
                    "reason": f"cycle_error:{type(e).__name__}:{e}",
                })

    if not all_rows:
        pd.DataFrame(issues).to_csv(issues_csv, index=False)
        raise RuntimeError("No cycle time-series rows produced.")

    wide_df = pd.DataFrame(all_rows)
    meta_cols = ["file", "Cycle", "Label", "cycle_label_3class", "cycle_label_3name"]
    ts_cols = [c for c in wide_df.columns if c not in meta_cols]

    def sort_key(colname: str):
        m = re.match(r"(.+)_t(\d+)$", colname)
        if not m:
            return (999, 999, colname)
        feat, t = m.group(1), int(m.group(2))
        feat_order = feature_names.index(feat) if feat in feature_names else 999
        return (t, feat_order, feat)

    ts_cols = sorted(ts_cols, key=sort_key)
    wide_df = wide_df[meta_cols + ts_cols].sort_values(["file", "Cycle"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    prefix_dir.mkdir(parents=True, exist_ok=True)
    issues_csv.parent.mkdir(parents=True, exist_ok=True)

    wide_df.to_csv(out_csv, index=False)
    pd.DataFrame(issues).to_csv(issues_csv, index=False)

    print(f"Saved wide time-series CSV: {out_csv}")
    print(f"Saved issues CSV:           {issues_csv}")
    print(f"Rows kept:                  {len(wide_df)}")
    print(f"Unique files kept:          {wide_df['file'].nunique()}")

    print("\n3-class counts in kept time-series dataset:")
    print(wide_df["cycle_label_3name"].value_counts(dropna=False).to_string())

    # Prefix datasets
    for w in windows:
        keep_cols = meta_cols[:]
        for col in ts_cols:
            m = re.match(r"(.+)_t(\d+)$", col)
            if m and int(m.group(2)) <= w:
                keep_cols.append(col)

        prefix_df = wide_df[keep_cols].copy()
        prefix_path = prefix_dir / f"cycle_timeseries_prefix_{w}s.csv"
        prefix_df.to_csv(prefix_path, index=False)
        print(f"Saved prefix dataset:       {prefix_path} | rows={len(prefix_df)}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-grid cycle-wise multivariate time-series dataset."
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Folder with raw Excel files."
    )
    parser.add_argument(
        "--master_csv",
        type=Path,
        default=Path("all_csv_for_training") / "ALL_cycles_master_from_raw.csv",
        help="Master cycle CSV from build_cycle_master.py"
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("all_csv_for_training") / "cycle_timeseries_wide_t0_60.csv",
        help="Output wide time-series CSV."
    )
    parser.add_argument(
        "--issues_csv",
        type=Path,
        default=Path("all_csv_for_training") / "cycle_timeseries_wide_t0_60_issues.csv",
        help="Issues CSV."
    )
    parser.add_argument(
        "--prefix_dir",
        type=Path,
        default=Path("all_csv_for_training") / "prefix_windows",
        help="Directory to save prefix datasets."
    )
    parser.add_argument(
        "--max_time_sec",
        type=int,
        default=60,
        help="Maximum aligned time in seconds."
    )
    parser.add_argument(
        "--dt_sec",
        type=int,
        default=1,
        help="Grid spacing in seconds."
    )
    parser.add_argument(
        "--keep_only_full_window",
        action="store_true",
        help="Keep only cycles that have non-NaN values over the full time grid."
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=DEFAULT_WINDOWS,
        help="Prefix window lengths in seconds."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_timeseries_dataset(
        data_dir=args.data_dir,
        master_csv=args.master_csv,
        out_csv=args.out_csv,
        issues_csv=args.issues_csv,
        prefix_dir=args.prefix_dir,
        max_time_sec=args.max_time_sec,
        dt_sec=args.dt_sec,
        keep_only_full_window=args.keep_only_full_window,
        windows=args.windows,
        feature_names=DEFAULT_FEATURES,
    )


if __name__ == "__main__":
    main()
