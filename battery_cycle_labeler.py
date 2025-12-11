
"""
battery_cycle_labeler.py

Label each cycle as GOOD/BAD for a single battery Excel file and export augmented data.
- Skips formation cycles (1..3) and the *last* cycle (may be incomplete).
- Adds per-cycle metrics (charge/discharge features, CE, IR proxy, etc.).
- Writes two new sheets: 'cycle_labels' (per cycle) and 'labeled_record' (record rows + label).
- Leaves original sheets intact by writing to a new output file.
Author: ChatGPT
"""

from __future__ import annotations
import argparse
import glob
import json
import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# ----------------------------
# Configurable thresholds
# ----------------------------
DEFAULT_CFG = {
    # Absolute voltage targets
    "V_END_CHG_MIN": 4.25,   # under-charge threshold
    "V_MAX_CHG_MAX": 4.35,   # overshoot threshold during charge
    "V_MIN_DCHG_MIN": 2.70,  # undervoltage floor during discharge
    "V_END_DCHG_MAX": 2.90,  # if discharge ended early (above cutoff), we flag/penalize

    # Time thresholds (hours)
    "DISCHARGE_TIME_MIN": 1.20,  # expect ~1.5h profile; below this and not at cutoff -> BAD
    "TARGET_DISCHARGE_TIME": 1.50,

    # Efficiency
    "CE_ABS_LOW": 0.93,
    "CE_ABS_HIGH": 1.05,

    # Charge linearity
    "CHARGE_LINEARITY_R2_MIN": 0.97,

    # IR proxy absolute cap (Ohms)
    "IR_ABS_MAX": 0.30,

    # Data quality
    "MIN_POINTS_CHG": 20,
    "MIN_POINTS_DCHG": 20,

    # Scoring (soft penalties) -- used only if all hard rules pass
    "SCORE_START": 100,
    "SCORE_GOOD_THRESHOLD": 70,
    "PENALTIES": {
        "AH_CHG_OUTLIER": 15,
        "AH_DCHG_OUTLIER": 20,
        "WH_DCHG_OUTLIER": 15,
        "DISCHARGE_TIME_DEVIATION": 10,
        "CE_MARGINAL": 10,
        "IR_MARGINAL": 10,
        "CHARGE_SLOPE_VARIABILITY": 10,
        "PROGRAM_NONCOMPLIANCE": 10
    },

    # Robust outlier multiplier (for per-file stats; not cross-battery)
    "MAD_K": 3.0,
}

# ----------------------------
# Utility functions
# ----------------------------

def r2_time_voltage_linear_fit(df: pd.DataFrame) -> float:
    """Compute R^2 for Voltage ~ Time (h) linear fit for given rows."""
    if df.empty or df["Time(h)"].nunique() < 3:
        return np.nan
    x = df["Time(h)"].to_numpy()
    y = df["Voltage(V)"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

def ir_proxy(charge_df: pd.DataFrame, dchg_df: pd.DataFrame, window_s: float = 10.0) -> float:
    """
    Estimate IR ~ ΔV / ΔI using the last seconds of charge and first seconds of discharge.
    Assumes 'Time(h)' is relative within step. We approximate by taking the last N points of charge
    and first N points of discharge (based on time deltas) if available.
    """
    if charge_df.empty or dchg_df.empty:
        return np.nan

    # Convert seconds to hours
    w = window_s / 3600.0

    # Charge window: last w hours of charge step
    t_max_chg = charge_df["Time(h)"].max()
    chg_win = charge_df[charge_df["Time(h)"] >= (t_max_chg - w)]
    # Discharge window: first w hours of discharge step (relative start assumed ~0)
    dchg_win = dchg_df[dchg_df["Time(h)"] <= (dchg_df["Time(h)"].min() + w)]

    if chg_win.empty or dchg_win.empty:
        return np.nan

    V1 = chg_win["Voltage(V)"].mean()
    I1 = chg_win["Current(A)"].mean()
    V2 = dchg_win["Voltage(V)"].mean()
    I2 = dchg_win["Current(A)"].mean()

    dV = V1 - V2  # drop
    dI = I2 - I1  # current step change
    if np.isclose(dI, 0, atol=1e-6):
        return np.nan
    return float(dV / dI)

def robust_bounds(x: np.ndarray, k: float) -> Tuple[float, float]:
    m = np.nanmedian(x)
    s = mad(x)
    return (m - k * s, m + k * s)

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# ----------------------------
# Core labeling
# ----------------------------

def label_file(
    input_xlsx: str,
    output_xlsx: Optional[str] = None,
    cfg: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single Excel workbook and return (cycle_labels_df, labeled_record_df).
    Also writes output_xlsx if provided.
    """
    cfg = (cfg or DEFAULT_CFG).copy()

    # Load sheets we care about
    xls = pd.ExcelFile(input_xlsx)

    def read_sheet(name: str) -> pd.DataFrame:
        return pd.read_excel(xls, sheet_name=name)

    # Skip non-essential sheets as requested
    sheet_names = set(xls.sheet_names)
    required = {"test", "cycle", "record", "step"}
    missing_req = [s for s in required if s not in sheet_names]
    if missing_req:
        raise ValueError(f"Missing required sheet(s): {missing_req}")

    test_df  = read_sheet("test")
    cycle_df = read_sheet("cycle")
    record_df = read_sheet("record")
    step_df  = read_sheet("step")

    # Basic schema checks & normalization
    # 'record' sheet columns we will rely on
    rec_cols = [
        "Cycle Index", "Step Type", "Time(h)", "Total Time(h)",
        "Current(A)", "Voltage(V)", "Capacity(Ah)",
        "Chg. Cap.(Ah)", "DChg. Cap.(Ah)", "Energy(Wh)"
    ]
    ensure_columns(record_df, rec_cols)

    # 'cycle' sheet helpful columns
    cyc_cols = [
        "Cycle Index", "Chg. Cap.(Ah)", "DChg. Cap.(Ah)",
        "Chg. Energy(Wh)", "DChg. Energy(Wh)",
        "Chg. Time(h)", "DChg. Time(h)"
    ]
    # Some files may miss energy/time; handle softly
    for c in cyc_cols:
        if c not in cycle_df.columns:
            cycle_df[c] = np.nan

    # Determine active material mass (g) from 'test' sheet
    # We search for a cell containing 'Active material' and take the numeric in the next column of the same row
    mass_g = np.nan
    try:
        mask = test_df.astype(str).apply(lambda col: col.str.contains("Active material", case=False, na=False)).any(axis=1)
        if mask.any():
            row = test_df[mask].iloc[0]
            # find first numeric in that row besides the match
            row_vals = row.tolist()
            # heuristic: mass value might be in the next column
            idx = row.index[(row.astype(str).str.contains("Active material", case=False, na=False)).to_numpy()].tolist()[0]
            pos = row.index.get_loc(idx)
            if pos + 1 < len(row_vals):
                cand = pd.to_numeric([row_vals[pos+1]], errors="coerce")[0]
                if pd.notna(cand):
                    mass_g = float(cand) / 1000.0 if cand > 5 else float(cand)  # if mg, convert to g
        # Fallback: try to find any numeric with 'mg' in same row string
        if np.isnan(mass_g):
            # Try scanning whole sheet for a numeric-looking field near 'mg'
            pass
    except Exception:
        pass

    # Prepare base per-cycle aggregation from record_df
    all_cycles = sorted([c for c in record_df["Cycle Index"].dropna().unique() if isinstance(c, (int, float))])
    if len(all_cycles) == 0:
        raise ValueError("No cycles found in 'record' sheet.")

    # Skip formation 1..3 and the last (possibly incomplete)
    max_cycle = int(max(all_cycles))
    valid_cycles = [int(c) for c in all_cycles if 4 <= int(c) < max_cycle]

    # Precompute robust stats for soft scoring (per-file, not per-battery fleet)
    # We'll compute these after first pass, so collect features
    per_cycle_rows: List[Dict[str, Any]] = []

    # Helper to pick charge/discharge data for a cycle
    def slice_cycle(df: pd.DataFrame, cycle_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cdf = df[df["Cycle Index"] == cycle_idx].copy()
        chg = cdf[cdf["Step Type"].astype(str).str.contains("Chg", case=False, na=False) &
                  cdf["Step Type"].astype(str).str.startswith("CC Chg")]
        dch = cdf[cdf["Step Type"].astype(str).str.contains("DChg", case=False, na=False)]
        return chg, dch

    # First pass: compute metrics and hard rules
    for cidx in valid_cycles:
        chg_df, dchg_df = slice_cycle(record_df, cidx)

        # Basic point counts
        n_chg = len(chg_df)
        n_dch = len(dchg_df)

        # Charge metrics
        V_end_chg = float(chg_df["Voltage(V)"].iloc[-1]) if n_chg else np.nan
        V_max_chg = float(chg_df["Voltage(V)"].max()) if n_chg else np.nan
        t_charge_h = float((chg_df["Time(h)"].max() - chg_df["Time(h)"].min())) if n_chg else np.nan
        Ah_chg = float(cycle_df.loc[cycle_df["Cycle Index"]==cidx, "Chg. Cap.(Ah)"].squeeze()) if cidx in set(cycle_df["Cycle Index"]) else np.nan
        if pd.isna(Ah_chg) and n_chg:
            # fallback from record cumulative
            Ah_chg = float(chg_df["Chg. Cap.(Ah)"].max())

        # Discharge metrics
        V_min_dchg = float(dchg_df["Voltage(V)"].min()) if n_dch else np.nan
        V_end_dchg = float(dchg_df["Voltage(V)"].iloc[-1]) if n_dch else np.nan
        t_discharge_h = float((dchg_df["Time(h)"].max() - dchg_df["Time(h)"].min())) if n_dch else 0.0
        Ah_dchg = float(cycle_df.loc[cycle_df["Cycle Index"]==cidx, "DChg. Cap.(Ah)"].squeeze()) if cidx in set(cycle_df["Cycle Index"]) else np.nan
        if pd.isna(Ah_dchg) and n_dch:
            Ah_dchg = float(dchg_df["DChg. Cap.(Ah)"].max())

        Wh_chg = float(cycle_df.loc[cycle_df["Cycle Index"]==cidx, "Chg. Energy(Wh)"].squeeze()) if cidx in set(cycle_df["Cycle Index"]) else np.nan
        Wh_dchg = float(cycle_df.loc[cycle_df["Cycle Index"]==cidx, "DChg. Energy(Wh)"].squeeze()) if cidx in set(cycle_df["Cycle Index"]) else np.nan
        if pd.isna(Wh_dchg) and n_dch:
            Wh_dchg = float(dchg_df["Energy(Wh)"].max())
        if pd.isna(Wh_chg) and n_chg:
            Wh_chg = float(chg_df["Energy(Wh)"].max())

        # Efficiency
        CE = (Ah_dchg / Ah_chg) if (Ah_chg and not pd.isna(Ah_chg) and Ah_chg != 0) else np.nan
        EE = (Wh_dchg / Wh_chg) if (Wh_chg and not pd.isna(Wh_chg) and Wh_chg != 0) else np.nan

        # Linearity of charge
        charge_r2 = r2_time_voltage_linear_fit(chg_df)

        # Charge slope variability (MAD of dV/dt)
        slope_mad = np.nan
        if n_chg >= 3:
            t = chg_df["Time(h)"].to_numpy()
            v = chg_df["Voltage(V)"].to_numpy()
            dvdt = np.diff(v) / np.diff(t)
            slope_mad = mad(dvdt)

        # IR proxy
        IR_est = ir_proxy(chg_df, dchg_df)

        # HARD RULES
        hard_flags = {
            "MISSING_DATA": (n_chg < cfg["MIN_POINTS_CHG"]) or (n_dch < cfg["MIN_POINTS_DCHG"]),
            "UNDERCHARGE": (not np.isnan(V_end_chg)) and (V_end_chg < cfg["V_END_CHG_MIN"]),
            "OVERSHOOT": (not np.isnan(V_max_chg)) and (V_max_chg > cfg["V_MAX_CHG_MAX"]),
            "EARLY_STOP_NO_CUTOFF": (t_discharge_h < cfg["DISCHARGE_TIME_MIN"]) and (not np.isnan(V_end_dchg)) and (V_end_dchg > cfg["V_END_DCHG_MAX"]),
            "UNDERVOLTAGE": (not np.isnan(V_min_dchg)) and (V_min_dchg < cfg["V_MIN_DCHG_MIN"]),
            "LOW_LINEARITY": (not np.isnan(charge_r2)) and (charge_r2 < cfg["CHARGE_LINEARITY_R2_MIN"]),
            "CE_ABS": (not np.isnan(CE)) and ((CE < cfg["CE_ABS_LOW"]) or (CE > cfg["CE_ABS_HIGH"])),
            "IR_ABS": (not np.isnan(IR_est)) and (IR_est > cfg["IR_ABS_MAX"])
        }
        hard_fail = any(hard_flags.values())

        row = dict(
            Cycle=int(cidx),
            V_end_chg=V_end_chg,
            V_max_chg=V_max_chg,
            t_charge_h=t_charge_h,
            Ah_chg=Ah_chg,
            V_min_dchg=V_min_dchg,
            V_end_dchg=V_end_dchg,
            t_discharge_h=t_discharge_h,
            Ah_dchg=Ah_dchg,
            Wh_chg=Wh_chg,
            Wh_dchg=Wh_dchg,
            CE=CE,
            EE=EE,
            charge_r2=charge_r2,
            slope_mad=slope_mad,
            IR_est=IR_est,
            hard_fail=hard_fail,
            **{f"HF_{k}": bool(v) for k, v in hard_flags.items()},
        )
        per_cycle_rows.append(row)

    cycle_metrics = pd.DataFrame(per_cycle_rows).sort_values("Cycle")

    # Soft scoring: compute robust bounds from the same file to define outliers
    # Note: We do not bootstrap on early cycles; we use file-level robust stats.
    k = cfg["MAD_K"]
    penalties = cfg["PENALTIES"]

    def outlier_mask(series: pd.Series) -> pd.Series:
        x = series.to_numpy(dtype=float)
        lo, hi = robust_bounds(x, k)
        return (series < lo) | (series > hi)

    cycle_metrics["soft_score"] = cfg["SCORE_START"]
    # Apply soft penalties only where hard_fail == False
    mask_ok = ~cycle_metrics["hard_fail"].astype(bool)

    # Define outliers
    if len(cycle_metrics) > 5:
        for col, pen_key in [
            ("Ah_chg", "AH_CHG_OUTLIER"),
            ("Ah_dchg", "AH_DCHG_OUTLIER"),
            ("Wh_dchg", "WH_DCHG_OUTLIER"),
        ]:
            m = outlier_mask(cycle_metrics[col])
            cycle_metrics.loc[mask_ok & m, "soft_score"] -= penalties[pen_key]

        # discharge time deviation from target (±10%)
        t = cycle_metrics["t_discharge_h"]
        dev = (np.abs(t - cfg["TARGET_DISCHARGE_TIME"]) > 0.10 * cfg["TARGET_DISCHARGE_TIME"])
        # penalize only if not at voltage cutoff (approx by V_end_dchg <= cutoff+0.05) to avoid punishing healthy cutoff endings
        cutoff_hit = cycle_metrics["V_end_dchg"] <= (cfg["V_END_DCHG_MAX"] + 0.02)
        cycle_metrics.loc[mask_ok & dev & (~cutoff_hit), "soft_score"] -= penalties["DISCHARGE_TIME_DEVIATION"]

        # CE marginal window (within absolute but outside tighter band)
        CE = cycle_metrics["CE"]
        CE_marg = CE.between(0.96, 1.02, inclusive="neither") == False  # outside tighter band
        CE_abs_ok = CE.between(cfg["CE_ABS_LOW"], cfg["CE_ABS_HIGH"], inclusive="both")
        cycle_metrics.loc[mask_ok & CE_abs_ok & CE_marg, "soft_score"] -= penalties["CE_MARGINAL"]

        # IR marginal (beyond robust bounds but below absolute max)
        IR = cycle_metrics["IR_est"]
        IR_out = outlier_mask(IR)
        cycle_metrics.loc[mask_ok & IR_out & (IR < cfg["IR_ABS_MAX"]), "soft_score"] -= penalties["IR_MARGINAL"]

        # Charge slope variability penalty vs robust bounds
        SM = cycle_metrics["slope_mad"]
        SM_out = outlier_mask(SM)
        cycle_metrics.loc[mask_ok & SM_out, "soft_score"] -= penalties["CHARGE_SLOPE_VARIABILITY"]

    # Final label
    cycle_metrics["Label"] = np.where(
        cycle_metrics["hard_fail"], "BAD",
        np.where(cycle_metrics["soft_score"] >= cfg["SCORE_GOOD_THRESHOLD"], "GOOD", "BAD")
    )

    # Reasons (concise): list which hard flags fired; if none, note soft penalties if score < 100
    def reasons_for_row(r: pd.Series) -> List[str]:
        reasons = []
        for k in [c for c in cycle_metrics.columns if c.startswith("HF_")]:
            if bool(r[k]):
                reasons.append(k.replace("HF_", ""))
        if not reasons and r["soft_score"] < cfg["SCORE_START"]:
            reasons.append(f"soft_penalty:{int(cfg['SCORE_START']-r['soft_score'])}")
        return reasons

    cycle_metrics["Reasons"] = cycle_metrics.apply(reasons_for_row, axis=1).apply(lambda lst: ", ".join(lst))

    # Augment 'record' sheet rows with cycle label
    label_map = dict(zip(cycle_metrics["Cycle"], cycle_metrics["Label"]))
    reason_map = dict(zip(cycle_metrics["Cycle"], cycle_metrics["Reasons"]))
    record_aug = record_df.copy()
    record_aug["Cycle Label"] = record_aug["Cycle Index"].map(label_map)
    record_aug["Cycle Reasons"] = record_aug["Cycle Index"].map(reason_map)

    # Save
    if output_xlsx is not None:
        with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
            # Write original sheets back (optional: to preserve, re-read from xls)
            for name in xls.sheet_names:
                try:
                    read_sheet(name).to_excel(writer, sheet_name=name, index=False)
                except Exception:
                    pass
            # Add new sheets
            cycle_metrics.to_excel(writer, sheet_name="cycle_labels", index=False)
            record_aug.to_excel(writer, sheet_name="labeled_record", index=False)

    return cycle_metrics, record_aug


def main():
    ap = argparse.ArgumentParser(description="Label battery cycles as GOOD/BAD and export augmented workbook.")
    ap.add_argument("path", help="Input Excel file path OR folder containing Excel files")
    ap.add_argument("-o", "--output", help="Output folder (defaults to same as input)", default=None)
    ap.add_argument("--cfg", help="JSON string or path for thresholds/penalties", default=None)
    args = ap.parse_args()

    cfg = DEFAULT_CFG
    if args.cfg:
        try:
            if args.cfg.strip().startswith("{"):
                cfg = json.loads(args.cfg)
            else:
                with open(args.cfg, "r") as f:
                    cfg = json.load(f)
        except Exception as e:
            print("Failed to parse cfg, using defaults. Error:", e)

    # Check if path is folder or file
    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "*.xlsx"))
    else:
        files = [args.path]

    outdir = args.output or args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    os.makedirs(outdir, exist_ok=True)

    for f in files:
        try:
            outfile = os.path.join(
                outdir,
                os.path.basename(f).replace(".xlsx", "_labeled.xlsx")
            )
            cycle_df, record_df = label_file(f, output_xlsx=outfile, cfg=cfg)
            print(f"[OK] {f} -> {outfile} ({len(cycle_df)} cycles)")
            print(cycle_df.head())

        except Exception as e:
            print(f"[FAIL] {f}: {e}")



# def main():
#     ap = argparse.ArgumentParser(description="Label battery cycles as GOOD/BAD and export augmented workbook.")
#     ap.add_argument("excel_path", help="Input Excel file path")
#     ap.add_argument("-o", "--output", help="Output Excel path; defaults to '<input>_labeled.xlsx'", default=None)
#     ap.add_argument("--cfg", help="JSON string or path for thresholds/penalties", default=None)
#     args = ap.parse_args()
#
#     cfg = DEFAULT_CFG
#     if args.cfg:
#         try:
#             # accept either JSON string or file path
#             if args.cfg.strip().startswith("{"):
#                 cfg = json.loads(args.cfg)
#             else:
#                 with open(args.cfg, "r") as f:
#                     cfg = json.load(f)
#         except Exception as e:
#             print("Failed to parse cfg, using defaults. Error:", e)
#
#     output = args.output or args.excel_path.replace(".xlsx", "_labeled.xlsx")
#     cycle_df, record_df = label_file(args.excel_path, output_xlsx=output, cfg=cfg)
#     print(f"Wrote: {output}")
#     print(cycle_df.head())


if __name__ == "__main__":
    main()
