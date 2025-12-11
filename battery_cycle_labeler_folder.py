
"""
battery_cycle_labeler_folder.py

Batch-labels battery cycles as GOOD/BAD across a file or an entire folder of Excel workbooks.
- Skips formation cycles (1..3) and the *last* cycle (possibly incomplete).
- Adds per-cycle metrics and labels; exports original core sheets + augmented sheets.
- Gracefully skips files where required sheets/columns are missing or where 'record' has no cycles.
- Prints a per-file summary line.

Author: ChatGPT
"""

from __future__ import annotations
import argparse
import json
import os
import glob
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style",
    category=UserWarning
)

DEFAULT_CFG = {
    "V_END_CHG_MIN": 4.25,
    "V_MAX_CHG_MAX": 4.35,
    "V_MIN_DCHG_MIN": 2.70,
    "V_END_DCHG_MAX": 2.90,
    "DISCHARGE_TIME_MIN": 1.20,
    "TARGET_DISCHARGE_TIME": 1.50,
    "CE_ABS_LOW": 0.93,
    "CE_ABS_HIGH": 1.05,
    "CHARGE_LINEARITY_R2_MIN": 0.93,
    "IR_ABS_MAX": 0.30,
    "MIN_POINTS_CHG": 20,
    "MIN_POINTS_DCHG": 20,
    "SCORE_START": 100,
    "SCORE_GOOD_THRESHOLD": 70,
    "PENALTIES": {
        "AH_CHG_OUTLIER": 15,
        "AH_DCHG_OUTLIER": 20,
        "WH_DCHG_OUTLIER": 15,
        "DISCHARGE_TIME_DEVIATION": 10,
        "CE_MARGINAL": 10,
        "IR_MARGINAL": 10,
        "CHARGE_SLOPE_VARIABILITY": 10
    },
    "MAD_K": 3.0,
    "SKIP_LAST_CYCLE": True   # as requested
}

# -------------- utils --------------

def r2_time_voltage_linear_fit(df: pd.DataFrame) -> float:
    if df.empty or df["Time(h)"].nunique() < 3:
        return float("nan")
    x = df["Time(h)"].to_numpy()
    y = df["Voltage(V)"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def ir_proxy(charge_df: pd.DataFrame, dchg_df: pd.DataFrame, window_s: float = 10.0) -> float:
    if charge_df.empty or dchg_df.empty:
        return float("nan")
    w = window_s / 3600.0
    t_max_chg = charge_df["Time(h)"].max()
    chg_win = charge_df[charge_df["Time(h)"] >= (t_max_chg - w)]
    dchg_win = dchg_df[dchg_df["Time(h)"] <= (dchg_df["Time(h)"].min() + w)]
    if chg_win.empty or dchg_win.empty:
        return float("nan")
    V1 = chg_win["Voltage(V)"].mean(); I1 = chg_win["Current(A)"].mean()
    V2 = dchg_win["Voltage(V)"].mean(); I2 = dchg_win["Current(A)"].mean()
    dV = V1 - V2; dI = I2 - I1
    if np.isclose(dI, 0, atol=1e-6):
        return float("nan")
    return float(dV / dI)

def robust_bounds(x: np.ndarray, k: float) -> Tuple[float, float]:
    m = float(np.nanmedian(x))
    s = mad(x)
    return (m - k * s, m + k * s)

def outlier_mask(series: pd.Series, k: float) -> pd.Series:
    x = series.to_numpy(dtype=float)
    lo, hi = robust_bounds(x, k)
    return (series < lo) | (series > hi)

# -------------- core --------------

def label_file(input_xlsx: str, output_xlsx: Optional[str], cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(input_xlsx)
    sheets = {name.lower(): name for name in xls.sheet_names}
    required = ["test", "cycle", "record", "step"]
    for r in required:
        if r not in sheets:
            raise ValueError(f"Missing required sheet '{r}' (available: {xls.sheet_names})")

    # Load
    test_df  = pd.read_excel(xls, sheet_name=sheets["test"])
    cycle_df = pd.read_excel(xls, sheet_name=sheets["cycle"])
    record_df = pd.read_excel(xls, sheet_name=sheets["record"])
    step_df  = pd.read_excel(xls, sheet_name=sheets["step"])

    # Validate essential columns (record)
    rec_needed = ["Cycle Index", "Step Type", "Time(h)", "Current(A)", "Voltage(V)",
                  "Chg. Cap.(Ah)", "DChg. Cap.(Ah)", "Energy(Wh)"]
    for col in rec_needed:
        if col not in record_df.columns:
            raise ValueError(f"'record' sheet missing column '{col}'")

    # Determine cycles
    if "Cycle Index" not in record_df.columns or record_df["Cycle Index"].dropna().nunique() == 0:
        raise ValueError("No cycles found in 'record' sheet.")
    all_cycles = sorted(set(int(c) for c in record_df["Cycle Index"].dropna().unique()))
    if len(all_cycles) <= 4:
        raise ValueError("Not enough cycles after formation to evaluate.")

    # skip 1..3 and the last
    last = max(all_cycles)
    valid = [c for c in all_cycles if 4 <= c < (last if cfg.get("SKIP_LAST_CYCLE", True) else last + 1)]

    def middle_window(df, tcol="Time(h)", low_q=0.2, high_q=0.8):
        if df.empty:
            return df
        lo = df[tcol].quantile(low_q)
        hi = df[tcol].quantile(high_q)
        return df[(df[tcol] >= lo) & (df[tcol] <= hi)]

    # helper
    def slice_cycle(cidx: int):
        cdf = record_df[record_df["Cycle Index"] == cidx].copy()
        chg = cdf[cdf["Step Type"].astype(str).str.startswith("CC Chg")]
        dch = cdf[cdf["Step Type"].astype(str).str.contains("DChg")]
        return chg, dch

    rows: List[Dict[str, Any]] = []
    for cidx in valid:
        chg_df, dchg_df = slice_cycle(cidx)
        n_chg, n_dch = len(chg_df), len(dchg_df)

        V_end_chg = float(chg_df["Voltage(V)"].iloc[-1]) if n_chg else np.nan
        V_max_chg = float(chg_df["Voltage(V)"].max()) if n_chg else np.nan
        V_min_dchg = float(dchg_df["Voltage(V)"].min()) if n_dch else np.nan
        V_end_dchg = float(dchg_df["Voltage(V)"].iloc[-1]) if n_dch else np.nan

        t_charge_h = float((chg_df["Time(h)"].max() - chg_df["Time(h)"].min())) if n_chg else np.nan
        t_discharge_h = float((dchg_df["Time(h)"].max() - dchg_df["Time(h)"].min())) if n_dch else 0.0

        cyc_row = cycle_df[cycle_df["Cycle Index"] == cidx]
        Ah_chg = float(cyc_row["Chg. Cap.(Ah)"].squeeze()) if not cyc_row.empty else np.nan
        Ah_dchg = float(cyc_row["DChg. Cap.(Ah)"].squeeze()) if not cyc_row.empty else np.nan
        Wh_chg = float(cyc_row["Chg. Energy(Wh)"].squeeze()) if not cyc_row.empty else np.nan
        Wh_dchg = float(cyc_row["DChg. Energy(Wh)"].squeeze()) if not cyc_row.empty else np.nan

        if (np.isnan(Ah_chg) or Ah_chg == 0) and n_chg:
            Ah_chg = float(chg_df["Chg. Cap.(Ah)"].max())
        if (np.isnan(Ah_dchg) or Ah_dchg == 0) and n_dch:
            Ah_dchg = float(dchg_df["DChg. Cap.(Ah)"].max())
        if (np.isnan(Wh_chg) or Wh_chg == 0) and n_chg:
            Wh_chg = float(chg_df["Energy(Wh)"].max())
        if (np.isnan(Wh_dchg) or Wh_dchg == 0) and n_dch:
            Wh_dchg = float(dchg_df["Energy(Wh)"].max())

        CE = (Ah_dchg / Ah_chg) if (Ah_chg and not np.isnan(Ah_chg)) else float("nan")
        EE = (Wh_dchg / Wh_chg) if (Wh_chg and not np.isnan(Wh_chg)) else float("nan")

        # charge_r2 = r2_time_voltage_linear_fit(chg_df)
        chg_mid = middle_window(chg_df, "Time(h)", 0.2, 0.8)
        charge_r2 = r2_time_voltage_linear_fit(chg_mid)

        slope_mad = float("nan")
        if n_chg >= 3:
            t = chg_df["Time(h)"].to_numpy()
            v = chg_df["Voltage(V)"].to_numpy()
            dvdt = np.diff(v) / np.diff(t)
            slope_mad = mad(dvdt)

        IR_est = ir_proxy(chg_df, dchg_df)

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

        rows.append(dict(
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
            CE=CE, EE=EE,
            charge_r2=charge_r2,
            slope_mad=slope_mad,
            IR_est=IR_est,
            hard_fail=hard_fail,
            **{f"HF_{k}": bool(v) for k, v in hard_flags.items()},
        ))

    cycle_metrics = pd.DataFrame(rows).sort_values("Cycle")

    def ended_by_cutoff(row, cutoff=2.80, tol=0.03):
        ve = row["V_end_dchg"]
        return pd.notna(ve) and (ve <= cutoff + tol)

    def is_well_behaved_ce(row, lo=0.95, hi=1.03):
        ce = row["CE"]
        return pd.notna(ce) and (lo <= ce <= hi)

    # soft scoring with file-level robust stats
    k = cfg["MAD_K"]
    start = cfg["SCORE_START"]
    good_thr = cfg["SCORE_GOOD_THRESHOLD"]
    P = cfg["PENALTIES"]

    cycle_metrics["soft_score"] = start
    ok = ~cycle_metrics["hard_fail"].astype(bool)

    def allow_penalties(idx):
        r = cycle_metrics.loc[idx]
        # If it ended by voltage cutoff and CE is near 1, don't penalize minor Ah/Wh stats.
        return not (ended_by_cutoff(r) and is_well_behaved_ce(r))

    if len(cycle_metrics) > 5:
        # Example for Ah_chg outlier
        m = outlier_mask(cycle_metrics["Ah_chg"], k)
        sel = ok & m & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["AH_CHG_OUTLIER"]

        m = outlier_mask(cycle_metrics["Ah_dchg"], k)
        sel = ok & m & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["AH_DCHG_OUTLIER"]

        m = outlier_mask(cycle_metrics["Wh_dchg"], k)
        sel = ok & m & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["WH_DCHG_OUTLIER"]


    if len(cycle_metrics) > 5:
        for col, pen_key in [("Ah_chg","AH_CHG_OUTLIER"),("Ah_dchg","AH_DCHG_OUTLIER"),("Wh_dchg","WH_DCHG_OUTLIER")]:
            m = outlier_mask(cycle_metrics[col], k)
            cycle_metrics.loc[ok & m, "soft_score"] -= P[pen_key]

        dev = (np.abs(cycle_metrics["t_discharge_h"] - cfg["TARGET_DISCHARGE_TIME"]) > 0.10 * cfg["TARGET_DISCHARGE_TIME"])
        cutoff_hit = cycle_metrics["V_end_dchg"] <= (cfg["V_END_DCHG_MAX"] + 0.02)
        cycle_metrics.loc[ok & dev & (~cutoff_hit), "soft_score"] -= P["DISCHARGE_TIME_DEVIATION"]

        CE = cycle_metrics["CE"]
        CE_abs_ok = CE.between(cfg["CE_ABS_LOW"], cfg["CE_ABS_HIGH"], inclusive="both")
        CE_marg = ~CE.between(0.96, 1.02, inclusive="both")
        cycle_metrics.loc[ok & CE_abs_ok & CE_marg, "soft_score"] -= P["CE_MARGINAL"]

        IR = cycle_metrics["IR_est"]
        IR_out = outlier_mask(IR, k)
        cycle_metrics.loc[ok & IR_out & (IR < cfg["IR_ABS_MAX"]), "soft_score"] -= P["IR_MARGINAL"]

        SM = cycle_metrics["slope_mad"]
        SM_out = outlier_mask(SM, k)
        cycle_metrics.loc[ok & SM_out, "soft_score"] -= P["CHARGE_SLOPE_VARIABILITY"]

    cycle_metrics["Label"] = np.where(
        cycle_metrics["hard_fail"], "BAD",
        np.where(cycle_metrics["soft_score"] >= good_thr, "GOOD", "BAD")
    )

    # reasons
    HF_cols = [c for c in cycle_metrics.columns if c.startswith("HF_")]
    def reasons_for_row(r: pd.Series) -> str:
        reasons = [c.replace("HF_","") for c in HF_cols if bool(r[c])]
        if not reasons and r["soft_score"] < start:
            reasons.append(f"soft_penalty:{int(start - r['soft_score'])}")
        return ", ".join(reasons)
    cycle_metrics["Reasons"] = cycle_metrics.apply(reasons_for_row, axis=1)

    # augment record
    record_aug = record_df.copy()
    lmap = dict(zip(cycle_metrics["Cycle"], cycle_metrics["Label"]))
    rmap = dict(zip(cycle_metrics["Cycle"], cycle_metrics["Reasons"]))
    record_aug["Cycle Label"] = record_aug["Cycle Index"].map(lmap)
    record_aug["Cycle Reasons"] = record_aug["Cycle Index"].map(rmap)

    # Save (write fewer sheets to keep IO small)
    if output_xlsx:
        with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as w:
            cycle_metrics.to_excel(w, sheet_name="cycle_labels", index=False)
            # write record with labels only to limit file size
            record_aug.to_excel(w, sheet_name="labeled_record", index=False)
    return cycle_metrics, record_aug

def main():
    ap = argparse.ArgumentParser(description="Label battery cycles for a file or all Excel files in a folder.")
    ap.add_argument("path", help="Path to an Excel file or a folder containing .xlsx files")
    ap.add_argument("-o", "--outdir", help="Output folder (default: same as input path)", default=None)
    ap.add_argument("--cfg", help="JSON string or path to JSON config", default=None)
    args = ap.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.cfg:
        try:
            if args.cfg.strip().startswith("{"):
                cfg.update(json.loads(args.cfg))
            else:
                with open(args.cfg, "r") as f:
                    cfg.update(json.load(f))
        except Exception as e:
            print("WARNING: Failed to parse cfg; using defaults.", e)

    # Build list of files
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.xlsx")))
        outdir = args.outdir or args.path
    else:
        files = [args.path]
        outdir = args.outdir or os.path.dirname(os.path.abspath(args.path)) or "."
    os.makedirs(outdir, exist_ok=True)

    # Process
    print(f"Found {len(files)} file(s). Output dir: {outdir}")
    summary = []
    for f in files:
        base = os.path.basename(f)
        try:
            out = os.path.join(outdir, base.replace(".xlsx", "_labeled.xlsx"))
            cyc, _ = label_file(f, out, cfg)
            good = int((cyc["Label"] == "GOOD").sum())
            bad = int((cyc["Label"] == "BAD").sum())
            print(f"[OK] {base}: cycles={len(cyc)}  GOOD={good}  BAD={bad} -> {os.path.basename(out)}")
            summary.append({"file": base, "cycles": len(cyc), "good": good, "bad": bad})
        except Exception as e:
            print(f"[SKIP] {base}: {e}")
            summary.append({"file": base, "cycles": 0, "good": 0, "bad": 0, "error": str(e)})

    # Emit a CSV summary in the output folder
    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(os.path.join(outdir, "batch_summary.csv"), index=False)
        print(f"Wrote batch_summary.csv in {outdir}")

if __name__ == "__main__":
    main()
