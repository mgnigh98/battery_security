
#!/usr/bin/env python3
"""
battery_cycle_labeler_folder_v3.py

- Robust to sheet/column name variations (case, spaces, dots, units).
- Works with any number of cycles (from cycle 4 onward); can include last cycle via cfg.
- Charge linearity measured on middle 60% of CC charge; threshold 0.93.
- Skips soft penalties when cycle ends by voltage cutoff (~2.8V) AND CE≈1.
- Gentler robust outliers (MAD_K=5). Exposes SP_* flags.
- Writer fallback (xlsxwriter -> openpyxl). Silences only the openpyxl default-style warning.
- Adds 'End Reason' column: 'voltage_cutoff', 'time_cutoff', or 'unknown'.

Usage:
  python battery_cycle_labeler_folder_v3.py data/ -o results/
  python battery_cycle_labeler_folder_v3.py file.xlsx -o results/ --cfg '{"SKIP_LAST_CYCLE": false}'
"""

from __future__ import annotations
import argparse, json, os, glob, warnings, re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning)

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
    "MAD_K": 5.0,
    "SKIP_LAST_CYCLE": True
}

# ---- helpers ---------------------------------------------------

def norm(s: str) -> str:
    """normalize header: lowercase alphanumerics only"""
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """Rename columns by normalized match against candidates list"""
    inv = {}
    current = {norm(c): c for c in df.columns}
    for canon, candidates in mapping.items():
        found = None
        for cand in candidates:
            nc = norm(cand)
            if nc in current:
                found = current[nc]
                break
        if found is not None:
            inv[found] = canon
    return df.rename(columns=inv)

def r2_time_voltage_linear_fit(df: pd.DataFrame) -> float:
    if df.empty or df["time_h"].nunique() < 3:
        return float("nan")
    x = df["time_h"].to_numpy()
    y = df["voltage_v"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def middle_window(df: pd.DataFrame, low_q=0.2, high_q=0.8) -> pd.DataFrame:
    if df.empty:
        return df
    lo = df["time_h"].quantile(low_q)
    hi = df["time_h"].quantile(high_q)
    return df[(df["time_h"] >= lo) & (df["time_h"] <= hi)]

def ir_proxy(charge_df: pd.DataFrame, dchg_df: pd.DataFrame, window_s: float = 10.0) -> float:
    if charge_df.empty or dchg_df.empty:
        return float("nan")
    w = window_s / 3600.0
    t_max_chg = charge_df["time_h"].max()
    chg_win = charge_df[charge_df["time_h"] >= (t_max_chg - w)]
    dchg_win = dchg_df[dchg_df["time_h"] <= (dchg_df["time_h"].min() + w)]
    if chg_win.empty or dchg_win.empty:
        return float("nan")
    V1 = chg_win["voltage_v"].mean(); I1 = chg_win["current_a"].mean()
    V2 = dchg_win["voltage_v"].mean(); I2 = dchg_win["current_a"].mean()
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

def ended_by_cutoff(ve: float, cutoff=2.80, tol=0.03) -> bool:
    return (pd.notna(ve)) and (ve <= cutoff + tol)

def is_well_behaved_ce(ce: float, lo=0.95, hi=1.03) -> bool:
    return (pd.notna(ce)) and (lo <= ce <= hi)

# ---- core ------------------------------------------------------

def load_record_sheet(xls: pd.ExcelFile) -> pd.DataFrame:
    # accept any case for name
    sheets = {s.lower(): s for s in xls.sheet_names}
    if "record" not in sheets:
        raise ValueError(f"Missing required sheet 'record' (available: {xls.sheet_names})")
    df = pd.read_excel(xls, sheet_name=sheets["record"])

    # remap typical variants
    mapping = {
        "cycle_index": ["Cycle Index","CycleIndex","Cycle","cycle index","cycle_id"],
        "step_type":   ["Step Type","StepType","Type","Step"],
        "time_h":      ["Time(h)","Time (h)","Step Time(h)","Step Time (h)","TimeHour","Time"],
        "total_time_h":["Total Time(h)","Total Time (h)","TotalTime(h)"],
        "current_a":   ["Current(A)","Current (A)","Current"],
        "voltage_v":   ["Voltage(V)","Voltage (V)","Voltage"],
        "chg_cap_ah":  ["Chg. Cap.(Ah)","Chg. Cap. (Ah)","Charge Cap(Ah)","Charge Capacity(Ah)","Charge Capacity (Ah)","Charge(Ah)"],
        "dchg_cap_ah": ["DChg. Cap.(Ah)","DChg. Cap. (Ah)","Discharge Cap(Ah)","Discharge Capacity(Ah)","Discharge(Ah)"],
        "energy_wh":   ["Energy(Wh)","Energy (Wh)","Energy"]
    }
    df = remap_columns(df, mapping)

    needed = ["cycle_index","step_type","time_h","current_a","voltage_v","chg_cap_ah","dchg_cap_ah","energy_wh"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"'record' sheet missing columns (after remap): {missing}")
    return df

def load_cycle_sheet(xls: pd.ExcelFile) -> pd.DataFrame:
    sheets = {s.lower(): s for s in xls.sheet_names}
    if "cycle" not in sheets:
        return pd.DataFrame()
    df = pd.read_excel(xls, sheet_name=sheets["cycle"])
    mapping = {
        "cycle_index": ["Cycle Index","CycleIndex","Cycle"],
        "chg_cap_ah":  ["Chg. Cap.(Ah)","Chg. Cap. (Ah)","Charge Cap(Ah)","Charge Capacity(Ah)","Charge Capacity (Ah)"],
        "dchg_cap_ah": ["DChg. Cap.(Ah)","DChg. Cap. (Ah)","Discharge Cap(Ah)","Discharge Capacity(Ah)","Discharge Capacity (Ah)"],
        "chg_energy_wh":["Chg. Energy(Wh)","Chg. Energy (Wh)","Charge Energy(Wh)","Charge Energy (Wh)"],
        "dchg_energy_wh":["DChg. Energy(Wh)","DChg. Energy (Wh)","Discharge Energy(Wh)","Discharge Energy (Wh)"],
        "chg_time_h":  ["Chg. Time(h)","Charge Time(h)","Charge Time (h)"],
        "dchg_time_h": ["DChg. Time(h)","Discharge Time(h)","Discharge Time (h)"]
    }
    df = remap_columns(df, mapping)
    return df

def label_file(input_xlsx: str, output_xlsx: Optional[str], cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(input_xlsx)
    rec = load_record_sheet(xls)
    cyc = load_cycle_sheet(xls)

    all_cycles = sorted(set(int(c) for c in rec["cycle_index"].dropna().unique()))
    if not all_cycles:
        raise ValueError("No cycles found in 'record' sheet.")

    # cycles from 4 onward
    valid = [c for c in all_cycles if c >= 4]
    if cfg.get("SKIP_LAST_CYCLE", True) and len(valid) >= 2:
        valid = valid[:-1]

    def slice_cycle(c):
        cdf = rec[rec["cycle_index"] == c].copy()
        chg = cdf[cdf["step_type"].astype(str).str.startswith("CC Chg")]
        dch = cdf[cdf["step_type"].astype(str).str.contains("DChg")]
        return chg, dch

    rows = []
    for cidx in valid:
        chg_df, dchg_df = slice_cycle(cidx)
        n_chg, n_dch = len(chg_df), len(dchg_df)

        V_end_chg = float(chg_df["voltage_v"].iloc[-1]) if n_chg else np.nan
        V_max_chg = float(chg_df["voltage_v"].max()) if n_chg else np.nan
        V_min_dchg = float(dchg_df["voltage_v"].min()) if n_dch else np.nan
        V_end_dchg = float(dchg_df["voltage_v"].iloc[-1]) if n_dch else np.nan

        t_charge_h = float((chg_df["time_h"].max() - chg_df["time_h"].min())) if n_chg else np.nan
        t_discharge_h = float((dchg_df["time_h"].max() - dchg_df["time_h"].min())) if n_dch else 0.0

        cyc_row = cyc[cyc["cycle_index"] == cidx]
        Ah_chg = float(cyc_row["chg_cap_ah"].squeeze()) if ("chg_cap_ah" in cyc.columns and not cyc_row.empty) else np.nan
        Ah_dchg = float(cyc_row["dchg_cap_ah"].squeeze()) if ("dchg_cap_ah" in cyc.columns and not cyc_row.empty) else np.nan
        Wh_chg = float(cyc_row["chg_energy_wh"].squeeze()) if ("chg_energy_wh" in cyc.columns and not cyc_row.empty) else np.nan
        Wh_dchg = float(cyc_row["dchg_energy_wh"].squeeze()) if ("dchg_energy_wh" in cyc.columns and not cyc_row.empty) else np.nan

        if (np.isnan(Ah_chg) or Ah_chg == 0) and n_chg:
            Ah_chg = float(chg_df["chg_cap_ah"].max())
        if (np.isnan(Ah_dchg) or Ah_dchg == 0) and n_dch:
            Ah_dchg = float(dchg_df["dchg_cap_ah"].max())
        if (np.isnan(Wh_chg) or Wh_chg == 0) and n_chg:
            Wh_chg = float(chg_df["energy_wh"].max())
        if (np.isnan(Wh_dchg) or Wh_dchg == 0) and n_dch:
            Wh_dchg = float(dchg_df["energy_wh"].max())

        CE = (Ah_dchg / Ah_chg) if (Ah_chg and not np.isnan(Ah_chg)) else float("nan")
        EE = (Wh_dchg / Wh_chg) if (Wh_chg and not np.isnan(Wh_chg)) else float("nan")

        # Linearity on middle
        chg_mid = middle_window(chg_df, 0.2, 0.8)
        charge_r2 = r2_time_voltage_linear_fit(chg_mid)

        slope_mad = float("nan")
        if len(chg_mid) >= 3:
            t = chg_mid["time_h"].to_numpy()
            v = chg_mid["voltage_v"].to_numpy()
            dvdt = np.diff(v) / np.diff(t)
            slope_mad = mad(dvdt)

        IR_est = ir_proxy(chg_df, dchg_df)

        # End reason
        if ended_by_cutoff(V_end_dchg, cfg["V_END_DCHG_MAX"] - 0.1, tol=0.15):
            end_reason = "voltage_cutoff"
        elif t_discharge_h >= (cfg["TARGET_DISCHARGE_TIME"] - 0.05):
            end_reason = "time_cutoff"
        else:
            end_reason = "unknown"

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
            end_reason=end_reason,
            hard_fail=hard_fail,
            **{f"HF_{k}": bool(v) for k, v in hard_flags.items()},
        ))

    cycle_metrics = pd.DataFrame(rows).sort_values("Cycle").reset_index(drop=True)

    # Soft scoring
    k = cfg["MAD_K"]; start = cfg["SCORE_START"]; good_thr = cfg["SCORE_GOOD_THRESHOLD"]; P = cfg["PENALTIES"]
    cycle_metrics["soft_score"] = start
    for col in ["SP_Ah_chg_outlier","SP_Ah_dchg_outlier","SP_Wh_dchg_outlier","SP_time_dev","SP_CE_marginal","SP_IR_marginal","SP_slope_var"]:
        cycle_metrics[col] = False
    ok = ~cycle_metrics["hard_fail"].astype(bool)

    def allow_penalties(idx):
        r = cycle_metrics.loc[idx]
        return not (ended_by_cutoff(r["V_end_dchg"], cfg["V_END_DCHG_MAX"], 0.03) and is_well_behaved_ce(r["CE"]))

    if len(cycle_metrics) > 5:
        m = outlier_mask(cycle_metrics["Ah_chg"], k); sel = ok & m & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["AH_CHG_OUTLIER"]; cycle_metrics.loc[sel, "SP_Ah_chg_outlier"] = True

        m = outlier_mask(cycle_metrics["Ah_dchg"], k); sel = ok & m & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["AH_DCHG_OUTLIER"]; cycle_metrics.loc[sel, "SP_Ah_dchg_outlier"] = True

        m = outlier_mask(cycle_metrics["Wh_dchg"], k); sel = ok & m & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["WH_DCHG_OUTLIER"]; cycle_metrics.loc[sel, "SP_Wh_dchg_outlier"] = True

        dev = (np.abs(cycle_metrics["t_discharge_h"] - cfg["TARGET_DISCHARGE_TIME"]) > 0.10 * cfg["TARGET_DISCHARGE_TIME"])
        cutoff_hit = cycle_metrics["V_end_dchg"] <= (cfg["V_END_DCHG_MAX"] + 0.02)
        sel = ok & dev & (~cutoff_hit); cycle_metrics.loc[sel, "soft_score"] -= P["DISCHARGE_TIME_DEVIATION"]; cycle_metrics.loc[sel, "SP_time_dev"] = True

        CE = cycle_metrics["CE"]; CE_abs_ok = CE.between(cfg["CE_ABS_LOW"], cfg["CE_ABS_HIGH"], inclusive="both"); CE_marg = ~CE.between(0.96, 1.02, inclusive="both")
        sel = ok & CE_abs_ok & CE_marg & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["CE_MARGINAL"]; cycle_metrics.loc[sel, "SP_CE_marginal"] = True

        IR = cycle_metrics["IR_est"]; IR_out = outlier_mask(IR, k)
        sel = ok & IR_out & (IR < cfg["IR_ABS_MAX"]) & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["IR_MARGINAL"]; cycle_metrics.loc[sel, "SP_IR_marginal"] = True

        SM = cycle_metrics["slope_mad"]; SM_out = outlier_mask(SM, k)
        sel = ok & SM_out & cycle_metrics.index.to_series().apply(allow_penalties)
        cycle_metrics.loc[sel, "soft_score"] -= P["CHARGE_SLOPE_VARIABILITY"]; cycle_metrics.loc[sel, "SP_slope_var"] = True

    cycle_metrics["Label"] = np.where(
        cycle_metrics["hard_fail"], "BAD",
        np.where(cycle_metrics["soft_score"] >= good_thr, "GOOD", "BAD")
    )

    # Reasons
    HF_cols = [c for c in cycle_metrics.columns if c.startswith("HF_")]
    SP_cols = [c for c in cycle_metrics.columns if c.startswith("SP_")]
    def reasons_for_row(r: pd.Series) -> str:
        reasons = [c.replace("HF_","") for c in HF_cols if bool(r[c])]
        reasons += [c for c in SP_cols if bool(r[c])]
        if not reasons and r["soft_score"] < start:
            reasons.append(f"soft_penalty:{int(start - r['soft_score'])}")
        if r.get("end_reason"):
            reasons.append(f"end:{r['end_reason']}")
        return ", ".join(reasons)
    cycle_metrics["Reasons"] = cycle_metrics.apply(reasons_for_row, axis=1)

    # Augment record (map by cycle_index)
    record_aug = rec.copy()
    lmap = dict(zip(cycle_metrics["Cycle"], cycle_metrics["Label"]))
    rmap = dict(zip(cycle_metrics["Cycle"], cycle_metrics["Reasons"]))
    record_aug["Cycle Label"] = record_aug["cycle_index"].map(lmap)
    record_aug["Cycle Reasons"] = record_aug["cycle_index"].map(rmap)

    if output_xlsx:
        try:
            with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as w:
                cycle_metrics.to_excel(w, sheet_name="cycle_labels", index=False)
                record_aug.to_excel(w, sheet_name="labeled_record", index=False)
        except Exception:
            with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
                cycle_metrics.to_excel(w, sheet_name="cycle_labels", index=False)
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

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.xlsx")))
        outdir = args.outdir or args.path
    else:
        files = [args.path]
        outdir = args.outdir or os.path.dirname(os.path.abspath(args.path)) or "."
    os.makedirs(outdir, exist_ok=True)

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
    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(os.path.join(outdir, "batch_summary.csv"), index=False)
        print(f"Wrote batch_summary.csv in {outdir}")

if __name__ == "__main__":
    main()
