#!/usr/bin/env python3
"""
cycle_label_v4.py

Cycle labelling using ONLY the requested parameters:

Monitored per cycle (from 'cycle' sheet, with fallbacks to 'record'):
1) Charge Capacity (Ah, mAh/g)
2) Discharge Capacity (Ah, mAh/g)
3) Charge Energy (Wh, mWh/g)
4) Discharge Energy (Wh, mWh/g)
5) Coulombic Efficiency (CE = DChg Cap / Chg Cap)

Hard rules (BAD if any triggered):
  H1. Charge specific capacity > 190 mAh/g
  H2. (Optional) Charging voltage fluctuation (dv/dt < 0) detected during CC-Chg
  H3. Discharge specific capacity < 120 mAh/g
  H4. Missing essential data points

Soft rules (score penalties):
  S1. IR proxy = (Wh/Ah)_charge − (Wh/Ah)_discharge  deviates strongly (outlier detection)
      -> Use robust MAD bounds (median ± K·MAD). Plots are generated to visualize outliers.
  S2. Coulombic Efficiency outside 0.95–1.05

Scoring:
  - Start score = 100; subtract penalties; GOOD if score >= 70 else BAD.
  - Penalty magnitudes are configurable.
"""

import argparse, json, os, glob, warnings, re
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning)

# ---------------- Config ----------------
DEFAULT_CFG = {
    "HARD_CHG_SPEC_CAP_MAX": 190.0,
    "HARD_DCHG_SPEC_CAP_MIN": 120.0,
    "CE_SOFT_LOW": 0.95,
    "CE_SOFT_HIGH": 1.05,
    "IR_PROXY_MAX": 0.422,
    # "IR_PROXY_MAX": 0.385,  # <= good, > bad (soft rule S1)
    # "IR_MAD_K": 3.5,
    "SCORE_START": 100,
    "SCORE_GOOD_THRESHOLD": 70,
    "PENALTIES": {
        "CE_SOFT": 10,
        "IR_OUTLIER": 20,
        "CE_EXCESS_PER_0p01": 2
    },
    "CHECK_DVDT_NEG": False,
    "DVDT_MIN_POINTS": 5,
    "DVDT_NEG_FRACTION": 0.05,
    "START_FROM_CYCLE": 4,
    "SKIP_LAST_CYCLE": True,
}

# ---------------- Helpers ----------------
def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    inv = {}
    current = {norm(c): c for c in df.columns}
    for canon, candidates in mapping.items():
        for cand in candidates:
            key = norm(cand)
            if key in current:
                inv[current[key]] = canon
                break
    return df.rename(columns=inv)

def load_cycle_sheet(xls: pd.ExcelFile) -> pd.DataFrame:
    sheets = {s.lower(): s for s in xls.sheet_names}
    if "cycle" not in sheets:
        raise ValueError("Missing required sheet 'cycle'.")
    df = pd.read_excel(xls, sheet_name=sheets["cycle"])
    mapping = {
        "cycle_index": ["Cycle Index","CycleIndex","Cycle"],
        "chg_cap_ah":  ["Chg. Cap.(Ah)","Charge Capacity(Ah)"],
        "dchg_cap_ah": ["DChg. Cap.(Ah)","Discharge Capacity(Ah)"],
        "chg_spec_mAhg": ["Chg. Spec. Cap.(mAh/g)"],
        "dchg_spec_mAhg":["DChg. Spec. Cap.(mAh/g)"],
        "chg_energy_wh": ["Chg. Energy(Wh)"],
        "dchg_energy_wh":["DChg. Energy(Wh)"],
        "chg_spec_mWhg": ["Chg. Spec. Energy(mWh/g)"],
        "dchg_spec_mWhg":["DChg. Spec. Energy(mWh/g)"],
    }
    return remap_columns(df, mapping)

# ---------------- Core ----------------
def label_file(path: str, out_xlsx: Optional[str], out_dir_plots: Optional[str], cfg: Dict[str,Any]) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    cyc = load_cycle_sheet(xls)

    # Select cycles
    all_cycles = sorted(set(int(c) for c in cyc["cycle_index"].dropna().unique()))
    valid = [c for c in all_cycles if c >= cfg["START_FROM_CYCLE"]]
    if cfg["SKIP_LAST_CYCLE"] and len(valid) >= 2:
        valid = valid[:-1]

    rows: List[Dict[str,Any]] = []
    for cidx in valid:
        row_cyc = cyc[cyc["cycle_index"]==cidx].iloc[0]

        chg_cap_ah  = float(row_cyc.get("chg_cap_ah", np.nan))
        dchg_cap_ah = float(row_cyc.get("dchg_cap_ah", np.nan))
        chg_spec    = float(row_cyc.get("chg_spec_mAhg", np.nan))
        dchg_spec   = float(row_cyc.get("dchg_spec_mAhg", np.nan))
        chg_wh      = float(row_cyc.get("chg_energy_wh", np.nan))
        dchg_wh     = float(row_cyc.get("dchg_energy_wh", np.nan))

        CE = dchg_cap_ah / chg_cap_ah if chg_cap_ah and not np.isnan(chg_cap_ah) else np.nan
        ir_chg = chg_wh / chg_cap_ah if chg_cap_ah else np.nan
        ir_dch = dchg_wh / dchg_cap_ah if dchg_cap_ah else np.nan
        ir_proxy = (ir_chg - ir_dch) if not np.isnan(ir_chg) and not np.isnan(ir_dch) else np.nan

        rows.append(dict(
            Cycle=cidx,
            Chg_Spec_mAhg=chg_spec, DChg_Spec_mAhg=dchg_spec,
            Chg_Cap_Ah=chg_cap_ah, DChg_Cap_Ah=dchg_cap_ah,
            Chg_Energy_Wh=chg_wh, DChg_Energy_Wh=dchg_wh,
            CE=CE, IR_proxy=ir_proxy
        ))

    df = pd.DataFrame(rows).sort_values("Cycle").reset_index(drop=True)

    # Hard rules
    df["HF_CHG_SPEC_HIGH"]  = df["Chg_Spec_mAhg"] > cfg["HARD_CHG_SPEC_CAP_MAX"]
    df["HF_DCHG_SPEC_LOW"]  = df["DChg_Spec_mAhg"] < cfg["HARD_DCHG_SPEC_CAP_MIN"]
    df["HF_MISSING"] = df[["Chg_Cap_Ah","DChg_Cap_Ah","Chg_Energy_Wh","DChg_Energy_Wh","CE"]].isna().any(axis=1)
    df["hard_fail"] = df[["HF_CHG_SPEC_HIGH","HF_DCHG_SPEC_LOW","HF_MISSING"]].any(axis=1)

    # Soft rules
    start = cfg["SCORE_START"]
    thr = cfg["SCORE_GOOD_THRESHOLD"]
    P = cfg["PENALTIES"]
    df["soft_score"] = start
    df["SP_IR_OUTLIER"] = False
    df["SP_CE_SOFT"] = False

    # IR soft rule (constant threshold): IR_proxy > IR_PROXY_MAX => soft penalty
    # ir_thresh = float(cfg.get("IR_PROXY_MAX", 0.385))
    ir_thresh = float(cfg.get("IR_PROXY_MAX", 0.422))
    ir_bad = df["IR_proxy"].astype(float) > ir_thresh

    # apply penalty only if not already hard-failed
    df.loc[ir_bad & (~df["hard_fail"]), "soft_score"] -= P.get("IR_OUTLIER", 20)
    df.loc[ir_bad, "SP_IR_OUTLIER"] = True
    df["IR_proxy_threshold"] = ir_thresh  # optional for traceability

    # CE soft band
    ce = df["CE"].astype(float)
    out_any = ((ce < cfg["CE_SOFT_LOW"]) | (ce > cfg["CE_SOFT_HIGH"])) & (~df["hard_fail"])
    df.loc[out_any, "soft_score"] -= P["CE_SOFT"]
    df.loc[out_any, "SP_CE_SOFT"] = True

    # Final label
    df["Label"] = np.where(df["hard_fail"], "BAD",
                    np.where(df["soft_score"] >= thr, "GOOD", "BAD"))

    if out_xlsx:
        df.to_excel(out_xlsx, sheet_name="cycle_labels", index=False)

        # ---------------- IR outlier plot ----------------
    if out_dir_plots:
        import matplotlib.pyplot as plt
        os.makedirs(out_dir_plots, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df["Cycle"], df["IR_proxy"], marker="o", linestyle="-", label="IR proxy")
        ax.axhline(ir_thresh, color="r", linestyle="--", label=f"IR threshold = {ir_thresh:.3f}")
        out_pts = df[ir_bad]
        ax.scatter(out_pts["Cycle"], out_pts["IR_proxy"], color="red", zorder=3, label="outlier")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("IR proxy = (Wh/Ah)_chg - (Wh/Ah)_dchg")
        ax.set_title("IR Proxy Outlier Detection")
        ax.legend()
        plot_path = os.path.join(out_dir_plots, os.path.basename(path).replace(".xlsx", "_IRproxy.png"))
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close(fig)

    return df

def main():
    ap = argparse.ArgumentParser(description="Cycle labelling script")
    ap.add_argument("path", help="Excel file or folder")
    ap.add_argument("-o","--outdir", default=None, help="Output folder")
    args = ap.parse_args()

    cfg = DEFAULT_CFG.copy()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.xlsx")))
        outdir = args.outdir or args.path
    else:
        files = [args.path]
        outdir = args.outdir or os.path.dirname(os.path.abspath(args.path)) or "."
    os.makedirs(outdir, exist_ok=True)

    print(f"Found {len(files)} file(s). Output dir: {outdir}")
    # for f in files:
    #     base = os.path.basename(f)
    #     try:
    #         out_xlsx = os.path.join(outdir, base.replace(".xlsx","_labeled.xlsx"))
    #         df = label_file(f, out_xlsx, None, cfg)
    #         good = (df["Label"]=="GOOD").sum(); bad = (df["Label"]=="BAD").sum()
    #         print(f"[OK] {base}: cycles={len(df)} GOOD={good} BAD={bad}")
    #     except Exception as e:
    #         print(f"[FAIL] {base}: {e}")
    summary = []
    for f in files:
        base = os.path.basename(f)
        try:
            out_xlsx = os.path.join(outdir, base.replace(".xlsx", "_labeled.xlsx"))
            plots_dir = os.path.join(outdir, "plots")
            df = label_file(f, out_xlsx, plots_dir, cfg)
            good = int((df["Label"] == "GOOD").sum())
            bad = int((df["Label"] == "BAD").sum())
            print(f"[OK] {base}: cycles={len(df)} GOOD={good} BAD={bad}")
            summary.append({"file": base, "cycles": len(df), "good": good, "bad": bad})
        except Exception as e:
            print(f"[FAIL] {base}: {e}")
            summary.append({"file": base, "cycles": 0, "good": 0, "bad": 0, "error": str(e)})

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(outdir, "batch_summary.csv"), index=False)
        print(f"Wrote batch_summary.csv in {outdir}")

if __name__ == "__main__":
    main()
