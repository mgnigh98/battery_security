#!/usr/bin/env python3
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

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    cur = { _norm(c): c for c in df.columns }
    ren = {}
    for canon, cand_list in mapping.items():
        for cand in cand_list:
            key = _norm(cand)
            if key in cur:
                ren[cur[key]] = canon
                break
    return df.rename(columns=ren)

def load_cycle_labels(labeled_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(labeled_path)
    sheet = None
    for cand in ["cycle_labels", "labels"]:
        if cand in xl.sheet_names:
            sheet = cand
            break
    if sheet is None:
        raise RuntimeError("No cycle label sheet found (expected cycle_labels or labels).")

    lab = pd.read_excel(xl, sheet_name=sheet)
    lab = remap_columns(lab, {
        "cycle": ["Cycle", "Cycle Index", "cycle_index", "cycle"],
        "label": ["Label", "label", "cycle_label", "Cycle Label"]
    })
    if "cycle" not in lab.columns or "label" not in lab.columns:
        raise RuntimeError("Label sheet missing Cycle/Label columns.")

    lab["cycle"] = pd.to_numeric(lab["cycle"], errors="coerce")
    lab = lab.dropna(subset=["cycle"]).copy()
    lab["cycle"] = lab["cycle"].astype(int)
    lab["label"] = lab["label"].astype(str).str.upper()
    return lab[["cycle", "label"]].drop_duplicates()

def load_step_sheet(xls: pd.ExcelFile) -> pd.DataFrame:
    name = None
    for s in xls.sheet_names:
        if s.lower() == "step":
            name = s; break
    if name is None:
        raise RuntimeError("Missing sheet 'step' in original workbook.")

    df = pd.read_excel(xls, sheet_name=name)
    mapping = {
        "cycle_index":     ["Cycle Index","Cycle","cycle index"],
        "step_number":     ["Step Number","StepNumber","Step No.","StepNo"],
        "step_type":       ["Step Type","StepType","Type","Step"],
        "step_time_h":     ["Step Time(h)","Step Time (h)","Time(h)","Time (h)"],
        "start_voltage_v": ["Oneset Volt.(V)","Onset Voltage(V)","Start Voltage(V)"],
        "end_voltage_v":   ["End Voltage(V)","End Voltage (V)","End Volt.(V)"],
        "chg_cap_ah":      ["Chg. Cap.(Ah)","Charge Capacity(Ah)"],
        "dchg_cap_ah":     ["DChg. Cap.(Ah)","Discharge Capacity(Ah)"],
        "chg_spec":        ["Chg. Spec. Cap.(mAh/g)","Chg Spec Cap (mAh/g)"],
        "dchg_spec":       ["DChg. Spec. Cap.(mAh/g)","DChg Spec Cap (mAh/g)"],
    }
    df = remap_columns(df, mapping)

    # numeric coercions
    for c in ["cycle_index","step_number","step_time_h","start_voltage_v","end_voltage_v",
              "chg_cap_ah","dchg_cap_ah","chg_spec","dchg_spec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # fallback if step_number absent: use original row order
    if "step_number" not in df.columns:
        df["step_number"] = np.arange(len(df), dtype=float)

    return df

def pick_1chg_5dchg(step_df: pd.DataFrame, cycle: int) -> Optional[Dict[str, float]]:
    cdf = step_df[step_df["cycle_index"] == cycle].copy()
    if cdf.empty:
        return None

    cdf = cdf.sort_values("step_number", na_position="last")

    # prefer spec-cap if available, else Ah
    if "chg_spec" in cdf.columns and cdf["chg_spec"].notna().any():
        chg_rows = cdf[cdf["chg_spec"] > 0]
    else:
        chg_rows = cdf[cdf.get("chg_cap_ah", pd.Series([np.nan]*len(cdf))) > 0]

    if chg_rows.empty:
        return None
    chg = chg_rows.iloc[0]
    chg_pos = chg.name

    after = cdf.loc[cdf.index > chg_pos].copy()
    if after.empty:
        return None

    if "dchg_spec" in after.columns and after["dchg_spec"].notna().any():
        d_rows = after[after["dchg_spec"] > 0]
    else:
        d_rows = after[after.get("dchg_cap_ah", pd.Series([np.nan]*len(after))) > 0]

    if len(d_rows) < 5:
        return None

    d5 = d_rows.iloc[:5]

    out = {
        "charge_V_end": float(chg["end_voltage_v"]) if pd.notna(chg.get("end_voltage_v")) else np.nan,
        "take_off_V_end": float(d5.iloc[0]["end_voltage_v"]) if pd.notna(d5.iloc[0].get("end_voltage_v")) else np.nan,
        "hover_V_end":    float(d5.iloc[1]["end_voltage_v"]) if pd.notna(d5.iloc[1].get("end_voltage_v")) else np.nan,
        "cruise_V_end":   float(d5.iloc[2]["end_voltage_v"]) if pd.notna(d5.iloc[2].get("end_voltage_v")) else np.nan,
        "landing_V_end":  float(d5.iloc[3]["end_voltage_v"]) if pd.notna(d5.iloc[3].get("end_voltage_v")) else np.nan,
        "standby_V_end":  float(d5.iloc[4]["end_voltage_v"]) if pd.notna(d5.iloc[4].get("end_voltage_v")) else np.nan,
        "standby_t_end_h": float(d5.iloc[4]["step_time_h"]) if pd.notna(d5.iloc[4].get("step_time_h")) else np.nan,
    }
    return out

def match_labeled(original_file: str, labels_dir: str) -> Optional[str]:
    stem = re.sub(r'_labeled$', '', os.path.splitext(os.path.basename(original_file))[0])
    cand = os.path.join(labels_dir, f"{stem}_labeled.xlsx")
    if os.path.isfile(cand):
        return cand
    # fuzzy match
    for lf in glob.glob(os.path.join(labels_dir, "*_labeled.xlsx")):
        lstem = re.sub(r'_labeled$', '', os.path.splitext(os.path.basename(lf))[0])
        if _norm(lstem) == _norm(stem):
            return lf
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", required=True, help="Folder with original .xlsx files")
    ap.add_argument("--labels", required=True, help="Folder with *_labeled.xlsx files")
    ap.add_argument("-o", "--outdir", default="step_stats_v4", help="Output folder")
    ap.add_argument("--start-cycle", type=int, default=4, help="Start cycle index (skip formation cycles)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    orig_files = sorted(glob.glob(os.path.join(args.originals, "*.xlsx")))
    all_rows = []

    for of in orig_files:
        lab_path = match_labeled(of, args.labels)
        if not lab_path:
            print(f"[WARN] No labeled match for {os.path.basename(of)}; skipping")
            continue

        try:
            labels_df = load_cycle_labels(lab_path)
            xls = pd.ExcelFile(of)
            step_df = load_step_sheet(xls)

            # only cycles we have labels for
            for _, r in labels_df.iterrows():
                cycle = int(r["cycle"])
                if cycle < args.start_cycle:
                    continue

                vals = pick_1chg_5dchg(step_df, cycle)
                row = {
                    "file": os.path.basename(of),
                    "cycle": cycle,
                    "label": r["label"],
                    "pattern_ok": vals is not None
                }
                if vals is None:
                    # keep row for auditing (optional)
                    row.update({
                        "charge_V_end": np.nan,
                        "take_off_V_end": np.nan,
                        "hover_V_end": np.nan,
                        "cruise_V_end": np.nan,
                        "landing_V_end": np.nan,
                        "standby_V_end": np.nan,
                        "standby_t_end_h": np.nan,
                    })
                else:
                    row.update(vals)

                all_rows.append(row)

            print(f"[OK] {os.path.basename(of)}: cycles={len(labels_df)}")

        except Exception as e:
            print(f"[FAIL] {os.path.basename(of)}: {e}")

    if not all_rows:
        print("No rows collected.")
        return

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(args.outdir, "all_cycles_3class.csv")
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote {out_path} (rows={len(out)})")

if __name__ == "__main__":
    main()
