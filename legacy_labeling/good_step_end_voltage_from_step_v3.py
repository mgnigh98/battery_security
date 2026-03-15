#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
good_step_end_voltage_from_step_v3.py
(See header comments for details)
"""
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

def load_good_cycles_from_labeled(labeled_path: str) -> List[int]:
    xl = pd.ExcelFile(labeled_path)
    sheet = None
    for cand in ["cycle_labels", "labels"]:
        if cand in xl.sheet_names:
            sheet = cand; break
    if sheet is None: return []
    lab = pd.read_excel(xl, sheet_name=sheet)
    lab = remap_columns(lab, {"cycle":["Cycle","Cycle Index","cycle_index","cycle"],
                               "label":["Label","label"]})
    if "cycle" not in lab.columns or "label" not in lab.columns:
        return []
    good = lab[lab["label"].astype(str).str.upper()=="GOOD"]["cycle"]
    good = pd.to_numeric(good, errors="coerce").dropna().astype(int).tolist()
    return sorted(list(set(good)))

def load_step_sheet(xls: pd.ExcelFile) -> Optional[pd.DataFrame]:
    name = None
    for s in xls.sheet_names:
        if s.lower() == "step":
            name = s; break
    if name is None: return None
    df = pd.read_excel(xls, sheet_name=name)
    mapping = {
        "cycle_index":     ["Cycle Index","Cycle","cycle index"],
        "step_index":      ["Step Index","StepIndex","Idx"],
        "step_number":     ["Step Number","StepNumber","Step No.","StepNo"],
        "step_type":       ["Step Type","StepType","Type","Step"],
        "step_time_h":     ["Step Time(h)","Step Time (h)","Time(h)","Time (h)"],
        "onset":           ["Oneset Date","Onset Date","Start Date","Oneset"],
        "end":             ["End Date","EndDate","End time","End"],
        "start_voltage_v": ["Oneset Volt.(V)","Onset Voltage(V)","Start Voltage(V)","Oneset Volt (V)"],
        "end_voltage_v":   ["End Voltage(V)","End Voltage (V)","End Volt.(V)","EndVolt(V)"],
        "chg_cap_ah":      ["Chg. Cap.(Ah)","Charge Capacity(Ah)","Chg Cap (Ah)","Chg.Cap.(Ah)"],
        "dchg_cap_ah":     ["DChg. Cap.(Ah)","Discharge Capacity(Ah)","DChg Cap (Ah)","DChg.Cap.(Ah)"],
    }
    df = remap_columns(df, mapping)
    for c in ["cycle_index","step_index","step_number","step_time_h","start_voltage_v","end_voltage_v","chg_cap_ah","dchg_cap_ah"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["onset","end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def pick_steps_from_step_sheet(step_df: pd.DataFrame, cycle: int):
    cdf = step_df[step_df["cycle_index"] == cycle].copy()
    if cdf.empty: return None
    sort_cols = []
    if "onset" in cdf.columns: sort_cols.append("onset")
    if "end"   in cdf.columns: sort_cols.append("end")
    if "step_number" in cdf.columns: sort_cols.append("step_number")
    if sort_cols: cdf = cdf.sort_values(sort_cols, na_position="last")
    has_chg = "chg_cap_ah" in cdf.columns
    has_dch = "dchg_cap_ah" in cdf.columns
    if has_chg:
        chg_rows = cdf[cdf["chg_cap_ah"] > 0]
    else:
        txt = cdf.get("step_type", pd.Series([""]*len(cdf))).astype(str).str.lower()
        chg_rows = cdf[txt.str.contains("chg")]
    if chg_rows.empty: return None
    chg = chg_rows.iloc[0]
    chg_pos = chg.name
    after = cdf.loc[chg_pos+1:] if chg_pos in cdf.index else cdf.iloc[0:0]
    if has_dch:
        d_after = after[after["dchg_cap_ah"] > 0]
    else:
        txt2 = after.get("step_type", pd.Series([""]*len(after))).astype(str).str.lower()
        d_after = after[txt2.str.contains("dchg") | txt2.str.contains("discharge")]
    if len(d_after) < 5: return None
    d5 = d_after.iloc[:5]
    out = [{"name":"charge", **chg.to_dict()}]
    for nm, (_, r) in zip(["take_off","hover","cruise","landing","standby"], d5.iterrows()):
        x = r.to_dict(); x["name"] = nm; out.append(x)
    return out

def process_one(original_path: str, labeled_path: str, outdir: str):
    base = os.path.basename(original_path)
    try:
        good_cycles = load_good_cycles_from_labeled(labeled_path)
        if not good_cycles:
            print(f"[INFO] {base}: no GOOD cycles; skipping"); return None
        xls = pd.ExcelFile(original_path)
        step_df = load_step_sheet(xls)
        if step_df is None or step_df.empty:
            print(f"[WARN] {base}: missing/empty 'step' sheet; skipping"); return None
        rows = []
        for c in sorted(good_cycles):
            steps = pick_steps_from_step_sheet(step_df, c)
            if steps is None:
                print(f"[SKIP cycle {c} in {base}]: need 1 charge + 5 discharge after charge")
                continue
            row = {"file": base, "cycle": c}
            for st in steps:
                nm = st["name"]
                row[f"{nm}_V_end"] = float(st["end_voltage_v"]) if "end_voltage_v" in st and pd.notna(st["end_voltage_v"]) else np.nan
                if nm == "standby":
                    row["standby_t_end_h"] = float(st["step_time_h"]) if "step_time_h" in st and pd.notna(st["step_time_h"]) else np.nan
            rows.append(row)
        if not rows:
            print(f"[INFO] {base}: no GOOD cycles with complete 1+5 pattern"); return None
        df = pd.DataFrame(rows).sort_values(["file","cycle"])
        step_cols = [f"{nm}_V_end" for nm in ["charge","take_off","hover","cruise","landing","standby"]]
        summary = []
        for col in step_cols + ["standby_t_end_h"]:
            if col in df.columns:
                summary.append({"metric": col, "mean": float(np.nanmean(df[col])), "median": float(np.nanmedian(df[col]))})
        summ_df = pd.DataFrame(summary)
        os.makedirs(outdir, exist_ok=True)
        ofile = os.path.join(outdir, base.replace(".xlsx", "_good_steps_v3.xlsx"))
        try:
            with pd.ExcelWriter(ofile, engine="xlsxwriter") as w:
                df.to_excel(w, sheet_name="good_step_ends", index=False)
                summ_df.to_excel(w, sheet_name="summary_mean_median", index=False)
        except Exception:
            with pd.ExcelWriter(ofile, engine="openpyxl") as w:
                df.to_excel(w, sheet_name="good_step_ends", index=False)
                summ_df.to_excel(w, sheet_name="summary_mean_median", index=False)
        print(f"[OK] {base}: wrote {os.path.basename(ofile)} (rows={len(df)})")
        return df
    except Exception as e:
        print(f"[FAIL] {base}: {e}"); return None

def match_labeled(original_file: str, labels_dir: str) -> Optional[str]:
    stem = re.sub(r'_labeled$', '', os.path.splitext(os.path.basename(original_file))[0])
    cand = os.path.join(labels_dir, f"{stem}_labeled.xlsx")
    if os.path.isfile(cand): return cand
    for lf in glob.glob(os.path.join(labels_dir, "*_labeled.xlsx")):
        lstem = re.sub(r'_labeled$', '', os.path.splitext(os.path.basename(lf))[0])
        if _norm(lstem) == _norm(stem): return lf
    return None

def main():
    ap = argparse.ArgumentParser(description="Compute end-of-step voltages (GOOD cycles) using only STEP sheet.")
    ap.add_argument("--originals", required=True, help="Folder with original .xlsx files")
    ap.add_argument("--labels", required=True, help="Folder with *_labeled.xlsx files")
    ap.add_argument("-o","--outdir", default="step_stats_v3", help="Output folder")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.originals, "*.xlsx")))
    if not files: print(f"No .xlsx files found in {args.originals}"); return

    all_rows = []
    for of in files:
        lab = match_labeled(of, args.labels)
        if not lab:
            print(f"[WARN] No labeled match for {os.path.basename(of)}; skipping"); continue
        df = process_one(of, lab, args.outdir)
        if df is not None and not df.empty: all_rows.append(df)

    if not all_rows: print("No data collected."); return

    ALL = pd.concat(all_rows, ignore_index=True)
    os.makedirs(args.outdir, exist_ok=True)
    ALL.to_csv(os.path.join(args.outdir, "ALL_good_step_ends_v3.csv"), index=False)
    step_cols = [c for c in ALL.columns if c.endswith("_V_end")]
    add_cols = step_cols + (["standby_t_end_h"] if "standby_t_end_h" in ALL.columns else [])
    gsum = [{"metric": col, "mean": float(np.nanmean(ALL[col])), "median": float(np.nanmedian(ALL[col]))} for col in add_cols]
    pd.DataFrame(gsum).to_csv(os.path.join(args.outdir, "GLOBAL_mean_median_v3.csv"), index=False)
    print(f"[DONE] Wrote:\n - {os.path.join(args.outdir, 'ALL_good_step_ends_v3.csv')}\n - {os.path.join(args.outdir, 'GLOBAL_mean_median_v3.csv')}")

if __name__ == "__main__":
    main()
