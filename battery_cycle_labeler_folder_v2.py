#!/usr/bin/env python3
import os, sys, json, warnings
import pandas as pd
import numpy as np

# silence only the specific benign warning from openpyxl
warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning)

# ---------- CONFIG DEFAULTS ----------
CFG = {
    "LINEARITY_THRESHOLD": 0.93,
    "MAD_K": 5,
    "SKIP_LAST_CYCLE": True,   # can override by --cfg '{"SKIP_LAST_CYCLE": false}'
}

# ---------- HELPERS ----------
def robust_mad(arr):
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return mad, med

def fit_linearity(time, volt):
    # restrict to middle 60% of CC-charge segment to avoid saturation
    n = len(time)
    lo, hi = int(0.2*n), int(0.8*n)
    if hi-lo < 5: return 1.0
    t, v = time[lo:hi], volt[lo:hi]
    A = np.vstack([t, np.ones(len(t))]).T
    m, c = np.linalg.lstsq(A, v, rcond=None)[0]
    v_pred = m*t + c
    ss_res = np.sum((v-v_pred)**2)
    ss_tot = np.sum((v-np.mean(v))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 1
    return r2

def label_cycles(record_df, cycle_df, step_df):
    cycle_labels = []
    out_rows = []
    unique_cycles = sorted(record_df['Cycle Index'].dropna().unique())
    if 1 in unique_cycles: 
        start_cycle = 4
    else:
        start_cycle = unique_cycles[0]
    if CFG.get("SKIP_LAST_CYCLE", True):
        unique_cycles = unique_cycles[:-1]
    for cyc in unique_cycles:
        cyc_records = record_df[record_df['Cycle Index']==cyc]
        if cyc_records.empty: continue
        chg_mask = cyc_records['Step Type'].str.contains("Chg", na=False) & (cyc_records['Current(A)']>0)
        dchg_mask = cyc_records['Step Type'].str.contains("DChg", na=False) & (cyc_records['Current(A)']<0)
        ce = None
        if cyc in cycle_df['Cycle Index'].values:
            row = cycle_df[cycle_df['Cycle Index']==cyc].iloc[0]
            chg_cap, dchg_cap = row['Chg. Cap.(Ah)'], row['DChg. Cap.(Ah)']
            ce = dchg_cap/chg_cap if chg_cap>0 else None
        linearity = None
        if chg_mask.sum()>10:
            tvals = cyc_records.loc[chg_mask, 'Time(h)'].values
            vvals = cyc_records.loc[chg_mask, 'Voltage(V)'].values
            linearity = fit_linearity(tvals, vvals)
        bad_reasons = []
        if ce is not None:
            if abs(ce-1) > 0.08:
                bad_reasons.append("CE_ABS")
        if linearity is not None:
            if linearity < CFG['LINEARITY_THRESHOLD']:
                bad_reasons.append("LOW_LINEARITY")
        label = "GOOD" if len(bad_reasons)==0 else "BAD"
        cycle_labels.append((cyc,label,bad_reasons,ce,linearity))
        out_rows.append({
            "Cycle": cyc, "Label": label,
            "Reasons": ",".join(bad_reasons),
            "CE": ce, "Linearity": linearity
        })
    return pd.DataFrame(out_rows)

def label_file(path, output_xlsx=None):
    xls = pd.ExcelFile(path)
    if 'record' not in xls.sheet_names:
        raise ValueError("missing record sheet")
    record_df = pd.read_excel(xls, sheet_name='record')
    cycle_df = pd.read_excel(xls, sheet_name='Cycle') if 'Cycle' in xls.sheet_names else pd.DataFrame()
    step_df = pd.read_excel(xls, sheet_name='step') if 'step' in xls.sheet_names else pd.DataFrame()
    labels = label_cycles(record_df, cycle_df, step_df)
    if output_xlsx:
        try:
            with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as w:
                record_df.to_excel(w, sheet_name="record", index=False)
                cycle_df.to_excel(w, sheet_name="Cycle", index=False)
                step_df.to_excel(w, sheet_name="step", index=False)
                labels.to_excel(w, sheet_name="labels", index=False)
        except Exception:
            with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
                record_df.to_excel(w, sheet_name="record", index=False)
                cycle_df.to_excel(w, sheet_name="Cycle", index=False)
                step_df.to_excel(w, sheet_name="step", index=False)
                labels.to_excel(w, sheet_name="labels", index=False)
    return labels

def process_folder(folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary = []
    for fn in os.listdir(folder):
        if not fn.endswith(".xlsx"): continue
        inpath = os.path.join(folder, fn)
        try:
            labels = label_file(inpath, output_xlsx=os.path.join(output_dir, fn.replace(".xlsx","_labeled.xlsx")))
            good = (labels['Label']=="GOOD").sum()
            bad = (labels['Label']=="BAD").sum()
            summary.append((fn,len(labels),good,bad))
            print(f"[OK] {fn}: cycles={len(labels)} GOOD={good} BAD={bad}")
        except Exception as e:
            print(f"[FAIL] {fn}: {e}")
    pd.DataFrame(summary, columns=["File","nCycles","GOOD","BAD"]).to_csv(os.path.join(output_dir,"batch_summary.csv"), index=False)

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Excel file or folder")
    ap.add_argument("-o","--output", default="results", help="Output folder")
    ap.add_argument("--cfg", help="JSON config overrides")
    args = ap.parse_args()
    if args.cfg:
        CFG.update(json.loads(args.cfg))
    if os.path.isdir(args.input):
        process_folder(args.input, args.output)
    else:
        labels = label_file(args.input, output_xlsx=os.path.join(args.output, os.path.basename(args.input).replace(".xlsx","_labeled.xlsx")))
        print(labels.head())
