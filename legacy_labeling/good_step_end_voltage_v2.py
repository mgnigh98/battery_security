#!/usr/bin/env python3
"""
good_step_end_voltage_v2.py

Goal
----
From ORIGINAL Excel workbooks (with sheets: unit, test, cycle, step, record, log, idle, curve),
extract end-of-step voltages/times for cycles labeled GOOD (labels come from *_labeled.xlsx).
Each cycle must have exactly: 1 charge step + 5 discharge steps (take_off, hover, cruise, landing, standby).
Discharge steps are not named in sheets; we detect them by segmentation using Time(h) resets in the RECORD sheet.

Fixes vs v1
-----------
- Use robust RECORD-based segmentation: split when Time(h) drops/resets to ~0.
- At the end of each discharge step, some files have TWO consecutive rows with Time(h)==0.
  We choose the FIRST of the duplicate-zeros as the segment end (ignore the second).
- Prevents incorrect 4.3 V end-of-step values for discharge segments.
- Falls back to STEP sheet windows if record segmentation fails, but prefers segmentation.
- Works only on GOOD cycles (from labeled files).

Outputs
-------
For each original workbook -> <basename>_good_steps_v2.xlsx with sheets:
  - good_step_ends: per GOOD cycle values (charge/take_off/hover/cruise/landing/standby end voltages; standby end time)
  - summary_mean_median: mean/median across GOOD cycles in that file

Global (across all processed files) in outdir:
  - ALL_good_step_ends_v2.csv
  - GLOBAL_mean_median_v2.csv

Usage
-----
python good_step_end_voltage_v2.py \
  --originals data/ \
  --labels results/ \
  -o step_stats_v2/ \
  [--prefer-record]    # default True; record segmentation first then fallback to step windows

"""

from __future__ import annotations
import argparse, glob, os, re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings

# Silence the common openpyxl default-style warning
warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
    module="openpyxl.styles.stylesheet",
)

# ------------------------- utilities -------------------------
_NORM = lambda s: re.sub(r'[^a-z0-9]+', '', str(s).lower())

def remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    cur = { _NORM(c): c for c in df.columns }
    ren = {}
    for canon, cand_list in mapping.items():
        for cand in cand_list:
            key = _NORM(cand)
            if key in cur:
                ren[cur[key]] = canon
                break
    return df.rename(columns=ren)

# ------------------------- loaders -------------------------
def load_good_cycles_from_labeled(labeled_path: str) -> List[int]:
    xl = pd.ExcelFile(labeled_path)
    sheet = None
    for cand in ["cycle_labels", "labels"]:
        if cand in xl.sheet_names:
            sheet = cand
            break
    if sheet is None:
        return []
    lab = pd.read_excel(xl, sheet_name=sheet)
    lab = remap_columns(lab, {"Cycle": ["Cycle","Cycle Index","cycle_index","cycle"],
                               "Label": ["Label","label"]})
    if "Cycle" not in lab.columns or "Label" not in lab.columns:
        return []
    good = lab[lab["Label"].astype(str).str.upper()=="GOOD"]["Cycle"]
    good = pd.to_numeric(good, errors="coerce").dropna().astype(int).tolist()
    return sorted(list(set(good)))

def load_step_sheet(xls: pd.ExcelFile) -> Optional[pd.DataFrame]:
    name = None
    for s in xls.sheet_names:
        if s.lower() == "step":
            name = s; break
    if name is None:
        return None
    df = pd.read_excel(xls, sheet_name=name)
    mapping = {
        "cycle_index":      ["Cycle Index","Cycle","cycle index"],
        "step_type":        ["Step Type","StepType","Type","Step"],
        "step_number":      ["Step Number","StepNumber","Step Index","StepIdx"],
        "onset":            ["Oneset Date","Onset Date","Start Date","Oneset"],
        "end":              ["End Date","EndDate","End time","End"],
        "start_voltage_v":  ["Oneset Volt.(V)","Start Voltage(V)","Onset Voltage(V)"],
        "end_voltage_v":    ["End Voltage(V)","End Voltage (V)","End Volt.(V)","EndVolt(V)"],
        "step_time_h":      ["Step Time(h)","Step Time (h)","Time(h)"],
        "chg_cap_ah":       ["Chg. Cap.(Ah)","Charge Capacity(Ah)","Chg Cap (Ah)"],
        "dchg_cap_ah":      ["DChg. Cap.(Ah)","Discharge Capacity(Ah)","DChg Cap (Ah)"],
    }
    df = remap_columns(df, mapping)
    for c in ["onset","end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["chg_cap_ah","dchg_cap_ah","step_number"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_record_sheet(xls: pd.ExcelFile) -> pd.DataFrame:
    name = None
    for s in xls.sheet_names:
        if s.lower() == "record":
            name = s; break
    if name is None:
        raise RuntimeError("Missing required sheet 'record'.")
    df = pd.read_excel(xls, sheet_name=name)
    mapping = {
        "cycle_index": ["Cycle Index","Cycle","cycle index"],
        "step_type":   ["Step Type","StepType","Type","Step"],
        "time_h":      ["Time(h)","Time (h)","Step Time(h)","Step Time (h)"],
        "date":        ["Date","Timestamp","TimeStamp"],
        "voltage_v":   ["Voltage(V)","Voltage (V)","Voltage"],
        "current_a":   ["Current(A)","Current (A)","Current"],
    }
    df = remap_columns(df, mapping)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # force numeric time/voltage
    for c in ["time_h","voltage_v","current_a","cycle_index"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------- segmentation on RECORD -------------------------
def segment_by_time_reset(rdf: pd.DataFrame, eps: float=1e-9):
    """
    Return (segments, rdf_sorted)
    segments: list of (start_iloc, end_iloc), where a new segment starts when Time(h) decreases.
    If a segment ends with consecutive Time(h)==0 rows, choose the FIRST zero as the end index.
    """
    if "time_h" not in rdf.columns or rdf.empty:
        return [], rdf

    # Sort stable: prefer Date then Time; else by Time keeping arrival order stable.
    if "date" in rdf.columns and rdf["date"].notna().any():
        rdf = rdf.sort_values(["date","time_h","voltage_v"], kind="mergesort").reset_index(drop=True)
    else:
        rdf = rdf.reset_index(drop=True).sort_values(["time_h","voltage_v"], kind="mergesort").reset_index(drop=True)

    t = rdf["time_h"].to_numpy()
    starts = [0]
    for i in range(1, len(t)):
        if (t[i] + eps) < (t[i-1] - eps):
            starts.append(i)
    starts.append(len(rdf))

    segs = []
    for s, e in zip(starts[:-1], starts[1:]):
        end_iloc = e - 1
        # Handle duplicate zero at end
        while end_iloc > s and (rdf.at[end_iloc, "time_h"] == 0) and (rdf.at[end_iloc-1, "time_h"] == 0):
            end_iloc -= 1
        segs.append((s, end_iloc))
    return segs, rdf

def classify_segment(rdf: pd.DataFrame, s: int, e: int, top_v_threshold: float=4.25) -> str:
    seg = rdf.iloc[s:e+1]
    if seg.empty: return "unknown"
    if "step_type" in seg.columns:
        txt = seg["step_type"].astype(str).str.lower()
        chg = (txt.str.contains("cc chg") | txt.str.contains("charge")).mean()
        dch = (txt.str.contains("cc dchg") | txt.str.contains("discharge")).mean()
        if chg > dch: return "charge"
        if dch > chg: return "discharge"
    # slope heuristic
    v = seg["voltage_v"].to_numpy(dtype=float)
    if len(v) >= 3:
        v_start = np.nanmedian(v[:min(3,len(v))])
        v_end   = np.nanmedian(v[-min(3,len(v)) : ])
        if (v_end - v_start) > 0.02 and (np.nanmax(v) >= top_v_threshold):
            return "charge"
    return "discharge"

def pick_steps_from_record(record_df: pd.DataFrame, cycle: int) -> Optional[List[dict]]:
    rdf = record_df[record_df["cycle_index"] == cycle].copy()
    if rdf.empty:
        return None
    (segments, rdf_sorted) = segment_by_time_reset(rdf)
    if not segments:
        return None

    labels = []
    for (s,e) in segments:
        cls = classify_segment(rdf_sorted, s, e)
        end_voltage = float(rdf_sorted.iloc[e]["voltage_v"]) if "voltage_v" in rdf_sorted.columns else None
        end_time    = float(rdf_sorted.iloc[e]["time_h"]) if "time_h" in rdf_sorted.columns else None
        labels.append({"class": cls, "s": s, "e": e, "end_voltage": end_voltage, "end_time_h": end_time})

    chg_idx = next((i for i,x in enumerate(labels) if x["class"]=="charge"), None)
    if chg_idx is None:
        end_volts = np.array([x["end_voltage"] if x["end_voltage"] is not None else -np.inf for x in labels], dtype=float)
        if np.isfinite(end_volts).any():
            chg_idx = int(np.nanargmax(end_volts))
            if end_volts[chg_idx] < 4.25:
                chg_idx = 0
        else:
            chg_idx = 0

    d_list = labels[chg_idx+1: chg_idx+6]
    if len(d_list) < 5:
        return None

    ordered = [{"name":"charge", **labels[chg_idx]}]
    for nm, seg in zip(["take_off","hover","cruise","landing","standby"], d_list):
        out = seg.copy(); out["name"] = nm
        ordered.append(out)
    return ordered

# Fallback to step-sheet if segmentation fails
def pick_steps_from_step_sheet(step_df: Optional[pd.DataFrame], record_df: pd.DataFrame, cycle: int) -> Optional[List[dict]]:
    if step_df is None or step_df.empty:
        return None
    cdf = step_df[step_df["cycle_index"] == cycle].copy()
    if cdf.empty:
        return None
    if "onset" in cdf.columns and cdf["onset"].notna().any():
        cdf = cdf.sort_values(["onset","end","step_number"], na_position="last")
    elif "step_number" in cdf.columns:
        cdf = cdf.sort_values("step_number")

    chg_mask = (cdf.get("chg_cap_ah") > 0) if "chg_cap_ah" in cdf.columns else None
    dch_mask = (cdf.get("dchg_cap_ah") > 0) if "dchg_cap_ah" in cdf.columns else None
    chg_steps = cdf[chg_mask] if chg_mask is not None else cdf.iloc[0:0]
    if chg_steps.empty:
        return None
    chg_row = chg_steps.iloc[0]

    chg_pos = chg_row.name
    after_chg = cdf.loc[chg_pos:]
    dchg_after = after_chg[dch_mask.loc[after_chg.index]] if dch_mask is not None else after_chg.iloc[0:0]
    if len(dchg_after) < 5:
        return None
    d5 = dchg_after.iloc[:5]

    out = [{"name":"charge", **chg_row.to_dict()}]
    for nm, (_, r) in zip(["take_off","hover","cruise","landing","standby"], d5.iterrows()):
        x = r.to_dict(); x["name"] = nm
        out.append(x)
    return out

# ------------------------- per-file processing -------------------------
def process_one(original_path: str, labeled_path: str, outdir: str, prefer_record: bool=True) -> Optional[pd.DataFrame]:
    base = os.path.basename(original_path)
    try:
        good_cycles = load_good_cycles_from_labeled(labeled_path)
        if not good_cycles:
            print(f"[INFO] {base}: no GOOD cycles in labels; skipping")
            return None

        xls = pd.ExcelFile(original_path)
        step_df = load_step_sheet(xls)  # may be None
        record_df = load_record_sheet(xls)

        rows = []
        for c in sorted(good_cycles):
            if prefer_record:
                steps = pick_steps_from_record(record_df, c)
                if steps is None:
                    steps = pick_steps_from_step_sheet(step_df, record_df, c)
            else:
                steps = pick_steps_from_step_sheet(step_df, record_df, c)
                if steps is None:
                    steps = pick_steps_from_record(record_df, c)

            if steps is None:
                try:
                    segs, _ = segment_by_time_reset(record_df[record_df["cycle_index"]==c].copy())
                    nseg = len(segs)
                except Exception:
                    nseg = -1
                print(f"[SKIP cycle {c} in {base}] need 1 charge + 5 discharge; record_segments={nseg}")
                continue

            row = {"file": base, "cycle": c}
            if "end_voltage" in steps[0]:  # record-based
                for st in steps:
                    nm = st["name"]
                    row[f"{nm}_V_end"] = st.get("end_voltage")
                    if nm == "standby":
                        row["standby_t_end_h"] = st.get("end_time_h")
            else:
                # step-sheet path: window on Date and pick last row (first-of-duplicate-zeros rule)
                for st in steps:
                    nm = st.get("name")
                    onset = st.get("onset"); end = st.get("end")
                    rdf = record_df[(record_df["cycle_index"]==c)].copy()
                    if onset is not None and end is not None and "date" in rdf.columns and rdf["date"].notna().any():
                        win = rdf[(rdf["date"]>=onset) & (rdf["date"]<=end)].copy()
                        if not win.empty:
                            win = win.sort_values(["date","time_h"], kind="mergesort")
                            # adjust for duplicate zeros
                            if len(win) >= 2 and (win.iloc[-1]["time_h"]==0) and (win.iloc[-2]["time_h"]==0):
                                last_row = win.iloc[-2]
                            else:
                                last_row = win.iloc[-1]
                            row[f"{nm}_V_end"] = float(last_row.get("voltage_v")) if "voltage_v" in last_row else None
                            if nm == "standby":
                                row["standby_t_end_h"] = float(last_row.get("time_h")) if "time_h" in last_row else None
                            continue
                    # fallback: use segmented values
                    seg_pick = pick_steps_from_record(record_df, c)
                    if seg_pick is not None:
                        for st2 in seg_pick:
                            if st2["name"] == nm:
                                row[f"{nm}_V_end"] = st2.get("end_voltage")
                                if nm == "standby":
                                    row["standby_t_end_h"] = st2.get("end_time_h")
                                break

            rows.append(row)

        if not rows:
            print(f"[INFO] {base}: no GOOD cycles with 1+5 steps; nothing to write")
            return None

        df = pd.DataFrame(rows).sort_values(["file","cycle"])

        # per-file summary
        step_cols = [f"{nm}_V_end" for nm in ["charge","take_off","hover","cruise","landing","standby"]]
        summary = []
        for col in step_cols + ["standby_t_end_h"]:
            if col in df.columns:
                summary.append({"metric": col,
                                "mean": float(np.nanmean(df[col])),
                                "median": float(np.nanmedian(df[col]))})
        summ_df = pd.DataFrame(summary)

        # write per-file excel
        os.makedirs(outdir, exist_ok=True)
        ofile = os.path.join(outdir, base.replace(".xlsx","_good_steps_v2.xlsx"))
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
        print(f"[SKIP] {base}: {e}")
        return None

# ------------------------- matching labeled vs original -------------------------
def match_labeled(original_file: str, labels_dir: str) -> Optional[str]:
    stem = re.sub(r'_labeled$', '', os.path.splitext(os.path.basename(original_file))[0])
    cand1 = os.path.join(labels_dir, f"{stem}_labeled.xlsx")
    if os.path.isfile(cand1):
        return cand1
    for lf in glob.glob(os.path.join(labels_dir, "*_labeled.xlsx")):
        lstem = re.sub(r'_labeled$', '', os.path.splitext(os.path.basename(lf))[0])
        if _NORM(lstem) == _NORM(stem):
            return lf
    return None

# ------------------------- CLI -------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract end-of-step voltages from ORIGINAL files using GOOD cycles from labeled files (record-based segmentation).")
    ap.add_argument("--originals", required=True, help="Folder with original .xlsx files (with all sheets)")
    ap.add_argument("--labels", required=True, help="Folder with *_labeled.xlsx files containing cycle_labels (GOOD cycles)")
    ap.add_argument("-o","--outdir", default="step_stats_v2", help="Output folder")
    ap.add_argument("--prefer-record", action="store_true", default=True,
                    help="Prefer record-based segmentation first (default True).")
    args = ap.parse_args()

    orig_files = sorted(glob.glob(os.path.join(args.originals, "*.xlsx")))
    if not orig_files:
        print(f"No .xlsx files in {args.originals}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    all_rows = []
    for of in orig_files:
        lab = match_labeled(of, args.labels)
        if not lab:
            print(f"[WARN] No labeled match found for {os.path.basename(of)}; skipping")
            continue
        df = process_one(of, lab, args.outdir, prefer_record=args.prefer_record)
        if df is not None and not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No data collected.")
        return

    allc = pd.concat(all_rows, ignore_index=True)
    all_csv = os.path.join(args.outdir, "ALL_good_step_ends_v2.csv")
    allc.to_csv(all_csv, index=False)

    # Global summary
    step_cols = [c for c in allc.columns if c.endswith("_V_end")] + (["standby_t_end_h"] if "standby_t_end_h" in allc.columns else [])
    gsum = [{"metric": col,
             "mean": float(np.nanmean(allc[col])),
             "median": float(np.nanmedian(allc[col]))} for col in step_cols]
    pd.DataFrame(gsum).to_csv(os.path.join(args.outdir, "GLOBAL_mean_median_v2.csv"), index=False)

    print(f"[DONE] Wrote:\n - {all_csv}\n - {os.path.join(args.outdir, 'GLOBAL_mean_median_v2.csv')}")

if __name__ == "__main__":
    main()
