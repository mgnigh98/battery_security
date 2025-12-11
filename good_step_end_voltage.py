#!/usr/bin/env python3
"""
good_step_end_voltage_from_originals.py

Use ORIGINAL Excel files (with all sheets) and LABELED files (to get GOOD cycles) to extract:
- End-of-step voltages for 1 charge + 5 discharge steps (take_off, hover, cruise, landing, standby)
- For standby, also report the end time (Time(h))
- Skip cycles that don't have exactly five discharge steps
- Produce per-file Excel + global CSV summaries

Usage:
  python good_step_end_voltage_from_originals.py \
      --originals data/ \
      --labels results/ \
      -o step_stats/

Outputs (in outdir):
- Per original file: <original_basename>_good_steps.xlsx with sheets:
    * good_step_ends
    * summary_mean_median
- Global:
    * ALL_good_step_ends.csv
    * GLOBAL_mean_median.csv
"""

from __future__ import annotations
import argparse, glob, os, re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
    module="openpyxl.styles.stylesheet"
)


# ---------- column normalization helpers ----------
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    inv = {}
    cur = { re.sub(r'[^a-z0-9]+', '', str(c).lower()): c for c in df.columns }
    for canon, candidates in mapping.items():
        for cand in candidates:
            key = re.sub(r'[^a-z0-9]+', '', str(cand).lower())
            if key in cur:
                inv[cur[key]] = canon
                break
    return df.rename(columns=inv)

# ---------- loaders ----------
def load_good_cycles_from_labeled(labeled_path: str) -> List[int]:
    xl = pd.ExcelFile(labeled_path)
    sheet = None
    for cand in ["cycle_labels", "labels"]:
        if cand in xl.sheet_names:
            sheet = cand
            break
    if sheet is None:
        raise RuntimeError(f"No cycle_labels/labels sheet in {os.path.basename(labeled_path)}")
    df = pd.read_excel(xl, sheet_name=sheet)
    # normalize columns
    df = _remap_columns(df, {"Cycle": ["Cycle","Cycle Index","cycle_index","cycle"], "Label": ["Label","label"]})
    if "Cycle" not in df.columns or "Label" not in df.columns:
        return []
    good = df[df["Label"].astype(str).str.upper() == "GOOD"]["Cycle"].dropna()
    try:
        return sorted(list(set(int(c) for c in good)))
    except Exception:
        return sorted(list(set(pd.to_numeric(good, errors="coerce").dropna().astype(int).tolist())))

def load_step_sheet(xls: pd.ExcelFile) -> pd.DataFrame:
    if "step" not in [s.lower() for s in xls.sheet_names]:
        raise RuntimeError("Missing required sheet 'step'.")
    name = [s for s in xls.sheet_names if s.lower() == "step"][0]
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
        # NEW: capacities to infer chg/dchg objectively
        "chg_cap_ah":       ["Chg. Cap.(Ah)","Charge Capacity(Ah)","Chg Cap (Ah)"],
        "dchg_cap_ah":      ["DChg. Cap.(Ah)","Discharge Capacity(Ah)","DChg Cap (Ah)"],
    }
    remap_columns = _remap_columns
    df = _remap_columns(df, mapping)
    for c in ["onset","end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # Ensure numeric
    for c in ["chg_cap_ah","dchg_cap_ah","step_number"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _classify_segment(seg_df: pd.DataFrame) -> str:
    """Return 'charge' or 'discharge' based on Step Type text (fallback to current sign if available)."""
    # Text majority vote
    if "step_type" in seg_df.columns:
        txt = seg_df["step_type"].astype(str).str.lower()
        # Prefer explicit labels if present
        chg_hits  = txt.str.contains("cc chg") | txt.str.contains("charge")
        dchg_hits = txt.str.contains("cc dchg") | txt.str.contains("discharge")
        if chg_hits.mean() > dchg_hits.mean():
            return "charge"
        if dchg_hits.mean() > chg_hits.mean():
            return "discharge"
    # Fallback: current sign (if available)
    if "current(a)" in (c.lower() for c in seg_df.columns):
        # find the actual column name
        cur_col = [c for c in seg_df.columns if c.lower()=="current(a)"][0]
        cur = seg_df[cur_col]
        # many cyclers log discharge as positive; if sign is unreliable, default to discharge after first charge
        # We'll just treat this as unknown and let ordering decide.
    return "unknown"

def segment_steps_by_time_reset(record_df: pd.DataFrame, cycle: int, eps: float = 1e-6) -> list[dict]:
    """
    Split a cycle's record rows into segments wherever Time(h) resets/drops.
    Returns list of dicts: { 'name': 'charge'/'discharge'/'unknown', 'start_idx', 'end_idx', 'end_voltage', 'end_time_h' }
    """
    rdf = record_df[record_df.get("cycle_index") == cycle].copy()
    if rdf.empty or "time_h" not in rdf.columns:
        return []

    rdf = rdf.sort_values("time_h").reset_index()  # keep original index if needed
    # When time resets, the natural order by ingest might not be monotonic; resort by Date then Time if available
    if "date" in rdf.columns and rdf["date"].notna().any():
        rdf = rdf.sort_values(["date", "time_h"])
    else:
        # Keep the within-cycle order stable: detect drops based on diff of time_h in original order if present
        rdf = rdf.sort_values("time_h")

    times = rdf["time_h"].to_numpy()
    # Identify starts: first row and any row where time decreases by >eps relative to previous row
    starts = [0]
    for i in range(1, len(times)):
        if times[i] + eps < times[i-1]:
            starts.append(i)
    # Build segments
    segs = []
    starts.append(len(rdf))  # sentinel
    for s, e in zip(starts[:-1], starts[1:]):
        seg = rdf.iloc[s:e]
        end_row = seg.iloc[-1]
        end_voltage = float(end_row.get("voltage_v")) if "voltage_v" in end_row else None
        end_time_h  = float(end_row.get("time_h"))    if "time_h" in end_row else None
        cls = _classify_segment(seg)
        segs.append({
            "class": cls,
            "start_iloc": s,
            "end_iloc": e-1,
            "end_voltage": end_voltage,
            "end_time_h": end_time_h
        })
    return segs

def pick_steps_from_record_segments(record_df: pd.DataFrame, cycle: int) -> list[dict] | None:
    """
    Use Time(h) reset segmentation to pick:
      - first 'charge' segment
      - next 5 segments as 'discharge' (accept 'unknown' but still treat as discharge by position)
    Returns ordered list with canonical names: charge, take_off, hover, cruise, landing, standby.
    """
    segs = segment_steps_by_time_reset(record_df, cycle)
    if not segs:
        return None

    # Find first explicit charge; if none, take the very first as charge.
    chg_idx = next((i for i, s in enumerate(segs) if s["class"] == "charge"), 0)
    # Pick subsequent 5 segments as discharge blocks
    d_list = []
    for s in segs[chg_idx+1:]:
        d_list.append(s)
        if len(d_list) == 5:
            break
    if len(d_list) < 5:
        return None

    ordered = []
    ordered.append({"name": "charge", **segs[chg_idx]})
    for nm, s in zip(["take_off","hover","cruise","landing","standby"], d_list):
        item = s.copy()
        item["name"] = nm
        ordered.append(item)
    return ordered

def load_record_sheet(orig_xls: pd.ExcelFile) -> pd.DataFrame:
    name = None
    for s in orig_xls.sheet_names:
        if s.lower() == "record":
            name = s
            break
    if name is None:
        raise RuntimeError("Missing required sheet 'record'.")
    df = pd.read_excel(orig_xls, sheet_name=name)
    mapping = {
        "cycle_index": ["Cycle Index","Cycle","cycle index"],
        "step_type": ["Step Type","StepType","Type","Step"],
        "time_h": ["Time(h)","Time (h)","Step Time(h)","Step Time (h)"],
        "date": ["Date","Timestamp","TimeStamp"],
        "voltage_v": ["Voltage(V)","Voltage (V)","Voltage"],
    }
    df = _remap_columns(df, mapping)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# ---------- step selection & extraction ----------
STEP_NAMES = ["charge","take_off","hover","cruise","landing","standby"]

def pick_steps_for_cycle(step_df: pd.DataFrame, cycle: int) -> Optional[List[dict]]:
    """
    Select, for a given cycle:
      - the earliest CHARGE step (chg_cap_ah > 0)
      - then the first FIVE DISCHARGE steps (dchg_cap_ah > 0) that occur after that charge step
    Order is by 'onset' if available, otherwise by 'step_number'.
    Skip cycles with fewer than 5 discharge steps.
    """
    cdf = step_df[step_df.get("cycle_index") == cycle].copy()
    if cdf.empty:
        return None

    # Sort so we have a temporal order
    if "onset" in cdf.columns and cdf["onset"].notna().any():
        cdf = cdf.sort_values(["onset","end","step_number"], na_position="last")
    elif "step_number" in cdf.columns:
        cdf = cdf.sort_values("step_number")

    # Identify charge/discharge by capacities (robust to missing/ambiguous Step Type text)
    chg_mask = (cdf.get("chg_cap_ah") > 0)
    dch_mask = (cdf.get("dchg_cap_ah") > 0)

    # Earliest charge step
    chg_steps = cdf[chg_mask] if chg_mask is not None else cdf.iloc[0:0]
    if chg_steps.empty:
        return None
    chg_row = chg_steps.iloc[0]  # earliest by sort order

    # All discharge steps after that charge step (by position)
    # Find index position of chosen charge step in cdf
    chg_pos = chg_row.name
    # Use position after that row; rely on mask + position to ensure chronological
    after_chg = cdf.loc[chg_pos:]
    dchg_after = after_chg[dch_mask.loc[after_chg.index]] if dch_mask is not None else after_chg.iloc[0:0]

    if len(dchg_after) < 5:
        return None  # per your instruction: do not consider cycles with <5 discharges

    d5 = dchg_after.iloc[:5]

    out = []
    out.append({"name": "charge", **chg_row.to_dict()})
    for nm, (_, r) in zip(["take_off","hover","cruise","landing","standby"], d5.iterrows()):
        d = r.to_dict()
        d["name"] = nm
        out.append(d)
    return out

def end_of_step_voltage(record_df: pd.DataFrame, cycle: int, onset: pd.Timestamp, end: pd.Timestamp) -> Tuple[Optional[float], Optional[float]]:
    rdf = record_df[record_df["cycle_index"] == cycle].copy()
    v_out, t_out = None, None

    if "date" in rdf.columns and pd.notna(onset) and pd.notna(end):
        window = rdf[(rdf["date"] >= onset) & (rdf["date"] <= end)]
        if not window.empty:
            last = window.sort_values(["date","time_h"]).iloc[-1]
            v_out = float(last.get("voltage_v")) if "voltage_v" in last else None
            t_out = float(last.get("time_h")) if "time_h" in last and pd.notna(last.get("time_h")) else None

    if v_out is None:
        # Fallback: last row by time within the cycle
        if "time_h" in rdf.columns and not rdf.empty:
            last = rdf.sort_values("time_h").iloc[-1]
            v_out = float(last.get("voltage_v")) if "voltage_v" in last else None
            t_out = float(last.get("time_h")) if pd.notna(last.get("time_h")) else None

    return v_out, t_out

# ---------- main processing ----------
def process_one(original_path: str, labeled_path: str, outdir: str, use_record_seg: bool = False) -> Optional[pd.DataFrame]:
    base = os.path.basename(original_path)
    try:
        good_cycles = load_good_cycles_from_labeled(labeled_path)
        if not good_cycles:
            print(f"[INFO] {base}: no GOOD cycles found in labels; skipping")
            return None

        xls = pd.ExcelFile(original_path)
        step_df = load_step_sheet(xls)
        record_df = load_record_sheet(xls)

        rows = []
        for c in sorted(good_cycles):
            steps = None
            if use_record_seg:
                steps = pick_steps_from_record_segments(record_df, c)
                if steps is None:
                    steps = pick_steps_for_cycle(step_df, c)
            else:
                steps = pick_steps_for_cycle(step_df, c)
                if steps is None:
                    steps = pick_steps_from_record_segments(record_df, c)

            if steps is None:
                # Count derived discharge-like segments from record
                segs = segment_steps_by_time_reset(record_df, c)
                print(f"[SKIP cycle {c}] Could not get 1+5 after charge. record_segments={len(segs)}")
                continue

            # When steps came from record segmentation, they already have end_voltage/time
            # When from step-sheet, we still need to query end-of-step voltage inside the window.
            row = {"file": base, "cycle": c}
            if "end_voltage" in steps[0]:
                # record-based: use provided end-of-step values
                for st in steps:
                    nm = st["name"]
                    row[f"{nm}_V_end"] = st.get("end_voltage")
                    if nm == "standby":
                        row["standby_t_end_h"] = st.get("end_time_h")
            else:
                # step-sheet path: fetch from record windows
                for st in steps:
                    nm = st.get("name")
                    onset = st.get("onset")
                    end = st.get("end")
                    v_end, t_end = end_of_step_voltage(record_df, c, onset, end)
                    row[f"{nm}_V_end"] = v_end
                    if nm == "standby":
                        row["standby_t_end_h"] = t_end

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
                summary.append({"metric": col, "mean": float(np.nanmean(df[col])), "median": float(np.nanmedian(df[col]))})
        summ_df = pd.DataFrame(summary)

        # write per-file excel
        os.makedirs(outdir, exist_ok=True)
        ofile = os.path.join(outdir, base.replace(".xlsx","_good_steps.xlsx"))
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

def match_labeled(original_file: str, labels_dir: str) -> Optional[str]:
    """
    Find the corresponding *_labeled.xlsx in labels_dir for the given original file.
    Matching strategy: same basename without any trailing '_labeled'.
    """
    base = os.path.basename(original_file)
    stem = re.sub(r'_labeled$', '', os.path.splitext(base)[0])
    # try exact match with _labeled suffix
    cand1 = os.path.join(labels_dir, f"{stem}_labeled.xlsx")
    if os.path.isfile(cand1):
        return cand1
    # fallback: any file that starts with the stem and ends with _labeled.xlsx
    patt = os.path.join(labels_dir, f"{stem}*_*labeled.xlsx")
    matches = glob.glob(patt)
    if matches:
        return matches[0]
    # last fallback: search any *_labeled.xlsx and compare normalized stems
    lab_files = glob.glob(os.path.join(labels_dir, "*_labeled.xlsx"))
    nstem = _norm(stem)
    for lf in lab_files:
        lf_stem = _norm(re.sub(r'_labeled$', '', os.path.splitext(os.path.basename(lf))[0]))
        if lf_stem == nstem:
            return lf
    return None

def main():
    ap = argparse.ArgumentParser(description="Extract end-of-step voltages from ORIGINAL files using GOOD cycles from labeled files.")
    ap.add_argument("--originals", required=True, help="Folder with original .xlsx files (with all sheets)")
    ap.add_argument("--labels", required=True, help="Folder with *_labeled.xlsx files containing cycle_labels")
    ap.add_argument("-o","--outdir", default="step_stats", help="Output folder")
    ap.add_argument("--use-record-seg", action="store_true",
                    help="Segment steps by Time(h) resets in the record sheet (preferred when step naming is unreliable).")


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
        df = process_one(of, lab, args.outdir, use_record_seg=args.use_record_seg)
        if df is not None and not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No data collected.")
        return

    allc = pd.concat(all_rows, ignore_index=True)
    all_csv = os.path.join(args.outdir, "ALL_good_step_ends.csv")
    allc.to_csv(all_csv, index=False)

    # Global summary (mean/median per step over ALL files)
    step_cols = [c for c in allc.columns if c.endswith("_V_end")] + (["standby_t_end_h"] if "standby_t_end_h" in allc.columns else [])
    gsum = []
    for col in step_cols:
        gsum.append({"metric": col, "mean": float(np.nanmean(allc[col])), "median": float(np.nanmedian(allc[col]))})
    pd.DataFrame(gsum).to_csv(os.path.join(args.outdir, "GLOBAL_mean_median.csv"), index=False)

    print(f"[DONE] Wrote:\n - {all_csv}\n - {os.path.join(args.outdir, 'GLOBAL_mean_median.csv')}")

if __name__ == "__main__":
    main()
