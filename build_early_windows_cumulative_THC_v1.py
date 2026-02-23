#!/usr/bin/env python3
"""
build_early_windows_cumulative_THC_v2.py

Same as v1 but fixes file-label matching by using a canonical join_key.
This handles datasets where actual xlsx files are named *_good.xlsx or *_bad.xlsx,
while labels CSV may use the base *.xlsx name.

Outputs early_{t}s.csv for each window.
"""

from __future__ import annotations
import os, re, glob, argparse
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

def canonical_stem(name: str) -> str:
    s = os.path.basename(str(name)).strip()
    s = os.path.splitext(s)[0]
    s = s.strip()
    # strip trailing tags
    s = re.sub(r"\s*_good_labeled_3class\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*_labeled\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*_good\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*_bad\s*$", "", s, flags=re.I)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def remap_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    cur = {_norm(c): c for c in df.columns}
    ren = {}
    for canon, cands in mapping.items():
        for cand in cands:
            k = _norm(cand)
            if k in cur:
                ren[cur[k]] = canon
                break
    return df.rename(columns=ren)

def is_charge_step(st: str) -> bool:
    st = str(st)
    return st.startswith("CC Chg") or ("CC Chg" in st) or ("chg" in st.lower() and "dchg" not in st.lower())

def is_discharge_step(st: str) -> bool:
    st = str(st).lower()
    return ("dchg" in st) or ("discharge" in st)

def compute_dvdt(group: pd.DataFrame) -> pd.Series:
    t = group["t_cum_s"].to_numpy(dtype=float)
    v = group["voltage_v"].to_numpy(dtype=float)
    out = np.full(len(group), np.nan, dtype=float)
    if len(group) >= 2:
        dt = np.diff(t)
        dv = np.diff(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = dv / dt
    return pd.Series(out, index=group.index)

def load_record(xls: pd.ExcelFile) -> pd.DataFrame:
    sheets = {s.lower(): s for s in xls.sheet_names}
    if "record" not in sheets:
        raise ValueError("Missing 'record' sheet")
    df = pd.read_excel(xls, sheet_name=sheets["record"])
    mapping = {
        "cycle_index": ["Cycle Index","CycleIndex","Cycle"],
        "step_type":   ["Step Type","StepType","Type","Step"],
        "current_a":   ["Current(A)","Current (A)","Current"],
        "voltage_v":   ["Voltage(V)","Voltage (V)","Voltage"],
        "power_w":     ["Power(W)","Power (W)"],
        "date":        ["Date"]
    }
    df = remap_columns(df, mapping)
    need = ["cycle_index","step_type","current_a","voltage_v","date"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"record missing columns after remap: {miss}")
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["current_a"] = pd.to_numeric(df["current_a"], errors="coerce")
    df["voltage_v"] = pd.to_numeric(df["voltage_v"], errors="coerce")
    if "power_w" in df.columns:
        df["power_w"] = pd.to_numeric(df["power_w"], errors="coerce")
    else:
        df["power_w"] = df["voltage_v"] * df["current_a"]
    return df.dropna(subset=["cycle_index","date"]).copy()

def load_step(xls: pd.ExcelFile) -> pd.DataFrame:
    sheets = {s.lower(): s for s in xls.sheet_names}
    if "step" not in sheets:
        raise ValueError("Missing 'step' sheet")
    df = pd.read_excel(xls, sheet_name=sheets["step"])
    mapping = {
        "cycle_index":  ["Cycle Index","CycleIndex","Cycle"],
        "step_type":    ["Step Type","StepType","Type","Step"],
        "onset_dt":     ["Oneset Date","Onset Date","Start Date","Oneset"],
        "end_dt":       ["End Date","EndDate","End time","End"],
        "step_number":  ["Step Number","StepNumber","Step No.","StepNo"],
        "chg_cap_ah":   ["Chg. Cap.(Ah)","Charge Capacity(Ah)","Chg Cap (Ah)","Chg.Cap.(Ah)"],
        "dchg_cap_ah":  ["DChg. Cap.(Ah)","Discharge Capacity(Ah)","DChg Cap (Ah)","DChg.Cap.(Ah)"],
    }
    df = remap_columns(df, mapping)
    need = ["cycle_index","step_type","onset_dt","end_dt"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"step missing columns after remap: {miss}")
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
    df["onset_dt"] = pd.to_datetime(df["onset_dt"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_dt"], errors="coerce")
    if "step_number" in df.columns:
        df["step_number"] = pd.to_numeric(df["step_number"], errors="coerce")
    if "chg_cap_ah" in df.columns:
        df["chg_cap_ah"] = pd.to_numeric(df["chg_cap_ah"], errors="coerce")
    if "dchg_cap_ah" in df.columns:
        df["dchg_cap_ah"] = pd.to_numeric(df["dchg_cap_ah"], errors="coerce")
    return df.dropna(subset=["cycle_index","onset_dt","end_dt"]).copy()

def pick_charge_then_5_discharge(step_cycle: pd.DataFrame) -> Optional[pd.DataFrame]:
    cdf = step_cycle.copy()
    sort_cols = [c for c in ["onset_dt","end_dt","step_number"] if c in cdf.columns]
    if sort_cols:
        cdf = cdf.sort_values(sort_cols, na_position="last")

    if "chg_cap_ah" in cdf.columns:
        chg_rows = cdf[cdf["chg_cap_ah"].fillna(0) > 0]
    else:
        chg_rows = cdf[cdf["step_type"].apply(is_charge_step)]
    if chg_rows.empty:
        return None
    chg = chg_rows.iloc[0]
    chg_pos = chg.name
    after = cdf.loc[chg_pos+1:] if chg_pos in cdf.index else cdf.iloc[0:0]

    if "dchg_cap_ah" in after.columns:
        d_after = after[after["dchg_cap_ah"].fillna(0) > 0]
    else:
        d_after = after[after["step_type"].apply(is_discharge_step)]
    if len(d_after) < 5:
        return None

    picked = pd.concat([chg_rows.iloc[[0]], d_after.iloc[:5]], axis=0).reset_index(drop=True)
    picked.loc[1:, "mission_step"] = ["take_off","hover","cruise","landing","standby"]
    picked.loc[0, "mission_step"] = "charge"
    return picked

def build_for_file(xlsx_path: str, labels: pd.DataFrame, windows: List[int], min_cycle: int = 4):
    fname = os.path.basename(xlsx_path)
    fkey = canonical_stem(fname)

    labf = labels[labels["join_key"] == fkey].copy()
    if labf.empty:
        raise ValueError(f"No labels for file_key={fkey}")

    cycle_to_label = dict(zip(labf["Cycle"].astype(int), labf["cycle_label_3name"]))

    xls = pd.ExcelFile(xlsx_path)
    rec = load_record(xls)
    step = load_step(xls)

    rec["cycle_index"] = rec["cycle_index"].astype(int)
    rec = rec[rec["cycle_index"] >= min_cycle].copy()
    step["cycle_index"] = step["cycle_index"].astype(int)
    step = step[step["cycle_index"] >= min_cycle].copy()

    outputs = {w: [] for w in windows}
    skipped_cycles = 0

    for c in sorted(step["cycle_index"].unique()):
        if c not in cycle_to_label:
            continue

        picked = pick_charge_then_5_discharge(step[step["cycle_index"] == c])
        if picked is None:
            skipped_cycles += 1
            continue

        thc = picked[picked["mission_step"].isin(["take_off","hover","cruise"])].copy()
        if len(thc) != 3:
            skipped_cycles += 1
            continue

        takeoff_onset = thc.loc[thc["mission_step"] == "take_off", "onset_dt"].iloc[0]

        segs = []
        for _, st in thc.iterrows():
            onset, end = st["onset_dt"], st["end_dt"]
            mask = (rec["cycle_index"] == c) & (rec["date"] >= onset) & (rec["date"] <= end)
            seg = rec.loc[mask, ["date","current_a","voltage_v","power_w","step_type"]].copy()
            if seg.empty:
                mask2 = (rec["cycle_index"] == c) & (rec["date"] >= onset) & (rec["date"] <= (end + pd.Timedelta(seconds=1)))
                seg = rec.loc[mask2, ["date","current_a","voltage_v","power_w","step_type"]].copy()
            if seg.empty:
                continue
            seg["mission_step"] = st["mission_step"]
            segs.append(seg)

        if not segs:
            skipped_cycles += 1
            continue

        df = pd.concat(segs, ignore_index=True).sort_values("date").reset_index(drop=True)
        df["t_cum_s"] = (df["date"] - takeoff_onset).dt.total_seconds()
        df = df[df["t_cum_s"] >= 0].copy()
        if df.empty:
            skipped_cycles += 1
            continue

        df["Cycle"] = int(c)
        df["file"] = fname                     # keep actual file name
        df["file_key"] = fkey                  # stable key
        df["label"] = cycle_to_label[int(c)]
        df["dvdt_v_per_s"] = compute_dvdt(df)

        for w in windows:
            dfw = df[df["t_cum_s"] <= float(w)].copy()
            if not dfw.empty:
                outputs[w].append(dfw)

    out = {w: (pd.concat(outputs[w], ignore_index=True) if outputs[w] else pd.DataFrame()) for w in windows}
    return out, skipped_cycles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--windows", default="1,2,5,10,20,30,50,60")
    ap.add_argument("--min_cycle", type=int, default=4)
    args = ap.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    need = {"file","Cycle","cycle_label_3name"}
    if not need.issubset(labels.columns):
        raise ValueError(f"labels_csv must contain columns {sorted(need)}")

    labels["Cycle"] = pd.to_numeric(labels["Cycle"], errors="coerce")
    labels = labels.dropna(subset=["Cycle"]).copy()
    labels["Cycle"] = labels["Cycle"].astype(int)
    labels["join_key"] = labels["file"].apply(canonical_stem)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.xlsx")))
    print(f"Found {len(files)} files in data_dir")

    accum = {w: [] for w in windows}
    total_skipped = 0
    ok_files = 0

    for fp in files:
        base = os.path.basename(fp)
        try:
            parts, skipped = build_for_file(fp, labels, windows, min_cycle=args.min_cycle)
            ok_files += 1
            total_skipped += skipped
            for w in windows:
                if not parts[w].empty:
                    accum[w].append(parts[w])
            print(f"[OK] {base}: " + ", ".join([f"{w}s={len(parts[w])}" for w in windows]) + f" | skipped_cycles={skipped}")
        except Exception as e:
            print(f"[SKIP FILE] {base}: {e}")

    for w in windows:
        out_df = pd.concat(accum[w], ignore_index=True) if accum[w] else pd.DataFrame()
        out_path = os.path.join(args.outdir, f"early_{w}s.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} rows={len(out_df)}")

    print(f"Files processed OK: {ok_files}/{len(files)}")
    print(f"Total skipped cycles (missing THC or no record match): {total_skipped}")

if __name__ == "__main__":
    main()

