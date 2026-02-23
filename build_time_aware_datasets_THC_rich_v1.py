#!/usr/bin/env python3
import os, re, glob, argparse
import numpy as np
import pandas as pd

def canonical_stem(name: str) -> str:
    s = os.path.basename(str(name)).strip()
    s = os.path.splitext(s)[0].strip()
    s = re.sub(r"\s*_good\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*_bad\s*$", "", s, flags=re.I)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def normcol(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def remap_columns(df: pd.DataFrame, mapping):
    cur = {normcol(c): c for c in df.columns}
    ren = {}
    for canon, cands in mapping.items():
        for cand in cands:
            k = normcol(cand)
            if k in cur:
                ren[cur[k]] = canon
                break
    return df.rename(columns=ren)

def load_record(xls: pd.ExcelFile) -> pd.DataFrame:
    sheets = {s.lower(): s for s in xls.sheet_names}
    if "record" not in sheets:
        raise ValueError("Missing 'record' sheet")
    df = pd.read_excel(xls, sheet_name=sheets["record"])

    mapping = {
        "cycle_index": ["Cycle Index","CycleIndex","Cycle"],
        "step_type":   ["Step Type","StepType","Type","Step"],
        "date":        ["Date"],
        "current_a":   ["Current(A)","Current (A)","Current"],
        "voltage_v":   ["Voltage(V)","Voltage (V)","Voltage"],
        "capacity_ah": ["Capacity(Ah)","Capacity (Ah)"],
        "spec_cap":    ["Spec. Cap.(mAh/g)","Spec Cap.(mAh/g)","Spec. Cap. (mAh/g)"],
        "chg_cap":     ["Chg. Cap.(Ah)","Chg. Cap. (Ah)"],
        "chg_spec":    ["Chg. Spec. Cap.(mAh/g)","Chg. Spec. Cap. (mAh/g)"],
        "dchg_cap":    ["DChg. Cap.(Ah)","DChg. Cap. (Ah)"],
        "dchg_spec":   ["DChg. Spec. Cap.(mAh/g)","DChg. Spec. Cap. (mAh/g)"],
        "energy_wh":   ["Energy(Wh)","Energy (Wh)"],
        "power_w":     ["Power(W)","Power (W)"],
    }
    df = remap_columns(df, mapping)

    need = ["cycle_index","date","step_type","current_a","voltage_v"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"record missing columns after remap: {miss}")

    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["current_a","voltage_v","capacity_ah","spec_cap","chg_cap","chg_spec","dchg_cap","dchg_spec","energy_wh","power_w"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "power_w" not in df.columns:
        df["power_w"] = df["voltage_v"] * df["current_a"]

    return df.dropna(subset=["cycle_index","date"]).copy()

def load_step(xls: pd.ExcelFile) -> pd.DataFrame:
    sheets = {s.lower(): s for s in xls.sheet_names}
    if "step" not in sheets:
        raise ValueError("Missing 'step' sheet")
    df = pd.read_excel(xls, sheet_name=sheets["step"])

    mapping = {
        "cycle_index": ["Cycle Index","CycleIndex","Cycle"],
        "step_type":   ["Step Type","StepType","Type","Step"],
        "onset_dt":    ["Oneset Date","Onset Date","Start Date","Oneset"],
        "end_dt":      ["End Date","EndDate","End time","End"],
        "step_number": ["Step Number","StepNumber","Step No.","StepNo"],
        "chg_cap":     ["Chg. Cap.(Ah)","Chg. Cap. (Ah)"],
        "dchg_cap":    ["DChg. Cap.(Ah)","DChg. Cap. (Ah)"],
    }
    df = remap_columns(df, mapping)

    need = ["cycle_index","step_type","onset_dt","end_dt"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"step missing columns after remap: {miss}")

    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
    df["onset_dt"] = pd.to_datetime(df["onset_dt"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_dt"], errors="coerce")
    for c in ["step_number","chg_cap","dchg_cap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["cycle_index","onset_dt","end_dt"]).copy()

def pick_charge_then_5_discharge(step_cycle: pd.DataFrame) -> pd.DataFrame | None:
    cdf = step_cycle.copy()
    sort_cols = [c for c in ["onset_dt","end_dt","step_number"] if c in cdf.columns]
    if sort_cols:
        cdf = cdf.sort_values(sort_cols, na_position="last")

    # charge
    if "chg_cap" in cdf.columns:
        chg_rows = cdf[cdf["chg_cap"].fillna(0) > 0]
    else:
        chg_rows = cdf[cdf["step_type"].astype(str).str.lower().str.contains("chg")]

    if chg_rows.empty:
        return None

    chg = chg_rows.iloc[0]
    after = cdf.loc[chg.name+1:] if chg.name in cdf.index else cdf.iloc[0:0]

    # discharge
    if "dchg_cap" in after.columns:
        d_after = after[after["dchg_cap"].fillna(0) > 0]
    else:
        d_after = after[after["step_type"].astype(str).str.lower().str.contains("dchg|discharge")]

    if len(d_after) < 5:
        return None

    picked = pd.concat([chg_rows.iloc[[0]], d_after.iloc[:5]], ignore_index=True)
    picked.loc[0, "mission_step"] = "charge"
    picked.loc[1:, "mission_step"] = ["take_off","hover","cruise","landing","standby"]
    return picked

def add_dvdt_per_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust dv/dt:
    - group by (file_key, Cycle)
    - within group: sort by t_cum_s
    - drop duplicate t values by averaging V at same t (prevents dt=0)
    - compute dv/dt using np.gradient
    """
    out = df.copy()
    out["dvdt_v_per_s"] = np.nan

    for (fk, cyc), g in out.groupby(["file_key","Cycle"], sort=False):
        gg = g.sort_values("t_cum_s")
        # aggregate duplicates in time
        gg2 = gg.groupby("t_cum_s", as_index=False).agg(
            voltage_v=("voltage_v","mean")
        )
        t = gg2["t_cum_s"].to_numpy(dtype=float)
        v = gg2["voltage_v"].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        dvdt = np.gradient(v, t)
        # map back: assign dvdt by matching t_cum_s
        dvdt_map = dict(zip(t, dvdt))
        out.loc[gg.index, "dvdt_v_per_s"] = gg["t_cum_s"].map(dvdt_map).to_numpy()

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--windows", default="1,2,5,10,20,30,50,60")
    ap.add_argument("--min_cycle", type=int, default=4)
    ap.add_argument("--discharge_only", action="store_true",
                    help="Keep only discharge rows (step_type contains DChg or |I|>1e-4)")
    args = ap.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    labels["Cycle"] = pd.to_numeric(labels["Cycle"], errors="coerce")
    labels = labels.dropna(subset=["Cycle"]).copy()
    labels["Cycle"] = labels["Cycle"].astype(int)
    labels["join_key"] = labels["file"].apply(canonical_stem)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.xlsx")))
    print(f"Found {len(files)} files")

    accum = {w: [] for w in windows}
    skipped_files = 0
    skipped_cycles = 0

    for fp in files:
        fname = os.path.basename(fp)
        fkey = canonical_stem(fname)

        labf = labels[labels["join_key"] == fkey]
        if labf.empty:
            skipped_files += 1
            continue

        cycle_to_label = dict(zip(labf["Cycle"], labf["cycle_label_3name"]))

        try:
            xls = pd.ExcelFile(fp)
            rec = load_record(xls)
            stp = load_step(xls)

            rec["cycle_index"] = rec["cycle_index"].astype(int)
            rec = rec[rec["cycle_index"] >= args.min_cycle].copy()
            stp["cycle_index"] = stp["cycle_index"].astype(int)
            stp = stp[stp["cycle_index"] >= args.min_cycle].copy()

            for cyc in sorted(stp["cycle_index"].unique()):
                if cyc not in cycle_to_label:
                    continue

                picked = pick_charge_then_5_discharge(stp[stp["cycle_index"] == cyc])
                if picked is None:
                    skipped_cycles += 1
                    continue

                thc = picked[picked["mission_step"].isin(["take_off","hover","cruise"])].copy()
                if len(thc) != 3:
                    skipped_cycles += 1
                    continue

                takeoff_onset = thc.loc[thc["mission_step"] == "take_off", "onset_dt"].iloc[0]

                segs = []
                for _, r in thc.iterrows():
                    onset, end = r["onset_dt"], r["end_dt"]
                    m = (rec["cycle_index"] == cyc) & (rec["date"] >= onset) & (rec["date"] <= end)
                    seg = rec.loc[m].copy()
                    if seg.empty:
                        # small tolerance
                        m2 = (rec["cycle_index"] == cyc) & (rec["date"] >= onset) & (rec["date"] <= (end + pd.Timedelta(seconds=1)))
                        seg = rec.loc[m2].copy()
                    if seg.empty:
                        continue
                    seg["mission_step"] = r["mission_step"]
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

                df["Cycle"] = int(cyc)
                df["file"] = fname
                df["file_key"] = fkey
                df["label"] = cycle_to_label[int(cyc)]

                if args.discharge_only:
                    stxt = df["step_type"].astype(str).str.lower()
                    keep = stxt.str.contains("dchg") | (df["current_a"].abs() > 1e-4)
                    df = df[keep].copy()

                if df.empty:
                    skipped_cycles += 1
                    continue

                # add dvdt robustly (needs per-cycle grouping, but we can do on this cycle chunk)
                df = add_dvdt_per_cycle(df)

                # write per window accum
                for w in windows:
                    dfw = df[df["t_cum_s"] <= float(w)].copy()
                    if not dfw.empty:
                        accum[w].append(dfw)

        except Exception:
            skipped_files += 1
            continue

    for w in windows:
        out_df = pd.concat(accum[w], ignore_index=True) if accum[w] else pd.DataFrame()
        out_path = os.path.join(args.outdir, f"early_{w}s.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} rows={len(out_df)}")

    print(f"Skipped files (no labels or read error): {skipped_files}")
    print(f"Skipped cycles (missing THC or no record match): {skipped_cycles}")

if __name__ == "__main__":
    main()



