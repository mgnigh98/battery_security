#!/usr/bin/env python3
"""
build_early_feature_datasets_v1.py

Build ML-ready early-window datasets.

For each window t:
- Collect cumulative takeoff→hover→cruise datapoints
- Compute per-cycle aggregated features
- Output one row per cycle

Produces:
    early_{t}s_features.csv
"""

import os, re, glob, argparse
import numpy as np
import pandas as pd


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def canonical_stem(name):
    s = os.path.basename(str(name))
    s = os.path.splitext(s)[0]
    s = re.sub(r"\s*_good\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*_bad\s*$", "", s, flags=re.I)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def linear_slope(x, y):
    if len(x) < 2:
        return np.nan
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m


# -------------------------------------------------
# Main builder
# -------------------------------------------------

def build_features_for_file(filepath, labels_df, windows):

    fname = os.path.basename(filepath)
    fkey = canonical_stem(fname)

    lab = labels_df[labels_df["join_key"] == fkey]
    if lab.empty:
        return None

    cycle_to_label = dict(zip(lab["Cycle"], lab["cycle_label_3name"]))

    xls = pd.ExcelFile(filepath)
    record = pd.read_excel(xls, sheet_name="record")
    step = pd.read_excel(xls, sheet_name="step")

    # Clean record
    record["Cycle Index"] = pd.to_numeric(record["Cycle Index"], errors="coerce")
    record["Date"] = pd.to_datetime(record["Date"], errors="coerce")

    record = record.dropna(subset=["Cycle Index","Date"])

    outputs = {w: [] for w in windows}

    for cycle in sorted(record["Cycle Index"].unique()):
        cycle = int(cycle)
        if cycle < 4:
            continue
        if cycle not in cycle_to_label:
            continue

        # get THC step intervals
        step_cycle = step[step["Cycle Index"] == cycle].copy()
        if step_cycle.empty:
            continue

        step_cycle = step_cycle.sort_values("Oneset Date")

        # find first charge
        chg = step_cycle[step_cycle["Chg. Cap.(Ah)"] > 0]
        if chg.empty:
            continue

        chg_idx = chg.index[0]
        after = step_cycle.loc[chg_idx+1:]

        dchg = after[after["DChg. Cap.(Ah)"] > 0]
        if len(dchg) < 5:
            continue

        d5 = dchg.iloc[:5]

        takeoff_onset = pd.to_datetime(d5.iloc[0]["Oneset Date"])

        thc_intervals = []
        for i in range(3):  # takeoff, hover, cruise
            onset = pd.to_datetime(d5.iloc[i]["Oneset Date"])
            end   = pd.to_datetime(d5.iloc[i]["End Date"])
            thc_intervals.append((onset,end))

        # collect record rows
        segs = []
        for onset,end in thc_intervals:
            mask = (
                (record["Cycle Index"] == cycle) &
                (record["Date"] >= onset) &
                (record["Date"] <= end)
            )
            segs.append(record[mask])

        if not segs:
            continue

        df = pd.concat(segs).sort_values("Date")
        if df.empty:
            continue

        df["t_cum_s"] = (df["Date"] - takeoff_onset).dt.total_seconds()
        df = df[df["t_cum_s"] >= 0]

        if df.empty:
            continue

        for w in windows:

            dfw = df[df["t_cum_s"] <= w].copy()
            if dfw.empty:
                continue

            # ----- feature engineering -----

            I = dfw["Current(A)"].values
            V = dfw["Voltage(V)"].values
            C = dfw["Capacity(Ah)"].values
            SC = dfw["Spec. Cap.(mAh/g)"].values
            EC = dfw["Energy(Wh)"].values

            t = dfw["t_cum_s"].values

            features = {
                "file": fname,
                "Cycle": cycle,
                "label": cycle_to_label[cycle],

                # current
                "I_mean": np.mean(I),
                "I_std": np.std(I),

                # voltage
                "V_mean": np.mean(V),
                "V_min": np.min(V),
                "V_max": np.max(V),
                "V_sag": V[0] - np.min(V),

                # cumulative deltas
                "delta_capacity": C[-1] - C[0],
                "delta_spec_cap": SC[-1] - SC[0],
                "delta_energy": EC[-1] - EC[0],

                # dv/dt via regression
                "dVdt_slope": linear_slope(t, V),
                "dVdt_std": np.std(np.diff(V)/np.diff(t)) if len(t)>1 else np.nan
            }

            outputs[w].append(features)

    return outputs


# -------------------------------------------------
# Runner
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--windows", default="1,2,5,10,20,30,50,60")
    args = parser.parse_args()

    windows = [int(x) for x in args.windows.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    labels["Cycle"] = pd.to_numeric(labels["Cycle"])
    labels["join_key"] = labels["file"].apply(canonical_stem)

    files = glob.glob(os.path.join(args.data_dir,"*.xlsx"))

    all_outputs = {w: [] for w in windows}

    for f in files:
        result = build_features_for_file(f, labels, windows)
        if result:
            for w in windows:
                all_outputs[w].extend(result[w])

    for w in windows:
        df = pd.DataFrame(all_outputs[w])
        df.to_csv(os.path.join(args.outdir, f"early_{w}s_features.csv"), index=False)
        print(f"early_{w}s_features.csv rows={len(df)}")


if __name__ == "__main__":
    main()
