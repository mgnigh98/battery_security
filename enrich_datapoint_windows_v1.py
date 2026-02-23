#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

WINDOWS_DEFAULT = "1,2,5,10,20,30,50,60"

def linreg_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    msk = np.isfinite(x) & np.isfinite(y)
    x = x[msk]; y = y[msk]
    if len(x) < 2:
        return np.nan
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

def enrich_one_cycle(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("t_cum_s").reset_index(drop=True)

    # baseline (first row)
    V0 = g["voltage_v"].iloc[0] if "voltage_v" in g else np.nan
    Cap0 = g["capacity_ah"].iloc[0] if "capacity_ah" in g else np.nan
    E0 = g["energy_wh"].iloc[0] if "energy_wh" in g else np.nan
    P0 = g["power_w"].iloc[0] if "power_w" in g else np.nan

    g["V0"] = V0
    g["Cap0"] = Cap0
    g["E0"] = E0
    g["P0"] = P0

    # deltas
    g["V_delta"] = g["voltage_v"] - V0
    g["Cap_delta"] = g["capacity_ah"] - Cap0
    g["E_delta"] = g["energy_wh"] - E0
    g["P_delta"] = g["power_w"] - P0

    # cumulative stats up to each row
    V = g["voltage_v"].to_numpy(dtype=float)
    I = g["current_a"].to_numpy(dtype=float)
    P = g["power_w"].to_numpy(dtype=float)
    t = g["t_cum_s"].to_numpy(dtype=float)

    # cumulative mean/std via expanding
    g["V_mean_upto"] = pd.Series(V).expanding().mean().to_numpy()
    g["V_std_upto"]  = pd.Series(V).expanding().std().to_numpy()
    g["V_min_upto"]  = pd.Series(V).expanding().min().to_numpy()

    g["I_mean_upto"] = pd.Series(I).expanding().mean().to_numpy()
    g["I_std_upto"]  = pd.Series(I).expanding().std().to_numpy()
    g["I_abs_mean_upto"] = pd.Series(np.abs(I)).expanding().mean().to_numpy()

    g["P_mean_upto"] = pd.Series(P).expanding().mean().to_numpy()
    g["P_std_upto"]  = pd.Series(P).expanding().std().to_numpy()

    # voltage sag up to current time
    g["Vsag_upto"] = V0 - g["V_min_upto"]
    g["Vsag_ratio_upto"] = g["Vsag_upto"] / (V0 + 1e-6)

    # dv/dt slope up to current time (regression)
    slopes = []
    for k in range(len(g)):
        slopes.append(linreg_slope(t[:k+1], V[:k+1]))
    g["dVdt_slope_upto"] = slopes

    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="out_early_windows_THC_rich")
    ap.add_argument("--out_dir", default="out_early_windows_THC_rich_enriched")
    ap.add_argument("--windows", default=WINDOWS_DEFAULT)
    args = ap.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    for w in windows:
        in_csv = os.path.join(args.in_dir, f"early_{w}s.csv")
        out_csv = os.path.join(args.out_dir, f"early_{w}s_enriched.csv")
        if not os.path.isfile(in_csv):
            print(f"[SKIP] missing {in_csv}")
            continue

        df = pd.read_csv(in_csv)

        # group by cycle identity
        df2 = []
        for (_, _), g in df.groupby(["file", "Cycle"], sort=False):
            df2.append(enrich_one_cycle(g))

        out = pd.concat(df2, ignore_index=True)
        out.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} rows={len(out)}")

if __name__ == "__main__":
    main()
