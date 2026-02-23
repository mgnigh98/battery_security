#!/usr/bin/env python3
"""
build_cycle_window_features_v1.py

Input:
  out_early_windows_THC_rich/early_{t}s.csv  (datapoint-level)

Output:
  out_cycle_window_features/early_{t}s_cycle_features.csv (cycle-level)

Each output row corresponds to one (file, Cycle) for that window.
Features are computed using ALL datapoints with t_cum_s <= t.
"""

import os
import argparse
import numpy as np
import pandas as pd


WINDOWS_DEFAULT = "1,2,5,10,20,30,50,60"


def safe_div(a, b, eps=1e-9):
    return a / (b + eps)


def linreg_slope(x: np.ndarray, y: np.ndarray):
    """Least-squares slope of y ~ m*x + c. Returns nan if insufficient data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.all(~np.isfinite(x)) or np.all(~np.isfinite(y)):
        return np.nan
    # remove non-finite
    msk = np.isfinite(x) & np.isfinite(y)
    x = x[msk]; y = y[msk]
    if len(x) < 2:
        return np.nan
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def agg_one_cycle(g: pd.DataFrame) -> dict:
    """
    g: datapoints for a single (file, Cycle) in a given window
    returns dict of cycle-level features
    """
    # sort by time to define "start" and "end"
    g = g.sort_values("t_cum_s").reset_index(drop=True)

    # core series
    t = g["t_cum_s"].to_numpy(dtype=float)
    I = g["current_a"].to_numpy(dtype=float) if "current_a" in g else np.full(len(g), np.nan)
    V = g["voltage_v"].to_numpy(dtype=float) if "voltage_v" in g else np.full(len(g), np.nan)
    P = g["power_w"].to_numpy(dtype=float) if "power_w" in g else np.full(len(g), np.nan)

    Cap = g["capacity_ah"].to_numpy(dtype=float) if "capacity_ah" in g else np.full(len(g), np.nan)
    E = g["energy_wh"].to_numpy(dtype=float) if "energy_wh" in g else np.full(len(g), np.nan)
    SC = g["spec_cap"].to_numpy(dtype=float) if "spec_cap" in g else np.full(len(g), np.nan)

    # start values (first valid)
    def first_valid(x):
        for v in x:
            if np.isfinite(v):
                return float(v)
        return np.nan

    def last_valid(x):
        for v in x[::-1]:
            if np.isfinite(v):
                return float(v)
        return np.nan

    V0, Vt = first_valid(V), last_valid(V)
    I0, It = first_valid(I), last_valid(I)
    P0, Pt = first_valid(P), last_valid(P)
    Cap0, Capt = first_valid(Cap), last_valid(Cap)
    E0, Et = first_valid(E), last_valid(E)
    SC0, SCt = first_valid(SC), last_valid(SC)

    # robust dt counts
    n = int(len(g))
    tspan = float(np.nanmax(t) - np.nanmin(t)) if np.isfinite(t).any() else np.nan

    # dv/dt slope (voltage vs time)
    dVdt_slope = linreg_slope(t, V)

    # if dvdt column exists, summarize it
    dvdt = g["dvdt_v_per_s"].to_numpy(dtype=float) if "dvdt_v_per_s" in g else np.full(len(g), np.nan)

    # voltage sag features
    Vmin = float(np.nanmin(V)) if np.isfinite(V).any() else np.nan
    Vmax = float(np.nanmax(V)) if np.isfinite(V).any() else np.nan
    Vsag = (V0 - Vmin) if np.isfinite(V0) and np.isfinite(Vmin) else np.nan
    Vsag_ratio = safe_div(Vsag, V0) if np.isfinite(Vsag) and np.isfinite(V0) else np.nan

    # energy/capacity deltas within window
    dCap = (Capt - Cap0) if np.isfinite(Capt) and np.isfinite(Cap0) else np.nan
    dE = (Et - E0) if np.isfinite(Et) and np.isfinite(E0) else np.nan
    dSC = (SCt - SC0) if np.isfinite(SCt) and np.isfinite(SC0) else np.nan

    # mean absolute current (helps if sign conventions vary)
    I_abs_mean = float(np.nanmean(np.abs(I))) if np.isfinite(I).any() else np.nan

    # simple “IR-like” proxy using early change: ΔV / ΔI over the window
    dV = (Vt - V0) if np.isfinite(Vt) and np.isfinite(V0) else np.nan
    dI = (It - I0) if np.isfinite(It) and np.isfinite(I0) else np.nan
    IR_proxy_win = safe_div(dV, dI) if np.isfinite(dV) and np.isfinite(dI) and abs(dI) > 1e-9 else np.nan

    # summarize mission_step composition (counts)
    # counts help the model know if a cycle only has takeoff points in early windows
    ms_counts = {}
    if "mission_step" in g.columns:
        vc = g["mission_step"].astype(str).value_counts()
        for k in ["take_off", "hover", "cruise"]:
            ms_counts[f"cnt_{k}"] = int(vc.get(k, 0))
    else:
        ms_counts["cnt_take_off"] = ms_counts["cnt_hover"] = ms_counts["cnt_cruise"] = 0

    out = {
        # bookkeeping
        "n_points": n,
        "t_span_s": tspan,

        # raw summary stats
        "I_mean": float(np.nanmean(I)) if np.isfinite(I).any() else np.nan,
        "I_std": float(np.nanstd(I)) if np.isfinite(I).any() else np.nan,
        "I_abs_mean": I_abs_mean,

        "V_mean": float(np.nanmean(V)) if np.isfinite(V).any() else np.nan,
        "V_std": float(np.nanstd(V)) if np.isfinite(V).any() else np.nan,
        "V_min": Vmin,
        "V_max": Vmax,

        "P_mean": float(np.nanmean(P)) if np.isfinite(P).any() else np.nan,
        "P_std": float(np.nanstd(P)) if np.isfinite(P).any() else np.nan,

        # dynamics
        "dVdt_slope": dVdt_slope,
        "dvdt_mean": float(np.nanmean(dvdt)) if np.isfinite(dvdt).any() else np.nan,
        "dvdt_std": float(np.nanstd(dvdt)) if np.isfinite(dvdt).any() else np.nan,

        # sag + deltas
        "V0": V0,
        "Vt": Vt,
        "Vsag": Vsag,
        "Vsag_ratio": Vsag_ratio,

        "Cap0": Cap0,
        "Capt": Capt,
        "dCap": dCap,

        "E0": E0,
        "Et": Et,
        "dE": dE,

        "SC0": SC0,
        "SCt": SCt,
        "dSC": dSC,

        # window proxy
        "IR_proxy_win": IR_proxy_win,
    }
    out.update(ms_counts)
    return out


def build_for_window(in_csv: str, out_csv: str):
    df = pd.read_csv(in_csv)

    # Required columns
    required = ["file", "Cycle", "label", "t_cum_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(in_csv)} missing columns: {missing}")

    # Keep only THC datapoints if present (should already be)
    # Optional: drop rows with missing label
    df = df.dropna(subset=["label"])

    # Group + aggregate
    rows = []
    for (f, cyc), g in df.groupby(["file", "Cycle"], sort=False):
        label = g["label"].iloc[0]
        feats = agg_one_cycle(g)
        feats.update({"file": f, "Cycle": int(cyc), "label": str(label)})
        rows.append(feats)

    out = pd.DataFrame(rows)

    # Stable column order (best effort)
    first_cols = ["file", "Cycle", "label", "n_points", "t_span_s",
                  "cnt_take_off", "cnt_hover", "cnt_cruise"]
    cols = first_cols + [c for c in out.columns if c not in first_cols]
    out = out[cols]

    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} rows={len(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="out_early_windows_THC_rich", help="Folder containing early_{t}s.csv (datapoint-level)")
    ap.add_argument("--out_dir", default="out_cycle_window_features", help="Output folder for cycle-level features")
    ap.add_argument("--windows", default=WINDOWS_DEFAULT)
    args = ap.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    for w in windows:
        in_csv = os.path.join(args.in_dir, f"early_{w}s.csv")
        out_csv = os.path.join(args.out_dir, f"early_{w}s_cycle_features.csv")
        if not os.path.isfile(in_csv):
            print(f"[SKIP] missing {in_csv}")
            continue
        build_for_window(in_csv, out_csv)


if __name__ == "__main__":
    main()
