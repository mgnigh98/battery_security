#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eUAM Battery Risk Scoring from Coupling Matrix (Likelihood × Severity with phase weights, coupling, detectability)

USAGE:
  python euam_risk_scoring.py input.csv --outdir . --prefix risk --beta 1.8

INPUT CSV (minimal, 4 columns):
  Security, Safety, CouplingLevel, Severity(1-5)

OUTPUTS:
  <prefix>_ranked.csv       # sorted list with all computed fields
  <prefix>_heatmap.png      # annotated heatmap (Security × Safety)
  <prefix>_top10_bar.png    # Top-10 cell risks (bar chart)

Notes:
- Likelihood per phase is generated from mission-phase heuristics (can be edited below).
- Detectability (1–5) is inferred by security class and safety column (can be edited below).
- Risk formula:
    Risk = (w·L_phase) × (Severity^β) × coupling_multiplier × ((6 − Detectability)/5)
  with phase weights w = {Takeoff:0.40, Cruise:0.30, Landing:0.30}
  coupling multipliers: L→1.0, M→1.4, H→1.9, C→2.5
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


font_size = 18
rc = {"text.usetex": True, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (14, 8),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 1, 'lines.linewidth': 2.5, "axes.linewidth": 2.5,
          'axes.axisbelow': True}
# plt.subplots_adjust(left=0.153, right=0.98, top=0.695, bottom=0.126, hspace=0.2, wspace=0.2)
plt.rcParams.update(rc)

# -----------------------------
# Tunable parameters & heuristics
# -----------------------------

# Mission-phase weights (must sum to 1.0)
PHASE_WEIGHTS = {"Takeoff": 0.40, "Cruise": 0.30, "Landing": 0.30}

# Coupling multipliers (stronger spread than 1+K)
COUPLING_MAP = {"L": 1.0, "M": 1.4, "H": 1.9, "C": 2.5}

# Non-linear severity exponent (β)
DEFAULT_BETA = 1.8

# Base phase likelihood by SAFETY column (Takeoff, Cruise, Landing), scale 1..5
BASE_L_BY_SAFETY = {
    "Thermal Risk":     (4, 3, 2),  # high burst at TO, fades by Landing
    "Electrical Risk":  (4, 3, 3),
    "Mechanical Risk":  (2, 2, 4),  # higher during Landing (gear/touchdown, rotor tilt, etc.)
    "Materials Risk":   (2, 3, 2),
    "Systemic Risk":    (3, 3, 4),  # integration/nav/comms more critical at Landing
}

# Security-class adjustments (delta added to base L per phase; clamp to [1,5])
ADJ_BY_SECURITY = {
    "Data Manipulation":    (0, 1, 1),
    "Spoofing":             (0, 0, 1),
    "Hardware Exploits":    (1, 0, 0),
    "Sensor Tampering":     (1, 0, 0),
    "Unauthorized Access":  (0, 0, 1),
}

# Detectability (1=hard to detect → worst, 5=easy to detect → best)
DETECT_BASE_BY_SECURITY = {
    "Data Manipulation": 3,
    "Spoofing": 2,
    "Hardware Exploits": 2,
    "Sensor Tampering": 3,
    "Unauthorized Access": 2,
}
# Column tweaks: Materials often leaves physical evidence (easier), Systemic is harder/pre-symptomatic
DETECT_COL_ADJ = {
    "Thermal Risk": 0,
    "Electrical Risk": 0,
    "Mechanical Risk": 0,
    "Materials Risk": +1,
    "Systemic Risk": -1,
}

# Order for pretty plots
SECURITY_ORDER = [
    "Data Manipulation", "Spoofing", "Hardware Exploits",
    "Sensor Tampering", "Unauthorized Access"
]
SAFETY_ORDER = [
    "Thermal Risk", "Electrical Risk", "Mechanical Risk",
    "Materials Risk", "Systemic Risk"
]

# -----------------------------
# Helpers
# -----------------------------
def _phase_likelihoods(security: str, safety: str):
    base = np.array(BASE_L_BY_SAFETY[safety], dtype=float)
    adj = np.array(ADJ_BY_SECURITY.get(security, (0, 0, 0)), dtype=float)
    Ls = np.clip(base + adj, 1, 5)
    return Ls  # (TO, CR, LD)

def _detectability(security: str, safety: str):
    d = DETECT_BASE_BY_SECURITY.get(security, 3) + DETECT_COL_ADJ.get(safety, 0)
    return int(np.clip(d, 1, 5))

def compute_risk(df: pd.DataFrame, beta: float):
    # Likelihoods per phase
    L = df.apply(lambda r: _phase_likelihoods(r["Security"], r["Safety"]), axis=1, result_type="expand")
    L.columns = ["L_Takeoff", "L_Cruise", "L_Landing"]
    df = pd.concat([df, L], axis=1)

    # Detectability
    df["Detectability(1-5)"] = df.apply(lambda r: _detectability(r["Security"], r["Safety"]), axis=1)

    # Coupling multiplier
    df["mK"] = df["CouplingLevel"].str.upper().map(COUPLING_MAP).fillna(1.4)

    # Phase-weighted likelihood
    w = PHASE_WEIGHTS
    df["L_phase_weighted"] = (
        df["L_Takeoff"] * w["Takeoff"] +
        df["L_Cruise"]  * w["Cruise"]  +
        df["L_Landing"] * w["Landing"]
    )

    # Detectability multiplier: (6-D)/5 => D=1 → 1.0, D=5 → 0.2
    df["fD"] = (6 - df["Detectability(1-5)"]) / 5.0

    # Risk
    sev = df["Severity(1-5)"].clip(1, 5)
    df["Risk"] = df["L_phase_weighted"] * (sev ** beta) * df["mK"] * df["fD"]
    return df

def plot_heatmap(pivot: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    ax.set_title("Risk Heatmap")
    # ax.set_title("Risk Heatmap (phase-weighted L $ \times S^\beta \times coupling \times (6−D)/5)$")
    # annotate
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Risk score")
    fig.tight_layout()
    plt.show()
    fig.savefig(outpath, dpi=300)

def plot_topN(ranked: pd.DataFrame, outpath: str, topN: int = 10):
    t = ranked.head(topN).copy()
    labels = [f"{r.Security} → {r.Safety}" for r in t.itertuples()]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y, t["Risk"].values)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Risk score")
    ax.set_title(f"Top-{topN} Coupled Safety–Security Cells by Risk")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()
    fig.savefig(outpath, dpi=300)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="Path to input CSV (Security, Safety, CouplingLevel, Severity(1-5))")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--prefix", default="risk", help="Output file prefix")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Severity exponent β (default: 1.8)")
    ap.add_argument("--topN", type=int, default=10, help="Top-N bars to show (default: 10)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read input
    df = pd.read_csv(args.input_csv)
    needed = {"Security", "Safety", "CouplingLevel", "Severity(1-5)"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # Compute
    scored = compute_risk(df.copy(), beta=args.beta)

    # Order rows/cols for plotting
    pivot = (scored
             .pivot_table(index="Security", columns="Safety", values="Risk", aggfunc="mean")
             .reindex(index=SECURITY_ORDER, columns=SAFETY_ORDER))

    # Rank and save
    ranked = scored.sort_values("Risk", ascending=False).reset_index(drop=True)
    ranked_path = os.path.join(args.outdir, f"{args.prefix}_ranked.csv")
    ranked.to_csv(ranked_path, index=False)

    # Figures
    heatmap_path = os.path.join(args.outdir, f"{args.prefix}_heatmap.png")
    plot_heatmap(pivot, heatmap_path)

    bar_path = os.path.join(args.outdir, f"{args.prefix}_top10_bar.png")
    plot_topN(ranked, bar_path, topN=args.topN)

    # print(f"[OK] Wrote:\n  {ranked_path}\n  {heatmap_path}\n  {bar_path}")

if __name__ == "__main__":
    main()
