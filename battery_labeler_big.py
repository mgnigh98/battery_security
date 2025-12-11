#!/usr/bin/env python3
"""
battery_labeler_big.py

Auto-label battery health (good/bad) for very large Excel datasets by
sampling cycles and aggregating robust features.

Features:
- Per-cycle summaries (capacity, energy, polarization, durations, etc.)
- Aggregates across chosen cycles (means, deltas, slopes per 100 cycles)
- Two labeling modes:
    * Composite health score (thresholded by median)
    * GMM clustering (unsupervised; higher-score cluster => "good")
- Optional 2D PCA plot with GMM contours for visualization

Usage examples:
  # Early condition only (cycles 4–10)
  python battery_labeler_big.py \
      --input_glob "/path/to/batts/*.xlsx" \
      --out_csv "/path/out/labels.csv" \
      --method both \
      --cycles 4-10

  # Long-term sampling (every 100th cycle after formation)
  python battery_labeler_big.py \
      --input_glob "/path/to/batts/*.xlsx" \
      --out_csv "/path/out/labels.csv" \
      --method both \
      --every_n 100

  # Combine both strategies + plot
  python battery_labeler_big.py \
      --input_glob "/path/to/batts/*.xlsx" \
      --out_csv "/path/out/labels.csv" \
      --method both \
      --cycles 4-10 --every_n 100 \
      --plot_out "/path/out/gmm_plot.png"
"""

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="openpyxl.styles.stylesheet"
)

# Expected column names (as in your samples)
REQUIRED_COLS = [
    'Cycle Index', 'Time(h)', 'Current(A)', 'Voltage(V)',
    'Chg. Cap.(Ah)', 'DChg. Cap.(Ah)'
]

# Cycles 1–3 are formation and ignored by default
FORMATION_END = 3


# ---------------------------- Utilities ---------------------------- #

def load_excel_sheet_safely(path, sheet="record"):
    """
    Robustly load a sheet from an Excel file.
    - If sheet='auto', search all sheets for one containing REQUIRED_COLS.
    - If a specific sheet name is given, try case-insensitive match first.
    - Raises a clear error if nothing suitable is found.
    """
    xls = pd.ExcelFile(path)  # uses openpyxl under the hood for .xlsx
    sheet_names = xls.sheet_names

    def has_required_cols(df):
        return all(col in df.columns for col in REQUIRED_COLS)

    # If user asked for a specific sheet (default: 'record')
    if sheet and sheet.lower() != "auto":
        # Case-insensitive match
        match = None
        for s in sheet_names:
            if s.strip().lower() == sheet.strip().lower():
                match = s
                break
        if match is None:
            raise ValueError(
                f"Sheet '{sheet}' not found in {path}. Available sheets: {sheet_names}"
            )

        USECOLS: List[str] = ['Cycle Index', 'Time(h)', 'Current(A)', 'Voltage(V)', 'Chg. Cap.(Ah)', 'DChg. Cap.(Ah)']
        df = pd.read_excel(path, sheet_name=match, usecols=USECOLS)

        # df = pd.read_excel(path, sheet_name=match)
        if not has_required_cols(df):
            raise ValueError(
                f"Sheet '{match}' in {path} does not have required columns: {REQUIRED_COLS}"
            )
        return df

    # Auto mode: scan all sheets for required columns
    for s in sheet_names:
        df = pd.read_excel(path, sheet_name=s)
        if has_required_cols(df):
            return df

    # Nothing matched
    raise ValueError(
        f"No sheet with required columns found in {path}. "
        f"Checked sheets: {sheet_names}. Needed: {REQUIRED_COLS}"
    )


def parse_cycles_arg(cycles_arg):
    """Parse '4-10' or '7' -> (start, end)."""
    if not cycles_arg:
        return None
    s = cycles_arg.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a), int(b))
    else:
        c = int(s)
        return (c, c)


def select_cycles(unique_cycles, cycles_range=None, every_n=None):
    """
    Select cycles to use given available unique cycles and sampling strategy.
    - Drops formation cycles (<= 3)
    - If neither cycles_range nor every_n given: default to {4,5}
    """
    uc = np.array(sorted(unique_cycles))
    uc = uc[uc > FORMATION_END]
    if uc.size == 0:
        return []

    selected = set()

    if cycles_range is not None:
        lo, hi = cycles_range
        selected.update([c for c in uc if lo <= c <= hi])

    if every_n is not None and every_n > 0:
        base = max(FORMATION_END + 1, uc.min())
        selected.update([c for c in uc if (c - base) % every_n == 0])

    if not selected and cycles_range is None and every_n is None:
        # default to cycles 4 & 5
        selected.update([c for c in uc if c in (4, 5)])

    return sorted(selected)


def summarize_features_one_cycle(cdf):
    """Compute per-cycle features for a single cycle dataframe cdf."""
    chg = cdf[cdf['Current(A)'] > 0]
    dchg = cdf[cdf['Current(A)'] < 0]

    chg_cap = float(chg['Chg. Cap.(Ah)'].max()) if not chg.empty else np.nan
    dchg_cap = float(dchg['DChg. Cap.(Ah)'].max()) if not dchg.empty else np.nan
    ce = (dchg_cap / chg_cap) if (chg_cap and chg_cap > 0) else np.nan

    v_dis_avg = float(dchg['Voltage(V)'].mean()) if not dchg.empty else np.nan
    v_chg_avg = float(chg['Voltage(V)'].mean()) if not chg.empty else np.nan

    t_dis = float(dchg['Time(h)'].iloc[-1] - dchg['Time(h)'].iloc[0]) if len(dchg) >= 2 else np.nan
    t_chg = float(chg['Time(h)'].iloc[-1] - chg['Time(h)'].iloc[0]) if len(chg) >= 2 else np.nan

    i_dis = float(dchg['Current(A)'].abs().mean()) if not dchg.empty else np.nan
    i_chg = float(chg['Current(A)'].abs().mean()) if not chg.empty else np.nan

    # Discharge energy (Wh) via trapezoidal rule on V*|I| vs time (time in hours)
    if not dchg.empty:
        dchgs = dchg.sort_values('Time(h)')
        energy_wh = float(np.trapz(dchgs['Voltage(V)'] * dchgs['Current(A)'].abs(), dchgs['Time(h)']))
        energy_wh = abs(energy_wh)
    else:
        energy_wh = np.nan

    return dict(
        chg_cap_Ah=chg_cap, dchg_cap_Ah=dchg_cap, coul_eff=ce,
        v_dis_avg=v_dis_avg, v_chg_avg=v_chg_avg,
        t_dis_h=t_dis, t_chg_h=t_chg,
        i_dis_A=i_dis, i_chg_A=i_chg,
        energy_dis_Wh=energy_wh
    )


def summarize_features(df, chosen_cycles):
    """Return per-cycle summary dataframe (index = cycle)."""
    rows = {}
    for c in chosen_cycles:
        cdf = df[df['Cycle Index'] == c]
        if cdf.empty:
            continue
        rows[c] = summarize_features_one_cycle(cdf)
    return pd.DataFrame(rows).T  # index: cycle


def safe_slope(x, y):
    """Slope of y vs x via least squares; NaN if insufficient data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    coeff = np.polyfit(x[mask], y[mask], 1)  # slope, intercept
    return float(coeff[0])


def aggregate_features(per_cycle_df):
    """
    Aggregate over cycles: means, deltas, and slopes per 100 cycles for key indicators.
    """
    if per_cycle_df is None or per_cycle_df.shape[0] == 0:
        return None

    out = {}

    # Means & deltas
    cols = [
        'dchg_cap_Ah', 'chg_cap_Ah', 'coul_eff', 'v_dis_avg', 'v_chg_avg',
        't_dis_h', 't_chg_h', 'i_dis_A', 'i_chg_A', 'energy_dis_Wh'
    ]
    for k in cols:
        out[f'mean_{k}'] = per_cycle_df[k].mean()
        out[f'delta_{k}'] = per_cycle_df[k].iloc[-1] - per_cycle_df[k].iloc[0]

    # Polarization (charge avg V − discharge avg V)
    pol = per_cycle_df['v_chg_avg'] - per_cycle_df['v_dis_avg']
    out['mean_polarization_V'] = float(pol.mean())
    out['delta_polarization_V'] = float(pol.iloc[-1] - pol.iloc[0])

    # Slopes per 100 cycles (trend)
    cyc = per_cycle_df.index.values.astype(float)
    out['slope_cap_per100']    = safe_slope(cyc, per_cycle_df['dchg_cap_Ah'])   * 100.0
    out['slope_energy_per100'] = safe_slope(cyc, per_cycle_df['energy_dis_Wh']) * 100.0
    out['slope_polar_per100']  = safe_slope(cyc, pol)                            * 100.0

    return out


def compute_health_score(df_features):
    """
    Composite score (higher = healthier).
    + mean capacity, energy, discharge voltage, discharge time, capacity slope, energy slope
    - mean polarization, polarization slope
    """
    pos = [
        'mean_dchg_cap_Ah', 'mean_energy_dis_Wh', 'mean_v_dis_avg',
        'mean_t_dis_h', 'slope_cap_per100', 'slope_energy_per100'
    ]
    neg = ['mean_polarization_V', 'slope_polar_per100']
    use = pos + neg

    X = df_features[use].values
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    score = Z[:, :len(pos)].sum(axis=1) - Z[:, len(pos):].sum(axis=1)
    return score


def label_by_gmm(df_features, score, random_state=0):
    """
    2-cluster GMM on compact robust feature set; higher-score cluster => "good".
    """
    cols = [
        'mean_dchg_cap_Ah', 'mean_energy_dis_Wh', 'mean_v_dis_avg', 'mean_t_dis_h',
        'mean_polarization_V', 'slope_cap_per100', 'slope_energy_per100', 'slope_polar_per100'
    ]
    X = df_features[cols].values
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state)
    gmm_labels = gmm.fit_predict(X)

    mean_scores = [float(np.mean(score[gmm_labels == k])) for k in [0, 1]]
    good_cluster = int(np.argmax(mean_scores))
    labels = np.where(gmm_labels == good_cluster, 'good', 'bad')
    return labels, gmm


# def make_plot(df_features, final_labels, out_png):
#     """
#     2D PCA projection + GMM contours (fitted in 2D for visualization only).
#     """
#     cols = [
#         'mean_dchg_cap_Ah', 'mean_energy_dis_Wh', 'mean_v_dis_avg', 'mean_t_dis_h',
#         'mean_polarization_V', 'slope_cap_per100', 'slope_energy_per100', 'slope_polar_per100'
#     ]
#     X = df_features[cols].values
#     scaler = StandardScaler()
#     Z = scaler.fit_transform(X)
#     pca = PCA(n_components=2, random_state=0)
#     Z2 = pca.fit_transform(Z)
#
#     gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
#     gmm2.fit(Z2)
#
#     xmin, ymin = Z2.min(axis=0) - 0.5
#     xmax, ymax = Z2.max(axis=0) + 0.5
#     xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
#     grid = np.c_[xx.ravel(), yy.ravel()]
#     zz = -gmm2.score_samples(grid)  # negative log-likelihood for nice contours
#
#     plt.figure(figsize=(7, 6))
#     plt.contourf(xx, yy, zz.reshape(xx.shape), levels=20, alpha=0.3)
#
#     mask_good = (final_labels == 'good')
#     mask_bad  = (final_labels == 'bad')
#     plt.scatter(Z2[mask_good, 0], Z2[mask_good, 1], label="good")
#     plt.scatter(Z2[mask_bad,  0], Z2[mask_bad,  1], label="bad")
#
#     for i, fname in enumerate(df_features.index):
#         plt.annotate(Path(fname).stem, (Z2[i, 0], Z2[i, 1]))
#
#     plt.xlabel("PCA-1")
#     plt.ylabel("PCA-2")
#     plt.title("Battery health clustering (sampled cycles + slopes)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=150)
#     plt.close()


def make_plot(df_features, final_labels, out_png, scores=None, cmap_name="viridis"):
    """
    2D PCA projection + GMM contours (fitted in 2D for visualization only),
    with points colored by health score and outlined by label.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    cols = [
        'mean_dchg_cap_Ah', 'mean_energy_dis_Wh', 'mean_v_dis_avg', 'mean_t_dis_h',
        'mean_polarization_V', 'slope_cap_per100', 'slope_energy_per100', 'slope_polar_per100'
    ]
    X = df_features[cols].values
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z2 = pca.fit_transform(Z)

    # Fit a 2D GMM just for the background contours
    gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm2.fit(Z2)

    xmin, ymin = Z2.min(axis=0) - 0.5
    xmax, ymax = Z2.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = -gmm2.score_samples(grid)  # negative log-likelihood for contours

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz.reshape(xx.shape), levels=20, alpha=0.25)

    # Colors by health score + outline by label
    if scores is None:
        scores = df_features['health_score'].values
    sc = plt.scatter(
        Z2[:, 0], Z2[:, 1],
        c=scores, cmap=cmap_name, s=70,
        edgecolors=np.where(final_labels=='good', 'black', 'white'),
        linewidths=1.0
    )
    cb = plt.colorbar(sc)
    cb.set_label("Health score (higher = healthier)")

    # Optional: annotate each point with filename stem
    for i, fname in enumerate(df_features.index):
        plt.annotate(Path(fname).stem, (Z2[i, 0], Z2[i, 1]), fontsize=8, alpha=0.8)

    # Legend proxy for outlines
    import matplotlib.lines as mlines
    good_proxy = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='good outline')
    bad_proxy  = mlines.Line2D([], [], color='white', marker='o', linestyle='None', markeredgecolor='black', label='bad outline')
    plt.legend(handles=[good_proxy, bad_proxy], title="Cluster label outline", loc="best")

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Battery health clustering (PCA + GMM) — colored by health score")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ------------------------------ Main ------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Auto-label battery health with cycle sampling for huge datasets."
    )
    ap.add_argument("--input_glob", type=str, required=True,
                    help="Glob for Excel files, e.g., '/data/*.xlsx'")
    ap.add_argument("--out_csv", type=str, default="battery_labels.csv",
                    help="Output CSV path")
    ap.add_argument("--method", type=str, choices=['score', 'gmm', 'both'], default='both',
                    help="Labeling method")
    ap.add_argument("--plot_out", type=str, default=None,
                    help="Optional PNG path for 2D plot")
    ap.add_argument("--cycles", type=str, default=None,
                    help="Cycle range '4-10' (inclusive).")
    ap.add_argument("--every_n", type=int, default=None,
                    help="Sample every n-th cycle (after formation).")
    ap.add_argument("--sheet", type=str, default="record",
                    help="Worksheet name to read (default: 'record'). Use 'auto' to search for the right sheet.")

    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    cycles_range = parse_cycles_arg(args.cycles) if args.cycles else None

    rows = []
    for f in files:
        try:
            # df = pd.read_excel(f)
            df = load_excel_sheet_safely(f, sheet=args.sheet)
            df = df[df['Cycle Index'] > 3].copy()

            # Validate columns
            for col in REQUIRED_COLS:
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' in {f}")

            # Choose cycles for this file
            unique_cycles = pd.unique(df['Cycle Index'])
            chosen = select_cycles(unique_cycles, cycles_range=cycles_range, every_n=args.every_n)
            if not chosen:
                print(f"[WARN] No chosen cycles in {f} (after filtering). Skipping.")
                continue

            per_cycle = summarize_features(df, chosen)
            feats = aggregate_features(per_cycle)
            if feats is None:
                print(f"[WARN] No usable data in {f}")
                continue

            feats['file'] = f
            rows.append(feats)

        except Exception as e:
            print(f"[WARN] Failed to process {f}: {e}")
            continue

    if not rows:
        raise SystemExit("No features extracted. Aborting.")

    feat_df = pd.DataFrame(rows).set_index('file')

    # Composite health score & label
    score = compute_health_score(feat_df)
    feat_df['health_score'] = score
    median_thr = float(np.median(score))
    feat_df['label_score'] = np.where(score >= median_thr, 'good', 'bad')

    # GMM labels
    gmm_labels, _ = label_by_gmm(feat_df, score)
    feat_df['label_gmm'] = gmm_labels

    # Final label
    if args.method == 'score':
        feat_df['label'] = feat_df['label_score']
    elif args.method == 'gmm':
        feat_df['label'] = feat_df['label_gmm']
    else:
        agree = feat_df['label_score'] == feat_df['label_gmm']
        feat_df['label'] = np.where(agree, feat_df['label_score'], feat_df['label_score'])

    # Save results
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_csv, index=True)
    print(f"[INFO] Saved labels to: {out_csv}")

    # Optional plot
    # if args.plot_out:
    #     plot_path = Path(args.plot_out)
    #     plot_path.parent.mkdir(parents=True, exist_ok=True)
    #     make_plot(feat_df, feat_df['label'].values, str(plot_path))
    #     print(f"[INFO] Saved plot to: {plot_path}")

    if args.plot_out:
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        make_plot(feat_df, feat_df['label'].values, str(plot_path))
        print(f"[INFO] Saved plot to: {plot_path}")

    # Console summary
    print(feat_df[['health_score', 'label_score', 'label_gmm', 'label']].to_string())


if __name__ == "__main__":
    main()
