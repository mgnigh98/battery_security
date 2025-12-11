#!/usr/bin/env python3
"""
plot_battery_pca.py

Make a PCA + GMM visualization from an existing labels.csv produced by
battery_labeler_big.py — without recomputing features.

Usage:
  python plot_battery_pca.py \
    --csv labels.csv \
    --out gmm_plot.png \
    --annotate_k 8 \
    --clip_quantile 0.0 \
    --point_size 60 \
    --alpha 0.9
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# The feature columns battery_labeler_big.py writes (adjust if you added/renamed):
FEATURE_COLS = [
    'mean_dchg_cap_Ah','mean_energy_dis_Wh','mean_v_dis_avg','mean_t_dis_h',
    'mean_polarization_V','slope_cap_per100','slope_energy_per100','slope_polar_per100'
]

def load_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only rows having all required columns
    need = FEATURE_COLS + ['file','health_score','label']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.set_index('file', drop=False)
    # Drop rows with NaNs in features
    df = df.dropna(subset=FEATURE_COLS)
    return df

def clip_outliers(df: pd.DataFrame, q: float) -> pd.DataFrame:
    if q <= 0:  # no clipping
        return df
    # clip each feature to [q, 1-q] quantiles
    dfc = df.copy()
    for c in FEATURE_COLS:
        lo, hi = dfc[c].quantile([q, 1 - q])
        dfc[c] = dfc[c].clip(lo, hi)
    return dfc

def pca_transform(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].values
    Z = StandardScaler().fit_transform(X)
    return PCA(n_components=2, random_state=0).fit_transform(Z)

def draw_gmm_contours(Z2: np.ndarray):
    gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm2.fit(Z2)
    xmin, ymin = Z2.min(axis=0) - 0.5
    xmax, ymax = Z2.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 220), np.linspace(ymin, ymax, 220))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = -gmm2.score_samples(grid)
    plt.contourf(xx, yy, zz.reshape(xx.shape), levels=24, alpha=0.25)

def annotate_extremes(df: pd.DataFrame, Z2: np.ndarray, k: int):
    if k <= 0:
        return
    # annotate top-k and bottom-k by health score
    order_hi = df['health_score'].nlargest(k).index
    order_lo = df['health_score'].nsmallest(k).index
    selected = list(dict.fromkeys(list(order_hi) + list(order_lo)))  # unique in order
    idx_map = {idx: i for i, idx in enumerate(df.index)}
    for key in selected:
        i = idx_map.get(key)
        if i is None:
            continue
        label = Path(df.loc[key, 'file']).stem
        plt.annotate(label, (Z2[i, 0], Z2[i, 1]), fontsize=8, alpha=0.9)

def main():
    ap = argparse.ArgumentParser(description="Plot PCA+GMM from labels.csv produced by battery_labeler_big.py")
    ap.add_argument("--csv", required=True, help="Path to labels.csv")
    ap.add_argument("--out", default="gmm_plot.png", help="Output PNG")
    ap.add_argument("--annotate_k", type=int, default=8, help="Annotate top/bottom K by health score (0 to disable)")
    ap.add_argument("--clip_quantile", type=float, default=0.0, help="Symmetric feature clipping quantile, e.g. 0.01")
    ap.add_argument("--point_size", type=int, default=60, help="Scatter marker size")
    ap.add_argument("--alpha", type=float, default=0.9, help="Point alpha (0–1)")
    args = ap.parse_args()

    df = load_features(args.csv)
    dfc = clip_outliers(df, args.clip_quantile)
    Z2 = pca_transform(dfc)

    plt.figure(figsize=(10, 7))
    draw_gmm_contours(Z2)

    # color by health score, outline by label
    scores = dfc['health_score'].values
    labels = dfc['label'].values
    edge = np.where(labels == 'good', 'black', 'red')

    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=scores, cmap="viridis",
                     s=args.point_size, edgecolors=edge, linewidths=1.0, alpha=args.alpha)
    cb = plt.colorbar(sc)
    cb.set_label("Health score (higher = healthier)")

    # annotate_extremes(dfc, Z2, args.annotate_k)

    # Legend proxies for outlines
    import matplotlib.lines as mlines
    n_good = (dfc['label'] == 'good').sum()
    n_bad = (dfc['label'] == 'bad').sum()

    good_proxy = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                               label=f'good ({n_good})')
    bad_proxy = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                              markeredgecolor='black', label=f'bad ({n_bad})')

    plt.legend(handles=[good_proxy, bad_proxy],
               title="Cluster label outline", loc="best")
    # good_proxy = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='good outline')
    # bad_proxy  = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markeredgecolor='black', label='bad outline')
    # plt.legend(handles=[good_proxy, bad_proxy], title="Cluster label outline", loc="best")

    plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
    plt.title("Battery health clustering (PCA + GMM) — colored by health score")
    plt.tight_layout()
    # plt.show()
    plt.savefig(args.out, dpi=300)
    plt.close()
    print(f"[INFO] Saved plot to {args.out}")

if __name__ == "__main__":
    main()
