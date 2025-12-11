
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_COLS = ['Cycle Index','Time(h)','Current(A)','Voltage(V)','Chg. Cap.(Ah)','DChg. Cap.(Ah)']

def summarize_features_fixed(df, cycles=(4,5)):
    df = df.copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input file.")
    summary = {}
    for c in cycles:
        cdf = df[df['Cycle Index'] == c]
        if cdf.empty:
            continue
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

        # Discharge energy (Wh) via trapezoidal rule on V*|I| vs time (time already in hours)
        if not dchg.empty:
            dchgs = dchg.sort_values('Time(h)')
            energy_wh = float(np.trapz(dchgs['Voltage(V)'] * dchgs['Current(A)'].abs(), dchgs['Time(h)']))
            energy_wh = abs(energy_wh)
        else:
            energy_wh = np.nan

        summary[c] = dict(
            chg_cap_Ah=chg_cap, dchg_cap_Ah=dchg_cap, coul_eff=ce,
            v_dis_avg=v_dis_avg, v_chg_avg=v_chg_avg,
            t_dis_h=t_dis, t_chg_h=t_chg,
            i_dis_A=i_dis, i_chg_A=i_chg,
            energy_dis_Wh=energy_wh
        )
    return pd.DataFrame(summary).T

def feature_vector(summary_df):
    if summary_df.shape[0] == 0:
        return None
    out = {}
    keys = ['dchg_cap_Ah','chg_cap_Ah','coul_eff','v_dis_avg','v_chg_avg','t_dis_h','t_chg_h','i_dis_A','i_chg_A','energy_dis_Wh']
    for k in keys:
        out[f'mean_{k}'] = summary_df[k].mean()
        out[f'delta_{k}'] = summary_df[k].iloc[-1] - summary_df[k].iloc[0]
    pol = summary_df['v_chg_avg'] - summary_df['v_dis_avg']
    out['mean_polarization_V']   = float(pol.mean())
    out['delta_polarization_V']  = float(pol.iloc[-1] - pol.iloc[0])
    return out

def compute_health_score(df_features):
    pos_feats = ['mean_dchg_cap_Ah','mean_energy_dis_Wh','mean_v_dis_avg','mean_t_dis_h']
    neg_feats = ['mean_polarization_V']
    use_feats = pos_feats + neg_feats

    X = df_features[use_feats].values
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    score = Z[:, :len(pos_feats)].sum(axis=1) - Z[:, len(pos_feats):].sum(axis=1)
    return score

def label_by_gmm(df_features, score, random_state=0):
    cols = ['mean_dchg_cap_Ah','mean_energy_dis_Wh','mean_v_dis_avg','mean_t_dis_h','mean_polarization_V']
    X = df_features[cols].values
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state)
    gmm_labels = gmm.fit_predict(X)

    # Higher composite health score cluster => GOOD
    mean_scores = [float(np.mean(score[gmm_labels == k])) for k in [0,1]]
    good_cluster = int(np.argmax(mean_scores))
    labels = np.where(gmm_labels == good_cluster, 'good', 'bad')
    return labels, gmm

def make_plot(df_features, final_labels, gmm, out_png):
    # 2D PCA projection for visualization only
    cols = ['mean_dchg_cap_Ah','mean_energy_dis_Wh','mean_v_dis_avg','mean_t_dis_h','mean_polarization_V']
    X = df_features[cols].values
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z2 = pca.fit_transform(Z)

    # Fit GMM in 2D for contour visualization
    gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm2.fit(Z2)

    xmin, ymin = Z2.min(axis=0) - 0.5
    xmax, ymax = Z2.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = -gmm2.score_samples(grid)  # negative log-likelihood

    plt.figure(figsize=(7,6))
    CS = plt.contourf(xx, yy, zz.reshape(xx.shape), levels=20, alpha=0.3)
    # Plot points: good vs bad
    mask_good = (final_labels == 'good')
    mask_bad  = (final_labels == 'bad')
    plt.scatter(Z2[mask_good,0], Z2[mask_good,1], label="good")
    plt.scatter(Z2[mask_bad,0],  Z2[mask_bad,1],  label="bad")
    for i, fname in enumerate(df_features.index):
        plt.annotate(Path(fname).stem, (Z2[i,0], Z2[i,1]))

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Battery health clustering (PCA projection + GMM contours)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Auto-label battery health (good/bad) from Excel cycle data with optional GMM plot.")
    ap.add_argument("--input_glob", type=str, required=True, help="Glob pattern for Excel files, e.g., '/data/*.xlsx'")
    ap.add_argument("--out_csv", type=str, default="battery_labels.csv", help="Path to save labels CSV")
    ap.add_argument("--method", type=str, choices=['score','gmm','both'], default='both', help="Labeling method")
    ap.add_argument("--plot_out", type=str, default=None, help="Optional PNG path to save a 2D GMM plot")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    rows = []
    for f in files:
        try:
            df = pd.read_excel(f)
            summary = summarize_features_fixed(df, cycles=(4,5))
            feats = feature_vector(summary)
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

    # GMM
    gmm_labels, gmm = label_by_gmm(feat_df, score)
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
    if args.plot_out:
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        make_plot(feat_df, feat_df['label'].values, gmm, str(plot_path))
        print(f"[INFO] Saved plot to: {plot_path}")

    # Print compact summary
    print(feat_df[['health_score','label_score','label_gmm','label']].to_string())

if __name__ == "__main__":
    main()
