
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def summarize_features_fixed(df, cycles=(4,5)):
    df = df.copy()
    # Ensure expected columns exist
    expected = ['Cycle Index','Time(h)','Current(A)','Voltage(V)','Chg. Cap.(Ah)','DChg. Cap.(Ah)']
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input file.")
    # Derive stats per cycle
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
        # Energy delivered during discharge (Wh), computed via trapezoid on V*|I| over time
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
    """Aggregate to a single vector per file; include means and deltas across cycles 4->5, plus polarization."""
    if summary_df.shape[0] == 0:
        return None
    out = {}
    for k in ['dchg_cap_Ah','chg_cap_Ah','coul_eff','v_dis_avg','v_chg_avg','t_dis_h','t_chg_h','i_dis_A','i_chg_A','energy_dis_Wh']:
        out[f'mean_{k}'] = summary_df[k].mean()
        out[f'delta_{k}'] = summary_df[k].iloc[-1] - summary_df[k].iloc[0]
    pol = summary_df['v_chg_avg'] - summary_df['v_dis_avg']
    out['mean_polarization_V'] = float(pol.mean())
    out['delta_polarization_V'] = float(pol.iloc[-1] - pol.iloc[0])
    return out

def compute_health_score(df_features):
    """Compute a composite score: +capacity, +energy, +discharge voltage, +discharge time, -polarization.
       Features are standardized first."""
    # Select features for scoring
    pos_feats = ['mean_dchg_cap_Ah','mean_energy_dis_Wh','mean_v_dis_avg','mean_t_dis_h']
    neg_feats = ['mean_polarization_V']
    use_feats = pos_feats + neg_feats
    # Standardize
    X = df_features[use_feats].values
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    # Score = sum positive z's - sum negative z's
    score = Z[:, :len(pos_feats)].sum(axis=1) - Z[:, len(pos_feats):].sum(axis=1)
    return score

def label_by_gmm(df_features, score, random_state=0):
    """Optional 2-cluster GMM; map the higher-score cluster to GOOD."""
    X = df_features[['mean_dchg_cap_Ah','mean_energy_dis_Wh','mean_v_dis_avg','mean_t_dis_h','mean_polarization_V']].values
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state)
    gmm_labels = gmm.fit_predict(X)
    # Decide which cluster is GOOD: compare mean composite score
    mean_scores = [float(np.mean(score[gmm_labels == k])) for k in [0,1]]
    good_cluster = int(np.argmax(mean_scores))
    labels = np.where(gmm_labels == good_cluster, 'good', 'bad')
    return labels

def main():
    parser = argparse.ArgumentParser(description="Auto-label battery health (good/bad) from Excel cycle data (cycles 4 & 5).")
    parser.add_argument("--input_glob", type=str, required=True, help="Glob pattern for Excel files, e.g., '/data/*.xlsx'")
    parser.add_argument("--out_csv", type=str, default="battery_labels.csv", help="Path to save labels CSV")
    parser.add_argument("--method", type=str, choices=['score','gmm','both'], default='both', help="Labeling method")
    args = parser.parse_args()

    import glob
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    rows = []
    for f in files:
        try:
            df = pd.read_excel(f)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
            continue
        try:
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
    # Compute composite score
    score = compute_health_score(feat_df)
    feat_df['health_score'] = score

    # Threshold by median for a deterministic split if using 'score'
    median_thr = float(np.median(score))
    feat_df['label_score'] = np.where(score >= median_thr, 'good', 'bad')

    # Optional GMM clustering
    gmm_labels = label_by_gmm(feat_df, score)
    feat_df['label_gmm'] = gmm_labels

    # Final label
    if args.method == 'score':
        feat_df['label'] = feat_df['label_score']
    elif args.method == 'gmm':
        feat_df['label'] = feat_df['label_gmm']
    else:
        # Combine: if both agree, use it; else fall back to higher score
        agree = feat_df['label_score'] == feat_df['label_gmm']
        feat_df['label'] = np.where(agree, feat_df['label_score'], feat_df['label_score'])

    feat_df.to_csv(args.out_csv, index=True)
    print(f"[INFO] Saved labels to: {args.out_csv}")
    print(feat_df[['health_score','label_score','label_gmm','label']].to_string())

if __name__ == "__main__":
    main()
