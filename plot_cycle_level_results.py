# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Plot cycle-level model comparisons from models_out/*.csv
# """
#
import os
import re
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--metrics_dir", required=True,
#                     help="Folder containing metrics_*.csv from train_models_3class.py")
#     ap.add_argument("-o", "--outdir", default=None)
#     args = ap.parse_args()
#
#     outdir = args.outdir or args.metrics_dir
#     os.makedirs(outdir, exist_ok=True)
#
#     # Load all metrics files
#     files = glob.glob(os.path.join(args.metrics_dir, "metrics_*.csv"))
#     df_list = [pd.read_csv(f) for f in files]
#     df = pd.concat(df_list, ignore_index=True)
#
#     # --- Accuracy comparison ---
#     plt.figure(figsize=(10,6))
#     sns.barplot(data=df, x="feature_set", y="accuracy", hue="model")
#     plt.xticks(rotation=45, ha="right")
#     plt.title("Cycle-Level Accuracy Comparison")
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, "cycle_accuracy_comparison.png"))
#     plt.close()
#
#     # --- Macro-F1 comparison ---
#     plt.figure(figsize=(10,6))
#     sns.barplot(data=df, x="feature_set", y="macro_f1", hue="model")
#     plt.xticks(rotation=45, ha="right")
#     plt.title("Cycle-Level Macro-F1 Comparison")
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, "cycle_f1_comparison.png"))
#     plt.close()
#
#     print(f"[OK] Saved plots into {outdir}")
#
#
# if __name__ == "__main__":
#     main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load results
# -----------------------------
csv_path = "models_out_balanced/all_models_summary (copy).csv"
df = pd.read_csv(csv_path)

# -----------------------------
# Normalize + filter feature sets
# -----------------------------
# Drop early_plus_base (and any variants that contain it)
df = df[~df["feature_set"].astype(str).str.contains("early_plus_base", case=False, na=False)].copy()

# Keep only: full, no_early, early<number>_only
early_pat = re.compile(r"^early(\d+)_only$", re.IGNORECASE)

def is_allowed(fs: str) -> bool:
    if fs in ("full", "no_early"):
        return True
    return bool(early_pat.match(fs))

df = df[df["feature_set"].astype(str).apply(is_allowed)].copy()

# -----------------------------
# Build an ordered x-axis
# -----------------------------
def feature_key(fs: str):
    # Sort full, no_early first, then early windows numerically
    if fs == "full":
        return (0, 0)
    if fs == "no_early":
        return (1, 0)
    m = early_pat.match(fs)
    if m:
        return (2, int(m.group(1)))
    return (9, 9999)

ordered_feature_sets = sorted(df["feature_set"].unique(), key=feature_key)
df["feature_set"] = pd.Categorical(df["feature_set"], categories=ordered_feature_sets, ordered=True)

# Optional: prettier labels on x-axis
def pretty_label(fs: str) -> str:
    if fs == "full":
        return "full"
    if fs == "no_early":
        return "no_early"
    m = early_pat.match(fs)
    if m:
        return f"early{m.group(1)}_only"
    return fs

df["feature_set_label"] = df["feature_set"].astype(str).map(pretty_label)
df["feature_set_label"] = pd.Categorical(df["feature_set_label"],
                                         categories=[pretty_label(x) for x in ordered_feature_sets],
                                         ordered=True)

# -----------------------------
# Column names safety
# -----------------------------
# If your CSV uses different metric column names, adjust here.
ACC_COL = "accuracy"
F1_COL  = "macro_f1"

missing = [c for c in ["model", "feature_set", ACC_COL, F1_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}\nAvailable columns: {list(df.columns)}")

# -----------------------------
# Plot settings
# -----------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold"
})

# -----------------------------
# Accuracy bar plot
# -----------------------------
plt.figure(figsize=(12, 4.8))
ax = sns.barplot(
    data=df,
    x="feature_set_label",
    y=ACC_COL,
    hue="model",
    errorbar=None
)
ax.set_xlabel("Feature Set")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Across Feature Sets (full, no_early, earlyX_only)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
# plt.savefig("accuracy_barplot.png", dpi=300)
plt.show()

# -----------------------------
# F1 bar plot
# -----------------------------
plt.figure(figsize=(12, 4.8))
ax = sns.barplot(
    data=df,
    x="feature_set_label",
    y=F1_COL,
    hue="model",
    errorbar=None
)
ax.set_xlabel("Feature Set")
ax.set_ylabel("F1-score")
ax.set_title("F1-score Across Feature Sets (full, no_early, earlyX_only)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
# plt.savefig("f1_barplot.png", dpi=300)
plt.show()