import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

font_size = 16
rc = {"text.usetex": True, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (10, 8),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 3, 'lines.linewidth': 4.5, "axes.linewidth":3,
          'axes.axisbelow': True}
plt.rcParams.update(rc)

# -----------------------------
# User settings
# -----------------------------

# CSV_PATH = "./all_csv_for_training/ALL_cycles_3class_filtered.csv"   # change if needed
CSV_PATH = "./all_csv_for_training/ALL_cycles_3class_filtered_balanced.csv"
OUT_DIR = "./Figures/csv_training_variation_plots"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL = "cycle_label_3name"   # or "cycle_label_3class"
CYCLE_COL = "Cycle"

# 1) Pick a compact set of informative features
#    You can edit this list based on what you want to emphasize.
candidate_features = [
    "IR_proxy", "CE",
    "Chg_Cap_Ah", "DChg_Cap_Ah",
    "Chg_Energy_Wh", "DChg_Energy_Wh",
]

# Add mission-aware early features automatically if present
# Example columns include takeoff_*, hover_*, cruise_*, landing_*, steady_*
def pick_mission_features(cols, max_per_phase=2):
    phases = ["takeoff_", "hover_", "cruise_", "landing_", "steady_"]
    picked = []
    for ph in phases:
        ph_cols = [c for c in cols if c.startswith(ph)]
        # prefer slope-like or delta-like features if they exist
        preferred = [c for c in ph_cols if ("slope" in c.lower() or "dvdt" in c.lower() or "delta" in c.lower())]
        use = preferred[:max_per_phase] if preferred else ph_cols[:max_per_phase]
        picked.extend(use)
    return picked

df = pd.read_csv(CSV_PATH)

# Keep only rows with label and cycle
df = df.dropna(subset=[LABEL_COL, CYCLE_COL]).copy()

mission_feats = pick_mission_features(df.columns, max_per_phase=2)
features = [f for f in candidate_features if f in df.columns]

# Drop features that are entirely missing
features = [f for f in features if df[f].notna().any()]

print("Selected features for plots:")
for f in features:
    print("  ", f)

print("Class counts before filtering:")
print(df[LABEL_COL].value_counts())
#
print("\nClass counts after feature selection but before dropna:")
print(df[[LABEL_COL] + features][LABEL_COL].value_counts())

print("\nRows with any NaN per class:")
print(df.groupby(LABEL_COL)[features].apply(lambda x: x.isna().any(axis=1).sum()))

# Clean dataframe to numeric features
plot_df = df[[CYCLE_COL, LABEL_COL] + features].copy()

# Clean label strings (important if there are trailing spaces)
plot_df[LABEL_COL] = plot_df[LABEL_COL].astype(str).str.strip()

# Convert features to numeric
for f in features:
    plot_df[f] = pd.to_numeric(plot_df[f], errors="coerce")

print("Class counts in plot_df (before any filtering):")
print(plot_df[LABEL_COL].value_counts(dropna=False))

# classes = sorted([c for c in plot_df[LABEL_COL].unique().tolist() if c not in ["nan", "None"]])

DESIRED_ORDER = ["good_drone", "good_not_drone", "bad"]

classes = [c for c in DESIRED_ORDER if c in plot_df[LABEL_COL].unique()]


# Fixed color mapping for consistency across figures
# You can reorder if you want a specific color per class
# cmap = plt.get_cmap("tab10")
# class_to_color = {cls: cmap(i) for i, cls in enumerate(classes)}

class_to_color = {
    "good_drone": "green",
    "good_not_drone": "blue",
    "bad": "red",
}


# -----------------------------
# Plot A: Distributions by class (boxplots), feature-wise NaN handling
# -----------------------------
n_rows, n_cols = 2, 3
fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(14,8),
    sharey=False
)

axs = axs.flatten()  # make it iterable

for ax, feat in zip(axs, features):
    data = []
    used_labels = []
    used_colors = []

    for cls in classes:
        vals = plot_df.loc[plot_df[LABEL_COL] == cls, feat].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        used_labels.append(cls)
        used_colors.append(class_to_color[cls])

    bp = ax.boxplot(data, labels=used_labels,
                    patch_artist=True, showfliers=False)

    for patch, col in zip(bp["boxes"], used_colors):
        patch.set_facecolor(col)

    ax.set_title(feat, fontweight="bold", fontsize=20)
    ax.tick_params(axis="x", rotation=8)
    ax.grid(False)

# Hide any unused axes (safety, if features < 6)
for ax in axs[len(features):]:
    ax.axis("off")

# fig.suptitle(
#     "Training Data Variation by Class (Cycle-level features)",
#     fontweight="bold", fontsize=22, y=1.02
# )

fig.tight_layout()
plt.show()



# # -----------------------------
# # Plot B: Feature vs Cycle trends (scatter), feature-wise NaN handling
# # -----------------------------
# trend_feats = [f for f in ["IR_proxy", "CE", "DChg_Cap_Ah", "DChg_Energy_Wh"] if f in features]
# if len(trend_feats) == 0:
#     trend_feats = features[:3]
#
# fig, axs = plt.subplots(len(trend_feats), 1, figsize=(10, 3.2 * len(trend_feats)), sharex=True)
# if len(trend_feats) == 1:
#     axs = [axs]
#
# for ax, feat in zip(axs, trend_feats):
#     for cls in classes:
#         sub = plot_df[plot_df[LABEL_COL] == cls][[CYCLE_COL, feat]].dropna().sort_values(CYCLE_COL)
#         if sub.empty:
#             continue
#         ax.scatter(sub[CYCLE_COL], sub[feat], s=12, alpha=0.55, label=cls, color=class_to_color[cls])
#
#     ax.set_ylabel(feat, fontweight="bold")
#     ax.grid(False)
#
# axs[-1].set_xlabel("Cycle", fontweight="bold")
# axs[0].legend(frameon=False, ncol=3, loc="best")
#
# fig.suptitle("Feature trends across cycles (colored by class)", fontweight="bold", y=1.02)
# fig.tight_layout()
# plt.show()


# # -----------------------------
# # Plot C: PCA scatter (2D separability), use robust feature subset
# # -----------------------------
# # PCA requires complete rows. Since BAD has NaNs in mission-end features,
# # do PCA using a robust subset that exists for BAD.
# pca_features = [f for f in ["IR_proxy", "CE", "Chg_Cap_Ah", "DChg_Cap_Ah", "Chg_Energy_Wh", "DChg_Energy_Wh"] if f in features]
#
# pca_df = plot_df[[LABEL_COL] + pca_features].dropna().copy()
# print("\nClass counts used for PCA:")
# print(pca_df[LABEL_COL].value_counts())
#
# X = pca_df[pca_features].values
# y = pca_df[LABEL_COL].values
#
# X_scaled = StandardScaler().fit_transform(X)
# pca = PCA(n_components=2, random_state=0)
# X_pca = pca.fit_transform(X_scaled)
#
# fig = plt.figure(figsize=(8, 6))
#
# for cls in classes:
#     m = (y == cls)
#     if np.sum(m) == 0:
#         continue
#     plt.scatter(
#         X_pca[m, 0], X_pca[m, 1],
#         s=21, alpha=0.6,
#         label=cls,
#         color=class_to_color[cls]
#     )
#
# plt.xlabel("PCA1", fontweight="bold", fontsize=22)
# plt.ylabel("PCA2", fontweight="bold", fontsize=22)
#
# # FIXED AXIS RANGES
# plt.xlim(-2.5, 2)
# plt.ylim(-2, 2)
#
# plt.legend(frameon=False, fontsize=18, markerscale=3.5)
# plt.grid(False)
# plt.tight_layout()
# plt.show()


# # -----------------------------
# # Plot D: Class mean signature (standardized), feature-wise means
# # -----------------------------
# # Standardize using the PCA dataframe (complete rows), so scaling is stable
# scaled_df = pd.DataFrame(X_scaled, columns=pca_features)
# scaled_df[LABEL_COL] = y
#
# means = scaled_df.groupby(LABEL_COL)[pca_features].mean().reindex(classes)
#
# fig = plt.figure(figsize=(12, 4))
# x = np.arange(len(pca_features))
# width = 0.25 if len(classes) == 3 else 0.8 / max(1, len(classes))
#
# for i, cls in enumerate(classes):
#     if cls not in means.index:
#         continue
#     plt.bar(x + (i - (len(classes)-1)/2)*width, means.loc[cls].values, width=width, label=cls, color=class_to_color[cls])
#
# plt.xticks(x, pca_features, rotation=35, ha="right")
# plt.ylabel("Standardized mean", fontweight="bold")
# plt.title("Class mean feature signature (standardized)", fontweight="bold")
# plt.legend(frameon=False)
# plt.grid(False)
# plt.tight_layout()
# plt.show()

# fig.savefig(os.path.join(OUT_DIR, "D_class_mean_signature.png"), dpi=300, bbox_inches="tight")
# plt.close(fig)

# print(f"Saved plots to: {OUT_DIR}")
