# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ----------------------------
# # USER SETTINGS
# # ----------------------------
# DATA_DIR = "./data"         # folder that contains many cell Excel files
# FILE_GLOB = "*.xlsx"           # or a narrower pattern, eg "GraSi NMC811*.xlsx"
#
# SHEET = "record"               # time series sheet
# CYCLE_START = 4                # skip preparation cycles 1-3
# MAX_CYCLES_PER_CELL = 20       # overlay at most 20 cycles per cell
# RANDOM_SEED = 7
# N_CELLS_TO_PLOT = 3
#
# # Optional truncation per cycle (seconds). Set to None for full cycle length.
# TMAX_PER_CYCLE_S = 2000        # eg 1200 if you want only first 1200 s of each cycle
#
# # Plot styling (good for sparse sampling like 10 s)
# USE_MARKERS = True
# MARKER = "."
# MARKER_SIZE = 2
# LINEWIDTH = 0.9
# ALPHA = 0.35
#
# OUT_PATH = os.path.join(DATA_DIR, "training_set_cycles_overlay_3cells.png")
#
# # ----------------------------
# # Columns expected in record sheet
# # ----------------------------
# COLS = [
#     "Cycle Index",
#     "Time(h)",
#     "Voltage(V)",
#     "Current(A)",
#     "Power(W)",
#     "Energy(Wh)",
#     "Chg. Cap.(Ah)",
#     "DChg. Cap.(Ah)",
# ]
#
# def load_record_df(path: str) -> pd.DataFrame:
#     df = pd.read_excel(path, sheet_name=SHEET, usecols=COLS)
#     df = df.dropna(subset=["Cycle Index", "Time(h)"]).copy()
#
#     df["Cycle Index"] = pd.to_numeric(df["Cycle Index"], errors="coerce")
#     df["Time(h)"] = pd.to_numeric(df["Time(h)"], errors="coerce")
#     df = df.dropna(subset=["Cycle Index", "Time(h)"]).copy()
#
#     df["Cycle Index"] = df["Cycle Index"].astype(int)
#     df["t_s"] = df["Time(h)"] * 3600.0
#     return df
#
# def pick_cycles(df: pd.DataFrame) -> list[int]:
#     cycles = sorted(df["Cycle Index"].unique().tolist())
#     cycles = [c for c in cycles if c >= CYCLE_START]
#     if len(cycles) > MAX_CYCLES_PER_CELL:
#         # Uniform sampling across the available range for readability
#         idx = np.linspace(0, len(cycles) - 1, MAX_CYCLES_PER_CELL).round().astype(int)
#         cycles = [cycles[i] for i in idx]
#     return cycles
#
# def plot_cell_row(axs_row, df, cycles):
#     """
#     axs_row: list of 5 axes (Voltage, Current, Power, Energy, Capacity)
#     Overlays selected cycles, each cycle aligned to its own start time (t=0).
#     """
#     axV, axI, axP, axE, axC = axs_row
#     marker = MARKER if USE_MARKERS else None
#
#     for cyc in cycles:
#         d = df[df["Cycle Index"] == cyc].copy()
#         if d.empty:
#             continue
#
#         d = d.sort_values("t_s")
#         t = d["t_s"].to_numpy()
#         t = t - t[0]  # cycle-relative time
#
#         if TMAX_PER_CYCLE_S is not None:
#             keep = t <= TMAX_PER_CYCLE_S
#             if not np.any(keep):
#                 continue
#             d = d.iloc[np.where(keep)[0]]
#             t = t[keep]
#
#         axV.plot(t, d["Voltage(V)"].to_numpy(), linewidth=LINEWIDTH, alpha=ALPHA,
#                  marker=marker, markersize=MARKER_SIZE)
#         axI.plot(t, d["Current(A)"].to_numpy(), linewidth=LINEWIDTH, alpha=ALPHA,
#                  marker=marker, markersize=MARKER_SIZE)
#         axP.plot(t, d["Power(W)"].to_numpy(), linewidth=LINEWIDTH, alpha=ALPHA,
#                  marker=marker, markersize=MARKER_SIZE)
#         axE.plot(t, d["Energy(Wh)"].to_numpy(), linewidth=LINEWIDTH, alpha=ALPHA,
#                  marker=marker, markersize=MARKER_SIZE)
#
#         # Capacity panel: overlay both charge and discharge
#         axC.plot(t, d["Chg. Cap.(Ah)"].to_numpy(), linewidth=LINEWIDTH, alpha=ALPHA,
#                  marker=marker, markersize=MARKER_SIZE)
#         axC.plot(t, d["DChg. Cap.(Ah)"].to_numpy(), linewidth=LINEWIDTH, alpha=ALPHA,
#                  marker=marker, markersize=MARKER_SIZE)
#
# def main():
#     paths = sorted(glob.glob(os.path.join(DATA_DIR, FILE_GLOB)))
#     if not paths:
#         raise FileNotFoundError(f"No Excel files found: {os.path.join(DATA_DIR, FILE_GLOB)}")
#
#     rng = np.random.default_rng(RANDOM_SEED)
#     chosen = list(rng.choice(paths, size=min(N_CELLS_TO_PLOT, len(paths)), replace=False))
#
#     # 3 rows (cells) x 5 cols (signals)
#     fig, axs = plt.subplots(nrows=len(chosen), ncols=5, figsize=(20, 4.2 * len(chosen)), sharex=False)
#
#     if len(chosen) == 1:
#         axs = np.array([axs])
#
#     # Column titles
#     col_titles = ["Voltage (V)", "Current (A)", "Power (W)", "Energy (Wh)", "Capacity (Ah)"]
#     for c, t in enumerate(col_titles):
#         axs[0, c].set_title(t, fontweight="bold")
#
#     for r, path in enumerate(chosen):
#         base = os.path.splitext(os.path.basename(path))[0]
#         df = load_record_df(path)
#         cycles = pick_cycles(df)
#
#         plot_cell_row(axs[r, :], df, cycles)
#
#         # Row label = cell name
#         axs[r, 0].set_ylabel(base, fontweight="bold")
#
#         for c in range(5):
#             axs[r, c].set_xlabel("Time within cycle (s)", fontweight="bold")
#             axs[r, c].grid(False)
#
#     # Global title
#     cyc_info = f"Cycles {CYCLE_START}+ (sampled up to {MAX_CYCLES_PER_CELL} cycles per cell)"
#     trunc_info = f", truncated to {int(TMAX_PER_CYCLE_S)} s" if TMAX_PER_CYCLE_S is not None else ""
#     fig.suptitle(f"Training Set: Cycle-overlaid waveforms for 3 randomly selected cells | {cyc_info}{trunc_info}",
#                  fontweight="bold", y=1.01)
#
#     fig.tight_layout()
#     # fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
#     # plt.close(fig)
#     plt.show()
#
#     # print(f"Saved: {OUT_PATH}")
#     print("Cells used:")
#     for p in chosen:
#         print("  ", os.path.basename(p))
#
# if __name__ == "__main__":
#     main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

font_size = 16
rc = {"text.usetex": True, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (10, 8),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 3, 'lines.linewidth': 4.5, "axes.linewidth":3,
          'axes.axisbelow': True}
plt.rcParams.update(rc)

# -----------------------
# Settings
# -----------------------
XLSX_PATH = "./drone_labels_out/2025-06-25 GraSi NMC811 LP57 XFC15 2_6_8_2025-06-25 18-08-00_good_labeled_3class (copy).xlsx" # change to your path
# XLSX_PATH = "./drone_labels_out/2025-06-25 GraSi NMC811 LP57 1_5_5_2025-06-25 18-08-00_good_labeled_3class (copy).xlsx"
OUT_PATH = "/mnt/data/training_cycle_level_6features.png"

CYCLE_COL = "Cycle"
LABEL_COL = "cycle_label_3name"   # GOOD_drone, GOOD_not_drone, BAD

FEATURES = [
    "Chg_Spec_mAhg",
    "DChg_Spec_mAhg",
    # "Chg_Cap_Ah",
    # "DChg_Cap_Ah",
    "Chg_Energy_mWh",
    "DChg_Energy_mWh",
    "CE",
    "IR_proxy",
]

# Optional cycle range (set to None to use all)
CYCLE_MIN = 4
CYCLE_MAX = 200  # e.g., 300

# Scatter styling
POINT_SIZE = 18
ALPHA = 0.5

# Rolling trend (set window to 0 to disable)
ROLLING_WINDOW = 15  # cycles, use 0 to disable

# Fixed colors per class
CLASS_COLORS = {
    "GOOD_drone": "green",
    "GOOD_not_drone": "blue",
    "BAD": "red",
}

# -----------------------
# Load data
# -----------------------
df = pd.read_excel(XLSX_PATH)  # only one sheet
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

# Keep only needed columns
keep_cols = [CYCLE_COL, LABEL_COL] + FEATURES
df = df[keep_cols].copy()

# Numeric conversion
df[CYCLE_COL] = pd.to_numeric(df[CYCLE_COL], errors="coerce")
for f in FEATURES:
    df[f] = pd.to_numeric(df[f], errors="coerce")

df = df.dropna(subset=[CYCLE_COL, LABEL_COL]).copy()

# Apply cycle range
if CYCLE_MIN is not None:
    df = df[df[CYCLE_COL] >= CYCLE_MIN]
if CYCLE_MAX is not None:
    df = df[df[CYCLE_COL] <= CYCLE_MAX]

classes = [c for c in ["GOOD_drone", "GOOD_not_drone", "BAD"] if c in df[LABEL_COL].unique()]
if not classes:
    classes = sorted(df[LABEL_COL].unique())

# -----------------------
# Plot: 2 rows x 3 cols
# -----------------------
fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
axs = axs.flatten()

for i, feat in enumerate(FEATURES):
    ax = axs[i]
    for cls in classes:
        sub = df[df[LABEL_COL] == cls][[CYCLE_COL, feat]].dropna().sort_values(CYCLE_COL)
        if sub.empty:
            continue

        color = CLASS_COLORS.get(cls, None)
        ax.scatter(sub[CYCLE_COL], sub[feat], s=POINT_SIZE, alpha=ALPHA, label=cls, color=color)

        # # Rolling median trend per class (optional)
        # if ROLLING_WINDOW and ROLLING_WINDOW > 1:
        #     roll = sub[feat].rolling(ROLLING_WINDOW, center=True).median()
        #     ax.plot(sub[CYCLE_COL], roll, linewidth=2)

    ax.set_title(feat, fontweight="bold",fontsize=22)
    ax.grid(False)

# Axis labels
# for ax in axs[3:]:
#     ax.set_xlabel("Cycle", fontweight="bold", fontsize=24)
#
# axs[0].set_ylabel("Value", fontweight="bold",fontsize=24)
# axs[3].set_ylabel("Value", fontweight="bold",fontsize=24)

for ax in axs:
    ax.set_xlabel("")
    ax.set_ylabel("")

# Add global labels
fig.supxlabel("Cycle", fontsize=22, fontweight="bold")
fig.supylabel("Value", fontsize=22, fontweight="bold")


# One legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=min(3, len(labels)),
    frameon=False,
    fontsize=18,
    markerscale=3.5,
    handlelength=2.0,
    handletextpad=0.6,
    columnspacing=1.2
)

# fig.suptitle("Training Set Variation by Class (Cycle-level features)", fontweight="bold", y=0.98)
# fig.tight_layout(rect=[0, 0, 1, 0.93])
# fig.tight_layout(rect=[0.06, 0.06, 1, 1])
fig.tight_layout()
plt.show()
# fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
# plt.close(fig)

# print(f"Saved: {OUT_PATH}")
