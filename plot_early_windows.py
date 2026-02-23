#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Accuracy and Macro-F1 vs Early Window using all_models_summary.csv

Expected CSV columns:
    model, feature_set, accuracy, macro_f1

Typical feature_set values:
    full, no_early,
    early5_only, early10_only, early20_only, early30_only, early50_only, early60_only
"""

import re
import pandas as pd
import matplotlib.pyplot as plt

font_size = 24
rc = {"text.usetex": True, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (10, 8),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 2, 'lines.linewidth': 4.5, "axes.linewidth":4.5,
          'axes.axisbelow': True}
plt.rcParams.update(rc)

# ==== CONFIG ====
# CSV_PATH = "./models_out_balanced/all_models_summary_rev.csv"
# CSV_PATH = "./models_out_balanced/all_models_summary (copy).csv"      # path to your summary csv
CSV_PATH = "./models_out_balanced/all_models_summary_opt.csv"      # path to your summary csv
OUT_PREFIX = "early_window_"             # prefix for saved plot files
MODELS_ORDER = ["RandomForest", "XGBoost", "MLP"]  # for legend ordering


def main():
    # ---- Load data ----
    df = pd.read_csv(CSV_PATH)

    # ---- Separate baseline and early-window rows ----
    # Baselines: full & no_early
    baseline = df[df["feature_set"].isin(["full", "no_early"])].copy()

    # Early windows: entries like "early5_only", "early10_only", ...
    early_mask = df["feature_set"].str.contains(r"^early\d+_only$")
    early = df[early_mask].copy()

    # Extract numeric window (seconds) from feature_set (e.g., "early20_only" -> 20)
    early["window_s"] = (
        early["feature_set"]
        .str.extract(r"early(\d+)_only", expand=False)
        .astype(int)
    )

    # Sort by window for nice lines
    early = early.sort_values("window_s")

    # ---------------- Plot: Accuracy vs Early Window ----------------
    plt.figure(figsize=(10, 6))

    for model in MODELS_ORDER:
        sub = early[early["model"] == model]
        if sub.empty:
            continue
        plt.plot(
            sub["window_s"],
            sub["accuracy"],
            marker="o",
            ms=15,
            linestyle="-",
            label=model,
        )

    # Optional: overlay full & no_early baselines as horizontal dashed lines (RF only, for reference)
    # You can comment this block out if you don't want baselines.
    # for fs, color, label_suffix in [("full", "gray", " (full)"),
    #                                 ("no_early", "black", " (no_early)")]:
    #     bsub = baseline[(baseline["feature_set"] == fs) &
    #                     (baseline["model"] == "RandomForest")]
    #     if not bsub.empty:
    #         acc = float(bsub["accuracy"].iloc[0])
    #         plt.axhline(
    #             acc,
    #             linestyle="--",
    #             color=color,
    #             linewidth=1,
    #             alpha=0.7,
    #             label=f"RF {fs}"
    #         )

    plt.xlabel("Early Window (seconds)", fontsize=24)
    plt.ylabel("Accuracy", fontsize=24)
    plt.title("Accuracy vs Early Window for synthetic dataset", fontsize=24)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.ylim(0.80, 0.92)
    plt.legend(loc="best", frameon=True, fontsize=24)
    plt.tight_layout()
    # plt.savefig(OUT_PREFIX + "accuracy.png", dpi=300)
    plt.show()

    # # ---------------- Plot: Macro-F1 vs Early Window ----------------
    plt.figure(figsize=(10, 6))

    for model in MODELS_ORDER:
        sub = early[early["model"] == model]
        if sub.empty:
            continue
        plt.plot(
            sub["window_s"],
            sub["macro_f1"],
            marker="o",
            ms=15,
            linestyle="-",
            label=model,
        )
    #
    # # Optional baselines (macro-F1) for RF
    # for fs, color in [("full", "gray"), ("no_early", "black")]:
    #     bsub = baseline[(baseline["feature_set"] == fs) &
    #                     (baseline["model"] == "RandomForest")]
    #     if not bsub.empty:
    #         f1 = float(bsub["macro_f1"].iloc[0])
    #         plt.axhline(
    #             f1,
    #             linestyle="--",
    #             color=color,
    #             linewidth=1,
    #             alpha=0.7,
    #             label=f"RF {fs}"
    #         )
    #
    plt.xlabel("Early Window (seconds)", fontsize=24)
    plt.ylabel("Macro F1", fontsize=24)
    plt.title("Macro F1 vs Early Window for synthetic dataset", fontsize=24)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.ylim(0.78, 0.92)
    plt.legend(loc="best", frameon=True, fontsize=24)
    plt.tight_layout()
    # # plt.savefig(OUT_PREFIX + "macro_f1.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
