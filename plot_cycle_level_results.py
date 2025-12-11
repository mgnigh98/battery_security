#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot cycle-level model comparisons from models_out/*.csv
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", required=True,
                    help="Folder containing metrics_*.csv from train_models_3class.py")
    ap.add_argument("-o", "--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or args.metrics_dir
    os.makedirs(outdir, exist_ok=True)

    # Load all metrics files
    files = glob.glob(os.path.join(args.metrics_dir, "metrics_*.csv"))
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    # --- Accuracy comparison ---
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="feature_set", y="accuracy", hue="model")
    plt.xticks(rotation=45, ha="right")
    plt.title("Cycle-Level Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cycle_accuracy_comparison.png"))
    plt.close()

    # --- Macro-F1 comparison ---
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="feature_set", y="macro_f1", hue="model")
    plt.xticks(rotation=45, ha="right")
    plt.title("Cycle-Level Macro-F1 Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cycle_f1_comparison.png"))
    plt.close()

    print(f"[OK] Saved plots into {outdir}")


if __name__ == "__main__":
    main()
