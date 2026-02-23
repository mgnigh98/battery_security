#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt

EARLY_ONLY_RE = re.compile(r"^early(\d+)_only$", re.IGNORECASE)

def parse_window(fs: str):
    m = EARLY_ONLY_RE.match(str(fs))
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to all_models_summary.csv")
    ap.add_argument("--metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    ap.add_argument("--out", default=None, help="Optional output folder for PNGs")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # ----- Plot 1: curve for early-only sets -----
    df_early = df.copy()
    df_early["window_s"] = df_early["feature_set"].apply(parse_window)
    df_early = df_early[df_early["window_s"].notna()].copy()
    df_early["window_s"] = df_early["window_s"].astype(int)

    if not df_early.empty:
        plt.figure()
        for model_name, g in df_early.groupby("model"):
            g2 = g.sort_values("window_s")
            plt.plot(g2["window_s"], g2[args.metric], marker="o", label=model_name)
        plt.xlabel("Early window (s)")
        plt.ylabel(args.metric)
        plt.title(f"{args.metric} vs early window (early*_only)")
        plt.grid(True)
        plt.legend()
        if args.out:
            import os
            os.makedirs(args.out, exist_ok=True)
            plt.savefig(f"{args.out}/curve_{args.metric}_early_only.png", dpi=200, bbox_inches="tight")
        plt.show()

    # ----- Plot 2: bar chart for full/no_early + early-only best per model -----
    base_sets = ["full", "no_early"]
    df_base = df[df["feature_set"].isin(base_sets)].copy()

    if not df_base.empty:
        plt.figure()
        # pivot: rows=model, cols=feature_set
        piv = df_base.pivot_table(index="model", columns="feature_set", values=args.metric, aggfunc="mean")
        piv.plot(kind="bar")
        plt.ylabel(args.metric)
        plt.title(f"{args.metric} for full vs no_early")
        plt.grid(True)
        plt.tight_layout()
        if args.out:
            import os
            os.makedirs(args.out, exist_ok=True)
            plt.savefig(f"{args.out}/bar_{args.metric}_full_vs_no_early.png", dpi=200, bbox_inches="tight")
        plt.show()

    # ----- Print best early-only window per model -----
    if not df_early.empty:
        best = (df_early.sort_values(args.metric, ascending=False)
                .groupby("model", as_index=False)
                .first()[["model", "feature_set", "window_s", args.metric]])
        print("\nBest early-only per model:")
        print(best.to_string(index=False))

if __name__ == "__main__":
    main()
