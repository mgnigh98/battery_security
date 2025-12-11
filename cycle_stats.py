#!/usr/bin/env python3
"""
cycle_stats.py  (enhanced)

Reads *_labeled.xlsx files (created by your cycle labelling pipeline) and produces:
1) Per-file statistics CSV (one row per file)
2) Global summary CSV (aggregated across all files)
3) An all-cycles CSV (concatenated cycles with counts)
4) (Optional) Augmented labeled workbooks with extra columns
5) NEW: Distribution breakdowns for rule counts:
   - Hard_Count == 0, == 1, >= 2 (counts & percentages)
   - Soft_Count == 0, == 1, >= 2 (counts & percentages)
   - Total_Count == 0, == 1, >= 2 (counts & percentages)

Rules considered:
- Hard rules = any columns that start with "HF_" (e.g., HF_CHG_SPEC_HIGH, HF_DCHG_SPEC_LOW, HF_MISSING)
- Soft rules = any columns that start with "SP_" (e.g., SP_IR_OUTLIER, SP_CE_SOFT)

Specifically highlighted (as requested):
- H1 = HF_CHG_SPEC_HIGH  (Charge specific capacity > 190 mAh/g)
- H3 = HF_DCHG_SPEC_LOW  (Discharge specific capacity < 120 mAh/g)
- H4 = HF_MISSING        (Missing essential data points)

Usage:
  python cycle_stats.py results/  -o stats_out/  [--augment]
"""

from __future__ import annotations
import argparse, glob, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_flag_cols(df, prefix):
    return [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]

def load_cycle_labels(xlsx_path):
    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception as e:
        raise RuntimeError(f"Cannot open {xlsx_path}: {e}")
    # Support both "cycle_labels" and "labels" as sheet names
    sheet_name = None
    for cand in ["cycle_labels", "labels"]:
        if cand in xl.sheet_names:
            sheet_name = cand
            break
    if sheet_name is None:
        raise RuntimeError(f"No 'cycle_labels' or 'labels' sheet in {xlsx_path} (found: {xl.sheet_names})")
    df = pd.read_excel(xl, sheet_name=sheet_name)
    return df

def add_counts_columns(df):
    """Add Hard_Count, Soft_Count, Total_Count. Ensure H1/H3/H4 exist (default False)."""
    # Ensure the key ones exist (if not, add False)
    for k in ["HF_CHG_SPEC_HIGH","HF_DCHG_SPEC_LOW","HF_MISSING"]:
        if k not in df.columns:
            df[k] = False

    hard_cols = find_flag_cols(df, "HF_")
    soft_cols = find_flag_cols(df, "SP_")

    df["Hard_Count"] = df[hard_cols].sum(axis=1) if hard_cols else 0
    df["Soft_Count"] = df[soft_cols].sum(axis=1) if soft_cols else 0
    df["Total_Count"] = df["Hard_Count"] + df["Soft_Count"]
    return df, hard_cols, soft_cols

def dist_counts(series: pd.Series):
    """Return counts for ==0, ==1, >=2 and percentages (as tuple dict)."""
    n = len(series)
    c0 = int((series == 0).sum())
    c1 = int((series == 1).sum())
    c2p = int((series >= 2).sum())
    return {
        "n": n,
        "cnt_0": c0, "cnt_1": c1, "cnt_2p": c2p,
        "pct_0": round(100*c0/n, 2) if n else 0.0,
        "pct_1": round(100*c1/n, 2) if n else 0.0,
        "pct_2p": round(100*c2p/n, 2) if n else 0.0,
    }

def compute_per_file_stats(df):
    """Compute per-file aggregates, including distributions."""
    df, hard_cols, soft_cols = add_counts_columns(df)

    n_cycles = len(df)
    h1 = int(df["HF_CHG_SPEC_HIGH"].sum())
    h3 = int(df["HF_DCHG_SPEC_LOW"].sum())
    h4 = int(df["HF_MISSING"].sum())

    hard_any = int((df["Hard_Count"] > 0).sum())
    soft_any = int((df["Soft_Count"] > 0).sum())
    both_any = int(((df["Hard_Count"] > 0) & (df["Soft_Count"] > 0)).sum())

    avg_hard_cnt = float(df["Hard_Count"].mean()) if n_cycles else 0.0
    avg_soft_cnt = float(df["Soft_Count"].mean()) if n_cycles else 0.0
    avg_total_cnt = float(df["Total_Count"].mean()) if n_cycles else 0.0

    # Distributions
    hard_dist = dist_counts(df["Hard_Count"])
    soft_dist = dist_counts(df["Soft_Count"])
    total_dist = dist_counts(df["Total_Count"])

    per_file = {
        "file_cycles": n_cycles,
        "H1_HF_CHG_SPEC_HIGH_cnt": h1,
        "H3_HF_DCHG_SPEC_LOW_cnt": h3,
        "H4_HF_MISSING_cnt": h4,
        "hard_any_cnt": hard_any,
        "soft_any_cnt": soft_any,
        "both_any_cnt": both_any,
        "avg_Hard_Count": round(avg_hard_cnt, 3),
        "avg_Soft_Count": round(avg_soft_cnt, 3),
        "avg_Total_Count": round(avg_total_cnt, 3),
        "pct_H1": round(100*h1/n_cycles, 2) if n_cycles else 0.0,
        "pct_H3": round(100*h3/n_cycles, 2) if n_cycles else 0.0,
        "pct_H4": round(100*h4/n_cycles, 2) if n_cycles else 0.0,
        "pct_hard_any": round(100*hard_any/n_cycles, 2) if n_cycles else 0.0,
        "pct_soft_any": round(100*soft_any/n_cycles, 2) if n_cycles else 0.0,
        "pct_both_any": round(100*both_any/n_cycles, 2) if n_cycles else 0.0,
        # Hard dist
        "Hard_cnt0": hard_dist["cnt_0"],
        "Hard_cnt1": hard_dist["cnt_1"],
        "Hard_cnt2p": hard_dist["cnt_2p"],
        "Hard_pct0": hard_dist["pct_0"],
        "Hard_pct1": hard_dist["pct_1"],
        "Hard_pct2p": hard_dist["pct_2p"],
        # Soft dist
        "Soft_cnt0": soft_dist["cnt_0"],
        "Soft_cnt1": soft_dist["cnt_1"],
        "Soft_cnt2p": soft_dist["cnt_2p"],
        "Soft_pct0": soft_dist["pct_0"],
        "Soft_pct1": soft_dist["pct_1"],
        "Soft_pct2p": soft_dist["pct_2p"],
        # Total dist
        "Total_cnt0": total_dist["cnt_0"],
        "Total_cnt1": total_dist["cnt_1"],
        "Total_cnt2p": total_dist["cnt_2p"],
        "Total_pct0": total_dist["pct_0"],
        "Total_pct1": total_dist["pct_1"],
        "Total_pct2p": total_dist["pct_2p"],
    }

    return df, per_file, hard_cols, soft_cols

def _plot_three_bars(ax, title, pct0, pct1, pct2p):
    """Draw a simple 3-bar chart for 0 / 1 / 2+ percentages."""
    cats = ["0", "1", "2+"]
    vals = [pct0, pct1, pct2p]
    ax.bar(cats, vals)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of cycles")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

def plot_file_distributions(per_file_row: dict, outdir: str):
    """
    Make one figure per file showing three panels:
      - Hard: % with 0/1/2+ hard rules
      - Soft: % with 0/1/2+ soft rules
      - Total: % with 0/1/2+ total rules
    """
    os.makedirs(outdir, exist_ok=True)
    file_name = per_file_row["file"]

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

    _plot_three_bars(
        axs[0], "Hard rules",
        per_file_row.get("Hard_pct0", 0.0),
        per_file_row.get("Hard_pct1", 0.0),
        per_file_row.get("Hard_pct2p", 0.0),
    )
    _plot_three_bars(
        axs[1], "Soft rules",
        per_file_row.get("Soft_pct0", 0.0),
        per_file_row.get("Soft_pct1", 0.0),
        per_file_row.get("Soft_pct2p", 0.0),
    )
    _plot_three_bars(
        axs[2], "Total rules",
        per_file_row.get("Total_pct0", 0.0),
        per_file_row.get("Total_pct1", 0.0),
        per_file_row.get("Total_pct2p", 0.0),
    )

    fig.suptitle(file_name, fontsize=10, y=1.02)
    outfile = os.path.join(outdir, f"{os.path.splitext(file_name)[0]}_rule_dist.png")
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_global_distributions(df_all: pd.DataFrame, outdir: str):
    """
    One figure for global distributions across all cycles:
      - Hard_Count 0/1/2+
      - Soft_Count 0/1/2+
      - Total_Count 0/1/2+
    """
    os.makedirs(outdir, exist_ok=True)

    def dist(series):
        n = len(series)
        c0 = int((series == 0).sum())
        c1 = int((series == 1).sum())
        c2p = int((series >= 2).sum())
        p0 = round(100 * c0 / n, 2) if n else 0.0
        p1 = round(100 * c1 / n, 2) if n else 0.0
        p2 = round(100 * c2p / n, 2) if n else 0.0
        return (p0, p1, p2)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

    p0, p1, p2 = dist(df_all["Hard_Count"])
    _plot_three_bars(axs[0], "Hard rules (global)", p0, p1, p2)

    p0, p1, p2 = dist(df_all["Soft_Count"])
    _plot_three_bars(axs[1], "Soft rules (global)", p0, p1, p2)

    p0, p1, p2 = dist(df_all["Total_Count"])
    _plot_three_bars(axs[2], "Total rules (global)", p0, p1, p2)

    outfile = os.path.join(outdir, "GLOBAL_rule_distributions.png")
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_ranked_bars(df_files: pd.DataFrame, metrics: list[tuple[str, str]], outdir: str):
    """
    Make one horizontal bar chart per metric, ranking files high→low.

    metrics: list of (column_name, pretty_title)
      e.g., [("pct_hard_any","% cycles with ≥1 HARD rule")]
    """
    os.makedirs(outdir, exist_ok=True)
    if df_files.empty:
        return

    # Ensure we have a friendly label (strip extension) for plotting
    dfp = df_files.copy()
    dfp["file_base"] = dfp["file"].apply(lambda s: os.path.splitext(s)[0])

    for col, title in metrics:
        if col not in dfp.columns:
            # skip silently if this metric doesn't exist in the CSV
            continue

        tmp = dfp[["file_base", col]].dropna().sort_values(col, ascending=False)
        if tmp.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 0.4*len(tmp) + 1.5))
        ax.barh(tmp["file_base"], tmp[col])
        ax.invert_yaxis()  # top = highest value
        ax.set_xlabel(title)
        ax.set_title(f"Ranked by {title}")
        # annotate bars
        for i, (name, val) in enumerate(zip(tmp["file_base"], tmp[col])):
            ax.text(val, i, f" {val:.2f}", va="center")

        plt.tight_layout()
        out = os.path.join(outdir, f"RANK_{col}.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)



def main():
    import argparse, glob, os
    ap = argparse.ArgumentParser(description="Aggregate statistics from *_labeled.xlsx cycle label files (with count distributions).")
    ap.add_argument("input_dir", help="Directory containing *_labeled.xlsx files")
    ap.add_argument("-o","--outdir", default="stats_out", help="Output directory for statistics")
    ap.add_argument("--augment", action="store_true", help="Also overwrite each labeled file with added columns (Hard_Count, Soft_Count, Total_Count)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*_labeled.xlsx")))
    if not files:
        print(f"No *_labeled.xlsx files found in {args.input_dir}")
        return

    per_file_rows = []
    all_cycles_rows = []

    for path in files:
        base = os.path.basename(path)
        try:
            df = load_cycle_labels(path)
            df, one_stats, hard_cols, soft_cols = compute_per_file_stats(df)

            # annotate cycles with file
            df_out = df.copy()
            df_out.insert(0, "file", base)
            all_cycles_rows.append(df_out)

            # per-file row
            one_stats_row = {"file": base}
            one_stats_row.update(one_stats)
            per_file_rows.append(one_stats_row)

            print(f"[OK] {base}: cycles={one_stats['file_cycles']} hard_any={one_stats['hard_any_cnt']} soft_any={one_stats['soft_any_cnt']} both={one_stats['both_any_cnt']}  |  Hard dist 0/1/2+ = {one_stats['Hard_cnt0']}/{one_stats['Hard_cnt1']}/{one_stats['Hard_cnt2p']}")

            # optionally write back augmented workbook
            if args.augment:
                try:
                    with pd.ExcelWriter(path, engine="xlsxwriter", mode="a", if_sheet_exists="replace") as w:
                        df.to_excel(w, sheet_name="cycle_labels", index=False)
                except Exception:
                    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
                        df.to_excel(w, sheet_name="cycle_labels", index=False)

        except Exception as e:
            print(f"[SKIP] {base}: {e}")
            continue

    ## write per-file stats
    df_files = pd.DataFrame(per_file_rows).sort_values("file")
    df_files.to_csv(os.path.join(args.outdir, "per_file_stats.csv"), index=False)

    ## write all-cycles concatenated
    df_all = pd.concat(all_cycles_rows, ignore_index=True)
    df_all.to_csv(os.path.join(args.outdir, "all_cycles_with_counts.csv"), index=False)

    # global summary
    total_cycles = int(df_all.shape[0])
    for flag in ["HF_CHG_SPEC_HIGH","HF_DCHG_SPEC_LOW","HF_MISSING"]:
        if flag not in df_all.columns:
            df_all[flag] = False

    # Global distributions
    hard_dist = dist_counts(df_all["Hard_Count"])
    soft_dist = dist_counts(df_all["Soft_Count"])
    total_dist = dist_counts(df_all["Total_Count"])

    global_summary = {
        "total_files": len(per_file_rows),
        "total_cycles": total_cycles,
        "H1_total": int(df_all["HF_CHG_SPEC_HIGH"].sum()),
        "H3_total": int(df_all["HF_DCHG_SPEC_LOW"].sum()),
        "H4_total": int(df_all["HF_MISSING"].sum()),
        "H1_pct": round(100*df_all["HF_CHG_SPEC_HIGH"].mean(), 2) if total_cycles else 0.0,
        "H3_pct": round(100*df_all["HF_DCHG_SPEC_LOW"].mean(), 2) if total_cycles else 0.0,
        "H4_pct": round(100*df_all["HF_MISSING"].mean(), 2) if total_cycles else 0.0,
        "hard_any_total": int((df_all.filter(like="HF_").sum(axis=1) > 0).sum()),
        "soft_any_total": int((df_all.filter(like="SP_").sum(axis=1) > 0).sum()),
        "both_any_total": int(((df_all.filter(like="HF_").sum(axis=1) > 0) & (df_all.filter(like="SP_").sum(axis=1) > 0)).sum()),
        # Distributions (global)
        "Hard_cnt0": hard_dist["cnt_0"], "Hard_cnt1": hard_dist["cnt_1"], "Hard_cnt2p": hard_dist["cnt_2p"],
        "Hard_pct0": hard_dist["pct_0"], "Hard_pct1": hard_dist["pct_1"], "Hard_pct2p": hard_dist["pct_2p"],
        "Soft_cnt0": soft_dist["cnt_0"], "Soft_cnt1": soft_dist["cnt_1"], "Soft_cnt2p": soft_dist["cnt_2p"],
        "Soft_pct0": soft_dist["pct_0"], "Soft_pct1": soft_dist["pct_1"], "Soft_pct2p": soft_dist["pct_2p"],
        "Total_cnt0": total_dist["cnt_0"], "Total_cnt1": total_dist["cnt_1"], "Total_cnt2p": total_dist["cnt_2p"],
        "Total_pct0": total_dist["pct_0"], "Total_pct1": total_dist["pct_1"], "Total_pct2p": total_dist["pct_2p"],
    }

    pd.DataFrame([global_summary]).to_csv(os.path.join(args.outdir, "global_summary.csv"), index=False)
    print(f"Wrote stats to: {args.outdir}")

    df_files.to_csv(os.path.join(args.outdir, "per_file_stats.csv"), index=False)
    df_all.to_csv(os.path.join(args.outdir, "all_cycles_with_counts.csv"), index=False)
    pd.DataFrame([global_summary]).to_csv(os.path.join(args.outdir, "global_summary.csv"), index=False)

    # ---- Plots ----
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Per-file plots
    for row in df_files.to_dict(orient="records"):
        plot_file_distributions(row, plots_dir)

    # Global plot
    plot_global_distributions(df_all, plots_dir)

    print(f"Wrote plots to: {plots_dir}")

    # ---- Comparison (ranked) plots across files ----
    rank_metrics = [
        ("pct_hard_any", "% cycles with ≥1 HARD rule"),
        ("pct_soft_any", "% cycles with ≥1 SOFT rule"),
        ("pct_both_any", "% cycles with both HARD and SOFT"),
        ("pct_H1", "% cycles with H1 (Chg spec > 190 mAh/g)"),
        ("pct_H3", "% cycles with H3 (DChg spec < 120 mAh/g)"),
        ("pct_H4", "% cycles with H4 (Missing essentials)"),
        # You can also rank by averages if useful:
        ("avg_Hard_Count", "Avg HARD rules per cycle"),
        ("avg_Soft_Count", "Avg SOFT rules per cycle"),
        ("avg_Total_Count", "Avg TOTAL rules per cycle"),
    ]
    plot_ranked_bars(df_files, rank_metrics, plots_dir)
    print("Wrote ranked comparison plots.")


if __name__ == "__main__":
    main()
