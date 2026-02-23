# #!/usr/bin/env python3
# import os
# import argparse
# import pandas as pd
#
# GOOD_LABELS = {"good_drone", "good_no_drone"}
#
# def norm_label(x):
#     if pd.isna(x):
#         return ""
#     return str(x).strip().lower()
#
# def find_col(df, candidates):
#     cols_norm = {str(c).strip().lower(): c for c in df.columns}
#     for cand in candidates:
#         key = cand.strip().lower()
#         if key in cols_norm:
#             return cols_norm[key]
#     return None
#
# def balance_one_file(df: pd.DataFrame,
#                      cycle_col="cycle",
#                      label_col="cycle_label_3name"):
#     """
#     Cycle-aware balancing.
#
#     Rules:
#       1) If first cycle (min cycle) is BAD -> discard file (return None).
#       2) Let G = #GOOD cycles (good_drone + good_no_drone)
#       3) Keep all GOOD cycles
#       4) Keep first G BAD cycles in chronological order AFTER the first GOOD cycle appears
#       5) Drop remaining BAD cycles
#     """
#
#     # Required columns
#     cycle_col_found = find_col(df, [cycle_col, "Cycle", "cycle", "Cycle Index", "cycle_index"])
#     label_col_found = find_col(df, [label_col])
#
#     if cycle_col_found is None or label_col_found is None:
#         return None, {"status": "skipped", "reason": "missing required columns"}
#
#     cycle_col = cycle_col_found
#     label_col = label_col_found
#
#     df = df.copy()
#     df[label_col] = df[label_col].apply(norm_label)
#     df[cycle_col] = pd.to_numeric(df[cycle_col], errors="coerce")
#     df = df.dropna(subset=[cycle_col]).copy()
#     df[cycle_col] = df[cycle_col].astype(int)
#
#     # One label per cycle (if duplicates exist, take first occurrence)
#     cyc = (
#         df.sort_values(cycle_col)
#           .groupby(cycle_col, as_index=False)[label_col]
#           .first()
#     )
#
#     if cyc.empty:
#         return None, {"status": "skipped", "reason": "no cycles"}
#
#     first_cycle = int(cyc[cycle_col].min())
#     first_label = cyc.loc[cyc[cycle_col] == first_cycle, label_col].iloc[0]
#
#     # If cell is BAD from the first cycle -> discard completely
#     if first_label not in GOOD_LABELS:
#         return None, {
#             "status": "skipped",
#             "reason": "bad_from_first_cycle",
#             "first_cycle": first_cycle,
#             "first_label": first_label
#         }
#
#     good_cycles = cyc[cyc[label_col].isin(GOOD_LABELS)][cycle_col].tolist()
#     G = len(good_cycles)
#
#     if G == 0:
#         return None, {"status": "skipped", "reason": "no good cycles"}
#
#     first_good_cycle = int(min(good_cycles))
#
#     # BAD cycles after first good appears, in time order
#     bad_candidates = cyc[
#         (cyc[cycle_col] >= first_good_cycle) &
#         (~cyc[label_col].isin(GOOD_LABELS))
#     ][cycle_col].tolist()
#
#     bad_keep = bad_candidates[:G]
#
#     keep_cycles = set(good_cycles) | set(bad_keep)
#
#     # Keep ALL rows belonging to those cycles (handles 1-row-per-cycle and multi-row-per-cycle)
#     df_bal = df[df[cycle_col].isin(keep_cycles)].copy()
#     df_bal = df_bal.sort_values(cycle_col).reset_index(drop=True)
#
#     meta = {
#         "status": "processed",
#         "first_cycle": first_cycle,
#         "first_label": first_label,
#         "total_unique_cycles": len(cyc),
#         "n_good_cycles": G,
#         "n_bad_kept_cycles": len(bad_keep),
#         "kept_unique_cycles": len(keep_cycles),
#         "first_good_cycle": first_good_cycle,
#     }
#     return df_bal, meta
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--indir", required=True)
#     ap.add_argument("--outdir", required=True)
#     ap.add_argument("--cycle_col", default="cycle")
#     ap.add_argument("--label_col", default="cycle_label_3name")
#     args = ap.parse_args()
#
#     os.makedirs(args.outdir, exist_ok=True)
#
#     files = [
#         f for f in os.listdir(args.indir)
#         if (f.endswith(".csv") or f.endswith(".xlsx"))
#         and not f.startswith(("~", ".~", "~$"))
#     ]
#
#     summary = []
#
#     for fname in sorted(files):
#         in_path = os.path.join(args.indir, fname)
#
#         # Read
#         try:
#             if fname.endswith(".csv"):
#                 df = pd.read_csv(in_path)
#             else:
#                 df = pd.read_excel(in_path, engine="openpyxl")
#         except Exception as e:
#             print(f"[SKIP] {fname} read_error: {e}")
#             summary.append({"file": fname, "status": "skipped", "reason": f"read_error: {e}"})
#             continue
#
#         df_bal, meta = balance_one_file(df, cycle_col=args.cycle_col, label_col=args.label_col)
#         meta = {"file": fname, **meta}
#         summary.append(meta)
#
#         if df_bal is None:
#             print(f"[SKIP] {fname} :: {meta.get('reason')} (first={meta.get('first_cycle')}, {meta.get('first_label')})")
#             continue
#
#         # Write only processed files
#         out_path = os.path.join(args.outdir, fname)
#         try:
#             if fname.endswith(".csv"):
#                 df_bal.to_csv(out_path, index=False)
#             else:
#                 df_bal.to_excel(out_path, index=False)
#             print(f"[OK] {fname} -> good={meta['n_good_cycles']} bad_kept={meta['n_bad_kept_cycles']}")
#         except Exception as e:
#             print(f"[WARN] write_error {fname}: {e}")
#             meta["status"] = "skipped"
#             meta["reason"] = f"write_error: {e}"
#
#     # Save summary
#     pd.DataFrame(summary).to_csv(os.path.join(args.outdir, "balancing_summary.csv"), index=False)
#     print("\n[OK] Wrote balancing_summary.csv")
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import os
import argparse
import pandas as pd

GOOD_LABELS = {"good_drone", "good_no_drone"}

def norm_label(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def find_col(df, name):
    cols = {str(c).strip().lower(): c for c in df.columns}
    return cols.get(name.strip().lower(), None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cycle_col", default="cycle")              # will match Cycle too
    ap.add_argument("--label_col", default="cycle_label_3name")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = [
        f for f in os.listdir(args.indir)
        if (f.endswith(".xlsx") or f.endswith(".csv"))
        and not f.startswith(("~", ".~", "~$"))
    ]

    summary = []

    for fname in sorted(files):
        in_path = os.path.join(args.indir, fname)

        try:
            df = pd.read_excel(in_path, engine="openpyxl") if fname.endswith(".xlsx") else pd.read_csv(in_path)
        except Exception as e:
            summary.append({"file": fname, "status": "skipped", "reason": f"read_error: {e}"})
            continue

        cycle_col = find_col(df, args.cycle_col) or find_col(df, "Cycle") or find_col(df, "cycle")
        label_col = find_col(df, args.label_col)

        if cycle_col is None or label_col is None:
            summary.append({"file": fname, "status": "skipped", "reason": "missing required columns"})
            continue

        df = df.copy()
        df[label_col] = df[label_col].apply(norm_label)
        df[cycle_col] = pd.to_numeric(df[cycle_col], errors="coerce")
        df = df.dropna(subset=[cycle_col]).copy()
        df[cycle_col] = df[cycle_col].astype(int)

        if df.empty:
            summary.append({"file": fname, "status": "skipped", "reason": "empty after cleaning"})
            continue

        # first cycle (min cycle)
        first_cycle = int(df[cycle_col].min())
        first_label = df.loc[df[cycle_col] == first_cycle, label_col].iloc[0]

        if first_label not in GOOD_LABELS:
            summary.append({
                "file": fname, "status": "skipped",
                "reason": "bad_from_first_cycle",
                "first_cycle": first_cycle, "first_label": first_label
            })
            continue

        # keep file as-is (filter-only)
        out_path = os.path.join(args.outdir, fname)
        if fname.endswith(".xlsx"):
            df.to_excel(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)

        # counts for quick stats
        counts = df[label_col].value_counts().to_dict()
        summary.append({
            "file": fname, "status": "kept",
            "first_cycle": first_cycle, "first_label": first_label,
            "rows": len(df),
            **{f"count_{k}": v for k, v in counts.items()}
        })

    pd.DataFrame(summary).to_csv(os.path.join(args.outdir, "filter_summary.csv"), index=False)
    print("[OK] Wrote filter_summary.csv")

if __name__ == "__main__":
    main()

