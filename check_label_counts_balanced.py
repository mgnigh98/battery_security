#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from collections import Counter

# LABEL_COL_CANDIDATES = ["cycle_label_3name", "cycle_label_3class", "Label"]
# CYCLE_COL_CANDIDATES = ["Cycle", "cycle"]
#
# def find_col(df, candidates):
#     cols_norm = {str(c).strip().lower(): c for c in df.columns}
#     for cand in candidates:
#         key = str(cand).strip().lower()
#         if key in cols_norm:
#             return cols_norm[key]
#     return None
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--indir", required=True, help="drone_labels_out_balanced")
#     ap.add_argument("--out", default=None, help="optional: save per-file counts csv")
#     args = ap.parse_args()
#
#     files = [f for f in os.listdir(args.indir) if f.endswith(".xlsx") or f.endswith(".csv")]
#     per_file_rows = []
#     overall = Counter()
#
#     for f in sorted(files):
#         path = os.path.join(args.indir, f)
#         try:
#             df = pd.read_excel(path, engine="openpyxl") if f.endswith(".xlsx") else pd.read_csv(path)
#         except Exception as e:
#             print(f"[SKIP] {f}: read error {e}")
#             continue
#
#         label_col = find_col(df, LABEL_COL_CANDIDATES)
#         cycle_col = find_col(df, CYCLE_COL_CANDIDATES)
#
#         if label_col is None:
#             print(f"[SKIP] {f}: no label column found")
#             continue
#
#         # normalize labels
#         labels = df[label_col].astype(str).str.strip().str.lower()
#
#         # Count as cycles if cycle col exists; else count rows
#         if cycle_col is not None:
#             df[cycle_col] = pd.to_numeric(df[cycle_col], errors="coerce")
#             df2 = df.dropna(subset=[cycle_col]).copy()
#             df2[cycle_col] = df2[cycle_col].astype(int)
#             df2[label_col] = df2[label_col].astype(str).str.strip().str.lower()
#
#             # unique cycles per label
#             cts = df2.groupby(label_col)[cycle_col].nunique().to_dict()
#             total_cycles = df2[cycle_col].nunique()
#             basis = "unique_cycles"
#         else:
#             cts = labels.value_counts().to_dict()
#             total_cycles = len(df)
#             basis = "rows"
#
#         overall.update(cts)
#
#         row = {"file": f, "basis": basis, "total": total_cycles}
#         row.update({f"count_{k}": v for k, v in cts.items()})
#         per_file_rows.append(row)
#
#     per_file = pd.DataFrame(per_file_rows).fillna(0)
#
#     print("\n=== OVERALL COUNTS (summed) ===")
#     for k, v in overall.most_common():
#         print(f"{k:20s} {v}")
#
#     # Helpful derived metrics if 3name exists
#     # (works if your labels are good_drone, good_no_drone, bad)
#     good = overall.get("good_drone", 0) + overall.get("good_no_drone", 0)
#     bad = overall.get("bad", 0)
#     if good + bad > 0:
#         print(f"\nGOOD={good}, BAD={bad}, BAD/GOOD={bad / max(good,1):.3f}")
#
#     if args.out:
#         per_file.to_csv(args.out, index=False)
#         print(f"\n[OK] wrote {args.out}")
#
# if __name__ == "__main__":
#     main()

# IN_CSV  = "all_csv_for_training/ALL_cycles_3class_filtered.csv"
# OUT_CSV = "all_csv_for_training/ALL_cycles_3class_filtered_balanced.csv"
#
# GOOD = {"good_drone", "good_not_drone"}
#
# df = pd.read_csv(IN_CSV)
# df["cycle_label_3name"] = df["cycle_label_3name"].astype(str).str.strip().str.lower()
#
# df_good = df[df["cycle_label_3name"].isin(GOOD)]
# df_bad  = df[~df["cycle_label_3name"].isin(GOOD)]
#
# N_GOOD = len(df_good)
# BAD_TARGET = N_GOOD   # or int(0.9 * N_GOOD)
#
# print(f"GOOD={N_GOOD}, BAD(before)={len(df_bad)}, BAD(target)={BAD_TARGET}")
#
# # keep early bad cycles first
# df_bad = df_bad.sort_values("Cycle")
# df_bad = df_bad.iloc[:BAD_TARGET]
#
# df_final = pd.concat([df_good, df_bad], ignore_index=True)
# df_final = df_final.sort_values(["file", "Cycle"]).reset_index(drop=True)
#
# df_final.to_csv(OUT_CSV, index=False)
#
# print("\nFinal counts:")
# print(df_final["cycle_label_3name"].value_counts())
#
#
# df = pd.read_csv("all_csv_for_training/ALL_cycles_3class_early.csv")
# miss_cols = [c for c in df.columns if c.endswith("_missing")]
# print("Missing columns:", miss_cols[:8], "...")
# print("Missing rate (mean) for first few:")
# for c in miss_cols[:8]:
#     print(c, df[c].mean())


df = pd.read_csv("all_csv_for_training/ALL_cycles_3class_filtered_balanced.csv")
print("\nFinal counts:")
print(df["cycle_label_3name"].value_counts())
# print(df)


