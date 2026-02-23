import pandas as pd
import numpy as np

# MASTER_CSV   = "all_csv_for_training/ALL_cycles_3class.csv"          # your big file
# COMPUTED_CSV = "step_stats_v4/all_cycles_3class.csv"  # the CSV from the step-sheet extractor
# OUT_CSV      = "step_stats_v4/ALL_cycles_3class_filled.csv"
#
# STEP_COLS = [
#     "charge_V_end",
#     "take_off_V_end",
#     "hover_V_end",
#     "cruise_V_end",
#     "landing_V_end",
#     "standby_V_end",
#     "standby_t_end_h",
# ]
#
#
# def basename_only(s: pd.Series) -> pd.Series:
#     return s.astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1].str.strip()
#
# def stem_key(s: pd.Series) -> pd.Series:
#     """
#     Normalize filenames so:
#     - keep basename only
#     - remove extension
#     - remove trailing '_labeled' if present
#     - remove trailing '_good_steps_vX' if present (just in case)
#     """
#     x = basename_only(s)
#     x = x.str.replace(r"\.xlsx$", "", regex=True)
#     x = x.str.replace(r"_labeled$", "", regex=True)
#     x = x.str.replace(r"_good_steps_v\d+$", "", regex=True)
#     # also remove other common suffixes if needed
#     return x.str.lower()
#
# # ---- Load master ----
# df = pd.read_csv(MASTER_CSV)
# if "Cycle" not in df.columns or "file" not in df.columns:
#     raise ValueError("MASTER_CSV must contain 'Cycle' and 'file' columns.")
#
# df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce").astype("Int64")
# df["file_stem"] = stem_key(df["file"])
#
# for c in STEP_COLS:
#     if c not in df.columns:
#         df[c] = np.nan
#
# # ---- Load computed ----
# comp = pd.read_csv(COMPUTED_CSV)
#
# # tolerate cycle vs Cycle
# if "cycle" in comp.columns and "Cycle" not in comp.columns:
#     comp = comp.rename(columns={"cycle": "Cycle"})
#
# if "Cycle" not in comp.columns or "file" not in comp.columns:
#     raise ValueError("COMPUTED_CSV must contain 'Cycle' and 'file' (or 'cycle').")
#
# comp["Cycle"] = pd.to_numeric(comp["Cycle"], errors="coerce").astype("Int64")
# comp["file_stem"] = stem_key(comp["file"])
#
# keep_cols = ["file_stem", "Cycle"] + [c for c in STEP_COLS if c in comp.columns]
# comp = comp[keep_cols].dropna(subset=["file_stem", "Cycle"]).drop_duplicates(["file_stem", "Cycle"])
#
# # ---- Merge + fill ----
# merged = df.merge(comp, on=["file_stem", "Cycle"], how="left", suffixes=("", "_new"))
#
# filled_counts = {}
# for c in STEP_COLS:
#     newc = f"{c}_new"
#     if newc not in merged.columns:
#         filled_counts[c] = 0
#         continue
#     mask = merged[c].isna() & merged[newc].notna()
#     filled_counts[c] = int(mask.sum())
#     merged.loc[mask, c] = merged.loc[mask, newc]
#
# # cleanup
# drop_cols = ["file_stem"] + [f"{c}_new" for c in STEP_COLS if f"{c}_new" in merged.columns]
# merged = merged.drop(columns=drop_cols)
#
# merged.to_csv(OUT_CSV, index=False)
#
# print("Done.")
# print("Filled counts per column:")
# for k, v in filled_counts.items():
#     print(f"  {k}: {v}")
# print(f"\nWrote: {OUT_CSV}")

import pandas as pd

INPUT_CSV  = "step_stats_v4/ALL_cycles_3class_filled.csv"
OUTPUT_CSV = "step_stats_v4/ALL_cycles_3class_final.csv"

STEP_COLS = [
    "charge_V_end",
    "take_off_V_end",
    "hover_V_end",
    "cruise_V_end",
    "landing_V_end",
    "standby_V_end",
    "standby_t_end_h",
]

df = pd.read_csv(INPUT_CSV)

initial_rows = len(df)

# Drop rows with ANY missing step-end value
mask_complete = df[STEP_COLS].notna().all(axis=1)
df_clean = df[mask_complete]

final_rows = len(df_clean)

print(f"Initial rows: {initial_rows}")
print(f"Removed rows: {initial_rows - final_rows}")
print(f"Final rows: {final_rows}")

df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote cleaned file: {OUTPUT_CSV}")

