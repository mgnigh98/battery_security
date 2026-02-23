#!/usr/bin/env python3
import pandas as pd
import os, re

# labels = pd.read_csv("drone_labels_out/all_cycle_labels_3class.csv")
# steps  = pd.read_csv("step_stats_all/ALL_step_ends_all_cycles.csv")
#
# print("labels columns:", labels.columns.tolist())
# print("steps columns :", steps.columns.tolist())
#
# print("labels file sample:", labels["file"].head(3).tolist())
# print("steps file sample :", steps["file"].head(3).tolist())
#
# # How many exact file-name matches?
# print("Exact file-name overlap:", len(set(labels["file"]) & set(steps["file"])))


LABELS_CSV = "drone_labels_out/all_cycle_labels_3class.csv"
STEPS_CSV  = "step_stats_all/ALL_step_ends_all_cycles.csv"
OUT_CSV    = "drone_labels_out/all_cycle_labels_3class_with_steps.csv"

def canonical_stem(name: str) -> str:
    """
    Make a stable key:
    - basename
    - remove extension
    - remove trailing tags like: _good, _bad, _labeled, _good_labeled_3class, etc.
    - collapse to alphanumerics
    """
    s = os.path.basename(str(name)).strip()
    s = os.path.splitext(s)[0]  # remove .xlsx
    s = s.strip()

    # remove common trailing suffixes (allow optional spaces before underscore)
    # examples: "abc_good", "abc _good", "abc_good_labeled_3class"
    s = re.sub(r"\s*_good_labeled_3class\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*_labeled\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*_good\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*_bad\s*$", "", s, flags=re.I)

    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

labels = pd.read_csv(LABELS_CSV)
steps  = pd.read_csv(STEPS_CSV)

# drop junk columns
for df in (labels, steps):
    junk = [c for c in df.columns if str(c).startswith("Unnamed")]
    if junk:
        df.drop(columns=junk, inplace=True)

# ensure Cycle int
labels["Cycle"] = pd.to_numeric(labels["Cycle"], errors="coerce")
steps["Cycle"]  = pd.to_numeric(steps["Cycle"], errors="coerce")
labels = labels.dropna(subset=["Cycle"]).copy()
steps  = steps.dropna(subset=["Cycle"]).copy()
labels["Cycle"] = labels["Cycle"].astype(int)
steps["Cycle"]  = steps["Cycle"].astype(int)

# keys
labels["join_key"] = labels["file"].apply(canonical_stem)
steps["join_key"]  = steps["file"].apply(canonical_stem)

overlap = len(set(labels["join_key"]) & set(steps["join_key"]))
print("Canonical join_key overlap:", overlap)

# merge on join_key + Cycle
merged = labels.merge(
    steps.drop(columns=["file"], errors="ignore"),
    on=["join_key", "Cycle"],
    how="left"
)

# diagnostics
if "charge_V_end" in merged.columns:
    print("Matched step rows:", merged["charge_V_end"].notna().sum(), "/", len(merged))
else:
    print("WARNING: charge_V_end not found after merge — check steps CSV columns.")

merged.drop(columns=["join_key"], inplace=True)
merged.to_csv(OUT_CSV, index=False)
print(f"[DONE] wrote {OUT_CSV}")


df = pd.read_csv("drone_labels_out/all_cycle_labels_3class_with_steps.csv")
print(df["charge_V_end"].notna().sum())

df = pd.read_csv("drone_labels_out/all_cycle_labels_3class_with_steps.csv")
df["missing_step_pattern"] = df["charge_V_end"].isna().astype(int)
df.to_csv("drone_labels_out/all_cycle_labels_3class_with_steps_flag.csv", index=False)
print(df["missing_step_pattern"].value_counts())




# steps = pd.read_csv("step_stats_all/ALL_step_ends_all_cycles.csv")
# print(steps["file"].head(20).tolist())
# print("unique files:", steps["file"].nunique())
#
#
# def norm_stem(x):
#     x=str(x).strip()
#     x=os.path.splitext(x)[0].lower()
#     x=re.sub(r"[^a-z0-9]+","",x)
#     return x
#
# print("steps file_key sample:", steps["file"].head(5).apply(norm_stem).tolist())

