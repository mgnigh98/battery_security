import pandas as pd
from pathlib import Path

# ---- Paths ----
base = Path("/mnt/data")

csv_files = [
     "models_out/all_models_summary.csv",     # early 1s, 2s
     "models_out/metrics_early5_only.csv",
     "models_out/metrics_early10_only.csv",
     "models_out/metrics_early20_only.csv",
     "models_out/metrics_early30_only.csv",
     "models_out/metrics_early50_only.csv",
     "models_out/metrics_early60_only.csv",
     "models_out/metrics_full.csv",
]

# ---- Load & concat ----
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# ---- Ensure numeric ordering ----
df_all["early_window"] = df_all["early_window"].astype(int)

# Optional: sort
df_all = df_all.sort_values("early_window")

print(df_all.head())
