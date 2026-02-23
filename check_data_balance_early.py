import pandas as pd
import os

windows = [1,2,5,10,20,30,50,60]
base_dir = "out_early_windows_THC_rich"

# for w in windows:
#     path = os.path.join(base_dir, f"early_{w}s.csv")
#     df = pd.read_csv(path)
#
#     print(f"\n=== Early {w}s ===")
#     print("Total rows:", len(df))
#     print(df["label"].value_counts())
#     print("\nClass ratio:")
#     print(df["label"].value_counts(normalize=True))
#
# for w in windows:
#     path = os.path.join(base_dir, f"early_{w}s.csv")
#     df = pd.read_csv(path)
#
#     cyc = df.groupby(["file","Cycle"])["label"].first().reset_index()
#
#     print(f"\n=== Early {w}s (Cycle-level) ===")
#     print("Unique cycles:", len(cyc))
#     print(cyc["label"].value_counts())
#     print(cyc["label"].value_counts(normalize=True))

# df = pd.read_csv("out_early_windows_THC_rich/early_60s.csv")

# file_label = df.groupby("file")["label"].value_counts().unstack(fill_value=0)
# print(file_label.head())

df = pd.read_csv("out_early_windows_THC_rich/early_60s.csv")

file_label = (
    df.groupby("file")["label"]
      .value_counts()
      .unstack(fill_value=0)
)

file_label["dominant_class"] = file_label.idxmax(axis=1)

print(file_label["dominant_class"].value_counts())