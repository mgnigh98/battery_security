#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_3class.py

Train and evaluate 3 models (RF, XGBoost, MLP) on
ALL_cycles_3class_early.csv to predict:

    cycle_label_3class ∈ {0=BAD, 1=good_not_drone, 2=drone_ready}

using different feature subsets:
    - full (all numeric features except IDs/labels)
    - no_early (exclude earlyXX_* features)
    - early20_only, early30_only, early50_only, early60_only

Grouped train/test split by 'file' so cycles from the same
battery do not leak across train and test.

Usage:
    python train_models_3class.py \
        --csv drone_labels_out/ALL_cycles_3class_early.csv \
        -o models_out
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# from plot_confusion_matrix import plot_cm

# Try to import XGBoost if available
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost is not installed. XGB model will be skipped.")


# ------------- Helpers -------------

def build_feature_sets(df: pd.DataFrame, target_col: str):
    """
    Build different feature subsets:
      - full: all numeric except IDs/labels
      - no_early: numeric excluding earlyXX_*
      - earlyT_only: numeric columns starting with earlyT_ for T in [20,30,50,60]
    """
    # Identify numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Columns to exclude from features
    label_cols = [
        target_col,
        "cycle_label_3class",
        "battery_label_3class",
    ]
    id_cols = ["Cycle"]  # keep 'file' out (non-numeric anyway)
    exclude = set(label_cols + id_cols)

    numeric_features = [c for c in num_cols if c not in exclude]

    # Full set
    feature_sets = {}
    feature_sets["full"] = numeric_features

    # No early features (baseline using only non-early stats)
    no_early = [c for c in numeric_features if not re.match(r"^early\d+_", c)]
    feature_sets["no_early"] = no_early

    # Early-only sets
    for T in [5, 10, 20, 30, 50, 60]:
        prefix = f"early{T}_"
        cols_T = [c for c in numeric_features if c.startswith(prefix)]
        if cols_T:
            feature_sets[f"{prefix}only"] = cols_T

    return feature_sets


import re

def build_feature_sets_from_df(df, label_col="label", ignore_cols=None):
    """
    Returns:
      feature_sets: dict[str, list[str]]
      base_features: list[str]
      early_windows: sorted list[int]  (e.g., [1,2,5,10,20,...])
    """
    if ignore_cols is None:
        ignore_cols = []

    # --- detect early window columns like early1_V_mean, early2_IR_early, etc.
    early_re = re.compile(r"^early(\d+)_", re.IGNORECASE)
    early_cols = []
    early_windows = set()

    for c in df.columns:
        m = early_re.match(str(c))
        if m:
            early_cols.append(c)
            early_windows.add(int(m.group(1)))

    early_windows = sorted(list(early_windows))

    # --- define candidate feature columns = numeric columns except ignored/label
    drop_cols = set([label_col] + (ignore_cols or []))
    candidate_cols = [c for c in df.columns if c not in drop_cols]

    # Keep only numeric features (important if you have strings like file names)
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    # base features = numeric cols excluding early*
    early_set = set(early_cols)
    base_features = [c for c in numeric_cols if c not in early_set]

    # Build feature sets
    feature_sets = {
        "full": base_features + early_cols,
        "no_early": base_features,
    }

    for T in early_windows:
        cols_T = [c for c in early_cols if early_re.match(str(c)) and int(early_re.match(str(c)).group(1)) == T]
        # keep stable ordering
        cols_T = sorted(cols_T)
        feature_sets[f"early{T}_only"] = base_features + cols_T

    return feature_sets, base_features, early_windows



def train_and_eval_models(X_train, X_test, y_train, y_test, feature_set_name: str, outdir: str = None):
    """
    Train RF, XGB (if available), and MLP on the given features.
    Print metrics and optionally save simple reports.
    """
    import pandas as pd
    results = []

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== RandomForest ({feature_set_name}) ===")
    print(f"Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, digits=3))

    results.append(("RandomForest", feature_set_name, acc, f1))

    # Save raw confusion matrix if outdir is given
    if outdir:
        cm_df = pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2])
        cm_path = os.path.join(outdir, f"cm_RF_{feature_set_name}.csv")
        cm_df.to_csv(cm_path, index=True)

    # ---- XGBoost ----
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)

        acc_x = accuracy_score(y_test, y_pred)
        f1_x = f1_score(y_test, y_pred, average="macro")
        cm_x = confusion_matrix(y_test, y_pred)

        print(f"\n=== XGBoost ({feature_set_name}) ===")
        print(f"Accuracy: {acc_x:.4f}, Macro-F1: {f1_x:.4f}")
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification report:\n", classification_report(y_test, y_pred, digits=3))

        results.append(("XGBoost", feature_set_name, acc_x, f1_x))

        if outdir:
            import pandas as pd
            cm_df = pd.DataFrame(cm_x, index=[0, 1, 2], columns=[0, 1, 2])
            cm_path = os.path.join(outdir, f"cm_XGB_{feature_set_name}.csv")
            cm_df.to_csv(cm_path, index=True)
    else:
        print(f"\n Skipping XGBoost for feature set '{feature_set_name}' (not installed).")



    # ---- MLP (Neural Net) ----
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            alpha=1e-4,
            random_state=42
        ))
    ])
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    acc_m = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    cm_m = confusion_matrix(y_test, y_pred)

    print(f"\n=== MLP ({feature_set_name}) ===")
    print(f"Accuracy: {acc_m:.4f}, Macro-F1: {f1_m:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, digits=3))


    results.append(("MLP", feature_set_name, acc_m, f1_m))

    # Save metrics if outdir provided
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        df_res = pd.DataFrame(results, columns=["model", "feature_set", "accuracy", "macro_f1"])
        csv_path = os.path.join(outdir, f"metrics_{feature_set_name}.csv")
        df_res.to_csv(csv_path, index=False)
        print(f" Saved metrics for {feature_set_name} to {csv_path}")

        cm_df = pd.DataFrame(cm_m, index=[0, 1, 2], columns=[0, 1, 2])
        cm_path = os.path.join(outdir, f"cm_MLP_{feature_set_name}.csv")
        cm_df.to_csv(cm_path, index=True)

    return results


def select_feature_sets(feature_sets: dict, spec: str):
    """
    spec examples:
      - "all"
      - "full,no_early,early1_only"
      - "early:1,2,5"
      - "early:1-10"
    """
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return feature_sets

    selected = {}

    if spec.startswith("early:"):
        s = spec.split(":", 1)[1].strip()
        wanted = set()

        # ranges like 1-10
        if "-" in s and "," not in s:
            a, b = s.split("-", 1)
            for t in range(int(a), int(b) + 1):
                wanted.add(f"early{t}_only")
        else:
            # list like 1,2,5
            parts = [p.strip() for p in s.split(",") if p.strip()]
            for p in parts:
                wanted.add(f"early{int(p)}_only")

        for k in feature_sets.keys():
            if k.lower() in wanted:
                selected[k] = feature_sets[k]

        return selected

    # comma-separated explicit names
    names = [x.strip() for x in spec.split(",") if x.strip()]
    for name in names:
        # allow user to pass early1_only or early1_only etc
        if name in feature_sets:
            selected[name] = feature_sets[name]
        else:
            # try case-insensitive match
            for k in feature_sets.keys():
                if k.lower() == name.lower():
                    selected[k] = feature_sets[k]
                    break

    return selected



# ------------- Main -------------

def main():
    ap = argparse.ArgumentParser(
        description="Train RF, XGB, and MLP on ALL_cycles_3class_early.csv "
                    "to predict 3-class cycle labels with different feature sets."
    )
    ap.add_argument("--csv", required=True,
                    help="Path to ALL_cycles_3class_early.csv")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Optional output folder for metrics CSVs")
    ap.add_argument("--target", choices=["cycle", "battery"], default="cycle",
                    help="Predict cycle_label_3class (cycle) or battery_label_3class (battery). "
                         "Default: cycle")
    ap.add_argument(
        "--feature_sets",
        default="all",
        help=(
            "Which feature sets to run. Options:\n"
            "  all                -> run everything\n"
            "  full,no_early      -> run a comma-separated list\n"
            "  early:1,2          -> run early1_only and early2_only\n"
            "  early:1-10         -> run early1_only..early10_only if present\n"
        )
    )

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Adjust these to match your file
    LABEL_COL = "label"  # or "battery_level_3class", etc.
    IGNORE_COLS = ["file", "netlist_name", "Date_Time", "DateTime", "Date_Time_str"]  # keep any IDs here

    FEATURE_SETS, BASE_FEATURES, EARLY_WINDOWS = build_feature_sets_from_df(
        df,
        label_col=LABEL_COL,
        ignore_cols=IGNORE_COLS
    )

    print("Detected early windows:", EARLY_WINDOWS)
    print(" Feature sets:", list(FEATURE_SETS.keys()))

    if args.target == "cycle":
        target_col = "cycle_label_3class"
        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found in CSV.")
        y = df[target_col].values
        groups = df["file"].values  # group by battery file
        X_source = df.copy()
    else:
        # battery-level: one row per file (aggregating aready in battery_level_3class_summary normally),
        # but here we'll just take the most frequent label per file.
        if "battery_label_3class" not in df.columns:
            raise ValueError("battery_label_3class not found in CSV.")

        agg = df.groupby("file")["battery_label_3class"].agg(lambda x: x.mode().iloc[0])
        y = agg.values
        groups = agg.index.values  # group name = file
        # For X, aggregate numeric features by mean per battery
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ["battery_label_3class", "cycle_label_3class"]]
        X_source = df.groupby("file")[num_cols].mean().reset_index()
        X_source = X_source.rename(columns={"file": "file_agg"})

    # Filter rows with NaN target
    mask_valid = ~pd.isna(y)
    if args.target == "cycle":
        df = df[mask_valid].reset_index(drop=True)
        y = y[mask_valid]
        groups = groups[mask_valid]
    else:
        # Already aggregated
        pass

    # Build feature sets from the appropriate df
    # if args.target == "cycle":
    #     feature_sets = build_feature_sets(df, target_col="cycle_label_3class")
    # else:
    #     feature_sets = build_feature_sets(X_source, target_col="battery_label_3class")

    # Build feature sets from df (cycle) or X_source (battery)
    if args.target == "cycle":
        feature_sets, base_feats, early_windows = build_feature_sets_from_df(
            df,
            label_col="cycle_label_3class",
            ignore_cols=IGNORE_COLS
        )
    else:
        feature_sets, base_feats, early_windows = build_feature_sets_from_df(
            X_source,
            label_col="battery_label_3class",
            ignore_cols=["file_agg"]  # battery-level uses file_agg id col
        )

    print("[INFO] Detected early windows:", early_windows)
    print("[INFO] Available feature sets:", list(feature_sets.keys()))

    # Use GroupShuffleSplit (group by file) for train/test
    if args.target == "cycle":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        # We need X matrix later per feature set; for now store df and y, groups only
        splits = list(gss.split(df, y, groups=groups))
        train_idx, test_idx = splits[0]
    else:
        # For battery-level, each group is already a battery
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        X_batt = X_source
        y_batt = y
        groups_batt = groups
        splits = list(gss.split(X_batt, y_batt, groups=groups_batt))
        train_idx, test_idx = splits[0]

    all_results = []

    selected_feature_sets = select_feature_sets(feature_sets, args.feature_sets)

    if not selected_feature_sets:
        raise ValueError(
            f"No feature sets matched --feature_sets='{args.feature_sets}'. "
            f"Available: {list(feature_sets.keys())}"
        )

    feature_sets = selected_feature_sets
    print("Running feature sets:", list(feature_sets.keys()))

    for fs_name, fs_cols in feature_sets.items():
        if not fs_cols:
            print(f"\nFeature set '{fs_name}' has no columns.")
            continue

        print(f"\n===============================")
        print(f" Feature set: {fs_name}")
        print(f" #features: {len(fs_cols)}")
        print(f"===============================")

        if args.target == "cycle":
            X = df[fs_cols].values
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            X = X_batt[fs_cols].values
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_batt[train_idx], y_batt[test_idx]

        # Replace any NaNs with 0 for tree models; scaler in MLP will handle
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        res = train_and_eval_models(X_train, X_test, y_train, y_test,
                                    feature_set_name=fs_name,
                                    outdir=args.outdir)
        all_results.extend(res)

    # Summarize all results in one CSV if outdir given
    if args.outdir and all_results:
        df_all = pd.DataFrame(all_results, columns=["model", "feature_set", "accuracy", "macro_f1"])
        summary_csv = os.path.join(args.outdir, "all_models_summary_rev.csv")
        df_all.to_csv(summary_csv, index=False)
        print(f"\nSaved combined summary to {summary_csv}")


if __name__ == "__main__":
    main()
