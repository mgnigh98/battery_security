#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_models_3class_select.py

- Trains RF / XGB / MLP on a CSV with cycle_label_3class
- Builds feature sets: full, no_early, earlyT_only (T inferred from columns)
- Lets you run only chosen feature sets (e.g., early:1,2)
- Robust NaN handling using SimpleImputer + numeric coercion

Example:
  python train_models_3class_select.py \
    --csv drone_labels_out/ALL_cycles_3class_early_1s2s_trunc200.csv \
    -o models_out \
    --target cycle \
    --feature_sets early:1,2
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Try XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed; XGB will be skipped.")


EARLY_RE = re.compile(r"^early(\d+)_", re.IGNORECASE)


def coerce_numeric(df: pd.DataFrame, cols):
    """Force numeric; non-parsable -> NaN."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def detect_early_windows(df: pd.DataFrame):
    wins = set()
    early_cols = []
    for c in df.columns:
        m = EARLY_RE.match(str(c))
        if m:
            wins.add(int(m.group(1)))
            early_cols.append(c)
    return sorted(wins), sorted(early_cols)


# def build_feature_sets(df: pd.DataFrame, label_col: str, ignore_cols=None):
#     if ignore_cols is None:
#         ignore_cols = []
#
#     early_windows, early_cols = detect_early_windows(df)
#
#     drop_cols = set([label_col] + ignore_cols)
#     candidate_cols = [c for c in df.columns if c not in drop_cols]
#
#     # numeric candidates only
#     numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
#     base_features = [c for c in numeric_cols if not EARLY_RE.match(str(c))]
#
#     feature_sets = {
#         "full": base_features + early_cols,
#         "no_early": base_features,
#     }
#
#     for T in early_windows:
#         cols_T = [c for c in early_cols if c.lower().startswith(f"early{T}_")]
#         feature_sets[f"early{T}_only"] = cols_T  # EARLY ONLY (no base)
#         feature_sets[f"early{T}_plus_base"] = base_features + cols_T  # if you want base+early
#
#     return feature_sets, base_features, early_windows

def is_binary_like(series: pd.Series) -> bool:
    # bool dtype -> drop
    if pd.api.types.is_bool_dtype(series):
        return True

    # numeric 0/1 only -> drop (ignore NaN)
    s = series.dropna()
    if s.empty:
        return False

    # try numeric conversion if object slipped through
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return False

    uniq = set(pd.unique(s))
    return uniq.issubset({0, 1}) and len(uniq) <= 2


def build_feature_sets(
    df: pd.DataFrame,
    label_col: str,
    ignore_cols=None,
    drop_binary_features: bool = True,
    # name-based patterns to drop (edit freely)
    drop_name_regex: str = r"(?:^|_)(flag|missing)(?:_|$)|_ws$|_per_i$"
):
    if ignore_cols is None:
        ignore_cols = []

    BIN_DROP_RE = re.compile(r"(flag|missing)$", re.IGNORECASE)

    def filter_binary_like(cols):
        return [c for c in cols if not BIN_DROP_RE.search(str(c))]

    early_windows, early_cols = detect_early_windows(df)

    drop_cols = set([label_col] + ignore_cols)
    candidate_cols = [c for c in df.columns if c not in drop_cols]

    # numeric candidates only
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    # base features = numeric cols excluding early*
    base_features = [c for c in numeric_cols if not EARLY_RE.match(str(c))]

    early_cols = filter_binary_like(early_cols)
    base_features = filter_binary_like(base_features)

    # --------- NEW: drop binary/flag-like features everywhere ----------
    if drop_binary_features:
        name_pat = re.compile(drop_name_regex, re.IGNORECASE)

        def keep_col(c: str) -> bool:
            # drop by name pattern
            if name_pat.search(str(c)):
                return False
            # drop if binary-like values
            return not is_binary_like(df[c])

        base_features = [c for c in base_features if keep_col(c)]
        early_cols = [c for c in early_cols if keep_col(c)]
    # -----------------------------------------------------------------

    feature_sets = {
        "full": base_features + early_cols,
        "no_early": base_features,
    }

    for T in early_windows:
        cols_T = [c for c in early_cols if c.lower().startswith(f"early{T}_")]
        feature_sets[f"early{T}_only"] = cols_T  # EARLY ONLY (no base)
        feature_sets[f"early{T}_plus_base"] = base_features + cols_T  # base+early

    return feature_sets, base_features, early_windows



# def parse_feature_set_selection(selection: str, available: list[str]):
#     """
#     selection examples:
#       "all"
#       "full,no_early"
#       "early:1,2" -> early1_only, early2_only
#       "early_plus:1,2" -> early1_plus_base, early2_plus_base
#     """
#     selection = (selection or "all").strip().lower()
#     if selection == "all":
#         return available
#
#     chosen = []
#     parts = [p.strip() for p in selection.split(",") if p.strip()]
#
#     # handle early:1,2 as a single token maybe passed as "early:1,2"
#     # so detect prefixes first
#     if selection.startswith("early:"):
#         nums = selection.split(":", 1)[1]
#         ts = [int(x.strip()) for x in nums.split(",") if x.strip().isdigit()]
#         for t in ts:
#             name = f"early{t}_only"
#             if name in available:
#                 chosen.append(name)
#         return chosen
#
#     if selection.startswith("early_plus:"):
#         nums = selection.split(":", 1)[1]
#         ts = [int(x.strip()) for x in nums.split(",") if x.strip().isdigit()]
#         for t in ts:
#             name = f"early{t}_plus_base"
#             if name in available:
#                 chosen.append(name)
#         return chosen
#
#     # otherwise treat as explicit names
#     for p in parts:
#         if p in available:
#             chosen.append(p)
#
#     return chosen

def parse_feature_set_selection(selection: str, available: list[str]):
    """
    selection examples:
      "all"
      "full,no_early"
      "early:1,2"
      "early:1-5"
      "early_plus:1,2"
      "full,early:1,2"
    """
    selection = (selection or "all").strip().lower()

    if selection == "all":
        return available.copy()

    selected = []
    available_set = set(available)

    tokens = [t.strip() for t in selection.split(",") if t.strip()]

    def add_if_exists(name):
        if name in available_set:
            selected.append(name)
        else:
            print(f"[WARN] Requested feature set '{name}' not available.")

    for tok in tokens:
        # ---- early only ----
        if tok.startswith("early:"):
            spec = tok.split(":", 1)[1]

            # range early:1-5
            if "-" in spec:
                a, b = spec.split("-", 1)
                try:
                    a, b = int(a), int(b)
                    for t in range(a, b + 1):
                        add_if_exists(f"early{t}_only")
                except ValueError:
                    print(f"[WARN] Invalid early range '{tok}'")
            else:
                for s in spec.split(","):
                    if s.isdigit():
                        add_if_exists(f"early{int(s)}_only")

        # # ---- early + base ----
        # elif tok.startswith("early_plus:"):
        #     spec = tok.split(":", 1)[1]
        #     for s in spec.split(","):
        #         if s.isdigit():
        #             add_if_exists(f"early{int(s)}_plus_base")

        # ---- explicit feature set name ----
        else:
            add_if_exists(tok)

    # keep original ordering from `available`
    ordered = [fs for fs in available if fs in selected]

    if not ordered:
        raise ValueError(
            f"No feature sets matched selection='{selection}'. "
            f"Available={available}"
        )

    return ordered


def make_models():
    # Imputer shared
    imputer = SimpleImputer(strategy="median")

    rf = Pipeline([
        ("imputer", imputer),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    xgb = None
    if HAS_XGB:
        xgb = Pipeline([
            ("imputer", imputer),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
            ))
        ])

    mlp = Pipeline([
        ("imputer", imputer),
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

    return rf, xgb, mlp


def train_eval_one(model_name, model, X_train, X_test, y_train, y_test, outdir, tag):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {model_name} ({tag}) ===")
    print(f"Accuracy: {acc:.6f}, Macro-F1: {f1:.6f}")
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, y_pred, digits=3))

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2]).to_csv(
            os.path.join(outdir, f"cm_{model_name}_{tag}.csv")
        )

    return (model_name, tag, acc, f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--target", choices=["cycle"], default="cycle")
    ap.add_argument("--label_col", default="cycle_label_3class")
    ap.add_argument("--group_col", default="file")
    ap.add_argument("--ignore_cols", default="file,Date_Time,DateTime,Date_Time_str,netlist_name")
    ap.add_argument("--feature_sets", default="all",
                    help="all | full,no_early | early:1,2 | early_plus:1,2 | explicit names")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    ignore_cols = [c.strip() for c in args.ignore_cols.split(",") if c.strip()]
    label_col = args.label_col
    group_col = args.group_col

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in CSV.")

    # Build feature sets using dtype info; but first coerce numeric for early/base columns

    feature_sets, base_features, early_windows = build_feature_sets(
        df, label_col=label_col, ignore_cols=ignore_cols
    )
    available = list(feature_sets.keys())
    print(" Detected early windows:", early_windows)
    print("Available feature sets:", available)

    selected = parse_feature_set_selection(args.feature_sets, available)
    if not selected:
        raise ValueError(f"No feature sets selected. Input was '{args.feature_sets}'. Available: {available}")
    print(" Running feature sets:", selected)

    print("CSV early windows detected:", early_windows)
    for k in selected:
        print(k, "->", len(feature_sets[k]), "cols")

    # Group split
    y = df[label_col].values
    groups = df[group_col].values

    mask_valid = ~pd.isna(y)
    df = df.loc[mask_valid].reset_index(drop=True)
    y = y[mask_valid]
    groups = groups[mask_valid]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))

    rf, xgb, mlp = make_models()
    all_rows = []

    for fs_name in selected:
        cols = feature_sets[fs_name]
        if not cols:
            print(f" {fs_name} has 0 columns.")
            continue

        # Ensure numeric, then slice into numpy
        df_num = coerce_numeric(df, cols)
        X = df_num[cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print("\n===============================")
        print(f" Feature set: {fs_name}")
        print(f" #features: {len(cols)}")
        print("===============================")

        all_rows.append(train_eval_one("RandomForest", rf, X_train, X_test, y_train, y_test, args.outdir, fs_name))
        if xgb is not None:
            all_rows.append(train_eval_one("XGBoost", xgb, X_train, X_test, y_train, y_test, args.outdir, fs_name))
        else:
            print(" Skipping XGBoost (not installed).")
        all_rows.append(train_eval_one("MLP", mlp, X_train, X_test, y_train, y_test, args.outdir, fs_name))

    if args.outdir and all_rows:
        out = pd.DataFrame(all_rows, columns=["model", "feature_set", "accuracy", "macro_f1"])
        out_path = os.path.join(args.outdir, "all_models_summary_3.csv")
        out.to_csv(out_path, index=False)
        print(f"\n[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
