# simple_train_rf_xgb_group_table.py

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


DROP_COLS_FROM_X = {
    "Cycle",
    "battery_label_3class",
    "battery_label_3name",
}

IMPUTE_ZERO_COLS = [
    "charge_V_end",
    "take_off_V_end",
    "hover_V_end",
    "cruise_V_end",
    "landing_V_end",
    "standby_V_end",
    "standby_t_end_h",
    "landing_V_rel_global",
]


def make_preprocessor(feature_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_and_split(csv_path, label_col, group_col, test_size, seed):
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    # Save groups BEFORE dropping
    groups = df[group_col].astype(str).values

    # Clean labels
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    valid_mask = ~df[label_col].isna()
    df = df.loc[valid_mask].reset_index(drop=True)
    groups = groups[valid_mask]
    y = df[label_col].astype(int).values

    # Impute specified columns with 0
    for c in IMPUTE_ZERO_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Drop unwanted columns from features
    X_df = df.drop(columns=[label_col], errors="ignore").copy()
    drop_cols = set(DROP_COLS_FROM_X)
    drop_cols.add(group_col)
    X_df = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns], errors="ignore")

    # Drop boolean columns
    bool_cols = [c for c in X_df.columns if X_df[c].dtype == bool]
    X_df = X_df.drop(columns=bool_cols, errors="ignore")

    # Keep numeric only
    X_df = X_df.select_dtypes(include=[np.number]).copy()
    X_df = X_df.dropna(axis=1, how="all")

    # Group split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X_df, y, groups=groups))

    return X_df, y, train_idx, test_idx


def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ("pre", make_preprocessor(list(X_train.columns))),
        ("clf", model),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    return acc, macro_f1, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="all_csv_for_training")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--label_col", type=str, default="cycle_label_3class")
    ap.add_argument("--group_col", type=str, default="file")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary_rows = []

    for fname in args.files:
        csv_path = os.path.join(args.data_dir, fname)

        print("\n" + "#" * 80)
        print(f"FILE: {fname}")

        X_df, y, train_idx, test_idx = load_and_split(
            csv_path, args.label_col, args.group_col,
            args.test_size, args.seed
        )

        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

        acc, macro_f1, y_pred = evaluate_model(
            rf, X_train, X_test, y_train, y_test
        )

        print("\nRandomForest")
        print(classification_report(y_test, y_pred, digits=4))

        summary_rows.append({
            "CSV": fname,
            "Model": "RandomForest",
            "Accuracy": round(acc, 4),
            "Macro_F1": round(macro_f1, 4),
        })

        # XGBoost
        if HAS_XGB:
            n_classes = len(np.unique(y))
            xgb = XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=args.seed,
                n_jobs=-1,
            )

            acc, macro_f1, y_pred = evaluate_model(
                xgb, X_train, X_test, y_train, y_test
            )

            print("\nXGBoost")
            print(classification_report(y_test, y_pred, digits=4))

            summary_rows.append({
                "CSV": fname,
                "Model": "XGBoost",
                "Accuracy": round(acc, 4),
                "Macro_F1": round(macro_f1, 4),
            })

    # Final Summary Table
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
