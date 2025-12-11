#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train RF, XGB, and MLP on battery-level 3-class labels.
Input: battery_level_3class_summary.csv
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Target
    y = df["battery_label_3class"].values

    # Features: basic counts & ratios
    feature_cols = [
        "n_cycles", "n_bad", "n_good_not_drone", "n_drone_ready",
        "bad_ratio", "drone_ratio"
    ]
    X = df[feature_cols].values

    # Train/test split (simple)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    # ---- RF ----
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["RF"] = (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro"))
    print("\n=== RandomForest ===")
    print("Acc:", results["RF"][0])
    print("F1:", results["RF"][1])
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # ---- XGB ----
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss"
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        results["XGB"] = (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro"))
        print("\n=== XGBoost ===")
        print("Acc:", results["XGB"][0])
        print("F1:", results["XGB"][1])
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    # ---- MLP ----
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
    ])
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    results["MLP"] = (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro"))
    print("\n=== MLP ===")
    print("Acc:", results["MLP"][0])
    print("F1:", results["MLP"][1])
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
