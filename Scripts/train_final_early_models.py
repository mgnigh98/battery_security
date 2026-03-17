#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from xgboost import XGBClassifier

#added by M. Nigh
import time
try:
    from tim_utils.kde import KernelDensity
    from tim_utils.syn_data import alpha_beta_prec_recall
    from scipy.integrate import simpson
    from sklearn.svm import OneClassSVM, SVC
    import optuna
    HAS_SYN = True
except ImportError:
    HAS_SYN = False


GROUP_COL = "file"
LABEL_COL = "cycle_label_3name"
META_COLS = ["file", "Cycle", "Label", "cycle_label_3class", "cycle_label_3name"]

def feature_cols_for_window(df: pd.DataFrame, window_sec: int) -> list[str]:
    """
    Use only:
    - cumulative early features up to current window
    - growth features whose end window <= current window
    Drop noisy binary flags / missing indicators.
    """
    allowed = [w for w in [1, 2, 5, 10, 20, 30, 50, 60] if w <= window_sec]
    cols = []

    # cumulative early blocks
    for w in allowed:
        prefix = f"early{w}_"
        cols.extend([c for c in df.columns if c.startswith(prefix)])

    # only growth features valid up to this window
    growth_cols = []
    for c in df.columns:
        if "_growth_" not in c:
            continue
        m = re.search(r"_growth_(\d+)_to_(\d+)$", c)
        if m:
            w2 = int(m.group(2))
            if w2 <= window_sec:
                growth_cols.append(c)

    cols.extend(growth_cols)

    # drop noisy flags / missing indicators
    drop_patterns = [
        "_missing",
        "_voltage_collapse_flag",
        "_IR_high_flag",
        "_dvdt_fluct_flag",
    ]

    cols = [
        c for c in sorted(set(cols))
        if not any(p in c for p in drop_patterns)
    ]

    return cols


def infer_window_from_name(path: Path) -> int:
    stem = path.stem  # final_early_10s
    token = stem.split("_")[-1]
    return int(token.replace("s", ""))


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [GROUP_COL, LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], out_csv: Path, out_png: Path, title: str) -> None:
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_csv)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close('all')


def plot_metric_lines(summary_df: pd.DataFrame, out_dir: Path, syn=False) -> None:
    for metric in ["accuracy", "macro_f1"]:
        plt.figure(figsize=(7, 5))
        cmap = plt.get_cmap("tab10")
        for i, model_name in enumerate(sorted(summary_df["model"].unique())):
            sub = summary_df[summary_df["model"] == model_name].sort_values("window_sec")
            plt.plot(sub["window_sec"], sub[metric], marker="o", label=model_name, color=cmap(i), alpha=0.9)
            if syn:
                plt.plot(sub["window_sec"], sub[f"{metric}_syn"], marker="*", label=f"{model_name}_syn", color=cmap(i), alpha=0.5)

        plt.xlabel("Time window (s)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} vs Time Window")
        plt.xticks(sorted(summary_df["window_sec"].unique()))
        plt.ylim([.75, .95])
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"lineplot_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close('all')


def build_model(model_name: str, random_state: int):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "XGB":
        return XGBClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=4,
            reg_lambda=2.0,
            gamma=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )

    # if model_name == "XGB":
    #     return XGBClassifier(
    #         n_estimators=800,
    #         max_depth=5,
    #         learning_rate=0.04,
    #         subsample=0.8,
    #         colsample_bytree=0.7,
    #         min_child_weight=4,
    #         reg_lambda=2.0,
    #         objective="multi:softprob",
    #         eval_metric="mlogloss",
    #         random_state=random_state,
    #         n_jobs=-1,
    #     )

    if model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model: {model_name}")


def maybe_scale(model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame):
    if model_name == "MLP":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    return X_train.values, X_test.values


def train_syn(X_train, y_train, inner=0, outer=1, syn_multiply_factor=2, eval_syn=False, **kwargs):
    scaler = StandardScaler()
    alphas, betas, coverages = [], [], []
    eval_scores = {}
    for i in np.unique(y_train):
        # print(i)
        X_train_i = X_train[y_train==i]
        if outer-inner < 1:
            OCC = OneClassSVM(nu=.5, kernel="rbf", gamma="scale")
            OCC.fit(X_train_i)
            scores = OCC.score_samples(X_train_i)
            core = (scores < np.quantile(scores, inner))
            edge = (scores > np.quantile(scores, outer))
            band = ~((core)|(edge))
            X_train_i = X_train_i[band]        

        mask = (pd.DataFrame(X_train_i).nunique()>50).values # (len(X_train_i)//4)
        bws = ([.25]*len(mask)) * mask
        # print(bws)
        kde = KernelDensity(bandwidth=bws, kernel="epanechnikov")  
        kde.fit(scaler.fit_transform(X_train_i))
        X_train_syn = scaler.inverse_transform(kde.sample(syn_multiply_factor*len(X_train[y_train==i]), random_state=42))
        X_train = np.concatenate([X_train, X_train_syn], axis=0)
        y_train = np.concatenate([y_train, [i]*len(X_train_syn)])

        if eval_syn:
            ab_dict = alpha_beta_prec_recall.compute_metrics(X_train_i, X_train_syn, steps=10)
            alphas.append(round(1-2*simpson(abs(ab_dict['alphas'] - ab_dict["alpha_precision"]), ab_dict["alphas"]), 3))
            betas.append(round(1-2*simpson(abs(ab_dict['alphas'] - ab_dict["beta_recall"]), ab_dict["alphas"]), 3))
            coverages.append(round(1-2*simpson(abs(ab_dict['alphas'] - ab_dict["beta_coverage"]), ab_dict["alphas"]), 3))
    if eval_syn:
        eval_scores["alpha"] = np.mean(alphas)
        eval_scores["beta"] = np.mean(betas)
        eval_scores["coverage"] = np.mean(coverages)

    return X_train, y_train, eval_scores


def optuna_optimize_syn(X, y, groups, model=None):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[test_idx]
    y_val = y[test_idx]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        inner = trial.suggest_float("inner", 0, .5, step=0.01)
        outer = trial.suggest_float("outer", .75, 1, step=0.01)
        X_train_syn, y_train_syn, _ = train_syn(X_train, y_train, inner, outer, 2, eval_syn=False)
        nonlocal model
        if model is None:
            model = SVC(kernel="rbf", gamma="scale")
        model.fit(X_train_syn, y_train_syn)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        return acc + f1
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100)
    inner = study.best_params["inner"]
    outer = study.best_params["outer"]
    return inner, outer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("all_csv_for_training") / "final_early_model_data",
        help="Directory containing final_early_*s.csv files.",
    )
    ap.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results_final_early_models"),
        help="Directory to save model outputs.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["RF", "XGB", "MLP"],
        help="Models to train.",
    )
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--random_state", type=int, default=42)
    args, remaining = ap.parse_known_args()

    #added by M. Nigh
    syn_ap = argparse.ArgumentParser()
    syn_ap.add_argument("--use_syn", default=True, action="store_true")
    syn_ap.add_argument("--verbose", default=False, action="store_true")
    syn_ap.add_argument("--syn_multiply_factor", default=2, type=int)
    syn_ap.add_argument("--inner", default=0, type=float)
    syn_ap.add_argument("--outer", default=1, type=float)
    syn_ap.add_argument("--optimize_syn", default=False, action="store_true")
    syn_ap.add_argument("--eval_syn", default=False, action="store_true")
    syn_args = syn_ap.parse_args(remaining)
    

    if not HAS_SYN and syn_args.use_syn:
        print("[WARN] tim_utils, scipy, or optuna not installed; synthetic data will be skipped.")
        time.sleep(10) #wait for 10 seconds to read the warning
        syn_args.use_syn = False

    args.results_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(args.data_dir.glob("final_early_*s.csv"), key=infer_window_from_name)
    if not dataset_paths:
        raise FileNotFoundError(f"No final datasets found in {args.data_dir}")

    # Fixed split based on largest window for fair comparison
    ref_path = max(dataset_paths, key=infer_window_from_name)
    ref_df = load_dataset(ref_path)
    ref_df = ref_df.sort_values(["file", "Cycle"]).reset_index(drop=True)

    groups = ref_df[GROUP_COL].astype(str).values
    y_ref = ref_df[LABEL_COL].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(ref_df, y_ref, groups=groups))

    train_files = sorted(ref_df.iloc[train_idx][GROUP_COL].astype(str).unique())
    test_files = sorted(ref_df.iloc[test_idx][GROUP_COL].astype(str).unique())

    # Save split for reproducibility
    pd.DataFrame({"file": list(train_files)}).to_csv(args.results_dir / "train_files.csv", index=False)
    pd.DataFrame({"file": list(test_files)}).to_csv(args.results_dir / "test_files.csv", index=False)

    with open(args.results_dir / "split_info.json", "w") as f:
        json.dump(
            {
                "reference_dataset": str(ref_path),
                "train_files": train_files,
                "test_files": test_files,
                "test_size": args.test_size,
                "random_state": args.random_state,
            },
            f,
            indent=2,
        )

    le = LabelEncoder()
    le.fit(ref_df[LABEL_COL].astype(str).values)
    class_names = list(le.classes_)

    summary_rows = []
    feature_rows = []

    for path in dataset_paths:
        window_sec = infer_window_from_name(path)
        df = load_dataset(path)
        df = df.sort_values(["file", "Cycle"]).reset_index(drop=True)

        # feature_cols = get_feature_columns(df)
        feature_cols = feature_cols_for_window(df, window_sec)
        feature_rows.append({
            "window_sec": window_sec,
            "rows": len(df),
            "n_features": len(feature_cols),
            "files": df[GROUP_COL].nunique(),
            "input_csv": str(path),
        })

        print(f"Using {len(feature_cols)} features for {window_sec}s")

        train_df = df[df[GROUP_COL].astype(str).isin(train_files)].reset_index(drop=True)
        test_df = df[df[GROUP_COL].astype(str).isin(test_files)].reset_index(drop=True)

        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        for c in feature_cols:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)
        X_test = X_test.fillna(med)

        y_train = le.transform(train_df[LABEL_COL].astype(str).values)
        y_test = le.transform(test_df[LABEL_COL].astype(str).values)

        for model_name in args.models:
            print(f"Training {model_name} on {window_sec}s ...")

            model = build_model(model_name, args.random_state)
            X_train_arr, X_test_arr = maybe_scale(model_name, X_train, X_test)

            model.fit(X_train_arr, y_train)

            # Save feature importance for XGB
            if model_name == "XGB":
                importance = model.feature_importances_

                imp_df = pd.DataFrame({
                    "feature": feature_cols,
                    "importance": importance
                }).sort_values("importance", ascending=False)

                imp_df.to_csv(
                    args.results_dir / f"feature_importance_{window_sec}s.csv",
                    index=False
                )
            y_pred = model.predict(X_test_arr)

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")

            report = classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

            model_dir = args.results_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            pred_df = test_df[[GROUP_COL, "Cycle", LABEL_COL]].copy()
            pred_df["y_true"] = test_df[LABEL_COL].values
            pred_df["y_pred"] = le.inverse_transform(y_pred)
            pred_df.to_csv(model_dir / f"predictions_{window_sec}s.csv", index=False)

            if False:
                save_confusion_matrix(
                    cm=cm,
                    class_names=class_names,
                    out_csv=model_dir / f"confusion_matrix_{window_sec}s.csv",
                    out_png=model_dir / f"confusion_matrix_{window_sec}s.png",
                    title=f"{model_name} - {window_sec}s",
                )


            #added by M. Nigh
            if syn_args.use_syn:
                if syn_args.optimize_syn:
                    syn_args.inner, syn_args.outer = optuna_optimize_syn(X_train_arr, y_train, groups[train_idx], model=[None, model][1])
                    # print(f"Optimized inner: {inner}, outer: {outer}")
                X_train_, y_train_, scores = train_syn(X_train_arr, y_train, **syn_args.__dict__)
                model.fit(X_train_, y_train_)
                y_pred_syn = model.predict(X_test)
                acc_syn = accuracy_score(y_test, y_pred_syn)
                f1_syn = f1_score(y_test, y_pred_syn, average="macro")
                # syn_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

            row = {
                "window_sec": window_sec,
                "model": model_name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "n_features": len(feature_cols),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "train_files": len(train_files),
                "test_files": len(test_files),
            }

            for cls in class_names:
                row[f"{cls}_precision"] = report.get(cls, {}).get("precision", np.nan)
                row[f"{cls}_recall"] = report.get(cls, {}).get("recall", np.nan)
                row[f"{cls}_f1"] = report.get(cls, {}).get("f1-score", np.nan)
                row[f"{cls}_support"] = report.get(cls, {}).get("support", np.nan)
            
            if syn_args.use_syn:
                row["accuracy_syn"] = acc_syn
                row["macro_f1_syn"] = f1_syn
                row["accuracy_impr"] = acc_syn-acc
                row["macro_f1_impr"] = f1_syn-macro_f1

            summary_rows.append(row)

            print(f"{model_name} {window_sec}s -> accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "window_sec"]).reset_index(drop=True)
    feature_df = pd.DataFrame(feature_rows).drop_duplicates().sort_values("window_sec").reset_index(drop=True)

    summary_df.to_csv(args.results_dir / "metrics_summary.csv", index=False)
    feature_df.to_csv(args.results_dir / "feature_counts.csv", index=False)
    plot_metric_lines(summary_df, args.results_dir, syn_args.use_syn)

    print(f"\nSaved metrics to: {args.results_dir / 'metrics_summary.csv'}")
    print(f"Saved feature counts to: {args.results_dir / 'feature_counts.csv'}")
    print(f"Saved plots to: {args.results_dir}")


if __name__ == "__main__":
    main()
