import os, glob, re
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


DROP_ALWAYS = {
    "Cycle",
    "battery_label_3class", "battery_label_3name",
    "cycle_label_3name",
    # keep "file" only for grouping, not features
    "file",
}

# columns you said are often empty; impute with 0 if present
EMPTY_PHASE_COLS = [
    "charge_V_end","take_off_V_end","hover_V_end","cruise_V_end",
    "landing_V_end","standby_V_end","standby_t_end_h","landing_V_rel_global"
]


def canonical_group_file(fname: str) -> str:
    """Group by original identity, but make labeled/unlabeled match."""
    base = os.path.basename(str(fname))
    if base.lower().endswith(".xlsx"):
        stem = base[:-5].replace("_labeled", "")
        return stem + ".xlsx"
    # if already looks like a name from CSV
    return str(fname).replace("_labeled", "").strip()


def find_balanced_group_split(y, groups, seed=42, test_size=0.3, max_tries=500):
    """Try many GroupShuffleSplit seeds until train/test contain all classes; else >=2 classes."""
    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    idx = np.arange(len(y))
    classes = set(y.tolist())

    best = None
    # strict
    for t in range(max_tries):
        rs = seed + t
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr, te = next(gss.split(idx, y, groups=groups))
        if set(y[tr].tolist()) == classes and set(y[te].tolist()) == classes:
            return tr, te, rs, "all_classes"
        best = (tr, te, rs)

    # fallback >=2
    for t in range(max_tries):
        rs = seed + t
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr, te = next(gss.split(idx, y, groups=groups))
        if len(set(y[tr].tolist())) >= 2 and len(set(y[te].tolist())) >= 2:
            return tr, te, rs, ">=2_classes"

    # last resort: return first split
    tr, te, rs = best
    return tr, te, rs, "unbalanced"


def make_models(random_state=42):
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    models = [("RF", rf)]

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        models.append(("XGB", xgb))

    return models


def run_one_csv(csv_path, label_col="cycle_label_3class", test_size=0.3, seed=42):
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"{csv_path}: missing label col {label_col}")

    # label
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    # group file key
    if "file" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'file' column for group split")
    df["group_file"] = df["file"].astype(str).apply(canonical_group_file)

    # impute empties
    for c in EMPTY_PHASE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # drop boolean-like columns (TRUE/FALSE strings) + actual bool dtype
    bool_cols = []
    for c in df.columns:
        if df[c].dtype == bool:
            bool_cols.append(c)
        elif df[c].dtype == object:
            vals = set(df[c].dropna().astype(str).str.upper().unique().tolist()[:20])
            if vals.issubset({"TRUE", "FALSE"}):
                bool_cols.append(c)

    # features: numeric columns only, excluding drop list + label + group
    drop_cols = set(DROP_ALWAYS) | set(bool_cols) | {label_col, "group_file"}
    feat_cols = [c for c in df.columns if c not in drop_cols]

    # coerce numeric
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[label_col].values
    groups = df["group_file"].values

    # sanity
    overall_counts = Counter(y.tolist())

    tr, te, used_seed, split_mode = find_balanced_group_split(y, groups, seed=seed, test_size=test_size)

    X_train, X_test = X_df.iloc[tr].values, X_df.iloc[te].values
    y_train, y_test = y[tr], y[te]

    # scaler for tree models isn't necessary, but you asked earlier and it helps keep consistent
    scaler = StandardScaler()

    results = []
    for name, model in make_models(random_state=seed):
        pipe = Pipeline([
            ("scaler", scaler),
            ("clf", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro") if len(set(y_test.tolist())) > 1 else float("nan")

        results.append((name, acc, mf1))

        print("\n" + "="*80)
        print(f"{os.path.basename(csv_path)} | {name}")
        print(f"Split: {split_mode}, seed={used_seed}")
        print("Overall label counts:", dict(overall_counts))
        print("Train label counts:", dict(Counter(y_train.tolist())))
        print("Test  label counts:", dict(Counter(y_test.tolist())))
        print(f"Accuracy={acc:.4f} | Macro-F1={mf1:.4f}")
        print(classification_report(y_test, pred, digits=4))

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, default="all_csv_for_training", help="Folder with enriched CSVs")
    ap.add_argument("--pattern", type=str, default="early_*s_enriched.csv")
    ap.add_argument("--label_col", type=str, default="cycle_label_3class")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSVs found: {os.path.join(args.folder, args.pattern)}")

    table_rows = []
    for p in paths:
        res = run_one_csv(p, label_col=args.label_col, test_size=args.test_size, seed=args.seed)
        for model_name, acc, mf1 in res:
            table_rows.append({
                "csv": os.path.basename(p),
                "model": model_name,
                "accuracy": round(acc, 4),
                "macro_f1": round(mf1, 4) if np.isfinite(mf1) else np.nan,
            })

    out = pd.DataFrame(table_rows)
    print("\n" + "#"*80)
    print("SUMMARY TABLE")
    print("#"*80)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
