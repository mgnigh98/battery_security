import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix



full_data = pd.read_csv('./all_csv_for_training/ALL_cycles_3class_early.csv')

def prepare_data(use_base=True, use_early=True, early_num=None):
    base_cols = full_data.columns[1:9]
    if early_num is None:
                early_cols = full_data.columns[full_data.columns.str.contains('early')]
    else:
        early_cols = full_data.columns[full_data.columns.str.contains(f'early{early_num}_')]
    early_cols = early_cols[~early_cols.str.contains('flag')]
    early_cols = early_cols[~early_cols.str.contains('missing')]
    cols = base_cols.append(early_cols)
    if use_base and use_early:
        data = full_data[cols].dropna(axis=0)
    elif use_base:
        data = full_data[base_cols].dropna(axis=0)
    elif use_early:
        data = full_data[early_cols].dropna(axis=0)
    data = data.loc[:, data.nunique()>100]
    return data
rf = RandomForestClassifier(n_estimators=100, random_state=42)

results = pd.DataFrame(columns=["use_base", "use_early", "early_num", "accuracy", "f1_score"])

early_nums = [None, 1,2,5,10,20,30,50,60]
for use_base, use_early, early_num in [(True, False, None)]+[(use_base, True, early_num) for early_num in early_nums for use_base in [True, False] ] :
    data = prepare_data(use_base=use_base, use_early=use_early, early_num=early_num)
    X_train, X_test, y_train, y_test = train_test_split(data, full_data.loc[data.index, "cycle_label_3class"], train_size=0.70, random_state=42)
    results.loc[len(results)] = [use_base, use_early, early_num, rf.fit(X_train, y_train).score(X_test, y_test), f1_score(y_test, rf.predict(X_test), average='macro')]

results.loc[len(results), results.columns[3:]] = results.iloc[:, 3:].mean()
results.index = results.index[:-1].append(pd.Index(["mean"]))
print(results)



