# phase_token_gru_run_fixed.py
#
# FIXED + SIMPLIFIED (paste and run):
# - Uses ALL_cycles_3class.csv for labels (cycle_label_3class) keyed by (file, Cycle)
# - Reads Excel raw files: step + record
# - Skips formation cycles: Cycle < 4
# - Cycle-level filter: keep cycles that have exactly 5 "CC DChg" steps (skip only bad cycles)
# - Phase assignment is TIME-BASED using record Time(h) and step Step Time(h) (NO timestamp matching!)
# - Builds variable-length early-window sequences with phase one-hot:
#     [t_norm, V, I, dVdt, onehot(5)]
# - Group split by file (no leakage) with a robust retry; if impossible, it falls back gracefully
# - Caches each window dataset to cache/phase_seq_T{T}.pt
#
# Run:
#   python phase_token_gru_run_fixed.py --excel_dir /path/to/original_excels \
#     --labels_csv all_csv_for_training/ALL_cycles_3class.csv \
#     --windows 1 2 5 10 20 30 50 60 --epochs 8 --batch_size 32

import os
import glob
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
    module="openpyxl.styles.stylesheet",
)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import WeightedRandomSampler


# -----------------------------
# Aliases
# -----------------------------
STEP_CYCLE_COLS  = ["Cycle Index", "Cycle_Index"]
STEP_TYPE_COLS   = ["Step Type", "Step_Type"]
STEP_TIMEH_COLS  = ["Step Time(h)", "Step_Time(h)", "Step Time (h)"]
STEP_NUM_COLS    = ["Step Number", "Step_Number", "StepIndex", "Step Index", "Step_Index"]

REC_CYCLE_COLS   = ["Cycle Index", "Cycle_Index"]
REC_TIMEH_COLS   = ["Time(h)", "Time (h)"]
REC_V_COLS       = ["Voltage(V)", "Voltage (V)", "Voltage"]
REC_I_COLS       = ["Current(A)", "Current (A)", "Current"]

PHASES = ["takeoff", "hover", "cruise", "landing", "standby"]  # 5 discharge blocks


# -----------------------------
# Helpers
# -----------------------------
def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None found from {candidates}. Available (first 40): {list(df.columns)[:40]}")

def canonical_file_key(fname: str) -> str:
    """Map *_labeled.xlsx -> *.xlsx so label CSV matches original raw file names."""
    base = os.path.basename(str(fname)).strip()
    if base.lower().endswith(".xlsx"):
        stem = base[:-5].replace("_labeled", "")
        return stem + ".xlsx"
    return base.replace("_labeled", "")

def save_cached_dataset(cache_path, X_list, y_list, g_list, T_sec):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "T_sec": int(T_sec),
        "X_list": X_list,  # list of np arrays (n_i, 9)
        "y_list": np.array(y_list, dtype=np.int64),
        "g_list": list(g_list),
        "feature_desc": ["t_norm", "V", "I", "dVdt", "phase_onehot_5"],
        "phase_names": PHASES,
    }
    torch.save(payload, cache_path)

def load_cached_dataset(cache_path):
    payload = torch.load(cache_path, map_location="cpu")
    X_list = payload["X_list"]
    y_list = payload["y_list"].tolist() if hasattr(payload["y_list"], "tolist") else payload["y_list"]
    g_list = payload["g_list"]
    return X_list, y_list, g_list, payload


# -----------------------------
# Label map
# -----------------------------
def build_label_map(labels_csv: str) -> dict:
    labels = pd.read_csv(labels_csv)
    required = {"Cycle", "cycle_label_3class", "file"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")

    labels["Cycle"] = pd.to_numeric(labels["Cycle"], errors="coerce")
    labels["cycle_label_3class"] = pd.to_numeric(labels["cycle_label_3class"], errors="coerce")

    print("CSV cycle_label_3class value counts:")
    print(labels["cycle_label_3class"].value_counts(dropna=False).head(10))

    label_map = {}
    for r in labels.itertuples(index=False):
        f = getattr(r, "file")
        cyc = getattr(r, "Cycle")
        lab = getattr(r, "cycle_label_3class")
        if pd.isna(f) or pd.isna(cyc) or pd.isna(lab):
            continue
        key = (canonical_file_key(str(f)), int(cyc))
        label_map[key] = int(lab)
    return label_map


# -----------------------------
# Read step + record
# -----------------------------
def read_excel_step_record(fpath: str):
    step = pd.read_excel(fpath, sheet_name="step")
    rec  = pd.read_excel(fpath, sheet_name="record")

    # step cols
    scyc = pick_col(step, STEP_CYCLE_COLS)
    styp = pick_col(step, STEP_TYPE_COLS)
    stim = pick_col(step, STEP_TIMEH_COLS)
    snum = None
    for c in STEP_NUM_COLS:
        if c in step.columns:
            snum = c
            break

    # record cols
    rcyc = pick_col(rec, REC_CYCLE_COLS)
    rtim = pick_col(rec, REC_TIMEH_COLS)
    rv   = pick_col(rec, REC_V_COLS)
    ri   = pick_col(rec, REC_I_COLS)

    step = step.rename(columns={scyc: "Cycle_Index", styp: "Step_Type", stim: "Step_Time_h"})
    if snum is not None:
        step = step.rename(columns={snum: "Step_Number"})
    else:
        step["Step_Number"] = np.arange(len(step), dtype=int)

    rec  = rec.rename(columns={rcyc: "Cycle_Index", rtim: "Time_h", rv: "VoltageV", ri: "CurrentA"})

    # numeric coercion
    # step["Cycle_Index"] = pd.to_numeric(step["Cycle_Index"], errors="coerce")
    # step["Step_Time_h"] = pd.to_numeric(step["Step_Time_h"], errors="coerce")
    # rec["Cycle_Index"]  = pd.to_numeric(rec["Cycle_Index"], errors="coerce")
    rec["Time_h"]       = pd.to_numeric(rec["Time_h"], errors="coerce")

    # clean cycle indices
    step["Cycle_Index"] = pd.to_numeric(step["Cycle_Index"], errors="coerce")
    step = step.dropna(subset=["Cycle_Index"])
    step["Cycle_Index"] = step["Cycle_Index"].astype(int)

    rec["Cycle_Index"] = pd.to_numeric(rec["Cycle_Index"], errors="coerce")
    rec = rec.dropna(subset=["Cycle_Index"])
    rec["Cycle_Index"] = rec["Cycle_Index"].astype(int)

    # drop bad
    step = step.dropna(subset=["Cycle_Index", "Step_Type", "Step_Time_h"]).copy()
    rec  = rec.dropna(subset=["Cycle_Index", "Time_h", "VoltageV", "CurrentA"]).copy()

    # ensure dtypes
    step["Step_Type"] = step["Step_Type"].astype(str)
    rec["VoltageV"] = pd.to_numeric(rec["VoltageV"], errors="coerce")
    rec["CurrentA"] = pd.to_numeric(rec["CurrentA"], errors="coerce")
    rec = rec.dropna(subset=["VoltageV", "CurrentA"]).copy()

    return step, rec


# -----------------------------
# Phase assignment (time-based)
# -----------------------------
def assign_phase_ids_timebased(rec_cyc: pd.DataFrame, dchg_steps: pd.DataFrame) -> np.ndarray:
    """
    dchg_steps must have exactly 5 rows with Step_Time_h in hours, in order.
    rec_cyc has Time_h in hours from cycle start.
    We assign phase by cumulative discharge-step durations.
    """
    dur = dchg_steps["Step_Time_h"].values.astype(float)
    dur = np.nan_to_num(dur, nan=0.0)

    # If durations are bogus, fail
    if (dur <= 0).sum() >= 2:
        return np.full(len(rec_cyc), -1, dtype=np.int64)

    edges = np.concatenate([[0.0], np.cumsum(dur)])  # length 6
    t = rec_cyc["Time_h"].values.astype(float)
    t = np.nan_to_num(t, nan=-1.0)

    pid = np.full(len(t), -1, dtype=np.int64)
    for k in range(5):
        a, b = edges[k], edges[k + 1]
        pid[(t >= a) & (t < b)] = k
    return pid

def assign_phase_ids_timebased_with_offset(step_cyc: pd.DataFrame, rec_cyc: pd.DataFrame):
    """
    step_cyc: all steps for this cycle (with Step_Number, Step_Time_h, Step_Type)
    rec_cyc : all record rows for this cycle (with Time_h)
    Returns:
      phase_id (len rec_cyc), t0_h (discharge start hour), t_end_h (discharge end hour)
    """
    step_cyc = step_cyc.sort_values("Step_Number").copy()

    dchg = step_cyc[step_cyc["Step_Type"] == "CC DChg"].copy()
    if len(dchg) != 5:
        return np.full(len(rec_cyc), -1, dtype=np.int64), None, None

    # find discharge start offset = sum of step times BEFORE first CC DChg
    first_dchg_pos = step_cyc.index.get_loc(dchg.index[0])
    pre = step_cyc.iloc[:first_dchg_pos]
    t0_h = float(np.nan_to_num(pre["Step_Time_h"].sum(), nan=0.0))

    dur = dchg["Step_Time_h"].values.astype(float)
    dur = np.nan_to_num(dur, nan=0.0)
    if (dur <= 0).sum() >= 2:
        return np.full(len(rec_cyc), -1, dtype=np.int64), None, None

    edges = t0_h + np.concatenate([[0.0], np.cumsum(dur)])  # length 6
    t_end_h = float(edges[-1])

    t = rec_cyc["Time_h"].values.astype(float)
    t = np.nan_to_num(t, nan=-1.0)

    pid = np.full(len(t), -1, dtype=np.int64)
    for k in range(5):
        a, b = edges[k], edges[k + 1]
        pid[(t >= a) & (t < b)] = k

    return pid, t0_h, t_end_h



# def build_sequence_timebased(rec_cyc: pd.DataFrame, phase_id: np.ndarray, T_sec: int, min_points: int) -> np.ndarray | None:
#     """
#     Variable-length early window sequence based on Time_h.
#     Features per timestep: [t_norm, V, I, dVdt, onehot(5)] => (n, 9)
#     """
#     m = phase_id >= 0
#     if int(m.sum()) < min_points:
#         return None
#
#     rec_cyc = rec_cyc.loc[m].copy()
#     phase_id = phase_id[m]
#
#     # Time to seconds, normalized to start at 0
#     t_sec = (rec_cyc["Time_h"].values.astype(np.float32) * 3600.0)
#     t_sec = t_sec - float(np.nanmin(t_sec))
#
#     keep = t_sec <= float(T_sec)
#     if int(keep.sum()) < min_points:
#         return None
#
#     rec_cyc = rec_cyc.loc[keep].copy()
#     phase_id = phase_id[keep]
#     t_sec = t_sec[keep]
#
#     V = rec_cyc["VoltageV"].values.astype(np.float32)
#     I = rec_cyc["CurrentA"].values.astype(np.float32)
#
#     ok = np.isfinite(V) & np.isfinite(I) & np.isfinite(t_sec)
#     V, I, t_sec, phase_id = V[ok], I[ok], t_sec[ok], phase_id[ok]
#     if len(V) < min_points:
#         return None
#
#     # --- normalize per sequence ---
#     V0 = float(V[0])
#     V_rel = V - V0
#
#     I_abs = np.abs(I)
#
#     # dv/dt on V_rel
#     dt = np.diff(t_sec, prepend=t_sec[0])
#     dV = np.diff(V_rel, prepend=V_rel[0])
#     dt[dt == 0] = 1e-6
#     dvdt = (dV / dt).astype(np.float32)
#
#     # optional z-score (helps a lot)
#     v_std = float(np.std(V_rel) + 1e-6)
#     i_std = float(np.std(I_abs) + 1e-6)
#     Vn = (V_rel / v_std).astype(np.float32)
#     In = (I_abs / i_std).astype(np.float32)
#
#
#     # # dv/dt
#     # dt = np.diff(t_sec, prepend=t_sec[0])
#     # dV = np.diff(V, prepend=V[0])
#     # dt[dt == 0] = 1e-6
#     # dvdt = (dV / dt).astype(np.float32)
#
#     t_norm = (t_sec / float(max(1, T_sec))).astype(np.float32)
#
#     # X: keep same 9 dims by swapping in normalized channels
#     X = np.zeros((len(V), 9), dtype=np.float32)
#     X[:, 0] = t_norm
#     X[:, 1] = Vn  # normalized relative voltage
#     X[:, 2] = In  # normalized abs current
#     X[:, 3] = dvdt  # slope of relative voltage
#     for i in range(len(V)):
#         p = int(phase_id[i])
#         if 0 <= p < 5:
#             X[i, 4 + p] = 1.0
#     return X

def build_sequence_timebased(rec_cyc: pd.DataFrame, phase_id: np.ndarray,
                             T_sec: int, min_points: int,
                             t0_h: float) -> np.ndarray | None:
    """
    Early window starts at discharge start t0_h.
    """
    m = phase_id >= 0
    if m.sum() < min_points:
        return None

    rec_cyc = rec_cyc.loc[m].copy()
    phase_id = phase_id[m]

    # time since discharge start
    t_sec = ((rec_cyc["Time_h"].values.astype(np.float32) - float(t0_h)) * 3600.0)

    keep = (t_sec >= 0) & (t_sec <= float(T_sec))

    min_points = 1 if T_sec == 1 else (2 if T_sec <= 5 else 8)

    if keep.sum() < min_points:
        return None

    rec_cyc = rec_cyc.loc[keep].copy()
    phase_id = phase_id[keep]
    t_sec = t_sec[keep]


    V = rec_cyc["VoltageV"].values.astype(np.float32)
    I = rec_cyc["CurrentA"].values.astype(np.float32)

    ok = np.isfinite(V) & np.isfinite(I) & np.isfinite(t_sec)
    V, I, t_sec, phase_id = V[ok], I[ok], t_sec[ok], phase_id[ok]
    if len(V) < min_points:
        return None

    # ---- (keep your normalized version here) ----
    V0 = float(V[0])
    V_rel = V - V0
    I_abs = np.abs(I)

    if len(V) == 1:
        dvdt = np.zeros_like(V, dtype=np.float32)
    else:
        dt = np.diff(t_sec, prepend=t_sec[0])
        dV = np.diff(V_rel, prepend=V_rel[0])
        dt[dt == 0] = 1e-6
        dvdt = (dV / dt).astype(np.float32)

    v_std = float(np.std(V_rel) + 1e-6)
    i_std = float(np.std(I_abs) + 1e-6)
    Vn = (V_rel / v_std).astype(np.float32)
    In = (I_abs / i_std).astype(np.float32)

    t_norm = (t_sec / float(max(1, T_sec))).astype(np.float32)

    X = np.zeros((len(V), 9), dtype=np.float32)
    X[:, 0] = t_norm
    X[:, 1] = Vn
    X[:, 2] = In
    X[:, 3] = dvdt
    for i in range(len(V)):
        p = int(phase_id[i])
        if 0 <= p < 5:
            X[i, 4 + p] = 1.0
    return X


# -----------------------------
# Dataset + model
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X_list, y_list, group_list):
        self.X = X_list
        self.y = y_list
        self.g = group_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(int(self.y[idx]), dtype=torch.long), self.g[idx]

def collate_pad(batch):
    Xs, ys, gs = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in Xs], dtype=torch.long)
    d = Xs[0].shape[1]
    max_len = int(lengths.max().item())

    X_pad = torch.zeros(len(Xs), max_len, d, dtype=torch.float32)
    mask = torch.zeros(len(Xs), max_len, dtype=torch.bool)
    for i, x in enumerate(Xs):
        n = x.shape[0]
        X_pad[i, :n] = x
        mask[i, :n] = True

    y = torch.stack(ys)
    return X_pad, mask, y, gs

class GRUClassifier(nn.Module):
    def __init__(self, in_dim=9, hidden=64, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden,
                          batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

        # self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True)
        # self.head = nn.Sequential(
        #     nn.Linear(hidden, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, num_classes),
        # )

    def forward(self, X_pad, mask):
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(X_pad, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = torch.cat([h[0], h[1]], dim=1)  # (B, 2*hidden)
        return self.head(h)
        # _, h = self.gru(packed)  # (1, B, hidden)
        # h = h.squeeze(0)
        # return self.head(h)


# -----------------------------
# Build sequences
# -----------------------------
def build_sequences(excel_dir: str, label_map: dict, T_sec: int, verbose: bool = True):
    excel_paths = sorted(glob.glob(os.path.join(excel_dir, "*.xlsx")))
    if not excel_paths:
        raise FileNotFoundError(f"No .xlsx files found in: {excel_dir}")

    X_list, y_list, g_list = [], [], []
    used_files = 0
    skipped_read = 0
    skipped_nolabel = 0
    skipped_short = 0
    skipped_bad_cycle = 0

    # min points: crucial for 1–5s windows
    min_points = 3 if T_sec <= 5 else 8

    for fpath in excel_paths:
        file_id = os.path.basename(fpath)
        file_key = canonical_file_key(file_id)

        try:
            step, rec = read_excel_step_record(fpath)
        except Exception as e:
            skipped_read += 1
            if verbose:
                print(f"[SKIP READ] {file_id}: {e}")
            continue

        used_files += 1

        # Iterate cycles in this file
        # cycle_indices = sorted(step["Cycle_Index"].dropna().unique())
        cycle_indices = sorted(step["Cycle_Index"].unique().tolist())

        for cyc in cycle_indices:
            cyc = int(cyc)
            if cyc < 4:
                continue

            # label lookup (by canonical file key + cycle)
            y = label_map.get((file_key, cyc), None)
            if y is None:
                skipped_nolabel += 1
                continue

            # 5 CC DChg steps (phase blocks)
            dchg = step[(step["Cycle_Index"] == cyc) & (step["Step_Type"] == "CC DChg")].copy()
            if "Step_Number" in dchg.columns:
                dchg = dchg.sort_values("Step_Number")
            else:
                dchg = dchg.sort_index()

            if len(dchg) != 5:
                skipped_bad_cycle += 1
                continue

            # record rows for this cycle
            rec_cyc = rec[rec["Cycle_Index"] == cyc].copy()
            if len(rec_cyc) < min_points:
                skipped_short += 1
                continue

            # phase ids based on Time(h) boundaries
            # phase_id = assign_phase_ids_timebased(rec_cyc, dchg)
            #
            # # build early window sequence
            # X = build_sequence_timebased(rec_cyc, phase_id, T_sec=T_sec, min_points=min_points)

            step_cyc = step[step["Cycle_Index"] == cyc].copy()
            phase_id, t0_h, t_end_h = assign_phase_ids_timebased_with_offset(step_cyc, rec_cyc)
            if t0_h is None:
                skipped_bad_cycle += 1
                continue

            X = build_sequence_timebased(rec_cyc, phase_id, T_sec=T_sec, min_points=min_points, t0_h=t0_h)

            if X is None:
                skipped_short += 1
                continue

            X_list.append(X)
            y_list.append(int(y))
            g_list.append(file_key)  # group split uses original file identity (no leakage)

    if verbose:
        print(f"[T={T_sec}s] Used files: {used_files}")
        print(f"[T={T_sec}s] Samples built: {len(X_list)}")
        print(f"[T={T_sec}s] Skipped read errors: {skipped_read}")
        print(f"[T={T_sec}s] Skipped no label: {skipped_nolabel}")
        print(f"[T={T_sec}s] Skipped too short/invalid: {skipped_short}")
        print(f"[T={T_sec}s] Skipped bad cycles (not 5 DChg): {skipped_bad_cycle}")

    return X_list, y_list, g_list


# -----------------------------
# Group split + train/eval
# -----------------------------
def find_group_split_retry(
    y_arr,
    g_arr,
    seed=42,
    test_size=0.3,
    max_tries=400,
    min_test_per_class=20,   # <-- important
):
    """
    Try to find a group split where:
      - train and test each have >=2 classes
      - test has at least `min_test_per_class` samples for every class present
    If impossible, return first split and mark as unbalanced.
    """

    idx_all = np.arange(len(y_arr))
    best = None

    for t in range(max_tries):
        rs = seed + t
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr, te = next(gss.split(idx_all, y_arr, groups=g_arr))

        y_train = y_arr[tr]
        y_test  = y_arr[te]

        train_classes = set(y_train.tolist())
        test_classes  = set(y_test.tolist())

        # require >=2 classes in both
        if len(train_classes) < 2 or len(test_classes) < 2:
            continue

        # require minimum samples per class in test
        test_counts = Counter(y_test.tolist())
        if min(test_counts.values()) < min_test_per_class:
            continue

        return tr, te, rs, "balanced"

        # keep first split as fallback
        if best is None:
            best = (tr, te, rs)

    # fallback if no good split found
    if best is not None:
        tr, te, rs = best
        return tr, te, rs, "unbalanced"

    raise RuntimeError("Could not generate any group split.")


def train_eval_gru(
    X_list, y_list, g_list,
    seed=42, test_size=0.3,
    epochs=8, batch_size=32, lr=1e-3,
):
    idx_all = np.arange(len(X_list))
    y_arr = np.array(y_list, dtype=int)
    g_arr = np.array(g_list, dtype=object)

    # classes_present = sorted(set(y_arr.tolist()))
    # num_classes = int(max(classes_present) + 1) if classes_present else 3
    # num_classes = max(num_classes, 3)  # keep output head size stable at 3
    num_classes = 3

    tr, te, used_seed, mode = find_group_split_retry(y_arr, g_arr, seed=seed, test_size=test_size)


    X_train = [X_list[i] for i in tr]
    y_train = y_arr[tr].tolist()
    g_train = g_arr[tr].tolist()

    X_test = [X_list[i] for i in te]
    y_test = y_arr[te].tolist()
    g_test = g_arr[te].tolist()

    train_files = set(g_train)
    test_files = set(g_test)
    overlap = len(train_files.intersection(test_files))

    print(f"\nSplit mode: {mode} | seed used: {used_seed}")
    print("Train label counts:", dict(Counter(y_train)))
    print("Test  label counts:", dict(Counter(y_test)))
    print(f"Train files: {len(train_files)} | Test files: {len(test_files)} | Overlap: {overlap}")

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print("[WARN] Train or Test has <2 classes. Metrics may be misleading for this window.")

    train_ds = SeqDataset(X_train, y_train, g_train)
    test_ds = SeqDataset(X_test, y_test, g_test)

    class_counts = Counter(y_train)
    w = np.array([1.0 / class_counts[int(c)] for c in y_train], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=torch.tensor(w), num_samples=len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_pad)

    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = GRUClassifier(in_dim=9, hidden=64, num_classes=num_classes).to(device)
    #
    # # class weights from train
    # counts = np.bincount(np.array(y_train, dtype=int), minlength=num_classes).astype(np.float32)
    # weights = counts.sum() / (counts + 1e-6)
    # weights = torch.tensor(weights, dtype=torch.float32).to(device)
    model = GRUClassifier(in_dim=9, hidden=64, num_classes=3).to(device)

    counts = np.bincount(np.array(y_train, dtype=int), minlength=3).astype(np.float32)
    weights = counts.sum() / (counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for X_pad, mask, y_batch, _ in train_loader:
            X_pad = X_pad.to(device)
            mask = mask.to(device)
            y_batch = y_batch.to(device)

            optim.zero_grad()
            logits = model(X_pad, mask)
            loss = criterion(logits, y_batch)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())

        print(f"  epoch {ep}/{epochs} loss={total_loss / max(1, len(train_loader)):.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_pad, mask, y_batch, _ in test_loader:
            X_pad = X_pad.to(device)
            mask = mask.to(device)
            logits = model(X_pad, mask)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()

            y_pred.extend(pred)
            y_true.extend(y_batch.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], digits=4, zero_division=0)

    meta = (len(train_files), len(test_files), len(y_train), len(y_test), used_seed, mode)
    return acc, mf1, report, overlap, meta


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_dir", type=str, required=True, help="Folder containing original .xlsx files")
    ap.add_argument("--labels_csv", type=str, default="all_csv_for_training/ALL_cycles_3class.csv")
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 5, 30, 60])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--no_cache", action="store_true", help="Disable cache and rebuild sequences")
    args = ap.parse_args()

    print("Loading labels...")
    label_map = build_label_map(args.labels_csv)
    if len(label_map) == 0:
        raise RuntimeError("Label map is empty. Check labels_csv path and columns.")

    for T in args.windows:
        print("\n" + "#" * 90)
        print(f"BUILD + TRAIN for T={T}s")

        cache_path = f"cache/phase_seq_T{T}.pt"
        if (not args.no_cache) and os.path.exists(cache_path):
            print(f"Loading cached dataset: {cache_path}")
            X_list, y_list, g_list, _ = load_cached_dataset(cache_path)
        else:
            X_list, y_list, g_list = build_sequences(args.excel_dir, label_map, T_sec=T, verbose=True)
            if not args.no_cache:
                save_cached_dataset(cache_path, X_list, y_list, g_list, T_sec=T)

        if len(X_list) < 50:
            print(f"[WARN] Too few samples for T={T}s ({len(X_list)}). Skipping.")
            continue

        print("Overall label counts:", dict(Counter(y_list)))
        print("Total unique files:", len(set(g_list)))

        print(f"Training GRU (variable-length + phase tokens) for T={T}s ...")
        acc, mf1, report, overlap, meta = train_eval_gru(
            X_list, y_list, g_list,
            seed=args.seed,
            test_size=args.test_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        n_train_files, n_test_files, n_train, n_test, used_seed, split_mode = meta

        print(f"\n[T={T}s] File overlap (should be 0): {overlap}")
        print(f"[T={T}s] Split mode: {split_mode} | seed used: {used_seed}")
        print(f"[T={T}s] Train files={n_train_files}, Test files={n_test_files}, Train samples={n_train}, Test samples={n_test}")
        print(f"[T={T}s] Accuracy={acc:.4f} | Macro-F1={mf1:.4f}")
        print(report)


if __name__ == "__main__":
    main()
