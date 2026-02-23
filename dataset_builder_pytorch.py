import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PHASES = ["takeoff", "hover", "cruise", "landing", "standby"]

def one_hot(idx, n=5):
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v

def assign_phase_ids(record_df, dchg_steps_df):
    """
    record_df must include 'Date' as datetime
    dchg_steps_df must include Oneset Date, End Date as datetime and be sorted
    returns phase_id array aligned with record_df
    """
    phase_id = np.full(len(record_df), -1, dtype=np.int64)
    dates = record_df["Date"].values.astype("datetime64[ns]")

    for k, row in enumerate(dchg_steps_df.itertuples(index=False)):
        start = np.datetime64(getattr(row, "Oneset Date"))
        end   = np.datetime64(getattr(row, "End Date"))
        mask = (dates >= start) & (dates <= end)
        phase_id[mask] = k  # 0..4

    return phase_id

def build_sequence(record_df, phase_id, T_sec):
    """
    record_df is already filtered to discharge mission rows for this cycle
    returns X (n, d) float32
    """
    # keep only points with valid phase
    m = phase_id >= 0
    record_df = record_df.loc[m].copy()
    phase_id = phase_id[m]
    if len(record_df) < 5:
        return None

    # mission-relative time
    t0 = record_df["Date"].iloc[0]
    t_sec = (record_df["Date"] - t0).dt.total_seconds().values.astype(np.float32)

    # truncate early window
    keep = t_sec <= float(T_sec)
    record_df = record_df.loc[keep].copy()
    phase_id = phase_id[keep]
    t_sec = t_sec[keep]
    if len(record_df) < 5:
        return None

    V = record_df["Voltage(V)"].astype(float).values.astype(np.float32)
    I = record_df["Current(A)"].astype(float).values.astype(np.float32)

    # dV/dt
    # (avoid division by zero if duplicate timestamps)
    dt = np.diff(t_sec, prepend=t_sec[0])
    dV = np.diff(V, prepend=V[0])
    dt[dt == 0] = 1e-6
    dvdt = (dV / dt).astype(np.float32)

    t_norm = (t_sec / float(T_sec)).astype(np.float32)

    # stack per-step features
    feats = []
    for i in range(len(V)):
        feats.append(np.concatenate([
            np.array([t_norm[i], V[i], I[i], dvdt[i]], dtype=np.float32),
            one_hot(int(phase_id[i]), 5)
        ]))
    X = np.stack(feats, axis=0).astype(np.float32)  # (n, 9)
    return X

class CyclePhaseSeqDataset(Dataset):
    """
    Each item: (X_seq, y, group) where:
      X_seq: (n, 9) float32 variable length
      y: int
      group: file_id (for group split)
    """
    def __init__(self, index_df, T_sec):
        self.df = index_df.reset_index(drop=True)
        self.T_sec = int(T_sec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        X = row["X"]           # numpy (n, d)
        y = int(row["y"])
        g = row["group"]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long), g


def collate_pad(batch):
    # batch: list of (X_seq, y, group)
    Xs, ys, groups = zip(*batch)
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
    return X_pad, mask, y, groups
