#!/usr/bin/env python3
"""
build_cycle_master.py

Rebuild a clean cycle-level master CSV from raw battery Excel files.

What it does
------------
- Reads raw Excel files from `data/`
- Uses the `step` sheet as the primary source
- Skips formation cycles (Cycle Index < start_cycle)
- Requires each kept cycle to have:
    * exactly 1 charge step
    * exactly 5 discharge steps
- If any kept cycle in a file has fewer than 5 discharge steps, skips the whole file
- Assigns discharge steps in order:
    1 -> take_off
    2 -> hover
    3 -> cruise
    4 -> landing
    5 -> standby
- Extracts per-step:
    * start/end voltage
    * step duration
    * capacity
    * specific capacity
    * energy
- Computes cycle summary metrics:
    * charge/discharge cap and energy
    * CE
    * IR_proxy
    * missing count
    * CE_dev_abs
- Applies 2-class GOOD/BAD rules
- Keeps placeholder columns for 3-class labels

Output
------
A clean CSV with one row per valid cycle.

Example
-------
python build_cycle_master.py
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
    module="openpyxl"
)


DEFAULT_CFG = {
    "DRONE_IR_MAX": 0.442,
    "DRONE_DCHG_SPEC_MIN": 140.0,
    "DRONE_CE_LOW": 0.97,
    "DRONE_CE_HIGH": 1.03,
    "DRONE_CHARGE_V_END_MIN": 4.28,
    "DRONE_STANDBY_V_END_MIN": 2.73,
    "DRONE_LANDING_V_REL_GLOBAL_MIN": 0.52,

    "HARD_CHG_SPEC_CAP_MAX": 190.0,
    "HARD_DCHG_SPEC_CAP_MIN": 120.0,
    "CE_SOFT_LOW": 0.95,
    "CE_SOFT_HIGH": 1.05,
    "IR_PROXY": 0.385,
    "SCORE_START": 100,
    "SCORE_GOOD_THRESHOLD": 70,
    "PENALTIES": {
        "CE_SOFT": 10,
        "IR_OUTLIER": 20,
        "CE_EXCESS_PER_0p01": 2,
    },
    "START_FROM_CYCLE": 4,
    "SKIP_LAST_CYCLE": True,
}

PHASE_NAMES = ["take_off", "hover", "cruise", "landing", "standby"]


# ============================================================
# Helpers
# ============================================================

def norm_text(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    x = re.sub(r"[^a-z0-9_\.]+", "", x)
    x = re.sub(r"_+", "_", x)
    return x


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_file_stem(name: str) -> str:
    stem = Path(name).stem
    stem = stem.replace("_labeled_3class", "")
    stem = stem.replace("_labeled", "")
    return stem


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lookup = {norm_text(c): c for c in df.columns}
    for cand in candidates:
        key = norm_text(cand)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Could not find any of columns: {candidates}")
    return None


def choose_sheet_name(xl: pd.ExcelFile, target: str) -> str:
    for s in xl.sheet_names:
        if norm_text(s) == norm_text(target):
            return s
    raise KeyError(f"Sheet '{target}' not found. Available: {xl.sheet_names}")


def parse_step_sheet(step_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the raw step sheet columns.
    """
    col_cycle = find_col(step_df, ["Cycle Index"])
    col_step_idx = find_col(step_df, ["Step Index"])
    col_step_num = find_col(step_df, ["Step Number"], required=False)
    col_step_type = find_col(step_df, ["Step Type"])
    col_step_time = find_col(step_df, ["Step Time(h)"])
    col_onset_date = find_col(step_df, ["Oneset Date"], required=False)
    col_end_date = find_col(step_df, ["End Date"], required=False)
    col_cap = find_col(step_df, ["Capacity(Ah)"], required=False)
    col_spec = find_col(step_df, ["Spec. Cap.(mAh/g)"], required=False)
    col_chg_cap = find_col(step_df, ["Chg. Cap.(Ah)"], required=False)
    col_chg_spec = find_col(step_df, ["Chg. Spec. Cap.(mAh/g)"], required=False)
    col_dchg_cap = find_col(step_df, ["DChg. Cap.(Ah)"], required=False)
    col_dchg_spec = find_col(step_df, ["DChg. Spec. Cap.(mAh/g)"], required=False)
    col_energy = find_col(step_df, ["Energy(Wh)"], required=False)
    col_v_start = find_col(step_df, ["Oneset Volt.(V)"], required=False)
    col_v_end = find_col(step_df, ["End Voltage(V)"], required=False)

    out = pd.DataFrame({
        "Cycle": safe_num(step_df[col_cycle]),
        "Step_Index": safe_num(step_df[col_step_idx]),
        "Step_Number": safe_num(step_df[col_step_num]) if col_step_num else np.nan,
        "Step_Type": step_df[col_step_type].astype(str).str.strip(),
        "Step_Time_h": safe_num(step_df[col_step_time]),
        "Oneset_Date": step_df[col_onset_date] if col_onset_date else pd.NaT,
        "End_Date": step_df[col_end_date] if col_end_date else pd.NaT,
        "Capacity_Ah": safe_num(step_df[col_cap]) if col_cap else np.nan,
        "Spec_Cap_mAhg": safe_num(step_df[col_spec]) if col_spec else np.nan,
        "Chg_Cap_Ah": safe_num(step_df[col_chg_cap]) if col_chg_cap else np.nan,
        "Chg_Spec_mAhg": safe_num(step_df[col_chg_spec]) if col_chg_spec else np.nan,
        "DChg_Cap_Ah": safe_num(step_df[col_dchg_cap]) if col_dchg_cap else np.nan,
        "DChg_Spec_mAhg": safe_num(step_df[col_dchg_spec]) if col_dchg_spec else np.nan,
        "Energy_Wh": safe_num(step_df[col_energy]) if col_energy else np.nan,
        "V_start": safe_num(step_df[col_v_start]) if col_v_start else np.nan,
        "V_end": safe_num(step_df[col_v_end]) if col_v_end else np.nan,
    })

    out = out.dropna(subset=["Cycle", "Step_Index"]).copy()
    out["Cycle"] = out["Cycle"].astype(int)
    out["Step_Index"] = out["Step_Index"].astype(int)
    out["Step_Type_norm"] = out["Step_Type"].map(norm_text)

    return out.sort_values(["Cycle", "Step_Index"]).reset_index(drop=True)


def parse_cycle_sheet(cycle_df: pd.DataFrame) -> pd.DataFrame:
    col_cycle = find_col(cycle_df, ["Cycle Index"])
    col_chg_cap = find_col(cycle_df, ["Chg. Cap.(Ah)"], required=False)
    col_chg_spec = find_col(cycle_df, ["Chg. Spec. Cap.(mAh/g)"], required=False)
    col_dchg_cap = find_col(cycle_df, ["DChg. Cap.(Ah)"], required=False)
    col_dchg_spec = find_col(cycle_df, ["DChg. Spec. Cap.(mAh/g)"], required=False)
    col_chg_energy = find_col(cycle_df, ["Chg. Energy(Wh)", "Chg Energy(Wh)"], required=False)
    col_dchg_energy = find_col(cycle_df, ["DChg. Energy(Wh)", "DChg Energy(Wh)"], required=False)

    out = pd.DataFrame({
        "Cycle": safe_num(cycle_df[col_cycle]),
        "Chg_Cap_Ah_cycle": safe_num(cycle_df[col_chg_cap]) if col_chg_cap else np.nan,
        "Chg_Spec_mAhg_cycle": safe_num(cycle_df[col_chg_spec]) if col_chg_spec else np.nan,
        "DChg_Cap_Ah_cycle": safe_num(cycle_df[col_dchg_cap]) if col_dchg_cap else np.nan,
        "DChg_Spec_mAhg_cycle": safe_num(cycle_df[col_dchg_spec]) if col_dchg_spec else np.nan,
        "Chg_Energy_Wh_cycle": safe_num(cycle_df[col_chg_energy]) if col_chg_energy else np.nan,
        "DChg_Energy_Wh_cycle": safe_num(cycle_df[col_dchg_energy]) if col_dchg_energy else np.nan,
    })

    out = out.dropna(subset=["Cycle"]).copy()
    out["Cycle"] = out["Cycle"].astype(int)
    return out.sort_values("Cycle").reset_index(drop=True)


def is_charge_step(step_type_norm: str) -> bool:
    return ("chg" in step_type_norm) and ("dchg" not in step_type_norm)


def is_discharge_step(step_type_norm: str) -> bool:
    return "dchg" in step_type_norm


def classify_cycle_good_bad(
    row: pd.Series,
    cfg: Dict,
) -> Dict[str, object]:
    """
    Apply 2-class GOOD/BAD logic.
    """
    chg_spec = row.get("Chg_Spec_mAhg", np.nan)
    dchg_spec = row.get("DChg_Spec_mAhg", np.nan)
    chg_cap = row.get("Chg_Cap_Ah", np.nan)
    dchg_cap = row.get("DChg_Cap_Ah", np.nan)
    chg_en = row.get("Chg_Energy_Wh", np.nan)
    dchg_en = row.get("DChg_Energy_Wh", np.nan)
    ce = row.get("CE", np.nan)
    ir_proxy = row.get("IR_proxy", np.nan)

    essential = [chg_cap, dchg_cap, chg_en, dchg_en, ce]
    missing_count = sum(pd.isna(v) for v in essential)

    hf_chg_spec_high = pd.notna(chg_spec) and (chg_spec > cfg["HARD_CHG_SPEC_CAP_MAX"])
    hf_dchg_spec_low = pd.notna(dchg_spec) and (dchg_spec < cfg["HARD_DCHG_SPEC_CAP_MIN"])
    hf_missing = missing_count > 0

    hard_fail = bool(hf_chg_spec_high or hf_dchg_spec_low or hf_missing)

    soft_score = cfg["SCORE_START"]

    sp_ir_outlier = pd.notna(ir_proxy) and (ir_proxy > cfg["IR_PROXY"])
    if sp_ir_outlier:
        soft_score -= cfg["PENALTIES"]["IR_OUTLIER"]

    sp_ce_soft = False
    ce_dev_abs = np.nan
    if pd.notna(ce):
        ce_dev_abs = abs(ce - 1.0)
        if (ce < cfg["CE_SOFT_LOW"]) or (ce > cfg["CE_SOFT_HIGH"]):
            sp_ce_soft = True
            soft_score -= cfg["PENALTIES"]["CE_SOFT"]

            # extra penalty for farther deviation
            low = cfg["CE_SOFT_LOW"]
            high = cfg["CE_SOFT_HIGH"]
            if ce < low:
                excess = low - ce
            else:
                excess = ce - high
            extra_steps = int(np.floor(excess / 0.01))
            soft_score -= extra_steps * cfg["PENALTIES"]["CE_EXCESS_PER_0p01"]

    label_2class = "BAD" if (hard_fail or soft_score < cfg["SCORE_GOOD_THRESHOLD"]) else "GOOD"

    return {
        "HF_CHG_SPEC_HIGH": bool(hf_chg_spec_high),
        "HF_DCHG_SPEC_LOW": bool(hf_dchg_spec_low),
        "HF_MISSING": bool(hf_missing),
        "hard_fail": bool(hard_fail),
        "soft_score": float(soft_score),
        "SP_IR_OUTLIER": bool(sp_ir_outlier),
        "SP_CE_SOFT": bool(sp_ce_soft),
        "IR_proxy_threshold": float(cfg["IR_PROXY"]),
        "Missing_Count": int(missing_count),
        "CE_dev_abs": ce_dev_abs,
        "Label": label_2class,
    }


def classify_cycle_3class(
    row: pd.Series,
    cfg: Dict,
) -> Dict[str, object]:
    """
    Split GOOD into GOOD_drone vs GOOD_not_drone using mission-readiness criteria.
    BAD stays BAD.
    """
    if row.get("Label", None) == "BAD":
        return {
            "cycle_label_3class": 0,
            "cycle_label_3name": "BAD",
        }

    is_low_ir = pd.notna(row.get("IR_proxy")) and (row["IR_proxy"] <= cfg["DRONE_IR_MAX"])
    is_high_capacity = pd.notna(row.get("DChg_Spec_mAhg")) and (row["DChg_Spec_mAhg"] >= cfg["DRONE_DCHG_SPEC_MIN"])
    ce_ideal = (
        pd.notna(row.get("CE")) and
        (cfg["DRONE_CE_LOW"] <= row["CE"] <= cfg["DRONE_CE_HIGH"])
    )
    charge_full = pd.notna(row.get("charge_V_end")) and (row["charge_V_end"] >= cfg["DRONE_CHARGE_V_END_MIN"])
    standby_ok = pd.notna(row.get("standby_V_end")) and (row["standby_V_end"] >= cfg["DRONE_STANDBY_V_END_MIN"])
    landing_ok = (
        pd.notna(row.get("landing_V_rel_global")) and
        (row["landing_V_rel_global"] >= cfg["DRONE_LANDING_V_REL_GLOBAL_MIN"])
    )

    if is_low_ir and is_high_capacity and ce_ideal and charge_full and standby_ok and landing_ok:
        return {
            "cycle_label_3class": 2,
            "cycle_label_3name": "GOOD_drone",
        }

    return {
        "cycle_label_3class": 1,
        "cycle_label_3name": "GOOD_not_drone",
    }

def build_cycle_row(
    file_name: str,
    cycle_num: int,
    charge_step: pd.Series,
    discharge_steps: pd.DataFrame,
    cycle_metrics: Optional[Dict[str, object]],
    cfg: Dict,
) -> Dict[str, object]:
    """
    Build one cycle row from step-sheet data.
    discharge_steps must already be ordered by Step_Index and have length 5.
    """
    row: Dict[str, object] = {
        "file": file_name,
        "Cycle": int(cycle_num),
    }

    # Charge summary
    row["charge_step_index"] = int(charge_step["Step_Index"])
    row["charge_t_h"] = charge_step["Step_Time_h"]
    row["charge_V_start"] = charge_step["V_start"]
    row["charge_V_end"] = charge_step["V_end"]

    # Keep step-derived values as fallback only
    row["Chg_Cap_Ah_step"] = charge_step["Chg_Cap_Ah"]
    row["Chg_Spec_mAhg_step"] = charge_step["Chg_Spec_mAhg"]
    row["Chg_Energy_Wh_step"] = charge_step["Energy_Wh"]

    # Five named discharge phases
    dchg_cap_total = 0.0
    dchg_energy_total = 0.0
    dchg_cap_any = False
    dchg_energy_any = False

    last_dchg_spec = np.nan

    for phase_name, (_, step_row) in zip(PHASE_NAMES, discharge_steps.iterrows()):
        row[f"{phase_name}_step_index"] = int(step_row["Step_Index"])
        row[f"{phase_name}_t_h"] = step_row["Step_Time_h"]
        row[f"{phase_name}_V_start"] = step_row["V_start"]
        row[f"{phase_name}_V_end"] = step_row["V_end"]

        row[f"{phase_name}_DChg_Cap_Ah"] = step_row["DChg_Cap_Ah"]
        row[f"{phase_name}_DChg_Spec_mAhg"] = step_row["DChg_Spec_mAhg"]
        row[f"{phase_name}_Energy_Wh"] = step_row["Energy_Wh"]

        if pd.notna(step_row["DChg_Cap_Ah"]):
            dchg_cap_total += float(step_row["DChg_Cap_Ah"])
            dchg_cap_any = True

        if pd.notna(step_row["Energy_Wh"]):
            dchg_energy_total += float(step_row["Energy_Wh"])
            dchg_energy_any = True

        # Use the LAST discharge step's specific capacity as cycle-level discharge specific capacity
        if pd.notna(step_row["DChg_Spec_mAhg"]):
            last_dchg_spec = float(step_row["DChg_Spec_mAhg"])

    # Step-derived totals kept for audit/fallback
    row["DChg_Cap_Ah_step_total"] = dchg_cap_total if dchg_cap_any else np.nan
    row["DChg_Spec_mAhg_step_last"] = last_dchg_spec
    row["DChg_Energy_Wh_step_total"] = dchg_energy_total if dchg_energy_any else np.nan

    # Use cycle-sheet totals for labeling metrics whenever available
    if cycle_metrics is not None:
        row["Chg_Cap_Ah"] = cycle_metrics.get("Chg_Cap_Ah_cycle", np.nan)
        row["Chg_Spec_mAhg"] = cycle_metrics.get("Chg_Spec_mAhg_cycle", np.nan)
        row["DChg_Cap_Ah"] = cycle_metrics.get("DChg_Cap_Ah_cycle", np.nan)
        row["DChg_Spec_mAhg"] = cycle_metrics.get("DChg_Spec_mAhg_cycle", np.nan)
        row["Chg_Energy_Wh"] = cycle_metrics.get("Chg_Energy_Wh_cycle", np.nan)
        row["DChg_Energy_Wh"] = cycle_metrics.get("DChg_Energy_Wh_cycle", np.nan)
    else:
        row["Chg_Cap_Ah"] = row["Chg_Cap_Ah_step"]
        row["Chg_Spec_mAhg"] = row["Chg_Spec_mAhg_step"]
        row["DChg_Cap_Ah"] = row["DChg_Cap_Ah_step_total"]
        row["DChg_Spec_mAhg"] = row["DChg_Spec_mAhg_step_last"]
        row["Chg_Energy_Wh"] = row["Chg_Energy_Wh_step"]
        row["DChg_Energy_Wh"] = row["DChg_Energy_Wh_step_total"]

    # CE
    if pd.notna(row["Chg_Cap_Ah"]) and row["Chg_Cap_Ah"] not in [0, 0.0] and pd.notna(row["DChg_Cap_Ah"]):
        row["CE"] = row["DChg_Cap_Ah"] / row["Chg_Cap_Ah"]
    else:
        row["CE"] = np.nan

    # IR proxy = (Wh/Ah)_charge - (Wh/Ah)_discharge
    charge_wh_per_ah = np.nan
    dchg_wh_per_ah = np.nan

    if pd.notna(row["Chg_Energy_Wh"]) and pd.notna(row["Chg_Cap_Ah"]) and row["Chg_Cap_Ah"] not in [0, 0.0]:
        charge_wh_per_ah = row["Chg_Energy_Wh"] / row["Chg_Cap_Ah"]

    if pd.notna(row["DChg_Energy_Wh"]) and pd.notna(row["DChg_Cap_Ah"]) and row["DChg_Cap_Ah"] not in [0, 0.0]:
        dchg_wh_per_ah = row["DChg_Energy_Wh"] / row["DChg_Cap_Ah"]

    if pd.notna(charge_wh_per_ah) and pd.notna(dchg_wh_per_ah):
        row["IR_proxy"] = charge_wh_per_ah - dchg_wh_per_ah
    else:
        row["IR_proxy"] = np.nan

    # Voltage / time features used later in 3-class logic
    row["take_off_V_end"] = row["take_off_V_end"]
    row["hover_V_end"] = row["hover_V_end"]
    row["cruise_V_end"] = row["cruise_V_end"]
    row["landing_V_end"] = row["landing_V_end"]
    row["standby_V_end"] = row["standby_V_end"]
    row["standby_t_end_h"] = (
        row["take_off_t_h"] +
        row["hover_t_h"] +
        row["cruise_t_h"] +
        row["landing_t_h"] +
        row["standby_t_h"]
        if all(pd.notna(row.get(k)) for k in ["take_off_t_h", "hover_t_h", "cruise_t_h", "landing_t_h", "standby_t_h"])
        else np.nan
    )

    # landing_V_rel_global: one useful interpretation = landing end voltage relative to charge end voltage
    if pd.notna(row["landing_V_end"]) and pd.notna(row["charge_V_end"]):
        row["landing_V_rel_global"] = row["landing_V_end"] / row["charge_V_end"]
    else:
        row["landing_V_rel_global"] = np.nan

    # Apply GOOD/BAD rules
    row.update(classify_cycle_good_bad(pd.Series(row), cfg))

    # Final 3-class split
    row.update(classify_cycle_3class(pd.Series(row), cfg))

    row["battery_label_3class"] = np.nan
    row["battery_label_3name"] = np.nan

    return row


def process_one_file(
    xlsx_path: Path,
    cfg: Dict,
) -> Tuple[Optional[pd.DataFrame], List[Dict[str, object]]]:
    """
    Process one raw battery Excel file.
    Returns:
      - cycle dataframe or None if file skipped
      - issues list
    """
    issues: List[Dict[str, object]] = []
    file_name = xlsx_path.name

    try:
        xl = pd.ExcelFile(xlsx_path)
        step_sheet = choose_sheet_name(xl, "step")
        cycle_sheet = choose_sheet_name(xl, "cycle")

        step_raw = pd.read_excel(xl, sheet_name=step_sheet)
        cycle_raw = pd.read_excel(xl, sheet_name=cycle_sheet)

        step_df = parse_step_sheet(step_raw)
        cycle_df = parse_cycle_sheet(cycle_raw)

        # FIX: define cycle_lookup here
        if not cycle_df.empty:
            cycle_lookup = cycle_df.set_index("Cycle").to_dict("index")
        else:
            cycle_lookup = {}

    except Exception as e:
        issues.append({
            "file": file_name,
            "Cycle": None,
            "reason": f"read_error:{type(e).__name__}:{e}",
        })
        return None, issues

    if step_df.empty:

        issues.append({
            "file": file_name,
            "Cycle": None,
            "reason": "empty_step_sheet",
        })
        return None, issues

    cycle_lookup = cycle_df.set_index("Cycle").to_dict("index") if not cycle_df.empty else {}

    cycles = sorted(c for c in step_df["Cycle"].unique() if c >= cfg["START_FROM_CYCLE"])

    if not cycles:
        issues.append({
            "file": file_name,
            "Cycle": None,
            "reason": "no_cycles_after_start",
        })
        return None, issues

    if cfg["SKIP_LAST_CYCLE"] and len(cycles) >= 1:
        cycles = cycles[:-1]

    if not cycles:
        issues.append({
            "file": file_name,
            "Cycle": None,
            "reason": "all_cycles_removed_after_skip_last",
        })
        return None, issues

    # First pass: check file-level structural validity
    # If any kept cycle has <5 discharge steps, skip whole file
    cycle_blocks: Dict[int, Dict[str, pd.DataFrame]] = {}

    for cyc in cycles:
        cdf = step_df.loc[step_df["Cycle"] == cyc].sort_values("Step_Index").copy()

        charge_steps = cdf.loc[cdf["Step_Type_norm"].map(is_charge_step)].copy()
        dchg_steps = cdf.loc[cdf["Step_Type_norm"].map(is_discharge_step)].copy()

        cycle_blocks[cyc] = {
            "charge_steps": charge_steps,
            "dchg_steps": dchg_steps,
            "all": cdf,
        }

        if len(dchg_steps) < 5:
            issues.append({
                "file": file_name,
                "Cycle": cyc,
                "reason": f"skip_file_cycle_has_lt5_dchg:{len(dchg_steps)}",
            })
            return None, issues

    # Second pass: keep only exact 1 charge + exact 5 discharge cycles
    rows = []
    for cyc in cycles:
        charge_steps = cycle_blocks[cyc]["charge_steps"]
        dchg_steps = cycle_blocks[cyc]["dchg_steps"]

        if len(charge_steps) != 1:
            issues.append({
                "file": file_name,
                "Cycle": cyc,
                "reason": f"skip_cycle_charge_count_{len(charge_steps)}",
            })
            continue

        if len(dchg_steps) != 5:
            issues.append({
                "file": file_name,
                "Cycle": cyc,
                "reason": f"skip_cycle_dchg_count_{len(dchg_steps)}",
            })
            continue

        charge_step = charge_steps.iloc[0]
        dchg_steps = dchg_steps.sort_values("Step_Index").copy()

        row = build_cycle_row(
            file_name=file_name,
            cycle_num=cyc,
            charge_step=charge_step,
            discharge_steps=dchg_steps,
            cycle_metrics=cycle_lookup.get(cyc, None),
            cfg=cfg,
        )
        rows.append(row)

    if not rows:
        issues.append({
            "file": file_name,
            "Cycle": None,
            "reason": "no_valid_cycles_after_exact_structure_filter",
        })
        return None, issues

    out_df = pd.DataFrame(rows).sort_values(["file", "Cycle"]).reset_index(drop=True)
    return out_df, issues


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean cycle-level master CSV from raw battery Excel files."
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Folder containing original raw battery Excel files.",
    )
    parser.add_argument(
        "--all_csv_dir",
        type=Path,
        default=Path("all_csv_for_training"),
        help="Folder containing prior training CSVs; also used as default output location.",
    )
    parser.add_argument(
        "--relabeled_dir",
        type=Path,
        default=Path("drones_label_out"),
        help="Folder containing relabeled Excel files (not required for this first script, but kept in skeleton).",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("all_csv_for_training") / "ALL_cycles_master_from_raw.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--issues_csv",
        type=Path,
        default=Path("all_csv_for_training") / "ALL_cycles_master_from_raw_issues.csv",
        help="CSV path for file/cycle issues log.",
    )

    parser.add_argument(
        "--start_cycle",
        type=int,
        default=DEFAULT_CFG["START_FROM_CYCLE"],
        help="Start from this cycle index (skip formation cycles).",
    )
    parser.add_argument(
        "--skip_last_cycle",
        action="store_true",
        default=DEFAULT_CFG["SKIP_LAST_CYCLE"],
        help="Skip the last cycle in each file.",
    )

    parser.add_argument(
        "--hard_chg_spec_cap_max",
        type=float,
        default=DEFAULT_CFG["HARD_CHG_SPEC_CAP_MAX"],
        help="Hard fail threshold: charge specific capacity > this => BAD.",
    )
    parser.add_argument(
        "--hard_dchg_spec_cap_min",
        type=float,
        default=DEFAULT_CFG["HARD_DCHG_SPEC_CAP_MIN"],
        help="Hard fail threshold: discharge specific capacity < this => BAD.",
    )
    parser.add_argument(
        "--ce_soft_low",
        type=float,
        default=DEFAULT_CFG["CE_SOFT_LOW"],
        help="Soft CE lower threshold.",
    )
    parser.add_argument(
        "--ce_soft_high",
        type=float,
        default=DEFAULT_CFG["CE_SOFT_HIGH"],
        help="Soft CE upper threshold.",
    )
    parser.add_argument(
        "--ir_proxy_thresh",
        type=float,
        default=DEFAULT_CFG["IR_PROXY"],
        help="Soft IR-proxy threshold.",
    )
    parser.add_argument(
        "--score_start",
        type=float,
        default=DEFAULT_CFG["SCORE_START"],
        help="Initial cycle score.",
    )
    parser.add_argument(
        "--score_good_threshold",
        type=float,
        default=DEFAULT_CFG["SCORE_GOOD_THRESHOLD"],
        help="Score threshold for GOOD vs BAD.",
    )
    parser.add_argument(
        "--penalty_ce_soft",
        type=float,
        default=DEFAULT_CFG["PENALTIES"]["CE_SOFT"],
        help="Penalty when CE outside soft band.",
    )
    parser.add_argument(
        "--penalty_ir_outlier",
        type=float,
        default=DEFAULT_CFG["PENALTIES"]["IR_OUTLIER"],
        help="Penalty when IR_proxy exceeds threshold.",
    )
    parser.add_argument(
        "--penalty_ce_excess_per_0p01",
        type=float,
        default=DEFAULT_CFG["PENALTIES"]["CE_EXCESS_PER_0p01"],
        help="Extra CE penalty per 0.01 beyond CE soft bound.",
    )

    parser.add_argument(
        "--drone_ir_max",
        type=float,
        default=DEFAULT_CFG["DRONE_IR_MAX"],
        help="Maximum IR_proxy for GOOD_drone.",
    )
    parser.add_argument(
        "--drone_dchg_spec_min",
        type=float,
        default=DEFAULT_CFG["DRONE_DCHG_SPEC_MIN"],
        help="Minimum discharge specific capacity for GOOD_drone.",
    )
    parser.add_argument(
        "--drone_ce_low",
        type=float,
        default=DEFAULT_CFG["DRONE_CE_LOW"],
        help="Lower CE bound for GOOD_drone.",
    )
    parser.add_argument(
        "--drone_ce_high",
        type=float,
        default=DEFAULT_CFG["DRONE_CE_HIGH"],
        help="Upper CE bound for GOOD_drone.",
    )
    parser.add_argument(
        "--drone_charge_v_end_min",
        type=float,
        default=DEFAULT_CFG["DRONE_CHARGE_V_END_MIN"],
        help="Minimum charge end voltage for GOOD_drone.",
    )
    parser.add_argument(
        "--drone_standby_v_end_min",
        type=float,
        default=DEFAULT_CFG["DRONE_STANDBY_V_END_MIN"],
        help="Minimum standby end voltage for GOOD_drone.",
    )
    parser.add_argument(
        "--drone_landing_v_rel_global_min",
        type=float,
        default=DEFAULT_CFG["DRONE_LANDING_V_REL_GLOBAL_MIN"],
        help="Minimum landing_V_rel_global for GOOD_drone.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = {
        "DRONE_IR_MAX": args.drone_ir_max,
        "DRONE_DCHG_SPEC_MIN": args.drone_dchg_spec_min,
        "DRONE_CE_LOW": args.drone_ce_low,
        "DRONE_CE_HIGH": args.drone_ce_high,
        "DRONE_CHARGE_V_END_MIN": args.drone_charge_v_end_min,
        "DRONE_STANDBY_V_END_MIN": args.drone_standby_v_end_min,
        "DRONE_LANDING_V_REL_GLOBAL_MIN": args.drone_landing_v_rel_global_min,

        "HARD_CHG_SPEC_CAP_MAX": args.hard_chg_spec_cap_max,
        "HARD_DCHG_SPEC_CAP_MIN": args.hard_dchg_spec_cap_min,
        "CE_SOFT_LOW": args.ce_soft_low,
        "CE_SOFT_HIGH": args.ce_soft_high,
        "IR_PROXY": args.ir_proxy_thresh,
        "SCORE_START": args.score_start,
        "SCORE_GOOD_THRESHOLD": args.score_good_threshold,
        "PENALTIES": {
            "CE_SOFT": args.penalty_ce_soft,
            "IR_OUTLIER": args.penalty_ir_outlier,
            "CE_EXCESS_PER_0p01": args.penalty_ce_excess_per_0p01,
        },
        "START_FROM_CYCLE": args.start_cycle,
        "SKIP_LAST_CYCLE": args.skip_last_cycle,
    }

    data_dir: Path = args.data_dir
    out_csv: Path = args.out_csv
    issues_csv: Path = args.issues_csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    issues_csv.parent.mkdir(parents=True, exist_ok=True)

    excel_files = sorted(list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls")))
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in: {data_dir}")

    all_rows = []
    all_issues = []

    for fp in excel_files:
        df_one, issues = process_one_file(fp, cfg)
        all_issues.extend(issues)
        if df_one is not None and not df_one.empty:
            all_rows.append(df_one)

    if not all_rows:
        issues_df = pd.DataFrame(all_issues)
        issues_df.to_csv(issues_csv, index=False)
        raise RuntimeError(
            f"No valid cycle rows produced. Check {issues_csv} for details."
        )

    final_df = pd.concat(all_rows, axis=0, ignore_index=True)
    final_df = final_df.sort_values(["file", "Cycle"]).reset_index(drop=True)

    # Nice column order
    preferred_cols = [
        "file", "Cycle",
        "Chg_Spec_mAhg", "DChg_Spec_mAhg", "Chg_Cap_Ah", "DChg_Cap_Ah",
        "Chg_Energy_Wh", "DChg_Energy_Wh", "CE", "IR_proxy",
        "HF_CHG_SPEC_HIGH", "HF_DCHG_SPEC_LOW", "HF_MISSING", "hard_fail",
        "soft_score", "SP_IR_OUTLIER", "SP_CE_SOFT", "IR_proxy_threshold",
        "Label",
        "charge_V_end", "take_off_V_end", "hover_V_end", "cruise_V_end",
        "landing_V_end", "standby_V_end", "standby_t_end_h", "landing_V_rel_global",
        "cycle_label_3class", "cycle_label_3name",
        "battery_label_3class", "battery_label_3name",
        "Missing_Count", "CE_dev_abs",
    ]

    existing_preferred = [c for c in preferred_cols if c in final_df.columns]
    remaining = [c for c in final_df.columns if c not in existing_preferred]
    final_df = final_df[existing_preferred + remaining]

    final_df.to_csv(out_csv, index=False)

    issues_df = pd.DataFrame(all_issues)
    issues_df.to_csv(issues_csv, index=False)

    print(f"Saved cycle master CSV: {out_csv}")
    print(f"Saved issues log:       {issues_csv}")
    print(f"Valid cycle rows:       {len(final_df)}")
    print(f"Unique files used:      {final_df['file'].nunique()}")

    label_counts = final_df["Label"].value_counts(dropna=False)
    print("\n2-class label counts:")
    print(label_counts.to_string())

    label3_counts = final_df["cycle_label_3name"].value_counts(dropna=False)
    print("\n3-class label counts:")
    print(label3_counts.to_string())


if __name__ == "__main__":
    main()