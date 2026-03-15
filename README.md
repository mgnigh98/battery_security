# Battery Security: Early Anomaly Detection for eUAM Batteries

This repository contains the data processing pipeline and machine learning models used to study **early detection of abnormal battery cycles** for electric Urban Air Mobility (eUAM) systems.

The goal is to determine whether a battery cycle is:

- **GOOD_drone** – suitable for drone/eUAM operation
- **GOOD_not_drone** – operational but not suitable for eUAM mission profiles
- **BAD** – unsafe or degraded battery behavior

The models focus on **early-cycle detection**, using only the first few seconds of battery data (2s–60s) to predict the final cycle label.

---

# Repository Structure



battery_security/
│
├── scripts
│
│ ├── build_cycle_master.py
│ ├── build_timeseries_wide.py
│ ├── build_early_features_from_prefix.py
│ ├── build_final_early_datasets.py
│ └── train_final_early_models.py
│
├── legacy_labeling
│
│ ├── cycle_label_v4.py
│ ├── cycle_stats_advanced.py
│ ├── good_step_end_voltage_from_step_v3.py
│ ├── label_3class_drone_v2.py
│ └── step_end_voltage_all_cycles_v1.py
│
├── all_csv_for_training
│
│ ├── ALL_cycles_master_from_raw.csv
│ ├── cycle_timeseries_wide_t0_60.csv
│ └── final_early_model_data
│
│ ├── final_early_2s.csv
│ ├── final_early_10s.csv
│ ├── final_early_30s.csv
│ ├── final_early_60s.csv
│ ├── feature_counts.csv
│ └── feature_list_*.txt
│
├── .gitignore
└── README.md


---

# Data Processing Pipeline

The full workflow converts raw battery cycling logs into structured machine-learning datasets.

## Step 1 — Build master cycle dataset

python scripts/build_cycle_master.py

This script merges raw cycle data and labeling information into:"ALL_cycles_master_from_raw.csv"

## Step 2 — Convert time-series into wide format
python scripts/build_timeseries_wide.py

Creates a time-aligned dataset containing the first 0–60 seconds of cycle measurements:

cycle_timeseries_wide_t0_60.csv

## Step 3 — Extract early-window features
python scripts/build_early_features_from_prefix.py

Computes early-cycle features such as:

voltage sag
dV/dt statistics
current statistics
early internal resistance proxy
power and energy features

These are extracted for multiple windows:

2 seconds
10 seconds
30 seconds
60 seconds

## Step 4 — Build final training datasets
python scripts/build_final_early_datasets.py

This produces the final ML datasets:

final_early_2s.csv
final_early_10s.csv
final_early_30s.csv
final_early_60s.csv

| Window | Features |
| -----: | -------: |
|     2s |       97 |
|    10s |      131 |
|    30s |      165 |
|    60s |      199 |

## Machine Learning Models

The following models are trained and evaluated:

Random Forest
XGBoost

## Training script:

python scripts/train_final_early_models.py

This script:
* performs group-based train/test split by battery file
* trains models on each time window
* evaluates accuracy and F1 score
* generates plots and metrics summaries

## Example Results

Typical performance observed:
| Window | Accuracy | Macro F1 |
| -----: | -------: | -------: |
|     2s |    ~0.86 |    ~0.84 |
|    10s |    ~0.86 |    ~0.85 |
|    30s |    ~0.87 |    ~0.86 |
|    60s |    ~0.89 |    ~0.88 |

The results demonstrate that useful battery health signals appear within the first few seconds of operation, enabling early anomaly detection.

## Legacy Labeling Code

The legacy_labeling/ directory contains scripts used during dataset construction:
voltage-based cycle health checks
internal resistance proxy thresholds
charge/discharge efficiency checks
drone suitability labeling

These scripts are preserved for reproducibility of the labeling process.

## Installation

Recommended Python version:
Python 3.10+

## Install dependencies:
pip install pandas numpy scikit-learn xgboost matplotlib

## Research Context

This repository supports research on:
Battery security for electric Urban Air Mobility (eUAM)
Early-cycle anomaly detection
Safety-critical battery diagnostics
Machine learning for cyber-physical system reliability

The work explores how early voltage/current behavior can reveal degradation or unsafe battery states before mission deployment.

## Author

Dipali Jain
PhD Candidate – Computer Engineering
University of Texas at Dallas

## Research areas:

Hardware security
Battery security for eUAM
AI for cyber-physical systems


