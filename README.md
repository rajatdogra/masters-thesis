# Cricket Score Prediction vs DLS Method

Master's Thesis — Rajat Dogra

## Overview

This thesis develops a machine learning framework to predict final cricket scores in ODI matches and compares its performance against the Duckworth-Lewis-Stern (DLS) method. The project covers first innings score prediction (46 features), second innings chase projection (53 features), and a revised target-setting engine for rain-affected matches.

## Key Results

| Model | RMSE | R² | MAE |
|---|---|---|---|
| CatBoost V2 (1st innings) | 43.57 | 0.618 | 32.29 |
| LightGBM V2 (1st innings) | 43.67 | 0.617 | 32.41 |
| XGBoost V2 (1st innings) | 44.05 | 0.610 | 32.84 |
| **DLS Baseline** | 65.03 | 0.150 | — |

- DM test vs DLS: p < 0.001 (T=542 matches)
- Bootstrap 95% CI for improvement over DLS: [−23.94, −18.72] runs
- Model Confidence Set (10%): {CatBoost V2, LightGBM V2}
- Conformal coverage (α=0.10): 86.6% empirical / 90% nominal

## Repository Structure

```
├── src/                   # Core Python modules
│   ├── pipeline.py        # Enhanced feature pipeline (V2, 46 features)
│   ├── ml_models.py       # XGBoost, LightGBM, CatBoost training
│   ├── second_innings.py  # Second innings pipeline + RevisedTargetEngine
│   ├── statistical_tests.py  # DM test, bootstrap, MCS, conformal, ablation
│   ├── feature_engineering.py
│   ├── player_features.py
│   ├── venue_features.py
│   ├── elo_tracker.py
│   ├── dls_method.py
│   └── ...
├── scripts/
│   ├── train_v2_models.py       # Train all V2 models
│   ├── run_full_evaluation.py   # Run all statistical tests + generate figures
│   └── regenerate_figures.py
├── notebooks/
│   ├── 01_data_collection_and_eda.ipynb
│   ├── 02_dls_analysis.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_extension_analysis.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_visualizations_and_results.ipynb
├── data/processed/        # Processed parquet datasets
├── results/
│   ├── figures/           # All plots (SHAP, actual vs predicted, etc.)
│   ├── metrics/           # CSVs/JSONs (RMSE, DM tests, bootstrap CI, etc.)
│   └── tables/            # LaTeX and CSV result tables
├── thesis.tex             # LaTeX source
├── thesis.pdf             # Compiled thesis
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Results

Train V2 models:
```bash
python scripts/train_v2_models.py
```

Run full evaluation (statistical tests + figures):
```bash
python scripts/run_full_evaluation.py
```

## Feature Sets

- **V1**: 22 features (original baseline)
- **V2**: 46 features — adds ELO ratings, rolling player stats, venue history, DLS resource features
- **Inn2**: 53 features — V2 + target score, required runs, required run rate, pressure index

## Data

Raw data sourced from [Cricsheet](https://cricsheet.org/) (not included due to size). Processed parquet files are included in `data/processed/`.
