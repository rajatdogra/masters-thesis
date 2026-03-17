"""
Train V3 Models — gradient boosting on extended 55-feature set.

V3 adds 9 new features to V2:
  H2H, Toss Advantage, Match Importance, Month

This script:
1. Loads V3 parquets (built by build_v3_snapshots.py)
2. Trains XGBoost, LightGBM, CatBoost on V3 features
3. Compares V3 vs V2 performance
4. Saves models and metrics

Runtime: ~45-90 minutes (3 × 50-100 Optuna trials)
"""

import sys, pickle, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml_models import train_xgboost, train_lightgbm, train_catboost, save_model
from src.pipeline import TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("train_v3")

PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "results" / "models"
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

N_XGB = 100
N_LGB = 50
N_CAT = 50


def rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def evaluate(model, X_test, y_test):
    p = model.predict(X_test)
    return {
        "rmse": round(rmse(y_test, p), 4),
        "r2":   round(r2_score(y_test, p), 4),
        "mae":  round(mean_absolute_error(y_test, p), 4),
    }


# ──────────────────────────────────────────────────────────────
# Load V3 parquets
# ──────────────────────────────────────────────────────────────
log.info("Loading V3 parquets...")
train = pd.read_parquet(PROC / "mens_odi_train_v3.parquet")
cal   = pd.read_parquet(PROC / "mens_odi_cal_v3.parquet")
test  = pd.read_parquet(PROC / "mens_odi_test_v3.parquet")

# Load V3 feature list
with open(METRICS / "feature_columns_v3.json") as f:
    v3_meta = json.load(f)

feats = [c for c in v3_meta["v3_all_features"] if c in train.columns]
log.info(f"V3 Features: {len(feats)} | train {len(train)} | cal {len(cal)} | test {len(test)}")

Xtr  = train[feats]; ytr  = train[TARGET_COLUMN]
Xcal = cal[feats];   ycal = cal[TARGET_COLUMN]
Xte  = test[feats];  yte  = test[TARGET_COLUMN]

results = {}

# ──────────────────────────────────────────────────────────────
# Train Models
# ──────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("XGBoost V3")
log.info("=" * 60)
xgb_v3, _ = train_xgboost(Xtr, ytr, Xcal, ycal, n_trials=N_XGB)
save_model(xgb_v3, "xgboost_v3", format_key="mens_odi")
results["XGBoost_V3"] = evaluate(xgb_v3, Xte.values, yte.values)
log.info(f"XGBoost V3: {results['XGBoost_V3']}")

log.info("=" * 60)
log.info("LightGBM V3")
log.info("=" * 60)
lgb_v3, _ = train_lightgbm(Xtr, ytr, Xcal, ycal, n_trials=N_LGB)
save_model(lgb_v3, "lightgbm_v3", format_key="mens_odi")
results["LightGBM_V3"] = evaluate(lgb_v3, Xte.values, yte.values)
log.info(f"LightGBM V3: {results['LightGBM_V3']}")

log.info("=" * 60)
log.info("CatBoost V3")
log.info("=" * 60)
cat_v3, _ = train_catboost(Xtr, ytr, Xcal, ycal, n_trials=N_CAT)
save_model(cat_v3, "catboost_v3", format_key="mens_odi")
results["CatBoost_V3"] = evaluate(cat_v3, Xte.values, yte.values)
log.info(f"CatBoost V3: {results['CatBoost_V3']}")

# DLS baseline
if "dls_predicted_final" in test.columns:
    dls_preds = test["dls_predicted_final"].values
    results["DLS"] = {
        "rmse": round(rmse(yte.values, dls_preds), 4),
        "r2":   round(r2_score(yte.values, dls_preds), 4),
        "mae":  round(mean_absolute_error(yte.values, dls_preds), 4),
    }

# ──────────────────────────────────────────────────────────────
# Load V2 results for comparison
# ──────────────────────────────────────────────────────────────
v2_results = {}
v2_path = METRICS / "inn1_v2_results.csv"
if v2_path.exists():
    v2_df = pd.read_csv(v2_path)
    for _, row in v2_df.iterrows():
        v2_results[row["model"]] = {"rmse": row["rmse"], "r2": row["r2"], "mae": row["mae"]}

# ──────────────────────────────────────────────────────────────
# Report and Save
# ──────────────────────────────────────────────────────────────
result_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
result_df.to_csv(METRICS / "inn1_v3_results.csv", index=False)

log.info("=" * 60)
log.info("V3 RESULTS SUMMARY")
log.info("=" * 60)
log.info(f"\n{result_df.to_string(index=False)}")

log.info("\n--- V2 vs V3 Comparison ---")
for m_v3, m_v2 in [("XGBoost_V3", "XGBoost_V2"), ("LightGBM_V3", "LightGBM_V2"), ("CatBoost_V3", "CatBoost_V2")]:
    if m_v3 in results and m_v2 in v2_results:
        d_rmse = results[m_v3]["rmse"] - v2_results[m_v2]["rmse"]
        d_r2   = results[m_v3]["r2"]   - v2_results[m_v2]["r2"]
        log.info(f"  {m_v3} vs {m_v2}: ΔRMSE={d_rmse:+.3f}, ΔR²={d_r2:+.4f}")

log.info("=" * 60)
log.info("TRAINING COMPLETE. Models saved to results/models/")
log.info("=" * 60)
