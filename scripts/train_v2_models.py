"""
Phase 3: Train enhanced (V2) models on both innings.

First innings  — 46 features, compare v1 baseline vs v2 enhanced
Second innings — 53 features, train on UNCENSORED innings only (teams that lost)

Saves models and a summary CSV to results/models/ and results/metrics/.
"""
import sys, pickle, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FEATURE_COLUMNS_V2, TARGET_COLUMN
from src.second_innings import FEATURE_COLUMNS_INN2, TARGET_COLUMN_INN2
from src.ml_models import (train_xgboost, train_lightgbm, train_catboost,
                            train_lgbm_quantile, save_model)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("train_v2")

PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "results" / "models"
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

N_XGB = 100
N_LGB = 50
N_CAT = 50

def rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))

def evaluate_quick(model, X_test, y_test):
    p = model.predict(X_test)
    return {
        "rmse": round(rmse(y_test, p), 4),
        "r2":   round(r2_score(y_test, p), 4),
        "mae":  round(mean_absolute_error(y_test, p), 4),
    }

# ──────────────────────────────────────────────────────────────
#  FIRST INNINGS  V2  (46 features)
# ──────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("FIRST INNINGS V2 TRAINING")
log.info("=" * 60)

train = pd.read_parquet(PROC / "mens_odi_train_v2.parquet")
cal   = pd.read_parquet(PROC / "mens_odi_cal_v2.parquet")
test  = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")

feats = [c for c in FEATURE_COLUMNS_V2 if c in train.columns]
log.info(f"Features: {len(feats)}  | train {len(train)} | cal {len(cal)} | test {len(test)}")

Xtr = train[feats]; ytr = train[TARGET_COLUMN]
Xcal = cal[feats];   ycal = cal[TARGET_COLUMN]
Xte = test[feats];   yte = test[TARGET_COLUMN]

inn1_results = {}

log.info("--- XGBoost V2 ---")
xgb_v2, _ = train_xgboost(Xtr, ytr, Xcal, ycal, n_trials=N_XGB)
save_model(xgb_v2, "xgboost_v2", format_key="mens_odi")
inn1_results["XGBoost_V2"] = evaluate_quick(xgb_v2, Xte.values, yte.values)
log.info(f"XGBoost V2:   {inn1_results['XGBoost_V2']}")

log.info("--- LightGBM V2 ---")
lgb_v2, _ = train_lightgbm(Xtr, ytr, Xcal, ycal, n_trials=N_LGB)
save_model(lgb_v2, "lightgbm_v2", format_key="mens_odi")
inn1_results["LightGBM_V2"] = evaluate_quick(lgb_v2, Xte.values, yte.values)
log.info(f"LightGBM V2:  {inn1_results['LightGBM_V2']}")

log.info("--- CatBoost V2 ---")
cat_v2, _ = train_catboost(Xtr, ytr, Xcal, ycal, n_trials=N_CAT)
save_model(cat_v2, "catboost_v2", format_key="mens_odi")
inn1_results["CatBoost_V2"] = evaluate_quick(cat_v2, Xte.values, yte.values)
log.info(f"CatBoost V2:  {inn1_results['CatBoost_V2']}")

log.info("--- LightGBM Quantile V2 ---")
qmodels = train_lgbm_quantile(
    Xtr, ytr, Xcal, ycal,
    quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
)
with open(MODELS / "mens_odi_lightgbm_quantile_v2.pkl", "wb") as f:
    pickle.dump(qmodels, f)
log.info("Quantile models saved.")

if "dls_predicted_final" in test.columns:
    dls_preds = test["dls_predicted_final"].values
    inn1_results["DLS_V1baseline"] = {
        "rmse": round(rmse(yte.values, dls_preds), 4),
        "r2":   round(r2_score(yte.values, dls_preds), 4),
        "mae":  round(mean_absolute_error(yte.values, dls_preds), 4),
    }
    log.info(f"DLS baseline: {inn1_results['DLS_V1baseline']}")

inn1_df = pd.DataFrame(inn1_results).T.reset_index().rename(columns={"index": "model"})
inn1_df.to_csv(METRICS / "inn1_v2_results.csv", index=False)
log.info(f"\nFirst innings V2 summary:\n{inn1_df.to_string(index=False)}")

# ──────────────────────────────────────────────────────────────
#  SECOND INNINGS  (53 features, uncensored only)
# ──────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("SECOND INNINGS TRAINING  (uncensored only)")
log.info("=" * 60)

tr2  = pd.read_parquet(PROC / "mens_odi_train_inn2.parquet")
cal2 = pd.read_parquet(PROC / "mens_odi_cal_inn2.parquet")
te2  = pd.read_parquet(PROC / "mens_odi_test_inn2.parquet")

tr2_unc  = tr2[tr2["is_censored"] == 0]
cal2_unc = cal2[cal2["is_censored"] == 0]
te2_unc  = te2[te2["is_censored"] == 0]

feats2 = [c for c in FEATURE_COLUMNS_INN2 if c in tr2.columns]
log.info(
    f"Inn2 features: {len(feats2)}  "
    f"| train-uncens: {len(tr2_unc)} | cal-uncens: {len(cal2_unc)} "
    f"| test-uncens: {len(te2_unc)}"
)

X2tr = tr2_unc[feats2];  y2tr = tr2_unc[TARGET_COLUMN_INN2]
X2cal = cal2_unc[feats2]; y2cal = cal2_unc[TARGET_COLUMN_INN2]
X2te = te2_unc[feats2];   y2te = te2_unc[TARGET_COLUMN_INN2]

inn2_results = {}

log.info("--- XGBoost Inn2 ---")
xgb_inn2, _ = train_xgboost(X2tr, y2tr, X2cal, y2cal, n_trials=N_XGB)
save_model(xgb_inn2, "xgboost_inn2", format_key="mens_odi")
inn2_results["XGBoost_Inn2"] = evaluate_quick(xgb_inn2, X2te.values, y2te.values)
log.info(f"XGBoost Inn2:  {inn2_results['XGBoost_Inn2']}")

log.info("--- LightGBM Inn2 ---")
lgb_inn2, _ = train_lightgbm(X2tr, y2tr, X2cal, y2cal, n_trials=N_LGB)
save_model(lgb_inn2, "lightgbm_inn2", format_key="mens_odi")
inn2_results["LightGBM_Inn2"] = evaluate_quick(lgb_inn2, X2te.values, y2te.values)
log.info(f"LightGBM Inn2: {inn2_results['LightGBM_Inn2']}")

log.info("--- CatBoost Inn2 ---")
cat_inn2, _ = train_catboost(X2tr, y2tr, X2cal, y2cal, n_trials=N_CAT)
save_model(cat_inn2, "catboost_inn2", format_key="mens_odi")
inn2_results["CatBoost_Inn2"] = evaluate_quick(cat_inn2, X2te.values, y2te.values)
log.info(f"CatBoost Inn2: {inn2_results['CatBoost_Inn2']}")

# DLS baseline for second innings: project score using DLS resource fraction
if "resource_pct_remaining_inn2" in te2_unc.columns:
    res_rem  = te2_unc["resource_pct_remaining_inn2"].values / 100.0
    res_full = 1.0   # team 2 started with 100% resources
    res_used = res_full - res_rem
    valid    = res_used > 0.02
    dls_inn2_preds = np.where(
        valid,
        te2_unc["current_score"].values / np.where(valid, res_used, 0.5),
        te2_unc["current_score"].values + 80,
    )
    dls_inn2_preds = np.minimum(dls_inn2_preds,
                                te2_unc["target_score"].values + 100)
    inn2_results["DLS_Inn2"] = {
        "rmse": round(rmse(y2te.values, dls_inn2_preds), 4),
        "r2":   round(r2_score(y2te.values, dls_inn2_preds), 4),
        "mae":  round(mean_absolute_error(y2te.values, dls_inn2_preds), 4),
    }
    log.info(f"DLS Inn2 projection: {inn2_results['DLS_Inn2']}")

inn2_df = pd.DataFrame(inn2_results).T.reset_index().rename(columns={"index": "model"})
inn2_df.to_csv(METRICS / "inn2_results.csv", index=False)
log.info(f"\nSecond innings summary:\n{inn2_df.to_string(index=False)}")

log.info("=" * 60)
log.info("ALL TRAINING COMPLETE")
log.info("=" * 60)
