"""
Stacking Ensemble for Cricket Score Prediction.

Motivation
----------
Individual gradient boosting models (XGBoost, LightGBM, CatBoost) each capture
slightly different aspects of the data due to their different tree-growing
algorithms, regularisation, and gradient approximations. A stacking ensemble
combines their predictions to reduce variance and potentially improve accuracy.

Architecture
-----------
Level-0 models (base learners):
  - XGBoost V2  (pre-trained)
  - LightGBM V2 (pre-trained)
  - CatBoost V2 (pre-trained)

Level-1 model (meta-learner):
  - Ridge Regression trained on calibration set base-model predictions
  - Ridge is preferred over another GBM because:
    (1) it cannot overfit on 3 features
    (2) it is fully interpretable (shows how much each model is weighted)
    (3) it is equivalent to optimal linear combination

Correct stacking protocol (avoiding data leakage):
  - Base models trained on TRAIN set only (already done)
  - Meta-learner trained on CALIBRATION set predictions (out-of-train predictions)
  - Final evaluation on TEST set (never seen by either level)

Output
------
  results/models/mens_odi_stacking_meta.pkl  — the Ridge meta-learner
  results/metrics/stacking_results.csv       — comparison with base models
  results/metrics/stacking_weights.json      — meta-learner coefficients

Runtime: ~5 minutes (loading pre-trained models + Ridge regression)
"""

import sys, pickle, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FEATURE_COLUMNS_V2, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("stacking")

PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "results" / "models"
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)


def rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def load_v2_model(name: str):
    """Load a pre-trained V2 model from pickle."""
    path = MODELS / f"mens_odi_{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train_v2_models.py first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("scaler")


# ──────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────
log.info("Loading V2 parquets...")
cal  = pd.read_parquet(PROC / "mens_odi_cal_v2.parquet")
test = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")

feats = [c for c in FEATURE_COLUMNS_V2 if c in cal.columns]
log.info(f"Features: {len(feats)} | cal: {len(cal)} | test: {len(test)}")

X_cal  = cal[feats].values
y_cal  = cal[TARGET_COLUMN].values
X_test = test[feats].values
y_test = test[TARGET_COLUMN].values

# ──────────────────────────────────────────────────────────────
# 2. Load pre-trained base models
# ──────────────────────────────────────────────────────────────
log.info("Loading pre-trained V2 base models...")
xgb_model, _  = load_v2_model("xgboost_v2")
lgb_model, _  = load_v2_model("lightgbm_v2")
cat_model, _  = load_v2_model("catboost_v2")
log.info("  Loaded: XGBoost V2, LightGBM V2, CatBoost V2")

# ──────────────────────────────────────────────────────────────
# 3. Generate base-model predictions on calibration set
# ──────────────────────────────────────────────────────────────
log.info("Generating calibration set predictions from base models...")
xgb_cal_preds = xgb_model.predict(X_cal)
lgb_cal_preds = lgb_model.predict(X_cal)
cat_cal_preds = cat_model.predict(X_cal)

# Stack predictions into meta-feature matrix
meta_X_cal = np.column_stack([xgb_cal_preds, lgb_cal_preds, cat_cal_preds])
log.info(f"  Meta-feature matrix shape: {meta_X_cal.shape}")

# ──────────────────────────────────────────────────────────────
# 4. Train Ridge meta-learner on calibration predictions
# ──────────────────────────────────────────────────────────────
log.info("Training Ridge meta-learner...")

# Cross-validate to select optimal Ridge alpha
alphas = np.logspace(-3, 3, 20)
best_alpha = 1.0
best_cv_score = np.inf

for a in alphas:
    ridge = Ridge(alpha=a)
    # Negative MSE → higher = better
    cv_scores = cross_val_score(ridge, meta_X_cal, y_cal, cv=5, scoring="neg_mean_squared_error")
    cv_rmse = float(np.sqrt(-cv_scores.mean()))
    if cv_rmse < best_cv_score:
        best_cv_score = cv_rmse
        best_alpha = a

log.info(f"  Best Ridge alpha: {best_alpha:.4f} (CV RMSE: {best_cv_score:.4f})")

meta_learner = Ridge(alpha=best_alpha)
meta_learner.fit(meta_X_cal, y_cal)

# Coefficients (how each base model is weighted)
coef_df = pd.DataFrame({
    "model":  ["XGBoost", "LightGBM", "CatBoost"],
    "weight": meta_learner.coef_.tolist(),
    "intercept": [meta_learner.intercept_] + [None, None],
})
log.info(f"  Meta-learner weights:\n{coef_df.to_string(index=False)}")

# ──────────────────────────────────────────────────────────────
# 5. Evaluate on test set
# ──────────────────────────────────────────────────────────────
log.info("Evaluating on test set...")
xgb_test_preds = xgb_model.predict(X_test)
lgb_test_preds = lgb_model.predict(X_test)
cat_test_preds = cat_model.predict(X_test)

meta_X_test = np.column_stack([xgb_test_preds, lgb_test_preds, cat_test_preds])
stack_preds  = meta_learner.predict(meta_X_test)

# Also compute simple average ensemble for comparison
avg_preds = (xgb_test_preds + lgb_test_preds + cat_test_preds) / 3.0

results = {
    "XGBoost_V2": {
        "rmse": round(rmse(y_test, xgb_test_preds), 4),
        "r2":   round(r2_score(y_test, xgb_test_preds), 4),
        "mae":  round(mean_absolute_error(y_test, xgb_test_preds), 4),
    },
    "LightGBM_V2": {
        "rmse": round(rmse(y_test, lgb_test_preds), 4),
        "r2":   round(r2_score(y_test, lgb_test_preds), 4),
        "mae":  round(mean_absolute_error(y_test, lgb_test_preds), 4),
    },
    "CatBoost_V2": {
        "rmse": round(rmse(y_test, cat_test_preds), 4),
        "r2":   round(r2_score(y_test, cat_test_preds), 4),
        "mae":  round(mean_absolute_error(y_test, cat_test_preds), 4),
    },
    "SimpleAverage": {
        "rmse": round(rmse(y_test, avg_preds), 4),
        "r2":   round(r2_score(y_test, avg_preds), 4),
        "mae":  round(mean_absolute_error(y_test, avg_preds), 4),
    },
    "StackingEnsemble": {
        "rmse": round(rmse(y_test, stack_preds), 4),
        "r2":   round(r2_score(y_test, stack_preds), 4),
        "mae":  round(mean_absolute_error(y_test, stack_preds), 4),
    },
}

if "dls_predicted_final" in test.columns:
    dls_preds = test["dls_predicted_final"].values
    results["DLS"] = {
        "rmse": round(rmse(y_test, dls_preds), 4),
        "r2":   round(r2_score(y_test, dls_preds), 4),
        "mae":  round(mean_absolute_error(y_test, dls_preds), 4),
    }

# ──────────────────────────────────────────────────────────────
# 6. Save results
# ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
results_df = results_df.sort_values("rmse")
results_df.to_csv(METRICS / "stacking_results.csv", index=False)

# Save meta-learner weights
weights = {
    "model_names":  ["XGBoost", "LightGBM", "CatBoost"],
    "weights":      meta_learner.coef_.tolist(),
    "intercept":    float(meta_learner.intercept_),
    "ridge_alpha":  float(best_alpha),
    "cv_rmse":      round(best_cv_score, 4),
}
with open(METRICS / "stacking_weights.json", "w") as f:
    json.dump(weights, f, indent=2)

# Save meta-learner pickle
with open(MODELS / "mens_odi_stacking_meta.pkl", "wb") as f:
    pickle.dump({
        "meta_learner": meta_learner,
        "base_model_names": ["xgboost_v2", "lightgbm_v2", "catboost_v2"],
        "feature_columns": feats,
    }, f)

log.info("=" * 60)
log.info("STACKING ENSEMBLE RESULTS")
log.info("=" * 60)
log.info(f"\n{results_df.to_string(index=False)}")

best_base   = min(["XGBoost_V2", "LightGBM_V2", "CatBoost_V2"], key=lambda m: results[m]["rmse"])
base_rmse   = results[best_base]["rmse"]
stack_rmse  = results["StackingEnsemble"]["rmse"]
improvement = base_rmse - stack_rmse

log.info(f"\nBest individual model: {best_base} (RMSE={base_rmse:.4f})")
log.info(f"Stacking ensemble RMSE:  {stack_rmse:.4f}")
log.info(f"Improvement from stacking: {improvement:.4f} runs RMSE")
log.info(f"Meta-learner weights: XGB={weights['weights'][0]:.4f}, LGB={weights['weights'][1]:.4f}, CAT={weights['weights'][2]:.4f}")
log.info("=" * 60)
