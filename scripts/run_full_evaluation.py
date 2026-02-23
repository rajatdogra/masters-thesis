"""
Full evaluation pipeline — run after train_v2_models.py completes.

Generates:
 - SHAP summary figure (V2 XGBoost, V2 LightGBM)
 - Hyperparameter tables (XGBoost, LightGBM, CatBoost V2)
 - DM test results (ML vs DLS, V1 vs V2)
 - Block bootstrap CIs
 - Phase-wise metrics
 - Conformal prediction coverage
 - Ablation study
 - Per-team fairness metrics

All outputs → results/metrics/ and results/figures/
"""
import sys, pickle, json, logging
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FEATURE_COLUMNS_V2, TARGET_COLUMN
from src.second_innings import FEATURE_COLUMNS_INN2, TARGET_COLUMN_INN2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("eval")

PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "results" / "models"
METRICS = ROOT / "results" / "metrics"
FIGS    = ROOT / "results" / "figures"
METRICS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  Load models and data
# ─────────────────────────────────────────────────────────────
def load_model(path):
    """Load model from pkl — handles both raw model and {'model':..,'scaler':..} dict."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("scaler")
    return obj, None

log.info("Loading models...")
xgb_v2,  xgb_v2_scaler  = load_model(MODELS / "mens_odi_xgboost_v2.pkl")
lgb_v2,  lgb_v2_scaler  = load_model(MODELS / "mens_odi_lightgbm_v2.pkl")
cat_v2,  cat_v2_scaler  = load_model(MODELS / "mens_odi_catboost_v2.pkl")
xgb_v1,  xgb_v1_scaler  = load_model(MODELS / "mens_odi_xgboost.pkl")
lgb_v1,  lgb_v1_scaler  = load_model(MODELS / "mens_odi_lightgbm.pkl")
with open(MODELS / "mens_odi_lightgbm_quantile_v2.pkl", "rb") as f:
    qmodels_raw = pickle.load(f)
# qmodels may be a dict of {quantile: model_dict}
qmodels = {}
for k, v in qmodels_raw.items():
    if isinstance(v, dict) and "model" in v:
        qmodels[k] = v["model"]
    else:
        qmodels[k] = v

log.info("Loading data...")
test  = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")
cal   = pd.read_parquet(PROC / "mens_odi_cal_v2.parquet")
train = pd.read_parquet(PROC / "mens_odi_train_v2.parquet")

feats = [c for c in FEATURE_COLUMNS_V2 if c in test.columns]
log.info(f"V2 features: {len(feats)}")

Xte_raw  = test[feats].values
yte      = test[TARGET_COLUMN].values
Xcal_raw = cal[feats].values
ycal     = cal[TARGET_COLUMN].values
Xtr_raw  = train[feats].values
ytr      = train[TARGET_COLUMN].values

# Apply scaler if present (LightGBM V2 uses scaler)
def apply_scaler(X, scaler):
    return scaler.transform(X) if scaler is not None else X

Xte   = apply_scaler(Xte_raw,  lgb_v2_scaler)
Xcal  = apply_scaler(Xcal_raw, lgb_v2_scaler)
Xtr   = apply_scaler(Xtr_raw,  lgb_v2_scaler)
# Per-model scaled arrays
Xte_xgb  = apply_scaler(Xte_raw, xgb_v2_scaler)
Xte_cat  = apply_scaler(Xte_raw, cat_v2_scaler)
Xte_v1   = Xte_raw  # V1 features applied below
Xcal_lgb = Xcal

# ─────────────────────────────────────────────────────────────
#  1. Extract and save hyperparameters
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("1. HYPERPARAMETER EXTRACTION")

def extract_params(model, model_name):
    """Extract best hyperparameters from Optuna-tuned model."""
    params = {}
    try:
        mn = model_name.lower()
        # XGBoost: XGBRegressor
        if "xgb" in mn or hasattr(model, 'get_xgb_params'):
            p = model.get_xgb_params() if hasattr(model, 'get_xgb_params') else {}
            keys = ['n_estimators','max_depth','learning_rate','subsample',
                    'colsample_bytree','min_child_weight','gamma','reg_alpha','reg_lambda']
            for k in keys:
                v = p.get(k) if k in p else getattr(model, k, None)
                if v is not None:
                    params[k] = v
            if hasattr(model, 'n_estimators'):
                params['n_estimators'] = int(model.n_estimators)
        # LightGBM: LGBMRegressor
        elif "lgb" in mn or "lightgbm" in mn:
            p = model.get_params() if hasattr(model, 'get_params') else {}
            keys = ['n_estimators','num_leaves','max_depth','learning_rate',
                    'min_child_samples','subsample','colsample_bytree',
                    'reg_alpha','reg_lambda','min_split_gain']
            params = {k: p[k] for k in keys if k in p and p[k] is not None}
        # CatBoost
        elif "cat" in mn or hasattr(model, 'get_all_params'):
            p = model.get_all_params() if hasattr(model, 'get_all_params') else {}
            keys = ['iterations','depth','learning_rate','l2_leaf_reg',
                    'bagging_temperature','random_strength','border_count']
            params = {k: p[k] for k in keys if k in p and p[k] is not None}
    except Exception as e:
        log.warning(f"Param extraction for {model_name}: {e}")
    return params

hp = {
    "XGBoost_V2":  extract_params(xgb_v2,  "xgb"),
    "LightGBM_V2": extract_params(lgb_v2,  "lightgbm"),
    "CatBoost_V2": extract_params(cat_v2,  "catboost"),
}
for name, p in hp.items():
    log.info(f"  {name}: {p}")

with open(METRICS / "hyperparameters_v2.json", "w") as f:
    json.dump(hp, f, indent=2, default=str)
log.info("Hyperparameters saved.")

# ─────────────────────────────────────────────────────────────
#  2. SHAP summary figures
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("2. SHAP ANALYSIS")

import shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use a background sample for TreeExplainer speed
Xtr_lgb = apply_scaler(Xtr_raw, lgb_v2_scaler)
Xtr_xgb = apply_scaler(Xtr_raw, xgb_v2_scaler)

bg_idx_lgb = np.random.default_rng(42).choice(len(Xtr_lgb), size=min(500, len(Xtr_lgb)), replace=False)
bg_idx_xgb = np.random.default_rng(42).choice(len(Xtr_xgb), size=min(500, len(Xtr_xgb)), replace=False)

for model, X_bg, Xtest, name, fname in [
    (xgb_v2, Xtr_xgb[bg_idx_xgb], Xte_xgb, "XGBoost V2 (46 features)", "mens_odi_shap_summary_xgboost"),
    (lgb_v2, Xtr_lgb[bg_idx_lgb], Xte,      "LightGBM V2 (46 features)", "mens_odi_shap_summary_lightgbm"),
]:
    log.info(f"  SHAP for {name} ...")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(Xtest[:500])  # first 500 test snapshots

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals, Xtest[:500],
        feature_names=feats,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — {name}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(FIGS / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {fname}.png")

# SHAP bar chart for V2 (mean |SHAP|)
log.info("  SHAP bar chart ...")
explainer_bar = shap.TreeExplainer(lgb_v2)
sv_bar        = explainer_bar.shap_values(Xte)
mean_abs_shap = pd.Series(np.abs(sv_bar).mean(axis=0), index=feats).sort_values(ascending=False)
mean_abs_shap.to_csv(METRICS / "shap_importance_lgb_v2.csv")

fig, ax = plt.subplots(figsize=(9, 7))
mean_abs_shap.head(20).sort_values().plot(kind="barh", ax=ax, color="#2196F3")
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Top 20 Features — LightGBM V2\n(mean absolute SHAP value)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_shap_bar_lgb_v2.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  SHAP bar chart saved.")

# ─────────────────────────────────────────────────────────────
#  3. Diebold-Mariano tests & block bootstrap
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("3. STATISTICAL TESTS")

from src.statistical_tests import diebold_mariano_test, block_bootstrap_ci

match_ids = test["match_id"].values if "match_id" in test.columns else np.arange(len(yte))

pred_lgb_v2 = lgb_v2.predict(Xte)
pred_xgb_v2 = xgb_v2.predict(Xte_xgb)
pred_cat_v2 = cat_v2.predict(Xte_cat)

# V1 features (22)
try:
    from src.feature_engineering import FEATURE_COLUMNS
    feats_v1 = [c for c in FEATURE_COLUMNS if c in test.columns]
except ImportError:
    feats_v1 = []
if feats_v1:
    Xte_v1_arr  = apply_scaler(test[feats_v1].values, lgb_v1_scaler)
    pred_lgb_v1 = lgb_v1.predict(Xte_v1_arr)
else:
    pred_lgb_v1 = None
    log.warning("V1 features not found in V2 test set; skipping V1 DM test.")

dls_col = "dls_predicted_final"
if dls_col in test.columns:
    pred_dls = test[dls_col].values
else:
    pred_dls = None
    log.warning(f"Column '{dls_col}' not in test set; DLS DM tests skipped.")

dm_results = {}

if pred_dls is not None:
    for name, pred in [("XGBoost_V2", pred_xgb_v2),
                       ("LightGBM_V2", pred_lgb_v2),
                       ("CatBoost_V2", pred_cat_v2)]:
        r = diebold_mariano_test(yte, pred, pred_dls, match_ids)
        dm_results[f"{name}_vs_DLS"] = r
        log.info(f"  DM {name} vs DLS: stat={r['dm_stat']:.3f}, p={r['p_value']:.4f}, "
                 f"sig={r['significant']}, dir={r['direction']}")

if pred_lgb_v1 is not None and pred_dls is not None:
    r = diebold_mariano_test(yte, pred_lgb_v1, pred_dls, match_ids)
    dm_results["LightGBM_V1_vs_DLS"] = r
    log.info(f"  DM LightGBM_V1 vs DLS: stat={r['dm_stat']:.3f}, p={r['p_value']:.4f}")
    r2 = diebold_mariano_test(yte, pred_lgb_v2, pred_lgb_v1, match_ids)
    dm_results["LightGBM_V2_vs_V1"] = r2
    log.info(f"  DM LightGBM_V2 vs V1: stat={r2['dm_stat']:.3f}, p={r2['p_value']:.4f}")

pd.DataFrame(dm_results).T.reset_index().rename(columns={"index":"comparison"}).to_csv(
    METRICS / "dm_test_results.csv", index=False)
log.info("  DM results saved.")

# Block bootstrap
if pred_dls is not None:
    log.info("  Block bootstrap CI (LightGBM_V2 vs DLS) ...")
    ci = block_bootstrap_ci(yte, pred_lgb_v2, pred_dls, match_ids, n_boot=5000)
    log.info(f"  Bootstrap CI: {ci}")
    with open(METRICS / "bootstrap_ci.json", "w") as f:
        json.dump(ci, f, indent=2, default=str)
    log.info("  Bootstrap CI saved.")

# ─────────────────────────────────────────────────────────────
#  4. Model Confidence Set
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("4. MODEL CONFIDENCE SET")

from src.statistical_tests import compute_mcs

preds_dict = {
    "XGBoost_V2":  pred_xgb_v2,
    "LightGBM_V2": pred_lgb_v2,
    "CatBoost_V2": pred_cat_v2,
}
if pred_lgb_v1 is not None:
    preds_dict["LightGBM_V1"] = pred_lgb_v1
if pred_dls is not None:
    preds_dict["DLS"] = pred_dls

try:
    mcs_result = compute_mcs(preds_dict, yte, match_ids, alpha=0.10)
    log.info(f"  MCS result: {mcs_result.get('mcs_set')}")
    with open(METRICS / "mcs_result.json", "w") as f:
        json.dump(mcs_result, f, indent=2, default=str)
    log.info("  MCS result saved.")
except Exception as e:
    log.warning(f"  MCS failed: {e}")

# ─────────────────────────────────────────────────────────────
#  5. Conformal prediction
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("5. CONFORMAL PREDICTION")

from src.statistical_tests import conformal_coverage

try:
    cov_dict = conformal_coverage(lgb_v2, Xcal_lgb, ycal, Xte, yte, feats,
                                  alphas=[0.05, 0.10, 0.20])
    if cov_dict:
        cov_df = pd.DataFrame(cov_dict).T.reset_index().rename(columns={"index": "alpha"})
        log.info(f"  Conformal coverage:\n{cov_df}")
        cov_df.to_csv(METRICS / "conformal_coverage.csv", index=False)
        log.info("  Conformal coverage saved.")
    else:
        log.warning("  Conformal coverage returned empty results.")
except Exception as e:
    log.warning(f"  Conformal prediction failed: {e}")

# ─────────────────────────────────────────────────────────────
#  6. Phase-wise metrics
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("6. PHASE-WISE METRICS")

from src.statistical_tests import phase_wise_metrics

overs_col = "overs_completed"
if overs_col in test.columns:
    preds_pw = {
        "XGBoost_V2":  pred_xgb_v2,
        "LightGBM_V2": pred_lgb_v2,
        "CatBoost_V2": pred_cat_v2,
    }
    if pred_dls is not None:
        preds_pw["DLS"] = pred_dls
    if pred_lgb_v1 is not None:
        preds_pw["LightGBM_V1"] = pred_lgb_v1

    pw_df = phase_wise_metrics(yte, preds_pw, test[overs_col].values)
    log.info(f"  Phase-wise metrics:\n{pw_df.to_string()}")
    pw_df.to_csv(METRICS / "phase_wise_metrics.csv", index=False)
    log.info("  Phase-wise metrics saved.")

# ─────────────────────────────────────────────────────────────
#  7. Ablation study
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("7. ABLATION STUDY (n_trials=30)")

from src.statistical_tests import run_ablation_study

try:
    abl_df = run_ablation_study(
        train[feats], train[TARGET_COLUMN],
        test[feats],  test[TARGET_COLUMN],
        feats, n_trials=20,
    )
    log.info(f"  Ablation:\n{abl_df.to_string(index=False)}")
except Exception as e:
    log.warning(f"  Ablation failed: {e}")

# ─────────────────────────────────────────────────────────────
#  8. Per-team fairness
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("8. PER-TEAM FAIRNESS")

from src.statistical_tests import per_team_metrics

if "batting_team" in test.columns:
    team_preds = {
        "LightGBM_V2": pred_lgb_v2,
        "CatBoost_V2": pred_cat_v2,
    }
    if pred_dls is not None:
        team_preds["DLS"] = pred_dls

    team_df = per_team_metrics(yte, team_preds, test["batting_team"].values)
    log.info(f"  Team metrics (top 5):\n{team_df.head(5).to_string(index=False)}")
else:
    log.warning("  'batting_team' column not found, skipping team metrics.")

# ─────────────────────────────────────────────────────────────
#  9. Quantile calibration
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("9. QUANTILE CALIBRATION")

from src.statistical_tests import quantile_calibration

try:
    qcal_df = quantile_calibration(qmodels, Xte, yte, feats)
    log.info(f"  Quantile ECE:\n{qcal_df}")
    qcal_df.to_csv(METRICS / "quantile_calibration.csv", index=False)
except Exception as e:
    log.warning(f"  Quantile calibration failed: {e}")

# ─────────────────────────────────────────────────────────────
#  10. Actual vs Predicted scatter (V2)
# ─────────────────────────────────────────────────────────────
log.info("="*50)
log.info("10. ACTUAL VS PREDICTED FIGURE")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, pred) in zip(axes, [("XGBoost V2", pred_xgb_v2),
                                    ("LightGBM V2", pred_lgb_v2),
                                    ("CatBoost V2", pred_cat_v2)]):
    ax.scatter(yte, pred, alpha=0.15, s=6, color="#1976D2")
    mn, mx = yte.min(), yte.max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.2, label="Perfect")
    ax.set_xlabel("Actual final score", fontsize=10)
    ax.set_ylabel("Predicted final score", fontsize=10)
    ax.set_title(name, fontsize=11)
    from sklearn.metrics import r2_score, mean_squared_error
    r2   = r2_score(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    ax.text(0.05, 0.92, f"R²={r2:.3f}  RMSE={rmse:.1f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(fc="white", ec="gray", alpha=0.8))
    ax.legend(fontsize=8, loc="lower right")

plt.suptitle("Actual vs Predicted — V2 Models (Men's ODI Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_actual_vs_predicted_v2.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Actual vs predicted figure saved.")

# ─────────────────────────────────────────────────────────────
#  11. Error distribution comparison V1 vs V2 vs DLS
# ─────────────────────────────────────────────────────────────
log.info("11. ERROR DISTRIBUTION FIGURE")

fig, ax = plt.subplots(figsize=(10, 5))
errors = {}
errors["CatBoost V2"]  = pred_cat_v2 - yte
errors["LightGBM V2"]  = pred_lgb_v2 - yte
if pred_lgb_v1 is not None:
    errors["LightGBM V1"]  = pred_lgb_v1 - yte
if pred_dls is not None:
    errors["DLS"] = pred_dls - yte

colors = ["#1976D2", "#43A047", "#E53935", "#FB8C00"]
for (name, err), color in zip(errors.items(), colors):
    ax.hist(err, bins=60, alpha=0.5, label=f"{name} (bias={err.mean():.1f})", color=color)

ax.axvline(0, color="black", lw=1.2, ls="--")
ax.set_xlabel("Prediction Error (runs)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Error Distribution — V2 Models vs DLS (Men's ODI)", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_error_dist_v2.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Error distribution saved.")

log.info("="*60)
log.info("ALL EVALUATION COMPLETE")
log.info("="*60)
