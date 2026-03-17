"""
Additional Baseline Comparisons.

Purpose
-------
A rigorous comparison requires evaluating the ML models against not just DLS
but also simpler ML baselines. This establishes a hierarchy:

  Naive Projection << Linear Regression < DLS < GBM Models

The DLS method should NOT be trivially beatable by any linear model —
if it were, the DLS comparison would be unfair (we'd just need a regression).
The fact that carefully engineered tree models with 46 features outperform DLS
is the genuine scientific contribution.

Baseline Models
---------------
1. Naive Projection (no training):
   score = current_score + current_run_rate × overs_remaining
   The simplest possible "extrapolate current rate" approach.

2. Linear Regression (V2 features):
   sklearn LinearRegression — no regularisation, shows raw linear performance.

3. Ridge Regression (V2 features, alpha tuned via CV):
   Best linear model with L2 regularisation.

4. Polynomial Regression (degree 2, limited features):
   Non-linear but parametric. Uses overs_completed, current_score, wickets_fallen,
   current_run_rate — a manageable polynomial without feature explosion.

5. DLS (already computed — listed for reference).

Output
------
  results/metrics/baseline_comparison.csv  — all baselines + ML models
  results/figures/baseline_comparison.png  — bar chart comparison

Runtime: ~10–15 minutes
"""

import sys, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FEATURE_COLUMNS_V2, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("baselines")

PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "results" / "models"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)


def rmse_score(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def metrics(yt, yp):
    return {
        "rmse": round(rmse_score(yt, yp), 4),
        "r2":   round(r2_score(yt, yp), 4),
        "mae":  round(mean_absolute_error(yt, yp), 4),
    }


def load_model(name: str):
    path = MODELS / f"mens_odi_{name}.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["model"]


# ──────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────
log.info("Loading data...")
train = pd.read_parquet(PROC / "mens_odi_train_v2.parquet")
# Combine train + cal for baseline training (baselines don't need cal for conformal)
cal   = pd.read_parquet(PROC / "mens_odi_cal_v2.parquet")
train_cal = pd.concat([train, cal], ignore_index=True)
test  = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")

feats = [c for c in FEATURE_COLUMNS_V2 if c in train.columns]

X_tr  = train_cal[feats].values
y_tr  = train_cal[TARGET_COLUMN].values
X_te  = test[feats].values
y_te  = test[TARGET_COLUMN].values

log.info(f"  Train+cal: {len(train_cal)} | Test: {len(test)} | Features: {len(feats)}")

results = {}

# ──────────────────────────────────────────────────────────────
# 2. Naive Projection (no training needed)
# ──────────────────────────────────────────────────────────────
log.info("Computing Naive Projection baseline...")
# score_pred = current_score + current_run_rate × overs_remaining
# Both columns are in FEATURE_COLUMNS_V2 (indices vary, look up by name)
feat_idx = {f: i for i, f in enumerate(feats)}

if "current_score" in feat_idx and "current_run_rate" in feat_idx and "overs_remaining" in feat_idx:
    cs  = X_te[:, feat_idx["current_score"]]
    crr = X_te[:, feat_idx["current_run_rate"]]
    # overs_remaining is in V2 (derived: 50 - overs_completed)
    # If not, compute from overs_completed
    if "overs_remaining" in feat_idx:
        orr = X_te[:, feat_idx["overs_remaining"]]
    else:
        oc  = X_te[:, feat_idx["overs_completed"]]
        orr = 50.0 - oc

    naive_preds = cs + crr * orr
    naive_preds = np.clip(naive_preds, 50, 500)
    results["NaiveProjection"] = metrics(y_te, naive_preds)
    log.info(f"  Naive Projection: {results['NaiveProjection']}")
else:
    log.warning("Required columns for naive projection not found; skipping.")

# ──────────────────────────────────────────────────────────────
# 3. Linear Regression (OLS)
# ──────────────────────────────────────────────────────────────
log.info("Training Linear Regression (OLS)...")
ols = LinearRegression()
ols.fit(X_tr, y_tr)
ols_preds = ols.predict(X_te)
results["LinearRegression"] = metrics(y_te, ols_preds)
log.info(f"  Linear Regression: {results['LinearRegression']}")

# ──────────────────────────────────────────────────────────────
# 4. Ridge Regression (alpha tuned via CV)
# ──────────────────────────────────────────────────────────────
log.info("Training Ridge Regression (RidgeCV)...")
alphas = np.logspace(-2, 4, 50)
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  RidgeCV(alphas=alphas, cv=5)),
])
ridge.fit(X_tr, y_tr)
ridge_preds = ridge.predict(X_te)
results["RidgeRegression"] = metrics(y_te, ridge_preds)
best_alpha_ridge = ridge.named_steps["ridge"].alpha_
log.info(f"  Ridge Regression (alpha={best_alpha_ridge:.4f}): {results['RidgeRegression']}")

# ──────────────────────────────────────────────────────────────
# 5. Polynomial Regression (degree 2, core features only)
# ──────────────────────────────────────────────────────────────
log.info("Training Polynomial Regression (degree=2, 4 core features)...")
poly_core_feats = ["overs_completed", "current_score", "wickets_fallen", "current_run_rate"]
poly_idx = [feat_idx[f] for f in poly_core_feats if f in feat_idx]

if len(poly_idx) >= 3:
    X_poly_tr = X_tr[:, poly_idx]
    X_poly_te = X_te[:, poly_idx]

    poly_pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=np.logspace(-2, 4, 30), cv=5)),
    ])
    poly_pipe.fit(X_poly_tr, y_tr)
    poly_preds = poly_pipe.predict(X_poly_te)
    results["PolyRegression_d2"] = metrics(y_te, poly_preds)
    log.info(f"  Polynomial d=2: {results['PolyRegression_d2']}")
else:
    log.warning("Not enough core features for polynomial regression.")

# ──────────────────────────────────────────────────────────────
# 6. Load pre-trained ML models for comparison
# ──────────────────────────────────────────────────────────────
log.info("Loading pre-trained V2 ML models for comparison...")
for model_name, pkl_name in [("XGBoost_V2", "xgboost_v2"),
                              ("LightGBM_V2", "lightgbm_v2"),
                              ("CatBoost_V2", "catboost_v2")]:
    try:
        model = load_model(pkl_name)
        preds = model.predict(X_te)
        results[model_name] = metrics(y_te, preds)
        log.info(f"  {model_name}: {results[model_name]}")
    except FileNotFoundError:
        log.warning(f"  {model_name}: not found, skipping.")

# DLS
if "dls_predicted_final" in test.columns:
    dls_preds = test["dls_predicted_final"].values
    results["DLS"] = metrics(y_te, dls_preds)
    log.info(f"  DLS: {results['DLS']}")

# ──────────────────────────────────────────────────────────────
# 7. Report and Save
# ──────────────────────────────────────────────────────────────
result_df = (
    pd.DataFrame(results).T
    .reset_index()
    .rename(columns={"index": "model"})
    .sort_values("rmse")
)
result_df.to_csv(METRICS / "baseline_comparison.csv", index=False)

log.info("=" * 60)
log.info("BASELINE COMPARISON RESULTS (sorted by RMSE)")
log.info("=" * 60)
log.info(f"\n{result_df.to_string(index=False)}")

# ──────────────────────────────────────────────────────────────
# 8. Bar chart comparison
# ──────────────────────────────────────────────────────────────
log.info("Generating baseline comparison plot...")

# Colour code by model type
type_colours = {
    "NaiveProjection":    "#aaaaaa",
    "LinearRegression":   "#cccccc",
    "RidgeRegression":    "#999999",
    "PolyRegression_d2":  "#bbbbbb",
    "DLS":                "#ff7f00",
    "XGBoost_V2":         "#e41a1c",
    "LightGBM_V2":        "#377eb8",
    "CatBoost_V2":        "#4daf4a",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

models = result_df["model"].tolist()
rmses  = result_df["rmse"].tolist()
r2s    = result_df["r2"].tolist()
colours = [type_colours.get(m, "#888888") for m in models]

bars1 = ax1.barh(models, rmses, color=colours, edgecolor="black", linewidth=0.5)
ax1.set_xlabel("RMSE (runs)", fontsize=12)
ax1.set_title("Model Comparison: RMSE", fontsize=13)
ax1.invert_yaxis()
for bar, val in zip(bars1, rmses):
    ax1.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va="center", ha="left", fontsize=9)

bars2 = ax2.barh(models, r2s, color=colours, edgecolor="black", linewidth=0.5)
ax2.set_xlabel("R²", fontsize=12)
ax2.set_title("Model Comparison: R²", fontsize=13)
ax2.invert_yaxis()
for bar, val in zip(bars2, r2s):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left", fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / "baseline_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"  Saved: {FIGURES / 'baseline_comparison.png'}")
log.info(f"  Results: {METRICS / 'baseline_comparison.csv'}")
