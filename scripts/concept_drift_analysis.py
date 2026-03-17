"""
Concept Drift / Temporal Stability Analysis.

Motivation
----------
Cricket evolves — Twenty20 influence, new power-play rules, fielding restrictions,
and changes in batting technique mean that models trained on 2006 data may not
apply well to 2023 cricket. This analysis answers:
  "Does the model performance degrade as time passes?"

Method
------
1. Load test set (most recent 20% of matches, temporally held out)
2. Load pre-trained V2 models (XGBoost, LightGBM, CatBoost)
3. Compute per-year RMSE, R², MAE for each model AND DLS baseline
4. Plot RMSE over time for visual inspection of drift
5. Compute Pearson correlation(year, RMSE) as a trend metric
   - Positive correlation → performance degrades over time (bad)
   - Negative correlation → performance improves over time (good)
   - Near-zero → no drift (expected/hoped for)

Output
------
  results/metrics/concept_drift.csv     — per-year metrics (all models + DLS)
  results/metrics/drift_trend.json      — trend statistics (Pearson r, p-value)
  results/figures/concept_drift.png     — RMSE over time plot
  results/figures/r2_over_time.png      — R² over time plot

Runtime: ~5–10 minutes
"""

import sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FEATURE_COLUMNS_V2, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("concept_drift")

PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "results" / "models"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)


def rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def load_model(name: str):
    path = MODELS / f"mens_odi_{name}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"]


# ──────────────────────────────────────────────────────────────
# 1. Load test set (+ optionally train for in-sample drift check)
# ──────────────────────────────────────────────────────────────
log.info("Loading test set...")
test = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")
test["date"] = pd.to_datetime(test["date"], errors="coerce")
test["year"] = test["date"].dt.year

log.info(f"  Test set: {len(test)} rows, years {test['year'].min()}–{test['year'].max()}")

feats = [c for c in FEATURE_COLUMNS_V2 if c in test.columns]
X_test = test[feats].values
y_test = test[TARGET_COLUMN].values

# ──────────────────────────────────────────────────────────────
# 2. Load models
# ──────────────────────────────────────────────────────────────
log.info("Loading models...")
xgb_model = load_model("xgboost_v2")
lgb_model = load_model("lightgbm_v2")
cat_model = load_model("catboost_v2")

# Generate predictions
xgb_preds = xgb_model.predict(X_test)
lgb_preds = lgb_model.predict(X_test)
cat_preds = cat_model.predict(X_test)

predictions = {
    "XGBoost": xgb_preds,
    "LightGBM": lgb_preds,
    "CatBoost": cat_preds,
}
if "dls_predicted_final" in test.columns:
    predictions["DLS"] = test["dls_predicted_final"].values

# ──────────────────────────────────────────────────────────────
# 3. Compute per-year metrics
# ──────────────────────────────────────────────────────────────
log.info("Computing per-year metrics...")
rows = []

years = sorted(test["year"].dropna().unique().astype(int))
for yr in years:
    mask = (test["year"] == yr).values
    if mask.sum() < 50:   # skip years with very few snapshots
        log.warning(f"  Year {yr}: only {mask.sum()} snapshots, skipping.")
        continue

    yt = y_test[mask]
    for model_name, preds in predictions.items():
        yp = preds[mask]
        rows.append({
            "year":  yr,
            "model": model_name,
            "n_snapshots": int(mask.sum()),
            "n_matches": test[mask]["match_id"].nunique(),
            "rmse": round(rmse(yt, yp), 4),
            "r2":   round(r2_score(yt, yp), 4),
            "mae":  round(mean_absolute_error(yt, yp), 4),
        })
    log.info(f"  {yr}: {mask.sum()} snapshots, {test[mask]['match_id'].nunique()} matches")

drift_df = pd.DataFrame(rows)
drift_df.to_csv(METRICS / "concept_drift.csv", index=False)

# ──────────────────────────────────────────────────────────────
# 4. Compute trend statistics (Pearson correlation)
# ──────────────────────────────────────────────────────────────
log.info("Computing drift trend statistics...")
trend_stats = {}
for model_name in predictions.keys():
    mdf = drift_df[drift_df["model"] == model_name].sort_values("year")
    if len(mdf) < 3:
        continue
    r, p = stats.pearsonr(mdf["year"], mdf["rmse"])
    trend_stats[model_name] = {
        "pearson_r":   round(float(r), 4),
        "p_value":     round(float(p), 4),
        "direction":   "improving" if r < -0.1 else ("degrading" if r > 0.1 else "stable"),
        "rmse_min":    round(float(mdf["rmse"].min()), 4),
        "rmse_max":    round(float(mdf["rmse"].max()), 4),
        "rmse_range":  round(float(mdf["rmse"].max() - mdf["rmse"].min()), 4),
    }
    log.info(f"  {model_name}: Pearson r={r:.4f}, p={p:.4f} ({trend_stats[model_name]['direction']})")

with open(METRICS / "drift_trend.json", "w") as f:
    json.dump(trend_stats, f, indent=2)

# ──────────────────────────────────────────────────────────────
# 5. Plots
# ──────────────────────────────────────────────────────────────
log.info("Generating concept drift plots...")

model_styles = {
    "XGBoost":  {"color": "#e41a1c", "marker": "o", "ls": "-"},
    "LightGBM": {"color": "#377eb8", "marker": "s", "ls": "--"},
    "CatBoost": {"color": "#4daf4a", "marker": "^", "ls": "-."},
    "DLS":      {"color": "#ff7f00", "marker": "D", "ls": ":"},
}

# RMSE over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for model_name, style in model_styles.items():
    mdf = drift_df[drift_df["model"] == model_name].sort_values("year")
    if mdf.empty:
        continue
    ax1.plot(mdf["year"], mdf["rmse"],
             color=style["color"], marker=style["marker"],
             linestyle=style["ls"], linewidth=2, markersize=7,
             label=model_name)
    ax2.plot(mdf["year"], mdf["r2"],
             color=style["color"], marker=style["marker"],
             linestyle=style["ls"], linewidth=2, markersize=7,
             label=model_name)

ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("RMSE (runs)", fontsize=12)
ax1.set_title("Temporal Stability: RMSE per Year", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax2.set_xlabel("Year", fontsize=12)
ax2.set_ylabel("R²", fontsize=12)
ax2.set_title("Temporal Stability: R² per Year", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(FIGURES / "concept_drift.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"  Saved: {FIGURES / 'concept_drift.png'}")

log.info("=" * 60)
log.info("CONCEPT DRIFT SUMMARY")
log.info("=" * 60)
log.info(f"\n{drift_df.pivot_table(index='year', columns='model', values='rmse').to_string()}")
log.info(f"\nTrend statistics: {json.dumps(trend_stats, indent=2)}")
log.info("=" * 60)
