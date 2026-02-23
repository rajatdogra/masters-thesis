"""
Regenerate all thesis figures with correct V1/V2 model data.

Fixes:
1. Restore V1 SHAP summary (overwritten by V2 run)
2. Regenerate model comparison bar with V2 models included
3. Regenerate phase RMSE figure with V2 + DLS R²=-1.008 data
4. Regenerate cross-model importance with V2 models
"""
import sys, pickle, logging
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("regen_figs")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "results" / "models"
FIGS    = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics"


def load_model(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("scaler")
    return obj, None


# ─────────────────────────────────────────────────────────────
# 1. Restore V1 SHAP summary (XGBoost V1 — 22 features)
# ─────────────────────────────────────────────────────────────
log.info("1. Regenerating V1 SHAP summary figure...")

from src.feature_engineering import FEATURE_COLUMNS as FEATURE_COLUMNS_V1
from src.pipeline import TARGET_COLUMN

test_v1 = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")   # V2 test set has V1 features too
feats_v1 = [c for c in FEATURE_COLUMNS_V1 if c in test_v1.columns]

xgb_v1, xgb_v1_scaler = load_model(MODELS / "mens_odi_xgboost.pkl")
lgb_v1, lgb_v1_scaler = load_model(MODELS / "mens_odi_lightgbm.pkl")
rf_v1,  rf_v1_scaler  = load_model(MODELS / "mens_odi_random_forest.pkl")

Xte_v1 = test_v1[feats_v1].values

log.info(f"   V1 features: {len(feats_v1)}, test set: {len(test_v1)}")

# XGBoost V1 SHAP
explainer_xgb_v1 = shap.TreeExplainer(xgb_v1)
sv_xgb_v1 = explainer_xgb_v1.shap_values(Xte_v1[:800])

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv_xgb_v1, Xte_v1[:800], feature_names=feats_v1,
                  max_display=20, show=False, plot_size=None)
plt.title("SHAP Summary — XGBoost (V1, 22 features)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_shap_summary_xgboost.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("   Saved mens_odi_shap_summary_xgboost.png (V1)")

# Also save V2 versions with explicit _v2 suffix for Phase 2 chapter reference
from src.pipeline import FEATURE_COLUMNS_V2
xgb_v2, xgb_v2_scaler = load_model(MODELS / "mens_odi_xgboost_v2.pkl")
lgb_v2, lgb_v2_scaler = load_model(MODELS / "mens_odi_lightgbm_v2.pkl")

test_v2 = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")
feats_v2 = [c for c in FEATURE_COLUMNS_V2 if c in test_v2.columns]
Xte_v2 = test_v2[feats_v2].values
if lgb_v2_scaler:
    Xte_v2_scaled = lgb_v2_scaler.transform(Xte_v2)
else:
    Xte_v2_scaled = Xte_v2

explainer_xgb_v2 = shap.TreeExplainer(xgb_v2)
sv_xgb_v2 = explainer_xgb_v2.shap_values(Xte_v2[:500])

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv_xgb_v2, Xte_v2[:500], feature_names=feats_v2,
                  max_display=20, show=False, plot_size=None)
plt.title("SHAP Summary — XGBoost V2 (46 features)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_shap_summary_xgboost_v2.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("   Saved mens_odi_shap_summary_xgboost_v2.png")

# LightGBM V1 SHAP
explainer_lgb_v1 = shap.TreeExplainer(lgb_v1)
sv_lgb_v1 = explainer_lgb_v1.shap_values(Xte_v1[:800])

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv_lgb_v1, Xte_v1[:800], feature_names=feats_v1,
                  max_display=20, show=False, plot_size=None)
plt.title("SHAP Summary — LightGBM (V1, 22 features)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_shap_summary_lightgbm.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("   Saved mens_odi_shap_summary_lightgbm.png (V1)")

# ─────────────────────────────────────────────────────────────
# 2. Cross-model feature importance: V1 (XGBoost, RF, LightGBM)
# ─────────────────────────────────────────────────────────────
log.info("2. Regenerating cross-model importance (V1)...")

sv_rf_v1 = None
try:
    explainer_rf_v1 = shap.TreeExplainer(rf_v1)
    sv_rf_v1 = explainer_rf_v1.shap_values(Xte_v1[:300])
    log.info("   RF SHAP computed")
except Exception as e:
    log.warning(f"   RF SHAP failed: {e}")

if sv_rf_v1 is not None:
    imp = {
        "XGBoost":     pd.Series(np.abs(sv_xgb_v1).mean(0), index=feats_v1),
        "LightGBM":    pd.Series(np.abs(sv_lgb_v1).mean(0), index=feats_v1),
        "RandomForest": pd.Series(np.abs(sv_rf_v1).mean(0), index=feats_v1),
    }
    imp_df = pd.DataFrame(imp)
    # Normalise each column
    imp_df = imp_df / imp_df.sum()
    imp_df["avg"] = imp_df.mean(axis=1)
    top20 = imp_df.nlargest(20, "avg")

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(top20))
    w = 0.25
    colors = ["#1976D2", "#43A047", "#E53935"]
    for i, (model, color) in enumerate(zip(["XGBoost","LightGBM","RandomForest"], colors)):
        ax.barh(x + (i - 1) * w, top20[model].values, height=w, label=model, color=color, alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_xlabel("Normalised mean |SHAP value|", fontsize=10)
    ax.set_title("Cross-Model Feature Importance — V1 Models (22 features)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS / "mens_odi_cross_model_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("   Saved mens_odi_cross_model_importance.png (V1)")


# ─────────────────────────────────────────────────────────────
# 3. Model comparison bar — V1 + V2 combined
# ─────────────────────────────────────────────────────────────
log.info("3. Regenerating model comparison bar chart...")

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))

yte_v1 = test_v1[TARGET_COLUMN].values

# V1 predictions
def predict(model, scaler, X):
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict(X)

preds_v1 = {
    "XGBoost V1":     predict(xgb_v1, xgb_v1_scaler, Xte_v1),
    "LightGBM V1":    predict(lgb_v1, lgb_v1_scaler, Xte_v1),
    "RandomForest V1": predict(rf_v1, rf_v1_scaler, Xte_v1),
}

# DLS
dls_col = "dls_predicted_final"
if dls_col in test_v1.columns:
    preds_v1["DLS"] = test_v1[dls_col].values

# V2 predictions
cat_v2, cat_v2_scaler = load_model(MODELS / "mens_odi_catboost_v2.pkl")
if xgb_v2_scaler:
    Xte_v2_xgb = xgb_v2_scaler.transform(Xte_v2)
else:
    Xte_v2_xgb = Xte_v2
if cat_v2_scaler:
    Xte_v2_cat = cat_v2_scaler.transform(Xte_v2)
else:
    Xte_v2_cat = Xte_v2

yte_v2 = test_v2[TARGET_COLUMN].values

preds_v2 = {
    "XGBoost V2":  xgb_v2.predict(Xte_v2_xgb),
    "LightGBM V2": lgb_v2.predict(Xte_v2_scaled),
    "CatBoost V2": cat_v2.predict(Xte_v2_cat),
}

metrics = {}
for name, pred in preds_v1.items():
    metrics[name] = {"RMSE": rmse(yte_v1, pred),
                     "R2":   r2_score(yte_v1, pred),
                     "MAE":  float(mean_absolute_error(yte_v1, pred))}

for name, pred in preds_v2.items():
    metrics[name] = {"RMSE": rmse(yte_v2, pred),
                     "R2":   r2_score(yte_v2, pred),
                     "MAE":  float(mean_absolute_error(yte_v2, pred))}

met_df = pd.DataFrame(metrics).T
log.info(f"   Metrics:\n{met_df.to_string()}")

# Ordered plot
order = ["DLS", "XGBoost V1", "LightGBM V1", "RandomForest V1",
         "XGBoost V2", "LightGBM V2", "CatBoost V2"]
order = [o for o in order if o in met_df.index]
met_df = met_df.loc[order]

colors_bar = (["#FB8C00"] +                        # DLS
              ["#90CAF9","#42A5F5","#1565C0"] +    # V1
              ["#A5D6A7","#43A047","#1B5E20"])      # V2
colors_bar = colors_bar[:len(order)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, ylabel in zip(axes,
                               ["RMSE","MAE","R2"],
                               ["RMSE (runs)","MAE (runs)","$R^2$"]):
    vals = met_df[metric].values
    bars = ax.bar(range(len(order)), vals, color=colors_bar, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(metric, fontsize=11)
    ax.axhline(0, color="black", lw=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

# Legend
v1_patch = mpatches.Patch(color="#42A5F5", label="V1 (22 features)")
v2_patch = mpatches.Patch(color="#43A047", label="V2 (46 features)")
dls_patch = mpatches.Patch(color="#FB8C00", label="DLS baseline")
axes[2].legend(handles=[dls_patch, v1_patch, v2_patch], fontsize=9, loc="lower right")

plt.suptitle("Model Performance Comparison — Men's ODI (V1 vs V2 vs DLS)", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_model_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("   Saved mens_odi_model_comparison_bar.png (V1+V2+DLS)")


# ─────────────────────────────────────────────────────────────
# 4. Phase RMSE figure — V1, V2, DLS
# ─────────────────────────────────────────────────────────────
log.info("4. Regenerating phase RMSE figure...")

phase_df = pd.read_csv(METRICS / "phase_wise_metrics.csv")
log.info(f"   Phase data:\n{phase_df.to_string(index=False)}")

phases = ["Early (1-10)", "Middle (11-40)", "Death (41-50)"]
models_phase = phase_df["model"].unique()

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(phases))
n_models = len(models_phase)
width = 0.8 / n_models

colors_phase = {
    "XGBoost_V2":   "#1976D2",
    "LightGBM_V2":  "#43A047",
    "CatBoost_V2":  "#7B1FA2",
    "DLS":          "#E53935",
    "LightGBM_V1":  "#90CAF9",
}

for i, model in enumerate(models_phase):
    sub = phase_df[phase_df["model"] == model].set_index("phase")
    vals = [sub.loc[p, "rmse"] if p in sub.index else np.nan for p in phases]
    offset = (i - n_models / 2 + 0.5) * width
    color = colors_phase.get(model, "#999999")
    bars = ax.bar(x + offset, vals, width=width * 0.9, label=model,
                  color=color, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=11)
ax.set_ylabel("RMSE (runs)", fontsize=11)
ax.set_title("Phase-wise RMSE: V1, V2, and DLS — Men's ODI", fontsize=12)
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(0, ax.get_ylim()[1] * 1.18)

# Annotate DLS R² in Early
ax.text(0, phase_df[(phase_df["model"]=="DLS") & (phase_df["phase"]=="Early (1-10)")]["rmse"].values[0] + 3,
        "$R^2=-1.01$", ha="center", va="bottom", fontsize=8, color="#E53935", fontweight="bold")

plt.tight_layout()
plt.savefig(FIGS / "mens_odi_phase_rmse.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("   Saved mens_odi_phase_rmse.png (V1+V2+DLS)")


# ─────────────────────────────────────────────────────────────
# 5. Error distributions — update to include V2
# ─────────────────────────────────────────────────────────────
log.info("5. Regenerating error distribution figure...")

fig, ax = plt.subplots(figsize=(12, 5))
color_map = {
    "DLS":          ("#FB8C00", 0.45),
    "LightGBM V1":  ("#90CAF9", 0.45),
    "LightGBM V2":  ("#43A047", 0.55),
    "CatBoost V2":  ("#7B1FA2", 0.50),
}

err_data = {
    "DLS":         (preds_v1["DLS"]         - yte_v1) if "DLS" in preds_v1 else None,
    "LightGBM V1": (preds_v1["LightGBM V1"] - yte_v1),
    "LightGBM V2": (preds_v2["LightGBM V2"] - yte_v2),
    "CatBoost V2": (preds_v2["CatBoost V2"] - yte_v2),
}
for name, (color, alpha) in color_map.items():
    err = err_data.get(name)
    if err is None:
        continue
    ax.hist(err, bins=70, alpha=alpha, label=f"{name}  (bias={err.mean():+.1f})", color=color)

ax.axvline(0, color="black", lw=1.2, ls="--")
ax.set_xlabel("Prediction Error (runs: predicted − actual)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Prediction Error Distribution — Men's ODI (V1, V2, DLS)", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_error_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("   Saved mens_odi_error_distributions.png (updated with V2)")


# ─────────────────────────────────────────────────────────────
# 6. Actual vs predicted — update to include V2 (update existing)
# ─────────────────────────────────────────────────────────────
log.info("6. Regenerating actual vs predicted scatter...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
pairs = [
    ("DLS",          preds_v1.get("DLS"),   yte_v1, "#FB8C00"),
    ("XGBoost V1",   preds_v1["XGBoost V1"],  yte_v1, "#90CAF9"),
    ("LightGBM V1",  preds_v1["LightGBM V1"], yte_v1, "#42A5F5"),
    ("XGBoost V2",   preds_v2["XGBoost V2"],  yte_v2, "#A5D6A7"),
    ("LightGBM V2",  preds_v2["LightGBM V2"], yte_v2, "#43A047"),
    ("CatBoost V2",  preds_v2["CatBoost V2"],  yte_v2, "#1B5E20"),
]
for ax, (name, pred, yte, color) in zip(axes.flat, pairs):
    if pred is None:
        ax.set_visible(False)
        continue
    ax.scatter(yte, pred, alpha=0.12, s=5, color=color)
    mn, mx = yte.min(), yte.max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.2)
    r2   = r2_score(yte, pred)
    rmse_val = np.sqrt(mean_squared_error(yte, pred))
    ax.text(0.05, 0.93, f"R²={r2:.3f}  RMSE={rmse_val:.1f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel("Actual", fontsize=9)
    ax.set_ylabel("Predicted", fontsize=9)
    ax.set_title(name, fontsize=10)

plt.suptitle("Actual vs Predicted Score — Men's ODI", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "mens_odi_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("   Saved mens_odi_actual_vs_predicted.png (V1+V2+DLS, 2×3 grid)")


log.info("=" * 60)
log.info("ALL FIGURES REGENERATED")
log.info("=" * 60)
