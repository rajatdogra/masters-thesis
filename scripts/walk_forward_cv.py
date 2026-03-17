"""
Walk-Forward Cross-Validation for Cricket Score Prediction.

Motivation
----------
A single 60/20/20 temporal split is susceptible to the lucky split problem:
the test performance might be unduly influenced by one particular test period.
Walk-forward CV with K=5 expanding folds provides:
  - RMSE mean ± std across folds (much more credible than a single number)
  - Evidence that model performance is stable across different test periods
  - Defence against reviewer concern about temporal generalisation

Methodology
-----------
5 folds with EXPANDING training window (the correct approach for time series data):

  Fold 1: train on 50%, validate on  50%–60%  (10% slice)
  Fold 2: train on 60%, validate on  60%–70%  (10% slice)
  Fold 3: train on 70%, validate on  70%–80%  (10% slice)
  Fold 4: train on 80%, validate on  80%–90%  (10% slice)
  Fold 5: train on 90%, validate on  90%–100% (10% slice)

Split is performed at MATCH LEVEL (not snapshot level) to prevent leakage.

For each fold:
  - LightGBM is trained with 20 Optuna trials (speed–quality trade-off)
  - DLS baseline evaluated on same fold for comparison
  - RMSE, R², MAE reported per fold

Output
------
  results/metrics/walk_forward_cv.csv   — per-fold metrics for LGB and DLS
  results/metrics/walk_forward_summary.json — mean ± std across all folds

Runtime: ~45–90 minutes (5 folds × 20 Optuna trials each)
"""

import sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FEATURE_COLUMNS_V2, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("walk_forward_cv")

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROC    = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

# Number of Optuna trials per fold (20 is fast enough for CV; full training uses 50–100)
N_TRIALS_PER_FOLD = 20
N_FOLDS = 5
SEED = 42


def rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def quick_lgbm(X_tr, y_tr, X_val, y_val, n_trials=N_TRIALS_PER_FOLD):
    """Train LightGBM with quick Optuna tuning on a single fold."""
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 200, 800),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("lr", 0.02, 0.2, log=True),
            num_leaves      = trial.suggest_int("leaves", 20, 150),
            subsample       = trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.7, 1.0),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 60),
        )
        model = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        return rmse(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = lgb.LGBMRegressor(**study.best_params, random_state=SEED, n_jobs=-1, verbose=-1)
    best.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return best, study.best_value


# ──────────────────────────────────────────────────────────────
# Load full V2 snapshot dataset
# ──────────────────────────────────────────────────────────────
log.info("Loading full V2 snapshots...")
snaps = pd.read_parquet(PROC / "mens_odi_snapshots_v2.parquet")
log.info(f"  {len(snaps)} snapshots, columns: {list(snaps.columns)[:10]}...")

# Verify features are available
feats = [c for c in FEATURE_COLUMNS_V2 if c in snaps.columns]
log.info(f"  Features available: {len(feats)} / {len(FEATURE_COLUMNS_V2)}")

# Ensure date column exists
if "date" not in snaps.columns:
    log.error("'date' column missing from snapshots! Cannot perform temporal CV.")
    sys.exit(1)

snaps["date"] = pd.to_datetime(snaps["date"], errors="coerce")
snaps = snaps.dropna(subset=["date", TARGET_COLUMN])

# ──────────────────────────────────────────────────────────────
# Build match-level timeline
# ──────────────────────────────────────────────────────────────
log.info("Building match timeline...")
match_dates = (
    snaps[["match_id", "date"]]
    .drop_duplicates("match_id")
    .sort_values("date")
    .reset_index(drop=True)
)
N_MATCHES = len(match_dates)
log.info(f"  Total matches: {N_MATCHES}  ({match_dates['date'].min().date()} → {match_dates['date'].max().date()})")

# ──────────────────────────────────────────────────────────────
# Walk-forward folds
# ──────────────────────────────────────────────────────────────
# Starting point: train on first 50%, then expand in 10% steps
start_frac = 0.50
fold_width  = 0.10   # each validation fold = 10% of all matches

fold_records = []

for fold_idx in range(N_FOLDS):
    train_frac = start_frac + fold_idx * fold_width
    val_end_frac = train_frac + fold_width

    train_end = int(N_MATCHES * train_frac)
    val_end   = min(int(N_MATCHES * val_end_frac), N_MATCHES)

    if train_end >= val_end:
        log.warning(f"Fold {fold_idx+1}: train_end >= val_end, skipping.")
        continue

    train_ids = set(match_dates.iloc[:train_end]["match_id"])
    val_ids   = set(match_dates.iloc[train_end:val_end]["match_id"])

    # Date info for logging
    val_start_date = match_dates.iloc[train_end]["date"].date()
    val_end_date   = match_dates.iloc[val_end - 1]["date"].date()

    train_snap = snaps[snaps["match_id"].isin(train_ids)]
    val_snap   = snaps[snaps["match_id"].isin(val_ids)]

    X_tr  = train_snap[feats].values
    y_tr  = train_snap[TARGET_COLUMN].values
    X_val = val_snap[feats].values
    y_val = val_snap[TARGET_COLUMN].values

    log.info(
        f"Fold {fold_idx+1}/{N_FOLDS}: "
        f"train {len(train_ids)} matches ({len(train_snap)} rows), "
        f"val {len(val_ids)} matches ({len(val_snap)} rows), "
        f"val period {val_start_date}→{val_end_date}"
    )

    # Train LightGBM
    model, val_opt_rmse = quick_lgbm(X_tr, y_tr, X_val, y_val)
    lgb_preds = model.predict(X_val)

    lgb_rmse = rmse(y_val, lgb_preds)
    lgb_r2   = r2_score(y_val, lgb_preds)
    lgb_mae  = mean_absolute_error(y_val, lgb_preds)

    # DLS baseline
    if "dls_predicted_final" in val_snap.columns:
        dls_preds = val_snap["dls_predicted_final"].values
        dls_rmse = rmse(y_val, dls_preds)
        dls_r2   = r2_score(y_val, dls_preds)
        dls_mae  = mean_absolute_error(y_val, dls_preds)
    else:
        dls_rmse = dls_r2 = dls_mae = np.nan

    fold_records.append({
        "fold":            fold_idx + 1,
        "train_matches":   len(train_ids),
        "val_matches":     len(val_ids),
        "val_start":       str(val_start_date),
        "val_end":         str(val_end_date),
        "lgb_rmse":        round(lgb_rmse, 4),
        "lgb_r2":          round(lgb_r2, 4),
        "lgb_mae":         round(lgb_mae, 4),
        "dls_rmse":        round(dls_rmse, 4) if not np.isnan(dls_rmse) else None,
        "dls_r2":          round(dls_r2, 4)   if not np.isnan(dls_r2)   else None,
        "dls_mae":         round(dls_mae, 4)  if not np.isnan(dls_mae)  else None,
        "lgb_vs_dls_delta_rmse": round(lgb_rmse - dls_rmse, 4) if not np.isnan(dls_rmse) else None,
    })

    log.info(
        f"  Fold {fold_idx+1}: LGB RMSE={lgb_rmse:.3f}, R²={lgb_r2:.4f}, "
        f"DLS RMSE={dls_rmse:.3f} (delta={lgb_rmse - dls_rmse:.3f})"
    )

# ──────────────────────────────────────────────────────────────
# Summary statistics across folds
# ──────────────────────────────────────────────────────────────
cv_df = pd.DataFrame(fold_records)
cv_df.to_csv(METRICS / "walk_forward_cv.csv", index=False)

lgb_rmses = cv_df["lgb_rmse"].values
dls_rmses = cv_df["dls_rmse"].dropna().values

summary = {
    "n_folds":          N_FOLDS,
    "n_trials_per_fold": N_TRIALS_PER_FOLD,
    "lgb_rmse_mean":    round(float(np.mean(lgb_rmses)), 4),
    "lgb_rmse_std":     round(float(np.std(lgb_rmses)),  4),
    "lgb_r2_mean":      round(float(cv_df["lgb_r2"].mean()), 4),
    "lgb_r2_std":       round(float(cv_df["lgb_r2"].std()),  4),
    "lgb_mae_mean":     round(float(cv_df["lgb_mae"].mean()), 4),
    "lgb_mae_std":      round(float(cv_df["lgb_mae"].std()),  4),
    "dls_rmse_mean":    round(float(np.mean(dls_rmses)), 4) if len(dls_rmses) > 0 else None,
    "dls_rmse_std":     round(float(np.std(dls_rmses)),  4) if len(dls_rmses) > 0 else None,
    "delta_rmse_mean":  round(float(cv_df["lgb_vs_dls_delta_rmse"].dropna().mean()), 4),
    "delta_rmse_std":   round(float(cv_df["lgb_vs_dls_delta_rmse"].dropna().std()),  4),
}

with open(METRICS / "walk_forward_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

log.info("=" * 60)
log.info("WALK-FORWARD CV SUMMARY")
log.info("=" * 60)
log.info(f"\n{cv_df.to_string(index=False)}")
log.info(f"\nLightGBM RMSE: {summary['lgb_rmse_mean']:.3f} ± {summary['lgb_rmse_std']:.3f}")
log.info(f"LightGBM R²  : {summary['lgb_r2_mean']:.4f} ± {summary['lgb_r2_std']:.4f}")
log.info(f"DLS RMSE     : {summary['dls_rmse_mean']:.3f} ± {summary['dls_rmse_std']:.3f}")
log.info(f"ML–DLS ΔRMSE : {summary['delta_rmse_mean']:.3f} ± {summary['delta_rmse_std']:.3f}")
log.info("=" * 60)
log.info(f"Results saved to {METRICS / 'walk_forward_cv.csv'}")
