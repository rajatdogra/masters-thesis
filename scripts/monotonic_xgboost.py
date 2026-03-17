"""
Domain-Knowledge-Informed XGBoost with Monotonic Constraints.

Motivation
----------
A purely data-driven model may learn spurious patterns — for instance, it might
predict that a team with a higher score at over 30 will score LESS in total than
a team with a lower score at over 30 (if confounded by venue or opponent quality).

Domain knowledge provides hard constraints that should ALWAYS hold:
  - current_score ↑ → final_total ↑ (monotone +1)
  - wickets_fallen ↑ → final_total ↓ (monotone -1)
  - overs_completed ↑ → final_total ↑ (more overs = more runs realised)
  - dot_ball_percentage ↑ → final_total ↓ (more dots = less scoring)
  - recent_wickets_5 ↑ → final_total ↓ (recent collapse = lower total)
  - wicket_rate_10 ↑ → final_total ↓ (sustained pressure = lower total)
  - bowling_team_elo ↑ → final_total ↓ (stronger bowling team = lower batting total)

Monotonic constraints in XGBoost:
  monotone_constraints=(+1, +1, 0, ...) — one value per feature
  +1 = feature must monotonically increase the prediction
  -1 = feature must monotonically decrease the prediction
   0 = unconstrained

Scientific value of monotonic constraints:
  1. Eliminates physically impossible predictions
  2. Improves generalisation by injecting domain knowledge
  3. Often reduces overfitting on noisy early-innings data
  4. Makes the model more robust to out-of-distribution inputs
  5. Potentially recovers performance from V2 without extra data

Output
------
  results/models/mens_odi_xgboost_mono.pkl  — monotonic XGBoost model
  results/metrics/monotonic_results.csv     — comparison with unconstrained XGBoost

Runtime: ~30–45 minutes (100 Optuna trials)
"""

import sys, pickle, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FEATURE_COLUMNS_V2, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("monotonic_xgb")

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "results" / "models"
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

SEED     = 42
N_TRIALS = 100


def rmse_score(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def evaluate(model, X_test, y_test):
    p = model.predict(X_test)
    return {
        "rmse": round(rmse_score(y_test, p), 4),
        "r2":   round(r2_score(y_test, p), 4),
        "mae":  round(mean_absolute_error(y_test, p), 4),
    }


# ──────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────
log.info("Loading V2 parquets...")
train = pd.read_parquet(PROC / "mens_odi_train_v2.parquet")
cal   = pd.read_parquet(PROC / "mens_odi_cal_v2.parquet")
test  = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")

feats = [c for c in FEATURE_COLUMNS_V2 if c in train.columns]
log.info(f"Features: {len(feats)}")

X_tr  = train[feats].values
y_tr  = train[TARGET_COLUMN].values
X_cal = cal[feats].values
y_cal = cal[TARGET_COLUMN].values
X_te  = test[feats].values
y_te  = test[TARGET_COLUMN].values

# ──────────────────────────────────────────────────────────────
# 2. Define monotonic constraints
# ──────────────────────────────────────────────────────────────

# Map feature name → constraint direction
MONOTONE_DIRECTIONS = {
    # Positive (higher feature → higher predicted score)
    "overs_completed":            +1,
    "current_score":              +1,
    "current_run_rate":           +1,
    "recent_run_rate_5":          +1,
    "boundary_percentage":        +1,
    "partnership_runs":           +1,
    "cumulative_boundaries":      +1,
    "cumulative_sixes":           +1,
    "batting_team_elo":           +1,
    "batter1_avg_30":             +1,
    "batter1_sr_30":              +1,
    "batter1_boundary_rate_30":   +1,
    "batter2_avg_30":             +1,
    "batter2_sr_30":              +1,
    "partnership_quality":        +1,
    "batting_team_avg_score_5":   +1,
    "venue_avg_score":            +1,
    "venue_boundary_rate":        +1,
    "venue_high_score_rate":      +1,
    "batting_at_home":            +1,
    "powerplay_score":            +1,
    "run_rate_vs_venue":          +1,
    "dls_predicted_final":        +1,
    # Negative (higher feature → lower predicted score)
    "wickets_fallen":             -1,
    "dot_ball_percentage":        -1,
    "recent_wickets_5":           -1,
    "wicket_rate_10":             -1,
    "bowling_team_elo":           -1,
    "bowling_team_avg_economy_5": -1,
    "venue_avg_wickets_25":       -1,
    "balls_per_boundary":         -1,
    # Unconstrained (direction ambiguous or complex interaction)
    "scoring_acceleration":       0,  # can be negative when decelerating
    "is_powerplay":               0,
    "is_middle_overs":            0,
    "is_death_overs":             0,
    "year":                       0,
    "toss_bat_first":             0,
    "resource_pct_dls":          0,  # ambiguous: high resource = many overs left = low score so far
    "elo_gap":                    0,  # complex
    "batter1_innings_count":      0,
    "current_bowler_economy_30":  0,  # higher economy = weaker bowling = higher score, but context-dependent
    "current_bowler_sr_30":       0,
    "venue_std_score":            0,
    "venue_matches_count":        0,
    "run_rate_std_5":             0,
    "manhattan_gradient":         0,  # slope can go either way
}

# Build constraint tuple in feature order
mono_constraints = tuple(MONOTONE_DIRECTIONS.get(f, 0) for f in feats)

n_pos = sum(1 for c in mono_constraints if c == +1)
n_neg = sum(1 for c in mono_constraints if c == -1)
n_zero = sum(1 for c in mono_constraints if c == 0)
log.info(
    f"Monotonic constraints: +1={n_pos}, -1={n_neg}, 0={n_zero} (out of {len(feats)} features)"
)

# ──────────────────────────────────────────────────────────────
# 3. Train unconstrained XGBoost (baseline for comparison)
# ──────────────────────────────────────────────────────────────
log.info("Loading unconstrained XGBoost V2 (pre-trained)...")
try:
    with open(MODELS / "mens_odi_xgboost_v2.pkl", "rb") as f:
        d = pickle.load(f)
    xgb_unc = d["model"]
    unc_result = evaluate(xgb_unc, X_te, y_te)
    log.info(f"  Unconstrained XGBoost V2: {unc_result}")
except FileNotFoundError:
    log.warning("Pre-trained XGBoost V2 not found; will report N/A for comparison.")
    unc_result = None

# ──────────────────────────────────────────────────────────────
# 4. Train monotonic XGBoost with Optuna
# ──────────────────────────────────────────────────────────────
log.info(f"Training Monotonic XGBoost ({N_TRIALS} Optuna trials)...")

def objective(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":       trial.suggest_int("max_depth", 3, 10),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma":           trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }
    model = xgb.XGBRegressor(
        **params,
        monotone_constraints=mono_constraints,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], verbose=False)
    return rmse_score(y_cal, model.predict(X_cal))

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

log.info(f"Best validation RMSE: {study.best_value:.4f}")
log.info(f"Best params: {study.best_params}")

# Retrain with best params
mono_model = xgb.XGBRegressor(
    **study.best_params,
    monotone_constraints=mono_constraints,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
)
mono_model.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], verbose=False)

# ──────────────────────────────────────────────────────────────
# 5. Evaluate and compare
# ──────────────────────────────────────────────────────────────
mono_result = evaluate(mono_model, X_te, y_te)
log.info(f"Monotonic XGBoost:    {mono_result}")

results = {"MonotonicXGBoost": mono_result}
if unc_result:
    results["XGBoost_V2"] = unc_result
    d_rmse = mono_result["rmse"] - unc_result["rmse"]
    d_r2   = mono_result["r2"]   - unc_result["r2"]
    log.info(f"Monotonic vs Unconstrained: ΔRMSE={d_rmse:+.4f}, ΔR²={d_r2:+.4f}")

if "dls_predicted_final" in test.columns:
    dls_preds = test["dls_predicted_final"].values
    results["DLS"] = evaluate(mono_model, X_te, y_te)  # placeholder
    results["DLS"] = {
        "rmse": round(rmse_score(y_te, dls_preds), 4),
        "r2":   round(r2_score(y_te, dls_preds), 4),
        "mae":  round(mean_absolute_error(y_te, dls_preds), 4),
    }

# ──────────────────────────────────────────────────────────────
# 6. Save
# ──────────────────────────────────────────────────────────────
with open(MODELS / "mens_odi_xgboost_mono.pkl", "wb") as f:
    pickle.dump({"model": mono_model, "scaler": None,
                 "monotone_constraints": mono_constraints,
                 "feature_columns": feats}, f)

result_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
result_df.to_csv(METRICS / "monotonic_results.csv", index=False)

# Save constraint details
constraint_info = {
    "feature_constraints": {f: c for f, c in zip(feats, mono_constraints)},
    "n_positive":   n_pos,
    "n_negative":   n_neg,
    "n_unconstrained": n_zero,
    "best_params":  study.best_params,
    "val_rmse":     round(study.best_value, 4),
    "test_result":  mono_result,
}
with open(METRICS / "monotonic_constraints.json", "w") as f:
    json.dump(constraint_info, f, indent=2)

log.info("=" * 60)
log.info("MONOTONIC XGBOOST RESULTS")
log.info("=" * 60)
log.info(f"\n{result_df.to_string(index=False)}")
log.info(f"\nModel saved to: {MODELS / 'mens_odi_xgboost_mono.pkl'}")
log.info("=" * 60)
