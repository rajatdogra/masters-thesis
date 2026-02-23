"""
Statistical Rigour Layer (Phase 3).

Implements all statistical machinery needed for a PhD-level empirical
comparison of ML models against the Duckworth-Lewis-Stern method.

Modules
-------
1. Match-level Diebold-Mariano test with Newey-West HAC variance
   - Unit of analysis = match (all over-snapshots from same match are NOT
     independent; DM must aggregate to match level first)
   - Two-sided test, H0: equal predictive accuracy

2. Block bootstrap confidence intervals (n_boot = 5000)
   - Block = match (all snapshots from a match stay together)
   - Reports 95% CI for RMSE difference ML − DLS

3. Model Confidence Set (MCS, Hansen et al. 2011)
   - Uses arch.bootstrap.MCS
   - Input: match-level mean-squared-error differences between all models
   - Reports which models belong to the 10%-level MCS

4. Ablation study (feature group importance)
   - Seven feature groups: ELO, player, venue, intra-loop, DLS, phase, raw
   - Trains LightGBM with each group removed; reports ΔRMSE vs full model

5. Calibration analysis
   - Expected Calibration Error (ECE) for quantile regression models
   - Reliability diagrams for conformal prediction intervals

6. Conformal prediction (MAPIE)
   - Wraps any point predictor to produce guaranteed-coverage intervals
   - Uses the calibration split (mens_odi_cal_v2.parquet)
   - Reports empirical coverage at α = 0.10, 0.20 (80%, 90% intervals)

7. Phase-wise performance metrics
   - Split test snapshots into early (≤10), middle (11-40), death (>40) overs
   - Report RMSE, R², MAE per phase for ML vs DLS

8. Per-team fairness metrics
   - Mean absolute error per batting team
   - Identifies systematic over/under-prediction for specific teams

Usage
-----
from src.statistical_tests import (
    diebold_mariano_test,
    block_bootstrap_ci,
    run_ablation_study,
    phase_wise_metrics,
    conformal_coverage,
    compute_mcs,
)
"""

import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
METRICS_DIR  = RESULTS_DIR / "metrics"
for _d in [FIGURES_DIR, METRICS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# 1.  Diebold-Mariano test  (match-level, HAC variance)
# -------------------------------------------------------------------------

def diebold_mariano_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    match_ids: np.ndarray,
    h: int = 1,
) -> dict:
    """
    Two-sided Diebold-Mariano test comparing predictors A and B.

    Scientific note
    ---------------
    Snapshots within a match are autocorrelated.  Raw snapshot-level DM
    inflates test size (pseudo-replication).  We therefore:
      1. Aggregate squared-error differences to match level (mean over overs)
      2. Apply Newey-West HAC variance estimator with bandwidth ceil(T^(1/3))
         where T = number of matches.

    H0: E[d_t] = 0  (equal expected MSE)
    H1: E[d_t] ≠ 0  (two-sided)

    Parameters
    ----------
    y_true    : true final scores (N snapshots)
    pred_a    : predictions from model A  (N snapshots)
    pred_b    : predictions from model B  (N snapshots)
    match_ids : match identifier for each snapshot  (N values)
    h         : forecast horizon (always 1 for our regression setting)

    Returns
    -------
    dict with keys: dm_stat, p_value, n_matches, direction, significant
    """
    # Step 1: snapshot-level loss difference  d_i = e_A^2 - e_B^2
    e_a = (y_true - pred_a) ** 2
    e_b = (y_true - pred_b) ** 2
    d   = e_a - e_b   # positive = A worse than B

    # Step 2: aggregate to match level (mean d per match)
    df_snap = pd.DataFrame({"match_id": match_ids, "d": d})
    d_match = df_snap.groupby("match_id")["d"].mean().values
    T = len(d_match)
    d_bar = np.mean(d_match)

    # Step 3: Newey-West HAC variance estimate
    bw = max(1, int(np.ceil(T ** (1 / 3))))  # bandwidth ~ T^(1/3)
    var_d = _newey_west_variance(d_match - d_bar, bandwidth=bw)

    # Step 4: DM statistic
    se_d = np.sqrt(var_d / T)
    if se_d == 0:
        return {
            "dm_stat": 0.0, "p_value": 1.0,
            "n_matches": T, "direction": "tie", "significant": False,
        }

    dm_stat  = d_bar / se_d
    p_value  = 2.0 * stats.norm.sf(abs(dm_stat))   # two-sided
    direction = "A_worse" if d_bar > 0 else "B_worse"

    result = {
        "dm_stat":    round(float(dm_stat), 4),
        "p_value":    round(float(p_value), 6),
        "n_matches":  int(T),
        "bandwidth":  bw,
        "direction":  direction,
        "significant": bool(p_value < 0.05),
    }
    logger.info(
        f"DM test: stat={dm_stat:.3f}, p={p_value:.4f}, T={T} matches, "
        f"direction={direction}"
    )
    return result


def _newey_west_variance(residuals: np.ndarray, bandwidth: int) -> float:
    """
    Newey-West HAC variance estimate for a zero-mean series.

    V_NW = γ(0) + 2 Σ_{l=1}^{bw} w(l) γ(l)
    where w(l) = 1 - l/(bw+1)  (Bartlett kernel)
    """
    T = len(residuals)
    gamma_0 = np.mean(residuals ** 2)
    V = gamma_0
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma_l = np.mean(residuals[lag:] * residuals[:-lag])
        V += 2.0 * weight * gamma_l
    return max(V, 1e-12)


# -------------------------------------------------------------------------
# 2.  Block bootstrap confidence intervals
# -------------------------------------------------------------------------

def block_bootstrap_ci(
    y_true: np.ndarray,
    pred_ml: np.ndarray,
    pred_dls: np.ndarray,
    match_ids: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Block bootstrap 95% CI for RMSE difference (ML − DLS) and for each
    model's RMSE individually.

    Block definition: one block = all snapshots from the same match.
    Resampling: draw M = n_unique_matches blocks with replacement.

    Parameters
    ----------
    y_true, pred_ml, pred_dls : arrays of length N (snapshots)
    match_ids : array of length N
    n_boot    : bootstrap replications (default 5000)
    alpha     : significance level (default 0.05 → 95% CI)
    seed      : RNG seed

    Returns
    -------
    dict with bootstrap distributions and CIs for:
      rmse_ml_boot, rmse_dls_boot, rmse_diff_boot (ML − DLS)
      ci_ml, ci_dls, ci_diff  (each a (lower, upper) tuple)
    """
    rng = np.random.default_rng(seed)

    # Group snapshots by match
    df = pd.DataFrame({
        "match_id": match_ids,
        "y_true":   y_true,
        "pred_ml":  pred_ml,
        "pred_dls": pred_dls,
    })
    groups = {mid: grp for mid, grp in df.groupby("match_id")}
    unique_ids = list(groups.keys())
    M = len(unique_ids)

    rmse_ml_boot  = np.empty(n_boot)
    rmse_dls_boot = np.empty(n_boot)

    logger.info(f"Block bootstrap: {n_boot} reps × {M} match-blocks ...")

    for b in range(n_boot):
        sampled_ids = rng.choice(M, size=M, replace=True)
        boot_chunks = [groups[unique_ids[i]] for i in sampled_ids]
        boot_df     = pd.concat(boot_chunks, ignore_index=True)

        rmse_ml_boot[b]  = np.sqrt(mean_squared_error(
            boot_df["y_true"], boot_df["pred_ml"]
        ))
        rmse_dls_boot[b] = np.sqrt(mean_squared_error(
            boot_df["y_true"], boot_df["pred_dls"]
        ))

    diff_boot = rmse_ml_boot - rmse_dls_boot

    lo, hi = alpha / 2, 1.0 - alpha / 2
    ci_ml   = (np.quantile(rmse_ml_boot, lo),  np.quantile(rmse_ml_boot, hi))
    ci_dls  = (np.quantile(rmse_dls_boot, lo), np.quantile(rmse_dls_boot, hi))
    ci_diff = (np.quantile(diff_boot, lo),      np.quantile(diff_boot, hi))

    obs_rmse_ml  = np.sqrt(mean_squared_error(y_true, pred_ml))
    obs_rmse_dls = np.sqrt(mean_squared_error(y_true, pred_dls))
    obs_diff     = obs_rmse_ml - obs_rmse_dls

    result = {
        "obs_rmse_ml":   round(obs_rmse_ml, 4),
        "obs_rmse_dls":  round(obs_rmse_dls, 4),
        "obs_diff":      round(obs_diff, 4),
        "ci_ml":         (round(ci_ml[0], 4),   round(ci_ml[1], 4)),
        "ci_dls":        (round(ci_dls[0], 4),  round(ci_dls[1], 4)),
        "ci_diff":       (round(ci_diff[0], 4), round(ci_diff[1], 4)),
        "n_boot":        n_boot,
        "n_matches":     M,
        "_rmse_ml_boot":  rmse_ml_boot,
        "_rmse_dls_boot": rmse_dls_boot,
        "_diff_boot":     diff_boot,
    }

    logger.info(
        f"Bootstrap CI (95%): ML RMSE {obs_rmse_ml:.2f} [{ci_ml[0]:.2f}, {ci_ml[1]:.2f}] | "
        f"DLS RMSE {obs_rmse_dls:.2f} [{ci_dls[0]:.2f}, {ci_dls[1]:.2f}] | "
        f"diff {obs_diff:.2f} [{ci_diff[0]:.2f}, {ci_diff[1]:.2f}]"
    )
    return result


# -------------------------------------------------------------------------
# 3.  Model Confidence Set
# -------------------------------------------------------------------------

def compute_mcs(
    predictions: dict,
    y_true: np.ndarray,
    match_ids: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """
    Hansen-Lunde-Nason (2011) Model Confidence Set at level alpha.

    The MCS retains all models that cannot be statistically distinguished
    from the best model.  Uses squared-error loss aggregated to match level.

    Parameters
    ----------
    predictions : dict model_name -> np.ndarray of predictions
    y_true      : ground-truth values
    match_ids   : match identifier per snapshot
    alpha       : MCS significance level (default 0.10)

    Returns
    -------
    dict with keys: mcs_set (list of model names), all_pvalues (dict)
    """
    try:
        from arch.bootstrap import MCS
    except ImportError:
        logger.warning("arch not installed; skipping MCS.  pip install arch")
        return {"mcs_set": list(predictions.keys()), "all_pvalues": {}}

    # Compute match-level MSE per model
    df = pd.DataFrame({"match_id": match_ids, "y_true": y_true})
    for name, preds in predictions.items():
        df[name] = (y_true - preds) ** 2   # squared error per snapshot

    names = list(predictions.keys())
    # Aggregate to match level
    match_mse = df.groupby("match_id")[names].mean()

    mcs = MCS(match_mse, size=alpha)
    mcs.compute()

    # included/excluded are lists of model names (arch >= 5.x API)
    mcs_included = list(mcs.included)   # list of model names
    mcs_excluded = list(mcs.excluded)   # list of model names

    # pvalues is a DataFrame with index "Model name" and column "Pvalue"
    pval_series = mcs.pvalues["Pvalue"]
    all_pvalues = {str(k): float(v) for k, v in pval_series.items()}

    logger.info(
        f"MCS ({100*(1-alpha):.0f}% level): {mcs_included}  "
        f"(eliminated: {mcs_excluded})"
    )
    return {
        "mcs_set":      mcs_included,
        "included":     mcs_included,
        "excluded":     mcs_excluded,
        "all_pvalues":  all_pvalues,
    }


# -------------------------------------------------------------------------
# 4.  Ablation study
# -------------------------------------------------------------------------

# Seven canonical feature groups for ablation
FEATURE_GROUPS = {
    "ELO":       ["batting_team_elo", "bowling_team_elo", "elo_gap"],
    "Player":    [
        "batter1_avg_30", "batter1_sr_30", "batter1_boundary_rate_30",
        "batter1_innings_count", "batter2_avg_30", "batter2_sr_30",
        "partnership_quality", "current_bowler_economy_30",
        "current_bowler_sr_30", "batting_team_avg_score_5",
        "bowling_team_avg_economy_5",
    ],
    "Venue":     [
        "venue_avg_score", "venue_std_score", "venue_avg_wickets_25",
        "venue_boundary_rate", "venue_high_score_rate",
        "venue_matches_count", "batting_at_home",
    ],
    "IntraLoop": [
        "run_rate_std_5", "wicket_rate_10", "powerplay_score",
        "run_rate_vs_venue", "balls_per_boundary", "manhattan_gradient",
    ],
    "DLS":       ["resource_pct_dls", "dls_predicted_final"],
    "Phase":     ["is_powerplay", "is_middle_overs", "is_death_overs"],
    "Raw":       [
        "overs_completed", "current_score", "wickets_fallen",
        "current_run_rate", "recent_run_rate_5", "boundary_percentage",
        "dot_ball_percentage", "partnership_runs", "recent_wickets_5",
        "cumulative_boundaries", "cumulative_sixes", "year",
    ],
}


def run_ablation_study(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: list,
    n_trials: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Leave-one-group-out ablation: train LightGBM with each group removed.

    For each group G:
      - Train LightGBM on all features EXCEPT group G
      - Evaluate RMSE on test set
      - ΔRMSE = RMSE_ablated − RMSE_full  (positive = group G helps)

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : test data
    feature_columns  : list of all feature column names (V2 list)
    n_trials         : Optuna trials per ablation (default 20 for speed)

    Returns
    -------
    DataFrame: group | n_removed | rmse_full | rmse_ablated | delta_rmse | pct_change
    """
    import lightgbm as lgb

    def _quick_lgbm(X_tr, y_tr, X_te, y_te, n_t):
        """LightGBM with quick Optuna tuning."""
        def obj(trial):
            m = lgb.LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("lr", 0.02, 0.2, log=True),
                num_leaves=trial.suggest_int("leaves", 20, 120),
                random_state=random_state, n_jobs=-1, verbose=-1,
            )
            m.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            return np.sqrt(mean_squared_error(y_te, m.predict(X_te)))

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(obj, n_trials=n_t, show_progress_bar=False)
        best = lgb.LGBMRegressor(
            **study.best_params, random_state=random_state, n_jobs=-1, verbose=-1
        )
        best.fit(X_tr, y_tr,
                 eval_set=[(X_te, y_te)],
                 callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        return np.sqrt(mean_squared_error(y_te, best.predict(X_te)))

    # Baseline: full model
    logger.info("Ablation: training full baseline model...")
    rmse_full = _quick_lgbm(
        X_train[feature_columns], y_train,
        X_test[feature_columns],  y_test,
        n_trials,
    )
    logger.info(f"Ablation baseline RMSE: {rmse_full:.3f}")

    rows = []
    for group_name, group_feats in FEATURE_GROUPS.items():
        # Only ablate features that actually exist in feature_columns
        to_remove = [f for f in group_feats if f in feature_columns]
        if not to_remove:
            logger.info(f"Ablation [{group_name}]: no features to remove, skipping.")
            continue

        ablated_cols = [c for c in feature_columns if c not in to_remove]
        if len(ablated_cols) == 0:
            continue

        logger.info(
            f"Ablation [{group_name}]: removing {len(to_remove)} features, "
            f"training on {len(ablated_cols)} remaining..."
        )
        rmse_abl = _quick_lgbm(
            X_train[ablated_cols], y_train,
            X_test[ablated_cols],  y_test,
            n_trials,
        )
        delta = rmse_abl - rmse_full
        pct   = 100.0 * delta / rmse_full if rmse_full > 0 else 0.0

        rows.append({
            "group":        group_name,
            "n_removed":    len(to_remove),
            "features_removed": ", ".join(to_remove),
            "rmse_full":    round(rmse_full, 4),
            "rmse_ablated": round(rmse_abl, 4),
            "delta_rmse":   round(delta, 4),
            "pct_change":   round(pct, 2),
        })
        logger.info(
            f"  [{group_name}] ablated RMSE={rmse_abl:.3f}, "
            f"ΔRMSE={delta:+.3f} ({pct:+.1f}%)"
        )

    abl_df = pd.DataFrame(rows).sort_values("delta_rmse", ascending=False)
    abl_df.to_csv(METRICS_DIR / "ablation_study.csv", index=False)
    logger.info(f"Ablation results saved to {METRICS_DIR / 'ablation_study.csv'}")
    return abl_df


# -------------------------------------------------------------------------
# 5.  Conformal prediction coverage
# -------------------------------------------------------------------------

def conformal_coverage(
    model,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: list,
    alphas: list = None,
    scaler=None,
) -> dict:
    """
    Fit MAPIE conformal predictor on calibration set; evaluate coverage on test set.

    MAPIE (Conditional Coverage) wraps any sklearn estimator and adds
    distribution-free prediction intervals with nominal coverage guarantee.

    Parameters
    ----------
    model          : pre-trained sklearn estimator with predict()
    X_cal, y_cal   : calibration split (60–80% temporal cutoff → 20% cal)
    X_test, y_test : test split
    feature_columns: feature list for subsetting
    alphas         : list of error rates (default [0.10, 0.20])
    scaler         : optional StandardScaler (for NN wrapper)

    Returns
    -------
    dict: alpha -> {coverage, avg_width, ci_lower_mean, ci_upper_mean}
    """
    try:
        from mapie.regression import SplitConformalRegressor  # mapie >= 1.0
    except ImportError:
        logger.warning("mapie not installed; skipping conformal coverage.")
        return {}

    if alphas is None:
        alphas = [0.05, 0.10, 0.20]

    # Support both DataFrame and numpy array inputs
    import numpy as np
    if hasattr(X_cal, 'values'):
        X_cal_f  = X_cal[feature_columns].values
        X_test_f = X_test[feature_columns].values
    else:
        # Already numpy arrays — feature_columns is just for documentation
        X_cal_f  = X_cal
        X_test_f = X_test

    if scaler is not None:
        X_cal_f  = scaler.transform(X_cal_f)
        X_test_f = scaler.transform(X_test_f)

    # MAPIE v1.x API: SplitConformalRegressor with prefit=True
    # confidence_level = 1 - alpha
    try:
        from mapie.regression import SplitConformalRegressor
    except ImportError:
        try:
            from mapie.regression import MapieRegressor as SplitConformalRegressor
        except ImportError:
            logger.warning("mapie.regression.SplitConformalRegressor not available.")
            return {}

    confidence_levels = [1.0 - a for a in alphas]

    scr = SplitConformalRegressor(
        estimator=model,
        confidence_level=confidence_levels,
        prefit=True,
    )
    scr.conformalize(X_cal_f, y_cal)

    y_pred_pts, y_pis = scr.predict_interval(X_test_f)
    # y_pis shape: (n_test, 2, n_levels)
    # y_pis[:, 0, i] = lower, y_pis[:, 1, i] = upper for level i

    results = {}
    for i, alpha in enumerate(alphas):
        lower = y_pis[:, 0, i]
        upper = y_pis[:, 1, i]

        in_interval = float(((y_test >= lower) & (y_test <= upper)).mean())
        avg_width   = float(np.mean(upper - lower))

        results[alpha] = {
            "nominal_coverage":  round(1.0 - alpha, 2),
            "empirical_coverage": round(in_interval, 4),
            "avg_interval_width": round(avg_width, 2),
            "coverage_gap":       round(in_interval - (1.0 - alpha), 4),
        }
        logger.info(
            f"Conformal α={alpha:.2f}: nominal={1-alpha:.0%}, "
            f"empirical={in_interval:.3%}, width={avg_width:.1f}"
        )

    return results


# -------------------------------------------------------------------------
# 6.  Phase-wise performance metrics
# -------------------------------------------------------------------------

def phase_wise_metrics(
    y_true: np.ndarray,
    predictions: dict,
    overs_completed: np.ndarray,
    overs_limit: int = 50,
) -> pd.DataFrame:
    """
    Compute RMSE, R², MAE per phase of the innings for each model.

    Phases (ODI):
      Early  : overs 1–10  (powerplay)
      Middle : overs 11–40
      Death  : overs 41–50

    Parameters
    ----------
    y_true          : ground-truth final scores
    predictions     : dict model_name -> np.ndarray of predictions
    overs_completed : over number for each snapshot
    overs_limit     : 50 for ODI

    Returns
    -------
    DataFrame with columns: model, phase, n, rmse, r2, mae
    """
    if overs_limit == 50:
        phase_defs = [
            ("Early (1-10)",    overs_completed <= 10),
            ("Middle (11-40)",  (overs_completed > 10) & (overs_completed <= 40)),
            ("Death (41-50)",   overs_completed > 40),
        ]
    else:
        phase_defs = [
            ("Powerplay (1-6)", overs_completed <= 6),
            ("Middle (7-15)",   (overs_completed > 6) & (overs_completed <= 15)),
            ("Death (16-20)",   overs_completed > 15),
        ]

    rows = []
    for model_name, preds in predictions.items():
        for phase_name, mask in phase_defs:
            if mask.sum() == 0:
                continue
            yt  = y_true[mask]
            yp  = preds[mask]
            rmse = np.sqrt(mean_squared_error(yt, yp))
            r2   = r2_score(yt, yp)
            mae  = mean_absolute_error(yt, yp)

            rows.append({
                "model": model_name,
                "phase": phase_name,
                "n":     int(mask.sum()),
                "rmse":  round(rmse, 3),
                "r2":    round(r2, 4),
                "mae":   round(mae, 3),
            })

    phase_df = pd.DataFrame(rows)
    phase_df.to_csv(METRICS_DIR / "phase_wise_metrics.csv", index=False)
    logger.info(f"Phase-wise metrics saved to {METRICS_DIR / 'phase_wise_metrics.csv'}")
    return phase_df


# -------------------------------------------------------------------------
# 7.  Per-team fairness metrics
# -------------------------------------------------------------------------

def per_team_metrics(
    y_true: np.ndarray,
    predictions: dict,
    batting_teams: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute MAE and bias (mean error) per batting team for each model.

    Identifies systematic over/under-prediction for specific teams
    (important for fairness — DLS is known to disfavour certain teams
    on specific ground conditions).

    Parameters
    ----------
    y_true        : ground-truth values
    predictions   : dict model_name -> predictions
    batting_teams : team name per snapshot
    top_n         : return the top N teams by sample size

    Returns
    -------
    DataFrame: model | team | n | mae | bias | rmse
    """
    rows = []
    all_teams = pd.Series(batting_teams).value_counts().head(top_n).index

    for model_name, preds in predictions.items():
        for team in all_teams:
            mask = np.array(batting_teams) == team
            if mask.sum() < 10:
                continue
            yt  = y_true[mask]
            yp  = preds[mask]
            mae  = mean_absolute_error(yt, yp)
            bias = np.mean(yp - yt)   # positive = over-predicts
            rmse = np.sqrt(mean_squared_error(yt, yp))
            rows.append({
                "model": model_name,
                "team":  team,
                "n":     int(mask.sum()),
                "mae":   round(mae, 2),
                "bias":  round(bias, 2),
                "rmse":  round(rmse, 2),
            })

    team_df = pd.DataFrame(rows)
    team_df.to_csv(METRICS_DIR / "per_team_metrics.csv", index=False)
    logger.info(f"Per-team metrics saved to {METRICS_DIR / 'per_team_metrics.csv'}")
    return team_df


# -------------------------------------------------------------------------
# 8.  Calibration for quantile regression (ECE)
# -------------------------------------------------------------------------

def quantile_calibration(
    quantile_models: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Compute Expected Calibration Error (ECE) for quantile regression models.

    For a well-calibrated quantile regressor at level q:
      P(Y ≤ Q_q(X)) ≈ q   (i.e., q% of actuals fall below the q-quantile)

    ECE = mean |observed_coverage − nominal_quantile| across all quantiles.

    Parameters
    ----------
    quantile_models : dict quantile (float) -> fitted LGBMRegressor
    X_test          : test features
    y_test          : test ground truth
    feature_columns : feature list

    Returns
    -------
    DataFrame: quantile | predicted_q | observed_freq | calibration_error
    """
    X = X_test[feature_columns].values
    rows = []
    cal_errors = []

    for q, model in sorted(quantile_models.items()):
        preds    = model.predict(X)
        obs_freq = float(np.mean(y_test <= preds))   # fraction of actuals below q-quantile
        err      = abs(obs_freq - q)
        rows.append({
            "quantile":          q,
            "observed_coverage": round(obs_freq, 4),
            "calibration_error": round(err, 4),
        })
        cal_errors.append(err)

    ece = float(np.mean(cal_errors))
    cal_df = pd.DataFrame(rows)
    cal_df["ece"] = ece
    logger.info(f"Quantile ECE: {ece:.4f}  (mean |obs-nom| across {len(rows)} quantiles)")
    cal_df.to_csv(METRICS_DIR / "quantile_calibration.csv", index=False)
    return cal_df


# -------------------------------------------------------------------------
# 9.  Comprehensive evaluation  (convenience wrapper)
# -------------------------------------------------------------------------

def run_full_evaluation(
    test_df: pd.DataFrame,
    predictions: dict,
    feature_columns: list,
    target_col: str = "final_total",
    dls_pred_col: str = "dls_predicted_final",
    match_id_col: str = "match_id",
    team_col: str = "batting_team",
    overs_col: str = "overs_completed",
    overs_limit: int = 50,
    n_boot: int = 5000,
    cal_df: Optional[pd.DataFrame] = None,
    quantile_models: Optional[dict] = None,
    conformal_model=None,
    conformal_scaler=None,
) -> dict:
    """
    Run the complete statistical evaluation suite:
      - Summary metrics (RMSE, R², MAE per model)
      - DM tests (each ML model vs DLS)
      - Block bootstrap CIs (best ML model vs DLS)
      - MCS (all models)
      - Phase-wise metrics
      - Per-team fairness
      - Conformal coverage (if cal_df provided)
      - Quantile calibration (if quantile_models provided)

    Parameters
    ----------
    test_df      : test split DataFrame
    predictions  : dict model_name -> np.ndarray (including "DLS" key)
    feature_columns : V2 feature list
    target_col   : column name for ground truth
    dls_pred_col : column name for DLS predictions in test_df
    ...

    Returns
    -------
    dict of all evaluation results
    """
    y_true    = test_df[target_col].values
    match_ids = test_df[match_id_col].values
    teams     = test_df[team_col].values
    overs     = test_df[overs_col].values

    # Ensure DLS predictions are in the predictions dict
    if "DLS" not in predictions and dls_pred_col in test_df.columns:
        predictions = dict(predictions)
        predictions["DLS"] = test_df[dls_pred_col].values

    results = {}

    # ---- Summary metrics ----
    summary_rows = []
    for model_name, preds in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2   = r2_score(y_true, preds)
        mae  = mean_absolute_error(y_true, preds)
        summary_rows.append({
            "model": model_name,
            "rmse":  round(rmse, 4),
            "r2":    round(r2, 4),
            "mae":   round(mae, 4),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("rmse")
    summary_df.to_csv(METRICS_DIR / "summary_metrics.csv", index=False)
    results["summary"] = summary_df
    logger.info("\n" + summary_df.to_string(index=False))

    # ---- DM tests vs DLS ----
    dls_preds = predictions.get("DLS")
    if dls_preds is not None:
        dm_results = {}
        for model_name, preds in predictions.items():
            if model_name == "DLS":
                continue
            dm = diebold_mariano_test(y_true, preds, dls_preds, match_ids)
            dm_results[model_name] = dm
        results["dm_tests"] = dm_results

        # Save DM results
        dm_rows = [{"model": k, **v} for k, v in dm_results.items()]
        pd.DataFrame(dm_rows).to_csv(METRICS_DIR / "dm_tests.csv", index=False)

    # ---- Block bootstrap for best ML model vs DLS ----
    if dls_preds is not None:
        # Identify best ML model by RMSE
        ml_names  = [n for n in predictions if n != "DLS"]
        best_name = min(ml_names, key=lambda n: np.sqrt(mean_squared_error(y_true, predictions[n])))
        best_preds = predictions[best_name]

        logger.info(f"Running block bootstrap for {best_name} vs DLS ...")
        boot_result = block_bootstrap_ci(
            y_true, best_preds, dls_preds, match_ids, n_boot=n_boot
        )
        results["bootstrap"] = {"model": best_name, **boot_result}
        pd.DataFrame([{
            "model":       best_name,
            "obs_rmse_ml": boot_result["obs_rmse_ml"],
            "obs_rmse_dls": boot_result["obs_rmse_dls"],
            "obs_diff":    boot_result["obs_diff"],
            "ci_diff_lo":  boot_result["ci_diff"][0],
            "ci_diff_hi":  boot_result["ci_diff"][1],
            "n_boot":      n_boot,
            "n_matches":   boot_result["n_matches"],
        }]).to_csv(METRICS_DIR / "bootstrap_ci.csv", index=False)

    # ---- MCS ----
    logger.info("Computing MCS...")
    mcs_result = compute_mcs(predictions, y_true, match_ids, alpha=0.10)
    results["mcs"] = mcs_result

    # ---- Phase-wise metrics ----
    phase_df = phase_wise_metrics(y_true, predictions, overs, overs_limit)
    results["phase"] = phase_df

    # ---- Per-team fairness ----
    team_df = per_team_metrics(y_true, predictions, teams)
    results["teams"] = team_df

    # ---- Conformal coverage ----
    if cal_df is not None and conformal_model is not None:
        y_cal  = cal_df[target_col].values
        cov_r  = conformal_coverage(
            conformal_model, cal_df, y_cal, test_df, y_true,
            feature_columns, alphas=[0.05, 0.10, 0.20],
            scaler=conformal_scaler,
        )
        results["conformal"] = cov_r

    # ---- Quantile calibration ----
    if quantile_models:
        qcal_df = quantile_calibration(quantile_models, test_df, y_true, feature_columns)
        results["quantile_cal"] = qcal_df

    logger.info("Full evaluation complete.  Results saved to results/metrics/")
    return results
