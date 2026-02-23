"""
Enhanced Feature Pipeline (Phase 1+).

Orchestrates the full feature engineering process:
  1. Load and filter match data
  2. Create enriched first-innings over-boundary snapshots
     (base 22 features + 6 new intra-loop features)
  3. Add ELO team ratings (temporally safe)
  4. Add player rolling stats (temporally safe)
  5. Add venue historical statistics (temporally safe)
  6. Add DLS features (resource%, predicted final)
  7. Train / validation / test split (temporal, no leakage)
  8. Save all artefacts

New features added vs v1 (22 → 49):
  ELO (3):       batting_team_elo, bowling_team_elo, elo_gap
  Player (11):   batter1_avg_30, batter1_sr_30, batter1_boundary_rate_30,
                 batter1_innings_count, batter2_avg_30, batter2_sr_30,
                 partnership_quality, current_bowler_economy_30,
                 current_bowler_sr_30, batting_team_avg_score_5,
                 bowling_team_avg_economy_5
  Venue (7):     venue_avg_score, venue_std_score, venue_avg_wickets_25,
                 venue_boundary_rate, venue_high_score_rate,
                 venue_matches_count, batting_at_home
  Intra-loop (6): run_rate_std_5, wicket_rate_10, powerplay_score,
                  run_rate_vs_venue, balls_per_boundary, manhattan_gradient

The feature list FEATURE_COLUMNS_V2 is exported for use by ml_models.py
and evaluation.py.

Usage
-----
from src.pipeline import run_enhanced_pipeline
snapshots, train, val, test = run_enhanced_pipeline('mens_odi')
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_collection import load_processed
from src.data_processing import (
    filter_completed_matches,
    add_match_context,
)
from src.dls_method import DLSModel
from src.elo_tracker import add_elo_to_snapshots
from src.player_features import PlayerFeatureComputer
from src.venue_features import VenueFeatureComputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

# -------------------------------------------------------------------------
# Feature column definitions
# -------------------------------------------------------------------------

# Original 22 features (kept for backwards compatibility)
FEATURE_COLUMNS_V1 = [
    "overs_completed", "overs_remaining", "current_score",
    "wickets_fallen", "wickets_in_hand",
    "current_run_rate", "recent_run_rate_5", "scoring_acceleration",
    "boundary_percentage", "dot_ball_percentage", "partnership_runs",
    "is_powerplay", "is_middle_overs", "is_death_overs",
    "recent_wickets_5", "cumulative_boundaries", "cumulative_sixes",
    "innings_progress", "year", "toss_bat_first",
    "resource_pct_dls", "dls_predicted_final",
]

# Enhanced feature set (46 features).
# Dropped vs V1: overs_remaining (= 50 - overs_completed), wickets_in_hand
# (= 10 - wickets_fallen), innings_progress (= overs_completed/50).
# Perfect collinearity harms SHAP interpretability; tree models are unaffected
# by the removal since overs_completed and wickets_fallen carry identical info.
FEATURE_COLUMNS_V2 = [c for c in FEATURE_COLUMNS_V1
                      if c not in ("overs_remaining", "wickets_in_hand", "innings_progress")] + [
    # ELO
    "batting_team_elo", "bowling_team_elo", "elo_gap",
    # Player
    "batter1_avg_30", "batter1_sr_30", "batter1_boundary_rate_30",
    "batter1_innings_count", "batter2_avg_30", "batter2_sr_30",
    "partnership_quality", "current_bowler_economy_30",
    "current_bowler_sr_30", "batting_team_avg_score_5",
    "bowling_team_avg_economy_5",
    # Venue
    "venue_avg_score", "venue_std_score", "venue_avg_wickets_25",
    "venue_boundary_rate", "venue_high_score_rate",
    "venue_matches_count", "batting_at_home",
    # New intra-loop features
    "run_rate_std_5", "wicket_rate_10", "powerplay_score",
    "run_rate_vs_venue", "balls_per_boundary", "manhattan_gradient",
]

TARGET_COLUMN = "final_total"


# -------------------------------------------------------------------------
# Enhanced snapshot creation (first innings)
# -------------------------------------------------------------------------

def create_enriched_snapshots(
    deliveries_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    overs_limit: int = 50,
) -> pd.DataFrame:
    """
    Create first-innings over-boundary snapshots with all base + new intra-loop features.

    New intra-loop features (require per-over history; must be computed here):
      run_rate_std_5    : std dev of runs-per-over over last 5 overs (volatility)
      wicket_rate_10    : wickets per over over last 10 overs (recent pressure)
      powerplay_score   : cumulative score at end of powerplay (constant per match)
      balls_per_boundary: cumulative legal balls / cumulative boundaries (boundary frequency)
      manhattan_gradient: linear slope of runs-per-over over last 5 overs (trend)

    Note: run_rate_vs_venue is added later (requires venue_avg_score from VenueFeatureComputer).
    """
    first_innings = deliveries_df[deliveries_df["innings"] == 1].copy()

    # First innings totals
    totals = first_innings.groupby("match_id")["total_runs"].sum().reset_index()
    totals.columns = ["match_id", "final_total"]

    snapshots = []
    match_ids = first_innings["match_id"].unique()

    for match_id in match_ids:
        match_dels = first_innings[first_innings["match_id"] == match_id].sort_values(
            ["over", "ball"]
        )
        if match_dels.empty:
            continue

        match_info = matches_df[matches_df["match_id"] == match_id]
        if match_info.empty:
            continue
        match_info = match_info.iloc[0]

        final_total_row = totals[totals["match_id"] == match_id]
        if final_total_row.empty:
            continue
        final_total = final_total_row.iloc[0]["final_total"]

        batting_team = match_dels.iloc[0]["batting_team"]
        actual_max_over = match_dels["over"].max()

        # Accumulators
        cum_runs = cum_wickets = cum_balls = 0
        cum_boundaries = cum_dots = cum_sixes = cum_total_deliveries = 0
        current_partnership = 0
        over_runs_hist = []
        over_wickets_hist = []
        powerplay_score_val = None  # captured once

        for over_num in range(0, actual_max_over + 1):
            over_dels = match_dels[match_dels["over"] == over_num]
            if over_dels.empty:
                over_runs_hist.append(0)
                over_wickets_hist.append(0)
                continue

            over_runs = int(over_dels["total_runs"].sum())
            over_wickets = int(over_dels["is_wicket"].sum())
            over_legal = len(over_dels[~over_dels["is_wide"] & ~over_dels["is_noball"]])
            over_total = len(over_dels)
            over_boundaries = int(over_dels["is_boundary_four"].sum() + over_dels["is_boundary_six"].sum())
            over_dots = len(over_dels[
                (over_dels["total_runs"] == 0)
                & ~over_dels["is_wide"] & ~over_dels["is_noball"]
            ])
            over_sixes = int(over_dels["is_boundary_six"].sum())

            cum_runs += over_runs
            cum_wickets += over_wickets
            cum_balls += over_legal
            cum_boundaries += over_boundaries
            cum_dots += over_dots
            cum_sixes += over_sixes
            cum_total_deliveries += over_total

            # Partnership
            if over_wickets > 0:
                current_partnership = 0
                wicket_idx = over_dels[over_dels["is_wicket"]].index
                if len(wicket_idx) > 0:
                    after = over_dels.loc[wicket_idx[-1]:]
                    current_partnership = int(after["total_runs"].sum())
            else:
                current_partnership += over_runs

            over_runs_hist.append(over_runs)
            over_wickets_hist.append(over_wickets)

            overs_completed = over_num + 1
            overs_remaining = overs_limit - overs_completed

            # Run rates
            crr = cum_runs / overs_completed
            recent_rr5 = np.mean(over_runs_hist[-5:]) if over_runs_hist else 0.0

            if overs_completed > 5:
                old_runs = sum(over_runs_hist[:-5])
                old_rr = old_runs / (overs_completed - 5)
                scoring_acc = crr - old_rr
            else:
                scoring_acc = 0.0

            boundary_pct = cum_boundaries / cum_total_deliveries * 100 if cum_total_deliveries > 0 else 0.0
            dot_pct = cum_dots / cum_balls * 100 if cum_balls > 0 else 0.0
            recent_wickets5 = sum(over_wickets_hist[-5:])

            # Phase indicators
            if overs_limit == 50:
                is_pp = int(overs_completed <= 10)
                is_mid = int(11 <= overs_completed <= 40)
                is_death = int(overs_completed > 40)
            else:
                is_pp = int(overs_completed <= 6)
                is_mid = int(7 <= overs_completed <= 15)
                is_death = int(overs_completed > 15)

            # --- New intra-loop features ---

            # Powerplay score (captured at end of powerplay phase)
            pp_end = 10 if overs_limit == 50 else 6
            if overs_completed == pp_end:
                powerplay_score_val = cum_runs
            pp_score = powerplay_score_val if powerplay_score_val is not None else 0

            # Run rate std over last 5 overs (volatility)
            recent5 = over_runs_hist[-5:]
            run_rate_std_5 = float(np.std(recent5)) if len(recent5) >= 2 else 0.0

            # Wicket rate over last 10 overs
            recent_w10 = over_wickets_hist[-10:]
            wicket_rate_10 = sum(recent_w10) / len(recent_w10) if recent_w10 else 0.0

            # Balls per boundary (lower = more aggressive)
            balls_per_boundary = cum_balls / max(1, cum_boundaries)

            # Manhattan gradient: slope of runs-per-over over last 5 overs
            if len(over_runs_hist) >= 3:
                y = np.array(over_runs_hist[-5:], dtype=float)
                x = np.arange(len(y), dtype=float)
                if len(x) >= 2:
                    slope = float(np.polyfit(x, y, 1)[0])
                else:
                    slope = 0.0
            else:
                slope = 0.0

            snapshot = {
                "match_id": match_id,
                "batting_team": batting_team,
                "overs_completed": overs_completed,
                "overs_remaining": overs_remaining,
                "current_score": cum_runs,
                "wickets_fallen": cum_wickets,
                "wickets_in_hand": 10 - cum_wickets,
                "current_run_rate": round(crr, 4),
                "recent_run_rate_5": round(recent_rr5, 4),
                "scoring_acceleration": round(scoring_acc, 4),
                "boundary_percentage": round(boundary_pct, 4),
                "dot_ball_percentage": round(dot_pct, 4),
                "partnership_runs": current_partnership,
                "is_powerplay": is_pp,
                "is_middle_overs": is_mid,
                "is_death_overs": is_death,
                "recent_wickets_5": recent_wickets5,
                "cumulative_boundaries": cum_boundaries,
                "cumulative_sixes": cum_sixes,
                "innings_progress": round(overs_completed / overs_limit, 4),
                "final_total": final_total,
                # New intra-loop
                "run_rate_std_5": round(run_rate_std_5, 4),
                "wicket_rate_10": round(wicket_rate_10, 4),
                "powerplay_score": pp_score,
                "balls_per_boundary": round(balls_per_boundary, 2),
                "manhattan_gradient": round(slope, 4),
            }
            snapshots.append(snapshot)

    df = pd.DataFrame(snapshots)
    logger.info(f"Created {len(df)} enriched snapshots from {len(match_ids)} matches")
    return df


# -------------------------------------------------------------------------
# Temporal split (60 / 20 / 20: train / calibration / test)
# -------------------------------------------------------------------------

def temporal_three_way_split(
    snapshots_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> tuple:
    """
    Split data temporally at match level into train / calibration / test sets.

    The calibration set is used for conformal prediction coverage guarantees.
    All three splits are strictly non-overlapping in time.

    Returns: (train_df, cal_df, test_df)
    """
    match_dates = (
        matches_df[["match_id", "date"]]
        .drop_duplicates("match_id")
        .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    n = len(match_dates)
    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + cal_frac))

    train_ids = set(match_dates.iloc[:train_end]["match_id"])
    cal_ids = set(match_dates.iloc[train_end:cal_end]["match_id"])
    test_ids = set(match_dates.iloc[cal_end:]["match_id"])

    train_df = snapshots_df[snapshots_df["match_id"].isin(train_ids)].copy()
    cal_df = snapshots_df[snapshots_df["match_id"].isin(cal_ids)].copy()
    test_df = snapshots_df[snapshots_df["match_id"].isin(test_ids)].copy()

    train_cutoff = match_dates.iloc[train_end]["date"].date()
    cal_cutoff = match_dates.iloc[cal_end]["date"].date()

    logger.info(
        f"Temporal split: train {len(train_ids)} matches ({len(train_df)} rows, "
        f"up to {train_cutoff}) | "
        f"cal {len(cal_ids)} matches ({len(cal_df)} rows, "
        f"up to {cal_cutoff}) | "
        f"test {len(test_ids)} matches ({len(test_df)} rows)"
    )
    return train_df, cal_df, test_df


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------

def run_enhanced_pipeline(
    format_key: str,
    overs_limit: int = 50,
    force_recompute: bool = False,
) -> tuple:
    """
    Full enhanced feature pipeline for one format.

    Parameters
    ----------
    format_key     : e.g. 'mens_odi'
    overs_limit    : 50 for ODI, 20 for T20
    force_recompute: if False, load from cache if available

    Returns
    -------
    (snapshots_df, train_df, cal_df, test_df)
    """
    snaps_path = PROCESSED_DIR / f"{format_key}_snapshots_v2.parquet"
    train_path = PROCESSED_DIR / f"{format_key}_train_v2.parquet"
    cal_path = PROCESSED_DIR / f"{format_key}_cal_v2.parquet"
    test_path = PROCESSED_DIR / f"{format_key}_test_v2.parquet"

    if not force_recompute and snaps_path.exists():
        logger.info(f"[{format_key}] Loading cached v2 snapshots from disk.")
        snapshots = pd.read_parquet(snaps_path)
        train = pd.read_parquet(train_path)
        cal = pd.read_parquet(cal_path)
        test = pd.read_parquet(test_path)
        return snapshots, train, cal, test

    logger.info(f"[{format_key}] Starting enhanced pipeline...")

    # 1. Load raw data
    matches_raw, deliveries_raw = load_processed(format_key)

    # 2. Filter to non-DL completed matches (for training data)
    matches_clean, deliveries_clean = filter_completed_matches(matches_raw, deliveries_raw)

    # 3. Create enriched first-innings snapshots
    snapshots = create_enriched_snapshots(deliveries_clean, matches_clean, overs_limit=overs_limit)

    # 4. Add match-level context (date, venue, city, toss)
    snapshots = add_match_context(snapshots, matches_clean)

    # 5. ELO ratings (fit on ALL matches including DL, for maximum history)
    logger.info(f"[{format_key}] Computing ELO ratings...")
    snapshots = add_elo_to_snapshots(snapshots, matches_raw, format_key=format_key)

    # 6. Player rolling features
    logger.info(f"[{format_key}] Computing player features (this may take 1-2 min)...")
    pfc = PlayerFeatureComputer()
    pfc.fit(matches_raw, deliveries_raw)
    snapshots = pfc.transform(snapshots, deliveries_clean)

    # Save PlayerFeatureComputer for inference
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / f"{format_key}_player_feature_computer.pkl", "wb") as f:
        pickle.dump(pfc, f)

    # 7. Venue features
    logger.info(f"[{format_key}] Computing venue features...")
    vfc = VenueFeatureComputer(overs_limit=overs_limit)
    vfc.fit(matches_clean, snapshots)
    snapshots = vfc.transform(snapshots, matches_clean)

    # Save VenueFeatureComputer for inference
    with open(MODELS_DIR / f"{format_key}_venue_feature_computer.pkl", "wb") as f:
        pickle.dump(vfc, f)

    # 8. Derive run_rate_vs_venue (requires venue_avg_score from step 7)
    venue_avg_rr = snapshots["venue_avg_score"] / overs_limit
    venue_avg_rr = venue_avg_rr.replace(0, np.nan).fillna(snapshots["current_run_rate"].median())
    snapshots["run_rate_vs_venue"] = (snapshots["current_run_rate"] / venue_avg_rr).round(4)

    # 9. DLS features (load fitted DLS model; fit if not available)
    dls_path = MODELS_DIR / f"dls_model_{overs_limit}.pkl"
    dls = DLSModel(overs_limit=overs_limit)
    if dls_path.exists():
        dls.load(str(dls_path))
    else:
        logger.info(f"[{format_key}] Fitting DLS model...")
        dls.fit(snapshots)
        dls.save(str(dls_path))

    from src.feature_engineering import add_dls_features
    snapshots = add_dls_features(snapshots, dls)

    # 10. Final type cleanup and NaN handling
    for col in FEATURE_COLUMNS_V2:
        if col in snapshots.columns:
            snapshots[col] = pd.to_numeric(snapshots[col], errors="coerce").fillna(0)

    # 11. Temporal 60/20/20 split
    train, cal, test = temporal_three_way_split(snapshots, matches_clean)

    # 12. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(snaps_path, index=False)
    train.to_parquet(train_path, index=False)
    cal.to_parquet(cal_path, index=False)
    test.to_parquet(test_path, index=False)

    logger.info(
        f"[{format_key}] Pipeline complete. "
        f"Snapshots: {len(snapshots)} rows × {len(snapshots.columns)} cols. "
        f"Features: {len(FEATURE_COLUMNS_V2)}"
    )
    return snapshots, train, cal, test
