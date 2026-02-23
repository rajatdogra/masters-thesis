"""
Second Innings Pipeline (Phase 2).

Scientific Note on Censoring
------------------------------
In ODI cricket, ~51.5% of second innings are right-censored: the chasing team
reaches the target before using all allocated overs.  We observe the score at
the winning delivery (a lower bound on the team's potential) but not their
"full-innings" total.  This is classical right-censoring from survival analysis.

Treatment in this module
  is_censored = 1  : innings where chasing team won (right-censored)
  is_censored = 0  : innings that ended by all-wickets or overs exhausted

Training strategy (documented in thesis)
  Primary   – uncensored innings only (teams that lost or used all overs)
  Secondary – all data with is_censored as a covariate (selection-bias caveat)
  Evaluation – RMSE / R² restricted to uncensored test innings only

This mirrors the Tobit regression partition:
  Type I  = censored observations (observed score < potential score)
  Type II = uncensored observations (actual final score is directly observed)

Chase Context Features  (new in Phase 2)
------------------------------------------
  target_score             first_innings_total + 1  (runs needed to win)
  required_runs            target_score − current_score
  required_run_rate        required_runs / overs_remaining  (capped at MAX_RRR)
  pressure_index           required_run_rate / current_run_rate  (capped at MAX_PRESSURE)
  overs_remaining_inn2     overs_limit − overs_completed
  resource_pct_remaining_inn2  DLS resource % still available to chasing team
  runs_above_dls_par       current_score − DLS par at current (overs, wickets) state
  first_inn_powerplay      first innings runs scored in powerplay (overs 1-10)
  first_inn_final_rr       first innings final run rate (total / overs_limit)
  chasing_chose_chase      1 if chasing team won toss and chose to field

Revised Target Engine
------------------------
RevisedTargetEngine evaluates the ML second-innings model on 255 historical
DL matches.  For each interrupted match it reconstructs the delivery-level
feature vector at the interruption over and predicts:
  (a) what final score the chasing team would have reached
  (b) a revised target = predicted_score  (rounded up; must exceed current score)

The ML revised target is compared with the official DLS target and actual
outcome to quantify fairness, accuracy, and distributional bias.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_collection import load_processed
from src.data_processing import filter_completed_matches, add_match_context
from src.dls_method import DLSModel
from src.elo_tracker import add_elo_to_snapshots
from src.player_features import PlayerFeatureComputer
from src.venue_features import VenueFeatureComputer
from src.pipeline import temporal_three_way_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

# Hard caps for derived chase features
MAX_RRR = 36.0       # 6 runs/ball × 6 balls/over (theoretical maximum)
MAX_PRESSURE = 10.0  # beyond this the match is practically unwinnable


# -------------------------------------------------------------------------
# Feature column definitions
# -------------------------------------------------------------------------

FEATURE_COLUMNS_INN2 = [
    # --- Match state (second innings) ---
    "overs_completed",
    "current_score",
    "wickets_fallen",
    "current_run_rate",
    "recent_run_rate_5",
    "scoring_acceleration",
    "boundary_percentage",
    "dot_ball_percentage",
    "partnership_runs",
    "is_powerplay",
    "is_middle_overs",
    "is_death_overs",
    "recent_wickets_5",
    "cumulative_boundaries",
    "cumulative_sixes",
    "year",
    "chasing_chose_chase",
    # --- Chase context (unique to second innings) ---
    "target_score",
    "required_runs",
    "required_run_rate",
    "pressure_index",
    "overs_remaining_inn2",
    "resource_pct_remaining_inn2",
    "runs_above_dls_par",
    "first_inn_powerplay",
    "first_inn_final_rr",
    # --- ELO ---
    "batting_team_elo",
    "bowling_team_elo",
    "elo_gap",
    # --- Player features ---
    "batter1_avg_30",
    "batter1_sr_30",
    "batter1_boundary_rate_30",
    "batter1_innings_count",
    "batter2_avg_30",
    "batter2_sr_30",
    "partnership_quality",
    "current_bowler_economy_30",
    "current_bowler_sr_30",
    "batting_team_avg_score_5",
    "bowling_team_avg_economy_5",
    # --- Venue features (based on first-innings history at venue) ---
    "venue_avg_score",
    "venue_std_score",
    "venue_avg_wickets_25",
    "venue_boundary_rate",
    "venue_high_score_rate",
    "venue_matches_count",
    "batting_at_home",
    # --- Derived intra-loop features ---
    "run_rate_std_5",
    "wicket_rate_10",
    "powerplay_score",
    "run_rate_vs_venue",
    "balls_per_boundary",
    "manhattan_gradient",
]

TARGET_COLUMN_INN2 = "second_innings_final_total"


# -------------------------------------------------------------------------
# Core snapshot function
# -------------------------------------------------------------------------

def create_second_innings_snapshots(
    deliveries_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    dls_model: DLSModel,
    overs_limit: int = 50,
) -> pd.DataFrame:
    """
    Create second-innings over-boundary snapshots with chase context features.

    Each snapshot captures the cumulative match state at the end of each over
    in the second innings, enriched with:
      - Chase context: target, required runs/rate, pressure index, DLS resource %
      - First innings summary: powerplay score, final run rate
      - Intra-over derived features mirroring the first innings module
      - Censoring flag: 1 if the batting team had already reached the target
        or wins at this over (right-censored observation)

    Parameters
    ----------
    deliveries_df : all ball-by-ball deliveries (both innings)
    matches_df    : match metadata (winner, toss, venue, date)
    dls_model     : fitted DLSModel (used for resource % and par score)
    overs_limit   : 50 for ODI, 20 for T20

    Returns
    -------
    DataFrame: one row per (match_id, overs_completed) in second innings.
    Columns include all chase features, match state, and metadata columns
    (second_innings_final_total, is_censored, first_innings_total, batting_team).
    """
    second_inn = deliveries_df[deliveries_df["innings"] == 2].copy()
    first_inn  = deliveries_df[deliveries_df["innings"] == 1].copy()

    # --- First innings context aggregates ---
    fi_totals = (
        first_inn.groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "first_innings_total"})
    )

    pp_end_over_0idx = (9 if overs_limit == 50 else 5)  # 0-indexed max over in powerplay
    fi_pp = (
        first_inn[first_inn["over"] <= pp_end_over_0idx]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "first_inn_powerplay"})
    )

    fi_ctx = fi_totals.merge(fi_pp, on="match_id", how="left")
    fi_ctx["first_inn_powerplay"] = fi_ctx["first_inn_powerplay"].fillna(0)
    fi_ctx["first_inn_final_rr"] = (
        fi_ctx["first_innings_total"] / overs_limit
    ).round(4)

    # --- Second innings final totals ---
    si_totals = (
        second_inn.groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "second_innings_final_total"})
    )

    # --- Match outcome for censoring detection ---
    outcome_cols = [c for c in ["match_id", "winner", "team1", "team2"] if c in matches_df.columns]
    match_outcomes = matches_df[outcome_cols].drop_duplicates("match_id")

    # Powerplay end (1-indexed overs_completed)
    pp_end_1idx = 10 if overs_limit == 50 else 6

    snapshots = []
    match_ids = second_inn["match_id"].unique()
    skipped = 0

    for match_id in match_ids:
        match_dels = second_inn[
            second_inn["match_id"] == match_id
        ].sort_values(["over", "ball"])

        if match_dels.empty:
            skipped += 1
            continue

        # First innings context
        fi_rows = fi_ctx[fi_ctx["match_id"] == match_id]
        if fi_rows.empty:
            skipped += 1
            continue
        fi = fi_rows.iloc[0]
        first_innings_total = float(fi["first_innings_total"])
        first_inn_pp_val    = float(fi["first_inn_powerplay"])
        first_inn_rr_val    = float(fi["first_inn_final_rr"])

        # Second innings final total (the label, possibly censored)
        si_rows = si_totals[si_totals["match_id"] == match_id]
        if si_rows.empty:
            skipped += 1
            continue
        si_total = float(si_rows.iloc[0]["second_innings_final_total"])

        # Target (first innings total + 1 run to win)
        target_score = int(first_innings_total) + 1

        # Batting team in second innings
        batting_team = str(match_dels.iloc[0]["batting_team"])

        # Censoring: chasing team won if winner == batting_team
        outcome_rows = match_outcomes[match_outcomes["match_id"] == match_id]
        winner = outcome_rows.iloc[0]["winner"] if not outcome_rows.empty else None
        is_censored_match = int(
            (winner is not None) and (not pd.isna(winner)) and (str(winner) == batting_team)
        )

        actual_max_over = match_dels["over"].max()

        # ---- Per-over accumulators ----
        cum_runs = cum_wickets = cum_balls = 0
        cum_boundaries = cum_dots = cum_sixes = cum_total_deliveries = 0
        current_partnership = 0
        over_runs_hist    = []
        over_wickets_hist = []
        pp_score_val      = None  # second innings powerplay total (captured once)

        for over_num in range(0, actual_max_over + 1):
            over_dels = match_dels[match_dels["over"] == over_num]
            if over_dels.empty:
                over_runs_hist.append(0)
                over_wickets_hist.append(0)
                continue

            over_runs     = int(over_dels["total_runs"].sum())
            over_wickets  = int(over_dels["is_wicket"].sum())
            over_legal    = int(len(over_dels[~over_dels["is_wide"] & ~over_dels["is_noball"]]))
            over_total    = int(len(over_dels))
            over_bounds   = int(
                over_dels["is_boundary_four"].sum() + over_dels["is_boundary_six"].sum()
            )
            over_dots = int(len(over_dels[
                (over_dels["total_runs"] == 0)
                & ~over_dels["is_wide"] & ~over_dels["is_noball"]
            ]))
            over_sixes = int(over_dels["is_boundary_six"].sum())

            cum_runs              += over_runs
            cum_wickets           += over_wickets
            cum_balls             += over_legal
            cum_boundaries        += over_bounds
            cum_dots              += over_dots
            cum_sixes             += over_sixes
            cum_total_deliveries  += over_total

            # Partnership tracking (reset on last wicket in over)
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

            overs_completed     = over_num + 1
            overs_remaining_inn2 = overs_limit - overs_completed

            # ---- Run rates ----
            crr = cum_runs / overs_completed
            recent_rr5 = float(np.mean(over_runs_hist[-5:])) if over_runs_hist else 0.0

            if overs_completed > 5:
                old_runs  = sum(over_runs_hist[:-5])
                old_rr    = old_runs / (overs_completed - 5)
                scoring_acc = crr - old_rr
            else:
                scoring_acc = 0.0

            boundary_pct = (
                cum_boundaries / cum_total_deliveries * 100
                if cum_total_deliveries > 0 else 0.0
            )
            dot_pct = cum_dots / cum_balls * 100 if cum_balls > 0 else 0.0
            recent_wickets5 = sum(over_wickets_hist[-5:])

            # ---- Phase indicators ----
            if overs_limit == 50:
                is_pp   = int(overs_completed <= 10)
                is_mid  = int(11 <= overs_completed <= 40)
                is_death = int(overs_completed > 40)
            else:
                is_pp   = int(overs_completed <= 6)
                is_mid  = int(7 <= overs_completed <= 15)
                is_death = int(overs_completed > 15)

            # ---- Powerplay score (second innings) ----
            if overs_completed == pp_end_1idx:
                pp_score_val = cum_runs
            pp_score = pp_score_val if pp_score_val is not None else 0

            # ---- Run rate volatility and momentum ----
            recent5     = over_runs_hist[-5:]
            rr_std_5    = float(np.std(recent5)) if len(recent5) >= 2 else 0.0
            recent_w10  = over_wickets_hist[-10:]
            wicket_r10  = sum(recent_w10) / len(recent_w10) if recent_w10 else 0.0
            balls_per_b = cum_balls / max(1, cum_boundaries)

            if len(over_runs_hist) >= 3:
                y    = np.array(over_runs_hist[-5:], dtype=float)
                x    = np.arange(len(y), dtype=float)
                slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0
            else:
                slope = 0.0

            # ---- Chase context ----
            required_runs = max(0, target_score - cum_runs)

            if overs_remaining_inn2 > 0:
                rrr = min(required_runs / overs_remaining_inn2, MAX_RRR)
            else:
                rrr = MAX_RRR if required_runs > 0 else 0.0

            if crr > 0:
                pressure = min(rrr / crr, MAX_PRESSURE)
            else:
                pressure = MAX_PRESSURE if required_runs > 0 else 0.0

            # DLS resource remaining for chasing team at this state
            resource_rem = dls_model.resource_remaining(overs_remaining_inn2, cum_wickets)

            # DLS par score: fraction of first innings total equivalent to resources used
            resource_full = dls_model.resource_remaining(overs_limit, 0)  # ≈ 100%
            if resource_full > 0:
                prop_used = (resource_full - resource_rem) / resource_full
                dls_par   = round(first_innings_total * prop_used, 2)
            else:
                dls_par   = first_innings_total / 2.0

            runs_above_par = round(cum_runs - dls_par, 2)

            snapshots.append({
                # Identifiers / metadata
                "match_id":                  match_id,
                "batting_team":              batting_team,
                "innings":                   2,
                # Match state
                "overs_completed":           overs_completed,
                "overs_remaining_inn2":      overs_remaining_inn2,
                "current_score":             cum_runs,
                "wickets_fallen":            cum_wickets,
                "current_run_rate":          round(crr, 4),
                "recent_run_rate_5":         round(recent_rr5, 4),
                "scoring_acceleration":      round(scoring_acc, 4),
                "boundary_percentage":       round(boundary_pct, 4),
                "dot_ball_percentage":       round(dot_pct, 4),
                "partnership_runs":          current_partnership,
                "is_powerplay":              is_pp,
                "is_middle_overs":           is_mid,
                "is_death_overs":            is_death,
                "recent_wickets_5":          recent_wickets5,
                "cumulative_boundaries":     cum_boundaries,
                "cumulative_sixes":          cum_sixes,
                # Chase context
                "target_score":              target_score,
                "required_runs":             required_runs,
                "required_run_rate":         round(rrr, 4),
                "pressure_index":            round(pressure, 4),
                "resource_pct_remaining_inn2": round(resource_rem, 2),
                "runs_above_dls_par":        runs_above_par,
                "first_inn_powerplay":       first_inn_pp_val,
                "first_inn_final_rr":        first_inn_rr_val,
                # Intra-loop derived
                "run_rate_std_5":            round(rr_std_5, 4),
                "wicket_rate_10":            round(wicket_r10, 4),
                "powerplay_score":           pp_score,
                "balls_per_boundary":        round(balls_per_b, 2),
                "manhattan_gradient":        round(slope, 4),
                # Label and censoring
                "second_innings_final_total": si_total,
                "is_censored":               is_censored_match,
                "first_innings_total":        first_innings_total,
            })

    df = pd.DataFrame(snapshots)
    logger.info(
        f"Second innings: {len(df)} snapshots from {len(match_ids)-skipped} matches "
        f"({skipped} skipped). "
        f"Censored: {df['is_censored'].sum()} ({100*df['is_censored'].mean():.1f}%)"
    )
    return df


# -------------------------------------------------------------------------
# Helper: add chasing_chose_chase feature
# -------------------------------------------------------------------------

def _add_chasing_choice(snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add chasing_chose_chase = 1 if the batting team won the toss and
    elected to field first (i.e. chose to chase).
    Requires add_match_context() to have already been called.
    """
    df = snapshots_df.copy()
    if "toss_winner" not in df.columns or "toss_decision" not in df.columns:
        df["chasing_chose_chase"] = 0
        return df
    won_toss     = (df["toss_winner"] == df["batting_team"]).astype(int)
    chose_field  = (df["toss_decision"] == "field").astype(int)
    df["chasing_chose_chase"] = won_toss * chose_field
    return df


# -------------------------------------------------------------------------
# Main second-innings pipeline
# -------------------------------------------------------------------------

def run_second_innings_pipeline(
    format_key: str,
    overs_limit: int = 50,
    force_recompute: bool = False,
) -> tuple:
    """
    Full second-innings feature pipeline for one format.

    Reuses the already-fitted PlayerFeatureComputer, VenueFeatureComputer,
    and DLS model from the first-innings pipeline (saved as .pkl / .pkl files).
    If any artefact is missing, it is refitted here.

    Parameters
    ----------
    format_key     : e.g. 'mens_odi'
    overs_limit    : 50 for ODI, 20 for T20
    force_recompute: bypass disk cache

    Returns
    -------
    (snapshots_inn2, train_inn2, cal_inn2, test_inn2)

    Additional saved files
    ----------------------
    data/processed/{format_key}_snapshots_inn2.parquet
    data/processed/{format_key}_train_inn2.parquet
    data/processed/{format_key}_cal_inn2.parquet
    data/processed/{format_key}_test_inn2.parquet

    Censoring stats are logged for each split.
    """
    snaps_path = PROCESSED_DIR / f"{format_key}_snapshots_inn2.parquet"
    train_path = PROCESSED_DIR / f"{format_key}_train_inn2.parquet"
    cal_path   = PROCESSED_DIR / f"{format_key}_cal_inn2.parquet"
    test_path  = PROCESSED_DIR / f"{format_key}_test_inn2.parquet"

    if not force_recompute and snaps_path.exists():
        logger.info(f"[{format_key}] Loading cached second-innings snapshots from disk.")
        snapshots = pd.read_parquet(snaps_path)
        train     = pd.read_parquet(train_path)
        cal       = pd.read_parquet(cal_path)
        test      = pd.read_parquet(test_path)
        return snapshots, train, cal, test

    logger.info(f"[{format_key}] Starting second-innings pipeline...")

    # 1. Load raw data
    matches_raw, deliveries_raw = load_processed(format_key)

    # 2. Filter to non-DL completed matches (training data)
    matches_clean, deliveries_clean = filter_completed_matches(
        matches_raw, deliveries_raw
    )

    # 3. Load / fit DLS model (reuse from first-innings pipeline)
    dls_path = MODELS_DIR / f"dls_model_{overs_limit}.pkl"
    dls = DLSModel(overs_limit=overs_limit)
    if dls_path.exists():
        dls.load(str(dls_path))
        logger.info(f"[{format_key}] Loaded existing DLS model.")
    else:
        snaps_v2_path = PROCESSED_DIR / f"{format_key}_snapshots_v2.parquet"
        if snaps_v2_path.exists():
            logger.info(f"[{format_key}] Fitting DLS model on first-innings snapshots...")
            snaps_v2 = pd.read_parquet(snaps_v2_path)
            dls.fit(snaps_v2)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            dls.save(str(dls_path))
        else:
            logger.warning(
                f"[{format_key}] No DLS model or snapshots_v2 found; "
                "using default DLS parameters."
            )

    # 4. Create second-innings snapshots (all chase context computed here)
    logger.info(f"[{format_key}] Creating second-innings snapshots...")
    snapshots = create_second_innings_snapshots(
        deliveries_clean, matches_clean, dls, overs_limit=overs_limit
    )

    # 5. Add match context (date, year, venue, city, toss)
    snapshots = add_match_context(snapshots, matches_clean)

    # 6. Compute chasing_chose_chase (requires toss_winner / toss_decision from step 5)
    snapshots = _add_chasing_choice(snapshots)

    # 7. ELO ratings (fit on ALL matches for maximum history)
    logger.info(f"[{format_key}] Adding ELO features...")
    snapshots = add_elo_to_snapshots(snapshots, matches_raw, format_key=format_key)

    # 8. Player rolling features  ── reuse fitted computer from first-innings run
    pfc_path = MODELS_DIR / f"{format_key}_player_feature_computer.pkl"
    if pfc_path.exists():
        logger.info(f"[{format_key}] Loading fitted PlayerFeatureComputer...")
        with open(pfc_path, "rb") as f:
            pfc = pickle.load(f)
    else:
        logger.info(f"[{format_key}] Fitting PlayerFeatureComputer (no cache found)...")
        pfc = PlayerFeatureComputer()
        pfc.fit(matches_raw, deliveries_raw)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(pfc_path, "wb") as f:
            pickle.dump(pfc, f)

    # Pass innings_num=2 so the player lookup uses second-innings deliveries
    snapshots = pfc.transform(snapshots, deliveries_clean, innings_num=2)

    # 9. Venue features  ── reuse computer fitted on first-innings data
    vfc_path = MODELS_DIR / f"{format_key}_venue_feature_computer.pkl"
    if vfc_path.exists():
        logger.info(f"[{format_key}] Loading fitted VenueFeatureComputer...")
        with open(vfc_path, "rb") as f:
            vfc = pickle.load(f)
    else:
        logger.info(f"[{format_key}] VenueFeatureComputer not found; fitting on first-innings data...")
        snaps_v2_path = PROCESSED_DIR / f"{format_key}_snapshots_v2.parquet"
        if snaps_v2_path.exists():
            snaps_v2 = pd.read_parquet(snaps_v2_path)
            vfc = VenueFeatureComputer(overs_limit=overs_limit)
            vfc.fit(matches_clean, snaps_v2)
        else:
            # Fallback: fit a fresh VFC on second-innings snapshots (less ideal)
            logger.warning(
                "Fitting VenueFeatureComputer on second-innings data (first-innings "
                "snapshots_v2 not found). Venue averages will reflect second-innings scores."
            )
            vfc = VenueFeatureComputer(overs_limit=overs_limit)
            vfc.fit(matches_clean, snapshots)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(vfc_path, "wb") as f:
            pickle.dump(vfc, f)

    snapshots = vfc.transform(snapshots, matches_clean)

    # 10. run_rate_vs_venue (current run rate as fraction of venue's typical pace)
    venue_avg_rr = snapshots["venue_avg_score"] / overs_limit
    venue_avg_rr = venue_avg_rr.replace(0, np.nan).fillna(
        snapshots["current_run_rate"].median()
    )
    snapshots["run_rate_vs_venue"] = (
        snapshots["current_run_rate"] / venue_avg_rr
    ).round(4)

    # 11. Final NaN cleanup for all feature columns
    for col in FEATURE_COLUMNS_INN2:
        if col in snapshots.columns:
            snapshots[col] = pd.to_numeric(snapshots[col], errors="coerce").fillna(0)

    # 12. Temporal 60/20/20 split  ── same match-level cutoffs as first-innings pipeline
    train, cal, test = temporal_three_way_split(snapshots, matches_clean)

    # Report censoring statistics per split
    for split_name, split_df in [("train", train), ("cal", cal), ("test", test)]:
        n_total    = len(split_df)
        n_censored = int(split_df["is_censored"].sum())
        logger.info(
            f"  [{format_key}] {split_name}: {n_total} rows, "
            f"censored {n_censored} ({100*n_censored/max(n_total,1):.1f}%)"
        )

    # 13. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(snaps_path, index=False)
    train.to_parquet(train_path, index=False)
    cal.to_parquet(cal_path, index=False)
    test.to_parquet(test_path, index=False)

    logger.info(
        f"[{format_key}] Second-innings pipeline complete. "
        f"Snapshots: {len(snapshots)} rows × {len(snapshots.columns)} cols. "
        f"Features: {len(FEATURE_COLUMNS_INN2)}"
    )
    return snapshots, train, cal, test


# -------------------------------------------------------------------------
# Revised Target Engine
# -------------------------------------------------------------------------

class RevisedTargetEngine:
    """
    Evaluate ML-based revised targets against official DLS targets on
    historical rain-affected (D/L) matches.

    The engine reconstructs match states at the interruption point for each
    DL match and predicts what the chasing team would have scored,
    yielding an ML revised target.

    Parameters
    ----------
    dls_model  : fitted DLSModel  (for resource-based comparisons)
    pfc        : fitted PlayerFeatureComputer  (loaded from .pkl)
    vfc        : fitted VenueFeatureComputer   (loaded from .pkl)
    overs_limit: innings overs limit (50 for ODI)
    """

    def __init__(
        self,
        dls_model: DLSModel,
        pfc: PlayerFeatureComputer,
        vfc: VenueFeatureComputer,
        overs_limit: int = 50,
    ):
        self.dls = dls_model
        self.pfc = pfc
        self.vfc = vfc
        self.overs_limit = overs_limit

    # ------------------------------------------------------------------
    # Snapshot reconstruction for DL matches
    # ------------------------------------------------------------------

    def _build_dl_match_snapshots(
        self,
        dl_df: pd.DataFrame,
        deliveries_raw: pd.DataFrame,
        matches_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build second-innings feature snapshots for each DL match at the
        over immediately before rain interrupted the match.

        For each DL match we compute the cumulative state up to
        inn2_overs_bowled and add chase context with dl_target_overs as
        the allocated overs (rather than the full 50).

        Parameters
        ----------
        dl_df          : DataFrame from parse_dl_targets() with 255 rows
        deliveries_raw : all raw deliveries (including DL match deliveries)
        matches_raw    : all raw match metadata

        Returns
        -------
        DataFrame with one row per DL match representing the interruption state.
        Columns include all FEATURE_COLUMNS_INN2 entries plus metadata.
        """
        # Filter deliveries to DL match IDs
        dl_ids       = set(dl_df["match_id"].astype(str))
        dl_dels      = deliveries_raw[
            deliveries_raw["match_id"].astype(str).isin(dl_ids)
        ].copy()
        dl_matches   = matches_raw[
            matches_raw["match_id"].astype(str).isin(dl_ids)
        ].copy()
        dl_dels["match_id"] = dl_dels["match_id"].astype(str)
        dl_matches["match_id"] = dl_matches["match_id"].astype(str)
        dl_df = dl_df.copy()
        dl_df["match_id"] = dl_df["match_id"].astype(str)

        rows = []
        for _, dl_row in dl_df.iterrows():
            mid = str(dl_row["match_id"])
            interruption_over = int(dl_row["inn2_overs_bowled"])  # overs completed
            allocated_overs   = int(dl_row["dl_target_overs"]) if pd.notna(
                dl_row["dl_target_overs"]
            ) else self.overs_limit
            first_innings_total = float(dl_row["first_innings_total"])
            target_score        = int(first_innings_total) + 1

            # Second innings deliveries for this match
            match_dels = (
                dl_dels[
                    (dl_dels["match_id"] == mid) & (dl_dels["innings"] == 2)
                ]
                .sort_values(["over", "ball"])
            )
            if match_dels.empty or interruption_over == 0:
                continue

            # Reuse create_second_innings_snapshots logic for a single match
            # up to interruption_over
            cum_runs = cum_wickets = cum_balls = 0
            cum_boundaries = cum_dots = cum_sixes = cum_total_deliveries = 0
            current_partnership = 0
            over_runs_hist    = []
            over_wickets_hist = []
            pp_score_val      = None
            pp_end_1idx       = 10 if self.overs_limit == 50 else 6
            batting_team      = str(match_dels.iloc[0]["batting_team"])

            for over_num in range(0, min(interruption_over, match_dels["over"].max() + 1)):
                over_dels = match_dels[match_dels["over"] == over_num]
                if over_dels.empty:
                    over_runs_hist.append(0)
                    over_wickets_hist.append(0)
                    continue

                over_runs    = int(over_dels["total_runs"].sum())
                over_wickets = int(over_dels["is_wicket"].sum())
                over_legal   = int(len(over_dels[~over_dels["is_wide"] & ~over_dels["is_noball"]]))
                over_total   = int(len(over_dels))
                over_bounds  = int(
                    over_dels["is_boundary_four"].sum() + over_dels["is_boundary_six"].sum()
                )
                over_dots = int(len(over_dels[
                    (over_dels["total_runs"] == 0)
                    & ~over_dels["is_wide"] & ~over_dels["is_noball"]
                ]))
                over_sixes = int(over_dels["is_boundary_six"].sum())

                cum_runs             += over_runs
                cum_wickets          += over_wickets
                cum_balls            += over_legal
                cum_boundaries       += over_bounds
                cum_dots             += over_dots
                cum_sixes            += over_sixes
                cum_total_deliveries += over_total

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

                if (over_num + 1) == pp_end_1idx:
                    pp_score_val = cum_runs

            overs_completed = interruption_over
            # Key difference from training: overs remaining = allocated_overs - overs_completed
            overs_remaining_inn2 = max(0, allocated_overs - overs_completed)

            crr = cum_runs / max(overs_completed, 1)
            recent_rr5 = float(np.mean(over_runs_hist[-5:])) if over_runs_hist else 0.0
            if overs_completed > 5:
                old_runs   = sum(over_runs_hist[:-5])
                scoring_acc = crr - (old_runs / (overs_completed - 5))
            else:
                scoring_acc = 0.0

            boundary_pct = (
                cum_boundaries / cum_total_deliveries * 100
                if cum_total_deliveries > 0 else 0.0
            )
            dot_pct = cum_dots / cum_balls * 100 if cum_balls > 0 else 0.0
            recent_wickets5 = sum(over_wickets_hist[-5:])

            if self.overs_limit == 50:
                is_pp   = int(overs_completed <= 10)
                is_mid  = int(11 <= overs_completed <= 40)
                is_death = int(overs_completed > 40)
            else:
                is_pp   = int(overs_completed <= 6)
                is_mid  = int(7 <= overs_completed <= 15)
                is_death = int(overs_completed > 15)

            pp_score = pp_score_val if pp_score_val is not None else 0
            recent5  = over_runs_hist[-5:]
            rr_std_5 = float(np.std(recent5)) if len(recent5) >= 2 else 0.0
            recent_w10 = over_wickets_hist[-10:]
            wicket_r10 = sum(recent_w10) / len(recent_w10) if recent_w10 else 0.0
            balls_per_b = cum_balls / max(1, cum_boundaries)

            if len(over_runs_hist) >= 3:
                y    = np.array(over_runs_hist[-5:], dtype=float)
                x    = np.arange(len(y), dtype=float)
                slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0
            else:
                slope = 0.0

            required_runs = max(0, target_score - cum_runs)
            rrr = (
                min(required_runs / overs_remaining_inn2, MAX_RRR)
                if overs_remaining_inn2 > 0
                else (MAX_RRR if required_runs > 0 else 0.0)
            )
            pressure = (
                min(rrr / crr, MAX_PRESSURE) if crr > 0
                else (MAX_PRESSURE if required_runs > 0 else 0.0)
            )

            resource_rem  = self.dls.resource_remaining(overs_remaining_inn2, cum_wickets)
            resource_full = self.dls.resource_remaining(self.overs_limit, 0)
            if resource_full > 0:
                prop_used = (resource_full - resource_rem) / resource_full
                dls_par   = round(first_innings_total * prop_used, 2)
            else:
                dls_par   = first_innings_total / 2.0
            runs_above_par = round(cum_runs - dls_par, 2)

            # First innings powerplay and final RR from DL row (or compute from deliveries)
            fi_pp_val  = float(
                dl_dels[
                    (dl_dels["match_id"] == mid)
                    & (dl_dels["innings"] == 1)
                    & (dl_dels["over"] <= (9 if self.overs_limit == 50 else 5))
                ]["total_runs"].sum()
            )
            fi_rr_val  = round(first_innings_total / self.overs_limit, 4)

            rows.append({
                "match_id":                  mid,
                "batting_team":              batting_team,
                "innings":                   2,
                "overs_completed":           overs_completed,
                "overs_remaining_inn2":      overs_remaining_inn2,
                "current_score":             cum_runs,
                "wickets_fallen":            cum_wickets,
                "current_run_rate":          round(crr, 4),
                "recent_run_rate_5":         round(recent_rr5, 4),
                "scoring_acceleration":      round(scoring_acc, 4),
                "boundary_percentage":       round(boundary_pct, 4),
                "dot_ball_percentage":       round(dot_pct, 4),
                "partnership_runs":          current_partnership,
                "is_powerplay":              is_pp,
                "is_middle_overs":           is_mid,
                "is_death_overs":            is_death,
                "recent_wickets_5":          recent_wickets5,
                "cumulative_boundaries":     cum_boundaries,
                "cumulative_sixes":          cum_sixes,
                "target_score":              target_score,
                "required_runs":             required_runs,
                "required_run_rate":         round(rrr, 4),
                "pressure_index":            round(pressure, 4),
                "resource_pct_remaining_inn2": round(resource_rem, 2),
                "runs_above_dls_par":        runs_above_par,
                "first_inn_powerplay":       fi_pp_val,
                "first_inn_final_rr":        fi_rr_val,
                "run_rate_std_5":            round(rr_std_5, 4),
                "wicket_rate_10":            round(wicket_r10, 4),
                "powerplay_score":           pp_score,
                "balls_per_boundary":        round(balls_per_b, 2),
                "manhattan_gradient":        round(slope, 4),
                # DL metadata
                "first_innings_total":       first_innings_total,
                "dl_target_runs":            dl_row["dl_target_runs"],
                "dl_target_overs":           allocated_overs,
                "inn2_score_at_end":         dl_row["inn2_score_at_end"],
                "inn2_overs_bowled":         interruption_over,
                "inn2_wickets_at_end":       dl_row["inn2_wickets_at_end"],
                "winner":                    dl_row.get("winner", None),
                "date":                      dl_row.get("date", None),
                "venue":                     dl_row.get("venue", None),
                "city":                      dl_row.get("city", None),
                "team1":                     dl_row.get("team1", None),
                "team2":                     dl_row.get("team2", None),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Add year
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year.fillna(2010).astype(int)

        # Add match context columns needed for feature computers
        df["toss_bat_first"]     = 0   # unknown for DL eval; default to 0
        df["chasing_chose_chase"] = 0  # unknown; default to 0

        logger.info(f"DL evaluation snapshots: {len(df)} matches reconstructed.")
        return df

    # ------------------------------------------------------------------
    # Full DL evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        ml_model,
        dl_df: pd.DataFrame,
        deliveries_raw: pd.DataFrame,
        matches_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Evaluate ML model vs DLS on historical D/L matches.

        For each interrupted match:
          1. Reconstruct state at interruption (over = inn2_overs_bowled)
          2. Add ELO, player, venue features
          3. Predict final score with ml_model
          4. ML revised target = ceil(predicted_score)  (must exceed current score)
          5. Compare: DLS outcome vs ML outcome vs actual winner

        Parameters
        ----------
        ml_model        : fitted sklearn estimator with predict()
        dl_df           : DataFrame from parse_dl_targets()
        deliveries_raw  : all raw deliveries (including DL matches)
        matches_raw     : all raw match metadata

        Returns
        -------
        DataFrame with columns:
          match_id, date, team1, team2, winner,
          first_innings_total,
          dls_target, dls_overs,
          ml_target, ml_predicted_score,
          inn2_score_at_end,
          dls_win_prediction   (1 = DLS target says team2 wins)
          ml_win_prediction    (1 = ML target says team2 wins)
          actual_team2_won     (1 = team2 actually won under DLS)
          target_diff          (ml_target - dls_target)
        """
        # Step 1: Build delivery-level feature snapshots for DL matches
        snap_df = self._build_dl_match_snapshots(dl_df, deliveries_raw, matches_raw)
        if snap_df.empty:
            logger.warning("No DL match snapshots could be built; returning empty result.")
            return pd.DataFrame()

        # Step 2: Add ELO
        from src.elo_tracker import ELOTracker
        # ELO is match-level: look up pre-match ratings from matches_raw
        snap_df = add_elo_to_snapshots(snap_df, matches_raw, format_key="mens_odi")

        # Step 3: Add player features (innings_num=2)
        snap_df = self.pfc.transform(snap_df, deliveries_raw, innings_num=2)

        # Step 4: Add venue features
        snap_df = self.vfc.transform(snap_df, matches_raw)

        # Step 5: run_rate_vs_venue
        venue_avg_rr = snap_df["venue_avg_score"] / self.overs_limit
        venue_avg_rr = venue_avg_rr.replace(0, np.nan).fillna(
            snap_df["current_run_rate"].median()
        )
        snap_df["run_rate_vs_venue"] = (snap_df["current_run_rate"] / venue_avg_rr).round(4)

        # Step 6: NaN cleanup
        for col in FEATURE_COLUMNS_INN2:
            if col in snap_df.columns:
                snap_df[col] = pd.to_numeric(snap_df[col], errors="coerce").fillna(0)

        # Step 7: ML prediction
        missing = [c for c in FEATURE_COLUMNS_INN2 if c not in snap_df.columns]
        if missing:
            logger.warning(f"Missing features for ML prediction: {missing}")
            for c in missing:
                snap_df[c] = 0.0

        X_eval = snap_df[FEATURE_COLUMNS_INN2].values
        ml_predicted_scores = ml_model.predict(X_eval)

        snap_df["ml_predicted_score"] = np.round(ml_predicted_scores, 1)
        # ML revised target must exceed current score (can't set target lower than where they are)
        snap_df["ml_target"] = np.maximum(
            np.ceil(ml_predicted_scores).astype(int),
            snap_df["current_score"] + 1,
        )

        # Step 8: Build comparison DataFrame
        comparison = snap_df[[
            "match_id", "date", "team1", "team2", "winner",
            "first_innings_total", "dl_target_runs", "dl_target_overs",
            "inn2_score_at_end", "inn2_overs_bowled", "inn2_wickets_at_end",
            "current_score",
            "ml_predicted_score", "ml_target",
        ]].copy()

        comparison = comparison.rename(columns={
            "dl_target_runs":   "dls_target",
            "dl_target_overs":  "dls_overs",
        })

        # Under DLS: team2 wins if final score >= DLS target
        comparison["dls_win_prediction"] = (
            comparison["inn2_score_at_end"] >= comparison["dls_target"]
        ).astype(int)

        # Under ML: team2 wins if final score >= ML target
        comparison["ml_win_prediction"] = (
            comparison["inn2_score_at_end"] >= comparison["ml_target"]
        ).astype(int)

        # Actual: did team2 win? (resolve from match winner vs team2 identity)
        def _team2_won(row):
            winner = row.get("winner")
            team2  = row.get("team2")
            if pd.isna(winner) or pd.isna(team2):
                return np.nan
            return int(str(winner) == str(team2))

        comparison["actual_team2_won"] = comparison.apply(_team2_won, axis=1)
        comparison["target_diff"]      = comparison["ml_target"] - comparison["dls_target"]

        # ---- Summary statistics ----
        n = len(comparison)
        dls_correct = int((comparison["dls_win_prediction"] == comparison["actual_team2_won"]).sum())
        ml_correct  = int((comparison["ml_win_prediction"]  == comparison["actual_team2_won"]).sum())
        agreement   = int((comparison["dls_win_prediction"] == comparison["ml_win_prediction"]).sum())

        logger.info("=" * 60)
        logger.info(f"Revised Target Evaluation — {n} DL matches")
        logger.info(f"  DLS outcome accuracy : {dls_correct}/{n} ({100*dls_correct/n:.1f}%)")
        logger.info(f"  ML  outcome accuracy : {ml_correct}/{n}  ({100*ml_correct/n:.1f}%)")
        logger.info(f"  DLS/ML agreement     : {agreement}/{n}  ({100*agreement/n:.1f}%)")
        logger.info(
            f"  Mean target diff (ML − DLS): "
            f"{comparison['target_diff'].mean():.1f} runs  "
            f"(std {comparison['target_diff'].std():.1f})"
        )
        logger.info("=" * 60)

        return comparison
