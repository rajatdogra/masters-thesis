"""
Data Processing Module
Filters matches, creates match-state snapshots at over boundaries.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def filter_completed_matches(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> tuple:
    """
    Filter to only completed first-innings matches:
    - No D/L method applied
    - No 'no result'
    - Has a winner or clear result
    - First innings was completed (not shortened)
    """
    # Remove D/L matches
    mask = matches_df["method"].isna()
    # Remove 'no result'
    mask &= matches_df["result"] != "no result"
    # Keep matches that have a winner OR are ties
    mask &= (matches_df["winner"].notna()) | (matches_df["result"] == "tie")

    filtered_matches = matches_df[mask].copy()
    logger.info(
        f"Filtered from {len(matches_df)} to {len(filtered_matches)} completed matches "
        f"(removed {len(matches_df) - len(filtered_matches)} D/L, no-result, or incomplete)"
    )

    # Filter deliveries to only include these matches
    valid_ids = set(filtered_matches["match_id"])
    filtered_deliveries = deliveries_df[deliveries_df["match_id"].isin(valid_ids)].copy()

    return filtered_matches, filtered_deliveries


def compute_first_innings_totals(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """Compute first innings final total for each match."""
    first_innings = deliveries_df[deliveries_df["innings"] == 1]
    totals = first_innings.groupby("match_id")["total_runs"].sum().reset_index()
    totals.columns = ["match_id", "final_total"]
    return totals


def create_over_snapshots(
    deliveries_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    overs_limit: int = 50,
) -> pd.DataFrame:
    """
    Create match-state snapshots at each over boundary for first innings.
    Each row represents the state after over X is completed.
    """
    first_innings = deliveries_df[deliveries_df["innings"] == 1].copy()

    # Compute first-innings totals
    totals = compute_first_innings_totals(deliveries_df)

    # Track the maximum completed over per match
    # to avoid creating snapshots beyond what was actually bowled
    max_overs = first_innings.groupby("match_id")["over"].max().reset_index()
    max_overs.columns = ["match_id", "max_over_bowled"]

    snapshots = []
    match_ids = first_innings["match_id"].unique()

    for match_id in match_ids:
        match_deliveries = first_innings[first_innings["match_id"] == match_id].sort_values(
            ["over", "ball"]
        )

        if match_deliveries.empty:
            continue

        match_info = matches_df[matches_df["match_id"] == match_id]
        if match_info.empty:
            continue
        match_info = match_info.iloc[0]

        final_total_row = totals[totals["match_id"] == match_id]
        if final_total_row.empty:
            continue
        final_total = final_total_row.iloc[0]["final_total"]

        batting_team = match_deliveries.iloc[0]["batting_team"]
        actual_max_over = match_deliveries["over"].max()

        # Build cumulative state over by over
        cumulative_runs = 0
        cumulative_wickets = 0
        cumulative_balls = 0  # legal deliveries
        cumulative_boundaries = 0
        cumulative_dots = 0
        cumulative_sixes = 0
        total_deliveries = 0  # including extras

        # Track recent history for rolling features
        over_runs_history = []
        over_wickets_history = []

        # Partnership tracking
        current_partnership_runs = 0

        for over_num in range(0, actual_max_over + 1):
            over_deliveries = match_deliveries[match_deliveries["over"] == over_num]

            if over_deliveries.empty:
                # No deliveries in this over (shouldn't happen normally)
                over_runs_history.append(0)
                over_wickets_history.append(0)
                continue

            over_runs = over_deliveries["total_runs"].sum()
            over_wickets = over_deliveries["is_wicket"].sum()
            over_legal_balls = len(over_deliveries[
                ~over_deliveries["is_wide"] & ~over_deliveries["is_noball"]
            ])
            # Count wides and no-balls as deliveries too for total count
            over_total_deliveries = len(over_deliveries)
            over_boundaries = (
                over_deliveries["is_boundary_four"].sum()
                + over_deliveries["is_boundary_six"].sum()
            )
            over_dots = len(over_deliveries[
                (over_deliveries["total_runs"] == 0)
                & ~over_deliveries["is_wide"]
                & ~over_deliveries["is_noball"]
            ])
            over_sixes = over_deliveries["is_boundary_six"].sum()

            cumulative_runs += over_runs
            cumulative_wickets += over_wickets
            cumulative_balls += over_legal_balls
            cumulative_boundaries += over_boundaries
            cumulative_dots += over_dots
            cumulative_sixes += over_sixes
            total_deliveries += over_total_deliveries

            # Partnership tracking
            if over_wickets > 0:
                current_partnership_runs = 0
                # Approximate: reset after last wicket in over
                # Add runs scored after last wicket ball
                wicket_indices = over_deliveries[over_deliveries["is_wicket"]].index
                if len(wicket_indices) > 0:
                    last_wicket_idx = wicket_indices[-1]
                    after_wicket = over_deliveries.loc[last_wicket_idx:]
                    current_partnership_runs = after_wicket["total_runs"].sum()
            else:
                current_partnership_runs += over_runs

            over_runs_history.append(over_runs)
            over_wickets_history.append(over_wickets)

            overs_completed = over_num + 1  # 1-indexed
            overs_remaining = overs_limit - overs_completed

            # Current run rate
            current_run_rate = cumulative_runs / overs_completed if overs_completed > 0 else 0

            # Recent run rate (last 5 overs)
            recent_overs = over_runs_history[-5:]
            recent_run_rate_5 = sum(recent_overs) / len(recent_overs) if recent_overs else 0

            # Scoring acceleration (current RR - RR at 5 overs ago)
            if overs_completed > 5:
                old_runs = sum(over_runs_history[:-5])
                old_rr = old_runs / (overs_completed - 5)
                scoring_acceleration = current_run_rate - old_rr
            else:
                scoring_acceleration = 0.0

            # Boundary and dot percentages
            boundary_pct = (
                cumulative_boundaries / total_deliveries * 100
                if total_deliveries > 0
                else 0
            )
            dot_pct = (
                cumulative_dots / cumulative_balls * 100
                if cumulative_balls > 0
                else 0
            )

            # Recent wickets in last 5 overs
            recent_wickets_5 = sum(over_wickets_history[-5:])

            # Phase indicators (ODI phases)
            if overs_limit == 50:
                is_powerplay = 1 if overs_completed <= 10 else 0
                is_middle_overs = 1 if 11 <= overs_completed <= 40 else 0
                is_death_overs = 1 if overs_completed > 40 else 0
            else:
                # T20 phases
                is_powerplay = 1 if overs_completed <= 6 else 0
                is_middle_overs = 1 if 7 <= overs_completed <= 15 else 0
                is_death_overs = 1 if overs_completed > 15 else 0

            snapshot = {
                "match_id": match_id,
                "batting_team": batting_team,
                "overs_completed": overs_completed,
                "overs_remaining": overs_remaining,
                "current_score": cumulative_runs,
                "wickets_fallen": cumulative_wickets,
                "wickets_in_hand": 10 - cumulative_wickets,
                "current_run_rate": round(current_run_rate, 4),
                "recent_run_rate_5": round(recent_run_rate_5, 4),
                "scoring_acceleration": round(scoring_acceleration, 4),
                "boundary_percentage": round(boundary_pct, 4),
                "dot_ball_percentage": round(dot_pct, 4),
                "partnership_runs": current_partnership_runs,
                "is_powerplay": is_powerplay,
                "is_middle_overs": is_middle_overs,
                "is_death_overs": is_death_overs,
                "recent_wickets_5": recent_wickets_5,
                "cumulative_boundaries": cumulative_boundaries,
                "cumulative_sixes": cumulative_sixes,
                "innings_progress": round(overs_completed / overs_limit, 4),
                "final_total": final_total,
            }
            snapshots.append(snapshot)

    snapshots_df = pd.DataFrame(snapshots)
    logger.info(f"Created {len(snapshots_df)} over-boundary snapshots from {len(match_ids)} matches")
    return snapshots_df


def add_match_context(snapshots_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add match-level context features to snapshots."""
    context_cols = ["match_id", "date", "venue", "city", "toss_winner", "toss_decision"]
    available_cols = [c for c in context_cols if c in matches_df.columns]
    context = matches_df[available_cols].copy()

    merged = snapshots_df.merge(context, on="match_id", how="left")

    # Extract year from date
    if "date" in merged.columns:
        merged["year"] = pd.to_datetime(merged["date"], errors="coerce").dt.year

    # Encode toss decision
    if "toss_decision" in merged.columns:
        merged["toss_bat_first"] = (merged["toss_decision"] == "bat").astype(int)

    return merged


def time_based_train_test_split(
    snapshots_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple:
    """
    Split data by time: older matches for training, recent for testing.
    Splits at match level (all snapshots from a match go to same set).
    """
    # Get match dates
    match_dates = matches_df[["match_id", "date"]].drop_duplicates()
    match_dates["date"] = pd.to_datetime(match_dates["date"], errors="coerce")
    match_dates = match_dates.dropna(subset=["date"]).sort_values("date")

    # Find cutoff date
    n_matches = len(match_dates)
    cutoff_idx = int(n_matches * (1 - test_fraction))
    cutoff_date = match_dates.iloc[cutoff_idx]["date"]

    train_ids = set(match_dates[match_dates["date"] < cutoff_date]["match_id"])
    test_ids = set(match_dates[match_dates["date"] >= cutoff_date]["match_id"])

    train_df = snapshots_df[snapshots_df["match_id"].isin(train_ids)].copy()
    test_df = snapshots_df[snapshots_df["match_id"].isin(test_ids)].copy()

    logger.info(
        f"Train/Test split: {len(train_ids)} train matches ({len(train_df)} rows), "
        f"{len(test_ids)} test matches ({len(test_df)} rows). "
        f"Cutoff date: {cutoff_date.date()}"
    )

    return train_df, test_df


def process_format(
    format_key: str,
    overs_limit: int = 50,
) -> tuple:
    """
    Full processing pipeline for one format.
    Returns (snapshots_df, train_df, test_df, matches_df)
    """
    from src.data_collection import load_processed

    matches_df, deliveries_df = load_processed(format_key)
    matches_df, deliveries_df = filter_completed_matches(matches_df, deliveries_df)

    snapshots_df = create_over_snapshots(deliveries_df, matches_df, overs_limit=overs_limit)
    snapshots_df = add_match_context(snapshots_df, matches_df)

    train_df, test_df = time_based_train_test_split(snapshots_df, matches_df)

    # Save
    snapshots_df.to_parquet(PROCESSED_DIR / f"{format_key}_snapshots.parquet", index=False)
    train_df.to_parquet(PROCESSED_DIR / f"{format_key}_train.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / f"{format_key}_test.parquet", index=False)

    logger.info(f"[{format_key}] Saved snapshots, train, test to parquet.")
    return snapshots_df, train_df, test_df, matches_df
