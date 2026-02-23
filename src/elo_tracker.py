"""
ELO Rating Tracker for Cricket Teams.

Computes pre-match ELO ratings for every team in chronological order.
Separate ratings maintained per format (ODI / T20).

Key design decisions:
  - K-factor 32 base; scaled by match importance (World Cup 2x, ICC event 1.5x, bilateral 1x)
  - Home advantage: +50 ELO points to home team's expected score calculation
  - Starting ELO: 1500 for all teams
  - Burn-in: first MIN_MATCHES_FOR_RELIABILITY matches per team flagged as unreliable
  - Formats processed independently (ODI and T20 have separate ratings)
  - Draws/ties split the margin equally (0.5 each)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INITIAL_ELO = 1500.0
K_BASE = 32.0
HOME_ADVANTAGE = 50.0        # ELO points added to home team's effective rating
MIN_MATCHES_FOR_RELIABILITY = 20   # flag ELO as unreliable below this

# Match-importance multipliers derived from event name keywords
IMPORTANCE_KEYWORDS = {
    2.0: ["world cup", "wc ", "cricket world cup"],
    1.5: ["champions trophy", "world championship", "icc", "tri-series", "tri series"],
    1.0: [],  # fallback bilateral
}


def _importance_multiplier(event_name: str) -> float:
    """Return K-factor multiplier based on event name."""
    if not event_name:
        return 1.0
    name = str(event_name).lower()
    for mult, keywords in IMPORTANCE_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return mult
    return 1.0


class ELOTracker:
    """
    Processes a matches DataFrame in chronological order and produces
    pre-match ELO ratings for every team appearing in that match.

    Usage
    -----
    tracker = ELOTracker(format_key="mens_odi")
    match_elos_df = tracker.fit(matches_df)
    # Returns DataFrame with columns:
    #   match_id, batting_team_elo, bowling_team_elo, elo_gap,
    #   batting_team_elo_reliable, bowling_team_elo_reliable
    """

    def __init__(self, format_key: str = "mens_odi"):
        self.format_key = format_key
        self._ratings: dict[str, float] = {}
        self._match_counts: dict[str, int] = {}  # per team
        self._history: list[dict] = []            # list of per-match ELO records

    def _get_rating(self, team: str) -> float:
        return self._ratings.get(team, INITIAL_ELO)

    def _is_reliable(self, team: str) -> bool:
        return self._match_counts.get(team, 0) >= MIN_MATCHES_FOR_RELIABILITY

    def _expected_score(self, rating_a: float, rating_b: float, home_team: str,
                        team_a: str, venue_country: str) -> float:
        """
        Expected score for team A vs team B.
        Home advantage applied if team_a is the home team.
        """
        effective_a = rating_a
        effective_b = rating_b
        if home_team and venue_country:
            # Crude home-detection: team name matches venue country
            if team_a.lower() in venue_country.lower() or venue_country.lower() in team_a.lower():
                effective_a += HOME_ADVANTAGE
            else:
                effective_b += HOME_ADVANTAGE
        return 1.0 / (1.0 + 10.0 ** ((effective_b - effective_a) / 400.0))

    def _update(self, team: str, actual: float, expected: float, k: float):
        old = self._get_rating(team)
        self._ratings[team] = old + k * (actual - expected)
        self._match_counts[team] = self._match_counts.get(team, 0) + 1

    def fit(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process matches in chronological order and return a DataFrame with
        pre-match ELO for every (match_id, team1, team2) pair.

        Parameters
        ----------
        matches_df : must contain columns:
            match_id, date, team1, team2, winner, result, venue, city
            Optional: event (tournament name for importance scaling)

        Returns
        -------
        elo_df : DataFrame indexed by match_id with columns:
            match_id, team1_elo, team2_elo,
            team1_elo_reliable, team2_elo_reliable
        """
        df = matches_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "team1", "team2"]).sort_values("date").reset_index(drop=True)

        records = []

        for _, row in df.iterrows():
            team1 = str(row["team1"])
            team2 = str(row["team2"])
            winner = row.get("winner", None)
            result = row.get("result", None)
            event = row.get("event", "")
            venue_country = str(row.get("city", "") or row.get("venue", "") or "")

            # --- Record PRE-match ratings ---
            r1 = self._get_rating(team1)
            r2 = self._get_rating(team2)

            records.append({
                "match_id": row["match_id"],
                "team1": team1,
                "team2": team2,
                "team1_elo": r1,
                "team2_elo": r2,
                "team1_elo_reliable": self._is_reliable(team1),
                "team2_elo_reliable": self._is_reliable(team2),
            })

            # --- Determine match result ---
            if pd.isna(winner) and result == "tie":
                actual1, actual2 = 0.5, 0.5
            elif pd.isna(winner):
                # No result or abandoned — do not update ELO
                self._match_counts[team1] = self._match_counts.get(team1, 0) + 1
                self._match_counts[team2] = self._match_counts.get(team2, 0) + 1
                continue
            elif str(winner) == team1:
                actual1, actual2 = 1.0, 0.0
            elif str(winner) == team2:
                actual1, actual2 = 0.0, 1.0
            else:
                # Winner doesn't match either team (shouldn't happen but be safe)
                self._match_counts[team1] = self._match_counts.get(team1, 0) + 1
                self._match_counts[team2] = self._match_counts.get(team2, 0) + 1
                continue

            k = K_BASE * _importance_multiplier(event)
            exp1 = self._expected_score(r1, r2, str(winner), team1, venue_country)
            exp2 = 1.0 - exp1

            self._update(team1, actual1, exp1, k)
            self._update(team2, actual2, exp2, k)

        elo_df = pd.DataFrame(records)
        logger.info(
            f"[{self.format_key}] ELO computed for {len(elo_df)} matches, "
            f"{len(self._ratings)} teams. "
            f"ELO range: [{min(self._ratings.values()):.0f}, {max(self._ratings.values()):.0f}]"
        )
        return elo_df

    def get_final_ratings(self) -> dict[str, float]:
        """Return the most recent ELO for every team (post all processed matches)."""
        return dict(self._ratings)


def add_elo_to_snapshots(
    snapshots_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    format_key: str = "mens_odi",
) -> pd.DataFrame:
    """
    Convenience function: compute ELO and join onto snapshots.

    The snapshots DataFrame must have 'match_id' and 'batting_team'.
    The bowling team is inferred from match metadata (team1/team2 that is
    not the batting team).

    Returns snapshots_df with added columns:
        batting_team_elo, bowling_team_elo, elo_gap,
        batting_elo_reliable, bowling_elo_reliable
    """
    tracker = ELOTracker(format_key=format_key)
    elo_df = tracker.fit(matches_df)

    # Build a lookup: match_id -> {team -> elo}
    team1_map = elo_df.set_index("match_id")[["team1", "team1_elo", "team1_elo_reliable"]]
    team2_map = elo_df.set_index("match_id")[["team2", "team2_elo", "team2_elo_reliable"]]

    def get_team_elos(match_id: str, batting_team: str) -> tuple:
        if match_id not in team1_map.index:
            return INITIAL_ELO, INITIAL_ELO, False, False

        t1 = team1_map.loc[match_id, "team1"]
        t2 = team2_map.loc[match_id, "team2"]
        e1 = team1_map.loc[match_id, "team1_elo"]
        e2 = team2_map.loc[match_id, "team2_elo"]
        r1 = team1_map.loc[match_id, "team1_elo_reliable"]
        r2 = team2_map.loc[match_id, "team2_elo_reliable"]

        if batting_team == t1:
            return e1, e2, r1, r2
        else:
            return e2, e1, r2, r1

    df = snapshots_df.copy()
    elo_cols = df.apply(
        lambda row: get_team_elos(row["match_id"], row["batting_team"]),
        axis=1,
        result_type="expand",
    )
    elo_cols.columns = ["batting_team_elo", "bowling_team_elo",
                        "batting_elo_reliable", "bowling_elo_reliable"]

    df = pd.concat([df, elo_cols], axis=1)
    df["elo_gap"] = df["batting_team_elo"] - df["bowling_team_elo"]

    logger.info(
        f"[{format_key}] ELO joined to snapshots. "
        f"Mean batting ELO: {df['batting_team_elo'].mean():.1f}, "
        f"ELO gap range: [{df['elo_gap'].min():.0f}, {df['elo_gap'].max():.0f}]"
    )
    return df
