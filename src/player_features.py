"""
Player Feature Computer.

Computes rolling batting and bowling statistics for each player from
ball-by-ball delivery data, strictly in chronological order (no look-ahead).

For each over-boundary snapshot we add:
  Batting (current striker + non-striker):
    batter1_avg_30       - striker rolling batting average (last 30 innings, min 5)
    batter1_sr_30        - striker rolling strike rate
    batter1_boundary_rate_30  - striker boundary rate (4s+6s per ball)
    batter1_innings_count     - career innings count (proxy for experience)
    batter2_avg_30       - non-striker rolling batting average
    batter2_sr_30        - non-striker rolling strike rate
    partnership_quality  - harmonic mean of both batters' strike rates
  Bowling (current over's bowler):
    current_bowler_economy_30  - bowler economy rate (last 30 bowling innings, min 3)
    current_bowler_sr_30       - bowler strike rate (balls per wicket)
  Team rolling form:
    batting_team_avg_score_5   - batting team's average first-innings score, last 5 matches
    bowling_team_avg_economy_5 - bowling team's average economy rate, last 5 matches

Design decisions:
  - Window = 30 most recent innings/matches (not calendar time — avoids sparsity issues)
  - Minimum observations before using rolling stat: MIN_OBS (5 for batting, 3 for bowling)
  - Below minimum: impute with global mean from the entire training pool
  - All features computed from data BEFORE the current match (no same-match contamination)
  - Player name matching is exact (Cricsheet names are consistent within a dataset)
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATTING_WINDOW = 30
BOWLING_WINDOW = 30
TEAM_FORM_WINDOW = 5
MIN_BATTING_OBS = 5
MIN_BOWLING_OBS = 3

# Global fallbacks (will be overwritten after first pass of data)
_GLOBAL_BATTING_AVG = 28.0
_GLOBAL_BATTING_SR = 72.0
_GLOBAL_BOUNDARY_RATE = 0.12
_GLOBAL_BOWLER_ECONOMY = 5.5
_GLOBAL_BOWLER_SR = 35.0


class PlayerFeatureComputer:
    """
    Two-pass algorithm:
      Pass 1 (fit): iterate all matches chronologically, build up per-player
                    rolling windows in a streaming fashion.
      Pass 2 (transform): for each (match_id, over_num) snapshot, look up the
                    pre-match stats of the current batsmen and bowler.

    Usage
    -----
    pfc = PlayerFeatureComputer()
    pfc.fit(matches_df, deliveries_df)
    snapshots_enriched = pfc.transform(snapshots_df, deliveries_df)
    """

    def __init__(self):
        # Per-player batting history (updated after each completed innings)
        # Key: player_name -> deque of innings dicts {runs, balls, dismissal}
        self._batting: dict[str, list] = defaultdict(list)

        # Per-player bowling history (updated after each completed match for that bowler)
        # Key: player_name -> deque of match bowling dicts {runs, balls, wickets}
        self._bowling: dict[str, list] = defaultdict(list)

        # Team first-innings scores: key: team_name -> list of recent scores
        self._team_scores: dict[str, list] = defaultdict(list)

        # Team bowling economy: key: team_name -> list of recent economies
        self._team_economy: dict[str, list] = defaultdict(list)

        # Pre-match feature cache: match_id -> {player: stats_dict}
        self._match_cache: dict[str, dict] = {}

        # Global fallbacks computed from all data (set during fit)
        self._global_batting_avg = _GLOBAL_BATTING_AVG
        self._global_batting_sr = _GLOBAL_BATTING_SR
        self._global_boundary_rate = _GLOBAL_BOUNDARY_RATE
        self._global_bowler_economy = _GLOBAL_BOWLER_ECONOMY
        self._global_bowler_sr = _GLOBAL_BOWLER_SR

        # Pre-match team form cache: match_id -> {team: form_dict}
        self._team_form_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> "PlayerFeatureComputer":
        """
        Process all matches chronologically. After fitting, the internal
        state is ready for transform() calls.
        """
        df_m = matches_df.copy()
        df_m["date"] = pd.to_datetime(df_m["date"], errors="coerce")
        df_m = df_m.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Index deliveries by match_id for fast lookup
        del_idx = deliveries_df.groupby("match_id")

        total = len(df_m)
        logger.info(f"PlayerFeatureComputer: fitting on {total} matches...")

        for i, (_, match_row) in enumerate(df_m.iterrows()):
            mid = match_row["match_id"]

            # --- Snapshot pre-match state ---
            self._cache_pre_match_state(mid, del_idx)

            # --- Update rolling windows with this match's data ---
            if mid in del_idx.groups:
                match_dels = del_idx.get_group(mid)
                self._update_batting(match_dels, mid)
                self._update_bowling(match_dels, mid)
                self._update_team_form(match_dels, match_row, mid)

        # Compute global fallbacks from all accumulated data
        self._compute_global_fallbacks()

        logger.info(
            f"PlayerFeatureComputer: fit complete. "
            f"Players tracked (batting): {len(self._batting)}, "
            f"(bowling): {len(self._bowling)}"
        )
        return self

    def _cache_pre_match_state(self, match_id: str, del_idx):
        """Store current rolling windows for all players in this match."""
        if match_id not in del_idx.groups:
            return
        match_dels = del_idx.get_group(match_id)
        players = set(match_dels["batter"].dropna()) | set(match_dels["bowler"].dropna())

        cache = {}
        for player in players:
            cache[player] = {
                "batting": self._get_batting_stats(player),
                "bowling": self._get_bowling_stats(player),
            }

        self._match_cache[match_id] = cache

        # Cache team form
        teams = set(match_dels["batting_team"].dropna())
        bowling_teams = set()
        for team in match_dels["batting_team"].dropna().unique():
            # Bowling team = all teams in this match that are not the batting team
            all_teams = set(match_dels["batting_team"].dropna().unique())
            bowling_teams |= all_teams

        form_cache = {}
        for team in bowling_teams:
            form_cache[team] = {
                "avg_score_5": self._get_team_avg_score(team),
                "avg_economy_5": self._get_team_avg_economy(team),
            }
        self._team_form_cache[match_id] = form_cache

    def _update_batting(self, match_dels: pd.DataFrame, match_id: str):
        """Update per-batter rolling windows after match is complete."""
        inn1 = match_dels[match_dels["innings"] == 1]
        inn2 = match_dels[match_dels["innings"] == 2]

        for innings_dels in [inn1, inn2]:
            if innings_dels.empty:
                continue
            for batter, batter_dels in innings_dels.groupby("batter"):
                if pd.isna(batter):
                    continue
                runs = int(batter_dels["batter_runs"].sum())
                legal = int(batter_dels[~batter_dels["is_wide"]]["batter_runs"].count())
                dismissal = int(batter_dels["is_wicket"].any() and
                                batter_dels[batter_dels["is_wicket"]]["player_out"].eq(batter).any())
                fours = int(batter_dels["is_boundary_four"].sum())
                sixes = int(batter_dels["is_boundary_six"].sum())

                entry = {
                    "runs": runs,
                    "balls": legal,
                    "dismissed": dismissal,
                    "boundaries": fours + sixes,
                }
                history = self._batting[batter]
                history.append(entry)
                if len(history) > BATTING_WINDOW:
                    history.pop(0)

    def _update_bowling(self, match_dels: pd.DataFrame, match_id: str):
        """Update per-bowler rolling windows after match is complete."""
        for bowler, bowler_dels in match_dels.groupby("bowler"):
            if pd.isna(bowler):
                continue
            legal = bowler_dels[~bowler_dels["is_wide"] & ~bowler_dels["is_noball"]]
            runs_conceded = int(bowler_dels["total_runs"].sum()
                                - bowler_dels["is_boundary_six"].sum() * 0  # total includes all
                                )
            # runs conceded = total runs - byes - leg byes (not tracked here, approximate)
            runs_conceded = int(bowler_dels["total_runs"].sum())
            balls = int(len(legal))
            wickets = int(bowler_dels["is_wicket"].sum())

            # Exclude run-outs from bowler wickets
            if "dismissal_kind" in bowler_dels.columns:
                run_outs = int((bowler_dels["dismissal_kind"] == "run out").sum())
                wickets = max(0, wickets - run_outs)

            if balls == 0:
                continue

            entry = {
                "runs": runs_conceded,
                "balls": balls,
                "wickets": wickets,
            }
            history = self._bowling[bowler]
            history.append(entry)
            if len(history) > BOWLING_WINDOW:
                history.pop(0)

    def _update_team_form(self, match_dels: pd.DataFrame, match_row, match_id: str):
        """Update team-level rolling first-innings score and economy."""
        inn1 = match_dels[match_dels["innings"] == 1]
        inn2 = match_dels[match_dels["innings"] == 2]

        for innings_num, innings_dels in [(1, inn1), (2, inn2)]:
            if innings_dels.empty:
                continue
            batting_team = innings_dels["batting_team"].iloc[0]
            bowling_team = match_dels[match_dels["innings"] != innings_num]["batting_team"].unique()

            score = int(innings_dels["total_runs"].sum())
            legal_balls = int(innings_dels[
                ~innings_dels["is_wide"] & ~innings_dels["is_noball"]
            ].shape[0])
            overs_bowled = legal_balls / 6.0
            economy = score / overs_bowled if overs_bowled > 0 else 0.0

            if innings_num == 1:
                score_history = self._team_scores[batting_team]
                score_history.append(score)
                if len(score_history) > TEAM_FORM_WINDOW:
                    score_history.pop(0)

            for bt in bowling_team:
                economy_history = self._team_economy[bt]
                economy_history.append(economy)
                if len(economy_history) > TEAM_FORM_WINDOW:
                    economy_history.pop(0)

    # ------------------------------------------------------------------
    # Internal stat retrievers
    # ------------------------------------------------------------------

    def _get_batting_stats(self, player: str) -> dict:
        history = self._batting.get(player, [])
        n = len(history)
        if n < MIN_BATTING_OBS:
            return {
                "avg": None, "sr": None, "boundary_rate": None,
                "innings_count": n, "reliable": False,
            }
        total_runs = sum(h["runs"] for h in history)
        total_balls = sum(h["balls"] for h in history)
        total_dismissed = sum(h["dismissed"] for h in history)
        total_boundaries = sum(h["boundaries"] for h in history)

        avg = total_runs / max(1, total_dismissed)
        sr = (total_runs / max(1, total_balls)) * 100.0
        boundary_rate = total_boundaries / max(1, total_balls)

        return {
            "avg": avg, "sr": sr, "boundary_rate": boundary_rate,
            "innings_count": n, "reliable": True,
        }

    def _get_bowling_stats(self, player: str) -> dict:
        history = self._bowling.get(player, [])
        n = len(history)
        if n < MIN_BOWLING_OBS:
            return {"economy": None, "bowling_sr": None, "reliable": False}
        total_runs = sum(h["runs"] for h in history)
        total_balls = sum(h["balls"] for h in history)
        total_wickets = sum(h["wickets"] for h in history)

        economy = (total_runs / max(1, total_balls)) * 6.0
        bowling_sr = total_balls / max(1, total_wickets)

        return {"economy": economy, "bowling_sr": bowling_sr, "reliable": True}

    def _get_team_avg_score(self, team: str) -> float | None:
        history = self._team_scores.get(team, [])
        return float(np.mean(history)) if len(history) >= 2 else None

    def _get_team_avg_economy(self, team: str) -> float | None:
        history = self._team_economy.get(team, [])
        return float(np.mean(history)) if len(history) >= 2 else None

    def _compute_global_fallbacks(self):
        """Compute global mean stats across all tracked players for imputation."""
        all_avgs = []
        all_srs = []
        all_brs = []
        for stats_list in self._batting.values():
            if len(stats_list) >= MIN_BATTING_OBS:
                total_runs = sum(h["runs"] for h in stats_list)
                total_balls = sum(h["balls"] for h in stats_list)
                total_dismissed = sum(h["dismissed"] for h in stats_list)
                total_boundaries = sum(h["boundaries"] for h in stats_list)
                all_avgs.append(total_runs / max(1, total_dismissed))
                all_srs.append((total_runs / max(1, total_balls)) * 100)
                all_brs.append(total_boundaries / max(1, total_balls))

        all_economies = []
        all_bowling_srs = []
        for stats_list in self._bowling.values():
            if len(stats_list) >= MIN_BOWLING_OBS:
                tr = sum(h["runs"] for h in stats_list)
                tb = sum(h["balls"] for h in stats_list)
                tw = sum(h["wickets"] for h in stats_list)
                all_economies.append((tr / max(1, tb)) * 6)
                all_bowling_srs.append(tb / max(1, tw))

        if all_avgs:
            self._global_batting_avg = float(np.median(all_avgs))
            self._global_batting_sr = float(np.median(all_srs))
            self._global_boundary_rate = float(np.median(all_brs))
        if all_economies:
            self._global_bowler_economy = float(np.median(all_economies))
            self._global_bowler_sr = float(np.median(all_bowling_srs))

        logger.info(
            f"Global fallbacks — batting avg: {self._global_batting_avg:.1f}, "
            f"SR: {self._global_batting_sr:.1f}, "
            f"economy: {self._global_bowler_economy:.2f}"
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        snapshots_df: pd.DataFrame,
        deliveries_df: pd.DataFrame,
        innings_num: int = 1,
    ) -> pd.DataFrame:
        """
        Add player feature columns to snapshots_df.

        For each snapshot (match_id, overs_completed), looks up:
          - Who was the striker and non-striker at the END of that over
          - Who bowled that over
        Then retrieves their pre-match rolling stats from the cache.

        Parameters
        ----------
        snapshots_df : DataFrame with columns match_id, overs_completed
        deliveries_df : full deliveries DataFrame
        innings_num   : which innings to build the over-player lookup from (1 or 2)

        Returns
        -------
        snapshots_df enriched with player feature columns.
        """
        # Build per-(match, over) lookup for: striker, non_striker, bowler
        inn = deliveries_df[deliveries_df["innings"] == innings_num].copy()
        over_players = self._build_over_player_lookup(inn)

        df = snapshots_df.copy()

        player_rows = []
        for _, row in df.iterrows():
            mid = row["match_id"]
            over = int(row["overs_completed"]) - 1  # overs_completed is 1-indexed; over is 0-indexed

            # Look up players at this snapshot
            key = (mid, over)
            if key in over_players:
                striker, non_striker, bowler = over_players[key]
            else:
                striker, non_striker, bowler = None, None, None

            # Get pre-match cache
            cache = self._match_cache.get(mid, {})
            team_cache = self._team_form_cache.get(mid, {})
            batting_team = row.get("batting_team", None)

            # Batting team form
            batting_form = team_cache.get(batting_team, {}) if batting_team else {}
            batting_avg_score_5 = batting_form.get("avg_score_5", None)

            # Derive bowling team from match deliveries
            # (bowling team = opponent of batting team)
            bowling_team = None
            if mid in self._match_cache:
                all_teams_in_match = set()
                # We can infer from the team form cache
                for t in team_cache:
                    if t != batting_team:
                        bowling_team = t
                        break
            bowling_form = team_cache.get(bowling_team, {}) if bowling_team else {}
            bowling_avg_economy_5 = bowling_form.get("avg_economy_5", None)

            def get_batting_stat(player, stat, fallback):
                if player is None or player not in cache:
                    return fallback
                s = cache[player]["batting"]
                return s.get(stat) if s.get(stat) is not None else fallback

            def get_bowling_stat(player, stat, fallback):
                if player is None or player not in cache:
                    return fallback
                s = cache[player]["bowling"]
                return s.get(stat) if s.get(stat) is not None else fallback

            b1_avg = get_batting_stat(striker, "avg", self._global_batting_avg)
            b1_sr = get_batting_stat(striker, "sr", self._global_batting_sr)
            b1_br = get_batting_stat(striker, "boundary_rate", self._global_boundary_rate)
            b1_count = cache.get(striker, {}).get("batting", {}).get("innings_count", 0) if striker else 0

            b2_avg = get_batting_stat(non_striker, "avg", self._global_batting_avg)
            b2_sr = get_batting_stat(non_striker, "sr", self._global_batting_sr)

            bowler_eco = get_bowling_stat(bowler, "economy", self._global_bowler_economy)
            bowler_sr = get_bowling_stat(bowler, "bowling_sr", self._global_bowler_sr)

            # Partnership quality: harmonic mean of strike rates (handle zeros)
            if b1_sr > 0 and b2_sr > 0:
                partnership_quality = 2 * b1_sr * b2_sr / (b1_sr + b2_sr)
            else:
                partnership_quality = (b1_sr + b2_sr) / 2.0

            player_rows.append({
                "batter1_avg_30": round(b1_avg, 2),
                "batter1_sr_30": round(b1_sr, 2),
                "batter1_boundary_rate_30": round(b1_br, 4),
                "batter1_innings_count": b1_count,
                "batter2_avg_30": round(b2_avg, 2),
                "batter2_sr_30": round(b2_sr, 2),
                "partnership_quality": round(partnership_quality, 2),
                "current_bowler_economy_30": round(bowler_eco, 3),
                "current_bowler_sr_30": round(bowler_sr, 2),
                "batting_team_avg_score_5": round(batting_avg_score_5, 1) if batting_avg_score_5 else None,
                "bowling_team_avg_economy_5": round(bowling_avg_economy_5, 3) if bowling_avg_economy_5 else None,
            })

        player_df = pd.DataFrame(player_rows, index=df.index)

        # Impute team-level NaNs: use column median, fallback to hardcoded global if all NaN
        score_median = player_df["batting_team_avg_score_5"].median()
        economy_median = player_df["bowling_team_avg_economy_5"].median()
        player_df["batting_team_avg_score_5"] = player_df["batting_team_avg_score_5"].fillna(
            score_median if pd.notna(score_median) else 250.0
        )
        player_df["bowling_team_avg_economy_5"] = player_df["bowling_team_avg_economy_5"].fillna(
            economy_median if pd.notna(economy_median) else 5.5
        )

        result = pd.concat([df, player_df], axis=1)
        logger.info(
            f"Player features added. New columns: {list(player_df.columns)}. "
            f"Shape: {result.shape}"
        )
        return result

    def _build_over_player_lookup(self, deliveries: pd.DataFrame) -> dict:
        """
        Build dict: (match_id, over_num_0indexed) -> (striker, non_striker, bowler)
        using the LAST delivery of each over.
        """
        lookup = {}
        # Sort to ensure last delivery is accessible
        sorted_dels = deliveries.sort_values(["match_id", "over", "ball"])

        for (mid, over), group in sorted_dels.groupby(["match_id", "over"]):
            last = group.iloc[-1]
            striker = last.get("batter", None)
            non_striker = last.get("non_striker", None)
            bowler = last.get("bowler", None)
            # Convert NaN to None
            striker = None if pd.isna(striker) else str(striker)
            non_striker = None if pd.isna(non_striker) else str(non_striker)
            bowler = None if pd.isna(bowler) else str(bowler)
            lookup[(str(mid), int(over))] = (striker, non_striker, bowler)

        return lookup
