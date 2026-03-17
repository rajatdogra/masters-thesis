"""
Advanced Feature Engineering Module (V3 extension).

New feature groups beyond V2 (46 features):
  1. Head-to-Head (H2H) team statistics — 4 features
     Captures the historical matchup record between batting and bowling teams.
     Research motivation: H2H record subsumes conditions + team-specific weaknesses
     that ELO alone cannot capture (e.g., India vs Australia at Adelaide is different
     from generic ELO gap).

  2. Toss Advantage — 3 features
     Batting first vs chasing decision varies strongly by venue/conditions.
     Day/night matches, pitch behaviour, and dew factor all interact with the toss.

  3. Match Importance Context — 2 features
     ICC knockout vs bilateral bilateral ODIs have different pressure dynamics.
     Teams may pace differently in group stages vs knockouts.

  4. Season / Month Effects — 1 feature
     Month captures seasonal pitches (sub-continent spin-friendly in winter,
     pace-friendly in spring), which is partially absorbed by venue_avg_score
     but month adds within-season granularity.

Total new features: 10
V3 feature count: 46 (V2) + 10 = 56 features

Scientific design principles:
  - All features computed in strict chronological order (no look-ahead)
  - Minimum observation thresholds to avoid spurious estimates
  - Graceful fallback to global means when history is insufficient

Usage
-----
from src.advanced_features import (
    HeadToHeadComputer,
    TossAdvantageComputer,
    add_match_context_v3,
)

h2h = HeadToHeadComputer()
h2h.fit(matches_df)
snapshots = h2h.transform(snapshots, matches_df)

toss = TossAdvantageComputer()
toss.fit(matches_df)
snapshots = toss.transform(snapshots, matches_df)

snapshots = add_match_context_v3(snapshots, matches_df)
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Minimum past meetings before H2H features are considered reliable
MIN_H2H_MATCHES = 3
# Minimum past matches at a venue before venue toss win rate is considered reliable
MIN_VENUE_TOSS_MATCHES = 5

# Event importance multipliers (mirrors elo_tracker.py for consistency)
_IMPORTANCE_KW = {
    2.0: ["world cup", "wc ", "cricket world cup"],
    1.5: ["champions trophy", "world championship", "icc", "tri-series", "tri series"],
}


def _importance(event_name: str) -> float:
    """Return match importance multiplier from event name string."""
    if not event_name:
        return 1.0
    n = str(event_name).lower()
    for mult, kws in _IMPORTANCE_KW.items():
        if any(k in n for k in kws):
            return float(mult)
    return 1.0


def _infer_bowling_team(match_id, batting_team, team1_map, team2_map):
    """Return the bowling team for a first-innings match."""
    t1 = team1_map.get(match_id, None)
    t2 = team2_map.get(match_id, None)
    if t1 is None or t2 is None:
        return None
    return t2 if batting_team == t1 else t1


# ---------------------------------------------------------------------------
# 1. Head-to-Head Computer
# ---------------------------------------------------------------------------

class HeadToHeadComputer:
    """
    Computes H2H statistics between batting and bowling teams in a temporally
    safe, chronological manner (identical design to ELOTracker).

    For each match M (in chronological order), *before* processing M:
      - Record H2H stats for (batting_team, bowling_team) of M
    After M:
      - Update H2H history with the first-innings score and match result.

    Output features (4):
      h2h_avg_score      : rolling avg first-innings score, batting_team vs bowling_team
      h2h_win_rate       : batting_team win rate against bowling_team (all match outcomes)
      h2h_n_matches      : number of past meetings (reliability indicator)
      h2h_venue_avg_score: rolling avg score in first innings at THIS venue (batting_team vs bowling_team)
    """

    def __init__(self):
        # Keyed by (batting_team, bowling_team)
        self._scores: dict[tuple, list] = defaultdict(list)
        self._wins:   dict[tuple, list] = defaultdict(list)   # 1=batting_team won, 0=lost
        # Keyed by (batting_team, bowling_team, venue)
        self._venue_scores: dict[tuple, list] = defaultdict(list)

        # Per-match pre-computed H2H stats (lookup at transform time)
        self._match_h2h: dict = {}   # match_id -> dict of h2h feature values

    def fit(self, matches_df: pd.DataFrame, first_innings_totals: pd.DataFrame = None) -> "HeadToHeadComputer":
        """
        Process matches in chronological order and pre-compute H2H stats.

        Parameters
        ----------
        matches_df : must have columns match_id, date, team1, team2, winner, venue
        first_innings_totals : optional DataFrame with (match_id, final_total)
                               If None, scores are not tracked (only win rates).
        """
        df = matches_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "team1", "team2"]).sort_values("date").reset_index(drop=True)

        if first_innings_totals is not None:
            score_map = first_innings_totals.set_index("match_id")["final_total"].to_dict()
        else:
            score_map = {}

        global_scores: list = []  # running list for global fallback

        for _, row in df.iterrows():
            mid   = row["match_id"]
            t1    = str(row["team1"])
            t2    = str(row["team2"])
            winner = str(row.get("winner", "")) if not pd.isna(row.get("winner", np.nan)) else ""
            venue  = str(row.get("venue", "")) if not pd.isna(row.get("venue", np.nan)) else ""
            score  = score_map.get(mid, None)   # first innings score for this match

            # --- RECORD pre-match H2H stats for first innings (batting = t1) ---
            key_bat_t1 = (t1, t2)
            key_bat_t2 = (t2, t1)
            vkey_bat_t1 = (t1, t2, venue)
            vkey_bat_t2 = (t2, t1, venue)

            global_mean = float(np.mean(global_scores)) if len(global_scores) >= 5 else 260.0

            for (batting, bowling, bkey, vkey) in [
                (t1, t2, key_bat_t1, vkey_bat_t1),
                (t2, t1, key_bat_t2, vkey_bat_t2),
            ]:
                sc_hist = self._scores[bkey]
                w_hist  = self._wins[bkey]
                vsc_hist = self._venue_scores[vkey]

                n   = len(sc_hist)
                avg = float(np.mean(sc_hist)) if n >= MIN_H2H_MATCHES else global_mean
                wr  = float(np.mean(w_hist))  if len(w_hist) >= MIN_H2H_MATCHES else 0.5
                vn  = len(vsc_hist)
                vavg = float(np.mean(vsc_hist)) if vn >= MIN_H2H_MATCHES else avg

                self._match_h2h[(mid, batting)] = {
                    "h2h_avg_score":       round(avg, 2),
                    "h2h_win_rate":        round(wr, 4),
                    "h2h_n_matches":       n,
                    "h2h_venue_avg_score": round(vavg, 2),
                }

            # --- UPDATE H2H history AFTER match ---
            # First innings: batting = t1, score known
            if score is not None:
                self._scores[key_bat_t1].append(score)
                self._venue_scores[vkey_bat_t1].append(score)
                global_scores.append(score)

            # Win/loss record (match outcome — winner is the team that won overall)
            if winner == t1:
                self._wins[key_bat_t1].append(1)  # t1 won
                self._wins[key_bat_t2].append(0)  # t2 lost (as batting team)
            elif winner == t2:
                self._wins[key_bat_t1].append(0)
                self._wins[key_bat_t2].append(1)

        logger.info(
            f"[H2H] Fit complete. H2H records for {len(self._scores)} team pairs."
        )
        return self

    def transform(self, snapshots_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add H2H features to snapshots DataFrame.

        Requires snapshots_df to have: match_id, batting_team
        Requires matches_df to have: match_id, team1, team2
        """
        team1_map = matches_df.set_index("match_id")["team1"].to_dict()
        team2_map = matches_df.set_index("match_id")["team2"].to_dict()

        h2h_rows = []
        for _, row in snapshots_df[["match_id", "batting_team"]].iterrows():
            mid  = row["match_id"]
            bat  = row["batting_team"]
            key  = (mid, bat)
            rec  = self._match_h2h.get(key, None)
            if rec is None:
                rec = {
                    "h2h_avg_score":       260.0,
                    "h2h_win_rate":        0.5,
                    "h2h_n_matches":       0,
                    "h2h_venue_avg_score": 260.0,
                }
            h2h_rows.append(rec)

        h2h_df = pd.DataFrame(h2h_rows, index=snapshots_df.index)
        result = pd.concat([snapshots_df.reset_index(drop=True), h2h_df.reset_index(drop=True)], axis=1)

        logger.info(
            f"[H2H] Transform complete. Added 4 H2H features. "
            f"H2H known for {(h2h_df['h2h_n_matches'] > 0).sum()} / {len(h2h_df)} snapshots."
        )
        return result


# ---------------------------------------------------------------------------
# 2. Toss Advantage Computer
# ---------------------------------------------------------------------------

class TossAdvantageComputer:
    """
    Computes toss-related advantage features in a temporally safe manner.

    Features (3):
      toss_won_by_batting_team : 1 if batting team won the toss, else 0
      venue_bat_first_win_rate : historical P(batting-first team wins) at this venue
      toss_advantage_index     : toss_won * (venue_bat_first_win_rate - 0.5) * 2
                                 = +1 when toss won AND venue strongly favours bat-first
                                 = 0 when venue is neutral
                                 = -1 when toss won AND venue strongly favours chasing

    The toss advantage index is zero-centred and interpretable as a signed measure
    of how much the toss benefits the batting team at this specific venue.
    """

    def __init__(self):
        # Keyed by normalised venue string
        self._venue_bat_first_count: dict[str, int] = defaultdict(int)
        self._venue_total_count:     dict[str, int] = defaultdict(int)

        # Per-match pre-computed toss stats
        self._match_toss: dict = {}   # match_id -> dict of feature values

    @staticmethod
    def _norm_venue(v: str) -> str:
        """Normalise venue string for consistent lookup."""
        return str(v).lower().strip().replace(",", "").replace(".", "")

    def fit(self, matches_df: pd.DataFrame) -> "TossAdvantageComputer":
        """
        Process matches chronologically, pre-compute venue bat-first win rates.

        Parameters
        ----------
        matches_df : must have match_id, date, team1, team2, toss_winner,
                     toss_decision, winner, venue
        """
        df = matches_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "team1", "team2"]).sort_values("date").reset_index(drop=True)

        for _, row in df.iterrows():
            mid  = row["match_id"]
            t1   = str(row.get("team1", ""))
            t2   = str(row.get("team2", ""))
            winner = str(row.get("winner", "")) if not pd.isna(row.get("winner", np.nan)) else ""
            toss_w = str(row.get("toss_winner", "")) if not pd.isna(row.get("toss_winner", np.nan)) else ""
            toss_d = str(row.get("toss_decision", "")) if not pd.isna(row.get("toss_decision", np.nan)) else ""
            venue  = self._norm_venue(str(row.get("venue", "")))

            # Determine who batted first
            if toss_d in ("bat", "batting"):
                batting_first = toss_w
            elif toss_d in ("field", "bowling"):
                batting_first = t2 if toss_w == t1 else t1
            else:
                # Unknown toss decision — skip toss features for this match
                batting_first = ""

            # --- RECORD pre-match venue stats ---
            n   = self._venue_total_count[venue]
            cnt = self._venue_bat_first_count[venue]
            if n >= MIN_VENUE_TOSS_MATCHES:
                venue_bfw = cnt / n
            else:
                venue_bfw = 0.5   # neutral prior

            # For each team (as potential batting-first team):
            for bat_team in [t1, t2]:
                toss_won_by_bat = 1 if (toss_w == bat_team) else 0
                # Toss advantage: signed benefit for batting team
                tai = toss_won_by_bat * (venue_bfw - 0.5) * 2.0

                self._match_toss[(mid, bat_team)] = {
                    "toss_won_by_batting_team": toss_won_by_bat,
                    "venue_bat_first_win_rate": round(venue_bfw, 4),
                    "toss_advantage_index":     round(tai, 4),
                }

            # --- UPDATE venue stats AFTER match ---
            if batting_first and winner:
                self._venue_total_count[venue] += 1
                if winner == batting_first:
                    self._venue_bat_first_count[venue] += 1

        logger.info(
            f"[Toss] Fit complete. {len(self._venue_total_count)} unique venues tracked."
        )
        return self

    def transform(self, snapshots_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add toss advantage features to snapshots DataFrame.

        Requires snapshots_df to have: match_id, batting_team
        """
        toss_rows = []
        for _, row in snapshots_df[["match_id", "batting_team"]].iterrows():
            key = (row["match_id"], row["batting_team"])
            rec = self._match_toss.get(key, None)
            if rec is None:
                rec = {
                    "toss_won_by_batting_team": 0,
                    "venue_bat_first_win_rate": 0.5,
                    "toss_advantage_index":     0.0,
                }
            toss_rows.append(rec)

        toss_df = pd.DataFrame(toss_rows, index=snapshots_df.index)
        result  = pd.concat([snapshots_df.reset_index(drop=True), toss_df.reset_index(drop=True)], axis=1)

        logger.info(
            f"[Toss] Transform complete. Added 3 toss features. "
            f"Bat-first win rate mean: {toss_df['venue_bat_first_win_rate'].mean():.3f}"
        )
        return result


# ---------------------------------------------------------------------------
# 3. Match Context V3 (importance + month)
# ---------------------------------------------------------------------------

def add_match_context_v3(
    snapshots_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add match importance and seasonal features to snapshots.

    New columns:
      match_importance : float in {1.0, 1.5, 2.0} — tournament significance
      match_month      : int 1–12 — captures seasonal pitch conditions

    These are match-level features (constant across all snapshots from same match).
    """
    # Build match-level context
    ctx = matches_df[["match_id"]].copy()

    # Match importance from event name
    if "event" in matches_df.columns:
        ctx["match_importance"] = matches_df["event"].apply(_importance)
    else:
        ctx["match_importance"] = 1.0

    # Match month from date
    if "date" in matches_df.columns:
        ctx["match_month"] = (
            pd.to_datetime(matches_df["date"], errors="coerce").dt.month
        )
    else:
        ctx["match_month"] = 6  # neutral fallback

    ctx = ctx.drop_duplicates("match_id")

    result = snapshots_df.merge(ctx, on="match_id", how="left")
    result["match_importance"] = result["match_importance"].fillna(1.0)
    result["match_month"]      = result["match_month"].fillna(6).astype(int)

    logger.info(
        f"[V3 Context] Added match_importance and match_month. "
        f"Importance distribution: {result['match_importance'].value_counts().to_dict()}"
    )
    return result


# ---------------------------------------------------------------------------
# 4. Opposition Bowling Strength (from player features)
# ---------------------------------------------------------------------------

def add_bowling_attack_strength(
    snapshots_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Compute a rolling bowling-attack-strength feature:
      bowling_attack_economy : rolling mean economy rate of the bowling team's
                               top bowlers over the last `window` matches.

    This supplements bowling_team_elo with a more direct performance metric.
    Note: We use team-level economy (from player_features.batting_team_avg_score_5
    is already the batting team's metric; we need the bowling team's economy).

    If 'bowling_team_avg_economy_5' is already in snapshots (from player_features),
    this function is a no-op and just returns the input unchanged.
    """
    if "bowling_team_avg_economy_5" in snapshots_df.columns:
        logger.info("[BowlingStrength] 'bowling_team_avg_economy_5' already present; skipping.")
        return snapshots_df
    logger.info("[BowlingStrength] Feature not found; would need PlayerFeatureComputer refit.")
    return snapshots_df


# ---------------------------------------------------------------------------
# 5. Pinball Loss (for quantile model evaluation)
# ---------------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Pinball (quantile) loss for a single quantile level.

    L_q(y, f) = (y - f) * q        if y >= f
              = (f - y) * (1 - q)  if y < f

    Lower is better. For q=0.5 this equals half the MAE.
    """
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, quantile * err, (quantile - 1) * err)))


def pinball_loss_table(
    quantile_models: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Compute pinball loss for each quantile model and summarise into a table.

    Parameters
    ----------
    quantile_models : dict quantile (float) -> fitted LGBMRegressor
    X_test          : test features
    y_test          : test ground truth
    feature_columns : feature list for column selection

    Returns
    -------
    DataFrame: quantile | pinball_loss | observed_coverage | calibration_error
    """
    X = X_test[feature_columns].values
    rows = []

    for q, model in sorted(quantile_models.items()):
        preds      = model.predict(X)
        pb         = pinball_loss(y_test, preds, q)
        obs_cov    = float(np.mean(y_test <= preds))
        cal_err    = abs(obs_cov - q)

        rows.append({
            "quantile":           q,
            "pinball_loss":       round(pb, 4),
            "observed_coverage":  round(obs_cov, 4),
            "calibration_error":  round(cal_err, 4),
        })

    df = pd.DataFrame(rows)
    ece = float(df["calibration_error"].mean())
    df["ece"] = round(ece, 4)

    logger.info(f"Pinball loss table computed. ECE = {ece:.4f}")
    return df


# ---------------------------------------------------------------------------
# Feature column definitions for V3
# ---------------------------------------------------------------------------

# New features added in V3 (beyond V2's 46 features)
FEATURE_COLUMNS_V3_NEW = [
    # Head-to-head
    "h2h_avg_score",
    "h2h_win_rate",
    "h2h_n_matches",
    "h2h_venue_avg_score",
    # Toss advantage
    "toss_won_by_batting_team",
    "venue_bat_first_win_rate",
    "toss_advantage_index",
    # Match context
    "match_importance",
    "match_month",
    # Note: toss_bat_first was already in V2 — venue_bat_first_win_rate is new
]
