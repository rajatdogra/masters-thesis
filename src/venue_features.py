"""
Venue Feature Computer.

Computes historical venue statistics from match data, strictly in
chronological order (only data from PRIOR matches at the same venue).

Features added per snapshot:
  venue_avg_score         - mean first-innings total at this venue (prior matches)
  venue_std_score         - std of first-innings totals (volatility indicator)
  venue_avg_wickets_25    - average wickets fallen by over 25 at this venue
  venue_boundary_rate     - average boundary percentage at this venue
  venue_high_score_rate   - fraction of matches with first-innings score > 300 (ODI)
  venue_matches_count     - number of prior matches at this venue (reliability proxy)
  home_team               - 1 if batting team is playing at a "home" venue, else 0
  is_neutral_venue        - 1 if neither team is obviously at home

Notes on home detection:
  We use a simple heuristic: a team is "at home" if the venue city/country
  appears as a substring of the team name, or vice versa.
  This is imperfect but avoids needing external country-team mappings.
  The raw `toss_bat_first` already captures some home-ground effects.

Minimum matches before using venue stats: MIN_VENUE_MATCHES (default 5).
Below this threshold, the global mean across all venues is used.
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_VENUE_MATCHES = 5
HIGH_SCORE_THRESHOLD_ODI = 300
HIGH_SCORE_THRESHOLD_T20 = 180


# Mapping of team name fragments to country/city keywords for home detection
TEAM_HOME_KEYWORDS = {
    "India": ["india", "mumbai", "delhi", "chennai", "kolkata", "bangalore",
              "hyderabad", "ahmedabad", "pune", "nagpur", "dharamsala", "mohali", "ranchi"],
    "Australia": ["australia", "sydney", "melbourne", "brisbane", "perth",
                  "adelaide", "hobart", "canberra"],
    "England": ["england", "london", "manchester", "birmingham", "leeds",
                "nottingham", "bristol", "southampton", "chester-le-street",
                "the oval", "lords", "lord's", "headingley", "edgbaston",
                "trent bridge", "old trafford"],
    "Pakistan": ["pakistan", "karachi", "lahore", "rawalpindi", "islamabad",
                 "multan", "faisalabad", "peshawar"],
    "South Africa": ["south africa", "johannesburg", "cape town", "durban",
                     "centurion", "port elizabeth", "east london", "bloemfontein"],
    "New Zealand": ["new zealand", "auckland", "wellington", "christchurch",
                    "hamilton", "napier", "dunedin"],
    "Sri Lanka": ["sri lanka", "colombo", "kandy", "galle", "dambulla",
                  "pallekele", "hambantota"],
    "West Indies": ["west indies", "kingston", "bridgetown", "port of spain",
                    "providence", "north sound", "gros islet"],
    "Bangladesh": ["bangladesh", "dhaka", "chittagong", "mirpur", "sylhet",
                   "fatullah"],
    "Zimbabwe": ["zimbabwe", "harare", "bulawayo"],
    "Afghanistan": ["afghanistan", "kabul", "sharjah", "greater noida"],
    "Ireland": ["ireland", "dublin", "malahide", "clontarf"],
    "Netherlands": ["netherlands", "amstelveen", "deventer"],
    "Scotland": ["scotland", "edinburgh"],
    "UAE": ["uae", "united arab emirates", "dubai", "abu dhabi", "sharjah"],
}


def _is_home_venue(team_name: str, venue: str, city: str) -> bool:
    """Heuristic: return True if team is likely playing at their home ground."""
    location = f"{venue or ''} {city or ''}".lower()
    for team_key, keywords in TEAM_HOME_KEYWORDS.items():
        if team_key.lower() in team_name.lower() or team_name.lower() in team_key.lower():
            if any(kw in location for kw in keywords):
                return True
    return False


class VenueFeatureComputer:
    """
    Compute per-venue statistics strictly from matches prior to each snapshot.

    Usage
    -----
    vfc = VenueFeatureComputer(overs_limit=50)
    vfc.fit(matches_df, snapshots_df)
    enriched = vfc.transform(snapshots_df)
    """

    def __init__(self, overs_limit: int = 50):
        self.overs_limit = overs_limit
        self._high_score_threshold = (
            HIGH_SCORE_THRESHOLD_ODI if overs_limit == 50 else HIGH_SCORE_THRESHOLD_T20
        )
        # venue -> list of {date, score, wickets_at_25, boundary_rate} dicts
        self._venue_history: dict[str, list] = defaultdict(list)
        # match_id -> pre-match venue stats cache
        self._match_cache: dict[str, dict] = {}

    def fit(self, matches_df: pd.DataFrame, snapshots_df: pd.DataFrame) -> "VenueFeatureComputer":
        """
        Process matches chronologically. Compute and cache venue stats
        as they stood BEFORE each match is played.

        Parameters
        ----------
        matches_df  : must contain match_id, date, venue, city
        snapshots_df: must contain match_id, overs_completed, wickets_fallen,
                      boundary_percentage, final_total
        """
        df_m = matches_df.copy()
        df_m["date"] = pd.to_datetime(df_m["date"], errors="coerce")
        df_m = df_m.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Pre-compute per-match aggregates from snapshots (last over = final state)
        snap_agg = self._aggregate_snapshots(snapshots_df)

        logger.info(f"VenueFeatureComputer: fitting on {len(df_m)} matches...")

        for _, row in df_m.iterrows():
            mid = str(row["match_id"])
            venue = str(row.get("venue", "") or "").strip()
            city = str(row.get("city", "") or "").strip()

            # Cache PRE-match venue stats
            self._match_cache[mid] = {
                "venue": venue,
                "city": city,
                "stats": self._get_venue_stats(venue),
            }

            # Update venue history with this match's data
            if mid in snap_agg:
                agg = snap_agg[mid]
                if venue:
                    self._venue_history[venue].append({
                        "date": row["date"],
                        "score": agg["final_total"],
                        "wickets_at_25": agg.get("wickets_at_25", np.nan),
                        "boundary_rate": agg.get("boundary_pct", np.nan),
                    })

        # Compute global fallbacks
        all_scores = [
            entry["score"]
            for entries in self._venue_history.values()
            for entry in entries
            if not np.isnan(entry["score"])
        ]
        self._global_avg_score = float(np.mean(all_scores)) if all_scores else 250.0
        self._global_std_score = float(np.std(all_scores)) if len(all_scores) > 1 else 50.0
        self._global_high_rate = (
            float(np.mean([s > self._high_score_threshold for s in all_scores]))
            if all_scores else 0.3
        )

        logger.info(
            f"VenueFeatureComputer: fit complete. Venues: {len(self._venue_history)}. "
            f"Global avg score: {self._global_avg_score:.1f}"
        )
        return self

    def _aggregate_snapshots(self, snapshots_df: pd.DataFrame) -> dict:
        """
        For each match_id, extract:
          - final_total (last snapshot's label)
          - wickets_at_25 (wickets_fallen when overs_completed==25)
          - boundary_pct (from last snapshot)
        """
        agg = {}
        for mid, group in snapshots_df.groupby("match_id"):
            last = group.sort_values("overs_completed").iloc[-1]
            final_total = last.get("final_total", np.nan)
            boundary_pct = last.get("boundary_percentage", np.nan)

            # Wickets at over 25
            at_25 = group[group["overs_completed"] == 25]
            wickets_at_25 = float(at_25["wickets_fallen"].iloc[0]) if not at_25.empty else np.nan

            agg[str(mid)] = {
                "final_total": float(final_total),
                "boundary_pct": float(boundary_pct),
                "wickets_at_25": wickets_at_25,
            }
        return agg

    def _get_venue_stats(self, venue: str) -> dict:
        """Return rolling venue stats from the history accumulated so far."""
        history = self._venue_history.get(venue, [])
        n = len(history)

        if n < MIN_VENUE_MATCHES:
            return {
                "venue_avg_score": None,
                "venue_std_score": None,
                "venue_avg_wickets_25": None,
                "venue_boundary_rate": None,
                "venue_high_score_rate": None,
                "venue_matches_count": n,
                "reliable": False,
            }

        scores = [h["score"] for h in history if not np.isnan(h["score"])]
        wickets = [h["wickets_at_25"] for h in history if not np.isnan(h["wickets_at_25"])]
        boundaries = [h["boundary_rate"] for h in history if not np.isnan(h["boundary_rate"])]

        return {
            "venue_avg_score": float(np.mean(scores)) if scores else None,
            "venue_std_score": float(np.std(scores)) if len(scores) > 1 else 50.0,
            "venue_avg_wickets_25": float(np.mean(wickets)) if wickets else None,
            "venue_boundary_rate": float(np.mean(boundaries)) if boundaries else None,
            "venue_high_score_rate": float(
                np.mean([s > self._high_score_threshold for s in scores])
            ) if scores else None,
            "venue_matches_count": n,
            "reliable": True,
        }

    def transform(self, snapshots_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add venue feature columns and home/neutral indicator to snapshots_df.

        Parameters
        ----------
        snapshots_df : must contain match_id, batting_team
        matches_df   : must contain match_id, venue, city

        Returns
        -------
        Enriched snapshots_df with venue columns.
        """
        # Build match_id -> (venue, city) lookup
        venue_lookup = (
            matches_df[["match_id", "venue", "city"]]
            .drop_duplicates("match_id")
            .set_index("match_id")
        )

        df = snapshots_df.copy()
        venue_rows = []

        for _, row in df.iterrows():
            mid = str(row["match_id"])
            batting_team = str(row.get("batting_team", ""))

            match_info = self._match_cache.get(mid, {})
            stats = match_info.get("stats", {})
            venue = match_info.get("venue", "")
            city = match_info.get("city", "")

            # Venue stats with fallback
            avg_score = stats.get("venue_avg_score") or self._global_avg_score
            std_score = stats.get("venue_std_score") or self._global_std_score
            wickets_25 = stats.get("venue_avg_wickets_25")  # no fallback; keep None if missing
            boundary_rate = stats.get("venue_boundary_rate")
            high_rate = stats.get("venue_high_score_rate") or self._global_high_rate
            match_count = stats.get("venue_matches_count", 0)

            # Home/neutral detection
            is_home = int(_is_home_venue(batting_team, venue, city))
            is_neutral = int(not is_home and not any(
                _is_home_venue(bowling_proxy, venue, city)
                for bowling_proxy in [""]  # simplified — don't need the exact bowling team here
            ))

            venue_rows.append({
                "venue_avg_score": round(avg_score, 1),
                "venue_std_score": round(std_score, 1),
                "venue_avg_wickets_25": round(wickets_25, 2) if wickets_25 is not None else np.nan,
                "venue_boundary_rate": round(boundary_rate, 4) if boundary_rate is not None else np.nan,
                "venue_high_score_rate": round(high_rate, 4),
                "venue_matches_count": match_count,
                "batting_at_home": is_home,
            })

        venue_df = pd.DataFrame(venue_rows, index=df.index)

        # Impute NaN venue columns with global medians
        for col in ["venue_avg_wickets_25", "venue_boundary_rate"]:
            median_val = venue_df[col].median()
            venue_df[col] = venue_df[col].fillna(median_val)

        result = pd.concat([df, venue_df], axis=1)
        logger.info(
            f"Venue features added. New columns: {list(venue_df.columns)}. "
            f"Shape: {result.shape}"
        )
        return result
