"""
Data Collection Module
Downloads and parses Cricsheet JSON data for ODI and T20I matches.
Supports: Men's ODI, Women's ODI, Men's T20I, Women's T20I
"""

import json
import os
import zipfile
import io
import glob
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cricsheet download URLs (JSON format)
CRICSHEET_URLS = {
    "mens_odi": "https://cricsheet.org/downloads/all_male_json.zip",
    "womens_odi": "https://cricsheet.org/downloads/odis_female_json.zip",
    "mens_t20i": "https://cricsheet.org/downloads/t20s_male_json.zip",
    "womens_t20i": "https://cricsheet.org/downloads/t20s_female_json.zip",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def download_and_extract(format_key: str, force: bool = False) -> Path:
    """Download and extract cricsheet JSON zip for a given format."""
    url = CRICSHEET_URLS[format_key]
    extract_dir = RAW_DIR / format_key

    if extract_dir.exists() and not force:
        json_files = list(extract_dir.glob("*.json"))
        if len(json_files) > 10:
            logger.info(f"[{format_key}] Already have {len(json_files)} JSON files. Skipping download.")
            return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{format_key}] Downloading from {url}...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    data = io.BytesIO()
    with tqdm(total=total_size, unit="B", unit_scale=True, desc=format_key) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            data.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"[{format_key}] Extracting ZIP...")
    data.seek(0)
    with zipfile.ZipFile(data) as zf:
        zf.extractall(extract_dir)

    json_files = list(extract_dir.glob("*.json"))
    logger.info(f"[{format_key}] Extracted {len(json_files)} JSON files.")
    return extract_dir


def parse_single_match(filepath: str) -> tuple:
    """
    Parse a single Cricsheet JSON file into match info and delivery records.
    Returns (match_dict, list_of_delivery_dicts) or (None, None) if parsing fails.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return None, None

    info = data.get("info", {})
    match_id = Path(filepath).stem

    # Extract match-level information
    outcome = info.get("outcome", {})
    winner = outcome.get("winner", None)
    margin_runs = outcome.get("by", {}).get("runs", None)
    margin_wickets = outcome.get("by", {}).get("wickets", None)
    method = outcome.get("method", None)  # "D/L" or None
    result = info.get("outcome", {}).get("result", None)  # "no result", "tie", etc.

    toss = info.get("toss", {})
    teams = info.get("teams", [])
    dates = info.get("dates", [])

    match_dict = {
        "match_id": match_id,
        "date": dates[0] if dates else None,
        "venue": info.get("venue", None),
        "city": info.get("city", None),
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner", None),
        "toss_decision": toss.get("decision", None),
        "winner": winner,
        "margin_runs": margin_runs,
        "margin_wickets": margin_wickets,
        "method": method,
        "result": result,
        "match_type": info.get("match_type", None),
        "gender": info.get("gender", None),
        "overs_limit": info.get("overs", None),
        "player_of_match": (info.get("player_of_match") or [None])[0],
    }

    # Parse innings deliveries
    delivery_records = []
    innings_data = data.get("innings", [])

    for inn_idx, innings in enumerate(innings_data):
        innings_num = inn_idx + 1
        team = innings.get("team", "Unknown")
        overs_data = innings.get("overs", [])

        for over_obj in overs_data:
            over_num = over_obj.get("over", 0)  # 0-indexed over number
            deliveries = over_obj.get("deliveries", [])

            for ball_idx, delivery in enumerate(deliveries):
                runs_info = delivery.get("runs", {})
                batter_runs = runs_info.get("batter", 0)
                extras_runs = runs_info.get("extras", 0)
                total_runs = runs_info.get("total", 0)

                # Wicket info
                wickets = delivery.get("wickets", [])
                is_wicket = len(wickets) > 0
                dismissal_kind = wickets[0].get("kind", None) if is_wicket else None
                player_out = wickets[0].get("player_out", None) if is_wicket else None

                # Extras breakdown
                extras_detail = delivery.get("extras", {})

                # Determine if this is a legal delivery (not a wide or no-ball for ball counting)
                is_wide = "wides" in extras_detail
                is_noball = "noballs" in extras_detail

                delivery_records.append({
                    "match_id": match_id,
                    "innings": innings_num,
                    "batting_team": team,
                    "over": over_num,
                    "ball": ball_idx,
                    "batter": delivery.get("batter", None),
                    "bowler": delivery.get("bowler", None),
                    "non_striker": delivery.get("non_striker", None),
                    "batter_runs": batter_runs,
                    "extras_runs": extras_runs,
                    "total_runs": total_runs,
                    "is_wicket": is_wicket,
                    "dismissal_kind": dismissal_kind,
                    "player_out": player_out,
                    "is_wide": is_wide,
                    "is_noball": is_noball,
                    "is_boundary_four": batter_runs == 4,
                    "is_boundary_six": batter_runs == 6,
                })

    return match_dict, delivery_records


def parse_all_matches(format_key: str) -> tuple:
    """Parse all JSON files for a format. Returns (matches_df, deliveries_df)."""
    raw_dir = RAW_DIR / format_key
    json_files = sorted(raw_dir.glob("*.json"))

    # Filter out non-match files (like README or info files)
    json_files = [f for f in json_files if f.stem.isdigit() or f.stem.replace("_", "").isdigit()]

    if not json_files:
        # Try broader matching - cricsheet sometimes uses different naming
        json_files = [f for f in sorted(raw_dir.glob("*.json"))
                      if f.stem not in ("README", "info", "people")]

    logger.info(f"[{format_key}] Parsing {len(json_files)} match files...")

    all_matches = []
    all_deliveries = []

    for filepath in tqdm(json_files, desc=f"Parsing {format_key}"):
        match_dict, delivery_records = parse_single_match(str(filepath))
        if match_dict is not None:
            all_matches.append(match_dict)
        if delivery_records:
            all_deliveries.extend(delivery_records)

    matches_df = pd.DataFrame(all_matches)
    deliveries_df = pd.DataFrame(all_deliveries)

    if not matches_df.empty:
        matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")

    logger.info(
        f"[{format_key}] Parsed {len(matches_df)} matches, "
        f"{len(deliveries_df)} deliveries."
    )
    return matches_df, deliveries_df


def save_to_parquet(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, format_key: str):
    """Save DataFrames to parquet files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches_path = PROCESSED_DIR / f"{format_key}_matches.parquet"
    deliveries_path = PROCESSED_DIR / f"{format_key}_deliveries.parquet"

    matches_df.to_parquet(matches_path, index=False)
    deliveries_df.to_parquet(deliveries_path, index=False)

    logger.info(
        f"[{format_key}] Saved to {matches_path.name} ({len(matches_df)} rows) "
        f"and {deliveries_path.name} ({len(deliveries_df)} rows)"
    )


def collect_format(format_key: str, force_download: bool = False):
    """Full pipeline: download, parse, save for one format."""
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {format_key}")
    logger.info(f"{'='*60}")

    download_and_extract(format_key, force=force_download)
    matches_df, deliveries_df = parse_all_matches(format_key)
    save_to_parquet(matches_df, deliveries_df, format_key)
    return matches_df, deliveries_df


def collect_all(formats: list = None, force_download: bool = False) -> dict:
    """
    Collect data for all specified formats.
    Returns dict mapping format_key -> (matches_df, deliveries_df)
    """
    if formats is None:
        formats = list(CRICSHEET_URLS.keys())

    results = {}
    for fmt in formats:
        matches_df, deliveries_df = collect_format(fmt, force_download=force_download)
        results[fmt] = (matches_df, deliveries_df)

    return results


def load_processed(format_key: str) -> tuple:
    """Load already-processed parquet files."""
    matches_path = PROCESSED_DIR / f"{format_key}_matches.parquet"
    deliveries_path = PROCESSED_DIR / f"{format_key}_deliveries.parquet"

    if not matches_path.exists() or not deliveries_path.exists():
        raise FileNotFoundError(
            f"Processed data for '{format_key}' not found. Run collect_format() first."
        )

    matches_df = pd.read_parquet(matches_path)
    deliveries_df = pd.read_parquet(deliveries_path)
    return matches_df, deliveries_df


def parse_dl_targets(format_key: str) -> pd.DataFrame:
    """
    Re-scan raw JSON files for D/L-affected matches and extract the
    official revised target set by the umpires.

    Returns a DataFrame with one row per DL match:
      match_id, date, venue, team1, team2, winner,
      first_innings_total,          - Team 1's actual score
      dl_target_runs,               - Official DL revised target
      dl_target_overs,              - Overs allocated to Team 2
      inn2_score_at_end,            - Team 2's final score
      inn2_overs_bowled,            - Overs bowled in Team 2 before end
      inn2_wickets_at_end,          - Wickets fallen at end of Team 2 innings

    These matches are excluded from model training but used for
    evaluating the revised target system.
    """
    raw_dir = RAW_DIR / format_key
    json_files = [f for f in sorted(raw_dir.glob("*.json"))
                  if f.stem not in ("README", "info", "people")]

    records = []
    for filepath in json_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        info = data.get("info", {})
        outcome = info.get("outcome", {})
        method = outcome.get("method", None)

        if method != "D/L":
            continue

        match_id = filepath.stem
        innings = data.get("innings", [])

        # First innings total
        first_innings_total = 0
        if len(innings) > 0:
            for over_obj in innings[0].get("overs", []):
                for delivery in over_obj.get("deliveries", []):
                    first_innings_total += delivery.get("runs", {}).get("total", 0)

        # Second innings DL target
        dl_target_runs = None
        dl_target_overs = None
        inn2_score = 0
        inn2_overs = 0
        inn2_wickets = 0

        if len(innings) > 1:
            inn2 = innings[1]
            target_info = inn2.get("target", {})
            dl_target_runs = target_info.get("runs", None)
            dl_target_overs = target_info.get("overs", None)

            # Compute second innings actual state
            for over_obj in inn2.get("overs", []):
                over_num = over_obj.get("over", 0)
                for delivery in over_obj.get("deliveries", []):
                    inn2_score += delivery.get("runs", {}).get("total", 0)
                    if delivery.get("wickets"):
                        inn2_wickets += len(delivery["wickets"])
                inn2_overs = over_num + 1  # count of overs completed

        dates = info.get("dates", [])
        teams = info.get("teams", [])

        records.append({
            "match_id": match_id,
            "date": dates[0] if dates else None,
            "venue": info.get("venue", None),
            "city": info.get("city", None),
            "team1": teams[0] if len(teams) > 0 else None,
            "team2": teams[1] if len(teams) > 1 else None,
            "winner": outcome.get("winner", None),
            "first_innings_total": first_innings_total,
            "dl_target_runs": dl_target_runs,
            "dl_target_overs": dl_target_overs,
            "inn2_score_at_end": inn2_score,
            "inn2_overs_bowled": inn2_overs,
            "inn2_wickets_at_end": inn2_wickets,
        })

    dl_df = pd.DataFrame(records)
    if not dl_df.empty:
        dl_df["date"] = pd.to_datetime(dl_df["date"], errors="coerce")
        dl_df = dl_df.dropna(subset=["dl_target_runs"]).reset_index(drop=True)

    logger.info(
        f"[{format_key}] Found {len(dl_df)} D/L matches with valid revised targets."
    )

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{format_key}_dl_matches.parquet"
    dl_df.to_parquet(out_path, index=False)
    logger.info(f"[{format_key}] Saved DL matches to {out_path.name}")

    return dl_df


if __name__ == "__main__":
    # Default: collect Men's ODI first, then others
    collect_all(["mens_odi"])
