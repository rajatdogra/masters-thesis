"""
Build V3 Snapshots — extends V2 parquets with advanced features.

New features added (V3 = V2 + 9 new):
  H2H:  h2h_avg_score, h2h_win_rate, h2h_n_matches, h2h_venue_avg_score
  Toss: toss_won_by_batting_team, venue_bat_first_win_rate, toss_advantage_index
  Ctx:  match_importance, match_month

Steps
-----
1. Load raw matches_df (includes DL matches for maximum H2H history)
2. Load cleaned V2 snapshots (already feature-engineered)
3. Compute first-innings totals for H2H score tracking
4. Fit HeadToHeadComputer and TossAdvantageComputer on ALL matches (chronologically)
5. Transform each V2 split (train, cal, test) with new features
6. Save as *_v3.parquet

Runtime: ~5-10 minutes
"""

import sys, logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_collection import load_processed
from src.advanced_features import (
    HeadToHeadComputer,
    TossAdvantageComputer,
    add_match_context_v3,
    FEATURE_COLUMNS_V3_NEW,
)
from src.pipeline import FEATURE_COLUMNS_V2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("build_v3")

PROC = ROOT / "data" / "processed"

# V3 feature list = V2 + new
FEATURE_COLUMNS_V3 = FEATURE_COLUMNS_V2 + FEATURE_COLUMNS_V3_NEW
TARGET_COLUMN = "final_total"

log.info("=" * 60)
log.info("BUILDING V3 SNAPSHOTS (V2 + H2H + Toss + Context features)")
log.info("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. Load raw data
# ──────────────────────────────────────────────────────────────
log.info("Loading raw processed data...")
matches_raw, deliveries_raw = load_processed("mens_odi")
log.info(f"  matches_raw: {len(matches_raw)} rows")

# ──────────────────────────────────────────────────────────────
# 2. Load V2 snapshots
# ──────────────────────────────────────────────────────────────
log.info("Loading V2 snapshots...")
train_v2 = pd.read_parquet(PROC / "mens_odi_train_v2.parquet")
cal_v2   = pd.read_parquet(PROC / "mens_odi_cal_v2.parquet")
test_v2  = pd.read_parquet(PROC / "mens_odi_test_v2.parquet")
snaps_v2 = pd.read_parquet(PROC / "mens_odi_snapshots_v2.parquet")

log.info(f"  train: {len(train_v2)} | cal: {len(cal_v2)} | test: {len(test_v2)}")

# ──────────────────────────────────────────────────────────────
# 3. Compute first-innings totals for H2H score tracking
#    (use ALL matches including DL for maximum history)
# ──────────────────────────────────────────────────────────────
log.info("Computing first-innings totals for H2H tracking...")
first_inn = deliveries_raw[deliveries_raw["innings"] == 1].copy()
totals = first_inn.groupby("match_id")["total_runs"].sum().reset_index()
totals.columns = ["match_id", "final_total"]
log.info(f"  First-innings totals: {len(totals)} matches")

# ──────────────────────────────────────────────────────────────
# 4. Fit advanced feature computers on ALL matches
# ──────────────────────────────────────────────────────────────
log.info("Fitting HeadToHeadComputer (all matches, chronological)...")
h2h = HeadToHeadComputer()
h2h.fit(matches_raw, first_innings_totals=totals)

log.info("Fitting TossAdvantageComputer (all matches, chronological)...")
toss = TossAdvantageComputer()
toss.fit(matches_raw)

# ──────────────────────────────────────────────────────────────
# 5. Transform each split
# ──────────────────────────────────────────────────────────────

def apply_v3_features(df, split_name=""):
    log.info(f"  [{split_name}] Applying H2H features...")
    df = h2h.transform(df, matches_raw)
    log.info(f"  [{split_name}] Applying Toss features...")
    df = toss.transform(df)
    log.info(f"  [{split_name}] Applying match context V3 (importance + month)...")
    df = add_match_context_v3(df, matches_raw)

    # Fill any remaining NaNs in new feature columns
    for col in FEATURE_COLUMNS_V3_NEW:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


log.info("Transforming full snapshot set...")
snaps_v3 = apply_v3_features(snaps_v2.copy(), "snapshots")

log.info("Transforming train split...")
train_v3 = apply_v3_features(train_v2.copy(), "train")

log.info("Transforming cal split...")
cal_v3   = apply_v3_features(cal_v2.copy(), "cal")

log.info("Transforming test split...")
test_v3  = apply_v3_features(test_v2.copy(), "test")

# ──────────────────────────────────────────────────────────────
# 6. Validate and save
# ──────────────────────────────────────────────────────────────
log.info("Validating new features...")
for col in FEATURE_COLUMNS_V3_NEW:
    if col in test_v3.columns:
        null_pct = test_v3[col].isnull().mean() * 100
        log.info(f"  {col:35s}: null={null_pct:.1f}%  mean={test_v3[col].mean():.3f}")
    else:
        log.warning(f"  {col} NOT FOUND in test_v3!")

log.info("Saving V3 parquets...")
snaps_v3.to_parquet(PROC / "mens_odi_snapshots_v3.parquet", index=False)
train_v3.to_parquet(PROC / "mens_odi_train_v3.parquet", index=False)
cal_v3.to_parquet(PROC / "mens_odi_cal_v3.parquet", index=False)
test_v3.to_parquet(PROC / "mens_odi_test_v3.parquet", index=False)

log.info(f"V3 snapshots: {len(snaps_v3)} rows × {len(snaps_v3.columns)} cols")
log.info(f"V3 features: {len([c for c in FEATURE_COLUMNS_V3 if c in snaps_v3.columns])}")
log.info(f"Saved to: {PROC}")
log.info("=" * 60)
log.info("DONE — V3 snapshots ready for training.")
log.info("Next step: python scripts/train_v3_models.py")
log.info("=" * 60)

# Also save the V3 feature list for reference
import json
(ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
with open(ROOT / "results" / "metrics" / "feature_columns_v3.json", "w") as f:
    json.dump({
        "v2_features": FEATURE_COLUMNS_V2,
        "v3_new_features": FEATURE_COLUMNS_V3_NEW,
        "v3_all_features": FEATURE_COLUMNS_V3,
        "n_v2": len(FEATURE_COLUMNS_V2),
        "n_v3": len(FEATURE_COLUMNS_V3),
    }, f, indent=2)

log.info(f"V3 feature list saved to results/metrics/feature_columns_v3.json")
