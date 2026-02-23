"""
Feature Engineering Module
Prepares final feature matrices for ML models.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Core features used by ML models (22 features)
FEATURE_COLUMNS = [
    "overs_completed",
    "overs_remaining",
    "current_score",
    "wickets_fallen",
    "wickets_in_hand",
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
    "innings_progress",
    "year",
    "toss_bat_first",
    "resource_pct_dls",
    "dls_predicted_final",
]

TARGET_COLUMN = "final_total"

# Features available before DLS columns are added
BASE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("resource_pct_dls", "dls_predicted_final")]


def encode_categorical_features(df: pd.DataFrame, fit_encoders: dict = None) -> tuple:
    """
    Encode categorical features (batting_team, venue) using label encoding.
    Returns (df_encoded, encoders_dict)
    """
    encoders = fit_encoders or {}
    df = df.copy()

    for col in ["batting_team", "venue"]:
        if col not in df.columns:
            continue

        df[col] = df[col].fillna("Unknown").astype(str)

        if col in encoders:
            le = encoders[col]
            # Handle unseen labels
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "Unknown")
            if "Unknown" not in known:
                le.classes_ = np.append(le.classes_, "Unknown")
            df[f"{col}_encoded"] = le.transform(df[col])
        else:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list = None,
    include_dls: bool = True,
) -> tuple:
    """
    Prepare feature matrix X and target vector y.
    Returns (X, y, feature_names)
    """
    if feature_cols is None:
        if include_dls:
            feature_cols = FEATURE_COLUMNS
        else:
            feature_cols = BASE_FEATURE_COLUMNS

    # Only use columns that exist in the DataFrame
    available_features = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        logger.warning(f"Missing features (will be skipped): {missing}")

    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()

    # Fill NaN values
    X = X.fillna(0)

    # Ensure numeric types
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    logger.info(f"Feature matrix: {X.shape}, Target: {y.shape}")
    return X, y, list(X.columns)


def add_dls_features(df: pd.DataFrame, dls_model) -> pd.DataFrame:
    """
    Add DLS resource percentage and DLS predicted final score to DataFrame.
    Requires a fitted DLS model (from src.dls_method).
    """
    df = df.copy()

    df["resource_pct_dls"] = df.apply(
        lambda row: dls_model.resource_remaining(
            overs_remaining=row["overs_remaining"],
            wickets_fallen=int(row["wickets_fallen"]),
        ),
        axis=1,
    )

    df["dls_predicted_final"] = df.apply(
        lambda row: dls_model.predict_final_score(
            current_score=row["current_score"],
            overs_completed=row["overs_completed"],
            overs_remaining=row["overs_remaining"],
            wickets_fallen=int(row["wickets_fallen"]),
        ),
        axis=1,
    )

    return df


def get_phase_mask(df: pd.DataFrame, phase: str, overs_limit: int = 50) -> pd.Series:
    """Get boolean mask for a particular match phase."""
    if overs_limit == 50:
        phases = {
            "early": (1, 10),
            "middle": (11, 30),
            "late": (31, 40),
            "death": (41, 50),
        }
    else:  # T20
        phases = {
            "early": (1, 6),
            "middle": (7, 12),
            "late": (13, 16),
            "death": (17, 20),
        }

    lo, hi = phases[phase]
    return (df["overs_completed"] >= lo) & (df["overs_completed"] <= hi)


def get_wicket_state_mask(df: pd.DataFrame, state: str) -> pd.Series:
    """Get boolean mask for wicket state groupings."""
    states = {
        "0-2": (0, 2),
        "3-5": (3, 5),
        "6-8": (6, 8),
        "9": (9, 9),
    }
    lo, hi = states[state]
    return (df["wickets_fallen"] >= lo) & (df["wickets_fallen"] <= hi)


def load_prepared_data(format_key: str) -> tuple:
    """Load saved train/test parquet and prepare features."""
    train_df = pd.read_parquet(PROCESSED_DIR / f"{format_key}_train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / f"{format_key}_test.parquet")
    return train_df, test_df
