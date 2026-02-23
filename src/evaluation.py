"""
Evaluation Module
Computes metrics and comparisons between DLS and ML models.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove NaN / inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {k: np.nan for k in [
            "RMSE", "MAE", "R2", "MAPE",
            "Within_5", "Within_10", "Within_20", "N"
        ]}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan

    # Percentage within X runs
    abs_errors = np.abs(y_true - y_pred)
    within_5 = (abs_errors <= 5).mean() * 100
    within_10 = (abs_errors <= 10).mean() * 100
    within_20 = (abs_errors <= 20).mean() * 100

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "Within_5": round(within_5, 2),
        "Within_10": round(within_10, 2),
        "Within_20": round(within_20, 2),
        "N": len(y_true),
    }


def compare_models(
    y_true: np.ndarray,
    predictions: dict,
    label: str = "Overall",
) -> pd.DataFrame:
    """
    Compare multiple models.
    predictions: dict of model_name -> y_pred array
    Returns DataFrame with metrics per model.
    """
    rows = []
    for name, y_pred in predictions.items():
        metrics = compute_metrics(y_true, y_pred)
        metrics["Model"] = name
        metrics["Subset"] = label
        rows.append(metrics)

    df = pd.DataFrame(rows)
    cols = ["Model", "Subset", "RMSE", "MAE", "R2", "MAPE", "Within_5", "Within_10", "Within_20", "N"]
    return df[[c for c in cols if c in df.columns]]


def phase_wise_comparison(
    test_df: pd.DataFrame,
    predictions: dict,
    overs_limit: int = 50,
) -> pd.DataFrame:
    """
    Compare models across different match phases.
    predictions: dict of model_name -> full prediction array (aligned with test_df)
    """
    if overs_limit == 50:
        phases = {
            "Early (1-10)": (1, 10),
            "Middle (11-30)": (11, 30),
            "Late (31-40)": (31, 40),
            "Death (41-50)": (41, 50),
        }
    else:
        phases = {
            "Powerplay (1-6)": (1, 6),
            "Middle (7-12)": (7, 12),
            "Late (13-16)": (13, 16),
            "Death (17-20)": (17, 20),
        }

    all_rows = []
    y_true = test_df["final_total"].values

    for phase_name, (lo, hi) in phases.items():
        mask = (test_df["overs_completed"].values >= lo) & (test_df["overs_completed"].values <= hi)

        if mask.sum() == 0:
            continue

        phase_true = y_true[mask]
        phase_preds = {name: pred[mask] for name, pred in predictions.items()}
        df = compare_models(phase_true, phase_preds, label=phase_name)
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def wicket_state_comparison(
    test_df: pd.DataFrame,
    predictions: dict,
) -> pd.DataFrame:
    """Compare models across different wicket states."""
    states = {
        "0-2 wickets": (0, 2),
        "3-5 wickets": (3, 5),
        "6-8 wickets": (6, 8),
        "9 wickets": (9, 9),
    }

    all_rows = []
    y_true = test_df["final_total"].values

    for state_name, (lo, hi) in states.items():
        mask = (
            (test_df["wickets_fallen"].values >= lo)
            & (test_df["wickets_fallen"].values <= hi)
        )

        if mask.sum() == 0:
            continue

        state_true = y_true[mask]
        state_preds = {name: pred[mask] for name, pred in predictions.items()}
        df = compare_models(state_true, state_preds, label=state_name)
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def full_evaluation(
    test_df: pd.DataFrame,
    predictions: dict,
    format_key: str = "mens_odi",
    overs_limit: int = 50,
) -> dict:
    """
    Run complete evaluation suite.
    Returns dict with overall, phase-wise, and wicket-state comparison DataFrames.
    """
    y_true = test_df["final_total"].values

    # Overall
    overall_df = compare_models(y_true, predictions, label="Overall")
    logger.info(f"\n{'='*60}\nOverall Results ({format_key}):\n{'='*60}")
    logger.info(f"\n{overall_df.to_string(index=False)}")

    # Phase-wise
    phase_df = phase_wise_comparison(test_df, predictions, overs_limit=overs_limit)
    logger.info(f"\n{'='*60}\nPhase-wise Results:\n{'='*60}")
    logger.info(f"\n{phase_df.to_string(index=False)}")

    # Wicket-state
    wicket_df = wicket_state_comparison(test_df, predictions)
    logger.info(f"\n{'='*60}\nWicket-state Results:\n{'='*60}")
    logger.info(f"\n{wicket_df.to_string(index=False)}")

    # Save tables
    overall_df.to_csv(TABLES_DIR / f"{format_key}_overall_comparison.csv", index=False)
    phase_df.to_csv(TABLES_DIR / f"{format_key}_phase_comparison.csv", index=False)
    wicket_df.to_csv(TABLES_DIR / f"{format_key}_wicket_comparison.csv", index=False)

    logger.info(f"Tables saved to {TABLES_DIR}")

    return {
        "overall": overall_df,
        "phase_wise": phase_df,
        "wicket_state": wicket_df,
    }


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate a LaTeX table from a DataFrame."""
    n_cols = len(df.columns)
    col_format = "l" + "r" * (n_cols - 1)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule",
        " & ".join(df.columns) + " \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def cross_format_heatmap_data(format_results: dict) -> pd.DataFrame:
    """
    Create DataFrame for cross-format RMSE heatmap.
    format_results: dict of format_key -> overall comparison DataFrame
    """
    rows = []
    for fmt, df in format_results.items():
        for _, row in df.iterrows():
            rows.append({
                "Format": fmt,
                "Model": row["Model"],
                "RMSE": row["RMSE"],
            })

    pivot = pd.DataFrame(rows).pivot(index="Model", columns="Format", values="RMSE")
    return pivot
