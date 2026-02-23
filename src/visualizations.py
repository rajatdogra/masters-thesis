"""
Visualization Module
All plotting functions for the thesis.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Consistent styling
COLORS = {
    "DLS": "#e74c3c",
    "XGBoost": "#3498db",
    "RandomForest": "#2ecc71",
    "LightGBM": "#9b59b6",
    "NeuralNetwork": "#f39c12",
}

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_dls_resource_curves(dls_model, format_key: str = "mens_odi"):
    """Figure 1: DLS resource curves (fitted vs theoretical)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    overs_range = np.arange(0, dls_model.overs_limit + 1, 0.5)

    # Left: Resource remaining curves by wickets
    ax = axes[0]
    for w in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        resources = [dls_model.resource_remaining(u, w) for u in overs_range]
        ax.plot(overs_range, resources, label=f"{w} wkts lost", linewidth=1.5)

    ax.set_xlabel("Overs Remaining")
    ax.set_ylabel("Resources Remaining (%)")
    ax.set_title("DLS Resource Curves (Fitted)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: Z0 and b parameter values
    ax = axes[1]
    wickets = range(10)
    ax.bar([w - 0.2 for w in wickets], dls_model.Z0, width=0.4, label="Z₀(w)", color="#3498db")
    ax2 = ax.twinx()
    ax2.bar([w + 0.2 for w in wickets], dls_model.b, width=0.4, label="b(w)", color="#e74c3c")
    ax.set_xlabel("Wickets Lost")
    ax.set_ylabel("Z₀ (max runs)", color="#3498db")
    ax2.set_ylabel("b (decay rate)", color="#e74c3c")
    ax.set_title("Fitted DLS Parameters")
    ax.set_xticks(range(10))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_dls_resource_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_model_comparison_bar(overall_df: pd.DataFrame, format_key: str = "mens_odi"):
    """Figure 2: Model performance comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ["RMSE", "MAE", "R2"]
    for ax, metric in zip(axes, metrics):
        colors = [COLORS.get(m, "#95a5a6") for m in overall_df["Model"]]
        bars = ax.bar(overall_df["Model"], overall_df[metric], color=colors)
        ax.set_title(metric, fontsize=14)
        ax.set_ylabel(metric)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f"{height:.2f}", ha="center", va="bottom", fontsize=10
            )

    plt.suptitle(f"Model Performance Comparison ({format_key})", fontsize=16, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_model_comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_error_distributions(
    y_true: np.ndarray,
    predictions: dict,
    format_key: str = "mens_odi",
):
    """Figure 3: Prediction error distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (name, y_pred) in enumerate(predictions.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        errors = np.asarray(y_true) - np.asarray(y_pred)
        color = COLORS.get(name, "#95a5a6")

        ax.hist(errors, bins=50, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{name}\nMean Error: {errors.mean():.1f}, Std: {errors.std():.1f}")
        ax.set_xlabel("Prediction Error (Actual - Predicted)")
        ax.set_ylabel("Count")

    # Hide unused axes
    for j in range(len(predictions), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Prediction Error Distributions ({format_key})", fontsize=16, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_error_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_phase_rmse(phase_df: pd.DataFrame, format_key: str = "mens_odi"):
    """Figure 4: Phase-wise RMSE line plots."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = phase_df["Model"].unique()
    phases = phase_df["Subset"].unique()

    for model_name in models:
        model_data = phase_df[phase_df["Model"] == model_name]
        color = COLORS.get(model_name, "#95a5a6")
        ax.plot(
            model_data["Subset"], model_data["RMSE"],
            marker="o", linewidth=2, markersize=8,
            label=model_name, color=color,
        )

    ax.set_xlabel("Match Phase")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Phase-wise RMSE Comparison ({format_key})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_phase_rmse.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_cross_format_heatmap(heatmap_df: pd.DataFrame):
    """Figure 7: Cross-format RMSE heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        heatmap_df, annot=True, fmt=".1f", cmap="YlOrRd",
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Cross-Format RMSE Comparison", fontsize=14)
    ax.set_ylabel("Model")
    ax.set_xlabel("Format")
    plt.tight_layout()
    path = FIGURES_DIR / "cross_format_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    predictions: dict,
    format_key: str = "mens_odi",
):
    """Figure 8: Actual vs Predicted scatter plots."""
    n_models = len(predictions)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        color = COLORS.get(name, "#95a5a6")
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=color)

        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")

        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Actual vs Predicted ({format_key})", fontsize=16, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_residuals(
    y_true: np.ndarray,
    predictions: dict,
    format_key: str = "mens_odi",
):
    """Figure 9: Residual plots."""
    n_models = len(predictions)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        residuals = np.asarray(y_true) - np.asarray(y_pred)
        color = COLORS.get(name, "#95a5a6")
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, color=color)
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Residual Plots ({format_key})", fontsize=16, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_residuals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_learning_curves(history_dict: dict, format_key: str = "mens_odi"):
    """Figure 10: Learning curves for Neural Network."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "NeuralNetwork" in history_dict:
        history = history_dict["NeuralNetwork"]
        if hasattr(history, "history"):
            h = history.history

            ax = axes[0]
            ax.plot(h.get("loss", []), label="Train Loss")
            ax.plot(h.get("val_loss", []), label="Val Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.set_title("Neural Network - Loss Curves")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.plot(h.get("mae", []), label="Train MAE")
            ax.plot(h.get("val_mae", []), label="Val MAE")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MAE")
            ax.set_title("Neural Network - MAE Curves")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Learning Curves ({format_key})", fontsize=14, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_eda_score_distribution(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, format_key: str):
    """EDA plots: score distributions, run rates, etc."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # First innings totals
    first_inn = deliveries_df[deliveries_df["innings"] == 1]
    totals = first_inn.groupby("match_id")["total_runs"].sum()

    ax = axes[0, 0]
    ax.hist(totals, bins=40, color="#3498db", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("First Innings Total")
    ax.set_ylabel("Frequency")
    ax.set_title(f"First Innings Score Distribution ({format_key})")
    ax.axvline(totals.mean(), color="red", linestyle="--", label=f"Mean: {totals.mean():.0f}")
    ax.legend()

    # Matches per year
    ax = axes[0, 1]
    if "date" in matches_df.columns:
        years = pd.to_datetime(matches_df["date"], errors="coerce").dt.year
        year_counts = years.value_counts().sort_index()
        ax.bar(year_counts.index, year_counts.values, color="#2ecc71", alpha=0.7)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Matches")
        ax.set_title("Matches per Year")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Average run rate per over
    ax = axes[1, 0]
    over_runs = first_inn.groupby("over")["total_runs"].mean()
    ax.bar(over_runs.index, over_runs.values, color="#9b59b6", alpha=0.7)
    ax.set_xlabel("Over Number")
    ax.set_ylabel("Average Runs")
    ax.set_title("Average Runs per Over")

    # Wickets per over
    ax = axes[1, 1]
    over_wickets = first_inn.groupby("over")["is_wicket"].mean() * 100
    ax.bar(over_wickets.index, over_wickets.values, color="#e74c3c", alpha=0.7)
    ax.set_xlabel("Over Number")
    ax.set_ylabel("Wicket Probability (%)")
    ax.set_title("Wicket Probability per Over")

    plt.suptitle(f"Exploratory Data Analysis - {format_key}", fontsize=16, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_eda.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
