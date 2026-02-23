"""
Explainability Module
SHAP and LIME explanations for ML models.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SHAP Analysis
# ============================================================
def compute_shap_values(model, X, model_type: str = "tree", scaler=None, background_size: int = 100):
    """
    Compute SHAP values for a model.
    model_type: "tree" for XGBoost/RF/LightGBM, "kernel" for NN
    """
    import shap

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    elif model_type == "kernel":
        # For neural networks, use KernelExplainer with a background sample
        if scaler is not None:
            X_scaled = scaler.transform(X)
            bg = shap.sample(pd.DataFrame(X_scaled, columns=X.columns), background_size)
        else:
            bg = shap.sample(X, background_size)

        def predict_fn(data):
            return model.predict(data, verbose=0).flatten()

        explainer = shap.KernelExplainer(predict_fn, bg)

        if scaler is not None:
            shap_values = explainer.shap_values(scaler.transform(X), nsamples=200)
        else:
            shap_values = explainer.shap_values(X, nsamples=200)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return shap_values, explainer


def plot_shap_summary(
    shap_values,
    X: pd.DataFrame,
    model_name: str,
    format_key: str = "mens_odi",
    max_display: int = 15,
):
    """Create SHAP summary (beeswarm) plot."""
    import shap

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Summary - {model_name} ({format_key})", fontsize=14)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_shap_summary_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot: {path}")


def plot_shap_bar(
    shap_values,
    X: pd.DataFrame,
    model_name: str,
    format_key: str = "mens_odi",
    max_display: int = 15,
):
    """Create SHAP feature importance bar plot."""
    import shap

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Feature Importance - {model_name} ({format_key})", fontsize=14)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_shap_bar_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP bar plot: {path}")


def plot_shap_dependence(
    shap_values,
    X: pd.DataFrame,
    feature: str,
    model_name: str,
    format_key: str = "mens_odi",
    interaction_feature: str = "auto",
):
    """Create SHAP dependence plot for a specific feature."""
    import shap

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature, shap_values, X,
        interaction_index=interaction_feature,
        show=False,
        ax=ax,
    )
    plt.title(f"SHAP Dependence: {feature} - {model_name}", fontsize=14)
    plt.tight_layout()
    safe_name = feature.replace(" ", "_").lower()
    path = FIGURES_DIR / f"{format_key}_shap_dep_{safe_name}_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP dependence plot: {path}")


def plot_shap_waterfall(
    shap_values,
    X: pd.DataFrame,
    explainer,
    instance_idx: int,
    model_name: str,
    format_key: str = "mens_odi",
):
    """Create SHAP waterfall plot for a single prediction."""
    import shap

    fig, ax = plt.subplots(figsize=(10, 8))

    if hasattr(explainer, "expected_value"):
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
    else:
        base_value = 0

    explanation = shap.Explanation(
        values=shap_values[instance_idx],
        base_values=base_value,
        data=X.iloc[instance_idx].values,
        feature_names=X.columns.tolist(),
    )

    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP Waterfall (instance {instance_idx}) - {model_name}", fontsize=12)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_shap_waterfall_{instance_idx}_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP waterfall plot: {path}")


def full_shap_analysis(
    model,
    X: pd.DataFrame,
    model_name: str,
    model_type: str = "tree",
    scaler=None,
    format_key: str = "mens_odi",
    top_features: int = 5,
    sample_size: int = 1000,
):
    """
    Run complete SHAP analysis for a model.
    Returns shap_values for further use.
    """
    # Sample data if too large
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    logger.info(f"Computing SHAP values for {model_name} ({len(X_sample)} samples)...")
    shap_values, explainer = compute_shap_values(
        model, X_sample, model_type=model_type, scaler=scaler
    )

    # Summary plots
    plot_shap_summary(shap_values, X_sample, model_name, format_key=format_key)
    plot_shap_bar(shap_values, X_sample, model_name, format_key=format_key)

    # Dependence plots for top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_features]
    top_feature_names = [X_sample.columns[i] for i in top_indices]

    for feat in top_feature_names:
        plot_shap_dependence(shap_values, X_sample, feat, model_name, format_key=format_key)

    # Waterfall plots for a few instances
    for idx in [0, len(X_sample) // 2, len(X_sample) - 1]:
        plot_shap_waterfall(shap_values, X_sample, explainer, idx, model_name, format_key=format_key)

    return shap_values, explainer


# ============================================================
# LIME Analysis
# ============================================================
def lime_explain_instance(
    model,
    X_train: pd.DataFrame,
    X_instance: pd.Series,
    model_name: str,
    instance_idx: int = 0,
    scaler=None,
    format_key: str = "mens_odi",
):
    """Generate LIME explanation for a single instance."""
    import lime
    import lime.lime_tabular

    feature_names = X_train.columns.tolist()

    if scaler is not None:
        X_train_vals = scaler.transform(X_train)
        X_instance_vals = scaler.transform(X_instance.values.reshape(1, -1)).flatten()

        def predict_fn(data):
            return model.predict(data, verbose=0).flatten()
    else:
        X_train_vals = X_train.values
        X_instance_vals = X_instance.values

        def predict_fn(data):
            return model.predict(data)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_vals,
        feature_names=feature_names,
        mode="regression",
        verbose=False,
    )

    explanation = explainer.explain_instance(
        X_instance_vals,
        predict_fn,
        num_features=15,
    )

    # Save as figure
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(12, 8)
    plt.title(f"LIME Explanation (instance {instance_idx}) - {model_name}", fontsize=14)
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_lime_{instance_idx}_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved LIME plot: {path}")

    # Save HTML report
    html_path = FIGURES_DIR / f"{format_key}_lime_{instance_idx}_{model_name.lower().replace(' ', '_')}.html"
    explanation.save_to_file(str(html_path))
    logger.info(f"Saved LIME HTML: {html_path}")

    return explanation


def full_lime_analysis(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    scaler=None,
    format_key: str = "mens_odi",
    n_instances: int = 5,
):
    """
    Run LIME analysis on diverse sample predictions.
    Selects instances from different score ranges and over states.
    """
    explanations = []

    # Select diverse instances
    test_sorted = X_test.copy()
    test_sorted["_y"] = y_test.values
    test_sorted["_idx"] = range(len(test_sorted))

    # Pick from different score percentiles
    percentiles = np.linspace(10, 90, n_instances).astype(int)
    score_thresholds = np.percentile(test_sorted["_y"], percentiles)

    selected_indices = []
    for threshold in score_thresholds:
        closest = (test_sorted["_y"] - threshold).abs().idxmin()
        orig_idx = test_sorted.loc[closest, "_idx"]
        if orig_idx not in selected_indices:
            selected_indices.append(int(orig_idx))

    # Ensure we have enough
    while len(selected_indices) < n_instances:
        selected_indices.append(np.random.randint(0, len(X_test)))
    selected_indices = selected_indices[:n_instances]

    for i, idx in enumerate(selected_indices):
        instance = X_test.iloc[idx]
        exp = lime_explain_instance(
            model, X_train, instance,
            model_name=model_name,
            instance_idx=i,
            scaler=scaler,
            format_key=format_key,
        )
        explanations.append(exp)

    return explanations


# ============================================================
# Cross-Model Feature Importance
# ============================================================
def cross_model_feature_importance(
    shap_results: dict,
    feature_names: list,
    format_key: str = "mens_odi",
    top_n: int = 15,
):
    """
    Create cross-model feature importance comparison chart.
    shap_results: dict of model_name -> shap_values array
    """
    importance_df = pd.DataFrame()

    for model_name, shap_values in shap_results.items():
        mean_abs = np.abs(shap_values).mean(axis=0)
        # Normalize to [0, 1]
        if mean_abs.max() > 0:
            mean_abs = mean_abs / mean_abs.max()
        importance_df[model_name] = pd.Series(mean_abs, index=feature_names[:len(mean_abs)])

    # Sort by average importance
    importance_df["avg"] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values("avg", ascending=True).tail(top_n)
    importance_df = importance_df.drop(columns=["avg"])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    importance_df.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel("Normalized Feature Importance (SHAP)", fontsize=12)
    ax.set_title(f"Cross-Model Feature Importance Comparison ({format_key})", fontsize=14)
    ax.legend(title="Model", loc="lower right")
    plt.tight_layout()
    path = FIGURES_DIR / f"{format_key}_cross_model_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cross-model importance: {path}")

    return importance_df
