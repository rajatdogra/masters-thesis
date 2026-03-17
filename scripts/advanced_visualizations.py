"""
Advanced thesis-quality visualizations from stored metrics.
Generates 6 publication-ready figures missing from current thesis.

Run: python scripts/advanced_visualizations.py
Output: results/figures/{walk_forward_bands, phase_radar, ablation_tornado,
         conformal_calibration, model_hierarchy, concept_drift_enhanced}.png
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ─── Palette ────────────────────────────────────────────────────────────────
C = {
    "xgb":  "#F97316",   # orange
    "lgb":  "#3B82F6",   # blue
    "cat":  "#10B981",   # green
    "dls":  "#94A3B8",   # slate-grey
    "stk":  "#8B5CF6",   # purple
    "v1":   "#CBD5E1",   # light grey
    "pos":  "#EF4444",   # red (positive delta)
    "neg":  "#22C55E",   # green (negative delta)
}
FONT = {"family": "sans-serif", "sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]}
plt.rcParams.update({"font.family": FONT["family"], "font.sans-serif": FONT["sans-serif"],
                     "axes.spines.top": False, "axes.spines.right": False})

WHITE = "#FFFFFF"
LIGHT = "#F8FAFC"
DARK  = "#1E293B"

# ═══════════════════════════════════════════════════════════════════════════
# 1. WALK-FORWARD CV — per-fold annotated figure
# ═══════════════════════════════════════════════════════════════════════════
def fig_walk_forward():
    df = pd.read_csv(METRICS / "walk_forward_cv.csv")
    with open(METRICS / "walk_forward_summary.json") as f:
        summary = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=WHITE)

    folds = df["fold"].astype(str).tolist()
    periods = df["val_start"].str[:7] + "–" + df["val_end"].str[:7]
    x = np.arange(len(folds))
    w = 0.35

    # Left: RMSE per fold
    ax = axes[0]
    ax.set_facecolor(LIGHT)
    bars_lgb = ax.bar(x - w/2, df["lgb_rmse"], w, color=C["lgb"], alpha=0.88,
                      label="LightGBM V2", zorder=3)
    bars_dls = ax.bar(x + w/2, df["dls_rmse"], w, color=C["dls"], alpha=0.88,
                      label="DLS Baseline", zorder=3)

    # Annotate delta on top of LGB bar
    for i, (lgb, dls) in enumerate(zip(df["lgb_rmse"], df["dls_rmse"])):
        delta = lgb - dls
        ax.text(i, lgb + 0.6, f"Δ{delta:+.1f}", ha="center", va="bottom",
                fontsize=7.5, color=C["lgb"], fontweight="bold")

    # Mean lines
    lgb_mean = summary["lgb_rmse_mean"]
    dls_mean = summary["dls_rmse_mean"]
    ax.axhline(lgb_mean, color=C["lgb"], lw=1.4, ls="--", alpha=0.7, zorder=4)
    ax.axhline(dls_mean, color=C["dls"], lw=1.4, ls="--", alpha=0.7, zorder=4)
    ax.text(len(folds) - 0.45, lgb_mean + 0.8, f"μ={lgb_mean:.1f}", fontsize=7.5,
            color=C["lgb"], fontweight="bold")
    ax.text(len(folds) - 0.45, dls_mean + 0.8, f"μ={dls_mean:.1f}", fontsize=7.5,
            color=C["dls"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}\n{p}" for f, p in zip(folds, periods)], fontsize=7.5)
    ax.set_ylabel("RMSE (runs)", fontsize=10, color=DARK)
    ax.set_title("(a) Per-fold RMSE: LightGBM vs DLS", fontsize=11, fontweight="bold",
                 color=DARK, pad=8)
    ax.set_ylim(0, 82)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Right: Δ RMSE (improvement) per fold
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT)
    deltas = df["lgb_vs_dls_delta_rmse"].values
    bar_colors = [C["lgb"]] * len(deltas)
    bars = ax2.bar(x, np.abs(deltas), color=bar_colors, alpha=0.85, zorder=3)

    # Error bars from std
    lgb_std = summary["lgb_rmse_std"]
    dls_std = summary["dls_rmse_std"]
    combined_std = np.sqrt(lgb_std**2 + dls_std**2)
    ax2.errorbar(x, np.abs(deltas), yerr=combined_std, fmt="none",
                 color=DARK, capsize=4, lw=1.3, zorder=5)

    mean_delta = abs(summary["delta_rmse_mean"])
    delta_std  = summary["delta_rmse_std"]
    ax2.axhline(mean_delta, color="#EF4444", lw=1.8, ls="--", zorder=4)
    ax2.fill_between([-0.5, len(folds) - 0.5],
                     mean_delta - delta_std, mean_delta + delta_std,
                     color="#EF4444", alpha=0.12, zorder=2)
    ax2.text(len(folds) - 0.45, mean_delta + 0.3,
             f"μ={mean_delta:.1f}±{delta_std:.1f}", fontsize=7.5,
             color="#EF4444", fontweight="bold")

    for bar, d in zip(bars, np.abs(deltas)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{d:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color=DARK)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Fold {f}" for f in folds], fontsize=8.5)
    ax2.set_ylabel("|RMSE Improvement| (runs)", fontsize=10, color=DARK)
    ax2.set_title("(b) ML improvement per fold (ΔRMSE vs DLS)", fontsize=11,
                  fontweight="bold", color=DARK, pad=8)
    ax2.set_ylim(0, 32)
    ax2.grid(axis="y", lw=0.6, alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    plt.suptitle("Walk-Forward Temporal Cross-Validation (K=5 expanding folds)",
                 fontsize=12.5, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    out = FIGS / "walk_forward_bands.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ walk_forward_bands.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ABLATION TORNADO CHART
# ═══════════════════════════════════════════════════════════════════════════
def fig_ablation_tornado():
    df = pd.read_csv(METRICS / "ablation_study.csv")
    df = df.sort_values("delta_rmse", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=WHITE)
    ax.set_facecolor(LIGHT)

    colors = [C["pos"] if d > 0 else C["neg"] for d in df["delta_rmse"]]
    bars = ax.barh(df["group"], df["delta_rmse"], color=colors, alpha=0.85,
                   height=0.55, zorder=3)

    for bar, val, pct in zip(bars, df["delta_rmse"], df["pct_change"]):
        x_pos = bar.get_width()
        ha = "left" if x_pos >= 0 else "right"
        offset = 0.02 if x_pos >= 0 else -0.02
        ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f"+{val:.2f} (+{pct:.1f}%)" if val > 0 else f"{val:.2f} ({pct:.1f}%)",
                va="center", ha=ha, fontsize=8.5, fontweight="bold" if abs(val) > 0.5 else "normal",
                color=DARK)

    ax.axvline(0, color=DARK, lw=1.2, zorder=4)
    ax.set_xlabel("ΔRMSE when feature group removed (runs)", fontsize=10, color=DARK)
    ax.set_title("Feature Group Ablation Study — LightGBM V2\n"
                 "(Positive = group improves predictions; Negative = group hurts)",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)

    # Legend patches
    pos_p = mpatches.Patch(color=C["pos"], alpha=0.85, label="Improves RMSE (group is useful)")
    neg_p = mpatches.Patch(color=C["neg"], alpha=0.85, label="Worsens RMSE (group is noise)")
    ax.legend(handles=[pos_p, neg_p], fontsize=8.5, loc="lower right", framealpha=0.9)

    ax.grid(axis="x", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, 2.2)

    # Annotation for top feature
    ax.annotate("Player stats:\n+3.4% RMSE — most important",
                xy=(1.50, 6), xytext=(1.55, 5.4),
                fontsize=7.5, color=C["pos"],
                arrowprops=dict(arrowstyle="->", color=C["pos"], lw=0.9))

    plt.tight_layout()
    out = FIGS / "ablation_tornado.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ ablation_tornado.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONFORMAL CALIBRATION DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════
def fig_conformal_calibration():
    df = pd.read_csv(METRICS / "conformal_coverage.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), facecolor=WHITE)

    # Left: calibration curve
    ax = axes[0]
    ax.set_facecolor(LIGHT)
    nominal = df["nominal_coverage"].values
    empirical = df["empirical_coverage"].values

    # Perfect calibration reference
    ax.plot([0.7, 1.0], [0.7, 1.0], "--", color=DARK, lw=1.4, alpha=0.6,
            label="Perfect calibration", zorder=2)
    ax.fill_between([0.7, 1.0], [0.7, 1.0], [0.65, 0.95], alpha=0.08,
                    color="#EF4444", label="Under-coverage zone")

    # Actual points
    ax.plot(nominal, empirical, "o-", color=C["lgb"], lw=2.2, ms=9,
            zorder=4, label="Split-conformal (MAPIE)", markeredgewidth=1.5,
            markeredgecolor=WHITE)

    # Annotate each point
    alphas = [1 - n for n in nominal]
    for nom, emp, alpha in zip(nominal, empirical, alphas):
        gap = emp - nom
        ax.annotate(f"α={alpha:.2f}\n({emp:.3f} vs {nom:.2f})\ngap={gap:+.3f}",
                    xy=(nom, emp), xytext=(nom - 0.04, emp - 0.022),
                    fontsize=7, color=DARK,
                    arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.8))

    ax.set_xlabel("Nominal coverage (1 − α)", fontsize=10, color=DARK)
    ax.set_ylabel("Empirical coverage", fontsize=10, color=DARK)
    ax.set_title("(a) Conformal Calibration Curve\n(split-conformal, MAPIE)", fontsize=11,
                 fontweight="bold", color=DARK, pad=8)
    ax.set_xlim(0.72, 1.02); ax.set_ylim(0.68, 1.02)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(lw=0.6, alpha=0.4, zorder=0); ax.set_axisbelow(True)

    # Right: interval width vs alpha
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT)
    widths = df["avg_interval_width"].values
    gap = df["coverage_gap"].values

    ax2_twin = ax2.twinx()
    ax2.bar(alphas, widths, width=0.04, color=C["lgb"], alpha=0.8, zorder=3,
            label="Interval width (runs)")
    ax2_twin.plot(alphas, np.abs(gap), "s--", color=C["pos"], lw=1.8, ms=8,
                  label="|Coverage gap|", zorder=4)

    for a, w, g in zip(alphas, widths, gap):
        ax2.text(a, w + 1.5, f"{w:.0f}r", ha="center", fontsize=8.5,
                 fontweight="bold", color=C["lgb"])
        ax2_twin.text(a, abs(g) + 0.001, f"{g:+.3f}", ha="center", fontsize=7.5,
                      color=C["pos"])

    ax2.set_xlabel("Significance level α", fontsize=10, color=DARK)
    ax2.set_ylabel("Mean interval width (runs)", fontsize=10, color=C["lgb"])
    ax2_twin.set_ylabel("|Empirical − Nominal| coverage", fontsize=9, color=C["pos"])
    ax2.set_title("(b) Interval Width and Coverage Gap by α",
                  fontsize=11, fontweight="bold", color=DARK, pad=8)
    ax2.set_ylim(0, 195); ax2_twin.set_ylim(0, 0.12)
    ax2.set_xticks(alphas)
    ax2.set_xticklabels([f"α={a:.2f}" for a in alphas])

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right",
               framealpha=0.9)
    ax2.grid(axis="y", lw=0.6, alpha=0.4, zorder=0); ax2.set_axisbelow(True)
    ax2_twin.spines["top"].set_visible(False)

    plt.suptitle("Conformal Prediction Interval Diagnostics",
                 fontsize=12.5, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    out = FIGS / "conformal_calibration.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ conformal_calibration.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL HIERARCHY OVERVIEW — multi-metric, all architectures
# ═══════════════════════════════════════════════════════════════════════════
def fig_model_hierarchy():
    v2   = pd.read_csv(METRICS / "inn1_v2_results.csv")
    stk  = pd.read_csv(METRICS / "stacking_results.csv")
    base = pd.read_csv(METRICS / "baseline_comparison.csv")
    mono = pd.read_csv(METRICS / "monotonic_results.csv")

    # Unify label columns
    for df, col in [(v2, "model"), (stk, "model"), (base, "model"), (mono, "model")]:
        if col not in df.columns and "Model" in df.columns:
            df.rename(columns={"Model": "model"}, inplace=True)

    # Build a consolidated table of RMSE / R² / MAE
    rows = []
    def add(label, cat, rmse, r2, mae):
        rows.append({"label": label, "cat": cat, "rmse": rmse, "r2": r2, "mae": mae})

    # DLS baseline
    dls_row = v2[v2["model"].str.contains("DLS", case=False)].iloc[0]
    add("DLS Baseline", "baseline", dls_row["rmse"], dls_row["r2"], dls_row["mae"])

    # OLS / Ridge from base
    for _, row in base.iterrows():
        m = str(row["model"])
        if "OLS" in m or "Linear" in m:
            add("OLS (46 features)", "linear", row["rmse"], row["r2"], row["mae"]); break
    for _, row in base.iterrows():
        m = str(row["model"])
        if "Poly" in m:
            add("Polynomial (d=2)", "linear", row["rmse"], row["r2"], row["mae"]); break

    # Individual GBMs
    for name, cat in [("XGBoost_V2", "gbm"), ("LightGBM_V2", "gbm"), ("CatBoost_V2", "gbm")]:
        r = v2[v2["model"] == name]
        if not r.empty:
            add(name.replace("_V2", " V2"), "gbm", r.iloc[0]["rmse"], r.iloc[0]["r2"], r.iloc[0]["mae"])

    # Monotonic XGBoost
    r = mono[mono["model"].str.contains("Monotonic", case=False)]
    if not r.empty:
        add("Monotonic XGBoost", "constrained", r.iloc[0]["rmse"], r.iloc[0]["r2"], r.iloc[0]["mae"])

    # Stacking
    r = stk[stk["model"].str.contains("Stack", case=False)]
    if not r.empty:
        add("Stacking Ensemble", "ensemble", r.iloc[0]["rmse"], r.iloc[0]["r2"], r.iloc[0]["mae"])

    data = pd.DataFrame(rows).sort_values("rmse", ascending=False)

    cat_colors = {
        "baseline":   C["dls"],
        "linear":     "#64748B",
        "gbm":        C["lgb"],
        "constrained": C["xgb"],
        "ensemble":   C["stk"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor=WHITE)
    metrics = [("rmse", "RMSE (runs)", "(a) RMSE ↓ lower is better"),
               ("r2",   "R²",          "(b) R² ↑ higher is better"),
               ("mae",  "MAE (runs)",  "(c) MAE ↓ lower is better")]

    for ax, (col, ylabel, title) in zip(axes, metrics):
        ax.set_facecolor(LIGHT)
        colors = [cat_colors[c] for c in data["cat"]]
        bars = ax.barh(data["label"], data[col], color=colors, alpha=0.85,
                       height=0.62, zorder=3)
        for bar, val in zip(bars, data[col]):
            x = bar.get_width()
            ax.text(x + 0.003 * max(data[col]), bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}" if col == "r2" else f"{val:.2f}",
                    va="center", fontsize=7.8, color=DARK)
        ax.set_xlabel(ylabel, fontsize=9.5, color=DARK)
        ax.set_title(title, fontsize=10, fontweight="bold", color=DARK, pad=7)
        ax.grid(axis="x", lw=0.6, alpha=0.4, zorder=0); ax.set_axisbelow(True)
        if col == "rmse":
            ax.set_xlim(0, max(data[col]) * 1.18)
        elif col == "r2":
            ax.set_xlim(0, 1.0)
        else:
            ax.set_xlim(0, max(data[col]) * 1.18)

    # Legend
    legend_patches = [
        mpatches.Patch(color=cat_colors[k], alpha=0.85, label=k.capitalize())
        for k in ["baseline", "linear", "gbm", "constrained", "ensemble"]
    ]
    fig.legend(handles=legend_patches, fontsize=8.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    plt.suptitle("Model Hierarchy: Performance Across All Architectures (First Innings, Test Set)",
                 fontsize=12.5, fontweight="bold", color=DARK, y=1.02)
    plt.tight_layout()
    out = FIGS / "model_hierarchy.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ model_hierarchy.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PHASE-WISE PERFORMANCE HEATMAP + DELTA
# ═══════════════════════════════════════════════════════════════════════════
def fig_phase_heatmap():
    df = pd.read_csv(METRICS / "phase_wise_metrics.csv")
    models_order = ["LightGBM_V1", "XGBoost_V2", "LightGBM_V2", "CatBoost_V2", "DLS"]
    phases = ["Early (1-10)", "Middle (11-40)", "Death (41-50)"]
    phase_labels = ["Early\n(overs 1–10)", "Middle\n(overs 11–40)", "Death\n(overs 41–50)"]

    fig = plt.figure(figsize=(12, 6), facecolor=WHITE)
    gs = GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.08, figure=fig)
    ax_heat = fig.add_subplot(gs[0])
    ax_delta = fig.add_subplot(gs[1])

    # Build RMSE matrix
    rmse_mat = np.zeros((len(models_order), len(phases)))
    r2_mat   = np.zeros_like(rmse_mat)
    for i, m in enumerate(models_order):
        for j, p in enumerate(phases):
            row = df[(df["model"] == m) & (df["phase"] == p)]
            if not row.empty:
                rmse_mat[i, j] = row.iloc[0]["rmse"]
                r2_mat[i, j]   = row.iloc[0]["r2"]

    # Heatmap — RMSE
    im = ax_heat.imshow(rmse_mat, cmap="RdYlGn_r", aspect="auto",
                        vmin=10, vmax=110)

    ax_heat.set_xticks(range(len(phases)))
    ax_heat.set_xticklabels(phase_labels, fontsize=10)
    ax_heat.set_yticks(range(len(models_order)))
    model_labels = ["LightGBM V1", "XGBoost V2", "LightGBM V2", "CatBoost V2", "DLS Baseline"]
    ax_heat.set_yticklabels(model_labels, fontsize=10)

    for i in range(len(models_order)):
        for j in range(len(phases)):
            rmse = rmse_mat[i, j]
            r2   = r2_mat[i, j]
            text_color = "white" if rmse > 70 else DARK
            ax_heat.text(j, i, f"RMSE {rmse:.1f}\nR²={r2:.3f}",
                         ha="center", va="center", fontsize=8.0, color=text_color,
                         fontweight="bold" if i in [2, 3] else "normal")

    cbar = plt.colorbar(im, ax=ax_heat, pad=0.02, fraction=0.03)
    cbar.set_label("RMSE (runs)", fontsize=9)
    ax_heat.set_title("(a) Phase-wise RMSE Heatmap\n(green = better, red = worse)",
                      fontsize=11, fontweight="bold", color=DARK, pad=8)

    # Add DLS catastrophe annotation
    ax_heat.annotate("DLS fails\ncatastrophically\n(R²=−1.008)",
                     xy=(0, 4), xytext=(0.55, 3.45),
                     fontsize=7.5, color="white", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="white", lw=1.0))

    # Right: ML vs DLS Δ per phase (LightGBM V2)
    ax_delta.set_facecolor(LIGHT)
    lgb_v2_rows = df[df["model"] == "LightGBM_V2"].set_index("phase")
    dls_rows    = df[df["model"] == "DLS"].set_index("phase")

    delta_rmse = []
    delta_r2   = []
    for p in phases:
        delta_rmse.append(dls_rows.loc[p, "rmse"] - lgb_v2_rows.loc[p, "rmse"])
        delta_r2.append(lgb_v2_rows.loc[p, "r2"] - dls_rows.loc[p, "r2"])

    y = np.arange(len(phases))
    colors_d = [C["lgb"] if d > 0 else C["pos"] for d in delta_rmse]
    bars = ax_delta.barh(y, delta_rmse, color=colors_d, alpha=0.85, height=0.5, zorder=3)
    for bar, d in zip(bars, delta_rmse):
        ax_delta.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                      f"+{d:.1f}" if d > 0 else f"{d:.1f}",
                      va="center", fontsize=9, fontweight="bold", color=DARK)

    ax_delta.axvline(0, color=DARK, lw=1.2, zorder=4)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(phase_labels, fontsize=10)
    ax_delta.set_xlabel("RMSE improvement (DLS − LightGBM V2, runs)", fontsize=9.5, color=DARK)
    ax_delta.set_title("(b) LightGBM V2 RMSE\nimprovement over DLS",
                       fontsize=11, fontweight="bold", color=DARK, pad=8)
    ax_delta.grid(axis="x", lw=0.6, alpha=0.4, zorder=0); ax_delta.set_axisbelow(True)
    ax_delta.set_xlim(-5, max(delta_rmse) * 1.18)

    plt.suptitle("Phase-Wise Performance Analysis: Match Overs 1–50",
                 fontsize=12.5, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    out = FIGS / "phase_heatmap.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ phase_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. CONCEPT DRIFT — enhanced with R² panels and trend lines
# ═══════════════════════════════════════════════════════════════════════════
def fig_concept_drift_enhanced():
    df = pd.read_csv(METRICS / "concept_drift.csv")
    with open(METRICS / "drift_trend.json") as f:
        trend = json.load(f)

    models = ["XGBoost", "LightGBM", "CatBoost", "DLS"]
    colors_m = {"XGBoost": C["xgb"], "LightGBM": C["lgb"],
                "CatBoost": C["cat"], "DLS": C["dls"]}
    markers = {"XGBoost": "s", "LightGBM": "o", "CatBoost": "^", "DLS": "D"}
    years = sorted(df["year"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=WHITE)

    for ax, (metric, ylabel, title_sfx) in zip(
        axes,
        [("rmse", "RMSE (runs)", "RMSE ↓ lower is better"),
         ("r2",   "R²",          "R² ↑ higher is better")]
    ):
        ax.set_facecolor(LIGHT)
        for m in models:
            sub = df[df["model"] == m].sort_values("year")
            vals = sub[metric].values
            yrs  = sub["year"].values
            ax.plot(yrs, vals, f"{markers[m]}-", color=colors_m[m], lw=2.0,
                    ms=7.5, label=m, alpha=0.9, markeredgecolor=WHITE,
                    markeredgewidth=1.3, zorder=4)
            # Trend line
            if len(yrs) > 1:
                z = np.polyfit(yrs, vals, 1)
                p = np.poly1d(z)
                x_ext = np.linspace(min(yrs), max(yrs), 50)
                ax.plot(x_ext, p(x_ext), "--", color=colors_m[m], lw=0.9,
                        alpha=0.5, zorder=3)
            # Pearson r annotation (last model point)
            if m in trend and len(yrs) > 0:
                r = trend[m]["pearson_r"]
                direction = "↓ improving" if r < 0 else "↑ degrading"
                ax.annotate(f"r={r:+.2f} {direction}",
                            xy=(yrs[-1], vals[-1]),
                            xytext=(yrs[-1] + 0.08, vals[-1] + (2 if metric == "rmse" else 0.01)),
                            fontsize=6.8, color=colors_m[m],
                            fontweight="bold" if m in ["CatBoost", "DLS"] else "normal")

        ax.set_xlabel("Year", fontsize=10, color=DARK)
        ax.set_ylabel(ylabel, fontsize=10, color=DARK)
        ax.set_title(f"(a) {title_sfx}" if metric == "rmse" else f"(b) {title_sfx}",
                     fontsize=11, fontweight="bold", color=DARK, pad=8)
        ax.set_xticks(years)
        ax.grid(lw=0.6, alpha=0.4, zorder=0); ax.set_axisbelow(True)
        if metric == "rmse":
            ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    plt.suptitle("Concept Drift Analysis: Model Performance Over Time (Test Period 2022–2026)\n"
                 "Dashed lines = linear trend fit; r = Pearson correlation with year",
                 fontsize=12, fontweight="bold", color=DARK, y=1.02)
    plt.tight_layout()
    out = FIGS / "concept_drift_enhanced.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ concept_drift_enhanced.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. SHAP TOP-20 IMPORTANCE + CUMULATIVE EXPLAINED VARIANCE
# ═══════════════════════════════════════════════════════════════════════════
def fig_shap_ranked():
    df = pd.read_csv(METRICS / "shap_importance_lgb_v2.csv")
    df.columns = ["feature", "shap_mean"]
    df = df.sort_values("shap_mean", ascending=False).head(20)
    df["cumsum_pct"] = df["shap_mean"].cumsum() / df["shap_mean"].sum() * 100

    # Clean feature names for display
    def clean(name):
        return (name.replace("_", " ")
                    .replace("dls predicted final", "DLS predicted final")
                    .replace("current run rate", "Current run rate")
                    .replace("elo gap", "ELO gap")
                    .replace("resource pct dls", "DLS resource %")
                    .replace("wickets fallen", "Wickets fallen")
                    .replace("partnership quality", "Partnership quality")
                    .replace("batting team elo", "Batting team ELO")
                    .replace("batter1 innings count", "Batter-1 innings count")
                    .replace("run rate vs venue", "Run rate vs venue avg")
                    .replace("batting at home", "Home advantage")
                    .title())
    df["label"] = df["feature"].apply(clean)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=WHITE,
                              gridspec_kw={"width_ratios": [1.6, 1]})

    # Left: horizontal bar ranked
    ax = axes[0]
    ax.set_facecolor(LIGHT)
    gradient_colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(df)))[::-1]
    bars = ax.barh(df["label"][::-1], df["shap_mean"][::-1], color=gradient_colors,
                   height=0.62, zorder=3, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, df["shap_mean"][::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=7.5, color=DARK)

    ax.set_xlabel("Mean |SHAP| value (runs)", fontsize=10, color=DARK)
    ax.set_title("(a) Top-20 Feature Importance (LightGBM V2)\nMean absolute SHAP contribution",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    ax.grid(axis="x", lw=0.6, alpha=0.4, zorder=0); ax.set_axisbelow(True)

    # Right: cumulative % explained
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT)
    n_feat = np.arange(1, len(df) + 1)
    ax2.plot(n_feat, df["cumsum_pct"].values, "o-", color=C["lgb"], lw=2.2,
             ms=6, zorder=4, markeredgecolor=WHITE, markeredgewidth=1.3)
    ax2.fill_between(n_feat, df["cumsum_pct"].values, alpha=0.12, color=C["lgb"])

    # Annotate 80% threshold
    idx_80 = np.argmax(df["cumsum_pct"].values >= 80)
    ax2.axhline(80, color="#EF4444", lw=1.4, ls="--", alpha=0.8, zorder=3)
    ax2.axvline(idx_80 + 1, color="#EF4444", lw=1.0, ls=":", alpha=0.7, zorder=3)
    ax2.annotate(f"80% explained\nby top-{idx_80+1} features",
                 xy=(idx_80 + 1, 80), xytext=(idx_80 + 3, 72),
                 fontsize=8, color="#EF4444", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#EF4444", lw=0.9))

    ax2.set_xlabel("Number of features (ranked by |SHAP|)", fontsize=10, color=DARK)
    ax2.set_ylabel("Cumulative explained SHAP (%)", fontsize=10, color=DARK)
    ax2.set_title("(b) Cumulative Feature Explanation\n(% of total SHAP mass)",
                  fontsize=11, fontweight="bold", color=DARK, pad=8)
    ax2.set_ylim(0, 105); ax2.set_xlim(0.5, 20.5)
    ax2.grid(lw=0.6, alpha=0.4, zorder=0); ax2.set_axisbelow(True)

    plt.suptitle("SHAP Feature Attribution Analysis — LightGBM V2 (46 features)",
                 fontsize=12.5, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    out = FIGS / "shap_ranked_cumulative.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ shap_ranked_cumulative.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating advanced thesis visualizations…\n")
    fig_walk_forward()
    fig_ablation_tornado()
    fig_conformal_calibration()
    fig_model_hierarchy()
    fig_phase_heatmap()
    fig_concept_drift_enhanced()
    fig_shap_ranked()
    print(f"\nAll figures saved to {FIGS}")
