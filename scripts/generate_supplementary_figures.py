"""
Three targeted supplementary figures that are genuinely missing from the thesis:
  1. bootstrap_distribution.png  — bootstrap CI visual for RMSE difference
  2. economic_bubble.png         — revenue vs ML improvement bubble chart
  3. rain_targets_deep.png       — DLS vs ML rain match target analysis

Run: python scripts/generate_supplementary_figures.py
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import re

ROOT   = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"
FIGS    = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
})
WHITE = "#FFFFFF"
LIGHT = "#F8FAFC"
DARK  = "#1E293B"
C_LGB = "#3B82F6"
C_DLS = "#94A3B8"
C_RED = "#EF4444"
C_GRN = "#10B981"

# ─── helper: parse numpy array stored as string ─────────────────────────────
def _parse_np_str(s: str) -> np.ndarray:
    """Convert the stored numpy repr string back to a float array."""
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return np.array([float(x) for x in nums])


# ═══════════════════════════════════════════════════════════════════════════
# 1. BOOTSTRAP CI DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
def fig_bootstrap_distribution():
    with open(METRICS / "bootstrap_ci.json") as f:
        bs = json.load(f)

    diff_boot = _parse_np_str(bs["_diff_boot"])
    ml_boot   = _parse_np_str(bs["_rmse_ml_boot"])
    dls_boot  = _parse_np_str(bs["_rmse_dls_boot"])

    obs_diff = bs["obs_diff"]
    ci_lo, ci_hi = bs["ci_diff"]
    obs_ml  = bs["obs_rmse_ml"]
    obs_dls = bs["obs_rmse_dls"]
    ci_ml   = bs["ci_ml"]
    ci_dls  = bs["ci_dls"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=WHITE)

    # ── Left: Bootstrap RMSE distributions for ML and DLS ──────────────────
    ax = axes[0]
    ax.set_facecolor(LIGHT)
    bins = 50
    ax.hist(ml_boot, bins=bins, color=C_LGB, alpha=0.72, label="LightGBM V2", density=True,
            zorder=3, edgecolor="white", linewidth=0.4)
    ax.hist(dls_boot, bins=bins, color=C_DLS, alpha=0.72, label="DLS Baseline", density=True,
            zorder=3, edgecolor="white", linewidth=0.4)

    # Observed values
    ax.axvline(obs_ml,  color=C_LGB, lw=2.0, ls="--", zorder=5,
               label=f"Observed LGB: {obs_ml:.2f}")
    ax.axvline(obs_dls, color="#64748B", lw=2.0, ls="--", zorder=5,
               label=f"Observed DLS: {obs_dls:.2f}")

    # 95% CI spans
    ax.axvspan(ci_ml[0],  ci_ml[1],  alpha=0.15, color=C_LGB,  zorder=2)
    ax.axvspan(ci_dls[0], ci_dls[1], alpha=0.12, color=C_DLS, zorder=2)

    ax.text(obs_ml - 0.6, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.18,
            f"95% CI\n[{ci_ml[0]:.1f}, {ci_ml[1]:.1f}]",
            color=C_LGB, fontsize=7.5, ha="right", fontweight="bold")
    ax.text(obs_dls + 0.3, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.15,
            f"95% CI\n[{ci_dls[0]:.1f}, {ci_dls[1]:.1f}]",
            color="#64748B", fontsize=7.5, ha="left", fontweight="bold")

    ax.set_xlabel("Bootstrap RMSE (runs)", fontsize=10, color=DARK)
    ax.set_ylabel("Density", fontsize=10, color=DARK)
    ax.set_title("(a) Bootstrap RMSE Distributions\n(5,000 resamples, stratified by phase)",
                 fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.grid(lw=0.6, alpha=0.4, zorder=0); ax.set_axisbelow(True)

    # ── Centre: Bootstrap distribution of ΔRMSE ────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT)

    n, bins2, patches = ax2.hist(diff_boot, bins=60, color=C_LGB, alpha=0.8,
                                  density=True, zorder=3, edgecolor="white", linewidth=0.3)
    # Color bars that fall outside the CI red
    for patch, left in zip(patches, bins2[:-1]):
        if left > ci_hi or left + (bins2[1] - bins2[0]) < ci_lo:
            patch.set_facecolor(C_RED)
            patch.set_alpha(0.55)

    ax2.axvline(obs_diff, color=DARK, lw=2.2, ls="-", zorder=6,
                label=f"Observed Δ = {obs_diff:.2f} runs")
    ax2.axvline(ci_lo, color=C_RED, lw=1.6, ls="--", zorder=5,
                label=f"95% CI lower = {ci_lo:.2f}")
    ax2.axvline(ci_hi, color=C_RED, lw=1.6, ls="--", zorder=5,
                label=f"95% CI upper = {ci_hi:.2f}")
    ax2.axvline(0, color="#DC2626", lw=1.0, ls=":", alpha=0.8, zorder=4,
                label="H₀: Δ = 0 (null)")
    ax2.fill_between([ci_lo, ci_hi], 0,
                     ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 0.25,
                     alpha=0.10, color=C_LGB, zorder=2)

    ax2.set_xlabel("ΔRMSE (LightGBM − DLS, runs)", fontsize=10, color=DARK)
    ax2.set_ylabel("Density", fontsize=10, color=DARK)
    ax2.set_title("(b) Bootstrap Δ RMSE Distribution\n(entirely negative → ML always better)",
                  fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    ax2.legend(fontsize=7.2, framealpha=0.9, loc="upper left")
    ax2.grid(lw=0.6, alpha=0.4, zorder=0); ax2.set_axisbelow(True)

    # ── Right: Empirical p-value illustration ──────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(LIGHT)

    # What fraction of bootstraps show Δ ≥ 0 (null direction)?
    p_null = (diff_boot >= 0).mean()

    # Sort and plot cumulative distribution
    sorted_diff = np.sort(diff_boot)
    cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
    ax3.plot(sorted_diff, cdf, color=C_LGB, lw=2.0, zorder=4)
    ax3.axvline(0,       color=C_RED,  lw=1.5, ls=":", zorder=5,
                label=f"H₀ = 0  (p̂ = {p_null:.4f})")
    ax3.axvline(obs_diff, color=DARK, lw=2.0, ls="--", zorder=5,
                label=f"Observed = {obs_diff:.2f}")
    ax3.axhline(0.025, color="#F59E0B", lw=1.2, ls="--", alpha=0.8,
                label="2.5th percentile")
    ax3.axhline(0.975, color="#F59E0B", lw=1.2, ls="--", alpha=0.8,
                label="97.5th percentile")

    # Shade CI region
    lo_idx = np.searchsorted(sorted_diff, ci_lo)
    hi_idx = np.searchsorted(sorted_diff, ci_hi)
    ax3.fill_betweenx(cdf[lo_idx:hi_idx], sorted_diff[lo_idx:hi_idx],
                      alpha=0.18, color=C_LGB, zorder=2, label="95% CI span")

    ax3.set_xlabel("ΔRMSE (LightGBM − DLS, runs)", fontsize=10, color=DARK)
    ax3.set_ylabel("Cumulative probability", fontsize=10, color=DARK)
    ax3.set_title("(c) Empirical CDF of ΔRMSE\n(zero lies at extreme right tail → H₀ rejected)",
                  fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    ax3.legend(fontsize=7.2, framealpha=0.9, loc="upper left")
    ax3.grid(lw=0.6, alpha=0.4, zorder=0); ax3.set_axisbelow(True)
    ax3.set_ylim(0, 1.05)

    plt.suptitle("Bootstrap Validation: LightGBM V2 vs DLS (5,000 stratified resamples, n=542 matches)",
                 fontsize=12, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    out = FIGS / "bootstrap_distribution.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ bootstrap_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ECONOMIC BUBBLE CHART — Revenue vs ML improvement
# ═══════════════════════════════════════════════════════════════════════════
def fig_economic_bubble():
    eco = pd.read_csv(METRICS / "economic_impact.csv")
    pt  = pd.read_csv(METRICS / "per_team_metrics.csv")
    pt  = pt[pt["model"] == "LightGBM_V2"].copy()

    # Merge on team name
    df = eco.merge(pt[["team", "n", "rmse", "bias"]], on="team", how="inner")
    df["ml_improvement"] = df["DLS"] - df["best_ml_rmse"]

    tier_colors = {1: "#3B82F6", 2: "#10B981", 3: "#F59E0B", 4: "#EF4444"}
    tier_labels = {1: "Tier 1 (BCCI/CA/ECB)", 2: "Tier 2 (NZC/CSA/PCB/BCB)",
                   3: "Tier 3 (SLC/ZC/Cricket Ireland)", 4: "Tier 4 (Associate)"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=WHITE)

    # ── Left: Revenue (log) vs ML RMSE improvement (bubble = n_snapshots) ──
    ax = axes[0]
    ax.set_facecolor(LIGHT)

    for _, row in df.iterrows():
        color = tier_colors.get(int(row["tier"]), "#94A3B8")
        size  = max(40, min(900, row["n"] / 3.5))
        ax.scatter(np.log10(row["board_revenue"]), row["ml_improvement"],
                   s=size, color=color, alpha=0.82, edgecolors="white",
                   linewidths=1.2, zorder=4)
        ax.annotate(row["team"], (np.log10(row["board_revenue"]), row["ml_improvement"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color=DARK, fontweight="bold")

    # Trend line
    log_rev = np.log10(df["board_revenue"])
    z = np.polyfit(log_rev, df["ml_improvement"], 1)
    x_line = np.linspace(log_rev.min(), log_rev.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="#94A3B8", lw=1.4,
            alpha=0.8, label="Linear trend", zorder=3)

    # Spearman r annotation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(df["board_revenue"], df["ml_improvement"])
    ax.text(0.97, 0.06, f"Spearman ρ = {rho:.2f}\n(p = {pval:.2f})",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            color=DARK, bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, alpha=0.8))

    ax.set_xlabel("Board revenue (log₁₀ USD)", fontsize=10, color=DARK)
    ax.set_ylabel("ML improvement over DLS (RMSE runs)", fontsize=10, color=DARK)
    ax.set_title("(a) Board Revenue vs ML Accuracy Improvement\n"
                 "(bubble size = test-set snapshot count)",
                 fontsize=10.5, fontweight="bold", color=DARK, pad=8)

    legend_elems = [mpatches.Patch(color=c, label=tier_labels[t], alpha=0.85)
                    for t, c in tier_colors.items() if t in df["tier"].values]
    ax.legend(handles=legend_elems, fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.grid(lw=0.6, alpha=0.4, zorder=0); ax.set_axisbelow(True)

    # ── Right: Per-team DLS vs ML RMSE + bias bars ─────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT)

    df_sorted = df.sort_values("DLS", ascending=False)
    n_teams   = len(df_sorted)
    y         = np.arange(n_teams)
    colors    = [tier_colors.get(int(t), "#94A3B8") for t in df_sorted["tier"]]

    ax2.barh(y + 0.18, df_sorted["DLS"],           0.35, color=C_DLS,   alpha=0.8,
             label="DLS RMSE", zorder=3)
    ax2.barh(y - 0.18, df_sorted["best_ml_rmse"],  0.35, color=colors,  alpha=0.85,
             label="Best ML RMSE", zorder=3)

    # Improvement annotations
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        imp = row["DLS"] - row["best_ml_rmse"]
        ax2.text(row["best_ml_rmse"] + 0.3, i,
                 f"−{imp:.1f}", va="center", fontsize=7.5,
                 color=tier_colors.get(int(row["tier"]), DARK), fontweight="bold")

    ax2.set_yticks(y)
    ax2.set_yticklabels(df_sorted["team"], fontsize=9)
    ax2.set_xlabel("RMSE (runs)", fontsize=10, color=DARK)
    ax2.set_title("(b) Per-Nation DLS vs ML RMSE\n(colour = tier; DLS shown in grey)",
                  fontsize=10.5, fontweight="bold", color=DARK, pad=8)

    dls_p  = mpatches.Patch(color=C_DLS, alpha=0.8, label="DLS Baseline")
    tier_patches = [mpatches.Patch(color=c, alpha=0.85, label=tier_labels[t])
                    for t, c in tier_colors.items() if t in df_sorted["tier"].values]
    ax2.legend(handles=[dls_p] + tier_patches, fontsize=7.2, loc="lower right",
               framealpha=0.9)
    ax2.grid(axis="x", lw=0.6, alpha=0.4, zorder=0); ax2.set_axisbelow(True)

    plt.suptitle("Economic Fairness Analysis: Revenue, Tier, and ML Accuracy Improvement",
                 fontsize=12, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    out = FIGS / "economic_bubble.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ economic_bubble.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. RAIN MATCH TARGET DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def fig_rain_targets_deep():
    df = pd.read_csv(METRICS / "dl_revised_targets.csv")
    df["target_diff"] = df["ml_target"] - df["dls_target"]
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["abs_diff"] = df["target_diff"].abs()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=WHITE)

    # ── (a) Scatter: DLS target vs ML target ──────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor(LIGHT)

    # Color by agreement
    agree_mask = df["dls_win_prediction"] == df["ml_win_prediction"]
    ax.scatter(df.loc[agree_mask,  "dls_target"], df.loc[agree_mask,  "ml_target"],
               s=18, color=C_GRN, alpha=0.55, label=f"Agree ({agree_mask.sum()} matches)",
               zorder=4)
    ax.scatter(df.loc[~agree_mask, "dls_target"], df.loc[~agree_mask, "ml_target"],
               s=18, color=C_RED, alpha=0.65, label=f"Disagree ({(~agree_mask).sum()} matches)",
               zorder=5)

    # Perfect agreement line
    lo, hi = df["dls_target"].min() - 5, df["dls_target"].max() + 5
    ax.plot([lo, hi], [lo, hi], "--", color=DARK, lw=1.2, alpha=0.6, label="Perfect agreement")
    ax.plot([lo, hi], [lo + 9.9, hi + 9.9], ":", color="#F59E0B", lw=1.4,
            label=f"Mean ML offset (+9.9 runs)", alpha=0.9)

    ax.set_xlabel("DLS revised target (runs)", fontsize=10, color=DARK)
    ax.set_ylabel("ML revised target (runs)", fontsize=10, color=DARK)
    ax.set_title("(a) DLS vs ML Revised Targets\n(colour = winner agreement)",
                 fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(lw=0.5, alpha=0.4, zorder=0); ax.set_axisbelow(True)

    # ── (b) Distribution of target differences ────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor(LIGHT)

    n, bins, patches = ax2.hist(df["target_diff"], bins=35, color=C_LGB, alpha=0.78,
                                 density=False, zorder=3, edgecolor="white", linewidth=0.4)
    # Color by direction
    for patch, left in zip(patches, bins[:-1]):
        if left < 0:
            patch.set_facecolor(C_DLS)

    ax2.axvline(0,                        color=DARK,     lw=1.5, ls=":",  zorder=5)
    ax2.axvline(df["target_diff"].mean(), color=C_RED,    lw=2.0, ls="--", zorder=5,
                label=f"Mean = +{df['target_diff'].mean():.1f} runs (t=2.88, p=0.004)")
    ax2.axvline(df["target_diff"].median(), color="#F59E0B", lw=1.5, ls="-.", zorder=5,
                label=f"Median = {df['target_diff'].median():.1f} runs")

    pct_higher = (df["target_diff"] > 0).mean() * 100
    ax2.text(0.98, 0.96, f"{pct_higher:.0f}% of matches:\nML target > DLS target",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
             color=C_LGB, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, alpha=0.85))

    ax2.set_xlabel("ML target − DLS target (runs)", fontsize=10, color=DARK)
    ax2.set_ylabel("Count", fontsize=10, color=DARK)
    ax2.set_title("(b) Target Difference Distribution\n(blue = ML higher; grey = DLS higher)",
                  fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(axis="y", lw=0.5, alpha=0.4, zorder=0); ax2.set_axisbelow(True)

    # ── (c) First-innings total vs target difference (does bias correlate?) ──
    ax3 = axes[1, 0]
    ax3.set_facecolor(LIGHT)

    ax3.scatter(df["first_innings_total"], df["target_diff"],
                c=df["target_diff"], cmap="RdBu_r", alpha=0.6, s=20, zorder=4,
                vmin=-80, vmax=80)
    ax3.axhline(0, color=DARK, lw=1.2, ls=":", zorder=5)

    # Binned mean trend
    bins_fi  = np.arange(100, 400, 30)
    bin_mids = []
    bin_means = []
    for i in range(len(bins_fi) - 1):
        mask = (df["first_innings_total"] >= bins_fi[i]) & (df["first_innings_total"] < bins_fi[i+1])
        if mask.sum() >= 3:
            bin_mids.append((bins_fi[i] + bins_fi[i+1]) / 2)
            bin_means.append(df.loc[mask, "target_diff"].mean())
    ax3.plot(bin_mids, bin_means, "o-", color=C_RED, lw=2.0, ms=7,
             markeredgecolor=WHITE, markeredgewidth=1.0, zorder=6,
             label="Binned mean Δ")

    from scipy.stats import pearsonr
    r, p = pearsonr(df["first_innings_total"], df["target_diff"])
    ax3.text(0.97, 0.05, f"Pearson r = {r:+.2f}\n(p = {p:.3f})",
             transform=ax3.transAxes, ha="right", fontsize=8.5, color=DARK,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, alpha=0.85))

    ax3.set_xlabel("First-innings total (runs)", fontsize=10, color=DARK)
    ax3.set_ylabel("ML target − DLS target (runs)", fontsize=10, color=DARK)
    ax3.set_title("(c) First-innings Total vs Target Difference\n"
                  "(does DLS bias depend on match context?)",
                  fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    ax3.legend(fontsize=8.5, framealpha=0.9)
    ax3.grid(lw=0.5, alpha=0.4, zorder=0); ax3.set_axisbelow(True)

    # ── (d) Absolute difference over time (trend) ──────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor(LIGHT)

    annual = df.groupby("year").agg(
        mean_abs_diff=("abs_diff", "mean"),
        n=("abs_diff", "count"),
        pct_disagree=("dls_win_prediction", lambda x: (x != df.loc[x.index, "ml_win_prediction"]).mean() * 100)
    ).reset_index()

    ax4b = ax4.twinx()
    ax4.bar(annual["year"], annual["mean_abs_diff"], color=C_LGB, alpha=0.7,
            width=0.55, zorder=3, label="Mean |Δ target| (runs)")
    ax4b.plot(annual["year"], annual["pct_disagree"], "s--", color=C_RED,
              lw=2.0, ms=7, zorder=5, markeredgecolor=WHITE, markeredgewidth=1.0,
              label="% winner disagreement")

    ax4.set_xlabel("Year", fontsize=10, color=DARK)
    ax4.set_ylabel("Mean |ML − DLS target| (runs)", fontsize=10, color=C_LGB)
    ax4b.set_ylabel("Winner disagreement (%)", fontsize=10, color=C_RED)
    ax4.set_title("(d) Target Disagreement Over Time\n"
                  "(recent years: DLS bias growing?)",
                  fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    ax4.set_xticks(annual["year"])

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.9)
    ax4.grid(axis="y", lw=0.5, alpha=0.4, zorder=0); ax4.set_axisbelow(True)
    ax4b.spines["top"].set_visible(False)

    plt.suptitle("Rain-Affected Match Target Analysis (255 DLS-Affected ODIs, 2016–2026)",
                 fontsize=12.5, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    out = FIGS / "rain_targets_deep.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"✓ rain_targets_deep.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating supplementary figures…\n")
    fig_bootstrap_distribution()
    fig_economic_bubble()
    fig_rain_targets_deep()
    print(f"\nAll figures saved to {FIGS}")
