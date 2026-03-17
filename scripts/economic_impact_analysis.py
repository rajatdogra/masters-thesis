"""
Economic Impact Analysis of Score Prediction Accuracy in Rain-Affected ODI Cricket.

Motivation
----------
The DLS method is not merely a statistical tool — it is an economic mechanism that
redistributes outcomes (and therefore prize money) across competing nations.

Economic contributions of this analysis:

1. Equity / Distributional Fairness
   ─────────────────────────────────
   Does DLS prediction error fall uniformly across ICC member nations, or do
   lower-tier cricket boards (and their players/fans) bear disproportionate costs?
   Measured via Gini coefficient and Kruskal-Wallis test across board-revenue tiers.

2. Economic Value of Prediction Accuracy
   ────────────────────────────────────────
   Decision-theoretic framework: RMSE → P(outcome change) → prize money at stake.
   Following the Expected Value of Information (EVOI) framework, each 1-run
   improvement in target accuracy has quantifiable economic value in terms of
   expected prize redistribution.

3. Rain Match Target Equity Analysis
   ─────────────────────────────────
   For 255 empirical rain-affected matches: distribution of ML vs DLS target
   revisions, directional bias test, and cases of disagreement.

4. Marginal Value of Accuracy (MVA) Curve
   ─────────────────────────────────────────
   How does prize-money-weighted expected outcome error change as RMSE improves
   from DLS baseline (75.74) toward a hypothetical perfect predictor (RMSE=0)?

Output
------
  results/metrics/economic_impact.csv     — full team-level fairness table
  results/metrics/economic_value.json     — decision-theoretic value estimates
  results/figures/economic_fairness.png   — per-tier RMSE comparison
  results/figures/economic_value.png      — MVA curve + rain match distribution

Runtime: ~2 minutes
"""

import sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("economic_impact")


# ──────────────────────────────────────────────────────────────
# A. Team economic tier classification
# Source: ICC financial distributions, board annual reports (2022–2024 estimates)
# ICC distributes ~$600M per cycle; BCCI retains largest share (~30%)
# ──────────────────────────────────────────────────────────────

TEAM_TIERS = {
    # Tier 1: Full Members, highest board revenues (>$100M/yr estimated)
    "India":                {"tier": 1, "label": "Tier 1\n(Major)", "revenue_m": 1700},
    "Australia":            {"tier": 1, "label": "Tier 1\n(Major)", "revenue_m": 180},
    "England":              {"tier": 1, "label": "Tier 1\n(Major)", "revenue_m": 200},
    # Tier 2: Full Members, mid-size ($20–100M/yr)
    "South Africa":         {"tier": 2, "label": "Tier 2\n(Mid)",   "revenue_m": 40},
    "New Zealand":          {"tier": 2, "label": "Tier 2\n(Mid)",   "revenue_m": 35},
    "Sri Lanka":            {"tier": 2, "label": "Tier 2\n(Mid)",   "revenue_m": 25},
    "Pakistan":             {"tier": 2, "label": "Tier 2\n(Mid)",   "revenue_m": 45},
    "West Indies":          {"tier": 2, "label": "Tier 2\n(Mid)",   "revenue_m": 30},
    # Tier 3: Full Members, smaller boards (<$20M/yr)
    "Bangladesh":           {"tier": 3, "label": "Tier 3\n(Small)", "revenue_m": 15},
    "Zimbabwe":             {"tier": 3, "label": "Tier 3\n(Small)", "revenue_m":  5},
    "Afghanistan":          {"tier": 3, "label": "Tier 3\n(Small)", "revenue_m":  8},
    "Ireland":              {"tier": 3, "label": "Tier 3\n(Small)", "revenue_m":  7},
    # Tier 4: Associate / Emerging nations
    "United Arab Emirates": {"tier": 4, "label": "Tier 4\n(Assoc)", "revenue_m":  2},
    "Scotland":             {"tier": 4, "label": "Tier 4\n(Assoc)", "revenue_m":  2},
    "Netherlands":          {"tier": 4, "label": "Tier 4\n(Assoc)", "revenue_m":  1},
    "Oman":                 {"tier": 4, "label": "Tier 4\n(Assoc)", "revenue_m":  1},
    "Namibia":              {"tier": 4, "label": "Tier 4\n(Assoc)", "revenue_m":  1},
}

# ICC prize money (USD) — ICC Men's ODI World Cup 2023
ICC_PRIZE = {
    "winner": 4_000_000,
    "runner_up": 2_000_000,
    "semifinal": 800_000,
    "group_stage": 100_000,
}


def gini_coefficient(values):
    """Compute Gini coefficient (0=perfect equality, 1=maximal inequality)."""
    v = np.sort(np.abs(np.array(values, dtype=float)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n)


def outcome_change_probability(delta_target_runs, typical_margins=None):
    """
    P(match outcome changes | target revised by delta_target_runs).

    Empirical distribution of ODI win margins (2010–2023, runs-based results):
    ~30% of ODI matches decided by <20 runs or <4 wickets when rain involved.
    We model run margins as Exponential(λ) calibrated to median ~18 runs.

    This gives a conservative lower bound on P(change).
    """
    if typical_margins is None:
        # Empirical: median ODI winning margin ≈ 18 runs (Cricinfo stats)
        # Use exponential: P(margin < delta) = 1 - exp(-delta/18)
        lam = 1.0 / 18.0
        return float(1.0 - np.exp(-lam * abs(delta_target_runs)))
    return float(np.mean(np.abs(typical_margins) < abs(delta_target_runs)))


# ──────────────────────────────────────────────────────────────
# 1. Load per-team metrics
# ──────────────────────────────────────────────────────────────
log.info("Loading per-team metrics...")
team_df = pd.read_csv(METRICS / "per_team_metrics.csv")
log.info(f"  {len(team_df)} rows, teams: {sorted(team_df['team'].unique())}")

# Attach tier info
team_df["tier"]          = team_df["team"].map(lambda t: TEAM_TIERS.get(t, {}).get("tier", 4))
team_df["tier_label"]    = team_df["team"].map(lambda t: TEAM_TIERS.get(t, {}).get("label", "Tier 4\n(Assoc)"))
team_df["board_revenue"] = team_df["team"].map(lambda t: TEAM_TIERS.get(t, {}).get("revenue_m", 1))

log.info(f"  Tier distribution:\n{team_df[['team','tier']].drop_duplicates().to_string(index=False)}")


# ──────────────────────────────────────────────────────────────
# 2. Fairness equity analysis
# ──────────────────────────────────────────────────────────────
log.info("\n=== FAIRNESS EQUITY ANALYSIS ===")

# Pivot: team × model → RMSE
rmse_pivot = team_df.pivot_table(index=["team","tier","tier_label","board_revenue"],
                                  columns="model", values="rmse").reset_index()
rmse_pivot.columns.name = None

# Only keep teams with all 3 models
rmse_pivot = rmse_pivot.dropna(subset=["DLS"] if "DLS" in rmse_pivot.columns else [])
log.info(f"  Teams with DLS + ML metrics: {len(rmse_pivot)}")

# Per-team disadvantage = DLS RMSE − best ML RMSE
ml_cols = [c for c in ["CatBoost_V2", "LightGBM_V2"] if c in rmse_pivot.columns]
if ml_cols:
    rmse_pivot["best_ml_rmse"]     = rmse_pivot[ml_cols].min(axis=1)
    rmse_pivot["dls_advantage"]    = rmse_pivot["DLS"] - rmse_pivot["best_ml_rmse"]  # positive = ML better
    rmse_pivot["relative_error"]   = rmse_pivot["dls_advantage"] / rmse_pivot["DLS"] * 100  # % improvement

    log.info("\n  Per-team DLS advantage (DLS RMSE - best ML RMSE):")
    log.info(rmse_pivot[["team","tier","DLS","best_ml_rmse","dls_advantage","relative_error"]]
             .sort_values("tier").to_string(index=False))

# Gini coefficient: equality of DLS errors across teams
if "DLS" in rmse_pivot.columns:
    gini_dls = gini_coefficient(rmse_pivot["DLS"].dropna())
    gini_ml  = gini_coefficient(rmse_pivot["best_ml_rmse"].dropna()) if ml_cols else None
    log.info(f"\n  Gini(DLS RMSE across teams) = {gini_dls:.4f}")
    if gini_ml is not None:
        log.info(f"  Gini(ML RMSE across teams)  = {gini_ml:.4f}  (lower = more equitable)")

# Kruskal-Wallis: do DLS errors differ across tiers?
tier_groups_dls = [
    rmse_pivot.loc[rmse_pivot["tier"] == t, "DLS"].dropna().values
    for t in sorted(rmse_pivot["tier"].unique())
]
tier_groups_dls = [g for g in tier_groups_dls if len(g) >= 2]
if len(tier_groups_dls) >= 2:
    kw_stat, kw_p = stats.kruskal(*tier_groups_dls)
    log.info(f"\n  Kruskal-Wallis (DLS RMSE across tiers): H={kw_stat:.3f}, p={kw_p:.4f}")
else:
    kw_stat, kw_p = np.nan, np.nan
    log.info("  Not enough tiers for Kruskal-Wallis test.")

# Spearman correlation: board revenue vs DLS RMSE
if "DLS" in rmse_pivot.columns and len(rmse_pivot) >= 4:
    sp_corr, sp_p = stats.spearmanr(rmse_pivot["board_revenue"], rmse_pivot["DLS"])
    log.info(f"  Spearman(board revenue, DLS RMSE) = {sp_corr:.3f}, p={sp_p:.4f}")
    sp_corr_ml, sp_p_ml = stats.spearmanr(
        rmse_pivot["board_revenue"], rmse_pivot["best_ml_rmse"])
    log.info(f"  Spearman(board revenue, ML RMSE)  = {sp_corr_ml:.3f}, p={sp_p_ml:.4f}")
else:
    sp_corr, sp_p, sp_corr_ml, sp_p_ml = np.nan, np.nan, np.nan, np.nan


# ──────────────────────────────────────────────────────────────
# 3. Rain match target equity analysis
# ──────────────────────────────────────────────────────────────
log.info("\n=== RAIN MATCH TARGET EQUITY ===")

dl_df = pd.read_csv(METRICS / "dl_revised_targets.csv")
# target_diff = ml_target - dls_target (positive = ML sets higher target)
dl_df["abs_diff"] = dl_df["target_diff"].abs()
dl_df["direction"] = np.where(dl_df["target_diff"] > 0, "ML higher", "DLS higher")

log.info(f"  Total rain-affected matches: {len(dl_df)}")
log.info(f"  Mean |ML - DLS target|: {dl_df['abs_diff'].mean():.1f} runs")
log.info(f"  Median |ML - DLS target|: {dl_df['abs_diff'].median():.1f} runs")
log.info(f"  Std:                       {dl_df['abs_diff'].std():.1f} runs")
log.info(f"  ML higher than DLS: {(dl_df['target_diff'] > 0).sum()} ({(dl_df['target_diff'] > 0).mean()*100:.1f}%)")
log.info(f"  DLS higher than ML: {(dl_df['target_diff'] < 0).sum()} ({(dl_df['target_diff'] < 0).mean()*100:.1f}%)")

# One-sample t-test: is mean target_diff significantly different from 0?
t_stat, t_p = stats.ttest_1samp(dl_df["target_diff"].dropna(), 0)
mean_diff = dl_df["target_diff"].mean()
log.info(f"  One-sample t-test (target_diff ≠ 0): t={t_stat:.3f}, p={t_p:.4f}")
log.info(f"  Mean target_diff (ML − DLS): {mean_diff:+.2f} runs")

# High-stakes: cases where |diff| > 20 runs (match-changing magnitude)
high_stakes = dl_df[dl_df["abs_diff"] > 20]
log.info(f"  High-stakes disagreements (|diff|>20 runs): {len(high_stakes)} ({len(high_stakes)/len(dl_df)*100:.1f}%)")

# Where ML and DLS predict DIFFERENT winners
if "dls_win_prediction" in dl_df.columns and "ml_win_prediction" in dl_df.columns:
    winner_disagree = dl_df[dl_df["dls_win_prediction"] != dl_df["ml_win_prediction"]]
    log.info(f"  Winner disagreements (ML ≠ DLS prediction): {len(winner_disagree)} ({len(winner_disagree)/len(dl_df)*100:.1f}%)")

    # Of disagreements, how often is ML correct?
    if "actual_team2_won" in dl_df.columns:
        ml_wins_disagree = (winner_disagree["ml_win_prediction"] == winner_disagree["actual_team2_won"]).mean()
        dls_wins_disagree = (winner_disagree["dls_win_prediction"] == winner_disagree["actual_team2_won"]).mean()
        log.info(f"  ML correct in disagreements: {ml_wins_disagree*100:.1f}%")
        log.info(f"  DLS correct in disagreements: {dls_wins_disagree*100:.1f}%")


# ──────────────────────────────────────────────────────────────
# 4. Economic value quantification
# ──────────────────────────────────────────────────────────────
log.info("\n=== ECONOMIC VALUE FRAMEWORK ===")

# Empirical: average |target_diff| from rain matches dataset
mean_abs_diff = dl_df["abs_diff"].mean()

# P(outcome change) given observed target differences
p_change_given_diff = outcome_change_probability(mean_abs_diff)
log.info(f"  Mean |ML − DLS target|: {mean_abs_diff:.1f} runs")
log.info(f"  P(outcome change | mean diff): {p_change_given_diff:.3f}")

# Expected value calculation for different tournament stages
stages = [
    ("ICC World Cup (Winner)", ICC_PRIZE["winner"] - ICC_PRIZE["runner_up"]),
    ("ICC World Cup (Semi-final)", ICC_PRIZE["semifinal"] - ICC_PRIZE["group_stage"]),
    ("ICC World Cup (Group stage)", ICC_PRIZE["group_stage"]),
]

ev_table = []
for stage_name, prize_diff in stages:
    ev = p_change_given_diff * prize_diff
    ev_table.append({
        "stage": stage_name,
        "prize_differential_usd": prize_diff,
        "p_outcome_change": round(p_change_given_diff, 4),
        "expected_value_per_match_usd": round(ev, 0),
    })
    log.info(f"  {stage_name}: Δprize=${prize_diff:,} → EV=${ev:,.0f}/match")

# Marginal Value of Accuracy (MVA): EV as function of RMSE improvement
# From DLS RMSE (75.74 inn2) down to perfect (0)
# We use the rain match mean diff as proxy: mean_diff ≈ 0.5 * RMSE_improvement
dls_rmse_inn2  = 75.74
ml_rmse_inn2   = 39.92
rmse_range     = np.linspace(0, dls_rmse_inn2, 200)
# proxy: target error ≈ 0.4 × RMSE (empirical ratio from our data)
target_err_scale = mean_abs_diff / (dls_rmse_inn2 - ml_rmse_inn2) if (dls_rmse_inn2 - ml_rmse_inn2) > 0 else 0.4
mva_p_change   = np.array([outcome_change_probability(r * target_err_scale) for r in rmse_range])
# World Cup semi-final stage
prize_diff_semi = ICC_PRIZE["semifinal"] - ICC_PRIZE["group_stage"]
mva_ev         = mva_p_change * prize_diff_semi

log.info(f"\n  MVA scale factor (target_err per RMSE unit): {target_err_scale:.4f}")
log.info(f"  EV at DLS RMSE (75.74): ${mva_ev[np.argmin(np.abs(rmse_range - dls_rmse_inn2))]:,.0f}")
log.info(f"  EV at ML RMSE  (39.92): ${mva_ev[np.argmin(np.abs(rmse_range - ml_rmse_inn2))]:,.0f}")
log.info(f"  Economic gain from DLS→ML: ${mva_ev[np.argmin(np.abs(rmse_range - dls_rmse_inn2))] - mva_ev[np.argmin(np.abs(rmse_range - ml_rmse_inn2))]:,.0f}/match")


# ──────────────────────────────────────────────────────────────
# 5. Save outputs
# ──────────────────────────────────────────────────────────────
rmse_pivot.to_csv(METRICS / "economic_impact.csv", index=False)
log.info(f"\n  Saved: {METRICS / 'economic_impact.csv'}")

ev_json = {
    "mean_abs_target_diff_runs": round(float(mean_abs_diff), 2),
    "p_outcome_change_given_mean_diff": round(float(p_change_given_diff), 4),
    "ev_per_match_usd": {row["stage"]: row["expected_value_per_match_usd"] for row in ev_table},
    "gini_dls": round(float(gini_dls), 4),
    "gini_ml": round(float(gini_ml), 4) if gini_ml is not None else None,
    "kruskal_wallis_H": round(float(kw_stat), 4) if not np.isnan(kw_stat) else None,
    "kruskal_wallis_p": round(float(kw_p), 4) if not np.isnan(kw_p) else None,
    "spearman_revenue_dls_rmse": round(float(sp_corr), 4) if not np.isnan(sp_corr) else None,
    "ttest_target_diff_t": round(float(t_stat), 4),
    "ttest_target_diff_p": round(float(t_p), 4),
    "mean_target_diff_ml_minus_dls": round(float(mean_diff), 2),
    "n_rain_matches": int(len(dl_df)),
    "pct_high_stakes_disagreement": round(float(len(high_stakes) / len(dl_df) * 100), 1),
}
with open(METRICS / "economic_value.json", "w") as f:
    json.dump(ev_json, f, indent=2)
log.info(f"  Saved: {METRICS / 'economic_value.json'}")


# ──────────────────────────────────────────────────────────────
# 6. Figures
# ──────────────────────────────────────────────────────────────
log.info("\nGenerating economic figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Economic Impact Analysis: Score Prediction vs DLS", fontsize=14, fontweight="bold")

TIER_COLORS = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3"}
tier_patch = [
    mpatches.Patch(color=TIER_COLORS[1], label="Tier 1 (Major)"),
    mpatches.Patch(color=TIER_COLORS[2], label="Tier 2 (Mid)"),
    mpatches.Patch(color=TIER_COLORS[3], label="Tier 3 (Small)"),
    mpatches.Patch(color=TIER_COLORS[4], label="Tier 4 (Associate)"),
]

# ─ Panel A: Per-team DLS vs ML RMSE ─
ax = axes[0, 0]
if ml_cols and "DLS" in rmse_pivot.columns:
    subset = rmse_pivot.sort_values("tier").dropna(subset=["DLS", "best_ml_rmse"])
    x = np.arange(len(subset))
    w = 0.35
    bar_colors = [TIER_COLORS[t] for t in subset["tier"]]
    b1 = ax.bar(x - w/2, subset["DLS"],          width=w, color="#ff7f00", alpha=0.85, label="DLS", edgecolor="k", lw=0.5)
    b2 = ax.bar(x + w/2, subset["best_ml_rmse"], width=w, color=bar_colors, alpha=0.85, label="Best ML", edgecolor="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["team"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (runs)", fontsize=10)
    ax.set_title("A: Per-Team Prediction Error (DLS vs Best ML)", fontsize=10)
    ax.legend(fontsize=8)
    ax.legend(handles=[mpatches.Patch(color="#ff7f00", label="DLS")] + tier_patch, fontsize=7)

# ─ Panel B: Distribution of rain match target differences ─
ax = axes[0, 1]
ax.hist(dl_df["target_diff"].dropna(), bins=30, color="#377eb8", edgecolor="k", linewidth=0.5, alpha=0.8)
ax.axvline(0, color="k", lw=1.5, ls="--", label="No difference")
ax.axvline(mean_diff, color="red", lw=2, ls="-", label=f"Mean diff: {mean_diff:+.1f} runs")
ax.axvspan(-20, 20, alpha=0.08, color="green", label="Low-stakes zone (±20 runs)")
ax.set_xlabel("ML Target − DLS Target (runs)", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.set_title("B: Distribution of Target Differences (255 Rain Matches)", fontsize=10)
ax.legend(fontsize=8)

# ─ Panel C: Marginal Value of Accuracy curve ─
ax = axes[1, 0]
ax.fill_between(rmse_range, mva_ev, alpha=0.2, color="#e41a1c")
ax.plot(rmse_range, mva_ev, color="#e41a1c", lw=2)
# Mark key points
for rmse_val, label, color in [
    (dls_rmse_inn2, f"DLS\n({dls_rmse_inn2:.1f})", "#ff7f00"),
    (ml_rmse_inn2,  f"ML\n({ml_rmse_inn2:.1f})",  "#377eb8"),
]:
    idx = np.argmin(np.abs(rmse_range - rmse_val))
    ax.axvline(rmse_val, color=color, lw=1.5, ls="--")
    ax.scatter([rmse_val], [mva_ev[idx]], color=color, s=80, zorder=5)
    ax.annotate(label, (rmse_val, mva_ev[idx]),
                xytext=(10, 15), textcoords="offset points", fontsize=8, color=color)
ax.set_xlabel("RMSE (runs)", fontsize=10)
ax.set_ylabel("Expected Economic Value at Stake (USD)", fontsize=10)
ax.set_title("C: Marginal Value of Accuracy\n(ICC semi-final prize differential, per rain match)", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ─ Panel D: Gini / equity illustration ─
ax = axes[1, 1]
if ml_cols and "DLS" in rmse_pivot.columns:
    subset_sorted = rmse_pivot.sort_values("DLS").dropna(subset=["DLS", "best_ml_rmse"])
    cum_share = np.linspace(0, 1, len(subset_sorted) + 1)
    # Lorenz curve for DLS RMSE
    dls_vals = np.sort(subset_sorted["DLS"].values)
    ml_vals  = np.sort(subset_sorted["best_ml_rmse"].values)
    lorenz_dls = np.concatenate([[0], np.cumsum(dls_vals) / dls_vals.sum()])
    lorenz_ml  = np.concatenate([[0], np.cumsum(ml_vals) / ml_vals.sum()])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect equality")
    ax.plot(cum_share, lorenz_dls, color="#ff7f00", lw=2, label=f"DLS (Gini={gini_dls:.3f})")
    if gini_ml is not None:
        ax.plot(cum_share, lorenz_ml, color="#4daf4a", lw=2, label=f"ML  (Gini={gini_ml:.3f})")
    ax.fill_between(cum_share, lorenz_dls, cum_share, alpha=0.15, color="#ff7f00")
    ax.set_xlabel("Cumulative share of teams", fontsize=10)
    ax.set_ylabel("Cumulative share of RMSE burden", fontsize=10)
    ax.set_title("D: Lorenz Curve — Equity of Prediction Error\nAcross ICC Member Nations", fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIGURES / "economic_fairness.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"  Saved: {FIGURES / 'economic_fairness.png'}")

log.info("=" * 60)
log.info("ECONOMIC IMPACT ANALYSIS COMPLETE")
log.info("=" * 60)
log.info(f"  Gini DLS: {gini_dls:.4f}  |  Gini ML: {gini_ml:.4f}" if gini_ml else f"  Gini DLS: {gini_dls:.4f}")
log.info(f"  Mean rain target diff: {mean_diff:+.1f} runs (t={t_stat:.2f}, p={t_p:.4f})")
log.info(f"  P(outcome change | mean diff={mean_abs_diff:.1f} runs): {p_change_given_diff:.3f}")
log.info(f"  EV at WC semi-final stage: ${mva_ev[np.argmin(np.abs(rmse_range-dls_rmse_inn2))]:,.0f} (DLS) → ${mva_ev[np.argmin(np.abs(rmse_range-ml_rmse_inn2))]:,.0f} (ML)")
log.info(f"  Economic equity improvement: Gini reduced from {gini_dls:.3f} to {gini_ml:.3f}")
