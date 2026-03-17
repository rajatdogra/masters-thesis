"""
Phase-grouped methodology diagram — the standard thesis style.
Portrait, compact, groups stages by research phase with subtle backgrounds.
Output: results/figures/research_flowchart.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

# ── Palette ───────────────────────────────────────────────────────────────────
WHITE  = "#FFFFFF"
DARK   = "#1C2833"
ARROW  = "#717D7E"

# Phase colours — subtle, desaturated
PH = {
    "data"  : ("#D6EAF8", "#2E86C1"),   # fill, border
    "feat"  : ("#D5F5E3", "#1E8449"),
    "model" : ("#FEF9E7", "#B7950B"),
    "eval"  : ("#FDEDEC", "#C0392B"),
    "out"   : ("#F4ECF7", "#7D3C98"),
}

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 4.4, 8.2
fig, ax = plt.subplots(figsize=(FW, FH), dpi=260)
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.axis("off"); fig.patch.set_facecolor(WHITE)

# ── Geometry ──────────────────────────────────────────────────────────────────
CX  = FW / 2        # 2.2
MW  = 3.8           # main box width
BW  = 1.72          # branch box width
MH  = 0.54          # main box height
BH  = 0.64          # branch box height
GAP = 0.18          # vertical gap

# ── Box drawing ───────────────────────────────────────────────────────────────
def box(cx, cy, w, h, title, sub, phase):
    fill, border = PH[phase]
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.04",
        linewidth=1.1, edgecolor=border,
        facecolor=fill, zorder=3))
    if sub:
        ax.text(cx, cy + h*0.14, title,
                ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=DARK, zorder=4)
        ax.text(cx, cy - h*0.26, sub,
                ha="center", va="center", fontsize=5.9,
                color="#555555", zorder=4)
    else:
        ax.text(cx, cy, title,
                ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=DARK, zorder=4)

def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>", color=ARROW,
                    lw=1.0, mutation_scale=9))

def phase_label(y_top, y_bot, label, phase):
    """Thin coloured left-margin label for each phase group."""
    _, border = PH[phase]
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, y_bot), 0.18, y_top - y_bot,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor=border, zorder=2, alpha=0.85))
    ax.text(0.13, (y_top + y_bot) / 2, label,
            ha="center", va="center", fontsize=5.2,
            fontweight="bold", color=WHITE, rotation=90, zorder=3)

# ── Layout — Y positions (top → bottom) ──────────────────────────────────────
y = FH - 0.40

def ny(h=MH):
    global y
    cy = y - h/2
    y -= h + GAP
    return cy

# Phase: DATA
y_data_top = y
y_raw  = ny(); y_proc = ny()
y_data_bot = y + GAP

# Phase: FEATURES
y_feat_top = y - 0.06
y_feat_ = ny(); y_split = ny()
y_feat_bot = y + GAP

# Phase: MODELS
y_mod_top = y - 0.06
y_branch1 = ny(BH)   # DLS / ML same row
y_ext  = ny()
y_mod_bot = y + GAP

# Phase: EVALUATION
y_eval_top = y - 0.06
y_eval_ = ny()
y_branch2 = ny(BH)   # Explainability / Rain same row
y_eval_bot = y + GAP

# Phase: OUTPUT
y_out_top = y - 0.06
y_econ = ny(); y_res  = ny()
y_out_bot = y + GAP/2

# ── Phase background bands ────────────────────────────────────────────────────
def phase_band(y_top, y_bot, phase):
    fill, _ = PH[phase]
    ax.add_patch(mpatches.Rectangle(
        (0.25, y_bot - 0.06), FW - 0.30, y_top - y_bot + 0.12,
        linewidth=0, facecolor=fill, alpha=0.35, zorder=1))

phase_band(y_data_top,  y_data_bot,  "data")
phase_band(y_feat_top,  y_feat_bot,  "feat")
phase_band(y_mod_top,   y_mod_bot,   "model")
phase_band(y_eval_top,  y_eval_bot,  "eval")
phase_band(y_out_top,   y_out_bot,   "out")

# ── Phase margin labels ───────────────────────────────────────────────────────
phase_label(y_data_top,  y_data_bot,  "DATA",       "data")
phase_label(y_feat_top,  y_feat_bot,  "FEATURES",   "feat")
phase_label(y_mod_top,   y_mod_bot,   "MODELS",     "model")
phase_label(y_eval_top,  y_eval_bot,  "EVALUATION", "eval")
phase_label(y_out_top,   y_out_bot,   "OUTPUT",     "out")

# ── Nodes ─────────────────────────────────────────────────────────────────────
LX = CX - BW/2 - 0.06
RX = CX + BW/2 + 0.06

box(CX, y_raw,    MW, MH, "Data Collection",
    "3,500+ Men's ODI matches  ·  Cricsheet (2002–2026)", "data")
box(CX, y_proc,   MW, MH, "Data Processing",
    "129k first-innings  ·  113k second-innings  ·  49.2% right-censored", "data")

box(CX, y_feat_,  MW, MH, "Feature Engineering  (V2 — 46 features)",
    "Match state · Player stats · Elo ratings · Venue · DLS-derived", "feat")
box(CX, y_split,  MW, MH, "Temporal Three-Way Split",
    "60% Train  ·  20% Calibration  ·  20% Test (2022–2026)", "feat")

box(LX, y_branch1, BW, BH, "DLS Baseline",
    "scipy.optimize fit\n3,086 complete matches", "model")
box(RX, y_branch1, BW, BH, "ML Models",
    "XGBoost · LightGBM · CatBoost\nOptuna Bayesian HPO", "model")
box(CX, y_ext,    MW, MH, "Extended Architectures",
    "Stacking Ensemble  ·  Monotonic XGBoost  ·  Walk-Forward CV", "model")

box(CX, y_eval_,  MW, MH, "Statistical Evaluation",
    "DM Test · Bootstrap CI · MCS · Bonferroni · Ablation · Conformal", "eval")
box(LX, y_branch2, BW, BH, "Explainability",
    "SHAP (global & local)\nLIME — 5 match cases", "eval")
box(RX, y_branch2, BW, BH, "Rain Match Evaluation",
    "255 D/L-affected matches\nML vs official DLS targets", "eval")

box(CX, y_econ,   MW, MH, "Economic Impact Analysis",
    "Gini fairness  ·  EVI framework  ·  Target bias  ·  ICC prizes", "out")
box(CX, y_res,    MW, MH, "Results & Conclusions",
    "CatBoost RMSE −33% vs DLS  ·  All DM tests p < 0.001", "out")

# ── Arrows ────────────────────────────────────────────────────────────────────
def bot(cy, h=MH): return cy - h/2
def top(cy, h=MH): return cy + h/2

arr(CX, bot(y_raw),    CX, top(y_proc))
arr(CX, bot(y_proc),   CX, top(y_feat_))
arr(CX, bot(y_feat_),  CX, top(y_split))

# Split → branches
arr(CX, bot(y_split), LX, top(y_branch1, BH))
arr(CX, bot(y_split), RX, top(y_branch1, BH))

# Branches → ext
arr(LX, bot(y_branch1, BH), CX, top(y_ext))
arr(RX, bot(y_branch1, BH), CX, top(y_ext))

arr(CX, bot(y_ext),    CX, top(y_eval_))

# Eval → side branches
arr(CX, bot(y_eval_), LX, top(y_branch2, BH))
arr(CX, bot(y_eval_), RX, top(y_branch2, BH))

# Side branches → econ
arr(LX, bot(y_branch2, BH), CX, top(y_econ))
arr(RX, bot(y_branch2, BH), CX, top(y_econ))

arr(CX, bot(y_econ),   CX, top(y_res))

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.1)
plt.savefig(FIGS / "research_flowchart.png",
            dpi=260, bbox_inches="tight", facecolor=WHITE)
plt.close()
print(f"Saved → {FIGS}/research_flowchart.png")
