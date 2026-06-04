"""Generate the README charts from the real results. Clean, few, no clutter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT))
from calorie_pipeline.models import combine_estimates  # noqa: E402

INK = "#222b35"
SLATE = "#48586a"
MUTED = "#b07a52"   # "worse than baseline" steps
WIN = "#2e8b6f"     # beats baseline
BASE = "#d9534f"
plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#d7dde3", "axes.linewidth": 1.0,
    "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.titlepad": 14,
    "figure.dpi": 150, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#5b6b7a", "ytick.color": INK, "xtick.labelsize": 10,
})


def _bar_labels(ax, bars, fmt, pad, inside=False):
    for b in bars:
        w = b.get_width()
        if inside:
            ax.text(w - pad, b.get_y() + b.get_height() / 2, fmt(w),
                    ha="right", va="center", color="white", fontweight="bold")
        else:
            ax.text(w + pad, b.get_y() + b.get_height() / 2, fmt(w),
                    ha="left", va="center", color=INK, fontweight="bold")


def recipe_ablation():
    steps = ["single prompt  (baseline)", "+ decompose & ground in USDA",
             "+ model picks the match", "+ blend with the prompt", "+ clamp to prompt  (shipped)"]
    mae = [90, 141, 125, 87, 86]
    colors = [SLATE, MUTED, MUTED, WIN, WIN]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    bars = ax.barh(range(len(steps)), mae, color=colors, height=0.66, zorder=3)
    ax.axvline(90, color="#9aa7b4", ls=(0, (4, 3)), lw=1.2, zorder=1)
    _bar_labels(ax, bars, lambda w: f"{w:.0f}", 2.5, inside=True)
    ax.set_yticks(range(len(steps)), steps)
    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    ax.set_xlabel("mean error vs lab-measured calories, kcal   ·   lower is better")
    ax.set_title("How to beat a single prompt", loc="left")
    ax.text(90, -0.78, "baseline 90", color="#7d8a98", fontsize=9.5, ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(DOCS / "recipe_ablation.png", bbox_inches="tight")
    plt.close(fig)


def predicted_vs_measured():
    exp = json.loads((ROOT / "benchmark/results/experiment_llm_match.json").read_text())
    t = [x["truth"] for x in exp]
    o = [x["oneshot"] for x in exp]
    comb = [combine_estimates(x["oneshot"], x["llm"], 0.5, 0.4) for x in exp]
    lim = max(max(t), max(o), max(comb)) * 1.07
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.plot([0, lim], [0, lim], "-", color="#cdd5dc", lw=1.2, zorder=0)
    ax.scatter(t, o, c=BASE, s=44, alpha=0.8, edgecolor="white", linewidth=0.5, label="single prompt  (MAE 90)")
    ax.scatter(t, comb, c=WIN, s=44, alpha=0.9, edgecolor="white", linewidth=0.5, label="recipe / blend  (MAE 86)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("measured calories (ground truth)")
    ax.set_ylabel("predicted calories")
    ax.set_title("Closer to the line is better", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "predicted_vs_measured.png", bbox_inches="tight")
    plt.close(fig)


def token_cost():
    labels = ["single\nprompt", "decomposed\npipeline", "combined\n(naive)", "combined\n(shipped)"]
    vals = [1.00, 1.66, 2.66, 1.71]
    colors = [SLATE, "#8aa0b5", MUTED, WIN]
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    bars = ax.bar(labels, vals, color=colors, width=0.64, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}×", ha="center", fontweight="bold")
    ax.set_ylim(0, 3.05)
    ax.set_ylabel("tokens, relative to a single prompt")
    ax.set_title("The cost, defused: encode the image once — and grounding is free", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(DOCS / "token_cost.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    recipe_ablation()
    predicted_vs_measured()
    token_cost()
    # remove charts no longer referenced
    for stale in ("hero_results.png", "noise_vs_signal.png"):
        (DOCS / stale).unlink(missing_ok=True)
    print("wrote charts to", DOCS)
