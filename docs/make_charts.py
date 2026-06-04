"""Generate the presentation charts embedded in the README, from the real results."""

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

INK = "#1b2733"
MUTED = "#9aa7b4"
BASE = "#d9534f"      # one-shot / baseline
PIPE = "#6c8ebf"      # pipeline
WIN = "#2e8b6f"       # the shipped winner
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#c9d2da", "axes.linewidth": 1.0,
    "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 140,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def load():
    exp = json.loads((ROOT / "benchmark/results/experiment_llm_match.json").read_text())
    t = [x["truth"] for x in exp]
    o = [x["oneshot"] for x in exp]
    llm = [x["llm"] for x in exp]
    comb = [combine_estimates(a, b, 0.5, 0.4) for a, b in zip(o, llm)]
    return t, o, llm, comb


def recipe_ablation():
    """The headline: each recipe step's measured effect, building past the baseline."""
    steps = ["single prompt  (baseline)", "+ decompose & ground in USDA", "+ model picks the match",
             "+ blend with the prompt", "+ clamp to the prompt  (shipped)"]
    mae = [90, 141, 125, 87, 86]
    colors = ["#5b6b7a", MUTED, "#b9876b", WIN, WIN]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bars = ax.barh(range(len(steps)), mae, color=colors, height=0.6)
    ax.axvline(90, color="#5b6b7a", ls="--", lw=1.3, zorder=0)
    ax.text(90, 4.62, " baseline to beat", color="#5b6b7a", fontsize=9, va="center")
    for i, (b, m) in enumerate(zip(bars, mae)):
        beat = m < 90
        ax.text(b.get_width() - 4, i, f"{m:.0f}", ha="right", va="center", color="white", fontweight="bold")
        if beat:
            ax.text(b.get_width() + 3, i, "beats baseline", va="center", color=WIN, fontsize=8.5)
    ax.set_yticks(range(len(steps)), steps)
    ax.invert_yaxis()
    ax.set_xlabel("Mean error vs lab-measured calories, kcal  (lower is better)")
    ax.set_title("How to beat a single prompt: a measured recipe")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 158)
    fig.tight_layout()
    fig.savefig(DOCS / "recipe_ablation.png", bbox_inches="tight")
    plt.close(fig)


def hero_results():
    methods = ["Decomposed pipeline\n(alone)", "Pipeline +\nLLM match", "Single prompt\n(baseline)",
               "Combined\n(shipped)"]
    mae = [140.7, 125.1, 89.9, 86.1]
    colors = [MUTED, PIPE, BASE, WIN]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(methods, mae, color=colors, height=0.62)
    ax.axvline(89.9, color=BASE, ls="--", lw=1.3, zorder=0)
    ax.text(89.9, 3.65, " single-prompt baseline", color=BASE, fontsize=9, va="center")
    for b, m in zip(bars, mae):
        ax.text(b.get_width() - 4, b.get_y() + b.get_height() / 2, f"{m:.0f}",
                ha="right", va="center", color="white", fontweight="bold")
    ax.set_xlabel("Mean error vs lab-measured calories, kcal  (lower is better)")
    ax.set_title("Decomposition alone loses to a single prompt — the blend wins")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 155)
    fig.tight_layout()
    fig.savefig(DOCS / "hero_results.png", bbox_inches="tight")
    plt.close(fig)


def predicted_vs_measured():
    t, o, _llm, comb = load()
    lim = max(max(t), max(o), max(comb)) * 1.08
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot([0, lim], [0, lim], "--", color="#aab4bd", lw=1, zorder=0, label="perfect")
    ax.scatter(t, o, c=BASE, s=46, alpha=0.85, label="single prompt (MAE 90)")
    ax.scatter(t, comb, c=WIN, s=46, alpha=0.9, label="combined (MAE 86)")
    ax.set_xlabel("measured calories (Nutrition5k ground truth)")
    ax.set_ylabel("predicted calories")
    ax.set_title("Closer to the line is better")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "predicted_vs_measured.png", bbox_inches="tight")
    plt.close(fig)


def noise_vs_signal():
    labels = ["How much meals\nvary (the signal)", "Single-prompt\nerror", "Decomposed\nerror"]
    vals = [115, 111, 157]
    colors = ["#5b6b7a", BASE, PIPE]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.axhline(115, color="#5b6b7a", ls="--", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v}", ha="center", fontweight="bold")
    ax.set_ylabel("standard deviation, kcal")
    ax.set_title("Why decomposition loses: its error is noisier than the signal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 180)
    ax.text(2, 168, "an estimate noisier than the thing it estimates\nloses to guessing the average",
            ha="center", fontsize=9, color="#5b6b7a")
    fig.tight_layout()
    fig.savefig(DOCS / "noise_vs_signal.png", bbox_inches="tight")
    plt.close(fig)


def token_cost():
    labels = ["single\nprompt", "decomposed\npipeline", "combined\n(naive)", "combined\n(fused, shipped)"]
    vals = [1.00, 1.66, 2.66, 1.71]
    colors = [BASE, PIPE, MUTED, WIN]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}x", ha="center", fontweight="bold")
    ax.set_ylabel("tokens, relative to a single prompt")
    ax.set_title("The token cost, defused: encode the image once, not twice")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 3.0)
    ax.text(3, 2.4, "USDA grounding = 0 tokens\n(an HTTP lookup, not a model)", ha="center",
            fontsize=9, color="#5b6b7a")
    fig.tight_layout()
    fig.savefig(DOCS / "token_cost.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    recipe_ablation()
    hero_results()
    predicted_vs_measured()
    noise_vs_signal()
    token_cost()
    print("wrote charts to", DOCS)
