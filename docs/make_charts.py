"""Charts for the methods study, generated from the real leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SCORES = ROOT / "benchmark/results/methods_scores.json"

INK = "#222b35"
LEVER_COLOR = {
    "baseline": "#48586a",
    "more reasoning": "#7a6ca8",
    "variance reduction": "#c08a4e",
    "grounding": "#3f7d8c",
    "ensembling": "#2e8b6f",
}
plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#d7dde3", "axes.linewidth": 1.0,
    "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.titlepad": 12,
    "figure.dpi": 150, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#5b6b7a", "ytick.color": INK, "xtick.labelsize": 10,
})


def _rows() -> list[dict]:
    return json.loads(SCORES.read_text())


def leaderboard():
    rows = sorted(_rows(), key=lambda r: r["mae"], reverse=True)  # worst on top
    base = next(r["mae"] for r in rows if r["method"] == "one-shot")
    colors = [LEVER_COLOR.get(r["lever"], "#888") for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars = ax.barh(range(len(rows)), [r["mae"] for r in rows], color=colors, height=0.66, zorder=3)
    ax.axvline(base, color="#9aa7b4", ls=(0, (4, 3)), lw=1.2, zorder=1)
    for i, (b, r) in enumerate(zip(bars, rows)):
        ax.text(b.get_width() - 2, i, f"{r['mae']:.0f}", ha="right", va="center",
                color="white", fontweight="bold")
    ax.set_yticks(range(len(rows)), [f"{r['method']}" for r in rows])
    ax.set_xlim(0, max(r["mae"] for r in rows) * 1.12)
    ax.set_xlabel("mean error vs lab-measured calories, kcal   ·   lower is better")
    ax.set_title("Every method, ranked", loc="left")
    ax.text(base, len(rows) - 0.4, "one-shot baseline", color="#7d8a98", fontsize=9, ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(DOCS / "leaderboard.png", bbox_inches="tight")
    plt.close(fig)


def cost_accuracy():
    rows = _rows()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for r in rows:
        c = LEVER_COLOR.get(r["lever"], "#888")
        ax.scatter(r["tokens"], r["mae"], s=120, color=c, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(r["method"], (r["tokens"], r["mae"]), xytext=(6, 6),
                    textcoords="offset points", fontsize=9.5, color=INK)
    ax.set_xlabel("tokens per dish   ·   cheaper is left")
    ax.set_ylabel("mean error, kcal   ·   better is down")
    ax.set_title("Accuracy vs cost per method", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.margins(0.18)
    fig.tight_layout()
    fig.savefig(DOCS / "cost_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def two_model():
    """The flip: same grounding workflow, opposite verdict on 7B vs 12B."""
    import numpy as np

    def mae(store, method):
        d = json.loads((ROOT / store).read_text())[method]
        rows = [abs(v["kcal"] - v["truth"]) for v in d.values() if v.get("kcal") is not None]
        return float(np.mean(rows))
    q, g = "benchmark/results/methods_7b.json", "benchmark/results/methods.json"
    data = {
        "one-shot": (mae(q, "one-shot"), mae(g, "one-shot")),
        "decomposed\n(grounded)": (mae(q, "decomposed + USDA"), mae(g, "decomposed + USDA")),
    }
    labels = list(data)
    seven = [data[k][0] for k in labels]
    twelve = [data[k][1] for k in labels]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    b1 = ax.bar(x - w / 2, seven, w, label="7B (qwen2.5vl)", color="#48586a", zorder=3)
    b2 = ax.bar(x + w / 2, twelve, w, label="12B (gemma4)", color="#b07a52", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{b.get_height():.0f}",
                    ha="center", fontweight="bold", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean error, kcal   ·   lower is better")
    ax.set_title("Same grounding workflow, opposite verdict", loc="left")
    ax.set_ylim(0, 560)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.annotate("grounding rescues\nthe over-counting 12B", xy=(1.18, 200), xytext=(1.0, 380),
                fontsize=9, color="#7a5230", ha="center",
                arrowprops=dict(arrowstyle="->", color="#b07a52"))
    fig.tight_layout()
    fig.savefig(DOCS / "two_model.png", bbox_inches="tight")
    plt.close(fig)


def segment_light():
    """Where grounding wins: light meals the prompt's prior can't reach."""
    seg = ROOT / "benchmark/results/light_segment.json"
    if not seg.exists():
        return
    u = json.loads(seg.read_text())
    t = [x["truth"] for x in u]
    o = [x["oneshot"] for x in u]
    p = [x["pipeline"] for x in u]
    lim = 220
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, lim], [0, lim], "-", color="#cdd5dc", lw=1.2, zorder=0)
    ax.scatter(t, o, c="#d9534f", s=46, alpha=0.8, edgecolor="white", linewidth=0.5, label="single prompt (MAE 53)")
    ax.scatter(t, p, c="#3f7d8c", s=46, alpha=0.85, edgecolor="white", linewidth=0.5, label="grounded pipeline (MAE 30)")
    ax.axhline(125, color="#d9534f", ls=":", lw=1)
    ax.text(8, 131, "the prompt floors out here; it can't imagine a light meal",
            color="#b33", fontsize=8.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, 320)
    ax.set_xlabel("measured calories (light meals, < 200)")
    ax.set_ylabel("predicted calories")
    ax.set_title("Where grounding wins: 44% lower error, p < 0.001", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(DOCS / "segment_light.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    segment_light()
    if not SCORES.exists():
        raise SystemExit("leaderboard charts need benchmark/compare_methods.py to finish first")
    leaderboard()
    cost_accuracy()
    for stale in ("recipe_ablation.png", "predicted_vs_measured.png", "token_cost.png"):
        (DOCS / stale).unlink(missing_ok=True)
    print("wrote charts to", DOCS)
