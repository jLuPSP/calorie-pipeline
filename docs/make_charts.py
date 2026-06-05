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


def gemma_result():
    """The aggregate: one-shot vs workflow on the 12B, across all 24 dishes."""
    import numpy as np

    def mae(method):
        d = json.loads((ROOT / "benchmark/results/methods.json").read_text())[method]
        return float(np.mean([abs(v["kcal"] - v["truth"]) for v in d.values() if v.get("kcal") is not None]))
    one, work = mae("one-shot"), mae("decomposed + USDA")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(["one-shot", "workflow\n(grounded)"], [one, work],
                  color=["#d9534f", "#2e8b6f"], width=0.55, zorder=3)
    for b, v in zip(bars, [one, work]):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.0f}", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylabel("mean error, kcal   ·   lower is better")
    ax.set_title(f"gemma4 12B, 24 dishes: the workflow cuts error {100*(one-work)/one:.0f}%", loc="left")
    ax.set_ylim(0, one * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(DOCS / "gemma_result.png", bbox_inches="tight")
    plt.close(fig)


def worked_example():
    """One dish, both approaches, side by side. The article in a single figure."""
    import matplotlib.image as mpimg

    img = mpimg.imread(ROOT / "benchmark/images/dish_1563302447/rgb.png")
    ledger = [
        ("fried fish (150 g)", "351"),
        ("potato wedges (80 g)", "103"),
        ("carrot, lettuce, mushroom", "29"),
        ("+ oil and butter for frying", "~64"),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 5.0), gridspec_kw={"width_ratios": [1, 1.35]})
    axL.imshow(img)
    axL.set_title("the photo", loc="left", fontsize=12)
    axL.axis("off")

    axR.axis("off")
    axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    # one-shot
    axR.text(0, 0.95, "ONE-SHOT", fontsize=12, fontweight="bold", color="#d9534f")
    axR.text(0.30, 0.95, "ask the 12B for the total", fontsize=10, color="#7d8a98", va="center")
    axR.text(0.02, 0.84, "1,250 kcal", fontsize=22, fontweight="bold", color="#d9534f")
    axR.text(0.46, 0.85, "wrong by 689", fontsize=10, color="#d9534f", va="center")
    axR.axhline(0.74, 0, 1, color="#e3e8ec", lw=1)
    # workflow
    axR.text(0, 0.68, "WORKFLOW", fontsize=12, fontweight="bold", color="#2e8b6f")
    axR.text(0.34, 0.68, "see the food, look up each calorie, sum", fontsize=10, color="#7d8a98", va="center")
    y = 0.58
    for name, kc in ledger:
        axR.text(0.02, y, name, fontsize=10.5, family="DejaVu Sans")
        axR.text(0.62, y, kc, fontsize=10.5, fontweight="bold", ha="right")
        y -= 0.075
    axR.axhline(y + 0.03, 0.02, 0.64, color="#cdd5dc", lw=1)
    axR.text(0.02, y - 0.04, "547 kcal", fontsize=22, fontweight="bold", color="#2e8b6f")
    axR.text(0.46, y - 0.03, "off by 14", fontsize=10, color="#2e8b6f", va="center")
    axR.text(0, 0.02, "measured truth: 561 kcal", fontsize=11, color="#48586a", fontweight="bold")
    fig.suptitle("Same 12B model, same photo: one-shot vs a grounded workflow",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DOCS / "worked_example.png", bbox_inches="tight")
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
    # The focused article uses three figures: the worked example, the aggregate
    # result, and the two-model caveat. The leaderboard/segment functions above
    # remain available for the fuller survey, but are not part of the main piece.
    worked_example()
    gemma_result()
    two_model()
    for stale in ("leaderboard.png", "cost_accuracy.png", "segment_light.png",
                  "recipe_ablation.png", "predicted_vs_measured.png", "token_cost.png"):
        (DOCS / stale).unlink(missing_ok=True)
    print("wrote charts to", DOCS)
