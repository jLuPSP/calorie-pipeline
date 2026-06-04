"""CLI — run the one-shot baseline and the decomposed pipeline side by side.

    python -m calorie_pipeline.run path/to/meal.jpg
    python -m calorie_pipeline.run path/to/meal.jpg --json

The rendering functions are pure (``Comparison``/``Estimate`` -> ``str``) so the
formatting is unit-tested with simulated stage data — no live model required.
"""

from __future__ import annotations

import argparse
import sys

from calorie_pipeline.config import Config
from calorie_pipeline.models import Comparison, Estimate, OneShotEstimate, to_json


def render_estimate(estimate: Estimate) -> str:
    """Render the decomposed estimate as an inspectable, auditable report."""
    lines: list[str] = []
    lines.append("DECOMPOSED PIPELINE")
    lines.append("  Grounded ingredients (USDA FoodData Central):")
    for g in estimate.grounded:
        if g.matched:
            src = g.fdc_description or "matched"
            lines.append(
                f"    - {g.ingredient.name} ({g.ingredient.grams:.0f} g)"
                f" -> {g.kcal:.0f} kcal  [{src}]"
            )
        else:
            lines.append(
                f"    - {g.ingredient.name} ({g.ingredient.grams:.0f} g)"
                f" -> no USDA match (excluded from base)"
            )
    lines.append(f"  Base (database) total: {estimate.base_kcal:.0f} kcal")

    if estimate.adjustments:
        lines.append("  Hidden-calorie adjustments (model judgment, as ranges):")
        for a in estimate.adjustments:
            lines.append(f"    + {a.reason}: {a.low:.0f}-{a.high:.0f} kcal")
    else:
        lines.append("  Hidden-calorie adjustments: none proposed")

    if estimate.unmatched:
        lines.append(f"  Unmatched (surfaced, not hidden): {', '.join(estimate.unmatched)}")

    lines.append(
        f"  ESTIMATE: {estimate.low:.0f}-{estimate.high:.0f} kcal"
        f"  (midpoint {estimate.point:.0f})"
    )
    return "\n".join(lines)


def render_oneshot(oneshot: OneShotEstimate) -> str:
    """Render the one-shot baseline."""
    if oneshot.kcal is None:
        body = "  ESTIMATE: (model returned no parseable number)"
    else:
        body = f"  ESTIMATE: {oneshot.kcal:.0f} kcal  (single point, no provenance)"
    return "ONE-SHOT BASELINE (same model, whole job at once)\n" + body


def render_comparison(comparison: Comparison) -> str:
    """Render both methods with timings — the side-by-side the thesis lives in."""
    sep = "=" * 68
    parts = [
        sep,
        f"IMAGE: {comparison.image_path}",
        sep,
        render_oneshot(comparison.oneshot),
        f"  time: {comparison.oneshot_seconds:.1f}s",
        "",
        render_estimate(comparison.estimate),
        f"  time: {comparison.pipeline_seconds:.1f}s",
        sep,
    ]
    if comparison.combined_kcal is not None:
        parts.append(
            f"RECOMMENDED (variance-weighted blend of both): "
            f"{comparison.combined_kcal:.0f} kcal"
        )
        parts.append(sep)
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m calorie_pipeline.run",
        description="Estimate meal calories: one-shot baseline vs decomposed pipeline.",
    )
    parser.add_argument("image", help="path to a meal photo")
    parser.add_argument(
        "--json", action="store_true", help="emit the full Comparison as JSON"
    )
    args = parser.parse_args(argv)

    config = Config.from_env()
    # Imported here so --help and arg errors don't require ollama installed.
    from calorie_pipeline.pipeline import compare

    try:
        comparison = compare(args.image, config)
    except FileNotFoundError:
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    if args.json:
        print(to_json(comparison))
    else:
        print(render_comparison(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
