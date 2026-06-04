"""Orchestration — wire the stages together, and the one-shot baseline to beat.

Two callables matter here:

* :func:`run_pipeline` — the decomposed estimator: vision -> lookup -> reason ->
  aggregate. Each arrow is a typed contract; each stage is independently
  testable; the only model judgments are "what/how much" and "what's hidden".
* :func:`run_oneshot` — the control: the *same* vision model handed the whole
  job at once and asked for a single number. This is what the thesis is measured
  against.

:func:`compare` runs both on one photo with wall-clock timing so the CLI and the
eval harness can put them side by side.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from calorie_pipeline.config import Config
from calorie_pipeline.lookup import lookup_ingredient
from calorie_pipeline.models import (
    Adjustment,
    Comparison,
    Estimate,
    GroundedIngredient,
    Ingredient,
    OneShotEstimate,
    combine_estimates,
)
from calorie_pipeline.reason import probe_hidden_calories
from calorie_pipeline.vision import extract_ingredients, extract_total_and_ingredients

_ONESHOT_PROMPT = """Look at this meal photo and estimate its TOTAL calories.
Respond with ONLY this JSON object: {"kcal": 0}
"""


def aggregate(
    grounded: list[GroundedIngredient], adjustments: list[Adjustment]
) -> Estimate:
    """Stage 4 — sum the grounded base and the adjustment ranges into an interval.

    Deterministic arithmetic, no model. The base is the sum of matched lookups
    (a point, because the database is exact). The adjustments contribute a
    low/high band. The result is a range; the point is its midpoint, reported
    only as a convenience and never as a claim of precision.

    Pure function — this is the math the offline verification exercises.
    """
    base = sum(g.kcal for g in grounded if g.kcal is not None)
    adj_low = sum(a.low for a in adjustments)
    adj_high = sum(a.high for a in adjustments)
    low = base + adj_low
    high = base + adj_high
    point = (low + high) / 2.0
    return Estimate(
        grounded=tuple(grounded),
        adjustments=tuple(adjustments),
        base_kcal=base,
        low=low,
        high=high,
        point=point,
    )


def run_pipeline(
    image_path: str,
    config: Config,
    *,
    on_stage: Callable[[str, Any], None] | None = None,
) -> Estimate:
    """Run the full decomposed pipeline on one photo.

    ``on_stage(name, payload)`` is an optional observer invoked after each stage
    with its output — used by the CLI to stream progress. The pipeline itself
    stays a pure data flow.
    """

    def emit(name: str, payload: Any) -> None:
        if on_stage is not None:
            on_stage(name, payload)

    ingredients: list[Ingredient] = extract_ingredients(image_path, config)
    emit("vision", ingredients)

    grounded = [lookup_ingredient(i, config) for i in ingredients]
    emit("lookup", grounded)

    adjustments = probe_hidden_calories(grounded, config)
    emit("reason", adjustments)

    estimate = aggregate(grounded, adjustments)
    emit("aggregate", estimate)
    return estimate


def run_oneshot(image_path: str, config: Config) -> OneShotEstimate:
    """The baseline: ask the vision model for the total in a single shot."""
    from ollama import Client  # lazy import

    client = Client(host=config.ollama_host)
    response = client.chat(
        model=config.vision_model,
        format="json",
        options={"temperature": config.oneshot_temperature},
        messages=[{"role": "user", "content": _ONESHOT_PROMPT, "images": [image_path]}],
    )
    raw = response["message"]["content"]
    return parse_oneshot(raw)


def parse_oneshot(raw: str) -> OneShotEstimate:
    """Parse the one-shot model response into a single kcal number (or None)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return OneShotEstimate(kcal=None, raw=raw)
    value: Any = data.get("kcal") if isinstance(data, dict) else None
    try:
        kcal = float(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        kcal = None
    return OneShotEstimate(kcal=kcal, raw=raw)


def estimate_fused(image_path: str, config: Config) -> Comparison:
    """The efficient production path: one vision call, then the rest is cheap.

    Identical result shape to :func:`compare`, but the one-shot total and the
    ingredient breakdown come from a *single* vision pass (the image is encoded
    once, not twice), the USDA grounding costs no tokens at all, and only the
    small text "reason" call is added. This is how you get the combined estimate's
    accuracy without paying ~2x the tokens of a single prompt — see
    ``benchmark/measure_tokens.py``.
    """
    t0 = time.perf_counter()
    total, ingredients = extract_total_and_ingredients(image_path, config)
    grounded = [lookup_ingredient(i, config) for i in ingredients]
    adjustments = probe_hidden_calories(grounded, config)
    estimate = aggregate(grounded, adjustments)
    seconds = time.perf_counter() - t0

    return Comparison(
        image_path=image_path,
        oneshot=OneShotEstimate(kcal=total, raw=""),
        oneshot_seconds=0.0,  # shared with the pipeline vision call
        estimate=estimate,
        pipeline_seconds=seconds,
        combined_kcal=combine_estimates(
            total, estimate.point, config.combine_weight, config.combine_clamp
        ),
    )


def compare(image_path: str, config: Config) -> Comparison:
    """Run both methods on one photo, timing each independently."""
    t0 = time.perf_counter()
    oneshot = run_oneshot(image_path, config)
    oneshot_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    estimate = run_pipeline(image_path, config)
    pipeline_seconds = time.perf_counter() - t1

    return Comparison(
        image_path=image_path,
        oneshot=oneshot,
        oneshot_seconds=oneshot_seconds,
        estimate=estimate,
        pipeline_seconds=pipeline_seconds,
        combined_kcal=combine_estimates(
            oneshot.kcal, estimate.point, config.combine_weight, config.combine_clamp
        ),
    )
