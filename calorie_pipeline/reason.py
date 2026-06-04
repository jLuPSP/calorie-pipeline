"""Stage 3 — Hidden-variable probe: the calories the database cannot see.

The lookup stage grounds *visible, named* foods. But a photo hides calories that
are real and large: the oil a vegetable was sautéed in, the butter on the toast,
the sugar in the glaze, the breading's absorbed fat, the dressing under the
salad. No database lookup of "broccoli" will ever surface the tablespoon of oil.

This is the one place in the pipeline where a genuine judgment is required —
estimating an unobservable from context — so it is the one place we spend a
model. And because it is a judgment about something invisible, its output is a
*range*, never a point. The model is told what it is already standing on (the
grounded base) so it adds only what is missing and does not double-count.
"""

from __future__ import annotations

import json
from typing import Any

from calorie_pipeline.config import Config
from calorie_pipeline.models import Adjustment, GroundedIngredient

_SYSTEM_PROMPT = """You audit meal calorie estimates for HIDDEN calories that a
food-database lookup structurally misses.

You are given a list of foods whose base calories have ALREADY been counted from
a database. Your job is to add ONLY the calories that such a lookup cannot see:
- cooking oil, butter, ghee, or fat foods were cooked in
- added sugar, honey, syrup, or glaze
- sauces, dressings, mayonnaise, gravy
- breading or batter and the oil it absorbs

Rules:
- Do NOT restate or recount the base foods. Their calories are already counted.
- Do NOT double-count. If oil is already implied by a "fried" item's database
  entry, do not add it again.
- Only add calories genuinely plausible for THESE specific foods. Do NOT invent a
  cooking method that is not implied (e.g. a cold deli sandwich was not pan-fried
  in butter; raw salad greens were not sauteed).
- Estimate conservatively and express genuine uncertainty as a low/high range.
- If nothing is plausibly hidden, return an empty list.

Respond with ONLY this JSON object:
{"adjustments": [{"reason": "...", "low": 0, "high": 0}]}
"""


def probe_hidden_calories(
    grounded: list[GroundedIngredient], config: Config
) -> list[Adjustment]:
    """Ask the text model for hidden-calorie adjustments over the grounded list."""
    from ollama import Client  # lazy import

    client = Client(host=config.ollama_host)
    response = client.chat(
        model=config.text_model,
        format="json",
        options={"temperature": config.reason_temperature},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": render_grounded(grounded)},
        ],
    )
    return parse_adjustments(response["message"]["content"])


def render_grounded(grounded: list[GroundedIngredient]) -> str:
    """Render the grounded ingredients into the compact brief the model reasons over."""
    lines = ["Foods on the plate (base calories already counted):"]
    for g in grounded:
        prep = f", {g.ingredient.prep}" if g.ingredient.prep else ""
        if g.matched:
            kcal = f"{g.kcal:.0f} kcal"
        else:
            kcal = "no database match"
        lines.append(
            f"- {g.ingredient.name} ({g.ingredient.grams:.0f} g{prep}): {kcal}"
        )
    lines.append(
        "\nList only ADDITIONAL hidden calories not already in the numbers above."
    )
    return "\n".join(lines)


def parse_adjustments(raw: str) -> list[Adjustment]:
    """Parse the model's JSON into validated :class:`Adjustment` ranges.

    Drops malformed entries and normalizes inverted ranges (low > high). Pure
    function — unit-tested without a live model.
    """
    data = json.loads(raw)
    items = _coerce_to_list(data)

    adjustments: list[Adjustment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        reason = item.get("reason")
        low = _coerce_float(item.get("low"))
        high = _coerce_float(item.get("high"))
        if not reason or low is None or high is None:
            continue
        low, high = min(low, high), max(low, high)
        adjustments.append(Adjustment(reason=str(reason).strip(), low=low, high=high))
    return adjustments


def _coerce_to_list(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("adjustments", "items", "hidden", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if "reason" in data:
            return [data]
    return []


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().split()[0])
        except (ValueError, IndexError):
            return None
    return None
