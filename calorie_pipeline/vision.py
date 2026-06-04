"""Stage 1 — Vision: food photo -> portioned ingredient list.

This stage has exactly one job: look at the photo and say *what* is on the plate
and roughly *how much*. It does not estimate calories. It does not editorialize.
It emits JSON and nothing else.

Narrowing the job is the whole trick. A vision model is good at "that's a fried
chicken thigh, ~120 g" and bad at "that meal is 640 kcal" — the first is
perception, the second smuggles in arithmetic and a memorized food database the
model only approximates. We take the perception and hand the arithmetic to code.
"""

from __future__ import annotations

import json
from typing import Any

from calorie_pipeline.config import Config
from calorie_pipeline.models import Ingredient

_SYSTEM_PROMPT = """You are a food portioning expert. You are shown one photo of a meal.

Break the meal into its INDIVIDUAL constituent foods — the separate single-food
ingredients a nutrition database would list — NOT the name of the assembled dish.

CRITICAL: never output a composite/assembled dish as one item. Decompose it into
its parts, and portion each part separately:
- a sandwich  -> bread, sliced deli turkey, cheese, lettuce, tomato, onion
- a slice of pizza -> pizza crust, mozzarella cheese, tomato sauce, pepperoni
- a burger    -> hamburger bun, beef patty, cheese, lettuce, tomato
- a salad     -> each vegetable, the protein, and each topping, separately
A single base food that is not assembled (a bagel, an apple, a boiled egg) is
just itself: emit one item with that plain name.

For each constituent food, provide:
- "name": a SPECIFIC, database-searchable SINGLE-FOOD name. Good: "whole wheat
  bread", "sliced deli turkey breast", "mozzarella cheese", "white rice, cooked".
  Bad: "sandwich", "pizza slice", "burger" (these are dishes, not foods).
- "grams": estimated edible weight of THAT component in grams (a number).
- "prep": preparation if visible (toasted, fried, grilled, raw), else null.

Do NOT estimate calories. Do NOT add commentary, units, or fields.
Respond with ONLY this JSON object:
{"ingredients": [{"name": "...", "grams": 0, "prep": "..."}]}
"""


def extract_ingredients(image_path: str, config: Config) -> list[Ingredient]:
    """Run the vision model on ``image_path`` and return portioned ingredients.

    Raises ``FileNotFoundError`` if the image is missing and ``ValueError`` if
    the model returns something we cannot parse into ingredients.
    """
    from ollama import Client  # lazy: keeps the package importable without ollama

    client = Client(host=config.ollama_host)
    response = client.chat(
        model=config.vision_model,
        format="json",
        options={"temperature": config.vision_temperature},
        messages=[
            {
                "role": "user",
                "content": _SYSTEM_PROMPT,
                "images": [image_path],
            }
        ],
    )
    content = response["message"]["content"]
    return parse_ingredients(content)


_FUSED_PROMPT = _SYSTEM_PROMPT + (
    '\nAlso include, in the SAME JSON object, your single best estimate of the '
    'whole meal\'s TOTAL calories as a number under the key "total_kcal".'
)


def extract_total_and_ingredients(
    image_path: str, config: Config
) -> tuple[float | None, list[Ingredient]]:
    """One vision call returning BOTH a one-shot total and the breakdown.

    The image is the expensive part of a vision prompt — it encodes to hundreds of
    tokens — so a combined system that wants both a one-shot number and an itemized
    decomposition should ask for them in a *single* pass rather than encode the
    image twice. This is the token optimization behind the efficient production
    path (``pipeline.estimate_fused``); the benchmark keeps the two calls separate
    only so it can score each method cleanly.
    """
    from ollama import Client  # lazy import

    response = Client(host=config.ollama_host).chat(
        model=config.vision_model,
        format="json",
        options={"temperature": config.vision_temperature},
        messages=[{"role": "user", "content": _FUSED_PROMPT, "images": [image_path]}],
    )
    content = response["message"]["content"]
    return _parse_total(content), parse_ingredients(content)


def _parse_total(raw: str) -> float | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return _coerce_float(data.get("total_kcal")) if isinstance(data, dict) else None


def parse_ingredients(raw: str) -> list[Ingredient]:
    """Parse the model's JSON into a list of :class:`Ingredient`.

    Tolerant of the two shapes a JSON-mode model tends to emit: a bare list, or
    an object wrapping the list under a key. Skips entries missing a name or a
    parseable weight rather than fabricating values. Pure function — unit-tested
    without a live model.
    """
    data = json.loads(raw)
    items = _coerce_to_list(data)

    ingredients: list[Ingredient] = []
    seen: set[tuple[str, float, str | None]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        grams = _coerce_float(item.get("grams"))
        if not name or grams is None:
            continue
        prep = item.get("prep")
        ingredient = Ingredient(
            name=str(name).strip(),
            grams=grams,
            prep=str(prep).strip() if prep else None,
        )
        # Drop exact duplicates. JSON-mode 7B models sometimes repeat a whole
        # sub-list verbatim (e.g. a fruit salad listed twice), which would
        # double-count calories downstream. Identical (name, grams, prep) entries
        # are far more likely a model repeat than two genuinely separate portions.
        key = (ingredient.name.lower(), ingredient.grams, ingredient.prep)
        if key in seen:
            continue
        seen.add(key)
        ingredients.append(ingredient)
    return ingredients


def _coerce_to_list(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Accept the documented {"ingredients": [...]} and a few likely synonyms.
        for key in ("ingredients", "items", "foods", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        # A single object that *is* an ingredient.
        if "name" in data:
            return [data]
    return []


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().split()[0])  # tolerate "120 g"
        except (ValueError, IndexError):
            return None
    return None
