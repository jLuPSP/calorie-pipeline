"""calorie_pipeline — decomposition beats model size.

A multi-stage calorie estimator that runs entirely on local 7B models (via
Ollama) plus a deterministic nutrition database. The thesis: a decomposed
pipeline produces better-calibrated, auditable estimates than the *same* local
model asked to one-shot a food photo. Accuracy comes from architecture, not
model strength.

The four stages are deliberately inspectable:

    vision   -> identify foods and portion them            (model: judgment)
    lookup   -> ground each food in USDA FoodData Central   (no model: facts)
    reason   -> probe for hidden calories the DB misses     (model: judgment)
    aggregate-> sum base + ranges into an honest interval   (no model: math)

See ``calorie_pipeline.pipeline`` for the orchestration and
``calorie_pipeline.run`` for the side-by-side CLI.
"""

from __future__ import annotations

from calorie_pipeline.config import Config
from calorie_pipeline.models import (
    Adjustment,
    Comparison,
    Estimate,
    GroundedIngredient,
    Ingredient,
    OneShotEstimate,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Ingredient",
    "GroundedIngredient",
    "Adjustment",
    "Estimate",
    "OneShotEstimate",
    "Comparison",
    "__version__",
]
