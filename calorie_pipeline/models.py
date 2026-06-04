"""Typed dataclasses that act as the contracts between stages.

Each stage of the pipeline consumes one of these types and emits the next. They
are the seams that make the pipeline *inspectable*: you can print, serialize, or
assert on the boundary between any two stages without reaching inside either.

Why dataclasses and not loose dicts? Because the whole argument of this project
is that the value lives in the *interfaces between stages*, not inside any one
model call. A dict says "trust me." A typed contract says "here is exactly what
crosses this boundary, and a checker can prove it."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Ingredient:
    """A single food item, identified and portioned by the vision stage.

    ``name`` is intentionally specific ("scrambled eggs", not "eggs") so it is
    searchable against a nutrition database. ``grams`` is an estimated portion.
    ``prep`` captures preparation that hints at hidden calories (fried, glazed).
    """

    name: str
    grams: float
    prep: str | None = None


@dataclass(frozen=True, slots=True)
class GroundedIngredient:
    """An :class:`Ingredient` after deterministic USDA lookup.

    ``kcal`` is energy scaled to the portion (``kcal_per_100g * grams / 100``),
    or ``None`` when the database had no usable match — we never invent a number
    to paper over a miss. ``fdc_description`` and ``fdc_id`` preserve *what we
    actually matched against* so any estimate can be audited back to its source.
    """

    ingredient: Ingredient
    kcal: float | None
    kcal_per_100g: float | None
    fdc_description: str | None
    fdc_id: int | None

    @property
    def matched(self) -> bool:
        return self.kcal is not None


@dataclass(frozen=True, slots=True)
class Adjustment:
    """A hidden-calorie correction proposed by the reasoning stage.

    Captures calories a raw database lookup structurally misses — cooking oil,
    butter, added sugar, sauces, breading. Always a *range*, because the model
    is being asked to estimate something genuinely unobservable in the photo.
    """

    reason: str
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class Estimate:
    """The final decomposed result: an honest interval, never false precision."""

    grounded: tuple[GroundedIngredient, ...]
    adjustments: tuple[Adjustment, ...]
    base_kcal: float  # sum of matched grounded kcal (deterministic)
    low: float
    high: float
    point: float

    @property
    def unmatched(self) -> tuple[str, ...]:
        """Names of ingredients the database could not ground — surfaced, not hidden."""
        return tuple(g.ingredient.name for g in self.grounded if not g.matched)

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class OneShotEstimate:
    """The baseline: the same vision model asked for a single total directly.

    ``kcal`` is a single number — the false precision the thesis critiques.
    ``raw`` keeps the model's raw response for debugging and honesty.
    """

    kcal: float | None
    raw: str


@dataclass(frozen=True, slots=True)
class Comparison:
    """Both methods run on one photo, with wall-clock timings for each.

    ``combined_kcal`` is the variance-weighted blend of the one-shot and the
    decomposed point estimate — the actual recommended answer, since on the
    benchmark the blend beats either method alone (their errors are only weakly
    correlated). ``None`` only if the one-shot produced no parseable number.
    """

    image_path: str
    oneshot: OneShotEstimate
    oneshot_seconds: float
    estimate: Estimate
    pipeline_seconds: float
    combined_kcal: float | None = None


def combine_estimates(
    oneshot_kcal: float | None,
    pipeline_point: float,
    weight_oneshot: float,
    clamp_band: float = 0.0,
) -> float:
    """Combine the monolithic and decomposed estimates into the final answer.

    Two estimators with weakly-correlated errors beat either alone — the
    decomposition is an independent correction on the one-shot's regress-to-the-
    prior bias. But the decomposition occasionally *blows up* (compounded part-
    errors), so we first clamp it to within ``clamp_band`` of the one-shot, using
    the stable cheap estimate as a sanity bound, then take a weighted average.

    Falls back to the pipeline point if the one-shot produced no number — exactly
    the case where the monolith is worthless and the decomposition's graceful
    degradation matters most.
    """
    if oneshot_kcal is None:
        return pipeline_point
    point = pipeline_point
    if clamp_band > 0.0:
        lo = (1.0 - clamp_band) * oneshot_kcal
        hi = (1.0 + clamp_band) * oneshot_kcal
        point = max(lo, min(hi, point))
    w = max(0.0, min(1.0, weight_oneshot))
    return w * oneshot_kcal + (1.0 - w) * point


def to_json(obj: object, *, indent: int | None = 2) -> str:
    """Serialize any contract dataclass (or container of them) to JSON.

    Tuples become arrays; nested dataclasses are expanded. Handy for snapshot
    tests and for the CLI's ``--json`` mode.
    """

    def _default(o: object) -> object:
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)  # type: ignore[arg-type]
        raise TypeError(f"not serializable: {type(o).__name__}")

    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)  # type: ignore[arg-type]
    return json.dumps(obj, indent=indent, default=_default)
