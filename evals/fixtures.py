"""Deterministic offline fixtures — the always-runnable proof.

These are hand-authored, internally-consistent stage outputs for a handful of
common meals. Per-100 g energies are USDA-plausible; portions and hidden-calorie
amounts are illustrative. They are **not** Nutrition5k measurements — their job
is to exercise the aggregation and metrics with zero network or model, and to
show the *idealized* case: what the pipeline looks like when every stage behaves
(matches are right, portions are right, the probe brackets the hidden calories).

IMPORTANT: this idealized case is NOT the live result. On the real benchmark
(``benchmark/``) the one-shot baseline actually *wins* on accuracy, because real
matching and portioning are noisy and a decomposed estimate compounds that noise.
These fixtures prove the math and the scoring harness; they do not prove the
thesis. See ``benchmark/README.md`` and the blog post for the honest result.
"""

from __future__ import annotations

from dataclasses import dataclass

from calorie_pipeline.models import Adjustment, GroundedIngredient, Ingredient


@dataclass(frozen=True, slots=True)
class FixtureDish:
    """A simulated dish: ground truth plus what each stage would have emitted."""

    dish_id: str
    ground_truth_kcal: float
    grounded: tuple[GroundedIngredient, ...]
    adjustments: tuple[Adjustment, ...]
    oneshot_kcal: float
    note: str


def _g(
    name: str,
    grams: float,
    kcal_per_100g: float,
    description: str,
    prep: str | None = None,
) -> GroundedIngredient:
    """Build a grounded ingredient with energy scaled to the portion."""
    return GroundedIngredient(
        ingredient=Ingredient(name=name, grams=grams, prep=prep),
        kcal=kcal_per_100g * grams / 100.0,
        kcal_per_100g=kcal_per_100g,
        fdc_description=description,
        fdc_id=None,
    )


def fixture_dishes() -> list[FixtureDish]:
    """The eight illustrative dishes scored by the offline harness."""
    return [
        FixtureDish(
            dish_id="chicken_rice_broccoli",
            ground_truth_kcal=648.0,
            grounded=(
                _g("grilled chicken breast", 150, 165, "Chicken, breast, grilled", "grilled"),
                _g("white rice, cooked", 200, 130, "Rice, white, cooked"),
                _g("broccoli, steamed", 90, 34, "Broccoli, cooked", "steamed"),
            ),
            adjustments=(Adjustment("cooking oil on broccoli and rice", 60, 150),),
            oneshot_kcal=540.0,
            note="oil the photo cannot show; one-shot under-counts it",
        ),
        FixtureDish(
            dish_id="caesar_salad",
            ground_truth_kcal=621.0,
            grounded=(
                _g("romaine lettuce", 100, 17, "Lettuce, cos or romaine, raw"),
                _g("grilled chicken breast", 120, 165, "Chicken, breast, grilled", "grilled"),
                _g("parmesan cheese", 20, 431, "Cheese, parmesan, grated"),
                _g("croutons", 30, 465, "Croutons, seasoned"),
            ),
            adjustments=(Adjustment("caesar dressing", 120, 250),),
            oneshot_kcal=350.0,
            note="the 'salad = healthy' prior; one-shot badly under-counts dressing",
        ),
        FixtureDish(
            dish_id="pancakes_butter_syrup",
            ground_truth_kcal=650.0,
            grounded=(
                _g("buttermilk pancakes", 150, 227, "Pancakes, plain, prepared"),
            ),
            adjustments=(
                Adjustment("butter", 50, 130),
                Adjustment("maple syrup", 120, 240),
            ),
            oneshot_kcal=480.0,
            note="toppings dominate and are invisible to a database lookup",
        ),
        FixtureDish(
            dish_id="spaghetti_marinara",
            ground_truth_kcal=574.0,
            grounded=(
                _g("spaghetti, cooked", 220, 158, "Pasta, cooked"),
                _g("marinara sauce", 120, 60, "Sauce, marinara"),
                _g("parmesan cheese", 15, 431, "Cheese, parmesan, grated"),
            ),
            adjustments=(Adjustment("olive oil in sauce and on pasta", 50, 140),),
            oneshot_kcal=700.0,
            note="one-shot over-counts a carb-heavy plate",
        ),
        FixtureDish(
            dish_id="fried_rice",
            ground_truth_kcal=720.5,
            grounded=(
                _g("white rice, cooked", 250, 130, "Rice, white, cooked", "fried"),
                _g("scrambled egg", 50, 155, "Egg, cooked"),
                _g("peas and carrots", 40, 70, "Peas and carrots, cooked"),
            ),
            adjustments=(Adjustment("stir-fry oil and sauces", 120, 260),),
            oneshot_kcal=500.0,
            note="DELIBERATE miss: true oil exceeds the probe's high bound (under-ranged)",
        ),
        FixtureDish(
            dish_id="yogurt_granola_berries",
            ground_truth_kcal=350.0,
            grounded=(
                _g("greek yogurt, plain", 200, 59, "Yogurt, Greek, plain, nonfat"),
                _g("blueberries", 80, 57, "Blueberries, raw"),
                _g("granola", 30, 471, "Granola, plain"),
            ),
            adjustments=(Adjustment("honey drizzle", 20, 70),),
            oneshot_kcal=250.0,
            note="low hidden calories; the probe correctly adds little",
        ),
        FixtureDish(
            dish_id="cheeseburger",
            ground_truth_kcal=693.0,
            grounded=(
                _g("beef patty", 110, 250, "Beef, ground, cooked", "grilled"),
                _g("hamburger bun", 80, 265, "Bread, hamburger roll"),
                _g("cheddar cheese", 20, 403, "Cheese, cheddar"),
                _g("lettuce and tomato", 30, 18, "Vegetables, raw"),
            ),
            adjustments=(Adjustment("condiments and grill oil", 70, 180),),
            oneshot_kcal=650.0,
            note="a dish one-shot handles well — and the pipeline still matches it",
        ),
        FixtureDish(
            dish_id="salmon_asparagus",
            ground_truth_kcal=502.0,
            grounded=(
                _g("salmon fillet", 170, 208, "Fish, salmon, cooked", "pan-seared"),
                _g("asparagus", 90, 20, "Asparagus, cooked"),
            ),
            adjustments=(Adjustment("butter on salmon and asparagus", 60, 160),),
            oneshot_kcal=620.0,
            note="one-shot over-counts fish; the lookup grounds it precisely",
        ),
    ]
