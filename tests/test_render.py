"""Rendering — the CLI output, exercised with simulated stage data."""

import unittest

from calorie_pipeline.models import (
    Adjustment,
    Comparison,
    GroundedIngredient,
    Ingredient,
    OneShotEstimate,
)
from calorie_pipeline.pipeline import aggregate
from calorie_pipeline.run import render_comparison, render_estimate, render_oneshot


def _estimate() -> object:
    grounded = [
        GroundedIngredient(Ingredient("rice", 200), 260.0, 130.0, "Rice, cooked", 1),
        GroundedIngredient(Ingredient("mystery sauce", 30), None, None, None, None),
    ]
    return aggregate(grounded, [Adjustment("cooking oil", 50, 150)])


class TestRender(unittest.TestCase):
    def test_estimate_report_is_auditable(self) -> None:
        text = render_estimate(_estimate())
        self.assertIn("Rice, cooked", text)          # provenance shown
        self.assertIn("cooking oil", text)            # adjustment shown
        self.assertIn("no USDA match", text)          # miss surfaced, not hidden
        self.assertIn("mystery sauce", text)
        self.assertIn("310-410 kcal", text)           # 260 + [50,150]

    def test_oneshot_flags_false_precision(self) -> None:
        text = render_oneshot(OneShotEstimate(kcal=640.0, raw="{}"))
        self.assertIn("640 kcal", text)
        self.assertIn("no provenance", text)

    def test_oneshot_handles_unparseable(self) -> None:
        text = render_oneshot(OneShotEstimate(kcal=None, raw="hmm"))
        self.assertIn("no parseable number", text)

    def test_comparison_shows_both_methods_and_timings(self) -> None:
        comp = Comparison(
            image_path="meal.jpg",
            oneshot=OneShotEstimate(640.0, "{}"),
            oneshot_seconds=1.2,
            estimate=_estimate(),
            pipeline_seconds=3.4,
        )
        text = render_comparison(comp)
        self.assertIn("meal.jpg", text)
        self.assertIn("ONE-SHOT BASELINE", text)
        self.assertIn("DECOMPOSED PIPELINE", text)
        self.assertIn("1.2s", text)
        self.assertIn("3.4s", text)


if __name__ == "__main__":
    unittest.main()
