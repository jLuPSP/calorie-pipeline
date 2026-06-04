"""The aggregation math — the deterministic core, proven without any model."""

import unittest

from calorie_pipeline.models import (
    Adjustment,
    GroundedIngredient,
    Ingredient,
    combine_estimates,
    to_json,
)
from calorie_pipeline.pipeline import aggregate


def grounded(name: str, grams: float, kcal: float | None) -> GroundedIngredient:
    return GroundedIngredient(
        ingredient=Ingredient(name, grams),
        kcal=kcal,
        kcal_per_100g=(kcal / grams * 100) if kcal is not None else None,
        fdc_description=None if kcal is None else f"{name} (USDA)",
        fdc_id=None,
    )


class TestAggregate(unittest.TestCase):
    def test_base_is_sum_of_matched_kcal(self) -> None:
        est = aggregate([grounded("a", 100, 200), grounded("b", 100, 150)], [])
        self.assertEqual(est.base_kcal, 350)

    def test_unmatched_excluded_from_base_and_surfaced(self) -> None:
        est = aggregate([grounded("a", 100, 200), grounded("mystery", 50, None)], [])
        self.assertEqual(est.base_kcal, 200)
        self.assertEqual(est.unmatched, ("mystery",))

    def test_adjustments_widen_into_a_range(self) -> None:
        est = aggregate(
            [grounded("a", 100, 400)],
            [Adjustment("oil", 50, 150), Adjustment("sugar", 10, 30)],
        )
        self.assertEqual(est.low, 400 + 60)
        self.assertEqual(est.high, 400 + 180)
        self.assertEqual(est.point, (460 + 580) / 2)
        self.assertEqual(est.width, 120)

    def test_no_adjustments_is_a_degenerate_range_at_base(self) -> None:
        est = aggregate([grounded("a", 100, 500)], [])
        self.assertEqual(est.low, 500)
        self.assertEqual(est.high, 500)
        self.assertEqual(est.point, 500)
        self.assertEqual(est.width, 0)

    def test_empty_plate(self) -> None:
        est = aggregate([], [])
        self.assertEqual((est.low, est.high, est.point), (0, 0, 0))

    def test_estimate_serializes_to_json(self) -> None:
        est = aggregate([grounded("a", 100, 200)], [Adjustment("oil", 20, 40)])
        payload = to_json(est)
        self.assertIn('"base_kcal": 200', payload)
        self.assertIn('"reason": "oil"', payload)


class TestCombine(unittest.TestCase):
    def test_weighted_blend(self) -> None:
        # 0.65*100 + 0.35*200 = 135
        self.assertAlmostEqual(combine_estimates(100, 200, 0.65), 135.0)

    def test_falls_back_to_pipeline_when_oneshot_failed(self) -> None:
        # The case where the monolith is worthless; use the pipeline's number.
        self.assertEqual(combine_estimates(None, 530, 0.65), 530)

    def test_weight_is_clamped(self) -> None:
        self.assertEqual(combine_estimates(100, 200, 1.5), 100)  # all one-shot
        self.assertEqual(combine_estimates(100, 200, -1.0), 200)  # all pipeline

    def test_clamp_caps_pipeline_blowup(self) -> None:
        # Pipeline says 300 but one-shot is 100; clamp to +/-40% -> 140, then
        # 0.5 blend -> 120 (instead of 0.5*100+0.5*300 = 200).
        self.assertAlmostEqual(combine_estimates(100, 300, 0.5, clamp_band=0.4), 120.0)

    def test_clamp_leaves_reasonable_pipeline_alone(self) -> None:
        # 120 is within +/-40% of 100, so it passes through; 0.5 blend -> 110.
        self.assertAlmostEqual(combine_estimates(100, 120, 0.5, clamp_band=0.4), 110.0)


if __name__ == "__main__":
    unittest.main()
