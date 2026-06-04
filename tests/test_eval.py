"""End-to-end offline eval — proves the comparison harness and the data flow.

This does not prove the *real* result (that needs models + Nutrition5k). It
proves that, given the fixtures' simulated stage outputs, the harness wires
aggregation into metrics correctly and reports the expected *shape*: the
decomposed pipeline beats the one-shot baseline on point error, and — unlike a
single number — actually covers the truth with its interval.
"""

import unittest

from evals.harness import evaluate_fixtures


class TestEvaluateFixtures(unittest.TestCase):
    def setUp(self) -> None:
        self.report = evaluate_fixtures()

    def test_all_dishes_scored(self) -> None:
        self.assertEqual(len(self.report.records), 8)
        self.assertEqual(self.report.pipeline.point.n, 8)

    def test_pipeline_beats_oneshot_on_point_error(self) -> None:
        self.assertLess(self.report.pipeline.point.mae, self.report.oneshot.point.mae)

    def test_oneshot_point_estimate_never_covers_truth(self) -> None:
        # The honesty axis: a single number makes no interval claim.
        assert self.report.oneshot.interval is not None
        self.assertEqual(self.report.oneshot.interval.coverage, 0.0)

    def test_pipeline_interval_mostly_covers_truth(self) -> None:
        assert self.report.pipeline.interval is not None
        # 7/8 by construction (one dish is deliberately under-ranged).
        self.assertGreaterEqual(self.report.pipeline.interval.coverage, 0.75)
        self.assertLess(self.report.pipeline.interval.coverage, 1.0)

    def test_pipeline_is_less_biased(self) -> None:
        self.assertLess(
            abs(self.report.pipeline.point.bias), abs(self.report.oneshot.point.bias)
        )


if __name__ == "__main__":
    unittest.main()
