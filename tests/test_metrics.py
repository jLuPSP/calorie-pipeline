"""Eval metrics — the numbers the thesis is argued with, proven on known inputs."""

import unittest

from evals.metrics import interval_metrics, point_metrics


class TestPointMetrics(unittest.TestCase):
    def test_known_values(self) -> None:
        # predicted - actual = [+10, -30] -> abs [10, 30]
        m = point_metrics([110, 70], [100, 100])
        self.assertEqual(m.n, 2)
        self.assertEqual(m.mae, 20.0)
        self.assertEqual(m.bias, -10.0)  # (10 + -30) / 2
        self.assertEqual(m.median_abs_error, 20.0)
        self.assertAlmostEqual(m.mape, (10.0 + 30.0) / 2)  # both actuals are 100

    def test_rmse_penalizes_large_errors(self) -> None:
        m = point_metrics([100, 200], [100, 100])  # errors 0, 100
        self.assertAlmostEqual(m.rmse, (0 + 100**2) ** 0.5 / (2**0.5))

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            point_metrics([], [])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            point_metrics([1, 2], [1])


class TestIntervalMetrics(unittest.TestCase):
    def test_coverage_and_width(self) -> None:
        lows = [90, 200, 50]
        highs = [110, 300, 60]
        actual = [100, 100, 55]  # in, out, in -> 2/3 covered
        m = interval_metrics(lows, highs, actual)
        self.assertAlmostEqual(m.coverage, 2 / 3)
        self.assertEqual(m.mean_width, (20 + 100 + 10) / 3)
        self.assertEqual(m.median_width, 20)

    def test_point_estimate_has_zero_coverage(self) -> None:
        # A degenerate interval (low == high) covers truth only on an exact hit.
        m = interval_metrics([100, 100], [100, 100], [100, 101])
        self.assertEqual(m.coverage, 0.5)
        self.assertEqual(m.mean_width, 0.0)


if __name__ == "__main__":
    unittest.main()
