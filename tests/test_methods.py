"""The method registry is the spine of the comparison; keep it well-formed."""

import unittest

from calorie_pipeline.methods import METHODS, Method, MethodResult


class TestMethodRegistry(unittest.TestCase):
    def test_every_entry_conforms_to_the_protocol(self) -> None:
        for m in METHODS:
            self.assertIsInstance(m, Method)
            self.assertTrue(m.name and m.lever, f"{m} missing name/lever")
            self.assertTrue(callable(m.estimate))

    def test_names_are_unique(self) -> None:
        names = [m.name for m in METHODS]
        self.assertEqual(len(names), len(set(names)))

    def test_the_baseline_is_present(self) -> None:
        # Every comparison is anchored on the one-shot baseline.
        self.assertIn("one-shot", [m.name for m in METHODS])

    def test_method_result_defaults(self) -> None:
        r = MethodResult(kcal=300.0)
        self.assertEqual((r.tokens, r.calls, r.detail), (0, 0, ""))


if __name__ == "__main__":
    unittest.main()
