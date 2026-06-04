"""Defensive parsing of model JSON — the seams where a 7B model gets sloppy."""

import unittest

from calorie_pipeline.pipeline import parse_oneshot
from calorie_pipeline.reason import parse_adjustments
from calorie_pipeline.vision import _parse_total, parse_ingredients


class TestParseIngredients(unittest.TestCase):
    def test_wrapped_object(self) -> None:
        out = parse_ingredients('{"ingredients": [{"name": "rice", "grams": 200, "prep": null}]}')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].name, "rice")
        self.assertEqual(out[0].grams, 200.0)
        self.assertIsNone(out[0].prep)

    def test_bare_list(self) -> None:
        out = parse_ingredients('[{"name": "egg", "grams": 50}]')
        self.assertEqual(out[0].name, "egg")

    def test_grams_with_unit_string(self) -> None:
        out = parse_ingredients('{"ingredients": [{"name": "toast", "grams": "60 g"}]}')
        self.assertEqual(out[0].grams, 60.0)

    def test_fused_response_yields_total_and_items(self) -> None:
        # One vision call returns both the one-shot total and the breakdown.
        raw = '{"total_kcal": 620, "ingredients": [{"name": "rice", "grams": 200}]}'
        self.assertEqual(_parse_total(raw), 620.0)
        self.assertEqual(parse_ingredients(raw)[0].name, "rice")

    def test_fused_total_absent_is_none(self) -> None:
        self.assertIsNone(_parse_total('{"ingredients": []}'))

    def test_entries_missing_name_or_grams_are_dropped(self) -> None:
        out = parse_ingredients(
            '{"ingredients": [{"grams": 100}, {"name": "ok", "grams": 10}, {"name": "noweight"}]}'
        )
        self.assertEqual([i.name for i in out], ["ok"])

    def test_exact_duplicates_are_deduped(self) -> None:
        # A 7B in JSON mode sometimes repeats a whole sub-list verbatim.
        out = parse_ingredients(
            '{"ingredients": ['
            '{"name": "strawberry", "grams": 100},'
            '{"name": "blueberry", "grams": 50},'
            '{"name": "strawberry", "grams": 100},'  # exact repeat -> dropped
            '{"name": "blueberry", "grams": 50}'  # exact repeat -> dropped
            "]}"
        )
        self.assertEqual([(i.name, i.grams) for i in out], [("strawberry", 100.0), ("blueberry", 50.0)])

    def test_same_food_different_portion_is_kept(self) -> None:
        # Genuinely different portions are NOT duplicates.
        out = parse_ingredients(
            '[{"name": "egg", "grams": 50}, {"name": "egg", "grams": 100}]'
        )
        self.assertEqual(len(out), 2)


class TestParseAdjustments(unittest.TestCase):
    def test_wrapped(self) -> None:
        out = parse_adjustments('{"adjustments": [{"reason": "oil", "low": 50, "high": 120}]}')
        self.assertEqual(out[0].reason, "oil")
        self.assertEqual((out[0].low, out[0].high), (50.0, 120.0))

    def test_inverted_range_is_normalized(self) -> None:
        out = parse_adjustments('[{"reason": "butter", "low": 120, "high": 40}]')
        self.assertEqual((out[0].low, out[0].high), (40.0, 120.0))

    def test_empty_list_when_nothing_hidden(self) -> None:
        self.assertEqual(parse_adjustments('{"adjustments": []}'), [])

    def test_malformed_entries_dropped(self) -> None:
        out = parse_adjustments('[{"reason": "x"}, {"reason": "y", "low": 1, "high": 2}]')
        self.assertEqual([a.reason for a in out], ["y"])


class TestParseOneShot(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(parse_oneshot('{"kcal": 640}').kcal, 640.0)

    def test_missing_key(self) -> None:
        self.assertIsNone(parse_oneshot('{"calories": 640}').kcal)

    def test_non_json_preserved_as_raw(self) -> None:
        est = parse_oneshot("about 600 calories")
        self.assertIsNone(est.kcal)
        self.assertEqual(est.raw, "about 600 calories")


if __name__ == "__main__":
    unittest.main()
