"""USDA lookup: dual nutrient schema, unit discipline, graceful misses."""

import unittest

from calorie_pipeline.config import Config
from calorie_pipeline.lookup import extract_energy_kcal, lookup_ingredient
from calorie_pipeline.models import Ingredient


def flat_food(kcal: float | None = 52.0, *, with_kj: bool = False) -> dict:
    nutrients = []
    if kcal is not None:
        nutrients.append({"nutrientNumber": "208", "unitName": "KCAL", "value": kcal})
    if with_kj:
        nutrients.append({"nutrientNumber": "268", "unitName": "KJ", "value": 999})
    return {"description": "Apple, raw", "fdcId": 1, "foodNutrients": nutrients}


def nested_food(kcal: float = 89.0) -> dict:
    return {
        "description": "Banana, raw",
        "fdcId": 2,
        "foodNutrients": [
            {"nutrient": {"number": "208", "unitName": "KCAL"}, "amount": kcal},
            {"nutrient": {"number": "203", "unitName": "G"}, "amount": 1.1},
        ],
    }


class TestExtractEnergy(unittest.TestCase):
    def test_flat_schema(self) -> None:
        self.assertEqual(extract_energy_kcal(flat_food(52)), 52.0)

    def test_nested_schema(self) -> None:
        self.assertEqual(extract_energy_kcal(nested_food(89)), 89.0)

    def test_kilojoule_energy_is_not_mistaken_for_kcal(self) -> None:
        # Only a kJ energy entry present -> no kcal available.
        food = {"foodNutrients": [{"nutrientNumber": "268", "unitName": "KJ", "value": 880}]}
        self.assertIsNone(extract_energy_kcal(food))

    def test_kcal_chosen_even_when_kj_also_present(self) -> None:
        self.assertEqual(extract_energy_kcal(flat_food(52, with_kj=True)), 52.0)

    def test_no_nutrients(self) -> None:
        self.assertIsNone(extract_energy_kcal({"foodNutrients": []}))
        self.assertIsNone(extract_energy_kcal({}))


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict, *, boom: bool = False) -> None:
        self._payload = payload
        self._boom = boom
        self.calls: list[dict] = []

    def get(self, url: str, params: dict, timeout: float) -> _FakeResponse:
        self.calls.append(params)
        if self._boom:
            raise ConnectionError("network down")
        return _FakeResponse(self._payload)


class TestLookupIngredient(unittest.TestCase):
    def setUp(self) -> None:
        # Keyword path: deterministic and offline (no embed service needed).
        self.config = Config(semantic_match=False)

    def test_scales_energy_to_portion(self) -> None:
        session = _FakeSession({"foods": [flat_food(52)]})  # 52 kcal / 100 g
        g = lookup_ingredient(Ingredient("apple", 150), self.config, session=session)
        self.assertAlmostEqual(g.kcal, 78.0)  # 52 * 150 / 100
        self.assertEqual(g.kcal_per_100g, 52.0)
        self.assertEqual(g.fdc_description, "Apple, raw")
        self.assertTrue(g.matched)

    def test_no_results_is_an_unmatched_ingredient(self) -> None:
        session = _FakeSession({"foods": []})
        g = lookup_ingredient(Ingredient("unobtanium stew", 100), self.config, session=session)
        self.assertIsNone(g.kcal)
        self.assertFalse(g.matched)

    def test_match_without_energy_keeps_provenance_but_no_kcal(self) -> None:
        session = _FakeSession({"foods": [flat_food(None)]})
        g = lookup_ingredient(Ingredient("apple", 100), self.config, session=session)
        self.assertIsNone(g.kcal)
        self.assertEqual(g.fdc_description, "Apple, raw")  # we matched, just no energy

    def test_rejects_irrelevant_top_match(self) -> None:
        # "bagel" must NOT resolve to a cheeseburger that ranked first.
        cheeseburger = {
            "description": "Fast foods, cheeseburger; single, large patty; plain",
            "fdcId": 99,
            "foodNutrients": [{"nutrientNumber": "208", "unitName": "KCAL", "value": 300}],
        }
        session = _FakeSession({"foods": [cheeseburger]})
        g = lookup_ingredient(Ingredient("bagel", 100), self.config, session=session)
        self.assertFalse(g.matched)  # rejected: shares no token with "bagel"

    def test_picks_most_relevant_energetic_candidate(self) -> None:
        # Irrelevant high-energy hit ranked first; the real bagel is below it.
        cheeseburger = {
            "description": "Fast foods, cheeseburger; single, large patty",
            "foodNutrients": [{"nutrientNumber": "208", "unitName": "KCAL", "value": 300}],
        }
        bagel = {
            "description": "Bagels, plain, enriched",
            "fdcId": 7,
            "foodNutrients": [{"nutrientNumber": "208", "unitName": "KCAL", "value": 250}],
        }
        session = _FakeSession({"foods": [cheeseburger, bagel]})
        g = lookup_ingredient(Ingredient("bagel", 100), self.config, session=session)
        self.assertTrue(g.matched)
        self.assertEqual(g.kcal_per_100g, 250.0)
        self.assertEqual(g.fdc_description, "Bagels, plain, enriched")

    def test_prefers_whole_food_over_unrequested_concentrate(self) -> None:
        # "salmon" must not resolve to fish OIL (902 kcal/100 g) when a whole-fish
        # match exists. The concentrate form is deprioritized below it.
        fish_oil = {
            "description": "Fish oil, salmon",
            "foodNutrients": [{"nutrientNumber": "208", "unitName": "KCAL", "value": 902}],
        }
        salmon = {
            "description": "Fish, salmon, Atlantic, cooked",
            "fdcId": 5,
            "foodNutrients": [{"nutrientNumber": "208", "unitName": "KCAL", "value": 206}],
        }
        session = _FakeSession({"foods": [fish_oil, salmon]})
        g = lookup_ingredient(Ingredient("salmon fillet", 100), self.config, session=session)
        self.assertEqual(g.kcal_per_100g, 206.0)
        self.assertEqual(g.fdc_description, "Fish, salmon, Atlantic, cooked")

    def test_concentrate_used_when_explicitly_requested(self) -> None:
        # If the query asks for oil, the oil form is fine (not penalized).
        oil = {
            "description": "Oil, olive, extra virgin",
            "fdcId": 6,
            "foodNutrients": [{"nutrientNumber": "208", "unitName": "KCAL", "value": 884}],
        }
        session = _FakeSession({"foods": [oil]})
        g = lookup_ingredient(Ingredient("olive oil", 10), self.config, session=session)
        self.assertTrue(g.matched)
        self.assertEqual(g.kcal_per_100g, 884.0)

    def test_network_error_degrades_to_unmatched(self) -> None:
        session = _FakeSession({}, boom=True)
        g = lookup_ingredient(Ingredient("apple", 100), self.config, session=session)
        self.assertIsNone(g.kcal)
        self.assertFalse(g.matched)

    def test_search_params_carry_config(self) -> None:
        session = _FakeSession({"foods": [nested_food()]})
        lookup_ingredient(Ingredient("banana", 100), self.config, session=session)
        params = session.calls[0]
        # First search uses the first data type in the fallback chain; an
        # energy-bearing top match short-circuits the rest.
        self.assertEqual(params["dataType"], self.config.fdc_data_types[0])
        self.assertEqual(params["pageSize"], self.config.fdc_page_size)
        self.assertEqual(params["query"], "banana")

    def test_falls_back_across_data_types_until_energy_found(self) -> None:
        # Foundation-style match with no energy, then a set that has energy.
        class _ChainSession:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def get(self, url: str, params: dict, timeout: float):
                self.calls.append(params)
                if params["dataType"] == "Foundation":
                    return _FakeResponse({"foods": [flat_food(None)]})  # no energy
                return _FakeResponse({"foods": [flat_food(52)]})  # SR Legacy: energy

        session = _ChainSession()
        g = lookup_ingredient(Ingredient("apple", 200), self.config, session=session)
        self.assertAlmostEqual(g.kcal, 104.0)  # 52 * 200 / 100, from the fallback
        self.assertEqual([c["dataType"] for c in session.calls][:2], ["Foundation", "SR Legacy"])


import calorie_pipeline.lookup as lookup_mod


class TestSemanticMatch(unittest.TestCase):
    """Semantic re-ranker logic, with embeddings injected (no live model)."""

    def _patch_embed(self, mapping: dict) -> None:
        self.addCleanup(setattr, lookup_mod, "_embed", lookup_mod._embed)
        lookup_mod._embed = lambda texts, config: [mapping[t] for t in texts]

    def test_ranks_by_cosine_similarity(self) -> None:
        self._patch_embed(
            {
                "search_query: salmon": [1.0, 0.0],
                "search_document: Cake, chocolate": [0.0, 1.0],
                "search_document: Fish, salmon, raw": [1.0, 0.0],
            }
        )
        pool = [
            ({"description": "Cake, chocolate"}, 370.0, 0),
            ({"description": "Fish, salmon, raw"}, 180.0, 1),
        ]
        food, kcal = lookup_mod._semantic_best_match("salmon", pool, Config())
        self.assertEqual(food["description"], "Fish, salmon, raw")
        self.assertEqual(kcal, 180.0)

    def test_whole_food_beats_concentrate_at_similar_cosine(self) -> None:
        self._patch_embed(
            {
                "search_query: salmon": [1.0, 0.0],
                "search_document: Fish oil, salmon": [1.0, 0.0],  # concentrate
                "search_document: Fish, salmon, raw": [0.95, 0.05],
            }
        )
        pool = [
            ({"description": "Fish oil, salmon"}, 902.0, 0),
            ({"description": "Fish, salmon, raw"}, 180.0, 0),
        ]
        food, _ = lookup_mod._semantic_best_match("salmon", pool, Config())
        self.assertEqual(food["description"], "Fish, salmon, raw")

    def test_rejects_semantically_distant_pool(self) -> None:
        self._patch_embed(
            {"search_query: rocket fuel": [1.0, 0.0], "search_document: Cake": [0.0, 1.0]}
        )
        pool = [({"description": "Cake"}, 370.0, 0)]
        food, _ = lookup_mod._semantic_best_match("rocket fuel", pool, Config())
        self.assertIsNone(food)


if __name__ == "__main__":
    unittest.main()
