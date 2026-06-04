"""Stage 2 — Calorie lookup: ground each ingredient in USDA FoodData Central.

This is the most important design decision in the project, and it is *the
absence of a model*. The single largest error source in photo calorie
estimation is not "what food is this" — it is "how many calories are in 100 g of
that food." That is a lookup against a measured database, not a judgment call.
So we look it up. The model never sees this number; it cannot get it wrong.

FDC's energy nutrient (number 208, kcal) has lived under two different JSON
shapes as the API has migrated, and search results mix them. We read both. See
:func:`extract_energy_kcal`.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

from calorie_pipeline.config import ENERGY_NUTRIENT_NUMBER, KCAL_UNIT, Config
from calorie_pipeline.models import GroundedIngredient, Ingredient

# Tokens too generic to carry relevance ("fresh tomato" shouldn't match on
# "fresh"). Kept tiny on purpose; the real signal is the food noun.
_STOPWORDS = frozenset(
    {"fresh", "raw", "cooked", "sliced", "slice", "whole", "plain", "with", "and"}
)

# Words that mark a CONCENTRATED form of a food — oil, flour, dried, juice — which
# carries several times the calorie density of the whole food. If the query did
# not ask for the concentrate but the candidate is one, it is almost always a bad
# match ("salmon fillet" -> "Fish oil, salmon" at 902 kcal/100 g). We deprioritize
# such candidates below any whole-food match. This was the single largest source
# of catastrophic over-estimates on the Nutrition5k benchmark.
_CONCENTRATE_MARKERS = frozenset(
    {"oil", "flour", "dried", "dehydrated", "powder", "concentrate", "paste", "juice"}
)

# Prepared-DISH forms that are far denser than the plain food a photo usually
# shows. FDC's searchable rows skew heavily toward these (search "potato" and you
# get chips, fries, "Soup, potato", "Pie, sweet potato"), which made every matcher
# — keyword AND semantic — ground foods ~22% too calorie-dense on the benchmark.
# Unless the query explicitly asks for the prepared form, we deprioritize it.
_DENSE_FORM_MARKERS = frozenset(
    {
        "pie", "cake", "soup", "stew", "sandwich", "pizza", "fried", "breaded",
        "battered", "snack", "snacks", "chips", "candy", "candies", "cookie",
        "cookies", "pastry", "muffin", "nuggets", "gravy", "dressing", "restaurant",
    }
)

# Either kind of non-plain form is penalized when the query didn't ask for it.
_FORM_MARKERS = _CONCENTRATE_MARKERS | _DENSE_FORM_MARKERS

# Shape/cut words that describe how a food was *portioned*, not what it *is*.
# Left in the FDC query they hijack the candidate pool ("potato chunk" ranks
# "Soup, beef and mushroom, chunk style" first, so plain potato never even enters
# the pool for the re-ranker to find). We strip them from the search query.
_DESCRIPTOR_WORDS = frozenset(
    {
        "chunk", "chunks", "slice", "sliced", "slices", "piece", "pieces",
        "fillet", "fillets", "strip", "strips", "diced", "cubed", "cube", "cubes",
        "wedge", "wedges", "half", "halves", "leaf", "leaves", "chopped", "minced",
    }
)


def lookup_ingredient(
    ingredient: Ingredient,
    config: Config,
    *,
    session: Any | None = None,
) -> GroundedIngredient:
    """Look up one ingredient and scale its energy to the portion.

    It pools candidates across *all* the configured data types and picks the
    single most query-relevant one that carries an energy value (see
    :func:`_best_match`), rather than blindly trusting FDC's top hit or the first
    data type that returns anything. This matters because Foundation is curated
    and lab-measured but thin: stopping at its top hit grounds ~1/7 real
    ingredients and returns dangerous near-matches ("whole wheat bread" -> Flour,
    "bagel" -> cheeseburger). Pooling + relevance keeps Foundation's quality when
    it is genuinely the best match, while letting a 2-token "Bread, whole-wheat"
    in SR Legacy beat a 1-token "Flour, whole wheat" in Foundation.

    On a total miss — nothing relevant anywhere, no energy nutrient, network/HTTP
    error — returns an *unmatched* GroundedIngredient with ``kcal=None`` rather
    than guessing, preserving the best provenance we saw for auditability. Misses
    are data, not failures.
    """
    fallback_description: str | None = None
    fallback_fdc_id: int | None = None
    pool: list[tuple[Mapping[str, Any], float, int]] = []
    # Query cleaning ("potato chunk" -> "potato") is part of the semantic
    # experiment; the default keyword path searches the name as-is.
    query = _clean_query(ingredient.name) if config.semantic_match else ingredient.name

    for priority, data_type in enumerate(config.fdc_data_types):
        try:
            candidates = _search_candidates(query, config, data_type, session=session)
        except Exception:  # noqa: BLE001 — a lookup miss must never crash the pipeline
            continue
        for c in candidates:
            energy = extract_energy_kcal(c)
            if energy is None:
                if fallback_description is None:
                    fallback_description = c.get("description")
                    fallback_fdc_id = c.get("fdcId")
                continue
            pool.append((c, energy, priority))

    food, kcal_per_100g = _select_match(query, pool, config)
    if food is None:
        return GroundedIngredient(ingredient, None, None, fallback_description, fallback_fdc_id)

    kcal = kcal_per_100g * ingredient.grams / 100.0
    return GroundedIngredient(
        ingredient, kcal, kcal_per_100g, food.get("description"), food.get("fdcId")
    )


def _select_match(
    query: str, pool: list[tuple[Mapping[str, Any], float, int]], config: Config
) -> tuple[Mapping[str, Any] | None, float]:
    """Choose the best candidate, semantically if possible, else by keyword.

    Semantic re-ranking (embed the query and each candidate description, rank by
    cosine) is the accurate path — it grounds "salmon fillet" to "Fish, salmon,
    raw" instead of "Vegetarian fillets". If embedding is disabled or the embed
    service is unreachable, we fall back to keyword token overlap, which is worse
    but needs no model and keeps the pipeline running offline-testably.
    """
    if config.llm_match:
        try:
            return _llm_best_match(query, pool, config)
        except Exception:  # noqa: BLE001 — degrade to keyword matching, never crash
            pass
    if config.semantic_match:
        try:
            return _semantic_best_match(query, pool, config)
        except Exception:  # noqa: BLE001
            pass
    return _best_match(query, pool)


def _llm_best_match(
    query: str, pool: list[tuple[Mapping[str, Any], float, int]], config: Config
) -> tuple[Mapping[str, Any] | None, float]:
    """Let the text model choose the best USDA row from the real candidates.

    Matching the right database entry is a *judgment* ("is 'salmon fillet' the raw
    fish, the breaded nuggets, or the oil?"), and judgment is what the model is
    for. Crucially it only *selects* among rows that exist — it cannot invent a
    calorie number — so the estimate stays grounded. On the benchmark this beat
    both keyword and embedding matching.
    """
    if not pool:
        return None, 0.0
    # Dedup by description, cap the option list to keep the prompt tight.
    deduped: list[tuple[Mapping[str, Any], float]] = []
    seen: set[str] = set()
    for food, kcal, _ in pool:
        desc = str(food.get("description", ""))
        if desc and desc not in seen:
            seen.add(desc)
            deduped.append((food, kcal))
    deduped = deduped[:12]

    options = "\n".join(f"{i}: {food.get('description', '')}" for i, (food, _) in enumerate(deduped))
    prompt = (
        f'A food identified in a meal photo is: "{query}".\n'
        f"Which USDA entry below is the SAME food in its plain, as-served form? "
        f"Avoid concentrates (oil, flour, juice) and unrelated prepared dishes "
        f"(pie, soup, chips) unless the food name itself says so.\n\n{options}\n\n"
        f'Respond with ONLY JSON: {{"index": <number>}}'
    )

    from ollama import Client  # lazy import

    response = Client(host=config.ollama_host).chat(
        model=config.text_model,
        format="json",
        options={"temperature": 0.0},
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        idx = int(json.loads(response["message"]["content"]).get("index", 0))
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        idx = 0
    if not 0 <= idx < len(deduped):
        idx = 0
    return deduped[idx]


# Minimum query/candidate cosine to accept a semantic match; below this the pool
# holds nothing genuinely about the query, so we reject (-> unmatched) rather than
# ground to noise. nomic-embed-text puts relevant food pairs around 0.6-0.75.
_MIN_SEMANTIC_SIM = 0.45

_EMBED_CACHE: dict[str, list[float]] = {}


def _semantic_best_match(
    query: str, pool: list[tuple[Mapping[str, Any], float, int]], config: Config
) -> tuple[Mapping[str, Any] | None, float]:
    """Rank candidates by embedding cosine similarity to the query.

    Still prefers a whole-food form over an unrequested concentrate (the oil/flour
    guard) as the primary key, then highest cosine. Rejects the whole pool if even
    the best match is too semantically distant.
    """
    if not pool:
        return None, 0.0
    descriptions = [str(food.get("description", "")) for food, _, _ in pool]
    vectors = _embed(
        [f"search_query: {query}"] + [f"search_document: {d}" for d in descriptions],
        config,
    )
    q_vec, d_vecs = vectors[0], vectors[1:]
    q_tokens = set(_tokens(query))

    scored: list[tuple[int, float, int, Mapping[str, Any], float]] = []
    for (food, kcal, priority), desc, d_vec in zip(pool, descriptions, d_vecs):
        scored.append((_basic_form(q_tokens, desc), _cosine(q_vec, d_vec), -priority, food, kcal))
    scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)

    best = scored[0]
    if best[1] < _MIN_SEMANTIC_SIM:
        return None, 0.0
    return best[3], best[4]


def _embed(texts: list[str], config: Config) -> list[list[float]]:
    """Embed texts via Ollama, memoized within the process to avoid re-embedding
    the same query/description across dishes."""
    missing = [t for t in texts if t not in _EMBED_CACHE]
    if missing:
        from ollama import Client  # lazy import

        result = Client(host=config.ollama_host).embed(model=config.embed_model, input=missing)
        for text, vector in zip(missing, result["embeddings"]):
            _EMBED_CACHE[text] = vector
    return [_EMBED_CACHE[t] for t in texts]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _best_match(
    query: str, pool: list[tuple[Mapping[str, Any], float, int]]
) -> tuple[Mapping[str, Any] | None, float]:
    """Pick the most query-relevant energy-bearing candidate, or reject all.

    Relevance = how many meaningful query tokens appear in the candidate's
    description. A candidate that shares *no* query token is rejected outright —
    that is the guard that stops "bagel" from resolving to a cheeseburger. Among
    relevant candidates we prefer, in order: higher overlap, then the
    higher-priority data type (Foundation's lab values over broader sets), then
    the shorter (more generic) description.
    """
    q_tokens = set(_tokens(query))
    scored: list[tuple[int, int, int, int, Mapping[str, Any], float]] = []
    for food, kcal, priority in pool:
        desc = str(food.get("description", ""))
        desc_lower = desc.lower()
        overlap = sum(1 for t in q_tokens if t in desc_lower)
        if overlap == 0:
            continue
        # Prefer a whole food over an unrequested *concentrate* (oil/flour) — the
        # guard that stops "salmon" -> "Fish oil". (The broader prepared-dish
        # penalty lives in the opt-in semantic path; it didn't help in aggregate.)
        whole_food = 0 if (_CONCENTRATE_MARKERS & set(_tokens(desc))) - q_tokens else 1
        # Sort key, all "bigger is better": whole food, overlap, curated data
        # type (-priority), brevity (-len).
        scored.append((whole_food, overlap, -priority, -len(desc), food, kcal))
    if not scored:
        return None, 0.0
    scored.sort(key=lambda s: (s[0], s[1], s[2], s[3]), reverse=True)
    best = scored[0]
    return best[4], best[5]


def _tokens(text: str) -> list[str]:
    """Meaningful lowercase word tokens from a query (>2 chars, not a stopword)."""
    raw = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in raw if len(t) > 2 and t not in _STOPWORDS]


def _basic_form(query_tokens: set[str], description: str) -> int:
    """1 if the candidate is a plain food form, 0 if an unrequested concentrate or
    prepared-dish form (oil, flour, pie, soup, fried...). Used as the top sort key
    so a plain food always outranks a denser prepared near-match."""
    unrequested = (_FORM_MARKERS & set(_tokens(description))) - query_tokens
    return 0 if unrequested else 1


def _clean_query(name: str) -> str:
    """Drop shape/cut descriptors so the FDC search returns the food, not the cut.

    "potato chunk" -> "potato"; "sliced deli turkey breast" -> "deli turkey
    breast". Falls back to the original name if stripping would empty it.
    """
    kept = [w for w in name.split() if w.lower().strip(",") not in _DESCRIPTOR_WORDS]
    return " ".join(kept) if kept else name


def _search_candidates(
    query: str, config: Config, data_type: str, *, session: Any | None
) -> list[dict[str, Any]]:
    import requests  # lazy: keeps the package importable without requests

    params = {
        "api_key": config.fdc_api_key,
        "query": query,
        "dataType": data_type,
        "pageSize": config.fdc_page_size,
    }
    getter = session.get if session is not None else requests.get
    response = getter(
        f"{config.fdc_base_url}/foods/search",
        params=params,
        timeout=config.request_timeout,
    )
    response.raise_for_status()
    return response.json().get("foods") or []


def extract_energy_kcal(food: Mapping[str, Any]) -> float | None:
    """Extract energy in kcal/100 g from an FDC food, schema-agnostically.

    FDC nutrient entries appear in two shapes, sometimes within one response:

    * **flat** (legacy search):
      ``{"nutrientNumber": "208", "unitName": "KCAL", "value": 52}``
    * **nested** (newer):
      ``{"nutrient": {"number": "208", "unitName": "KCAL"}, "amount": 52}``

    We match nutrient number 208 *and* require the kcal unit, so we never pick up
    the kilojoule energy entry (number 268) by mistake. Returns ``None`` if no
    kcal energy nutrient is present. Pure function — the core of the unit tests.
    """
    for entry in food.get("foodNutrients", []) or []:
        if not isinstance(entry, dict):
            continue
        if _nutrient_number(entry) != ENERGY_NUTRIENT_NUMBER:
            continue
        if _nutrient_unit(entry) != KCAL_UNIT:
            continue
        value = _nutrient_value(entry)
        if value is not None:
            return value
    return None


def _nutrient_number(entry: Mapping[str, Any]) -> str | None:
    if "nutrientNumber" in entry and entry["nutrientNumber"] is not None:
        return str(entry["nutrientNumber"])
    nutrient = entry.get("nutrient")
    if isinstance(nutrient, Mapping) and nutrient.get("number") is not None:
        return str(nutrient["number"])
    return None


def _nutrient_unit(entry: Mapping[str, Any]) -> str | None:
    if entry.get("unitName"):
        return str(entry["unitName"]).upper()
    nutrient = entry.get("nutrient")
    if isinstance(nutrient, Mapping) and nutrient.get("unitName"):
        return str(nutrient["unitName"]).upper()
    return None


def _nutrient_value(entry: Mapping[str, Any]) -> float | None:
    for key in ("value", "amount"):
        if entry.get(key) is not None:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                return None
    return None
