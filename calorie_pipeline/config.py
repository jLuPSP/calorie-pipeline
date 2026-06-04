"""Centralized, env-overridable configuration.

Every knob the pipeline touches lives here. Nothing else in the codebase reads
``os.environ`` directly. The design goal: a bigger box should be able to swap in
larger models (``VISION_MODEL=llama3.2-vision:90b``) or point at a remote Ollama
(``OLLAMA_HOST=http://tower.local:11434``) with **zero code changes** — only
environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

# --- Defaults -----------------------------------------------------------------
# 7B is the *point*, not a limitation. These are the models the thesis is about.
DEFAULT_VISION_MODEL = "qwen2.5vl:7b"
DEFAULT_TEXT_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

DEFAULT_FDC_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
# DEMO_KEY works with no signup (rate-limited). Override with a free key from
# https://fdc.nal.usda.gov/api-key-signup for real throughput.
DEFAULT_FDC_API_KEY = "DEMO_KEY"
# Ordered fallback across FDC data types. Foundation is the highest quality
# (curated, lab-measured) but has thin coverage of prepared foods, so on a miss
# we fall back to the broad SR Legacy set, then the generic FNDDS survey foods.
# Empirically, Foundation alone matches ~1/7 real meal ingredients (and returns
# dangerous near-matches like basil -> corn); the chain matches ~7/7. Override
# with a comma-separated FDC_DATA_TYPE (e.g. "Foundation" to force purity).
DEFAULT_FDC_DATA_TYPES = ("Foundation", "SR Legacy", "Survey (FNDDS)")
# We fetch many candidates rather than blindly trusting FDC's top hit, because
# the top hit is a trap: "bagel" can rank "Fast foods, cheeseburger; single..."
# first (the token "single"), "sandwich" can rank "Ice cream sandwich". The
# lookup re-ranks the pool to pick the best match. A larger pool gives the
# semantic re-ranker more to work with. See lookup._best_match / _semantic_match.
DEFAULT_FDC_PAGE_SIZE = 25

# Embedding model (Ollama) used to semantically re-rank USDA candidates. Keyword
# matching grounds foods to entries that merely share a word ("salmon fillet" ->
# "Vegetarian fillets"); on the benchmark this made the pipeline's matches ~28%
# too calorie-dense on average. Semantic re-ranking picks the food whose *meaning*
# is closest ("Fish, salmon, raw"). Falls back to keyword matching if unavailable.
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# FDC nutrient number for food energy in kilocalories. This is a stable
# identifier across the FDC schema migrations; the *shape* around it moves, not
# the number. See lookup.py for the dual-schema extraction.
ENERGY_NUTRIENT_NUMBER = "208"
KCAL_UNIT = "KCAL"


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable run configuration. Construct via :meth:`from_env`."""

    # Models
    vision_model: str = DEFAULT_VISION_MODEL
    text_model: str = DEFAULT_TEXT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST

    # Semantic re-ranking of USDA candidates (embeddings + query cleaning +
    # prepared-form penalty), vs the default keyword token-overlap matcher.
    # OFF by default: on the Nutrition5k benchmark it produced qualitatively nicer
    # individual matches but WORSE aggregate error than plain keyword matching
    # (MAE 188 vs 141) — kept as a documented, opt-in experiment. See benchmark/.
    semantic_match: bool = False

    # Let the text model PICK the USDA match from real candidates (a judgment) —
    # it can't invent a calorie number, only choose a row. Beats both keyword and
    # semantic matching on the benchmark (MAE 141/188 -> 125). Off by default
    # because it costs one model call per ingredient; turn on with LLM_MATCH=1.
    llm_match: bool = False

    # Final-answer blend of the one-shot baseline and the decomposed estimate.
    # The two make weakly-correlated errors, so combining beats either alone. We
    # first CLAMP the decomposed point to within +/- combine_clamp of the one-shot
    # (using the stable cheap estimate as a sanity bound that caps the pipeline's
    # occasional compounding blow-ups), then take a weighted average. On the
    # benchmark this takes MAE from 90 (one-shot) to ~85. combine_weight is the
    # weight on the one-shot.
    combine_weight: float = 0.5
    combine_clamp: float = 0.4

    # USDA FoodData Central
    fdc_api_key: str = DEFAULT_FDC_API_KEY
    fdc_base_url: str = DEFAULT_FDC_BASE_URL
    fdc_data_types: tuple[str, ...] = DEFAULT_FDC_DATA_TYPES
    fdc_page_size: int = DEFAULT_FDC_PAGE_SIZE

    # Sampling temperatures. 0.0-0.2 for anything that must be reproducible;
    # the pipeline never needs creativity, it needs consistency.
    vision_temperature: float = 0.1
    reason_temperature: float = 0.2
    oneshot_temperature: float = 0.1

    # Networking
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        """Build a Config from environment variables, falling back to defaults.

        Pass an explicit ``env`` mapping in tests to avoid touching the process
        environment.
        """
        e = os.environ if env is None else env
        return cls(
            vision_model=e.get("VISION_MODEL", DEFAULT_VISION_MODEL),
            text_model=e.get("TEXT_MODEL", DEFAULT_TEXT_MODEL),
            embed_model=e.get("EMBED_MODEL", DEFAULT_EMBED_MODEL),
            ollama_host=e.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            semantic_match=e.get("SEMANTIC_MATCH", "0").lower() in ("1", "true", "yes"),
            llm_match=e.get("LLM_MATCH", "0").lower() in ("1", "true", "yes"),
            combine_weight=float(e.get("COMBINE_WEIGHT", "0.5")),
            combine_clamp=float(e.get("COMBINE_CLAMP", "0.4")),
            fdc_api_key=e.get("FDC_API_KEY", DEFAULT_FDC_API_KEY),
            fdc_base_url=e.get("FDC_BASE_URL", DEFAULT_FDC_BASE_URL),
            fdc_data_types=_parse_data_types(e.get("FDC_DATA_TYPE")),
            fdc_page_size=int(e.get("FDC_PAGE_SIZE", str(DEFAULT_FDC_PAGE_SIZE))),
            vision_temperature=float(e.get("VISION_TEMPERATURE", "0.1")),
            reason_temperature=float(e.get("REASON_TEMPERATURE", "0.2")),
            oneshot_temperature=float(e.get("ONESHOT_TEMPERATURE", "0.1")),
            request_timeout=float(e.get("REQUEST_TIMEOUT", "30")),
        )


def _parse_data_types(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated FDC_DATA_TYPE into an ordered fallback tuple.

    Empty/unset -> the default chain. Whitespace around each type is trimmed so
    ``"Foundation, SR Legacy"`` works as expected.
    """
    if not raw or not raw.strip():
        return DEFAULT_FDC_DATA_TYPES
    parsed = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parsed or DEFAULT_FDC_DATA_TYPES
