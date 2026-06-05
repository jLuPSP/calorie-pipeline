"""The methods under comparison: every reasonable way to ship a calorie estimator
on a small local model, so the benchmark can rank them by accuracy AND by cost.

Each method is a swappable unit with one job: photo in, calorie number out, plus
the model tokens it spent. They are deliberately uniform and self-contained (each
sets its own config), so the comparison harness gives none of them special
treatment. The lineup spans the real levers for improving on a one-shot prompt:

    one-shot                 baseline                      one forward pass
    one-shot + CoT           more reasoning per call       think then answer
    one-shot + rubric        prompt engineering            calorie hints in-context
    few-shot                 in-context examples           2 solved photos first
    self-consistency         variance reduction            sample 5, take median
    self-refine              verify and revise             estimate, then critique it
    decomposed + USDA        grounding                     look the facts up
    decomposed (no reason)   ablation of grounding         drop the hidden-calorie pass
    grounded + blended       ensembling                    correct the prompt with facts

Add a method: one class, one line in ``METHODS``. That is the whole point.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from calorie_pipeline import reason as reason_mod
from calorie_pipeline import vision as vision_mod
from calorie_pipeline.config import Config
from calorie_pipeline.lookup import lookup_ingredient
from calorie_pipeline.models import GroundedIngredient, Ingredient, combine_estimates
from calorie_pipeline.pipeline import _ONESHOT_PROMPT, aggregate, parse_oneshot

_ROOT = Path(__file__).resolve().parent.parent
# Few-shot exemplars: real photos with known calories, NOT in the Nutrition5k
# benchmark, so there is no leakage.
_FEWSHOT = [(str(_ROOT / "evals/sample_images/pizza.jpg"), 310),
            (str(_ROOT / "evals/sample_images/bagels.jpg"), 500)]


@dataclass(frozen=True, slots=True)
class MethodResult:
    """One method's answer for one photo, plus what it cost to get there."""

    kcal: float | None
    tokens: int = 0
    calls: int = 0
    detail: str = ""


@runtime_checkable
class Method(Protocol):
    name: str
    lever: str

    def estimate(self, image_path: str, config: Config) -> MethodResult: ...


# --- shared plumbing ----------------------------------------------------------

def _client(config: Config) -> Any:
    from ollama import Client  # lazy: keeps the package importable without ollama

    return Client(host=config.ollama_host)


def _toks(resp: Any) -> int:
    return int(resp.get("prompt_eval_count", 0) or 0) + int(resp.get("eval_count", 0) or 0)


def _vision(client: Any, prompt: str, image: str, config: Config, temp: float | None = None) -> tuple[str, int]:
    resp = client.chat(
        model=config.vision_model, format="json",
        options={"temperature": config.vision_temperature if temp is None else temp},
        messages=[{"role": "user", "content": prompt, "images": [image]}],
    )
    return resp["message"]["content"], _toks(resp)


def _keyword(config: Config) -> Config:
    """Grounding with plain keyword matching (no hidden model calls, clean tokens)."""
    return dataclasses.replace(config, llm_match=False, semantic_match=False)


def _ground_and_reason(client: Any, ingredients: list[Ingredient], config: Config) -> tuple[list[GroundedIngredient], list, int]:
    grounded = [lookup_ingredient(i, config) for i in ingredients]  # USDA: 0 model tokens
    resp = client.chat(
        model=config.text_model, format="json",
        options={"temperature": config.reason_temperature},
        messages=[{"role": "system", "content": reason_mod._SYSTEM_PROMPT},
                  {"role": "user", "content": reason_mod.render_grounded(grounded)}],
    )
    return grounded, reason_mod.parse_adjustments(resp["message"]["content"]), _toks(resp)


# --- prompt-only methods ------------------------------------------------------

class OneShot:
    name, lever = "one-shot", "baseline"

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        content, tok = _vision(_client(config), _ONESHOT_PROMPT, image_path, config, config.oneshot_temperature)
        return MethodResult(parse_oneshot(content).kcal, tokens=tok, calls=1)


_COT_PROMPT = """Estimate the calories in this meal. Work through it: list each food
with its portion and calories, then sum. Respond ONLY JSON:
{"items": [{"food": "...", "kcal": 0}], "total_kcal": 0}"""


class OneShotCoT:
    name, lever = "one-shot + chain-of-thought", "more reasoning"

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        content, tok = _vision(_client(config), _COT_PROMPT, image_path, config, config.oneshot_temperature)
        return MethodResult(vision_mod._parse_total(content), tokens=tok, calls=1)


_RUBRIC_PROMPT = """Estimate the total calories in this meal. Reference densities,
kcal per 100 g: vegetables ~30, fruit ~60, cooked rice or pasta ~130, bread ~270,
lean cooked meat ~170, cheese ~380, fried or oily food ~280, oil or butter ~800.
Respond ONLY JSON: {"total_kcal": 0}"""


class OneShotRubric:
    name, lever = "one-shot + rubric", "prompt engineering"

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        content, tok = _vision(_client(config), _RUBRIC_PROMPT, image_path, config, config.oneshot_temperature)
        return MethodResult(vision_mod._parse_total(content), tokens=tok, calls=1)


class FewShot:
    name, lever = "few-shot (2 examples)", "in-context examples"

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        client = _client(config)
        msgs: list[dict] = []
        for ex_img, ex_kcal in _FEWSHOT:
            msgs.append({"role": "user", "content": _ONESHOT_PROMPT, "images": [ex_img]})
            msgs.append({"role": "assistant", "content": json.dumps({"kcal": ex_kcal})})
        msgs.append({"role": "user", "content": _ONESHOT_PROMPT, "images": [image_path]})
        resp = client.chat(model=config.vision_model, format="json",
                           options={"temperature": config.oneshot_temperature}, messages=msgs)
        return MethodResult(parse_oneshot(resp["message"]["content"]).kcal, tokens=_toks(resp), calls=1)


class SelfConsistency:
    name, lever, n = "self-consistency (n=5)", "variance reduction", 5

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        client = _client(config)
        vals, tok = [], 0
        for _ in range(self.n):
            content, t = _vision(client, _ONESHOT_PROMPT, image_path, config, 0.7)
            tok += t
            k = parse_oneshot(content).kcal
            if k is not None:
                vals.append(k)
        return MethodResult(statistics.median(vals) if vals else None, tokens=tok, calls=self.n)


class SelfRefine:
    name, lever = "self-refine", "verify & revise"

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        client = _client(config)
        content, tok = _vision(client, _ONESHOT_PROMPT, image_path, config, config.oneshot_temperature)
        first = parse_oneshot(content).kcal
        prompt = (f"Your first calorie estimate was {first}. Look again at portion sizes and "
                  f"hidden calories (oil, sauce, dressing), then give your best final number. "
                  f'Respond ONLY JSON: {{"total_kcal": 0}}')
        content2, t2 = _vision(client, prompt, image_path, config, config.oneshot_temperature)
        final = vision_mod._parse_total(content2)
        return MethodResult(final if final is not None else first, tokens=tok + t2, calls=2)


# --- grounded methods ---------------------------------------------------------

class Decomposed:
    name, lever = "decomposed + USDA", "grounding"

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        cfg = _keyword(config)
        client = _client(cfg)
        content, tok = _vision(client, vision_mod._SYSTEM_PROMPT, image_path, cfg)
        grounded, adj, t2 = _ground_and_reason(client, vision_mod.parse_ingredients(content), cfg)
        return MethodResult(aggregate(grounded, adj).point, tokens=tok + t2, calls=2,
                            detail=f"{sum(g.matched for g in grounded)} foods grounded")


class DecomposedNoReason:
    name, lever = "decomposed (no reason pass)", "ablation"

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        cfg = _keyword(config)
        client = _client(cfg)
        content, tok = _vision(client, vision_mod._SYSTEM_PROMPT, image_path, cfg)
        grounded = [lookup_ingredient(i, cfg) for i in vision_mod.parse_ingredients(content)]
        return MethodResult(aggregate(grounded, []).point, tokens=tok, calls=1)


class Blend:
    name, lever = "grounded + blended", "ensembling"

    def estimate(self, image_path: str, config: Config) -> MethodResult:
        cfg = _keyword(config)
        client = _client(cfg)
        content, tok = _vision(client, vision_mod._FUSED_PROMPT, image_path, cfg)
        total = vision_mod._parse_total(content)
        grounded, adj, t2 = _ground_and_reason(client, vision_mod.parse_ingredients(content), cfg)
        kcal = combine_estimates(total, aggregate(grounded, adj).point, cfg.combine_weight, cfg.combine_clamp)
        return MethodResult(kcal, tokens=tok + t2, calls=2)


# Order is execution priority only (the leaderboard sorts by accuracy). Grounding
# methods are listed first so a flaky bigger model still yields the key capstone
# before the expensive sampling methods run.
METHODS: list[Method] = [
    OneShot(), Decomposed(), Blend(), DecomposedNoReason(),
    OneShotCoT(), OneShotRubric(), FewShot(), SelfRefine(), SelfConsistency(),
]
