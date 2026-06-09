"""Measure real token usage per method, to answer "but you use more tokens".

Runs each model call on the sample photos and reads Ollama's token counters
(prompt_eval_count = input incl. the image, eval_count = output). The USDA lookup
stage uses NO model, so it contributes zero tokens — that matters.

Reports the split by MODEL (expensive vision model vs cheaper text model), so the
cost question can be answered honestly: not just "how many tokens" but "how many
tokens on the model you actually pay for." Small vision models like gemma4:12b
encode the image cheaply (~150 tok), so unlike a 1k-token-image model the raw
ratio is dominated by the breakdown the vision model generates, not by the image.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calorie_pipeline import reason, vision
from calorie_pipeline.config import Config
from calorie_pipeline.lookup import lookup_ingredient
from calorie_pipeline.pipeline import _ONESHOT_PROMPT

_ROOT = Path(__file__).resolve().parent.parent
IMAGES = [
    str(_ROOT / "evals/sample_images/pizza.jpg"),
    str(_ROOT / "evals/sample_images/turkey_sandwich.jpg"),
    str(_ROOT / "evals/sample_images/bagels.jpg"),
]


def toks(resp) -> tuple[int, int]:
    return int(resp.get("prompt_eval_count", 0) or 0), int(resp.get("eval_count", 0) or 0)


def _vision_breakdown(client, model, prompt, img):
    """A vision call for the ingredient breakdown, retried until it parses.

    Small models occasionally emit non-JSON; the benchmark retries rather than
    crash, so we do the same here to get an honest token count.
    """
    last = None
    for _ in range(4):
        r = client.chat(model=model, format="json",
                        messages=[{"role": "user", "content": prompt, "images": [img]}])
        last = r
        try:
            ingredients = vision.parse_ingredients(r["message"]["content"])
            return r, ingredients
        except Exception:  # noqa: BLE001 — non-JSON output, retry
            continue
    return last, []


def main() -> int:
    from ollama import Client

    cfg = Config.from_env()
    client = Client(host=cfg.ollama_host)
    # Per-model accounting: every vision call is the expensive model, the reason
    # call is the cheap text model, USDA is free.
    expensive = {"one-shot": 0, "pipeline": 0}
    cheap = {"pipeline": 0}

    for img in IMAGES:
        # one-shot: a vision call for a single number (expensive model)
        r = client.chat(model=cfg.vision_model, format="json",
                        messages=[{"role": "user", "content": _ONESHOT_PROMPT, "images": [img]}])
        os_in, os_out = toks(r)
        one = os_in + os_out

        # pipeline vision: a vision call for the ingredient breakdown (expensive model)
        r2, ingredients = _vision_breakdown(client, cfg.vision_model, vision._SYSTEM_PROMPT, img)
        dec_in, dec_out = toks(r2)
        vis = dec_in + dec_out

        # USDA grounding: NO model -> 0 tokens (the whole point)
        grounded = [lookup_ingredient(i, cfg) for i in ingredients]

        # reason: a text call (no image) for hidden calories (cheap model)
        r3 = client.chat(model=cfg.text_model, format="json",
                         messages=[{"role": "system", "content": reason._SYSTEM_PROMPT},
                                   {"role": "user", "content": reason.render_grounded(grounded)}])
        rea_in, rea_out = toks(r3)
        rea = rea_in + rea_out

        expensive["one-shot"] += one
        expensive["pipeline"] += vis
        cheap["pipeline"] += rea
        name = img.split("/")[-1]
        print(f"{name:<22} one-shot {one:>5} (img~{os_in}) | "
              f"pipeline: vision[{cfg.vision_model}] {vis:>5} + reason[{cfg.text_model}] {rea:>5}")

    one_total = expensive["one-shot"]
    pipe_expensive = expensive["pipeline"]
    pipe_cheap = cheap["pipeline"]
    pipe_total = pipe_expensive + pipe_cheap

    print(f"\n=== totals over {len(IMAGES)} photos (vision={cfg.vision_model}, text={cfg.text_model}) ===")
    print(f"  one-shot                       {one_total:>6} tok   (all on {cfg.vision_model})")
    print(f"  workflow total                 {pipe_total:>6} tok   {pipe_total / one_total:.2f}x one-shot")
    print(f"    on expensive ({cfg.vision_model}) {pipe_expensive:>6} tok   {pipe_expensive / one_total:.2f}x one-shot")
    print(f"    on cheap ({cfg.text_model})  {pipe_cheap:>6} tok   ({100 * pipe_cheap / pipe_total:.0f}% of workflow tokens)")
    print(f"  USDA lookup + arithmetic            0 tok   (no model)")
    # Cost-weight: count cheap-model tokens at half price (a ~7B vs ~12B proxy).
    cost = (one_total) and (pipe_expensive + 0.5 * pipe_cheap) / one_total
    print(f"  cost-weighted (cheap model at 0.5x): {cost:.2f}x one-shot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
