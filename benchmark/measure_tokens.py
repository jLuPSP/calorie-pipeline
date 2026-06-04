"""Measure real token usage per method, to answer "but you use more tokens".

Runs each model call on the sample photos and reads Ollama's token counters
(prompt_eval_count = input incl. the image, eval_count = output). The USDA lookup
stage uses NO model, so it contributes zero tokens — that matters.

Also measures a FUSED vision call that returns the one-shot total AND the
ingredient breakdown in a single pass, so the combined system doesn't have to
encode the (expensive) image twice.
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

_FUSED_PROMPT = vision._SYSTEM_PROMPT + (
    '\nAlso include your single best estimate of the meal\'s TOTAL calories as a '
    'number under the key "total_kcal", in the SAME JSON object as "ingredients".'
)


def toks(resp) -> tuple[int, int]:
    return int(resp.get("prompt_eval_count", 0) or 0), int(resp.get("eval_count", 0) or 0)


def main() -> int:
    from ollama import Client

    cfg = Config.from_env()
    client = Client(host=cfg.ollama_host)
    totals = {"one-shot": 0, "pipeline (vision+reason)": 0, "fused (1 vision+reason)": 0}

    for img in IMAGES:
        # one-shot: a vision call for a single number
        r = client.chat(model=cfg.vision_model, format="json",
                        messages=[{"role": "user", "content": _ONESHOT_PROMPT, "images": [img]}])
        os_in, os_out = toks(r)

        # pipeline vision: a vision call for the ingredient breakdown
        r2 = client.chat(model=cfg.vision_model, format="json",
                         messages=[{"role": "user", "content": vision._SYSTEM_PROMPT, "images": [img]}])
        dec_in, dec_out = toks(r2)
        ingredients = vision.parse_ingredients(r2["message"]["content"])

        # USDA grounding: NO model -> 0 tokens (the whole point)
        grounded = [lookup_ingredient(i, cfg) for i in ingredients]

        # reason: a text call (no image) for hidden calories
        r3 = client.chat(model=cfg.text_model, format="json",
                         messages=[{"role": "system", "content": reason._SYSTEM_PROMPT},
                                   {"role": "user", "content": reason.render_grounded(grounded)}])
        rea_in, rea_out = toks(r3)

        # fused: ONE vision call for total + breakdown
        rf = client.chat(model=cfg.vision_model, format="json",
                         messages=[{"role": "user", "content": _FUSED_PROMPT, "images": [img]}])
        fus_in, fus_out = toks(rf)

        one = os_in + os_out
        pipe = dec_in + dec_out + rea_in + rea_out
        fused = fus_in + fus_out + rea_in + rea_out
        totals["one-shot"] += one
        totals["pipeline (vision+reason)"] += pipe
        totals["fused (1 vision+reason)"] += fused
        name = img.split("/")[-1]
        print(f"{name:<22} one-shot {one:>6} | pipeline {pipe:>6} | fused {fused:>6}"
              f"  (image~{os_in} tok, USDA lookups=0)")

    print("\n=== totals over 3 photos (tokens) ===")
    base = totals["one-shot"]
    for k, v in totals.items():
        print(f"  {k:<26} {v:>7}   {v / base:.2f}x one-shot")
    combined = totals["one-shot"] + totals["pipeline (vision+reason)"]
    fused_combined = totals["fused (1 vision+reason)"]
    print(f"  {'combined (naive: 1shot+pipe)':<26} {combined:>7}   {combined / base:.2f}x one-shot")
    print(f"  {'combined (FUSED, shipped)':<26} {fused_combined:>7}   {fused_combined / base:.2f}x one-shot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
