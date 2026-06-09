"""Batch live comparison — run one-shot vs the decomposed pipeline on N photos.

Reproduces the "real run" numbers from the README/blog. Requires a reachable
Ollama (with the vision + text models) and network access to USDA.

    OLLAMA_HOST=http://your-ollama-host:11434 \
    TEXT_MODEL=qwen2.5:7b-instruct \
    FDC_API_KEY=... \
    python run_live.py                       # the three bundled sample photos
    python run_live.py path/to/a.jpg b.jpg   # or your own

Writes the full structured comparisons (no secrets) to
``evals/results/live_results.json`` for later analysis.
"""

from __future__ import annotations

import json
import pathlib
import sys

from calorie_pipeline.config import Config
from calorie_pipeline.models import to_json
from calorie_pipeline.pipeline import compare
from calorie_pipeline.run import render_comparison

_DEFAULT_IMAGES = [
    "evals/sample_images/pizza.jpg",
    "evals/sample_images/turkey_sandwich.jpg",
    "evals/sample_images/bagels.jpg",
]


def main(argv: list[str] | None = None) -> int:
    paths = argv if argv else _DEFAULT_IMAGES
    cfg = Config.from_env()
    print(f"vision={cfg.vision_model}  text={cfg.text_model}  host={cfg.ollama_host}")
    print(f"fdc chain={cfg.fdc_data_types}\n")

    results = []
    for path in paths:
        comp = compare(path, cfg)
        print(render_comparison(comp))
        print()
        results.append({"image": path, "comparison": json.loads(to_json(comp))})

    out = pathlib.Path("evals/results/live_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
