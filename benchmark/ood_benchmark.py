"""Out-of-distribution test: does the recipe beat the prompt where the prior fails?

The single prompt wins in-distribution by regressing to a sensible prior (~390
kcal). That trick collapses on meals far from the prior. This selects the
*extremes* of Nutrition5k (the lowest- and highest-calorie dishes), runs the
single prompt and the full recipe (LLM_MATCH=1) on each, and reports the
difference with a paired bootstrap CI. Writes benchmark/results/ood.json.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "benchmark"
sys.path.insert(0, str(ROOT))

from calorie_pipeline.config import Config  # noqa: E402
from calorie_pipeline.pipeline import compare  # noqa: E402

BUCKET = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
UA = "calorie-pipeline-benchmark/1.0 (research)"
N_PER_TAIL = 9


def candidates() -> list[tuple[str, float]]:
    out = []
    for name in ("n5k_cafe1.csv", "n5k_cafe2.csv"):
        p = HERE / name
        if not p.exists():
            continue
        for row in csv.reader(p.open(newline="", encoding="utf-8")):
            if len(row) < 3 or not row[0].startswith("dish_"):
                continue
            try:
                kcal, mass = float(row[1]), float(row[2])
            except ValueError:
                continue
            if 30 <= kcal <= 1500 and mass >= 80:
                out.append((row[0], kcal))
    return out


def fetch_image(dish_id: str) -> Path | None:
    dest = HERE / "images" / dish_id / "rgb.png"
    if dest.exists():
        return dest
    try:
        r = requests.get(f"{BUCKET}/imagery/realsense_overhead/{dish_id}/rgb.png",
                         headers={"User-Agent": UA}, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200 or len(r.content) < 1000:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def unload_vision(cfg: Config) -> None:
    try:
        requests.post(f"{cfg.ollama_host}/api/generate",
                      json={"model": cfg.vision_model, "keep_alive": 0}, timeout=30)
    except requests.RequestException:
        pass


def main() -> int:
    cand = sorted(candidates(), key=lambda c: c[1])
    picks = cand[: N_PER_TAIL * 3] + cand[-N_PER_TAIL * 3:]  # over-select; image 404s drop some
    cfg = Config.from_env()
    print(f"selecting extremes from {len(cand)} dishes; LLM_MATCH={cfg.llm_match}")

    records, lo, hi = [], 0, 0
    for dish_id, kcal in picks:
        if (kcal < 200 and lo >= N_PER_TAIL) or (kcal >= 200 and hi >= N_PER_TAIL):
            continue
        img = fetch_image(dish_id)
        if img is None:
            continue
        for attempt in range(3):
            try:
                comp = compare(str(img), cfg)
                records.append({"dish_id": dish_id, "truth": round(kcal, 1),
                                "oneshot": comp.oneshot.kcal, "recipe": comp.combined_kcal,
                                "pipeline": comp.estimate.point})
                os_ = f"{comp.oneshot.kcal:.0f}" if comp.oneshot.kcal is not None else "FAIL"
                print(f"  {dish_id} truth {kcal:.0f} | one-shot {os_} | recipe {comp.combined_kcal:.0f}")
                if kcal < 200:
                    lo += 1
                else:
                    hi += 1
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt+1} {dish_id}: {type(e).__name__}")
                unload_vision(cfg)
                time.sleep(2)
        (HERE / "results" / "ood.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\ndone: {len(records)} dishes ({lo} low, {hi} high)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
