"""Harden the light-meal finding: more low-calorie dishes, where the prompt's
prior floors out and the grounded pipeline wins. Appends to results/light.json."""

from __future__ import annotations

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
from benchmark.ood_benchmark import candidates, fetch_image, unload_vision  # noqa: E402

TARGET = 24
LO, HI = 20.0, 230.0  # "light meal" band


def main() -> int:
    cfg = Config.from_env()
    out = HERE / "results" / "light.json"
    done = set()
    records = []
    if out.exists():
        records = json.loads(out.read_text())
        done = {r["dish_id"] for r in records}
    # also skip dishes already measured in the OOD run
    ood = HERE / "results" / "ood.json"
    if ood.exists():
        done |= {r["dish_id"] for r in json.loads(ood.read_text())}

    light = sorted([c for c in candidates() if LO <= c[1] <= HI], key=lambda c: c[1])
    print(f"{len(light)} light candidates; targeting {TARGET} new, LLM_MATCH={cfg.llm_match}")
    for dish_id, kcal in light:
        if len(records) >= TARGET:
            break
        if dish_id in done:
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
                print(f"  {dish_id} truth {kcal:.0f} | one-shot {os_} | pipeline {comp.estimate.point:.0f}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt+1} {dish_id}: {type(e).__name__}")
                unload_vision(cfg)
                time.sleep(2)
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\ndone: {len(records)} light dishes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
