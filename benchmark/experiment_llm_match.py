"""Experiment: does letting the LLM PICK the USDA match beat heuristic matching?

Re-grounds the *already-saved* vision outputs (food names + grams) from the
benchmark with three matchers and re-scores — so we isolate the matching lever
without re-running the expensive vision stage.

  matcher = keyword  : the shipped heuristic (token overlap + concentrate guard)
  matcher = llm      : give the text model the food + real USDA candidates,
                       let it pick the best row (judgment), code does the kcal.

Usage: python benchmark/experiment_llm_match.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calorie_pipeline.config import Config
from calorie_pipeline.lookup import (
    _clean_query,
    _search_candidates,
    extract_energy_kcal,
)

HERE = Path(__file__).resolve().parent


def candidate_pool(name: str, config: Config) -> list[tuple[str, float]]:
    """All energy-bearing USDA candidates for a food, deduped by description."""
    query = _clean_query(name)
    seen: dict[str, float] = {}
    for data_type in config.fdc_data_types:
        try:
            cands = _search_candidates(query, config, data_type, session=None)
        except Exception:  # noqa: BLE001
            continue
        for c in cands:
            energy = extract_energy_kcal(c)
            desc = str(c.get("description", ""))
            if energy is not None and desc and desc not in seen:
                seen[desc] = energy
    return list(seen.items())[:12]


def llm_pick(name: str, pool: list[tuple[str, float]], client, model: str) -> int:
    """Ask the text model which candidate row best matches the food."""
    options = "\n".join(f"{i}: {desc}" for i, (desc, _) in enumerate(pool))
    prompt = (
        f"A food was identified in a meal photo as: \"{name}\".\n"
        f"Which USDA database entry below is the SAME food in its plain, as-served "
        f"form? Avoid concentrates (oil, flour, juice), and avoid unrelated "
        f"prepared dishes (pie, soup, chips) unless the food name itself says so.\n\n"
        f"{options}\n\n"
        f'Respond with ONLY JSON: {{"index": <number>}}'
    )
    resp = client.chat(
        model=model,
        format="json",
        options={"temperature": 0.0},
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        idx = int(json.loads(resp["message"]["content"]).get("index", 0))
    except (ValueError, TypeError, json.JSONDecodeError):
        idx = 0
    return idx if 0 <= idx < len(pool) else 0


def main() -> int:
    from ollama import Client

    config = Config.from_env()
    client = Client(host=config.ollama_host)
    mani = {d["dish_id"]: d for d in json.loads((HERE / "manifest.json").read_text())["dishes"]}
    records = json.loads((HERE / "results" / "results.json").read_text())

    rows = []
    for x in records:
        grounded = x["comparison"]["estimate"]["grounded"]
        adj = x["comparison"]["estimate"]["adjustments"]
        adj_mid = sum((a["low"] + a["high"]) / 2 for a in adj)

        kw_base = sum(g["kcal"] for g in grounded if g["kcal"] is not None)
        llm_base = 0.0
        for g in grounded:
            name = g["ingredient"]["name"]
            grams = g["ingredient"]["grams"]
            pool = candidate_pool(name, config)
            if not pool:
                continue
            idx = llm_pick(name, pool, client, config.text_model)
            llm_base += pool[idx][1] * grams / 100.0

        rows.append(
            {
                "dish_id": x["dish_id"],
                "truth": x["truth"],
                "oneshot": x["oneshot"],
                "kw": kw_base + adj_mid,
                "llm": llm_base + adj_mid,
            }
        )
        print(
            f"{x['dish_id']}: truth {x['truth']:.0f} | one-shot {x['oneshot']:.0f}"
            f" | keyword {rows[-1]['kw']:.0f} | LLM-match {rows[-1]['llm']:.0f}"
        )

    def mae(key):
        return round(st.mean([abs(r[key] - r["truth"]) for r in rows]))

    (HERE / "results" / "experiment_llm_match.json").write_text(json.dumps(rows, indent=2))
    print("\n=== matching-lever isolation (same vision outputs) ===")
    print(f"one-shot     MAE {mae('oneshot')}")
    print(f"keyword      MAE {mae('kw')}   (shipped)")
    print(f"LLM-match    MAE {mae('llm')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
