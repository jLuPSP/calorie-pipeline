"""Run every method head to head on the benchmark and rank them honestly.

Resumable: results are saved after each (method, dish), so a model wedge or a
restart never loses work; just run it again and it picks up where it stopped.
Reports a leaderboard ranked by accuracy, with tokens-per-dish and a paired
bootstrap significance test against the one-shot baseline.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "benchmark"
sys.path.insert(0, str(ROOT))

from calorie_pipeline.config import Config  # noqa: E402
from calorie_pipeline.methods import METHODS  # noqa: E402
from evals.datasets import load_manifest  # noqa: E402

RESULTS = HERE / "results" / "methods.json"


def _unload(cfg: Config) -> None:
    try:
        requests.post(f"{cfg.ollama_host}/api/generate",
                      json={"model": cfg.vision_model, "keep_alive": 0}, timeout=30)
    except requests.RequestException:
        pass


def run() -> dict:
    cfg = Config.from_env()
    dishes = load_manifest(HERE / "manifest.json")
    store: dict = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    print(f"{len(METHODS)} methods x {len(dishes)} dishes  (vision={cfg.vision_model})")

    for method in METHODS:
        done = store.setdefault(method.name, {})
        for d in dishes:
            if d.dish_id in done:
                continue
            for attempt in range(3):
                try:
                    r = method.estimate(d.image_path, cfg)
                    done[d.dish_id] = {"kcal": r.kcal, "tokens": r.tokens, "truth": d.ground_truth_kcal}
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  retry {attempt+1} {method.name}/{d.dish_id}: {type(e).__name__}")
                    _unload(cfg)
            RESULTS.write_text(json.dumps(store, indent=2), encoding="utf-8")
        n = len(done)
        print(f"  {method.name:<26} {n}/{len(dishes)} dishes done")
    return store


def score(store: dict) -> list[dict]:
    import numpy as np

    # common dish set across methods, where every method produced a number
    ids = None
    for m in store.values():
        good = {k for k, v in m.items() if v.get("kcal") is not None}
        ids = good if ids is None else (ids & good)
    ids = sorted(ids or [])
    if not ids:
        return []

    base = np.array([abs(store["one-shot"][i]["kcal"] - store["one-shot"][i]["truth"]) for i in ids]) \
        if "one-shot" in store else None
    rng = np.random.default_rng(0)
    rows = []
    for method in METHODS:
        m = store.get(method.name, {})
        if not all(i in m and m[i].get("kcal") is not None for i in ids):
            continue
        err = np.array([abs(m[i]["kcal"] - m[i]["truth"]) for i in ids])
        tok = st.mean([m[i]["tokens"] for i in ids])
        p = None
        if base is not None and method.name != "one-shot":
            diff = base - err
            bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
            p = float(min((bs <= 0).mean(), (bs >= 0).mean()) * 2)
        rows.append({"method": method.name, "lever": method.lever, "n": len(ids),
                     "mae": float(err.mean()), "median": float(np.median(err)),
                     "tokens": round(tok), "p_vs_oneshot": p})
    return sorted(rows, key=lambda r: r["mae"])


def leaderboard(rows: list[dict]) -> str:
    lines = ["# Methods leaderboard", "",
             f"{rows[0]['n']} lab-measured Nutrition5k dishes. Ranked by mean error (lower is better).", "",
             "| rank | method | lever | MAE | median | tokens/dish | vs one-shot |",
             "|--:|---|---|--:|--:|--:|---|"]
    for i, r in enumerate(rows, 1):
        if r["method"] == "one-shot":
            sig = "baseline"
        elif r["p_vs_oneshot"] is None:
            sig = "-"
        else:
            better = r["mae"] < next(x["mae"] for x in rows if x["method"] == "one-shot")
            sig = f"{'better' if better else 'worse'}, p={r['p_vs_oneshot']:.2f}" \
                  + ("" if r["p_vs_oneshot"] < 0.05 else " (n.s.)")
        lines.append(f"| {i} | {r['method']} | {r['lever']} | {r['mae']:.0f} | {r['median']:.0f} | {r['tokens']} | {sig} |")
    return "\n".join(lines)


def main() -> int:
    store = run()
    rows = score(store)
    if not rows:
        print("no scorable results yet")
        return 1
    (HERE / "results" / "methods_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    board = leaderboard(rows)
    (HERE / "results" / "methods_leaderboard.md").write_text(board, encoding="utf-8")
    print("\n" + board)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
