"""Fresh full-sweep re-run on both models, each to its own results file.

Runs the complete METHODS lineup on gemma4:12b (-> methods.json) and then on
qwen2.5vl:7b (-> methods_7b.json), scoring each consistently. Resumable: reuses
compare_methods.run()'s per-dish checkpointing. Prints the article's headline
numbers (one-shot vs decomposed MAE, win count) for each model at the end.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import compare_methods as cm  # noqa: E402


def sweep(model: str, results_name: str, scores_name: str, board_name: str):
    os.environ["VISION_MODEL"] = model
    cm.RESULTS = HERE / "results" / results_name
    print(f"\n##### SWEEP {model} -> {results_name} #####", flush=True)
    store = cm.run()
    rows = cm.score(store)
    (HERE / "results" / scores_name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        (HERE / "results" / board_name).write_text(cm.leaderboard(rows), encoding="utf-8")
    return store


def article_numbers(store: dict, tag: str) -> None:
    import numpy as np

    def m(method: str):
        d = store.get(method, {})
        return {k: (v["kcal"], v["truth"]) for k, v in d.items() if v.get("kcal") is not None}

    one, dec = m("one-shot"), m("decomposed + USDA")
    shared = sorted(set(one) & set(dec))
    if not shared:
        print(f"[{tag}] no shared dishes yet", flush=True)
        return
    eo = [abs(one[k][0] - one[k][1]) for k in shared]
    ed = [abs(dec[k][0] - dec[k][1]) for k in shared]
    wins = sum(1 for k in shared if abs(dec[k][0] - dec[k][1]) < abs(one[k][0] - one[k][1]))
    distinct = sorted({round(v[0]) for v in one.values()})
    print(f"[{tag}] n={len(shared)}  one-shot MAE={np.mean(eo):.1f}  "
          f"decomposed MAE={np.mean(ed):.1f}  workflow wins={wins}/{len(shared)}  "
          f"one-shot distinct outputs={distinct}", flush=True)


if __name__ == "__main__":
    for f in ("methods.json", "methods_7b.json"):
        p = HERE / "results" / f
        if p.exists():
            p.unlink()

    s12 = sweep("gemma4:12b", "methods.json", "methods_scores.json", "methods_leaderboard.md")
    article_numbers(s12, "12B")
    s7 = sweep("qwen2.5vl:7b", "methods_7b.json", "methods_7b_scores.json", "methods_7b_leaderboard.md")
    article_numbers(s7, "7B")
    print("\nDONE", flush=True)
