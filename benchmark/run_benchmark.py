"""Run the one-shot vs decomposed pipeline benchmark over the manifest.

    OLLAMA_HOST=http://your-ollama-host:11434 TEXT_MODEL=qwen2.5:7b-instruct \
    FDC_API_KEY=... python benchmark/run_benchmark.py

Resilient: each dish is independent, partial results are written after every dish
(so a long run is never lost), and a dish that errors is skipped, not fatal.
Writes results.json, a markdown results table, and a predicted-vs-truth chart.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running as `python benchmark/run_benchmark.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calorie_pipeline.config import Config
from calorie_pipeline.models import to_json
from calorie_pipeline.pipeline import compare
from evals.datasets import load_manifest
from evals.metrics import interval_metrics, point_metrics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def run(manifest_path: Path) -> list[dict]:
    dishes = load_manifest(manifest_path)
    cfg = Config.from_env()
    print(f"vision={cfg.vision_model} text={cfg.text_model} host={cfg.ollama_host}")
    print(f"benchmark: {len(dishes)} dishes\n")
    RESULTS.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for i, d in enumerate(dishes, 1):
        try:
            comp = compare(d.image_path, cfg)
        except Exception as e:  # noqa: BLE001 — skip a bad dish, keep the run alive
            print(f"[{i}/{len(dishes)}] {d.dish_id}: SKIP ({type(e).__name__}: {e})")
            continue
        est = comp.estimate
        rec = {
            "dish_id": d.dish_id,
            "truth": d.ground_truth_kcal,
            "oneshot": comp.oneshot.kcal,
            "pl_low": est.low,
            "pl_high": est.high,
            "pl_point": est.point,
            "oneshot_s": round(comp.oneshot_seconds, 1),
            "pipeline_s": round(comp.pipeline_seconds, 1),
            "comparison": json.loads(to_json(comp)),
        }
        records.append(rec)
        os_txt = f"{comp.oneshot.kcal:.0f}" if comp.oneshot.kcal is not None else "FAIL"
        print(
            f"[{i}/{len(dishes)}] {d.dish_id}: truth {d.ground_truth_kcal:.0f}"
            f" | one-shot {os_txt} | pipeline {est.low:.0f}-{est.high:.0f}"
        )
        (RESULTS / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def score(records: list[dict]) -> dict:
    truth = [r["truth"] for r in records]
    pl_point = [r["pl_point"] for r in records]

    os_pairs = [(r["oneshot"], r["truth"]) for r in records if r["oneshot"] is not None]
    os_pred = [p for p, _ in os_pairs]
    os_truth = [t for _, t in os_pairs]
    os_fail = sum(1 for r in records if r["oneshot"] is None)

    return {
        "n": len(records),
        "oneshot_failures": os_fail,
        "oneshot": {
            "point": asdict(point_metrics(os_pred, os_truth)),
            "coverage": interval_metrics(os_pred, os_pred, os_truth).coverage,
        },
        "pipeline": {
            "point": asdict(point_metrics(pl_point, truth)),
            "interval": asdict(
                interval_metrics(
                    [r["pl_low"] for r in records], [r["pl_high"] for r in records], truth
                )
            ),
        },
    }


def write_table(records: list[dict], s: dict) -> None:
    lines = [
        "# Benchmark results — Nutrition5k subset",
        "",
        f"{s['n']} dishes, lab-measured ground truth. Vision `qwen2.5vl:7b`, "
        "probe `qwen2.5:7b`, USDA grounding.",
        "",
        "| dish | truth | one-shot | err | pipeline | err | covered |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in sorted(records, key=lambda r: r["truth"]):
        os_txt = f"{r['oneshot']:.0f}" if r["oneshot"] is not None else "**FAIL**"
        os_err = f"{abs(r['oneshot'] - r['truth']):.0f}" if r["oneshot"] is not None else "—"
        pl_err = abs(r["pl_point"] - r["truth"])
        covered = "✅" if r["pl_low"] <= r["truth"] <= r["pl_high"] else ""
        lines.append(
            f"| {r['dish_id']} | {r['truth']:.0f} | {os_txt} | {os_err} "
            f"| {r['pl_low']:.0f}–{r['pl_high']:.0f} | {pl_err:.0f} | {covered} |"
        )
    o, p = s["oneshot"], s["pipeline"]
    lines += [
        "",
        "## Aggregate",
        "",
        "| metric | one-shot | pipeline |",
        "|---|---:|---:|",
        f"| MAE (kcal) | {o['point']['mae']:.1f} | {p['point']['mae']:.1f} |",
        f"| median abs err (kcal) | {o['point']['median_abs_error']:.1f} | {p['point']['median_abs_error']:.1f} |",
        f"| MAPE (%) | {o['point']['mape']:.1f} | {p['point']['mape']:.1f} |",
        f"| RMSE (kcal) | {o['point']['rmse']:.1f} | {p['point']['rmse']:.1f} |",
        f"| bias (kcal) | {o['point']['bias']:+.1f} | {p['point']['bias']:+.1f} |",
        f"| interval coverage | {o['coverage']:.0%} | **{p['interval']['coverage']:.0%}** |",
        f"| one-shot hard failures | {s['oneshot_failures']} | 0 |",
        "",
    ]
    (RESULTS / "results_table.md").write_text("\n".join(lines), encoding="utf-8")


def write_chart(records: list[dict], s: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(chart skipped: {e})")
        return

    truth = [r["truth"] for r in records]
    pl = [r["pl_point"] for r in records]
    os_t = [r["truth"] for r in records if r["oneshot"] is not None]
    os_p = [r["oneshot"] for r in records if r["oneshot"] is not None]
    lim = max(max(truth), max(pl), *(os_p or [0])) * 1.1

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, lim], [0, lim], "--", color="#888", lw=1, label="perfect")
    ax.scatter(os_t, os_p, c="#d9534f", s=55, alpha=0.8, label=f"one-shot (MAE {s['oneshot']['point']['mae']:.0f})")
    ax.scatter(truth, pl, c="#1f77b4", s=55, alpha=0.8, label=f"pipeline (MAE {s['pipeline']['point']['mae']:.0f})")
    ax.set_xlabel("measured calories (Nutrition5k ground truth)")
    ax.set_ylabel("predicted calories")
    ax.set_title("One-shot vs decomposed pipeline\n(closer to the dashed line is better)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "error_chart.png", dpi=130)
    print(f"wrote {RESULTS / 'error_chart.png'}")


def main(argv: list[str]) -> int:
    if "--rescore" in argv:
        # Re-score / re-chart from saved results without re-running inference.
        records = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
        print(f"rescoring {len(records)} saved dishes")
    else:
        manifest = Path(argv[0]) if argv else HERE / "manifest.json"
        records = run(manifest)
    if not records:
        print("no successful dishes", file=sys.stderr)
        return 1
    s = score(records)
    (RESULTS / "scores.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    write_table(records, s)
    write_chart(records, s)
    print("\n=== AGGREGATE ===")
    print(f"n={s['n']}  one-shot failures={s['oneshot_failures']}")
    print(f"MAE   one-shot {s['oneshot']['point']['mae']:.1f}  | pipeline {s['pipeline']['point']['mae']:.1f}")
    print(f"MAPE  one-shot {s['oneshot']['point']['mape']:.1f}% | pipeline {s['pipeline']['point']['mape']:.1f}%")
    print(f"cover one-shot {s['oneshot']['coverage']:.0%}   | pipeline {s['pipeline']['interval']['coverage']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
