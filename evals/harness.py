"""Harness — score one-shot vs decomposed pipeline against ground truth.

    python -m evals.harness                          # offline fixtures (default)
    python -m evals.harness --nutrition5k dish_metadata_cafe1.csv \\
                            --imagery imagery/realsense_overhead   # live, on your box
    python -m evals.harness --manifest evals/manifest.sample.json  # live, manifest

Offline mode uses the deterministic fixtures and touches no model or network, so
it always runs and always reproduces the same numbers. Live mode runs the real
pipeline (local models + USDA) over a dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from calorie_pipeline.config import Config
from calorie_pipeline.models import Estimate
from calorie_pipeline.pipeline import aggregate
from evals.datasets import GroundTruthDish, load_manifest, load_nutrition5k
from evals.fixtures import FixtureDish, fixture_dishes
from evals.metrics import (
    MethodReport,
    interval_metrics,
    point_metrics,
)


@dataclass(frozen=True, slots=True)
class DishRecord:
    """Per-dish outcome: ground truth and both methods' numbers."""

    dish_id: str
    ground_truth_kcal: float
    oneshot_kcal: float | None
    pipeline_low: float
    pipeline_high: float
    pipeline_point: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The scored comparison over a dataset."""

    records: tuple[DishRecord, ...]
    oneshot: MethodReport
    pipeline: MethodReport


def _score(records: list[DishRecord]) -> EvalReport:
    """Compute both method reports from per-dish records.

    One-shot dishes with no parseable number are dropped from the one-shot
    scoring (and that drop is visible in ``oneshot.point.n``) rather than
    silently counted as zero.
    """
    actual = [r.ground_truth_kcal for r in records]

    os_pairs = [(r.oneshot_kcal, r.ground_truth_kcal) for r in records if r.oneshot_kcal is not None]
    os_pred = [p for p, _ in os_pairs]
    os_actual = [a for _, a in os_pairs]
    oneshot = MethodReport(
        name="one-shot",
        point=point_metrics(os_pred, os_actual),
        # A point estimate's interval is degenerate (low == high == point); we
        # report its coverage explicitly to make the false-precision visible.
        interval=interval_metrics(os_pred, os_pred, os_actual),
    )

    pipeline = MethodReport(
        name="pipeline",
        point=point_metrics([r.pipeline_point for r in records], actual),
        interval=interval_metrics(
            [r.pipeline_low for r in records],
            [r.pipeline_high for r in records],
            actual,
        ),
    )
    return EvalReport(records=tuple(records), oneshot=oneshot, pipeline=pipeline)


def evaluate_fixtures(dishes: list[FixtureDish] | None = None) -> EvalReport:
    """Score the offline fixtures — no model, no network, fully deterministic."""
    dishes = fixture_dishes() if dishes is None else dishes
    records: list[DishRecord] = []
    for d in dishes:
        estimate: Estimate = aggregate(list(d.grounded), list(d.adjustments))
        records.append(
            DishRecord(
                dish_id=d.dish_id,
                ground_truth_kcal=d.ground_truth_kcal,
                oneshot_kcal=d.oneshot_kcal,
                pipeline_low=estimate.low,
                pipeline_high=estimate.high,
                pipeline_point=estimate.point,
            )
        )
    return _score(records)


def evaluate_live(dishes: list[GroundTruthDish], config: Config) -> EvalReport:
    """Score the real pipeline over a dataset. Requires images, models, network."""
    from calorie_pipeline.pipeline import compare  # lazy: pulls in ollama

    records: list[DishRecord] = []
    for d in dishes:
        if not d.image_path:
            continue
        comparison = compare(d.image_path, config)
        est = comparison.estimate
        records.append(
            DishRecord(
                dish_id=d.dish_id,
                ground_truth_kcal=d.ground_truth_kcal,
                oneshot_kcal=comparison.oneshot.kcal,
                pipeline_low=est.low,
                pipeline_high=est.high,
                pipeline_point=est.point,
            )
        )
    if not records:
        raise ValueError("no dishes with resolvable images to evaluate")
    return _score(records)


def render_report(report: EvalReport) -> str:
    """Render the scored comparison as a readable table."""
    lines = ["", "Per-dish:", "-" * 72]
    lines.append(f"{'dish':<26}{'truth':>7}{'one-shot':>10}{'pipeline (range)':>22}")
    for r in report.records:
        os_txt = f"{r.oneshot_kcal:.0f}" if r.oneshot_kcal is not None else "n/a"
        rng = f"{r.pipeline_low:.0f}-{r.pipeline_high:.0f}"
        lines.append(
            f"{r.dish_id:<26}{r.ground_truth_kcal:>7.0f}{os_txt:>10}{rng:>22}"
        )

    lines += ["", "Method scores:", "-" * 72]
    lines.append(f"{'metric':<22}{'one-shot':>14}{'pipeline':>14}")
    o, p = report.oneshot, report.pipeline
    lines.append(f"{'n':<22}{o.point.n:>14}{p.point.n:>14}")
    lines.append(f"{'MAE (kcal)':<22}{o.point.mae:>14.1f}{p.point.mae:>14.1f}")
    lines.append(f"{'MAPE (%)':<22}{o.point.mape:>14.1f}{p.point.mape:>14.1f}")
    lines.append(f"{'RMSE (kcal)':<22}{o.point.rmse:>14.1f}{p.point.rmse:>14.1f}")
    lines.append(f"{'bias (kcal)':<22}{o.point.bias:>+14.1f}{p.point.bias:>+14.1f}")
    if o.interval and p.interval:
        lines.append(
            f"{'interval coverage':<22}{o.interval.coverage:>13.0%}{p.interval.coverage:>14.0%}"
        )
        lines.append(
            f"{'mean width (kcal)':<22}{o.interval.mean_width:>14.0f}{p.interval.mean_width:>14.0f}"
        )
    lines.append("-" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.harness",
        description="Score one-shot vs decomposed pipeline against ground truth.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--nutrition5k", metavar="CSV", help="Nutrition5k dish_metadata CSV")
    src.add_argument("--manifest", metavar="JSON", help="JSON evaluation manifest")
    parser.add_argument("--imagery", metavar="DIR", help="Nutrition5k overhead imagery dir")
    args = parser.parse_args(argv)

    if args.nutrition5k:
        dishes = load_nutrition5k(args.nutrition5k, args.imagery)
        report = evaluate_live(dishes, Config.from_env())
    elif args.manifest:
        report = evaluate_live(load_manifest(args.manifest), Config.from_env())
    else:
        print("(offline fixtures - deterministic, no model or network)")
        report = evaluate_fixtures()

    print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
