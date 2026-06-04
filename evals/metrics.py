"""Metrics — chosen to expose *what kind* of estimate each method produces.

Point-error metrics (MAE, MAPE, bias) answer "how close is the single number."
That is necessary but not sufficient: it silently rewards confident guessing. A
method that always blurts "650 kcal" can post a respectable MAE on a dataset
centered near 650 while being epistemically worthless.

So we also measure the estimate's *honesty*: does its stated interval actually
contain the truth (coverage), and how wide does it have to be to do so (width)?
A point estimate has zero width and therefore near-zero coverage — by
construction. That is not a bug in the metric; it is the metric reporting that a
point estimate makes no falsifiable claim about its own uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PointMetrics:
    n: int
    mae: float
    mape: float  # mean absolute percentage error, in percent
    rmse: float
    bias: float  # mean signed error (predicted - actual); + = overestimate
    median_abs_error: float


@dataclass(frozen=True, slots=True)
class IntervalMetrics:
    n: int
    coverage: float  # fraction of truths falling within [low, high], 0..1
    mean_width: float
    median_width: float


@dataclass(frozen=True, slots=True)
class MethodReport:
    """Everything we measured about one method over an evaluation set."""

    name: str
    point: PointMetrics
    interval: IntervalMetrics | None  # None for methods that emit no interval


def point_metrics(predicted: Sequence[float], actual: Sequence[float]) -> PointMetrics:
    """Point-error metrics over paired predictions and ground truth."""
    _require_same_length(predicted, actual)
    if not predicted:
        raise ValueError("cannot compute point metrics over an empty set")

    errors = [p - a for p, a in zip(predicted, actual)]
    abs_errors = [abs(e) for e in errors]
    pct_errors = [
        abs(e) / a * 100.0 for e, a in zip(errors, actual) if a != 0
    ]
    return PointMetrics(
        n=len(predicted),
        mae=_mean(abs_errors),
        mape=_mean(pct_errors) if pct_errors else math.nan,
        rmse=math.sqrt(_mean([e * e for e in errors])),
        bias=_mean(errors),
        median_abs_error=_median(abs_errors),
    )


def interval_metrics(
    lows: Sequence[float], highs: Sequence[float], actual: Sequence[float]
) -> IntervalMetrics:
    """Coverage and width over paired intervals and ground truth."""
    _require_same_length(lows, highs)
    _require_same_length(lows, actual)
    if not lows:
        raise ValueError("cannot compute interval metrics over an empty set")

    widths = [hi - lo for lo, hi in zip(lows, highs)]
    covered = [1.0 if lo <= a <= hi else 0.0 for lo, hi, a in zip(lows, highs, actual)]
    return IntervalMetrics(
        n=len(lows),
        coverage=_mean(covered),
        mean_width=_mean(widths),
        median_width=_median(widths),
    )


def _require_same_length(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _median(xs: Sequence[float]) -> float:
    ordered = sorted(xs)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
