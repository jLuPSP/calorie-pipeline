"""Evaluation harness for the decomposition-beats-model-size thesis.

This package measures the claim instead of asserting it. It scores two methods —
the one-shot baseline and the decomposed pipeline — against ground-truth meal
calories, on metrics chosen to expose *what kind* of estimate each produces:

* point error (MAE / MAPE / bias) — how close the single number is;
* interval coverage and width — whether the estimate is honest about its own
  uncertainty, the axis on which a point estimate cannot compete.

Two entry points:

* :func:`evals.harness.evaluate_fixtures` — deterministic, offline, no model or
  network. Proves the metrics and data flow (and ships as the default so anyone
  can reproduce the *shape* of the result in one command).
* :func:`evals.harness.evaluate_live` — the real thing, over a Nutrition5k
  manifest, using local models on your box.
"""

from __future__ import annotations

__all__ = ["datasets", "metrics", "fixtures", "harness"]
