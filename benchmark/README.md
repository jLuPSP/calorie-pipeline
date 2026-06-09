# Benchmark — one-shot vs decomposed pipeline on Nutrition5k

This is the empirical backbone of the project's thesis: the *same* local 7B model
produces better, more honest calorie estimates when the work is decomposed than
when it one-shots the photo. We measure it against **lab-measured** ground truth.

## Why Nutrition5k

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (Thames et
al., CVPR 2021, CC BY 4.0) is ~5,000 real plated meals, each with an overhead RGB
photo and **physically measured** total calories. That last part is the point:
the ground truth is a scale-and-lab measurement, not a recipe card's estimate. A
benchmark built on measured truth can actually adjudicate which method is closer,
which a crowd-sourced calorie number cannot.

The same photo feeds both methods, so image quality (these are clinical overhead
shots, not food-blog glamour) affects one-shot and pipeline *equally* — the
relative comparison stays fair regardless.

## What's measured

For each dish, two estimates from the same models:

- **one-shot** — `qwen2.5vl:7b` shown the photo and asked for a single total.
- **pipeline** — vision (`qwen2.5vl:7b`) → USDA FoodData Central grounding (no
  model) → hidden-calorie probe (`qwen2.5:7b`) → aggregate.

Scored on point error (MAE, MAPE, RMSE, bias), interval **coverage** (does the
pipeline's range contain the measured truth — a one-shot point cannot, by
construction), and **hard failures** (the one-shot model returning no usable
number at all).

## Reproduce

```bash
pip install -r requirements.txt requests matplotlib

# 1. metadata (lab-measured calories) from the public Nutrition5k bucket
curl -H "User-Agent: research" -o benchmark/n5k_cafe1.csv \
  https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv
curl -H "User-Agent: research" -o benchmark/n5k_cafe2.csv \
  https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv

# 2. select a spread of dishes + download their overhead images -> manifest.json
python benchmark/build_manifest.py 24

# 3. run both methods on every dish, score, chart
OLLAMA_HOST=http://your-ollama-host:11434 TEXT_MODEL=qwen2.5:7b-instruct \
FDC_API_KEY=... python benchmark/run_benchmark.py
```

Selection is deterministic (dishes evenly spaced across the calorie-sorted
candidates that pass the filters: ≥3 ingredients, 150–800 kcal, ≥150 g), so the
manifest rebuilds identically.

## The headline result

See [`results/summary.md`](results/summary.md). In one line: the single-prompt
baseline beats the decomposed pipeline on accuracy (MAE 90 vs 141), but the two
make weakly-correlated errors, so a **variance-weighted blend of them beats either
alone** (MAE 87) — and letting the model pick the USDA match (`LLM_MATCH=1`) is the
best single-pipeline matcher (141 → 125). The decomposition's value is as an
independent, auditable correction to the prompt, not a replacement for it.

## Outputs (in `results/`)

- `summary.md` — the final method comparison (one-shot, each matcher, the blend).
- `results.json` — full per-dish comparisons, incl. the pipeline's audit trail
  (every grounded ingredient + USDA match + hidden-calorie adjustment).
- `results_table.md` / `scores.json` — per-dish and aggregate scores.
- `error_chart.png` — predicted vs measured calories for both methods.
- `experiment_llm_match.json` — the matcher-isolation experiment (re-grounds the
  saved vision outputs with each matcher, so it costs no vision re-run).
- `results_v{1..4}_*.json` — the four-version matching debugging trail.

## Token cost

`python benchmark/measure_tokens.py` reports real token usage, split by model
(expensive vision model vs cheaper text model). USDA lookups and arithmetic are
free (no model). The raw ratio is model-dependent: a vision model with a heavy
image encode (qwen2.5vl:7b, ~1,100 tok/image) is dominated by the image, so a
fused vision call amortizes it to ~1.7x a single prompt; gemma4:12b encodes the
image cheaply (~150-300 tok), so the cost is the breakdown it generates and the
workflow runs ~4x raw (~2.5x on the vision model, ~3.3x cost-weighted with the
7B at half price). See the blog's cost section.

## Honest caveats

- The overhead Nutrition5k images are lower-resolution and shot top-down; vision
  identification and portioning are harder than on a clean food-blog photo. This
  bounds *absolute* accuracy for both methods but not the *comparison*.
- USDA grounding can't ground what it can't name; unmatched items are surfaced,
  not hidden, and excluded from the base.
- The bundled images are not vendored (see `.gitignore`); the builder script
  re-downloads them from the public bucket. Nutrition5k is CC BY 4.0.
