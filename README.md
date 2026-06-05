# calorie-pipeline

An honest benchmark of nine ways to estimate calories from a photo, run across two
model sizes against lab-measured ground truth, all local. The finding: capability
is not the lever. Fancier prompting made a one-shot worse, and a bigger model made
it six times worse. The only thing that moved accuracy was grounding the model's
specific weakness in a multimodal workflow.

Full write-up and the reasoning: **[the article](blog/how-to-beat-a-single-prompt.md)**.

![Nine methods ranked by error](docs/leaderboard.png)

## The methods

Each is a small, swappable class in [`calorie_pipeline/methods.py`](calorie_pipeline/methods.py)
with one job: photo in, calories out, plus the tokens it spent. They span the real
levers for improving on a single prompt.

| method | lever |
|---|---|
| one-shot | baseline |
| one-shot + chain-of-thought | more reasoning per call |
| one-shot + rubric | prompt engineering |
| few-shot | in-context examples |
| self-consistency | sample N, take the median |
| self-refine | estimate, then critique it |
| decomposed + USDA | ground the facts in a database |
| decomposed, no reason pass | grounding, ablated |
| grounded + blended | ensemble the grounding with the prompt |

Adding a method is one class and one line in the `METHODS` registry. The study is
a leaderboard of interchangeable parts.

## How it works

The estimator decomposes the task into stages with typed contracts between them:
vision identifies and portions the food, a USDA FoodData Central lookup supplies
the calories per 100 g (no model, just a fact), a small model probes for hidden
oil and sugar, and arithmetic sums it. Two of the four stages use no model. The
grounding stage costs zero tokens because it is an HTTP lookup, not a generation.

The benchmark ([`benchmark/`](benchmark/)) downloads a spread of
[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) dishes,
runs every method on each, and ranks them by error and by tokens, with a paired
bootstrap significance test against the one-shot baseline.

## Reproduce

```bash
pip install -r requirements.txt
ollama pull qwen2.5vl:7b && ollama pull qwen2.5:7b
export OLLAMA_HOST=http://localhost:11434 FDC_API_KEY=DEMO_KEY   # DEMO_KEY works

python -m calorie_pipeline.run meal.jpg        # one photo, both methods + the blend
python benchmark/compare_methods.py            # the full survey + leaderboard
python benchmark/measure_tokens.py             # the token-cost table
python -m unittest discover -s tests           # 63 offline tests, under a second
```

The vision and text models are env-overridable (`VISION_MODEL`, `TEXT_MODEL`), so
swapping in a different or larger model is one variable, no code change.

## Layout

```
calorie_pipeline/   methods.py (the survey) + the staged estimator it draws on
benchmark/          Nutrition5k builder, the comparison harness, results/
docs/               the charts
blog/               the write-up
tests/              63 offline tests, no model or network needed
```

## License

MIT. Nutrition5k is CC BY 4.0; USDA FoodData Central is public domain.
