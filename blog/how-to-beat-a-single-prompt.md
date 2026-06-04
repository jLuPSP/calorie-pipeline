# How to Beat a Single Prompt: A Measured Recipe

### A local-LLM calorie estimator, an honest benchmark, and the five moves that actually improve on one-shot prompting.

---

You give a model a food photo and ask "how many calories?" It says **540** — one
confident number, no hedge, no provenance. It *feels* like an estimate; it's
really a vibe. And it's a genuinely *strong* baseline: the model has learned that
"a plate of food is about 400 calories" and regresses to that prior. So how do you
reliably beat it?

The recipe, measured on 24 meals with **lab-measured** calories (mean absolute
error vs ground truth — lower is better):

| step | MAE | |
|---|--:|---|
| single prompt (baseline) | 90 | the bar |
| + decompose & ground facts in USDA | 141 | **worse alone** |
| + let the model pick the database match | 125 | |
| + keep the prompt, blend the two | 87 | beats the bar |
| + clamp the decomposition to the prompt | **86** | **shipped** |

The headline is the second row: **the obvious idea — "just decompose it" — loses
on its own.** What beats the prompt is grounding the verifiable facts, then using
the decomposition as an independent, sanity-bounded *correction* to the cheap
baseline. Here's the evidence for each move — including the part where I built the
decomposition convinced it would win, benchmarked it, and watched it lose.

> Everything below reproduces with one command. Repo: MIT, runs local on a 16 GB GPU.

## Step 1 — Decompose, to *ground*, not for its own sake

"How many calories?" is really three problems with different character:
**perception** (what food, how much), a **measured fact** (cal/100 g — a database
has the right answer; a model only approximates it), and **the invisible** (hidden
oil, butter, sugar). A one-shot does all three at once with no way to check it. So
I split them and routed each to the tool that should own it — model for
perception, a **USDA database** for the fact, model for the invisible, arithmetic
for the total — four typed stages: `vision → lookup (no model) → reason →
aggregate`.

The single most important stage is the one with **no model**. The biggest
avoidable error in a calorie estimate is "cal/100 g of this food" — a *measured
fact* the model only approximates ("chicken ≈ 200" when it's 165). Take it off the
model's plate and look it up: free, and exactly right. (It also turned out to be
the *hardest* stage to get right — matching free text to the correct row — more
below.)

## The bet, and the loss

Before declaring victory I did the thing almost nobody does: built a benchmark
that could prove me wrong. **Nutrition5k** — real plated meals with *physically
measured* calories. 24 dishes, both methods, the same photo into each.

The one-shot won. **MAE 90 vs 141**, closer on 16 of 24 dishes.

Why? Two things. First, the one-shot wins by *regressing to the mean*: across 24
dishes it emitted ~10 distinct round numbers (mean 388 vs true 352). It's barely
looking — on in-distribution food, "guess the average" is a great, hollow
strategy. (Hand it an out-of-distribution bagel stack and it returns **0**.)

Second — the part I had to sit with — **the decomposition's error wasn't where I
assumed.** Portioning was fine (grams 1.03× truth). The whole bias was *database
matching*: foods grounded ~22% too calorie-dense. And the errors **compound**:
`kcal = density × grams` is a product, so a portion miss times a match miss becomes
a 2.5× blow-up (a 512-kcal plate → 1,344). The decomposed estimate is
*high-variance* — noisier (std 157) than meals even vary (std 115) — so it loses
to one prior-anchored guess.

## The debugging lesson: boring beats clever

I tried to fix the matching. Four versions:

| | matcher | MAE |
|---|---|--:|
| v1 | naive keyword | 245 |
| **v2** | **don't match a food to its oil; dedup repeats** | **141** |
| v3 | + semantic embeddings | 231 |
| v4 | + penalize prepared-dish forms | 188 |

Two lessons in one table. **The biggest win was the most boring** (v1→v2): stop
letting "salmon fillet" match *"Fish oil, salmon"* (902 kcal/100 g — a
1,350-calorie "salmon"), and dedup the list a 7B sometimes repeats. No ML. And
**the sophisticated fix made it worse** (v3): semantic embeddings looked better on
every example I hand-checked and were a disaster in aggregate. *Eyeballing
examples is not evaluation.*

## The win: correct the prompt, don't replace it

Even fixed, the decomposition loses *alone*. The breakthrough wasn't a better
pipeline — it was noticing the prompt and the pipeline make **different** mistakes
(their errors correlate just 0.29). Two estimators with uncorrelated errors is the
textbook case where a blend beats both:

- **Blend** them, weighted ~0.65 toward the prompt — the inverse-variance optimum,
  derived from the error spreads, *not* fit to the answer. → **MAE 87.**
- **Clamp** the decomposition to ±40% of the prompt first — the cheap, stable
  estimate as a sanity bound on the fragile one, so its blow-ups can't run away.
  → **MAE 86, beats the baseline.**

That's the shipped `combine_estimates`, and it's the whole thesis in one function:
*let the structured estimate correct the prompt, but never let it run from the
sanity check.* (Two asides: letting the model **pick** the database row — a
judgment, not arithmetic — beat the heuristics, 141→125. And an agentic
self-correcting version caught some blow-ups but looped, cost ~2.5× the tokens,
and didn't beat the static clamp — autonomy doesn't pay on a fixed-path task.)

## "But you use more tokens"

The fair objection. Measured:

```
  single prompt                       1.00×   (the image is ~1,100 tok; text is noise)
  decomposed pipeline                 1.66×
  combined, naive (2 image encodes)   2.66×   ← the critique, when true
  combined, fused (1 image encode)    1.71×   ← shipped
  USDA grounding (drives accuracy)    free    (an HTTP lookup, not a model)
```

The image is the whole bill, so the only question is *how many times you encode
it.* A fused vision call returns the total **and** the breakdown in one pass →
**1.71×**, and the stage that fixes the accuracy costs **zero tokens**. ~70% more
compute for a few points of accuracy plus an audit trail — a decision made with
eyes open, not the 3× the objection assumes.

## What I'd tell another engineer

1. **Build the eval that can falsify you, then publish what it says.** Mine
   refuted my own thesis — that's the project working, not failing.
2. **The boring stages hold the error.** All the systematic bias lived in the
   "trivial" database lookup, not the AI. Audit the part you assume is easy.
3. **Sophistication is a liability until a benchmark says otherwise.** Embeddings
   beat my eyeballs and lost the aggregate. Measure on the distribution.
4. **Don't replace the prompt — correct it.** Ground the verifiable facts for
   free, keep the cheap baseline, and ensemble the two with a sanity bound.

The recipe in one line: **ground the parts that have a right answer, and use a
decomposition to *correct* the prompt — not to replace it.**

## Reproduce

```bash
git clone <repo> && cd calorie-pipeline && pip install -r requirements.txt
ollama pull qwen2.5vl:7b && ollama pull qwen2.5:7b
export OLLAMA_HOST=http://localhost:11434 FDC_API_KEY=DEMO_KEY

python -m calorie_pipeline.run meal.jpg          # one photo, both methods + the blend
python benchmark/build_manifest.py 24 && python benchmark/run_benchmark.py   # the benchmark
python benchmark/measure_tokens.py               # the token table
python -m unittest discover -s tests             # 59 offline tests, < 1s
```

Everything — the four-version trail, the per-dish table, the chart of every dish —
is in [`benchmark/`](../benchmark/). The negative result is the feature, not a
footnote.
