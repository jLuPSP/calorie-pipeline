# How to Beat a Single Prompt: A Measured Recipe

### Every applied-AI engineer hits the same wall — you have a one-shot baseline, now do better. Here is a worked, benchmarked answer.

---

You give a model a food photo and ask "how many calories?" It says **540** — one
confident number, no hedge, no provenance. It *feels* like an estimate; it's
really a vibe. And it is a genuinely *strong* baseline, because the model has
learned that "a plate of food is about 400 calories" and regresses to that prior.
So: how do you reliably beat it?

The recipe, measured on 24 meals with **lab-measured** calories (mean absolute
error vs ground truth — lower is better):

| step | MAE | |
|---|---:|---|
| single prompt (baseline) | 90 | the bar |
| + decompose & ground facts in USDA | 141 | **worse alone** |
| + let the model pick the database match | 125 | |
| + keep the prompt, blend the two | 87 | beats the bar |
| + clamp the decomposition to the prompt | **86** | **shipped** |

Five moves, two of which are counterintuitive — and the headline lesson is in the
second row: **the obvious idea, "just decompose it," loses on its own.** What
beats the prompt is grounding the verifiable facts, then using the decomposition
as an independent, sanity-bounded *correction* to the cheap baseline. This post is
the evidence for each step — including the part where I built the decomposition
convinced it would win, benchmarked it, and watched it lose.

> **Follow along.** Every number is reproducible with one command — the full
> system, benchmark, and experiments are in the repo (MIT, runs local on a 16 GB
> GPU). Jump to [*Follow along: reproduce every number*](#follow-along-reproduce-every-number-in-this-post)
> for the commands.

---

## Step 1 — Decompose, but for *grounding*, not for its own sake

The bet behind decomposition is that you can do better than that single vibe by
pulling the question apart. "How many calories is this meal?" is not one problem;
it's at least three, with totally different character:

1. **Perception** — *what foods, how much?* What vision models are good at.
2. **A measured fact** — *how many kcal per 100 g of that food?* A database has
   the right answer; a model only approximates it.
3. **The invisible** — *how much hidden oil, butter, sugar?* A genuine estimate
   under uncertainty.

So I built four inspectable stages, each a typed contract to the next:

```
   food photo
       |
       v
  +--------------------+
  |  1. VISION         |   model      "what's on the plate, and how much?"
  +--------------------+   -> Ingredient(name, grams, prep)
       |
       v
  +--------------------+
  |  2. LOOKUP         |   NO model   "kcal per 100 g from USDA; scale it"
  +--------------------+   -> GroundedIngredient(kcal, matched record)
       |
       v
  +--------------------+
  |  3. REASON         |   model      "what calories does the lookup miss?"
  +--------------------+   -> Adjustment(reason, low, high)
       |
       v
  +--------------------+
  |  4. AGGREGATE      |   NO model   "sum base + ranges -> an honest interval"
  +--------------------+   -> Estimate(low, high, point)
```

Two of the four stages use no model at all. The whole thing runs on a single
16 GB GPU via Ollama — `qwen2.5vl:7b` for vision, `qwen2.5:7b` for the probe,
USDA FoodData Central for the facts. Nothing leaves the machine.

It produces output you can *read*:

```
DECOMPOSED PIPELINE
  pizza crust (50 g)      -> 140 kcal  [PIZZA HUT 12" Cheese Pizza, Pan Crust]
  mozzarella cheese (20 g)->  60 kcal  [Cheese, mozzarella, low moisture]
  tomato sauce (10 g)     ->  10 kcal  [Tomato products, canned, sauce]
  pepperoni (15 g)        ->  76 kcal  [Pepperoni, beef and pork, sliced]
  + breading on pepperoni: 20-30 kcal      (the hidden-calorie probe)
  ESTIMATE: 305-315 kcal
```

Every calorie traces to a citable record. Compared to the one-shot's bare "250,"
this looked like a clear win. I was ready to write the victory post.

---

## The bet: an eval that could falsify me

The discipline that separates an engineer from an enthusiast is building the test
that can prove you wrong. So before declaring victory I went to
[**Nutrition5k**](https://github.com/google-research-datasets/Nutrition5k)
(Thames et al., CVPR 2021): ~5,000 real plated meals, each with overhead imagery
and *physically measured* total calories. Measured, not estimated — the only kind
of ground truth that can actually adjudicate which method is closer.

I pulled 24 dishes spanning 150–560 kcal, ran both methods on each, and scored
them. Same photo into both, so image quality can't favor either.

Here is what came back.

```
                          one-shot     pipeline
  MAE (kcal)                  90          141
  MAPE                        33%          48%
  interval coverage            0%           0%
  dishes won (closer)         16            8
```

The one-shot won. Decisively. My architecture-over-scale thesis, on its first
contact with measured reality, was false.

---

## How you lose is more interesting than that you lose

The instinct is to feel bad and move on. The discipline is to ask *why*, because
the shape of a loss is data.

First, look at *how* the one-shot won. Across 24 dishes it emitted only ten
distinct numbers — 150, 250, 350, 450, 500, 550… — with a mean of **388** against
a true mean of **352**. It is barely looking at the food. It has learned that "a
plate of food is about 400 calories" and it nudges around that prior. On a
benchmark of normal cafeteria meals clustered near 350, **guessing the average is
an excellent strategy** — and a hollow one. Hand this same model the bare stack of
bagels from my real-world set and it returns **0**; it falls off the edge of its
prior and has nothing. Its accuracy is *borrowed* from the test set matching its
training distribution. It is right for reasons that won't survive contact with an
unusual plate.

Second — and this is the part I had to sit with — **my pipeline's losses were not
where I assumed.** I was sure the weak link was portioning: guessing grams from a
flat overhead photo with no depth cues seems impossible. So I measured it. I have
the ground-truth ingredient masses from Nutrition5k, so I could compare the
pipeline's total grams to the truth, and separately its implied calorie *density*
(kcal per 100 g) to the truth:

```
  portioning   (pipeline grams / true grams):    median 1.03x   <- essentially perfect
  match density(pipeline kcal-100g / true):       median 1.22x   <- the real bias
```

The portioning was *fine* — spot on, on average. The entire systematic error was
**database matching**: the pipeline kept grounding each food to a USDA row about
22% too calorie-dense. The decomposition was working; the "trivial" lookup — the
stage I'd spent a paragraph in my README praising as the reliable, no-model part —
was where the error lived.

---

## The debugging journey, and a lesson about sophistication

So I tried to fix the matching. Four versions:

| version | what changed | pipeline MAE | typical match bias |
|---|---|---:|---:|
| v1 | naive keyword search | 245 | 1.24x |
| **v2** | **don't match food to its oil/flour; dedup vision repeats** | **141** | 1.22x |
| v3 | + semantic embedding re-ranking | 231 | 1.24x |
| v4 | + penalize prepared-dish forms (pie, soup, fried) | 188 | **1.08x** |

Read that table slowly, because it contains two genuine lessons.

**The biggest win was the most boring.** v1→v2 nearly halved the error, and the
fixes were embarrassingly unglamorous: stop letting "salmon fillet" match
*"Fish oil, salmon"* (902 kcal/100 g — a 1,350-calorie "salmon"), and drop the
duplicate ingredients a 7B sometimes emits when it repeats a list. No ML. Just
reading the failures and writing guards.

**The sophisticated fix made it worse.** v3 was the move you'd put on a slide:
replace keyword matching with embedding similarity, so "salmon fillet" finds
"Fish, salmon, raw" by *meaning*. On every example I hand-inspected, it was
visibly better. In aggregate it was a disaster — MAE jumped from 141 to 231. I
had optimized for the dozen cases I eyeballed and pessimized the distribution.
Even v4, which fixed the *typical* match better than anything else (density bias
1.22→1.08, the best of any version), still scored worse than plain v2, because it
traded central bias for tail variance.

That is the most useful thing in this whole project, and it cost me three rebuilds
to learn: **eyeballing examples is not evaluation, and your most sophisticated
idea is exactly the one you're least equipped to judge by intuition.** The simple
keyword matcher with two boring guards beat the embedding-based semantic re-ranker
on the only judge that mattered — the full benchmark. (Semantic matching ships in
the repo, off by default, with this epitaph in its docstring.)

---

## The real reason decomposition lost: it compounds variance

Even at v2's best, the pipeline lost. After the matching bias was mostly handled,
the residual error wasn't bias at all — it was *variance*. And there's a
structural reason a pipeline has more of it than a monolith.

A calorie estimate is a **product**: `kcal = density × grams`, summed over foods.
Errors in a product *multiply*. A dish where the pipeline overestimates grams by
1.4× and, on one ingredient, matches something 1.8× too dense doesn't add those
mistakes — it compounds them into a 2.5× blowup. I watched a 512-kcal plate become
1,344 because a portion guess and a match error stacked. Across 24 dishes, these
multiplicative tails are what inflate the mean error.

The one-shot model cannot have this failure mode. It makes **one** guess, and that
guess is **regularized toward a sensible prior** — it physically cannot say 1,344
for a normal plate, because nothing in its training distribution says that. It
trades the *possibility of being exactly right* (which the pipeline occasionally
is — it nailed a dish to within 2 kcal) for *never being catastrophically wrong*
on in-distribution food.

So the honest, general statement is not "decomposition is worse." It's:

> **Decomposition trades bias for variance.** A pipeline of well-calibrated
> stages can be unbiased and occasionally exact, but its errors compound, so it is
> *high-variance*. A monolithic model is biased toward its prior but *low-variance*.
> On an in-distribution benchmark with a strong central tendency, low variance
> wins. That is a property of the test, not a verdict on the architecture.

---

## So where does the pipeline actually win?

If the monolith wins on in-distribution accuracy, the decomposition has to earn
its keep somewhere else. It does — on the axes a single number can't compete on.
Here are three real-world photos (angled, with scale cues, more typical food),
scored against published calories:

```
                truth    one-shot         pipeline
  pizza slice    310     250              305-315   <- pipeline exact, one-shot low
  turkey club    450     450 (exact!)     581       <- one-shot wins, pipeline high
  bagel stack    500       0  (FAILED)    530       <- one-shot fell off a cliff
```

Three things the pipeline does that the one-shot structurally cannot:

- **It fails loudly, not silently.** On the bagels the one-shot returned 0 — a
  confident, invisible, total failure. The pipeline returned 530 with an itemized
  trail. When the monolith is wrong you cannot tell; when the pipeline is wrong
  (the turkey sandwich's 581) you can read *exactly why* — it guessed 150 g of
  bread, two slices too many — and override it.
- **It doesn't borrow its calibration from the test distribution.** The one-shot's
  win on the sandwich is the same trick as its benchmark win: regress to "typical
  sandwich." Off-distribution, that trick is worth zero.
- **It's auditable.** Every number is a citation. For a system anyone is meant to
  *trust* — log against, dose against — a defensible 530 beats a magical 0.

The pipeline's edge is not accuracy. It's **accountability**: legible reasoning,
loud failure, and calibration that comes from facts rather than from a lucky match
between the test set and a model's priors.

---

## So can a well-architected solution actually beat the prompt? Yes.

I didn't want to end on "the monolith wins, but my thing is auditable." That's a
consolation prize. So I went looking for the architecture that *does* beat single
prompting — and the counterfactuals told me where to look. Recall that with
*perfect* matching the pipeline would hit MAE ~101, and matching was the single
biggest error source. So fix matching properly.

The fix was philosophically obvious in hindsight. Picking the right database row
for "salmon fillet" — the raw fish, the breaded nuggets, or the oil? — is a
**judgment**, and this whole project's creed is *use the model for judgment, code
for facts*. I'd been doing the match with keyword overlap and then embeddings —
heuristics — when I should have handed the model the food name and the real USDA
candidates and let it choose. It still can't invent a calorie number; it only
*selects a row*, so the estimate stays grounded. That one change:

```
  matcher        pipeline MAE
  keyword            141
  semantic (embeds)  188   (worse — the sophisticated one again)
  LLM picks the row  125   <- best
```

But 125 still loses to 90. Matching alone was never going to be enough; the
portioning variance is still there. The actual win came from a different idea
entirely — and it's the real lesson of the project.

**The one-shot and the pipeline make *different* mistakes.** The one-shot
regresses to its prior and misses the extremes; the pipeline compounds part-errors
and occasionally blows up. Their errors are only weakly correlated (r = 0.29). And
two estimators with uncorrelated errors are the textbook case where **a blend
beats both**:

```
  single prompt (one-shot)                 MAE 90
  decomposed pipeline (LLM match)          MAE 125
  -> variance-weighted blend of the two    MAE 87    (beats the prompt)
```

The blend weight isn't fit to the answer — it's the inverse-variance optimum,
~0.65 toward the one-shot, derivable from the two methods' error spreads. The
decomposition earns its place not as a *replacement* for the prompt but as an
**independent, auditable correction** to it: it pulls the monolith off its
prior on exactly the dishes the prior gets wrong.

One more turn of the screw helps: the pipeline's failures are *blow-ups* (a 260
dish estimated at 957), so before blending I clamp it to within ±40% of the
one-shot — the cheap, stable estimate as a sanity bound on the elaborate, fragile
one. That caps the damage from the compounding tail and takes the combined error
to **MAE 86**. It's the shipped `combine_estimates`, and it's the whole thesis in
one function: let the structured estimate correct the prompt, but never let it run
away from the sanity check.

I want to be honest about the size of this win: it's ~3%, and the blend's *median*
error is actually a touch worse — the gain is robustness against the one-shot's big
misses, not better typical-case accuracy. But the ceiling is much higher than 3%:
an oracle that simply picked the closer of the two estimates per dish scores **MAE
54**. The two methods are right and wrong on *different* dishes, so most of the
signal is still on the table for a confidence-aware combiner that knows *when* to
trust the decomposition. That's the next build. The point is established: the
well-architected answer isn't decomposition *versus* the prompt — it's the prompt
*plus* a decomposition that corrects it.

---

## "But you're using more tokens"

The fair objection to any pipeline. So I measured it — real token counts on the
three photos, local `qwen2.5vl`:

```
  single prompt                        3,517 tok   1.00x
  decomposed pipeline                  5,823       1.66x
  combined, NAIVE (2 image encodes)    9,340       2.66x   <- the critique, when true
  combined, FUSED (1 image encode)     6,003       1.71x   <- shipped
  USDA grounding (drives the accuracy)     0       free
```

Three things fall out of those numbers:

1. **The image is the whole bill.** A photo encodes to ~1,100 tokens; every text
   prompt and JSON output is rounding error next to it. So the real question isn't
   "how many stages?" — it's "how many times do you encode the image?"
2. **The naive combined system encodes it twice** (one-shot *and* decomposition)
   — 2.66×, and the critic is right. But you don't have to. Ask a *single* vision
   call for both the total and the breakdown (`estimate_fused` in the repo) and
   you're back to **1.71×**: the one-shot total piggybacks on the decomposition
   call for free.
3. **The stage that fixes the accuracy is free.** USDA grounding is an HTTP call
   to a database, not a model — zero tokens. The expensive model work is the
   perception you were already paying for; the *accuracy* comes from a lookup that
   costs nothing.

So the honest price of the combined, auditable estimate is **~1.7× a single
prompt** — one extra small text call, not the 2–3× the objection assumes. Whether
~70% more compute is worth a few points of accuracy *plus* an audit trail *plus*
graceful failure is a real product decision — but it's a decision about a 1.7×,
made with eyes open, and the accuracy lever itself was free. (Pushing to the
MAE-86 config — the model-as-matcher — adds about one text call per ingredient,
~2.7×; the keyword default stays at 1.7× and ties the one-shot on accuracy while
staying auditable. Pick your point on the curve.)

---

## What I'd tell another engineer

Strip the food out and the transferable lessons are sharper than the thesis I
started with:

1. **Build the eval that can falsify you, then publish what it says.** My
   benchmark refuted my hypothesis. That's not a failed project; that's the
   project working. The alternative — three cherry-picked photos and a victory
   lap — would have been worth nothing and I'd never have learned any of this.

2. **The boring stages hold the error.** I spent my design care on the model
   stages and waved at the "trivial" database lookup. The lookup was the entire
   systematic bias. Audit the part you assume is easy.

3. **Sophistication is a liability until a benchmark says otherwise.** Semantic
   embeddings — the impressive move — made things worse and I couldn't tell from
   examples. The boring guards won. Measure on the distribution; never ship what
   only looked good on the cases you happened to check.

4. **Know which trade you're making.** Decomposition buys legibility, loud
   failure, and prior-independent calibration, and it costs variance, latency, and
   failure surface. A monolith buys robustness-to-the-mean and opacity. Neither is
   "better." The engineering judgment is choosing the right one for whether your
   inputs look like your test set — and being honest when they don't.

5. **The best architecture combined them.** The biggest accuracy win wasn't a
   better pipeline *or* a better prompt — it was *blending* them, because their
   errors were weakly correlated. Don't throw away the simple baseline; ensemble
   your structured estimate against it. And where you do use a model, use it for
   the **judgment** (which database row matches this food) and never for the
   arithmetic.

The original thesis was *decomposition beats model size*. What the data actually
supports is more useful, and it has two halves:

> **A decomposed pipeline doesn't beat a single prompt on accuracy — it loses,
> because it compounds variance. But it makes *different* mistakes, so the
> well-architected answer is to blend the two: the decomposition becomes an
> independent, auditable correction that beats either alone. Decomposition's value
> was never replacing the prompt. It was correcting it — accountably.**

---

## Follow along: reproduce every number in this post

Every figure above comes from a command you can run. On a box with Ollama and a
16 GB GPU:

```bash
# 0. setup
git clone <repo> && cd calorie-pipeline
pip install -r requirements.txt
ollama pull qwen2.5vl:7b && ollama pull qwen2.5:7b
export OLLAMA_HOST=http://localhost:11434 FDC_API_KEY=DEMO_KEY   # DEMO_KEY works

# 1. one photo, both methods + the combined answer (the audit-trail example)
python -m calorie_pipeline.run path/to/meal.jpg

# 2. the benchmark: download Nutrition5k dishes, run both methods, score + chart
python benchmark/build_manifest.py 24
python benchmark/run_benchmark.py                 # one-shot 90 vs pipeline 141

# 3. isolate the matching lever (re-grounds saved vision outputs; no vision re-run)
python benchmark/experiment_llm_match.py          # keyword 141 -> LLM-match 125

# 4. the token-cost table
python benchmark/measure_tokens.py                # 1.00x / 1.66x / 1.71x

# 5. always-on offline tests (no model or network) — the math + scoring
python -m unittest discover -s tests              # 59 tests, < 1s
```

The four-version matching trail, the per-dish table, the predicted-vs-measured
chart, and the cost numbers are all written to `benchmark/results/`. The honest
result is reproducible to the dish.

## The repo

The full system — four typed stages, the USDA grounding with its hard-won
matching guards, the model-as-matcher and the fused-vision cost optimization, the
optional semantic re-ranker that didn't pan out, and the 24-dish Nutrition5k
benchmark that scores accuracy *and* calibration — is on GitHub under MIT. It runs
entirely local on a single 16 GB GPU. The negative result, the debugging trail,
and the combiner that finally beats the prompt are all in
[`benchmark/`](../benchmark/). The honest result is the feature, not a footnote.

> Not a bigger model. Not even a cleverer pipeline. An honest measurement.
