# A 12B model can see your food. It cannot count the calories.

### So I stopped one-shotting it. The same model, wrapped in a small workflow, goes from useless to within 14 calories of the truth. A measured case for workflows over raw prompting on local models.

I wanted to compare two ways of guesstimating the calories in a photo with a small
local model: one-shotting it, versus breaking it into a multi-step workflow. I
scored it against Nutrition5k, a dataset of real plated meals whose calories were
measured in a lab. I chose to run this locally on a small 16GB 4080 in lieu of using
Claude or ChatGPT to better determine gains.

## The OneShot way: ask the model

Show gemma4:12b a photo and ask for the total. It answers instantly and the estimate is not close. Across 24 dishes its mean error was
504 calories, and it tends to guess somewhere between 700 and 1,250 almost
regardless of what is on the plate.

The model is not blind. Ask it to describe the same photo and it is near perfect:
"roasted fish, potato wedges, carrots, a side salad." It sees the food fine. It
just cannot turn what it sees into a calorie number, because that number is a fact
it half-remembers, not something it can read off the image.

## The Multi-step way: a workflow

The workflow splits the
job along that exact seam.

```mermaid
flowchart LR
    P([food photo]) --> O["ONE-SHOT<br/>ask the 12B for the total"] --> OR(["1,250 kcal (wrong)"])
    P --> V["WORKFLOW<br/>12B identifies and portions each food"] --> L["USDA database<br/>looks up each calorie"] --> SUM["sum"] --> WR(["547 kcal (right)"])
    classDef model fill:#6c8ebf,color:#fff,stroke:#34506b;
    classDef bad fill:#d9534f,color:#fff,stroke:#a33;
    classDef good fill:#2e8b6f,color:#fff,stroke:#1f5d4a;
    class O,V model
    class OR bad
    class L,SUM,WR good
```

The model does what it is good at: naming and portioning the food. A database does
what it is good at: the calorie lookup. Code adds it up. The model never produces a
calorie number at all.

Here is one real dish, both ways:

![A worked example: one-shot vs the grounded workflow on one photo](../docs/worked_example.png)

One-shot says 1,250. The workflow identifies the food, grounds each item in USDA
FoodData Central, sums it, and lands at 547. The measured truth was 561, and I found this is not a one off.

## The result

Across all 24 dishes:

![gemma4 12B: one-shot 504 vs workflow 181](../docs/gemma_result.png)

One-shot 504 calories off, workflow was only 181 calories off. The same vision model, wrapped into a small multi step workflow, cuts its error by 64
percent.

## Why it works

A vision model bundles two very different skills: perception (what is this) and
recall (how many calories is that). It is strong at the first and unreliable at the
second. When a model is forced to use both at once, the context is more limited, and the weaker capability drags the
answer down. The workflow uses only the strong skill and hands the weak one to a
source that is actually reliable.

When a model is confidently wrong about facts, do not prompt it harder. Find the one step it is bad
at, and give that step to something built for it, a database, a tool, a calculator.
Let the model do judgment, not lookup.

## What it costs

The honest catch first: the workflow uses more tokens than the one-shot, about 1.7x.
On paper that sounds more expensive, but in actuality it matters which models are using what tokens.

The one-shot spends every token on the 12B. The workflow touches the 12B exactly
once, for the only step that needs eyes, then switches models for the rest: the
reasoning about hidden oil and sugar runs on a 7B, roughly half the size and faster,
and the lookup and the sum run on no model at all. So the extra tokens are cheap
tokens and free tokens, and the expensive model gets used once. The 64 percent gain
itself comes from the USDA lookup, which is an HTTP request to a database, not a
generation, so it costs nothing. I bought the accuracy with a free lookup, not with
more inference.

That is the whole cost argument: spend tokens where they are cheap, and spend the
expensive model only where it is earned. The naive way to chase the same number is
to make the big model work harder, more reasoning, more samples, a bigger model,
which piles cost onto the priciest thing you own. The workflow goes the other way.
And because each stage's model is a single environment variable, you keep
right-sizing: a smaller vision model here, a smaller text model there, nothing for
the lookup. A one-shot is all or nothing, one model paying full price for every job.
Model switching is what lets you stop doing that.

## Limitations

Twenty-four dishes is a small benchmark and the matching is
the weak link: when USDA has no clean row for the food the model named, that item
falls back to the model's own estimate and the grounding cannot help. And this is
one task, calories, where the facts happen to live in a clean public database. Not
every problem has a USDA.

## Reproduce

The estimator and every method are in the repo. The workflow is four stages with
typed contracts: vision identifies and portions, USDA supplies the calories, a
small model probes for hidden oil and sugar, arithmetic sums it.

```bash
python -m calorie_pipeline.run meal.jpg        # one photo, one-shot vs workflow
python benchmark/compare_methods.py            # the full benchmark across methods
python -m unittest discover -s tests           # 63 offline tests
```

Runs local on a 16 GB GPU. Ground truth is Nutrition5k (CC BY 4.0); facts are USDA
FoodData Central. MIT.
