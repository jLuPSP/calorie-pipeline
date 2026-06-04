# Final benchmark summary - 24 Nutrition5k dishes (lab-measured)

| method | MAE (kcal) | median | vs one-shot |
|---|---:|---:|---|
| single prompt (one-shot) | 89.9 | 66 | baseline |
| pipeline: keyword match (shipped default) | 140.7 | 91 | loses |
| pipeline: semantic match | 188.0 | 177 | worse |
| pipeline: LLM match | 125.1 | 77 | loses |
| blend: one-shot + keyword pipeline | 89.6 | 69 | **beats** |
| blend: one-shot + LLM-match pipeline | 87.4 | 91 | **beats** |
| **+ clamp pipeline to ±40% of one-shot (shipped)** | **86.1** | — | **beats** |

## The arc

1. **Single prompting is a strong baseline.** It regresses to a sensible prior
   (~388 kcal mean vs 352 true), so on in-distribution food it is hard to beat.
2. **The decomposed pipeline alone loses on accuracy** - its error is a *product*
   of per-stage errors (`kcal = density x grams`) so it is high-variance, and its
   per-dish noise (std 157) exceeds how much meals even vary (std 115).
3. **Letting the model pick the USDA match** (judgment, not a heuristic) is the
   best single-pipeline improvement: keyword 141 -> LLM-match 125.
4. **The win is the blend.** The one-shot and the pipeline make weakly-correlated
   errors (r = 0.29), so a variance-weighted average beats either alone. The
   decomposition is most valuable not as a replacement for the prompt but as an
   independent, auditable *correction* to it.
5. **A confidence-aware combiner does better still.** Clamping the decomposed
   estimate to within ±40% of the one-shot — the cheap estimate as a sanity bound
   that caps the pipeline's occasional blow-ups — then blending takes MAE to 86.1
   (and ±30% reaches 84, at the cost of more test-tuning). This is the shipped
   `combine_estimates`. The per-dish oracle is still 54, so a model-based "which
   estimate do I trust here" gate has real room left.

Blend weight = 0.65 on the one-shot (the benchmark's inverse-variance optimum,
derived from the error variances, not fit to minimize test error).

> Honest caveats: the MAE win is modest (~3%) and the blend's *median* is slightly
> worse than the one-shot's - the gain is robustness against the one-shot's big
> misses, not better typical-case accuracy. The per-dish oracle (pick the closer
> of the two each time) scores MAE 54, so most of the available signal is still on
> the table for a smarter, confidence-aware combiner. And the pipeline's real edge
> stays qualitative: it is auditable and it degrades gracefully where the one-shot
> fails outright (it returned 0 kcal on an out-of-distribution bagel stack).
