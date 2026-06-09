# Methods leaderboard

12 lab-measured Nutrition5k dishes. Ranked by mean error (lower is better).

| rank | method | lever | MAE | median | tokens/dish | vs one-shot |
|--:|---|---|--:|--:|--:|---|
| 1 | few-shot (2 examples) | in-context examples | 104 | 77 | 641 | better, p=0.00 |
| 2 | decomposed (no reason pass) | ablation | 147 | 129 | 583 | better, p=0.00 |
| 3 | decomposed + USDA | grounding | 148 | 122 | 960 | better, p=0.00 |
| 4 | grounded + blended | ensembling | 162 | 108 | 1049 | better, p=0.00 |
| 5 | one-shot + rubric | prompt engineering | 231 | 122 | 254 | better, p=0.01 |
| 6 | one-shot + chain-of-thought | more reasoning | 238 | 242 | 274 | better, p=0.01 |
| 7 | self-refine | verify & revise | 467 | 198 | 398 | better, p=0.61 (n.s.) |
| 8 | one-shot | baseline | 507 | 498 | 188 | baseline |
| 9 | self-consistency (n=5) | variance reduction | 598 | 782 | 938 | worse, p=0.26 (n.s.) |