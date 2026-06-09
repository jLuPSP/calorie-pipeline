# Methods leaderboard

24 lab-measured Nutrition5k dishes. Ranked by mean error (lower is better).

| rank | method | lever | MAE | median | tokens/dish | vs one-shot |
|--:|---|---|--:|--:|--:|---|
| 1 | grounded + blended | ensembling | 82 | 60 | 1933 | better, p=0.57 (n.s.) |
| 2 | few-shot (2 examples) | in-context examples | 86 | 78 | 3511 | better, p=0.73 (n.s.) |
| 3 | self-consistency (n=5) | variance reduction | 88 | 66 | 5460 | better, p=0.63 (n.s.) |
| 4 | one-shot | baseline | 95 | 63 | 1092 | baseline |
| 5 | one-shot + chain-of-thought | more reasoning | 113 | 112 | 1224 | worse, p=0.54 (n.s.) |
| 6 | self-refine | verify & revise | 142 | 99 | 2206 | worse, p=0.02 |
| 7 | one-shot + rubric | prompt engineering | 149 | 162 | 1157 | worse, p=0.01 |
| 8 | decomposed (no reason pass) | ablation | 153 | 117 | 1510 | worse, p=0.04 |
| 9 | decomposed + USDA | grounding | 202 | 181 | 1907 | worse, p=0.00 |