# Methods leaderboard

24 lab-measured Nutrition5k dishes. Ranked by mean error (lower is better).

| rank | method | lever | MAE | median | tokens/dish | vs one-shot |
|--:|---|---|--:|--:|--:|---|
| 1 | few-shot (2 examples) | in-context examples | 80 | 69 | 3511 | better, p=0.96 (n.s.) |
| 2 | one-shot | baseline | 81 | 59 | 1092 | baseline |
| 3 | self-consistency (n=5) | variance reduction | 101 | 102 | 5460 | worse, p=0.05 (n.s.) |
| 4 | one-shot + chain-of-thought | more reasoning | 110 | 82 | 1224 | worse, p=0.20 (n.s.) |
| 5 | grounded + blended | ensembling | 111 | 107 | 1927 | worse, p=0.15 (n.s.) |
| 6 | one-shot + rubric | prompt engineering | 145 | 153 | 1157 | worse, p=0.00 |
| 7 | self-refine | verify & revise | 150 | 115 | 2206 | worse, p=0.00 |
| 8 | decomposed (no reason pass) | ablation | 150 | 80 | 1507 | worse, p=0.04 |
| 9 | decomposed + USDA | grounding | 192 | 180 | 1923 | worse, p=0.00 |