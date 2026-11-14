# Spanish word similarity rebuild (exploratory, 2026-11-14; upstream Z25)

SemEval-2017 Task 2 Spanish test set (500 pairs; the task has no French data, Z7). Script
`scripts/word_similarity_es.py` (committed before running, d952f87). Pairs with an item outside the
vocabulary are dropped.

| Vocabulary | Space | Pairs | Pearson | Spearman |
|---|---|---|---|---|
| 50k (Zhou's notebook) | original | 317 | 0.708 | 0.736 |
| 50k | Shift Ori (as implemented) | 317 | 0.708 | 0.737 |
| 200k | original | 369 | 0.717 | 0.740 |
| 200k | Shift Ori | 369 | 0.718 | 0.740 |

Zhou's Table 1 (Pearson, stated): Original 0.7392, Shift Ori 0.7363, Hybrid Ori 0.7358.

**Reading.** "Mitigation preserves word similarity" holds, trivially: Shift moves only the 116
occupation words, which touch 15 of 317 test pairs, and changes the correlation by under 0.001 (in the
opposite direction to Table 1's small drop). Hybrid Ori is identical to Shift on Spanish-internal
cosines (orthogonal alignment, Z11). Table 1's exact values don't reproduce under any combination
tried: our Pearson is 0.71–0.72; the closest matches are Spearman values (0.740 at 200k vs 0.7392;
0.736 at 50k vs Shift's 0.7363), which may be coincidence. Coverage handling, the statistic actually
used, and the evaluation data are unknown without released code (Z25). The Eq. 2 inanimate projection
(never implemented, Z15) would touch many more pairs and isn't tested here.
