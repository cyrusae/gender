# Zhou et al. (2019) French Shift Ori, rebuilt (2026-11-08)

`embeddings/scripts/zhou_shift.py fr`: MUSE `wiki.multi.fr.vec` (50k), the repo's 23 French
occupation pairs, French definitional pairs (also the attribute words, as notebook cell 215).

| quantity | Table 1 (FR) | notebook's own saved output | rebuilt |
|---|---|---|---|
| MWEAT–Diff, original | 2.3437 (p 0.000) | (cell 232: no output saved) | 2.3958 (p 0.000) |
| Shift Ori | **0.2446 (p 0.147)** | **0.0649 (p 0.019)**, cell 306 | 0.0650 (p 0.005; 94.5% of permutations tie) |
| cos(semantic, grammatical) | not reported | — | 0.363 (SVC), 0.360 (LDA) |

- **The French Shift Ori value in Table 1 doesn't come from the released notebook.** The
  notebook's saved output (0.0649) matches our rebuild (0.0650); Table 1 says 0.2446.
- In the notebook's own run French was **significant** after Shift Ori (p 0.019); Table 1 says
  0.147. Either way 94.5% of permutations tie the observed value, so neither p is a test (Z10).
- The original French statistic is 2% off Table 1 (2.396 vs 2.344); the cell that computed it has
  no saved output, so the difference can't be traced. Possibly a different occupation list
  (cell 228 has a commented-out `FR_occupation_pairs.json` version, not in the repo).
- The notebook's Spanish Hybrid Ori output (cell 305) is 0.309022 with **p 0.269**, a third p-value
  for the same statistic (Table 1: Shift 0.113, Hybrid 0.733).
- Hybrid Ori FR = Shift Ori FR on this statistic by rotation invariance (Table 1 prints 0.2446
  twice, consistent with that); not rebuilt separately.
