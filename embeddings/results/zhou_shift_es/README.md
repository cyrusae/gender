# Zhou et al. (2019) Spanish Shift Ori, rebuilt (2026-11-08)

`embeddings/scripts/zhou_shift.py`: MUSE `wiki.multi.es.vec` (first 50k words), the repo's 58
occupation pairs, 10 definitional pairs, `masculine_nouns_es`/`feminine_nouns_es`. Their repo has
no mitigated embeddings, so Shift Ori is re-implemented from the paper (§3.1–3.2) and notebook.

| quantity | Table 1 / paper | rebuilt |
|---|---|---|
| MWEAT–Diff, original | 3.6918 (p 0.000) | **3.6918** (p 0.000) |
| cos(PCA semantic, grammatical direction) | 0.389 | **0.3885** with a LinearSVC direction; 0.377 with LDA |
| MWEAT–Diff, Shift Ori | 0.3090 (p 0.113) | **0.3087** (p 0.757) with the SVC direction; 0.365 with LDA |
| nouns in vocabulary (pairs) | notebook: 3,371 pairs | 3,371 m / 4,322 f → 3,371 pairs |

- The original statistic reproduces exactly; Shift Ori's statistic to 3 decimals (the unseeded
  LinearSVC refit explains the 4th).
- **The paper says the grammatical direction comes from LDA; the published numbers match a LinearSVC
  direction** (cos 0.3885 vs 0.389; Shift Ori 0.3087 vs 0.3090). LDA gives 0.377 and 0.365.
- **The p-value doesn't reproduce**: 0.757 vs the table's 0.113. Table 1 gives Shift Ori and Hybrid
  Ori the identical statistic (0.3090) but p 0.113 vs 0.733, so the p-values came from different
  runs of a seeded-but-shared random stream (`random.seed(520)` is set once per cell in the
  notebook, and earlier draws shift it). Either way Shift Ori is non-significant, as reported.
- Shifting along the *un-orthogonalised* PCA direction (not in the paper) drives the statistic to 0.048.
- Not yet rebuilt: the inanimate-noun half of Shift (project the semantic direction out of
  inanimate nouns; doesn't affect Table 1, which scores occupations only) and Hybrid (needs
  hard-debiased English MUSE + realignment).
