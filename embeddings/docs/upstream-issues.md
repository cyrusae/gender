# Issues found in the upstream code, data and papers

A running log for the embeddings study. Each entry: what we found, how we know, and where the
evidence lives. "Verified" = checked against code output or the paper text; "from code reading"
= not yet confirmed by running anything. Paper-claim checks: `docs/design/embeddings-lit-verification.md`.

## Omrani Sabbaghi & Caliskan 2022 (`external/GG_Disentangling`, commit 725d813)

| # | issue | status | evidence |
|---|---|---|---|
| O1 | Single-category GG-WEAT scores 1,000 + 1,000 nouns taken from the 3,000 pairs the grammatical-gender direction was **trained** on; a held-out split exists but isn't scored. | verified (code) | `main.ipynb` cell 48 (`shortened_3000`); check: `scripts/gg_heldout.py` |
| O2 | Noun lists contain **duplicates**: the de-dup test `word not in L or word_raw not in L` is always true for lowercase words (`word` is capitalised). Spanish masculine: 5,381 entries, 5,213 distinct; feminine 4,837 / 4,688. | verified | cell 19; counted 2026-11-08 |
| O3 | **48 nouns appear in both gender lists** (e.g. *agua*). | verified | counted 2026-11-08 |
| O4 | Because of O2–O3 the "held-out" pairs share ~150 words (151 distinct) with the training pairs. | verified | our safeguard assertion failed on it |
| O5 | Spanish has only ~1,837 pairs after equalising, so `rest` holds ~1,837 pairs, not 3,000; the held-out **accuracy matrix is padded with zero rows** (6,000 rows, ~3,674 real) carrying alternating labels. Reported held-out accuracy is partly computed on blank rows. | from code reading; being quantified | cells 23, 40; `scripts/gg_heldout.py` reports both versions |
| O6 | The paper's text/figures say **15** removal iterations; its tables match iteration **28** of `Results/es.xlsx`; the notebook default for Spanish is **14**. Spanish GG-WEAT: 1.83 → 1.305 (iter 15), 1.235 (iter 28), 1.32 (our 14-iter rerun). | verified | `results/repro_es/`, lit-verification table |
| O7 | Removal is applied to **every** vocabulary word, including the man/woman attribute words used to score the nouns. | verified (code) | cell 40 |
| O8 | After removal both noun groups lean the same way (Spanish: masc −0.24, fem −0.29), so raw means mostly reflect an attribute/baseline shift; the masculine-minus-feminine difference is the interpretable quantity. | interpretation | `results/repro_es/README.md` |
| O9 | ValNorm word lists: Polish `"smród,atak"` (missing comma merges two words); German `"das Gefängnis"` (with article), *faul* for "rotten", all lowercased (German fastText keeps capitalised nouns, so most German nouns likely drop out); duplicates (*prigione*, *paradis*). | from code reading | `ValNorm.py` |
| O10 | No requirements file, no seeds (LinearSVC refits, permutation nulls): reruns differ in the 3rd decimal. | verified | `results/repro_es/README.md` |
| O11 | No licence file in the repo. | verified | — |
| O12 | The method is essentially INLP (Ravfogel et al. 2020), which the paper doesn't cite. | verified (full text) | lit-verification table |

## Zhou et al. 2019 (`external/Bias_in_Gendered_Languages`, commit 6392702)

| # | issue | status | evidence |
|---|---|---|---|
| Z1 | Paper says the grammatical direction is from **LDA**; the published numbers (cos 0.389, Shift Ori 0.3090) match a **LinearSVC** direction (0.3885, 0.3087); LDA gives 0.377 and 0.365. | verified | `results/zhou_shift_es/` |
| Z2 | Table 1 gives Shift Ori and Hybrid Ori the identical statistic (0.3090) but p 0.113 vs 0.733; our rebuild gets p 0.757. P-values depend on a shared `random.seed(520)` stream across cells. | verified | `results/zhou_shift_es/` |
| Z3 | `doPCA` centres pairs at `a + b/2` (operator precedence), not the midpoint. Kept in our rebuild because it defines the published numbers. | verified (code) | notebook cell 36 |
| Z4 | Repo ships none of the mitigated embeddings (`*_offset_debiased_*`, `hybrid_*`, `debiased_aligned_*`) or the hard-debiased English `.bin`; Hybrid has to be rebuilt. | verified | — |
| Z5 | The grammatical-gender noun list includes animate nouns (e.g. *mujer*). | verified | lit-verification table |
| Z6 | Inanimate-noun bias score b_w is defined (§2.1) but never reported; Table 1 scores occupation pairs only. | verified (full text) | lit-verification table |
| Z7 | French word-similarity check cites SemEval-2017 Task 2, which has no French data. | verified | `docs/feynman/outputs/zhao-gender-bibliography-guide.verification.md` |
| Z8 | No licence file. | verified | — |

## Consequences for our design (to pre-register)

- Deduplicated noun lists, gold genders from Wiktionary, nouns in both genders dropped.
- Held-out nouns with zero overlap with training, checked by assertion.
- Accuracy on real rows only; report both stopping rules (chance and fixed iteration count).
- Report masculine-minus-feminine differences, not raw group means; consider English attributes
  in aligned space so attribute words carry no grammatical gender.
