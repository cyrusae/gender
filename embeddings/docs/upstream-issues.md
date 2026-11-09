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
| O9 | ValNorm word lists: Polish `"smród,atak"` (missing comma merges two words); German `"das Gefängnis"` (with article) and *faul* for "rotten"; duplicates (*prigione*, *paradiso*, *paradis*, *accident*, *prison*, *glücklich*); French multi-word items (*arc en ciel*, *lever du soleil*) can't be in a word vocabulary. **Words silently dropped** (not in the first 1M vectors): attributes es 7/47, fr 2/50, de 10/50, pl 17/49, it 0/50; rated words de 55/371 (54 exist capitalised: lowercasing drops German nouns), fr 37, it 11, pl 10, es 1. | verified 2026-11-08 | coverage check (`/tmp/o9.py` logic, see O16) |
| O16 | **Unicode bug:** the Spanish and Polish attribute words in `ValNorm.py` are stored in decomposed form (NFD: *a* + combining accent) while fastText uses composed form (NFC), so every accented attribute word fails lookup: Spanish *ánimo, ponzoña, cárcel, cáncer, vómito, agonía, prisión*; Polish *miłość, pokój, tęcza*… (36 combining marks in `ValNorm.py`; no other file affected). The Spanish notebook's own output prints these words as missing. ValNorm Spanish therefore runs on 40 of 47 attributes, mostly losing *unpleasant* words. | verified 2026-11-08 | NFC vs NFD lookup test |
| O13 | **Contradictory labels in training:** 24 Spanish words appear in the 3,000 training pairs under both labels. Some are genuine homographs whose gender changes meaning (*capital, corte, orden, frente, guía, editorial*), some are adjectives or plurals (*central, final, personal, canales, finales*), and some are just mislabelled (*madera, mano, motos* are feminine; *cero, aplazamiento* masculine). | verified 2026-11-08 | count on the loaded lists |
| O14 | Training pairs aren't distinct: 3,000 pairs contain 2,951 distinct feminine and 2,941 distinct masculine words (O2's duplicates). The 1,000 + 1,000 SC-GG-WEAT nouns contain 991 / 990 distinct words, 4 scored under both genders. | verified 2026-11-08 | same |
| O15 | The noun lists aren't restricted to singular inanimate nouns: plurals and adjectives occur (see O13). | verified (examples) | same |
| O17 | ~40% of the noun-list entries aren't singular nouns: 3,021 of 10,056 distinct Spanish words are plurals, plus adjectives/participles; ~9% of the dictionary nouns are animate. Gender labels on real nouns are 99.8% right vs Wiktionary. | verified 2026-11-08 | `results/label_audit_es/` |
| O18 | Training X is column-normalised (`normalize(X, axis=0)`) with its own column norms, held-out X with *its* own; the in-loop matrices aren't normalised at all. | from code reading | cells 23, 27, 40 |
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
| Z9 | Noun lists: 860 of 12,233 distinct words listed under both genders; 4,888 not noun lemmas (adjectives, determiners, abbreviations, numerals, quote-tokens). Disagreements with Wiktionary (56) include 18 feminine *el agua*-type nouns labelled masculine: labels look tagger/agreement-derived. | verified 2026-11-08 | `results/label_audit_es/` |
| Z10 | **MWEAT p-values after Shift are tie artefacts.** The statistic is ‖Σ_m| − |Σ_f‖; whenever a random re-partition gives the two sums opposite signs it equals \|Σ_m + Σ_f\|, which is the same for every partition. After Shift Ori, **75.7% of the 1,000 permutations give exactly the observed statistic**, so p is anywhere between 0 and 0.757 depending on floating-point noise. Our Shift rebuild gets 0.757; the same statistic in a rotated copy of the space gets 0.012; Table 1 reports 0.113 (Shift Ori) and 0.733 (Hybrid Ori) for the identical 0.3090. "Non-significant after mitigation" is not a test result. | verified 2026-11-08 | `results/zhou_hybrid_es/README.md` |
| Z11 | **Hybrid Ori = Shift Ori on Table 1's statistic, mathematically, if the alignment is orthogonal** (rotations preserve every Spanish-internal dot product; Procrustes rebuild: identical to 1e-10). That explains Table 1's identical 0.3090. But then De-Align must equal Original (3.6918), while Table 1 says 3.5748: their alignment was not a pure rotation (or another setting differed); the paper doesn't say how it aligned. | verified 2026-11-08 | `results/zhou_hybrid_es/` |
| Z12 | 12 of the 16 Spanish WEAT attribute words (*hombre, niño, padre, abuelo, hijo, niña, tía, hija, esposa, mujer, madre, abuela*) are in Zhou et al.'s grammatical-gender ("inanimate") noun lists (see Z5), so Eq. 2 applied to their list moves the attribute words used to measure bias. | verified 2026-11-08 | `docs/c1-prereg.md` results |
| Z13 | **French Table 1 doesn't match the released notebook.** Shift Ori FR: Table 1 0.2446 (p 0.147); the notebook's saved output 0.0649 (p 0.019, i.e. significant); our rebuild 0.0650. Original FR: Table 1 2.3437 vs rebuild 2.3958 (that cell has no saved output). The notebook also saved a third p-value (0.269) for the Spanish 0.3090. | verified 2026-11-08 | `results/zhou_shift_fr/README.md` |
| Z14 | **Shift Ori's MWEAT drop is an algebraic identity.** MWEAT per word is w·a with a = mean(male attributes) − mean(female attributes); a lies largely along the semantic direction ds (cos 0.82 ES, 0.88 FR; 68%/77% of its squared length). Shift moves only the ds-coordinate, making each pair's ds-coordinates sum to 0, so after Shift the statistic equals MWEAT computed on the *original* space with a's ds-part deleted: ES 0.308748 = 0.308748, FR 0.065019 = 0.065019 (exact to 1e-15). The same 58 (ES) / 23 (FR) pairs are shifted and scored (notebook cells 234, 273–275). 3.69 → 0.31 measures "how much of the attribute axis lies along ds", not a mitigation outcome; it also explains Z10's ties (post-Shift value = \|Σ over all occupation words of w·a⊥\|, the same for every opposite-sign split). | verified 2026-11-08 | `scripts/zhou_construct.py`, `results/zhou_construct.json` |


## Results of checks

- **O1 + O4 (held-out nouns), Spanish, 14 iterations** (`results/gg_heldout_es/scores_iter14.json`;
  1,000 + 1,000 held-out pairs with no word in training, from 1,689 clean pairs): the effect
  **generalises**. Masculine-minus-feminine SC-GG-WEAT gap: training nouns 1.741 → 0.050,
  held-out 1.516 → −0.026; weakening rate 99.8% / 93.5% held-out (99.8% / 96.8% training).
  Held-out nouns start with a ~13% smaller gap. Both groups end negative (masc −0.36, fem −0.34),
  consistent with O8.
- **Same at 28 iterations** (`scores_iter28.json`): gap training 1.741 → 0.056, held-out 1.516 →
  0.007; both groups drift further negative (held-out masc −0.41, fem −0.42). More iterations
  don't change the gap, only the shared shift.
- **O5 (zero-padded accuracy)**: low impact. Final held-out accuracy 0.506 as reported, 0.510 on
  real rows, 0.510 on clean held-out pairs (28 iter: 0.514 / 0.523 / 0.523): all at chance.

- **Ending-exception test**: neither paper's direction is a pure -o/-a spelling direction
  (exceptions classified 100%, n = 2–9). `results/label_audit_es/`.

- **Minimal-fix run, 14 iterations** (`results/gg_minfix_es/`; dedup + drop 48 both-gender words,
  unpadded accuracy, NFC ValNorm; method otherwise unchanged). Lists: masc 5,381 → 5,213, fem
  4,837 → 4,688, 4,640 pairs; held-out 1,640 pairs, none overlapping training. Compared with the
  as-published rerun (14 iter):
  | | as published | minimal fix |
  |---|---|---|
  | SVC accuracy train / held-out, before | 0.992 / 0.985 | 0.997 / 0.991 |
  | held-out accuracy after removal | 0.506 (padded) | 0.519 |
  | GG-WEAT after 14 iter | 1.32 | 1.32 |
  | SC-GG-WEAT gap, scored (training) nouns | 1.741 → 0.050 | 1.750 → 0.034 |
  | SC-GG-WEAT gap, held-out nouns | 1.516 → −0.026 | 1.526 → −0.031 |
  | ValNorm r before / after | 0.807 / 0.818 | **0.796 / 0.833** |
  The bugs don't change the GG-WEAT or SC-GG-WEAT conclusions. ValNorm moves most (now on all 47
  attributes): slightly lower before, higher after, so the "removal preserves meaning" result gets
  a little stronger. 28 iterations: same conclusions (results in `scores_iter28.json`).

## Consequences for our design (to pre-register)

- Never use the MWEAT ‖Σ_m| − |Σ_f‖ permutation p-value (Z10); C1's Δ with bootstrap CI and
  label permutation has a continuous null.
- Hybrid differs from Shift only cross-lingually (Z11): in C1 the Hybrid arm is informative only
  with English attribute words in the debiased-English space.

- Deduplicated noun lists, gold genders from Wiktionary, nouns in both genders dropped;
  singular nouns only; homographs excluded (static vectors have one vector per spelling).
- NFC-normalise every word list and assert full vocabulary coverage (fail, don't silently drop);
  case-sensitive lookup for German.
- Held-out nouns with zero overlap with training, checked by assertion.
- Accuracy on real rows only; report both stopping rules (chance and fixed iteration count).
- Report masculine-minus-feminine differences, not raw group means; consider English attributes
  in aligned space so attribute words carry no grammatical gender.
