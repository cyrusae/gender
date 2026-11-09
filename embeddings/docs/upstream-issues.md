# Issues found in the upstream code, data and papers

A running log for the embeddings study. Each entry: what we found, how we know, and where the
evidence lives. "Verified" = checked against code output or the paper text; "from code reading"
= not yet confirmed by running anything. Paper-claim checks: `docs/design/embeddings-lit-verification.md`.
Zhou et al. Shift Ori code map, derivation and blind check: `docs/zhou-shift-identity.md`.

**Severity** (judged by effect on the paper's *reported conclusions*, after our checks):
- **High**: invalidates or removes the support for a reported claim.
- **Medium**: changes reported numbers, blocks reproduction, or muddies what a claim is about, but
  the conclusion survives (or hasn't been tested yet).
- **Low**: hygiene or documentation; checked and no effect on results.

## Omrani Sabbaghi & Caliskan 2022 (`external/GG_Disentangling`, commit 725d813)

None of these overturns a conclusion: every one we could test leaves the paper's results standing
(see "Results of checks").

| # | issue | severity | status | evidence |
|---|---|---|---|---|
| O1 | Single-category GG-WEAT scores 1,000 + 1,000 nouns taken from the 3,000 pairs the grammatical-gender direction was **trained** on; a held-out split exists but isn't scored. | Medium (tested: effect generalises; starting gap ~13% larger on training nouns) | verified (code + rerun) | `main.ipynb` cell 48 (`shortened_3000`); `scripts/gg_heldout.py` |
| O2 | Noun lists contain **duplicates**: the de-dup test `word not in L or word_raw not in L` is always true for lowercase words (`word` is capitalised). Spanish masculine: 5,381 entries, 5,213 distinct; feminine 4,837 / 4,688. | Low (tested) | verified | cell 19; counted 2026-11-08 |
| O3 | **48 nouns appear in both gender lists** (e.g. *agua*). | Low (tested) | verified | counted 2026-11-08 |
| O4 | Because of O2–O3 the "held-out" pairs share ~150 words (151 distinct) with the training pairs. | Low (tested) | verified | our safeguard assertion failed on it |
| O5 | Spanish has only ~1,837 pairs after equalising, so `rest` holds ~1,837 pairs, not 3,000; the held-out **accuracy matrix is padded with zero rows** (6,000 rows, ~3,674 real) carrying alternating labels. | Low (tested: 0.506 padded vs 0.510 real rows) | verified (rerun) | cells 23, 40; `scripts/gg_heldout.py` |
| O6 | The paper's text/figures say **15** removal iterations; its tables match iteration **28** of `Results/es.xlsx`; the notebook default for Spanish is **14**. Spanish GG-WEAT: 1.83 → 1.305 (iter 15), 1.235 (iter 28), 1.32 (our 14-iter rerun). | Medium (numbers; conclusions same at 14 and 28) | verified | `results/repro_es/`, lit-verification table |
| O7 | Removal is applied to **every** vocabulary word, including the man/woman attribute words used to score the nouns. | Medium (design choice; makes raw means hard to read, see O8) | verified (code) | cell 40 |
| O8 | After removal both noun groups lean the same way (Spanish: masc −0.24, fem −0.29), so raw means mostly reflect an attribute/baseline shift; the masculine-minus-feminine difference is the interpretable quantity. | Medium (interpretation of reported means) | interpretation | `results/repro_es/README.md` |
| O9 | ValNorm word lists: Polish `"smród,atak"` (missing comma merges two words); German `"das Gefängnis"` (with article) and *faul* for "rotten"; duplicates (*prigione*, *paradiso*, *paradis*, *accident*, *prison*, *glücklich*); French multi-word items (*arc en ciel*, *lever du soleil*) can't be in a word vocabulary. **Words silently dropped** (not in the first 1M vectors): attributes es 7/47, fr 2/50, de 10/50, pl 17/49, it 0/50; rated words de 55/371 (54 exist capitalised: lowercasing drops German nouns), fr 37, it 11, pl 10, es 1. | Medium (validity check runs on lopsided sets; ES fixed version slightly *better*; DE/PL untested) | verified 2026-11-08 | coverage check (see O16) |
| O10 | No requirements file, no seeds (LinearSVC refits, permutation nulls): reruns differ in the 3rd decimal. | Low | verified | `results/repro_es/README.md` |
| O11 | No licence file in the repo. | Low | verified | — |
| O12 | The method is essentially INLP (Ravfogel et al. 2020), which the paper doesn't cite. | Low (citation) | verified (full text) | lit-verification table |
| O13 | **Contradictory labels in training:** 24 Spanish words appear in the 3,000 training pairs under both labels: genuine homographs (*capital, corte, orden, frente, guía, editorial*), adjectives/plurals (*central, final, personal, canales, finales*), mislabels (*madera, motos* f; *cero, aplazamiento* m; *mano* is two-gender in Wiktionary). | Low (tested via minimal fix) | verified 2026-11-08 | count on the loaded lists |
| O14 | Training pairs aren't distinct: 3,000 pairs contain 2,951 distinct feminine and 2,941 distinct masculine words (O2's duplicates). The 1,000 + 1,000 SC-GG-WEAT nouns contain 991 / 990 distinct words, 4 scored under both genders. | Low (tested) | verified 2026-11-08 | same |
| O15 | The noun lists aren't restricted to singular inanimate nouns: plurals and adjectives occur (see O13, O17). | Medium (see O17) | verified (examples) | same |
| O16 | **Unicode bug:** the Spanish and Polish attribute words in `ValNorm.py` are stored in decomposed form (NFD) while fastText uses composed form (NFC), so every accented attribute word fails lookup: Spanish *ánimo, ponzoña, cárcel, cáncer, vómito, agonía, prisión*; Polish *miłość, pokój, tęcza*… The Spanish notebook's own output prints these words as missing. Spanish ValNorm runs on 40 of 47 attributes, mostly losing *unpleasant* words. | Medium (ES tested: 0.807/0.818 → 0.796/0.833, conclusion strengthens; PL untested) | verified 2026-11-08 | NFC vs NFD lookup test; `results/gg_minfix_es/` |
| O17 | ~40% of the noun-list entries aren't singular nouns: 3,021 of 10,056 distinct Spanish words are plurals, plus adjectives/participles; ~9% of the dictionary nouns are animate. Gender labels on real nouns are 99.8% right vs Wiktionary. | Medium (changes what "10,000 inanimate nouns" means; effects survive) | verified 2026-11-08 | `results/label_audit_es/` |
| O18 | Training X is column-normalised (`normalize(X, axis=0)`) with its own column norms, held-out X with *its* own; the in-loop matrices aren't normalised at all. | Low (not tested separately; folded into the reruns) | from code reading | cells 23, 27, 40 |

## Zhou et al. 2019 (`external/Bias_in_Gendered_Languages`, commit 6392702)

The paper's Original value (3.6918 ES) reproduces exactly, but its significance test is degenerate
(Z10: the observed statistic is the maximum of its own permutation null, so p = 0 is guaranteed),
and in this configuration the statistic is the net lean of all occupation words. The mitigation
evidence carries most of the High-severity issues.

| # | issue | severity | status | evidence |
|---|---|---|---|---|
| Z1 | Paper says the grammatical direction is from **LDA**; the published numbers (cos 0.389, Shift Ori 0.3090) match a **LinearSVC** direction (0.3885, 0.3087) together with the `a + b/2` centring in Z3; LDA gives 0.377 and 0.365. Their own LDA attempts **crashed** (cells 75, 86: `ValueError: Internal work array size computation failed: -10`); 0.389 is printed by cell 77 from the LinearSVC fit (cell 73). Confirmed by both blind checks. | Medium (reproduction from the paper alone gives different numbers) | verified | `results/zhou_shift_es/` |
| Z2 | Table 1 gives Shift Ori and Hybrid Ori the identical statistic (0.3090) but p 0.113 vs 0.733; the notebook saved a third, 0.269 (cell 305); our rebuild gets 0.757. Explained by Z10. | High (with Z10) | verified | `results/zhou_shift_es/` |
| Z3 | `doPCA` centres pairs at `a + b/2` (operator precedence), not the midpoint (cell 36, line 6). Kept in our rebuild because it defines the published numbers. | Low (numbers shift slightly: cos 0.374 with the midpoint) | verified (code) | notebook cell 36 |
| Z4 | Repo ships none of the mitigated embeddings (`*_offset_debiased_*`, `hybrid_*`, `debiased_aligned_*`) or the hard-debiased English `.bin`; the notebook can't be rerun; Hybrid had to be rebuilt. | Medium (blocks reproduction; see Z15) | verified | — |
| Z5 | The grammatical-gender noun list includes animate nouns (e.g. *mujer*). | Medium (feeds Z12/Z16) | verified | lit-verification table |
| Z6 | Inanimate-noun bias score b_w is defined (§2.1) but never reported; Table 1 scores occupation pairs only. (Our C1 fills this: the m/f gap survives every mitigation arm.) | High (a stated aim of the mitigation is never evaluated) | verified (full text) | lit-verification table; `docs/c1-prereg.md` |
| Z7 | French word-similarity check cites SemEval-2017 Task 2, which has no French data. | Low (citation; the French similarity numbers' source is unclear) | verified | `docs/feynman/outputs/zhao-gender-bibliography-guide.verification.md` |
| Z8 | No licence file. | Low | verified | — |
| Z9 | Noun lists: 860 of 12,233 distinct words listed under both genders; 4,888 not noun lemmas (adjectives, determiners, abbreviations, numerals, quote-tokens). Disagreements with Wiktionary (56) include 18 feminine *el agua*-type nouns labelled masculine: labels look tagger/agreement-derived. | Medium (the "grammatical gender" direction is partly an agreement direction) | verified 2026-11-08 | `results/label_audit_es/` |
| Z10 | **MWEAT p-values after Shift are tie artefacts.** Whenever a random re-partition gives the two sums opposite signs, the statistic equals \|Σ_all w·a\|, the same for every partition. After Shift Ori **75.7% (ES) / 94.5% (FR) of 1,000 permutations give exactly the observed statistic** (blind check: 90.4% of 20,000 on its reconstruction), so the strict `>` comparison (cell 230) is decided by floating-point noise: 0.757 in our rebuild, 0.012 in a rotated copy, 0.113 / 0.733 / 0.269 for the same 0.3090 in the paper and notebook. **Extended by the Gemini check:** ‖x| − |y‖ ≤ |x + y| for every split, so whenever the observed totals have opposite signs the observed statistic is the maximum of its permutation distribution. That holds **before** mitigation too (ES T_m +2.99 / T_f −6.68; 20,000 permutations: 0 greater, max = observed; FR likewise), so **"Original p = 0.000" is also guaranteed by construction**, and with opposite signs the statistic is the magnitude asymmetry (feminine forms lean further), so the number is meaningful but its null is not; it is offset-sensitive (a per-word tilt c changes it by 2nc; 0.032/word would produce all of 3.69): `docs/zhou-zero-point-prereg.md`. | **High**: no p-value in Table 1, before or after mitigation, is a valid test | verified 2026-11-08; blind check confirmed | `results/zhou_hybrid_es/README.md`; `docs/zhou-shift-identity.md` §3, §5 |
| Z11 | **Hybrid Ori = Shift Ori on Table 1's statistic, mathematically, if the alignment is orthogonal** (rotations preserve every Spanish-internal dot product; Procrustes rebuild: identical to 1e-10), which explains Table 1's identical 0.3090. De-Align must then equal Original (3.6918), but Table 1 says 3.5748: their alignment wasn't a pure rotation (or another setting differed); the paper doesn't say how it aligned. Hybrid's real effect is cross-lingual only (gender-ValNorm Addendum C: it removes the off-axis stereotype signal relative to English words, persisting with a fixed English yardstick). | **High**: Table 1 can't show any monolingual advantage of Hybrid; the De-Align row can't be reproduced | verified 2026-11-08 | `results/zhou_hybrid_es/`; `docs/gvalnorm-prereg.md` Addendum C |
| Z12 | 12 of the 16 Spanish WEAT attribute words (*hombre, niño, padre, abuelo, hijo, niña, tía, hija, esposa, mujer, madre, abuela*) are in Zhou et al.'s grammatical-gender ("inanimate") noun lists (Z5), so **Eq. (2) applied to their list would move the attribute words** used to measure bias. *Inference from the paper (the released code never applies Eq. 2, Z15).* The released code does move most of them through Eq. (4): see Z16. | Medium | inference (Eq. 2); observed for Eq. 4 (Z16) | `docs/c1-prereg.md` results; `docs/zhou-shift-identity.md` §2, §5 |
| Z13 | **French Table 1 doesn't match the released notebook.** Shift Ori FR: Table 1 0.2446 (p 0.147); the notebook's saved output 0.0649 (p 0.019, i.e. significant); our rebuild 0.0650. Original FR: Table 1 2.3437 vs rebuild 2.3958 (that cell has no saved output). | **High**: a reported value doesn't come from the released code, and the code's own run contradicts the reported significance (though Z10 makes both p-values meaningless) | verified 2026-11-08 | `results/zhou_shift_fr/README.md` |
| Z14 | **Shift Ori's MWEAT drop is fixed before the shift (an algebraic identity).** MWEAT per word is w·a (raw dot products, cell 222 lines 15–19; the paper says cosine, under which the identity doesn't hold), a = mean(male attributes) − mean(female attributes). Write a = αu + a⊥, u = d_s/‖d_s‖. Shift moves only u-coordinates, making each pair's sum zero, so the u-parts cancel in T_m + T_f and the m − f difference T_m − T_f is unchanged. Hence after Shift the statistic is \|Σ_all occupation words w·a⊥\| if T_m, T_f have opposite signs (true here: ES +4.99 / −4.68), or the original \|T_m − T_f\| if they have the same sign; either way it is set, given d_s and the attribute words, before anything moves. Observed: ES 0.308748 = 0.308748, FR 0.065019 = 0.065019 (to 1e-15); blind check 0.115519 = 0.115519 on its own reconstruction. cos(a, u) = 0.82 ES / 0.88 FR, so 3.69 → 0.31 mostly measures how much of the yardstick lies along d_s. The shifted and scored words are the same pairs (cell 312 vs cell 234). Also explains Z10. | **High**: the headline mitigation number isn't a measurement of mitigation | verified 2026-11-08; derivation `docs/zhou-shift-identity.md` §3; blind check confirmed (PARTLY: added the same-sign branch and conditions) | `scripts/zhou_construct.py`, `results/zhou_construct.json` |
| Z15 | **Eq. (2) (project d_s out of inanimate nouns) appears in no cell of the released notebook;** the Spanish Shift Ori embedding is loaded from an unreleased file (cell 14, lines 1–2). The only projections of a direction out of word lists (cells 247–248) target the WEAT career/family words. Whether their Shift Ori applied Eq. (2) can't be determined from the release. Blind check confirmed. | Medium (can't verify the method as described; with Z6, the inanimate half is undocumented in code and unevaluated) | verified 2026-11-08 | `docs/zhou-shift-identity.md` §2 |
| Z16 | The notebook also shifts the **definitional pairs** along d_s (cells 279, 292; FR 293/296/297), and 10 of the 16 Spanish attribute words are definitional words (*hija, hijo, hombre, madre, masculino, mujer, niña, niño, padre, tía*), so in the released pipeline **the yardstick moves under Eq. (4)**; never mentioned in the paper. Which embedding produced Table 1 is unknown (Z15). Smaller slips: cell 274 line 4 (aligned and original FR embeddings mixed in one offset); cell 296 line 8 (wrong normaliser); cell 211 (doctora/doctor appended, 59 pairs vs the paper's 58). Found by the blind check, spot-checked by us. | Medium (undocumented step; Z14 holds with the moved yardstick too) | verified 2026-11-08 | `docs/zhou-shift-identity.md` §5 |


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

- **Zhou et al.**: blind check by a fresh Claude Sonnet agent with no access to this repo
  (`~/zhou-blind-check-claude/REPORT.md`): Z10, Z14, Z15 confirmed with narrowings (folded in
  above); Z16 new. Gemini (different family, `~/zhou-blind-check/REPORT.md`): all three claims
  CONFIRMED; extended Z10 to the Original column (verified: `scripts/zhou_perm_max.py`); LDA crash
  (Z1). See `docs/zhou-shift-identity.md` §6.

## Consequences for our design (to pre-register)

- Never use the MWEAT ‖Σ_m| − |Σ_f‖ permutation p-value (Z10); C1's Δ with bootstrap CI and
  label permutation has a continuous null.
- Hybrid differs from Shift only cross-lingually (Z11): in C1 the Hybrid arm is informative only
  with English attribute words.
- Never score with a yardstick the mitigation can move (Z12, Z14, Z16): use attribute words
  outside the edited set, or the d_s-orthogonal yardstick (gender-ValNorm Addendum B).

- Deduplicated noun lists, gold genders from Wiktionary, nouns in both genders dropped;
  singular nouns only; homographs excluded (static vectors have one vector per spelling).
- NFC-normalise every word list and assert full vocabulary coverage (fail, don't silently drop);
  case-sensitive lookup for German.
- Held-out nouns with zero overlap with training, checked by assertion.
- Accuracy on real rows only; report both stopping rules (chance and fixed iteration count).
- Report masculine-minus-feminine differences, not raw group means; consider English attributes
  in aligned space so attribute words carry no grammatical gender.
