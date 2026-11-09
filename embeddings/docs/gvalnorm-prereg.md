# Gender ValNorm on Spanish inanimate nouns (pre-registration)

*Status: **ADOPTED by the PI, 2026-11-08** (decisions below). Written by Claude. The English pilot
(`results/gvalnorm_en/`) is a validity check and is done (r = 0.584 vs Lewis & Lupyan's 0.59).
C1's per-noun scores exist (`results/c1_es/c1_per_noun.csv`), but **no score has been compared
with Glasgow ratings**; only the coverage count below was computed.*

## Question

For Spanish inanimate nouns, does an embedding's gender lean follow (a) the noun's **grammatical**
gender, (b) how masculine or feminine **people rate the thing** (Glasgow GEND of its English
translation), or both? And does Zhou et al.'s mitigation remove the grammatical part, the
human-rated part, or neither? Prior art (`docs/gender-valnorm-prior-art.md`): the English
correlation is published (Lewis & Lupyan 2020; Valentini et al. 2023); no gendered-language or
inanimate-noun version was found.

## Materials

- **Nouns:** the frozen C1 list (`data/c1_nouns_es_v1.csv`, sha256 2c58b883…) restricted to nouns
  whose `concept_en` (Wiktionary English gloss, lowercased) is a Glasgow word (homonym senses
  averaged, as in the pilot). Coverage: **547 nouns in the 50k vocabulary (362 m, 185 f; primary)**,
  1,139 in 200k (sensitivity).
- **Human rating:** Glasgow GEND (1 feminine – 7 masculine), z-scored within the analysed set.
  Limitation stated up front: ratings are of the English word, from British raters; no Spanish
  word-gender norms exist (prior-art check).
- **Embedding scores:** s(w) = mean cos(w, male words) − mean cos(w, female words), exactly C1's
  per-noun score, for the same arms (original; Shift Ori as implemented / as specified; Hybrid Ori
  as implemented / as specified) and attribute sets (Spanish WEAT words; English definitional words
  in the aligned space).
- **English anchor:** the same concepts' English words scored in English MUSE with the pilot's
  attributes: how strongly the human rating shows up when no grammatical gender is involved.

## Model and measures

Per arm × attribute set, OLS on standardised variables:

  s_z = β_G · masc + β_H · GEND_z + β_F · zipf_z + ε  (masc = 1 if grammatically masculine)

- **β_H (primary):** does the embedding track human ratings of the thing beyond grammatical
  gender? 95% bootstrap CI over nouns (2,000 resamples, seed 0).
- **β_G:** the grammatical-gender part (C1's gap, now adjusted for GEND and frequency).
- Descriptive: Pearson r(s, GEND) overall and within each grammatical gender; the English anchor's
  r on the same concepts.

## Reading (fixed before data)

- **Tracks human ratings** if β_H's CI excludes 0 and β_H > 0 in the original space with English
  attributes (the attribute set without grammatical gender). Same test reported for Spanish
  attributes.
- **Mitigation removes the human-rated part** if β_H's CI includes 0 after mitigation and
  |β_H| < 0.25 × original; **survives** if the CI excludes 0 and β_H ≥ 0.5 × original;
  otherwise **reduced**. Same rule, separately, for β_G.
- If the English anchor's r on these concepts is itself < 0.2 (the human signal isn't in the
  English embeddings for this noun set), report "no valid test" for β_H.

## Arm added at adoption: Omrani Sabbaghi & Caliskan's erasure

- Space: fastText Common Crawl `cc.es.300.vec` (first 1M words, as their notebook), their
  iterative LinearSVC erasure with the **bugs-fixed** inputs (`scripts/gg_heldout.py --minfix`:
  deduplicated noun lists, two-gender words dropped), applied to every word as in their method.
- Stopping points: **14** iterations (their notebook) and **28** (their tables), both reported
  (upstream O6).
- Attributes: their (= Zhou's) Spanish 8 + 8 words, which are erased too, as in their method. CC
  spaces aren't aligned to English, so no English-attribute version and no English anchor here.
- Nouns: the same Glasgow-matched C1 nouns, restricted to the CC vocabulary; any C1 noun in their
  erasure training pairs is dropped from this arm (counted and reported).
- Same model, measures and reading rule; "original" for this arm = CC before erasure.

## PI decisions (2026-11-08)

1. 50k primary, 200k sensitivity, as C1. (50k = the slice of the MUSE file Zhou et al. load;
   200k = the whole MUSE file. Not a ratio, the two natural boundaries of these files.)
2. Bring in Omrani Sabbaghi & Caliskan's erasure (section above).
3. French: build the French Wiktionary lexicon first; French C1 and French gender ValNorm follow
   the same designs, frozen as addenda before they run.
