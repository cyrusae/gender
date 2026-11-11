# Generic masculine and Zhou's skew (DRAFT pre-registration)

*Status: **ADOPTED 2026-11-10** (PI checked the plural list; adoption record at the end). **Spanish has already been
looked at** (exploratory check, 2026-11-10, 46 pairs, numbers below), so Spanish is reported as a
replication of an exploratory observation; **French is the confirmatory test** (not looked at).*

## Question

Zhou et al.'s Figure 1 skew: feminine occupation forms sit about twice as far from neutral on
the semantic gender axis as masculine forms (ES, our redraw: +0.10 vs −0.21 from the epicene zero
point; 45/58 pairs). **Is this because the masculine form doubles as the generic (sex-unspecified)
form?** If so, the masculine plural, the most generic form (*los médicos* for mixed groups),
should move toward neutral far more than the feminine plural does.

Exploratory Spanish numbers that motivated this (semantic PCA axis, 46 pairs): singular m +0.122,
f −0.242; plural m +0.019, f −0.142; common-gender person nouns +0.024.

## Materials

- MUSE `wiki.multi.{es,fr}`, first 200k words; unit vectors; raw projection onto the semantic
  axis d_s (true-midpoint PCA of the 10 definitional pairs, grammatical direction from the C1
  Wiktionary nouns removed, as `scripts/zhou_figures.py` panel B).
- Zero point z: strict-epicene anchor, frozen list `data/zero_ref_{lang}_frozen_v1.csv`.
- Pairs: Zhou's occupation pairs, cleaned list primary (ES 54, FR 22; `data/occupation_mapping_v1.csv`
  `clean_keep`), as published secondary. Plurals by rule (ES: +s after a vowel, -z → -ces, +es after
  another consonant, dropping a written accent on a final stressed syllable: capitán → capitanes; FR: +s, -al → -aux, -eau → -eaux, words ending in s/x/z unchanged), kept only if
  all four forms are in the 200k vocabulary and the plural is not identical to the singular.
- Floor: ≥ 15 pairs per language with all four forms (FR may fail with 23; then "too few").

## French extra pairs (PI 2026-11-10)

Added by rule to secure the French floor (`data/gm_extra_fr_candidates_v0.csv`): masculine person
nouns with a feminine counterpart whose English gloss names a US job title and whose four forms are
in the 200k vocabulary; 32 found, 16 kept after review (*travailleur* dropped by the PI as a status,
not an occupation). French usable pairs: 16 Zhou + 16 extra = 32.

## Coverage (counted 2026-11-10, descriptive, before any score)

All four forms in the first 200k MUSE words: **ES 43 of 54** clean pairs (missing feminine plurals:
*ingenieras, entrenadoras, locutoras, barberas, capitanas, coreógrafas, consejeras, juezas,
psicólogas, herreras, redactoras*); **FR 16 of 22** (missing: *administrateur, professeur, avocat,
dessinateur, éditeur, mathématicien* pairs). French clears the floor of 15 by one pair. That the
missing forms are almost all feminine plurals is itself descriptive evidence that feminine
plurals are rare in Wikipedia text.

## Measures

For each pair, distances from the zero point: Ms, Fs (singular), Mp, Fp (plural).
- **Shrink ratios** (pooled): ρ_m = mean(Mp) / mean(Ms), ρ_f = mean(Fp) / mean(Fs).
- **Primary statistic: Δρ = ρ_f − ρ_m** (> 0: the masculine form moves toward neutral more in the
  plural than the feminine form does).
- Secondary: is the masculine plural at neutral? mean(Mp) vs the common-gender person nouns' mean
  distance from z (difference and CI).
- 95% bootstrap (10,000, seed 0) over pairs, epicene reference words (within gender) and the 10
  definitional pairs (re-estimating the axis in every draw, so the axis choice is in the CI).

## Reading (fixed before data)

1. **Generic-masculine supported** if Δρ's CI is above 0.
2. **Masculine plural ≈ generic** if the CI of mean(Mp) − mean(common nouns) lies entirely within
   ± half the masculine singular's lean, mean(Ms), computed in the same bootstrap draw (a relative
   margin, PI 2026-11-10; replaces the earlier fixed ±0.05). Otherwise "not shown".
3. Otherwise inconclusive. A Δρ CI below 0 counts against the account.
4. French is the confirmatory language; Spanish is replication of the exploratory observation
   and is reported as such whatever it shows.

## Known limits

- A static vector can't separate generic from male-specific uses of a form; this tests a
  prediction of the generic account, not the account itself.
- Plurals also differ from singulars in other ways (frequency, contexts); comparing the two
  genders' shrink ratios, not raw plural values, is meant to cancel what they share.
- Plural forms are built by rule; the PI may review the generated list before freezing.


## Adoption record (2026-11-10, before any score)

- Pairs frozen: `data/gm_pairs_frozen_v1.csv`, sha256
  1dfc6c399d9f9746eced15a4ce11061fade383964c5f19b4d07b4b99b4d4b3cd (ES 54 Zhou clean; FR 22 Zhou
  clean + 16 extra). Plural spellings checked by the PI.
- Margin: relative (± half the masculine singular's lean in the same draw).
- The "as published" secondary list is not run for this test: it differs from the clean list only by
  non-occupations (*tory, commissioned, recorder, reader*), to which the generic-masculine
  prediction doesn't apply. A no-homograph subset (H = 0) is reported as descriptive.
- Script: `scripts/generic_masculine.py`.
