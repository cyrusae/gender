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

## Results (2026-11-10, run at 52cc9ed; `results/generic_masculine/`)

Distances from the epicene zero point on the semantic axis (95% CI; axis re-estimated per draw):

| | Ms | Fs | Mp | Fp | ρ_m | ρ_f | Δρ | reading 1 | reading 2 |
|---|---|---|---|---|---|---|---|---|---|
| **FR (confirmatory, 32 pairs)** | +0.076 [0.023, 0.101] | −0.249 | +0.041 [−0.002, 0.078] | −0.166 | 0.54 [−0.06, 1.24] | 0.67 [0.54, 0.76] | 0.13 [−0.65, 0.78] | **inconclusive** | **not shown** |
| ES (replication, 43 pairs) | +0.104 [0.050, 0.134] | −0.205 | +0.024 [−0.020, 0.058] | −0.152 | 0.23 [−0.34, 0.49] | 0.74 [0.58, 0.86] | **0.51 [0.14, 1.14]** | supported | **at neutral** |

- French, the confirmatory test, is **inconclusive**. Its masculine singular already sits close
  to neutral (+0.076, lower bound 0.023), so ρ_m divides by a small, uncertain number and its
  interval spans −0.06 to 1.24. The ratio design is badly conditioned when the masculine singular
  lean is small; that is a weakness of the pre-registered measure, recorded here, not fixed after
  the fact. Descriptively the French masculine plural (+0.041) is about half its singular, and
  sits slightly more male than common-gender nouns (G +0.041 [0.001, 0.086], margin 0.038).
- Spanish replicates the exploratory observation: the masculine plural keeps about a quarter of its
  singular lean and is at neutral by the relative-margin rule; the feminine plural keeps three
  quarters. Holds without homographs (descriptive). Spanish was seen before the prereg, so this is
  replication, not confirmation.
- Missing forms were almost all feminine plurals in Spanish (*ingenieras, juezas, psicólogas…*).
