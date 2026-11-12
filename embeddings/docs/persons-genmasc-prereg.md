# Do masculine plurals of ordinary person nouns act generic? (DRAFT pre-registration)

*Status: **ADOPTED 2026-11-11** (PI). Nothing scored before adoption.*

## Why

The generic-masculine study (`generic-masculine-prereg.md`) tested occupations only: Spanish
supported the generic reading of the masculine plural (Δρ 0.51 [0.14, 1.14], masculine plural at
neutral), French was inconclusive. Ordinary person nouns are the textbook case (*los vecinos* for
neighbours of any sex), and the person-noun word-pair study showed occupations behave differently
from other person nouns. So: does the generic-masculine pattern hold for person nouns too, or is it
another occupation-specific effect?

## Questions

**G1 (primary).** For non-occupation person nouns, is Δρ = ρ_f − ρ_m above 0 (the masculine form
moves toward neutral more in the plural than the feminine does)?
**G2.** Is the masculine plural at neutral (relative margin ± half the masculine singular's lean)?
**G3 (descriptive).** The same for the rule-built occupations, and the difference persons −
occupations in Δρ.

## Materials

- Nouns: `data/wp_persons_es_frozen_v1.csv` (sha 75dd47c6…); plurals by rule
  (`data/wp_persons_es_plurals_v0.csv`, `scripts/wp_persons_plurals.py`: Wiktionary plural and
  feminine-plural forms). Pairs need all four forms in the first 50k MUSE words: **person 51 of 107,
  occupation 43 of 115** (counted before any score). Plural list frozen at adoption; the PI may review
  it (51 + 43 rows, four columns).
- Axis, zero point and common-gender reference exactly as the generic-masculine study (definitional
  pairs, made orthogonal to the C1 grammatical-gender direction; epicene strict anchors; common-gender
  person nouns), **with the name pair *maría/juan* removed from the definitional pairs** (Z19 name
  rule; the list audit showed it changes no verdict of the original study).
- Spanish only (as the persons list).

## Measures and readings

As `generic-masculine-prereg.md`: per pair, distances from z of Ms, Fs, Mp, Fp; ρ_m = mean Mp / mean
Ms, ρ_f = mean Fp / mean Fs; Δρ = ρ_f − ρ_m. 10,000 bootstrap draws (seed 0) over pairs, epicene
references (within gender) and definitional pairs (axis re-estimated per draw).
- **G1:** "generic masculine supported" if Δρ's interval is above 0; "against" if below; otherwise
  inconclusive.
- **G2:** "masculine plural at neutral" if the interval of mean(Mp) − mean(common nouns) lies within
  ± ½ mean(Ms) in the same draw; otherwise "not shown".
- **G3:** descriptive (persons − occupations Δρ with a shared-draw interval).
- Sensitivity: without flagged nouns. Floor: fewer than 25 usable person pairs → descriptive.

## Limits

Requiring both plurals in the 50k vocabulary keeps the more frequent nouns (whose feminine plurals are
common enough), which may favour nouns used for mixed groups. A static vector can't separate generic
from male-specific uses; this tests a prediction of the generic account.

## Open for the PI

1. Review the plural list (94 rows) or accept the rule?
2. Floor of 25 person pairs (51 available): fine?

## Adoption record (2026-11-11, before any score)

PI accepted all open points: Spanish only (French to revisit), per-pair A, the plural rule, floor 25.
Plural list frozen as is (rule accepted, PI): `data/wp_persons_es_plurals_v0.csv`, sha256 b302a102f40b7c00…. Script `scripts/persons_genmasc.py`.
