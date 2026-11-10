# Crime nouns as a single-stereotype probe (DRAFT pre-registration)

*Status: DRAFT, 2026-11-10, written by Claude for the PI's review. Nothing scored. Follow-up named
in `docs/zhou-zero-point-prereg.md` (adopted 2026-11-10).*

## Question

Grammatically gender-neutral crime nouns (*el/la delincuente, el/la terrorista, le/la complice*)
carry no grammatical gender cue, but share one strongly male-tilted stereotype. **Do they lean
toward the male attribute words relative to a sex-neutral zero point?** Secondary: is the lean
specific to crime, or do common-gender person nouns in general lean the same way?

## Why it is informative

The zero-point study fixed where "no lean" sits for each yardstick (strict epicene anchor,
`data/zero_ref_{es,fr}_frozen_v1.csv`). A set of grammatically neutral words sharing one
stereotype is the cleanest test that the yardstick picks up social gender at all, without
grammatical gender in the way: a positive control for the gender-ValNorm machinery.

## Materials

- Space, yardsticks, zero point: exactly as the zero-point study (MUSE `wiki.multi.{es,fr}`, raw
  dot products; native attributes primary, English attributes in the aligned space secondary;
  zero point = strict-epicene anchor c, inanimate anchor as check). Original space only.
- **Crime set** (built by rule, `data/crime_candidates_{es,fr}_v0.csv`; PI marks `pi_keep`):
  animate nouns whose gloss names a criminal role, unmarked, Zipf ≥ 2.5, and grammatically
  neutral: common-gender (one form, both sexes) or single-gender epicene with no gendered
  counterpart. Rule-pool noise to review (e.g. *porro* 'joint', *tenorio* 'womanizer',
  *forzado* 'galley slave', *orca/orque* 'killer whale', *compère*, *penalista* 'criminal
  lawyer'; *homicide/régicide/infanticide* are also the crime itself).
- **Gendered crime pairs** (*asesino/asesina*): **not analysed.** Feminine forms are mostly
  absent from the 50k vocabulary (ES 2 of 14 pairs, FR 5 of 10 have both forms), itself a
  descriptive finding worth reporting.
- **Vocabulary:** the 50k cut leaves ES 11 / FR 12 candidates before review. Proposal: allow
  crime nouns up to the first 200k words (same vectors; the zero point is unchanged), primary;
  50k-only as a sensitivity check.

## Measures

- Crime lean L = mean over crime words of s(w) − c (s as in the zero-point study; c = strict
  epicene zero point). Positive = toward the male attribute words.
- Contrast K = mean s(crime) − mean s(non-crime common-gender reference words) (from the frozen
  common set; crime nouns already excluded from it).
- 95% joint bootstrap (10,000, seed 0): crime words, epicene reference words (within gender),
  common words, attribute words.

## Reading (fixed before data)

1. **Male-leaning** if L's 95% CI is above 0; female-leaning if below; otherwise inconclusive.
2. **Crime-specific** if K's CI is above 0 (crime nouns lean more male than other common-gender
   person nouns).
3. Robustness: leave-one-out over crime words (a verdict counts only if no single word flips
   it); inanimate anchor in place of epicene; 50k-only.
4. Minimum: ≥ 8 kept crime nouns per language; otherwise "too few".

## Open decisions for the PI

1. Mark `pi_keep` in the candidate files; add crime nouns the rule missed (e.g. *criminal* is
   in; *sicario, secuestrador* are gendered pairs and excluded).
2. 200k vocabulary for crime nouns (recommended) vs 50k only.
3. Floor 8 (recommended, given the pool size).
