# Crime nouns as a single-stereotype probe (DRAFT pre-registration)

*Status: **ADOPTED by the PI 2026-11-10**. Nothing scored at adoption. Follow-up named
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
  lawyer'; act-dominant words go to the act set below).
- **Act-dominant set (sensitivity, PI 2026-11-10):** words whose main sense is the crime itself,
  which is grammatically gendered (FR *homicide, infanticide, régicide, fratricide*: masculine as
  the act, also the perpetrator; ES *infanticidio*: the act only). Scored separately, never in
  the crime set. Reported: L_act and the difference L_act − L. If act words lean more male than the
  person crime nouns, that is read as the grammatical (masculine) cue entering through the act
  sense, not as stereotype. Marked `act` in the `kind` column.
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


## Adoption record (2026-11-10, before any score)

PI decisions: 200k vocabulary for crime nouns (primary), 50k-only as a check; floor 8; *coupable*
dropped (mostly the adjective 'guilty'); *capo*, *gangster* kept, with a sensitivity check
without them (sex-coded by convention); act-dominant words kept as their own test set;
*fratricida* added (common gender, the person; MUSE rank 100,335).

Frozen lists (`data/crime_{es,fr}_frozen_v1.csv`):
- ES sha256 4c0ec0185879fd167eae0eb30d57675ae423ed5d5d2f7db9d3f61cdd8869b67a: crime 12 (*criminal, terrorista, delincuente, pirata, capo,
  narcotraficante, homicida, asaltante, contrabandista, reincidente, forajido, fratricida*); act 1
  (*infanticidio*).
- FR sha256 81cada96b4e0a344f56ba829d36b6779e98afe758558fec90729300f2b9081d5: crime 12 (*terroriste, pirate, escroc, bandit, gangster, récidiviste,
  incendiaire, faussaire, tortionnaire, pickpocket, malfrat, multirécidiviste*); act 4
  (*homicide, régicide, fratricide, infanticide*).
- Only 7 crime nouns per language are inside 50k, so the 50k-only check is "too few" by the floor;
  noted now, before scoring.
- Zero point: identical to the zero-point study (reference words restricted to the first 50k).

## Results (2026-11-10; script 48f47a6; `results/crime_probe_{es,fr}/`)

L = crime lean from the epicene zero point (positive: toward male attributes); raw cosine-free
dot-product units; for scale, the zero point itself is ≈ −0.01.

| | L, Spanish attr. | L, English attr. | K (vs other common nouns) |
|---|---|---|---|
| **Spanish** | **+0.040 [0.008, 0.074] male-leaning** | **+0.044 [0.013, 0.076] male-leaning** | +0.022 / +0.034, inconclusive |
| **French** | +0.030 [−0.004, 0.068] inconclusive | +0.038 [−0.002, 0.078] inconclusive* | +0.019 / +0.034, inconclusive |

\* French English-yardstick verdict flips to "male-leaning" when *pirate* or *terroriste* is
dropped (LOO), i.e. it sits on the boundary.

- **Spanish: crime nouns lean male** with both yardsticks; robust to leave-one-out and to dropping
  *capo*. With the inanimate anchor: male-leaning (English), inconclusive (Spanish attr.).
- **French: same direction and size, CIs just touch 0** (inconclusive); without *gangster* the same.
- **Crime-specific (K): not shown** in either language. Crime nouns lean more male than the other
  common-gender person nouns by +0.02–0.03, but the CI includes 0: with 12 words, the probe can't
  separate "crime" from "person nouns that refer to people in public roles".
- **Act set:** no sign of a masculine cue entering through the act sense. Act words lean *less*
  male than the crime person nouns (ES, English attr.: D = −0.093 [−0.160, −0.023], one word,
  descriptive; FR D −0.009 / −0.029, inconclusive).
- 50k-only check: too few (7 < 8 per language), as noted before scoring.
- Reading: the yardstick does pick up a male lean on grammatically neutral words with a male
  stereotype (positive control passes in Spanish, borderline in French); the size (~0.04) is
  comparable to C1's per-noun leans and to the zero-point offset.
