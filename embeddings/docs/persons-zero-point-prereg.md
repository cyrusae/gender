# Is Zhou's monolingual asymmetry specific to occupations? (DRAFT pre-registration)

*Status: **ADOPTED 2026-11-11** (PI). Nothing scored before adoption.*

## Why

The zero-point study (`zhou-zero-point-prereg.md`) asks whether Zhou et al.'s feminine occupation
forms lean further from a sex-neutral zero point than the masculine forms. With corrected attribute
words it is inconclusive in Spanish and borderline in French (flips with dropping one occupation,
`results/list_audit/`). The person-noun study showed that the cross-lingual intrusion is specific to
occupations. This asks the same of the monolingual asymmetry: is it a property of occupations, or of
gendered person nouns generally?

## Question

**Q1.** For non-occupation person nouns (*vecino/vecina*), what is the offset-corrected asymmetry A?
**Q2 (primary).** Does A differ between person nouns and occupations?

## Materials (all frozen already)

- Nouns: `data/wp_persons_es_frozen_v1.csv` (sha 75dd47c6…): person 107, occupation 115 (rule-built).
  Identity nouns not used. Zhou's 58 occupations reported alongside for reference (already analysed).
- Zero point: the frozen epicene reference words (`data/zero_ref_es_frozen_v1.csv`, strict tier),
  minus any word that is also a target or attribute word (rule of the original study).
- Attribute words: corrected Spanish WEAT lists (primary since 2026-11-11); English attribute words in
  the aligned space as the second yardstick, as before.
- **Spanish only**: the person list is Spanish (the PI reviewed it; French wasn't built). This can't
  settle the French borderline directly; it tells us whether the asymmetry is an occupation property
  in the language where both groups exist.

## Measures (as `scripts/zhou_zero_point.py`, original space only)

Per-word lean s(w) = mean over male attribute words of w·x − mean over female attribute words of w·y.
For a group of n pairs, T_m, T_f = sums of the masculine / feminine forms' leans; zero point c =
½(mean lean of masculine epicenes + mean lean of feminine epicenes). A = (|T_m − nc| − |T_f − nc|) / n
(**per pair**, so groups of different size are comparable; the original study reported the sum).
A < 0: feminine forms lean further.

95% intervals: 10,000 joint bootstrap draws (seed 0) over target pairs (within group), epicene
reference words (within gender) and attribute words; the reference and attribute draws are shared by
the two groups in each draw, so Q2's difference has the shared parts cancelled.

## Readings, fixed now

- **Q1:** "asymmetry: feminine further" if A's interval is below 0, "masculine further" if above,
  otherwise "inconclusive".
- **Q2:** with D = A(person) − A(occupation): "occupation-specific" if D's interval is above 0 and
  A(occupation) < 0 (occupations show more feminine-further asymmetry than persons); "shared with
  person nouns" if D's interval lies within ± half |A(occupation)| (relative margin, as in the
  generic-masculine study); otherwise "not distinguishable".
- Native attributes primary; English attributes reported with the same rules.
- Sensitivity: both groups without nouns carrying any flag (`fem_also_noun`, `masc_also_noun`,
  `language_name`, `masc_identical_en`, `place_name`).
- Floor: fewer than 30 pairs per group with both forms in the 50k vocabulary → descriptive only.

## Limits

Person nouns and occupations differ in more than occupation-ness (frequency of the feminine form,
abstractness). A difference shows the asymmetry isn't general to person nouns, not why.

## Open for the PI

1. Spanish only is acceptable? (A French person list would need building by rule and couldn't be
   reviewed by you; it could be run as descriptive.)
2. Per-pair A (so group sizes don't matter) instead of the original study's sum: agreed?

## Adoption record (2026-11-11, before any score)

PI accepted all open points: Spanish only (French to revisit), per-pair A, the plural rule, floor 25.
Script `scripts/persons_zero_point.py`.
