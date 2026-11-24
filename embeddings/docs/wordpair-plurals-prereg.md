# Word-pair test with number: singular and plural occupation forms (pre-registration, paper B)

*Status: **ADOPTED 2026-11-24** (adoption record at the end). Nothing has been scored. Builds on the adopted
expanded-cue study (`wordpair-cues-prereg.md`); everything not stated here is as there.*

## Why

The corrected word-pair test (paper A) scores only the two singular occupation forms. A query
such as *thoughtful : worker = réfléchie : ?* can return the feminine **plural** *ouvrières* above
both singulars (`results/wordpair_examples/`), and singular-only scoring counts that as a miss,
although the retrieved word has the right gender. Whether this inflates or deflates intrusion is
unknown. Number also matters for the generic masculine: Spanish and French masculine plurals
are the default for mixed groups (*los médicos*), so a masculine plural retrieved under a feminine
cue may reflect that reading rather than a failure of agreement.

## Questions

1. **Q1 (primary).** Counting either number, is there intrusion? *Gender-level intrusion* = mean
   over feminine cues of (RR_M − RR_F), where RR_M = max(RR masc sg, RR masc pl) and RR_F =
   max(RR fem sg, RR fem pl).
2. **Q2.** Does singular-only scoring over- or understate intrusion? Paired difference:
   gender-level intrusion − singular intrusion (the paper A measure, computed in the same run).
3. **Q3 (descriptive).** Where does retrieval go? For each cue type (masculine, invariant,
   feminine), the mean RR of each of the four forms, and under feminine cues RR masc pl − RR fem pl
   (does the masculine plural, the generic form, intrude more than the masculine singular?).
4. **Q4 (plural cues; added 2026-11-24 at the PI's suggestion, before scoring).** The same test with
   everything plural: *gorgeous : developers = hermosas : ?*. Under a feminine plural cue only the
   feminine plural agrees; under a masculine plural cue the masculine plural agrees and is also the
   generic form for a mixed group. *Plural intrusion* = mean over feminine plural cues of
   (RR masc pl − RR fem pl). Reported alongside, paired and descriptive: plural intrusion − singular
   intrusion (does the generic reading make intrusion stronger in the plural?), and the four forms'
   RRs under each plural cue type.

5. **Q5 (grammatical competence, descriptive; added 2026-11-24 at the PI's request, before scoring).**
   How often does the space get agreement wrong in general, so that intrusion can be read against
   a baseline for "the space is just bad at grammar"? From the four forms' RRs, as shares of the
   total RR mass of the four forms (ratio of means, as in paper A's masculine share):
   - *number error*: under singular cues (Q1–Q3), the plural forms' share; under plural cues (Q4),
     the singular forms' share;
   - *gender error*: under masculine cues, the feminine forms' share (both numbers); under feminine
     cues, the masculine forms' share (both numbers; the share version of intrusion).
   Reported per cue type and number, with intervals, no verdict. The comparison of interest:
   if the gender error under feminine cues is much larger than both the number error and the gender
   error under masculine cues, intrusion is specific to the feminine side rather than general
   grammatical noise.

## Design

- **Space:** original MUSE space only. The mitigated spaces are 50k-word rebuilds, and feminine
  plurals are mostly outside the first 50k words (Spanish 36 of 54, French 15 of 22; masculine
  plurals 2 and 0). Mitigation is therefore out of scope here.
- **Target vocabulary:** first 200k words (paper A used 50k). Singular intrusion is recomputed at
  200k so that Q2 compares like with like; the paper A value is not reused.
- **Occupations:** clean list with all four forms in the 200k vocabulary. Plural forms from the
  PI-reviewed frozen list `data/gm_pairs_frozen_v1.csv` (Wiktionary, by rule). Spanish 43 of 54
  (primary); French 16 of 22, **descriptive only** (below a floor of 20).
- **Cues:** frozen v1 (`data/wp_cues_frozen_v1.csv`, primary subset): ES 87 gendered pairs + 67
  invariant; FR 86 + 60. Cues stay singular.
- **Query and ranks:** as the cue study (EN[occupation] − EN[cue] + T[cue form]; cue form removed;
  rank 0 outside the top 999), with the four occupation forms looked up in the same ranked list.
- **Plural cues (Q4):** plural forms of the frozen cues and English plurals of the occupation glosses,
  built by rule from Wiktionary (`scripts/wp_plural_cues_build.py` → `data/wp_plural_cues_v0.csv`,
  `data/wp_plural_occ_en_v0.csv`): every Spanish cue has its plural forms in the 200k vocabulary
  (87 + 67); French 86 + 58 (two invariant plurals missing, those cues dropped from Q4); every English
  gloss has a plural in the first 200k English words. Query EN[occupation pl] − EN[cue] +
  T[cue plural form] (English adjectives do not inflect, so the English cue is unchanged); the cue's
  plural form is removed from the ranking. Same occupations as Q1.
- **Intervals:** 10,000 bootstrap draws (seed 0), occupations and cues resampled independently
  (cues within cue type; gendered pairs keep their two cues together), as the cue study. Q2 is
  paired within each draw.

## Reading rules (fixed before scoring)

- Q1: **present** if the 95% lower bound > 0; **absent** if the upper bound < 0; else inconclusive.
- Q2: **singular-only overstates** if the upper bound of (gender-level − singular) < 0;
  **understates** if the lower bound > 0; else **no clear difference**.
- Q3: descriptive; no verdict.
- Q4: plural intrusion **present / absent / inconclusive** as Q1; the paired plural − singular
  difference is descriptive.
- Sensitivity (reported, not used for verdicts): (a) sum instead of max of the two forms' RRs;
  (b) the 18 Spanish occupations with all four forms in the first 50k words, target vocabulary 50k
  (paper A's setting).

## Predictions (Claude, before scoring)

Q1 present: the masculine lead is large under every cue type in paper A, and adding plurals gives
each gender a second chance. Q2 no clear difference or a small overstatement: plurals should help
feminine retrieval most when the feminine singular is rare. Q3: masculine plurals rank lower than
masculine singulars under all cues, since the query is built from singular words. Q4: plural
intrusion present and larger than singular intrusion, because the masculine plural is also the
generic form.

## Open for the PI

1. Spanish primary / French descriptive, with French below a floor of 20?
2. max (primary) vs sum (sensitivity) for combining the two numbers?
3. Is the 200k vocabulary acceptable, given that paper A's numbers are at 50k? (Q2 recomputes
   singular intrusion at 200k, so the comparison is internal.)

## Adoption record (2026-11-24)

- PI accepted the three open points as proposed (Spanish primary, French descriptive; max primary,
  sum sensitivity; 200k vocabulary), suggested the plural-cue arm (Q4) and requested the
  grammatical-competence measures (Q5); adopted with Q4 and Q5 as written above.
- Input lists: `data/gm_pairs_frozen_v1.csv` (sha256 1dfc6c399d9f9746…), `data/wp_plural_cues_v0.csv`
  (da03bffa5009ef4e…), `data/wp_plural_occ_en_v0.csv` (60e4100b5b4f4c24…), cues `data/wp_cues_frozen_v1.csv`.
- Scoring script `scripts/wordpair_plurals.py`, committed with this record before any score. Its
  only run before adoption was the pre-stated check: singular intrusion at 50k on the full clean
  list reproduces the cue study exactly (ES 0.2912, FR 0.1914).
- Sensitivity (b) is run as `PLURALS_VOCAB=50000` (occupations with all four forms in the first 50k
  words); sensitivity (a), the sum version, is reported in the same output as `S_sum_intrusion`.
