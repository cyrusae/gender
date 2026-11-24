# Word-pair test with number: singular and plural occupation forms (pre-registration, paper B)

*Status: **DRAFT 2026-11-24**, for PI review. Nothing has been scored. Builds on the adopted
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
- **Intervals:** 10,000 bootstrap draws (seed 0), occupations and cues resampled independently
  (cues within cue type; gendered pairs keep their two cues together), as the cue study. Q2 is
  paired within each draw.

## Reading rules (fixed before scoring)

- Q1: **present** if the 95% lower bound > 0; **absent** if the upper bound < 0; else inconclusive.
- Q2: **singular-only overstates** if the upper bound of (gender-level − singular) < 0;
  **understates** if the lower bound > 0; else **no clear difference**.
- Q3: descriptive; no verdict.
- Sensitivity (reported, not used for verdicts): (a) sum instead of max of the two forms' RRs;
  (b) the 18 Spanish occupations with all four forms in the first 50k words, target vocabulary 50k
  (paper A's setting).

## Predictions (Claude, before scoring)

Q1 present: the masculine lead is large under every cue type in paper A, and adding plurals gives
each gender a second chance. Q2 no clear difference or a small overstatement: plurals should help
feminine retrieval most when the feminine singular is rare. Q3: masculine plurals rank lower than
masculine singulars under all cues, since the query is built from singular words.

## Open for the PI

1. Spanish primary / French descriptive, with French below a floor of 20?
2. max (primary) vs sum (sensitivity) for combining the two numbers?
3. Is the 200k vocabulary acceptable, given that paper A's numbers are at 50k? (Q2 recomputes
   singular intrusion at 200k, so the comparison is internal.)
