# Expanded-cue test with the corrected PCA centring (sensitivity, 2026-11-19; upstream Z3/Z33)

Same script and frozen cues as `results/wordpair_cues/`, run with `ZHOU_CENTRE=midpoint` (pair
midpoint (a + b) / 2 instead of the notebook's a + b / 2 when building the semantic direction for
Shift and Hybrid). Switch and output override committed before the run (28e81a4). Original-space rows
are unaffected by construction and identical.

| | Published centring | Midpoint centring |
|---|---|---|
| ES Shift − Original, intrusion | −0.109 | −0.103 [−0.141, −0.067], reduces |
| ES Hybrid − Original, intrusion | −0.095 | −0.086 [−0.146, −0.032], reduces |
| FR Shift − Original, intrusion | −0.108 | −0.096 [−0.130, −0.064], reduces |
| FR Hybrid − Original, intrusion | −0.062 | −0.047 [−0.100, 0.002], **no clear change** |
| FR Hybrid − Original, agreement gap | −0.045 | −0.037 [−0.083, 0.005], **no clear change** |

**Reading.** The word-pair result doesn't depend on the bug: in Spanish both mitigations still cut
intrusion by about a third (Shift −35%, Hybrid −30%), and every verdict holds. Effects shrink slightly
with the corrected direction everywhere; the one change is French Hybrid, whose already borderline
reduction (interval −0.116 to −0.013 before) now touches zero. Contrast MWEAT (Z33): the same 6°
change in direction moves Shift Ori's Table 1 value tenfold, because MWEAT after a shift is a small
residual; the retrieval measure isn't.
