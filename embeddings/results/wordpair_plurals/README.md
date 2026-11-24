# Word-pair test with number (paper B; pre-registered, `docs/wordpair-plurals-prereg.md`)

`scripts/wordpair_plurals.py` (adopted and committed 9abbc0b before any score; two crash fixes before
any output, 7bb1fcc, dfd6843). Original MUSE space, 200k target vocabulary. Spanish 43 occupations
(primary), French 16 (descriptive). Sensitivity at 50k: `../wordpair_plurals_50k/` (ES 18, FR 7).

| | Spanish (primary) | French (descriptive) |
|---|---|---|
| Q1 gender-level intrusion (either number) | 0.284 [0.160, 0.404] **present** | 0.129 [−0.117, 0.366] inconclusive |
| singular intrusion, same run | 0.278 [0.155, 0.400] | 0.124 [−0.123, 0.362] |
| Q2 gender-level − singular | 0.006 [−0.000, 0.014] **no clear difference** | 0.005 [−0.000, 0.013] |
| Q4 plural intrusion (plural cues) | 0.365 [0.244, 0.482] **present** | 0.198 [−0.065, 0.444] inconclusive |
| Q4 plural − singular (descriptive) | 0.086 [−0.055, 0.220] | 0.074 [−0.059, 0.202] |

Q3 (descriptive): under feminine singular cues the masculine plural ranks far below the masculine
singular (RR difference ES −0.50, FR −0.43) and only slightly above the feminine plural (ES 0.026
[0.002, 0.053]; FR −0.003).

Q5 (descriptive), shares of the four forms' retrieval:

| | number error | gender error, masculine cues | gender error, feminine cues |
|---|---|---|---|
| ES singular cues | 13–14% | 17% | **65%** |
| ES plural cues | 6–7% | 11% | **68%** |
| FR singular cues | 9–10% | 23% | **57%** |
| FR plural cues | 2–3% | 12% | **61%** |

Reading: the masculine form wins under feminine cues whether or not plurals are counted, and in the
plural as in the singular. Singular-only scoring does not misstate intrusion (plurals almost never
outrank the singular of the same gender). The space handles number well and masculine agreement
reasonably; the error is concentrated on the feminine side, so intrusion is not general grammatical noise.
Predictions: Q1 present (right), Q2 no clear difference (right), Q4 present (right) and larger than
singular (not shown). Sensitivity: 50k on 18 Spanish occupations gives the same verdicts (Q1 0.317,
Q4 0.371, Q2 0.007 [−0.001, 0.020]); sum instead of max: 0.305 [0.167, 0.438].
