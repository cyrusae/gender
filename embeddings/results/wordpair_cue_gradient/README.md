# Masculine lead by cue type, before and after mitigation (exploratory, 2026-11-23)

`scripts/wordpair_cue_gradient.py` (committed before running, 7b59c7c). Same queries, cues (frozen v1:
ES 87 gendered pairs + 67 invariant, FR 86 + 60), clean occupations (54 / 22) and scoring as the
pre-registered cue study. Per cue type: gap = mean(RR masc − RR fem); share = RR masc / (RR masc + RR fem).
Bootstrap 10,000 over pairs, invariant cues and occupations; changes paired. Invariant and feminine
gaps reproduce the cue study's invariant_gap and intrusion exactly.

Original space (both centrings):

| | cue | form that agrees | gap [95% CI] | masc. share |
|---|---|---|---|---|
| ES | masculine | masculine only | 0.585 [0.494, 0.671] | 84% |
| ES | invariant | either | 0.484 [0.392, 0.573] | 76% |
| ES | feminine | feminine only | 0.291 [0.186, 0.396] | 65% |
| FR | masculine | masculine only | 0.499 [0.319, 0.658] | 80% |
| FR | invariant | either | 0.399 [0.222, 0.555] | 73% |
| FR | feminine | feminine only | 0.191 [0.002, 0.370] | 61% |

Reading: the masculine form leads under every cue; a feminine cue removes only about half of the lead an
invariant cue leaves (ES 0.48 → 0.29). Shift lowers the lead under all three cue types (it is not specific
to feminine cues); Hybrid changes it mainly under feminine cues. Mitigated values: `summary_<centring>.csv`;
figure `cue_gradient_<centring>.png` (+ single panels).
