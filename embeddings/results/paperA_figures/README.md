# paperA_figures

Figures F2–F4 for paper A (replication report on Zhou et al. 2019), made by
`scripts/paperA_figures.py` (2026-11-19). Plots of results recorded elsewhere, plus the small
recomputation each figure needs. Draft styling; not final.

- **F2_permutation_null.png / .csv** (upstream Z10): Zhou's permutation test redone with 10,000
  re-partitions. Original: the observed MWEAT is the maximum of its own null (0% exceed), so p = 0 is
  guaranteed. Shift Ori: 75% (ES) / 93% (FR) of re-partitions give exactly the observed value, so the
  reported p-values are decided by rounding.
- **F3_same_choices_two_measures.png, F3_shift_mweat_by_choice.csv** (Z14, Z19, Z33): Shift Ori MWEAT
  under centring (notebook a + b/2 / midpoint) × attribute lists (published / corrected): ES 0.309,
  1.815, 0.037, 1.476; FR 0.065, 0.453, 0.294, 0.213. Beside it, the change in intrusion under Shift for
  the two centrings (from `wordpair_cues/` and `wordpair_cues_midpoint/`): ES −0.109 / −0.103, FR −0.108
  / −0.096. The Table 1 statistic swings by 50× across defensible choices; the retrieval measure doesn't.
- **F4_mitigation_intrusion.png / .csv**: intrusion with feminine cues in the original, Shift Ori and
  Hybrid Ori spaces, ES and FR, 95% intervals (from `wordpair_cues/gaps.csv`).

Note from F3: with the corrected Spanish attribute list (*tío*), Shift Ori's "reduction" is 2.25 → 1.82
(−19%), not 3.69 → 0.31 (−92%).
