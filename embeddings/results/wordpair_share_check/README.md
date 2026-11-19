# Word-pair test: share of queries vs size of the rank gap (exploratory, 2026-11-14)

Why: the authors' correction draft of Zhou et al. (2019) reports, for a feminine-cue word-pair test,
the share of queries in which the masculine occupation form outranks the feminine one (64% ES / 59%
FR, falling "only to 55–62%" after mitigation; their cues and code not yet available). Our
pre-registered measure is intrusion, the mean reciprocal-rank gap (RR_m − RR_f) under a feminine
cue, which Shift and Hybrid cut by about a third. This check computes their kind of measure on our
frozen cue set to see whether the two results conflict.

Script `scripts/wordpair_share_check.py`; output `share_check.csv`. Queries, spaces and scoring as
`scripts/wordpair_cues.py` (87 ES / 86 FR feminine adjective cues, cleaned occupation list, misses
scored 0). Not pre-registered; the script was first run from a scratch copy and then filed (the
rerun from the filed script is byte-identical); not committed before running. No intervals.

| | Original | Shift Ori | Hybrid Ori |
|---|---|---|---|
| ES share where masculine outranks feminine | 75.9% | 66.9% | 67.9% |
| ES intrusion | 0.291 | 0.182 | 0.196 |
| ES agreement gap | 0.390 | 0.327 | 0.348 |
| FR share where masculine outranks feminine | 70.2% | 61.0% | 63.5% |
| FR intrusion | 0.191 | 0.083 | 0.130 |
| FR agreement gap | 0.307 | 0.235 | 0.262 |

Ties (both forms outside the top 999) are about 1% of queries.

## Reading

- The intrusion column reproduces `results/wordpair_cues/gaps.csv` exactly, so the two measures
  are computed on the same queries.
- **No conflict with the authors' numbers.** The share falls by 8–9 points here (theirs: 64% → 55–62%),
  while intrusion falls by 37% (Shift) and 33% (Hybrid) in Spanish. A yes/no share changes only when
  the feminine form overtakes the masculine one; it doesn't register the feminine form moving from
  far behind to just behind. So "mitigation reduces it only slightly" and "mitigation cuts it by
  about a third" describe the same data with different measures.
- Their "MRR gap shrinks by at most 16%" equals our Spanish agreement-gap change under Shift
  (0.390 → 0.327, −16%), so that figure is probably an agreement-type gap, not intrusion. (Our guess.)

## Limits

- French has 22 occupations; its intrusion intervals in `results/wordpair_cues/gaps.csv` include zero
  after mitigation.
- Hybrid Ori is our rebuild (no mitigated embeddings were released, Z4).
- Our original shares (76% / 70%) are higher than the authors' (64% / 59%); cue sets, occupation
  lists and miss handling may all differ. Can't be resolved until their code is public.
