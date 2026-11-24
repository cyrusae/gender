# Word-pair test as counts (exploratory, 2026-11-24)

`scripts/wordpair_counts.py` (committed before running, 65ea780). Same queries, frozen v1 cues, clean
occupations (ES 54, FR 22) and scoring as the pre-registered cue study, reported as shares of queries
rather than mean reciprocal rank: the masculine / feminine occupation form is the first word retrieved
(top-1), in the top ten, or ranks above the other form; and neither form is retrieved (top 999). One file
per centring (`midpoint.csv` = as described in the paper, primary; `notebook.csv` = released code).
No intervals; descriptive companion to the MRR results.

Spanish, original space, feminine cues (4,698 queries): masculine form first 51%, feminine first 12%;
masculine ranks above feminine 76%. After Shift Ori / Hybrid Ori (midpoint): 67% / 69%.
