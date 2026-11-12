# Internal checks of our own pipeline

Checks of our code's assumptions (not upstream issues). Newest first.

## 2026-11-12: unit-length vectors (raised by an outside reviewer)

MWEAT and our lean computations use raw dot products, which equal cosines only for unit vectors.
`scripts/check_unit_norm.py` (output `results/unit_norm_check.csv`): every MUSE vector in the first
200k words of `wiki.multi.{es,fr,en}` has length 1 within 1e-6 (max deviation 9.9e-7). Holds for the
original spaces; vectors after Zhou's Shift are not renormalised (by Zhou's design; see Z14 and
`zhou-shift-identity.md`). Scripts that load fastText Common Crawl vectors (Omrani arm,
`gvalnorm_en.py`) normalise explicitly. Rerun the script if the vector files are replaced.

## 2026-11-12: standard deviation with n vs n − 1 (raised by an outside reviewer)

Where each convention is used:
- `wordpair_persons.py` `zs()` on numpy arrays: divides by n (numpy default, ddof 0).
- `wordpair_mechanism.py` `zs()` on pandas Series: divides by n − 1 (pandas default, ddof 1);
  `wordpair_cues.py` reuses it on numpy arrays inside `slope()` (ddof 0) for both x and y.
- `gvalnorm_run.py` `z()`: n − 1 explicitly.
- `gvalnorm_en.py` SC-WEAT: n − 1, matching Omrani Sabbaghi & Caliskan's `utils.py` (`statistics.stdev`).

Effect: none on any result. Within every regression, the outcome and every standardised predictor use
the same convention, so the factor √(n/(n−1)) cancels from standardised slopes (both sides scale by
it). The unstandardised indicator in `gvalnorm_run.py` (grammatical gender, 0/1) has its coefficient
scaled by that factor relative to an n convention: under 0.03% at n ≈ 2,000, and the same in every
bootstrap draw, so verdicts can't change. SC-WEAT per word is rescaled by a constant √(15/16) relative
to ddof 0, which leaves its correlation with ratings unchanged. Across scripts the conventions differ,
but no number is compared across scripts in a way the factor would affect. Decision: leave the frozen
scripts as they are (changing them would alter frozen outputs in the last digits for no gain); new
scripts should state the convention.
