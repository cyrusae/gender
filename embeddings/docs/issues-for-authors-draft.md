# Issues to report to the authors (DRAFT, not sent)

*Drafted by Claude, 2026-11-10, for the PI to send or hold. Nothing here has been sent; contacting
authors is on hold by PI decision until the data is exhausted. Full evidence for each item is in
`upstream-issues.md` under the same code. Wording is meant to be neutral: "the code doesn't match
the described design", not intent. Ordered by severity, then by how easy it is to check.*

Suggested channel: a GitHub issue on each repo (public, lets others see it) or an email to the
corresponding author. One issue per paper, with this list as the body, is probably kindest.

---

## Zhou et al. 2019 (Bias_in_Gendered_Languages)

### High: affects a reported conclusion

1. **Table 1 p-values are not valid tests (Z10, Z2).**
   When the masculine and feminine totals have opposite signs, MWEAT on the observed split equals
   the largest value any permutation can produce, so p = 0 is guaranteed for the Original
   column. After Shift, most permutations give exactly the same value (75.7% ES, 94.5% FR), so
   the p-value depends on floating-point rounding. This is why Shift Ori and Hybrid Ori show the
   same statistic (0.3090) with p 0.113 and 0.733. *Check:* `scripts/zhou_perm_max.py`.
2. **The Shift Ori drop is an algebraic identity (Z14).** With raw dot products (cell 222), the
   statistic after Shift equals the original statistic with the attribute axis's d_s component
   removed, so 3.69 → 0.31 is determined before any vector moves. Under cosine (which the paper
   states) the identity doesn't hold. *Check:* `docs/zhou-shift-identity.md` §3.
3. **Hybrid Ori equals Shift Ori on Table 1 (Z11).** An orthogonal alignment leaves all
   Spanish-internal dot products unchanged, so Table 1 can't show any difference; De-Align's
   3.5748 can't be reproduced (our rebuild 3.6918).
4. **The French Table 1 doesn't match the released notebook (Z13).** Shift Ori FR: table 0.2446
   (p 0.147), notebook output 0.0649 (p 0.019).
5. **The inanimate-noun score b_w (§2.1) is never reported (Z6),** although preserving inanimate
   nouns' grammatical gender is a stated aim; Eq. 2 appears in no released cell (Z15).
6. **Table 2's word-pair test uses no feminine adjective cues (Z17).** `adj_pairs` (cell 345) has
   masculine or invariant forms only, so "F MRR < M MRR" partly measures correct agreement. Also:
   forms outside the top 999 are dropped rather than scored 0, and the Spanish Hybrid cell (351)
   ends in KeyboardInterrupt. *Worth saying:* with feminine cues added, the masculine form still
   wins (ES 0.58 vs 0.37), so the corrected test supports their conclusion.

### Medium: changes numbers or blocks reproduction, conclusions largely stand

7. **Grammatical direction from LinearSVC, not LDA as stated (Z1);** LDA cells crash (75, 86).
8. **`doPCA` centres at `a + b/2` (Z3)** (operator precedence; Low on its own, but needed to match
   cos 0.389).
9. **Table 2 word translation compares 200k vs 50k candidate sets (Z18).** Original's 79.2 and
   Shift's 80.7 match the same unmitigated space at 200k and 50k candidates.
10. **Mitigated embeddings and the debiased English model aren't released (Z4).**
11. **Noun lists mix in animates, non-nouns and both-gender words (Z5, Z9);** 18 *el agua*-type
    feminines are labelled masculine, which suggests tagger-derived labels.
12. **The definitional pairs are also shifted (Z16)** and 10 of 16 Spanish attribute words are
    definitional, so the yardstick moves; cell 274 reads the French feminine offset from the
    unaligned space (pairs left off-centre by −0.033).
13. **12 of 16 attribute words are in the "inanimate" noun list (Z12).**

### Low

14. French similarity cites SemEval-2017 Task 2, which has no French data (Z7). 15. No licence
    (Z8).

---

## Omrani Sabbaghi & Caliskan 2022 (GG_Disentangling)

*No High items: every issue we tested leaves the paper's conclusions standing. Several make
their own results look slightly weaker than they are.*

### Medium

1. **SC-GG-WEAT scores training nouns (O1).** On 1,000 + 1,000 held-out nouns the effect
   generalises (gap 1.52 → −0.03); the starting gap is ~13% larger on training nouns.
2. **ValNorm attribute words in NFD (O16):** every accented Spanish/Polish word fails lookup
   (*cárcel, cáncer*…). Fixed, Spanish improves 0.807/0.818 → 0.796/0.833.
3. **ValNorm list errors (O9):** Polish `"smród,atak"`, German `"das Gefängnis"`, *faul*,
   duplicates; German loses most rated nouns to lowercasing.
4. **Iteration count (O6):** text says 15, tables match iteration 28, notebook default 14.
5. **Erasure applied to the attribute words too (O7, O8),** so raw means mostly reflect a
   baseline shift; the masculine–feminine difference is the interpretable number.
6. **~40% of noun-list entries aren't singular nouns (O15, O17);** labels on real nouns are 99.8%
   right.

### Low

7. Dedup test never fires for lowercase words (O2); 48 nouns in both lists (O3); held-out pairs
   overlap training (O4); padded held-out accuracy matrix (O5: 0.506 padded vs 0.510 real);
   contradictory training labels (O13); non-distinct pairs (O14); separate column normalisation
   for train and test (O18); no seeds or requirements (O10); no licence (O11); method is
   essentially INLP, uncited (O12).
