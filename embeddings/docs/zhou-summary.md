# Zhou et al. (2019): what holds, what doesn't, and what we added

*Draft for the write-up, 2026-11-10, written by Claude. Every number points to a file in this
repo; issue codes (Z1–Z18) are entries in `upstream-issues.md`. Spanish unless marked.*

**Paper:** Pei Zhou, Weijia Shi, Jieyu Zhao, Kuan-Hao Huang, Muhao Chen, Ryan Cotterell, Kai-Wei Chang, "Examining Gender Bias in Languages with
Grammatical Gender", EMNLP 2019 (arXiv 1909.02224). Code: `Bias_in_Gendered_Languages`, notebook
`bias_emnlp19.ipynb` at 6392702.

## One-paragraph version

Zhou et al. claim that in Spanish and French word embeddings, feminine occupation forms
(*médica*) lean further toward female words than masculine forms (*médico*) lean toward male
words, and propose mitigations (Shift, Hybrid) that remove this asymmetry while keeping
inanimate nouns' grammatical gender. **The bias is real.** With a fair zero point it holds in
French and is inconclusive in Spanish; in the cross-lingual test it is stronger than the
paper's own test could show. **The quantitative evidence for it and for the mitigations mostly
does not hold.** Every Table 1 p-value is guaranteed by construction; the headline Shift result
(3.69 → 0.31) is an algebraic identity fixed before anything moves; Hybrid's Table 1 number is
Shift's by rotation invariance; the Table 2 bias test never uses a feminine cue; and the stated
aim for inanimate nouns is never evaluated. What survives is the measurement's direction, the
inanimate-noun neutrality, the English–Spanish correlation, and Hybrid's real but cross-lingual
effect, which comes from Bolukbasi's English debiasing.

## Claim by claim

| Zhou's claim | Verdict | Evidence |
|---|---|---|
| Semantic and grammatical directions, cos 0.389 (§2.1) | Reproduces, but with a linear SVM (paper says LDA, whose cells crashed) and a centring slip `a + b/2` | Z1, Z3 |
| Inanimate nouns sit near zero on the semantic axis (Fig. 1) | Holds: masc −0.011, fem −0.012 on 2,000 Wiktionary-labelled nouns | `results/zhou_figures/` |
| Feminine occupation forms lean further (Fig. 1, MWEAT 3.69) | Number reproduces exactly. Against an epicene zero point with attribute words resampled: FR −1.91 [−3.42, −0.30], holds; ES −1.99 [−5.63, 1.34], inconclusive | `zhou-zero-point-prereg.md` |
| p = 0.000 (Table 1) | Guaranteed: with opposite-sign totals the observed statistic is the permutation maximum. After Shift, most permutations tie, so the p-values reflect rounding | Z10, Z2 |
| Shift Ori cuts MWEAT 3.69 → 0.31 | Identity: the post-shift value equals the original with the attribute axis's d_s component deleted (0.308748 = 0.308748) | Z14, `zhou-shift-identity.md` |
| Shift preserves inanimate nouns' grammatical gender (aim) | Never evaluated in the paper; Eq. 2 is in no released cell. Our C1: grammatical gender survives every arm | Z6, Z15, `c1-prereg.md` |
| Hybrid Ori 0.3090 (Table 1) | Same as Shift by construction (rotation preserves Spanish dot products). De-Align 3.5748 can't be reproduced (rebuild 3.6918) | Z11 |
| French Table 1 | Doesn't match the released notebook (Shift FR: table 0.2446 p 0.147; notebook 0.0649 p 0.019) | Z13 |
| Word-pair translation shows bias; Hybrid reduces it most (Table 2) | Test as built has only masculine/invariant cues (Z17). Original and Shift reproduce; Hybrid's gain doesn't. With a pre-registered list of 87 feminine adjective cues, the masculine form is still retrieved better (intrusion 0.29 [0.18, 0.39]), so the bias is real and robust. Shift and Hybrid cut it by about a third but don't close it. Which adjective is used, including its own gender stereotype, makes no difference | Z17, `wordpair-cues-prereg.md`, `results/wordpair_cues/` |
| Mitigation doesn't hurt word translation (Table 2) | Holds. The apparent gain (79.2 → 80.7) is from scoring mitigated spaces against 50k rather than 200k candidates | Z18 |
| English and Spanish occupation bias correlate, Spearman 0.45 (§4.4) | Reproduces: 0.450, p 0.0006 (clean list 0.42) | `results/zhou_sec43_44/` |

## What we found beyond the paper

- **Why the asymmetry exists (monolingual).** The masculine form doubles as the generic form. In
  Spanish the masculine plural keeps about a quarter of its singular lean and sits at neutral;
  the feminine plural keeps three quarters (pre-registered replication; French confirmatory test
  inconclusive because its masculine singular is already near neutral).
  `generic-masculine-prereg.md`
- **Why English "doctor" lands on *médico* (cross-lingual).** Not the bilingual dictionary
  (D null twice; removing occupations from the alignment changes nothing). Occupational stereotype
  is suggestive, not established: US labor share β 0.30 in both the mechanism study (CI above 0)
  and the expanded-cue replication (CI just touching 0); female-dominated jobs showed no intrusion
  with Zhou's cues. Feminine forms that also name a field (*química* 'chemistry') add to
  it. `wordpair-mechanism-prereg.md`
- **What Hybrid really does.** Measured against English gender words, it removes stereotype
  signal that lies off Zhou's axis, even with the original English yardstick. That is
  Bolukbasi's English debiasing carried over by alignment, not anything Table 1 can show.
  `gvalnorm-prereg.md` addenda B–C
- **A positive control.** Grammatically neutral crime nouns (*delincuente*) lean male in Spanish
  (+0.040 [0.008, 0.074]); French inconclusive; not shown to be crime-specific.
  `crime-noun-probe-prereg.md`

## Limits of our checks

- Hybrid and De-Align are rebuilt (orthogonal Procrustes onto hard-debiased English), because no
  mitigated embeddings were released (Z4). Results on those spaces are approximate.
- Several of our analyses were exploratory (figure redraw, word-pair rerun, §4.3–4.4); the
  zero-point, generic-masculine, crime, word-pair mechanism and expanded-cue studies were
  pre-registered.
- French judgments rely on English glosses and Wiktionary; the PI doesn't read French.
- Spanish generic-masculine evidence is replication of an observation we had already seen.
