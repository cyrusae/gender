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
inanimate nouns' grammatical gender. **The bias is real** in the cross-lingual test, where it is stronger than the paper's own
test could show; the monolingual asymmetry is inconclusive in Spanish and borderline in French with native attribute words (it flips with single-word list choices)
once a fair zero point is used and the paper's attribute-word errors are corrected (Z19); with
English attribute words French still shows it (−2.03 [−3.39, −0.63]), Spanish doesn't. **The quantitative evidence for it and for the mitigations mostly
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
| Feminine occupation forms lean further (Fig. 1, MWEAT 3.69) | Number reproduces exactly, but depends on errors in their attribute words (Z19: correcting *tío* alone gives 2.25). Against an epicene zero point with corrected attribute words, resampled: ES −1.90 [−5.51, 1.39], FR −1.53 [−3.25, 0.41], both inconclusive; French is borderline (restricted to the cleaned occupation list, i.e. without *conservateur* 'tory', it reads −1.91 [−3.34, −0.39]; `results/list_audit/`). With English attribute words (unaffected by the fix): FR −2.03 [−3.39, −0.63], asymmetry; ES −3.15 [−6.35, 0.17], inconclusive and dependent on two anchors (*bebé, hada*). (With their uncorrected lists French had held, −1.91 [−3.42, −0.30].) | `zhou-zero-point-prereg.md` deviation, Z19 |
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
- **Why English "doctor" lands on *médico* (cross-lingual).** It is specific to occupations: for
  107 ordinary person nouns (*vecino/vecina, amigo/amiga*) there is no intrusion (−0.005 [−0.077,
  0.070]) against 0.290 [0.221, 0.357] for occupations (pre-registered, robust to every homograph
  flag), so the generic masculine alone doesn't explain it. Without a cue the masculine form is the default
  translation for both groups, about twice as strongly for occupations (0.51 vs 0.28); a feminine
  cue shifts both by a similar amount, cancelling the person default but only part of the
  occupational one (exploratory). Within nouns, both how much rarer the
  feminine form is (β 0.35) and how male the English noun is rated (β 0.22) predict intrusion; the
  two are entangled for occupations. Not the bilingual dictionary (D null twice). Feminine forms that
  also name a field (*química*) add to it. `wordpair-mechanism-prereg.md`, `wordpair-persons-prereg.md`
- **Follow-ups on ordinary person nouns (Spanish).** Monolingual asymmetry: persons and occupations
  both inconclusive and not distinguishable (occupations' point estimate about twice persons').
  Generic masculine: masculine plurals move toward neutral for both groups (Δρ positive); occupations
  supported (0.46 [0.05, 1.18]), persons inconclusive (0.29 [−0.08, 0.76]), but the verdicts depend on
  which nouns are included (they swap at a 50k vocabulary). `persons-zero-point-prereg.md`,
  `persons-genmasc-prereg.md`
- **What Hybrid really does.** Measured against English gender words, it removes stereotype
  signal that lies off Zhou's axis, even with the original English yardstick. That is
  Bolukbasi's English debiasing carried over by alignment, not anything Table 1 can show.
  `gvalnorm-prereg.md` addenda B–C
- **A positive control.** Grammatically neutral crime nouns (*delincuente*) lean male in Spanish
  (+0.039 [0.008, 0.070]); French borderline (male-leaning with corrected attribute words,
  0.037 [0.001, 0.079]; inconclusive if the name pair is kept); not shown to be crime-specific.
  `crime-noun-probe-prereg.md`

## Limits of our checks

- Hybrid and De-Align are rebuilt (orthogonal Procrustes onto hard-debiased English), because no
  mitigated embeddings were released (Z4). Results on those spaces are approximate.
- Several of our analyses were exploratory (figure redraw, word-pair rerun, §4.3–4.4); the
  zero-point, generic-masculine, crime, word-pair mechanism, expanded-cue and person-noun studies were
  pre-registered.
- French judgments rely on English glosses and Wiktionary; the PI doesn't read French.
- Spanish generic-masculine evidence is replication of an observation we had already seen.
