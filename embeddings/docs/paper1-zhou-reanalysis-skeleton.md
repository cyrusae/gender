# Paper 1 skeleton: reanalysing Zhou et al. (2019)

*Drafted by Claude 2026-11-10 after the two-paper decision (`papers-shared-plan.md`). Numbers come
from `zhou-summary.md` and `upstream-issues.md`; those stay the source of truth. Short format
(findings / reproducibility track, 4–8 pages + appendix).*

## Working title

*The bias is real, the evidence isn't: reanalysing "Examining Gender Bias in Languages with
Grammatical Gender"*. Softer alternative: *Reproducing and correcting Zhou et al. (2019)*.

## One-sentence claim

Zhou et al.'s headline numbers reproduce, but none of their tests can show what they are cited
for: the p-values are guaranteed by construction, the Shift result is an identity, Hybrid equals
Shift monolingually, and the word-pair test has no feminine cue; with corrected tests the
masculine/feminine asymmetry survives in French and cross-lingually, and it comes from the
masculine form's generic use and from occupational stereotype, not from dictionary anchoring.

## Abstract (draft)

Zhou et al. (2019) report that feminine occupation forms in Spanish and French embeddings lean
further toward female words than masculine forms toward male words, and propose mitigations
(Shift, Hybrid). Rerunning their released code, we reproduce the headline statistics exactly and
find that the tests behind them are uninformative: the MWEAT permutation p-value equals zero by
construction whenever the two totals have opposite signs, and after mitigation most permutations
tie with the observed value; the Shift result (3.69 → 0.31) is an algebraic identity determined
before any vector moves; Hybrid cannot differ from Shift on any monolingual measure; the French
table does not match the notebook; the word-pair bias test uses no feminine adjective cue; and the
apparent translation gain is a 200k-vs-50k candidate-set artefact. Two independent blind checks
confirmed the core identities. We then ask whether the asymmetry is real. With a sex-neutral zero
point it holds in French and is inconclusive in Spanish; with feminine cues added, the masculine
form is still retrieved first; in Spanish the asymmetry shrinks by half in the plural, consistent
with the masculine singular's generic use; and English "doctor" lands on the masculine form in
proportion to how male-dominated the job is, not to which form the bilingual dictionary lists.
The bias Zhou et al. described is real; the measurements need rebuilding.

## 1. Introduction

- Zhou 2019 is the standard citation for "gender bias in gendered-language embeddings" (87
  citations; forward-citation screen in `zhou-downstream-reuse.md`: only Xiao & Li 2026 and
  Omrani 2022 reuse the method, nobody reuses the mitigation numbers).
- What we did: rerun the released notebook against the paper, derive why the numbers behave as
  they do, get two blind confirmations, then build tests that can fail.
- Contributions: (i) reproduction with issue log Z1–Z18; (ii) proofs of the identity and the
  degenerate null; (iii) four pre-registered or corrected tests of the asymmetry and its mechanism;
  (iv) released scripts and frozen lists.

## 2. What Zhou et al. did (one page)

Semantic direction (PCA over 10 definitional pairs), grammatical direction (stated LDA, actually
LinearSVC, Z1), cos 0.389, orthogonalised d_s; MWEAT on 58 ES / 23 FR occupation pairs with 8 + 8
attribute words; Shift (Eq. 4) and Hybrid (align to hard-debiased English); Table 2 word-pair
analogy and word translation. Pipeline map from `orientation.md` and `zhou-shift-identity.md` §1–2.

## 3. What reproduces

Table: Original 3.6918 ES exact; cos 0.389 with SVC + `a + b/2` centring (Z3); Spearman EN–ES
0.450 (paper 0.45); Table 2 Original and Shift rows; word translation 79.2/89.0 at 200k. Say this
first and plainly.

## 4. What the tests cannot show

| § | finding | code | evidence |
|---|---|---|---|
| 4.1 | Observed MWEAT is the permutation maximum when totals have opposite signs (‖x|−|y‖ ≤ |x+y|); p = 0.000 is guaranteed before mitigation; after Shift 75.7% (ES) / 94.5% (FR) of permutations tie, so 0.113 / 0.733 / 0.269 for the same 0.3090 are rounding | Z10, Z2 | `scripts/zhou_perm_max.py`; `zhou-shift-identity.md` §3 |
| 4.2 | Shift Ori identity: post-shift statistic = \|Σ w·a⊥\| (opposite signs) or original \|T_m − T_f\| (same sign); 0.308748 = 0.308748 ES, FR to 1e-15; cos(a, u) 0.82 / 0.88, so the drop measures how much of the yardstick lies on d_s | Z14 | `scripts/zhou_construct.py`, `results/zhou_construct.json` |
| 4.3 | Hybrid = Shift under orthogonal alignment (identical to 1e-10); De-Align 3.5748 can't be reproduced (3.6918) | Z11 | `results/zhou_hybrid_es/` |
| 4.4 | French Table 1 vs notebook: Shift FR 0.2446 (p 0.147) vs 0.0649 (p 0.019) | Z13 | `results/zhou_shift_fr/README.md` |
| 4.5 | Word-pair test: `adj_pairs` has only masculine/invariant cues, so "F MRR < M MRR" partly measures agreement; unranked forms dropped not scored 0; ES Hybrid cell ends in KeyboardInterrupt; Hybrid row doesn't reproduce (0.256/0.696 vs 0.345/0.640) | Z17 | `results/zhou_wordpair/` |
| 4.6 | Word translation: Original scored on 200k candidates, mitigated on 50k; 79.2 → 80.7 is the same space at two candidate sizes | Z18 | `results/zhou_sec43_44/` |
| 4.7 | Yardstick moves under the mitigation: 12/16 attribute words in the "inanimate" list (Z12), definitional pairs also shifted (Z16), cell 274 FR offset bug (−0.033) | Z12, Z16 | `zhou-shift-identity.md` §5 |
| 4.8 | Code vs paper: Eq. 2 in no cell, Spanish Shift Ori loaded from an unreleased file, mitigated embeddings not released; inanimate score b_w never reported | Z15, Z4, Z6 | `zhou-shift-identity.md` §2 |
| 4.9 | Noun lists: 4,888 non-noun lemmas, 860 both-gender, animates, 18 *el agua* feminines as masculine (tagger-derived) | Z5, Z9 | `results/label_audit_es/` |
| 4.10 | Blind checks: fresh Claude agent and Gemini, no repo access, both CONFIRMED Z10/Z14/Z15; Gemini extended Z10 to the Original column; Claude narrowed Z14 (same-sign branch) and found Z16 | | `zhou-shift-identity.md` §5–6; `blind-checks/` |

Write-up rule: "the code doesn't match the described design", never intent
(`issues-for-authors-draft.md` tone). Low items (Z3, Z7, Z8) go to the appendix.

## 5. Is the asymmetry real?

| § | test | status | result |
|---|---|---|---|
| 5.1 | Epicene zero point, attribute words resampled, leave-one-out, S1–S7 | pre-registered (`zhou-zero-point-prereg.md`) | FR A −1.91 [−3.42, −0.30] holds, robust; ES −1.99 [−5.63, 1.34] inconclusive; offset explains 17–46% of Zhou's statistic |
| 5.2 | Word-pair test with feminine cues | exploratory (`scripts/zhou_wordpair.py`) | masculine form still wins under a feminine cue (ES M 0.58 vs F 0.37; FR 0.54 vs 0.33); invariant-cue gap ES 0.45, FR 0.38; mitigations narrow only the invariant gap in ES |
| 5.3 | Crime-noun positive control | pre-registered (`crime-noun-probe-prereg.md`) | ES epicene crime nouns lean male +0.040 [0.008, 0.074]; FR inconclusive; not crime-specific |

Conclusion of §5: the direction of Zhou's claim is right; the French evidence is clean, the
Spanish evidence rests on the cross-lingual test and §6.

## 6. Why the asymmetry exists

| § | hypothesis | status | result |
|---|---|---|---|
| 6.1 | Generic masculine: masculine singular doubles as the generic, so its lean is diluted; plural should show it | pre-registered (`generic-masculine-prereg.md`); ES is a replication of an exploratory sighting, FR confirmatory | ES Δρ 0.51 [0.14, 1.14]: masculine plural keeps ~¼ of its lean and sits at neutral, feminine ¾; FR inconclusive (masc singular already +0.076, ratio ill-conditioned) |
| 6.2 | Cross-lingual: why "doctor" → *médico*: anchoring (dictionary lists only masculine), generic usage, or stereotype | pre-registered (`wordpair-mechanism-prereg.md`) | anchoring null (29/58 dictionary entries list both forms; removing occupations from the alignment changes nothing); stereotype supported (BLS share β 0.30, Glasgow β 0.32; BLS fragile without homographs); no intrusion into female-dominated jobs; field-noun feminines (*química*) add |

## 7. Discussion

- Five measurement lessons, each with the Zhou instance: bounded statistics give degenerate
  permutation nulls (cite Phipson & Smyth 2010; WEAT as specified by Caliskan uses a different
  statistic); a yardstick the mitigation can move; rotation invariance blinds monolingual tests to
  alignment-based mitigation; candidate-set mismatch; gold labels from a dictionary, not a tagger.
- What the paper's conclusion should have been: the asymmetry is a generic-masculine and
  stereotype effect; Shift removes the part of the yardstick on d_s and nothing else; Hybrid's
  real effect is cross-lingual (point to paper 2 for the measurement).
- Downstream: nobody reused the mitigation numbers, so the damage is to the method's reputation
  as a template, not to a chain of results.

## 8. Limitations

Hybrid/De-Align rebuilt (orthogonal Procrustes onto hard-debiased English), results on them
approximate; §5.2 exploratory; ES zero point inconclusive, not null; French judged through
glosses and Wiktionary; ES generic-masculine is a replication of an already-seen pattern; 50k
vocabulary primary, 200k sensitivity; MUSE vectors only.

## 9. Ethics / author contact

Nothing sent yet (`issues-for-authors-draft.md`). Norm for reproducibility work: contact the
authors before submission and report their reply. Decide before the repo goes public.

## Appendices

- A: Z1–Z18 table verbatim.
- B: identity derivation (`zhou-shift-identity.md` §3) and both blind-check reports with
  `TASK.md`.
- C: preregs in adoption order with commit hashes (zero point, crime, generic masculine, word-pair
  mechanism).
- D: list construction: epicene anchors (floor 15 + LOO), crime nouns, generic-masculine pairs,
  occupation mapping v1 (BLS 2025, Eurostat 2024).

## Must not overstate

- ES zero point and FR generic masculine: inconclusive.
- "Real" rests on FR zero point (prereg) + feminine-cue word pair (exploratory) + ES generic
  masculine (replication) + stereotype mechanism (prereg). Label each.
- Hybrid numbers are ours.

## Literature this paper leans on

From `papers-shared-plan.md` §B: B1 (targets and lineage), B3 (metric validity: Phipson & Smyth,
Goldfarb-Tarrant, Antoniak & Mimno, WEAT reliability), B5 (generic masculine psycholinguistics),
B6 (reproducibility reporting). B4 only for Glasgow/BLS as criteria in §6.2.
