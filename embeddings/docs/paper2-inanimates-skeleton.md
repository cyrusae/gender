# Paper 2 skeleton: grammatical and social gender in inanimate-noun embeddings

*Drafted by Claude 2026-11-10 after the two-paper decision (`papers-shared-plan.md`). Numbers from
`inanimates-summary.md`, `c1-prereg.md`, `gvalnorm-prereg.md`, `upstream-issues.md` (O-series).
Full-length paper; may still grow (German arm, LLM link).*

## Working title

*What debiasing leaves behind: grammatical and social gender in Spanish and French inanimate
nouns*.

## One-sentence claim

An inanimate noun's vector carries a large grammatical-gender component and a small, real
social-gender component (tracking human gender ratings of the object); of the three mitigations in
the literature, Zhou's Shift removes the social part only along its own axis, Hybrid removes it
cross-lingually through Bolukbasi's English debiasing, and Omrani Sabbaghi & Caliskan's erasure
removes the grammatical part and leaves the social part untouched. Neither paper measured the
half it wasn't targeting.

## Abstract (draft)

Two lines of work treat grammatical gender in gendered-language embeddings in opposite ways:
Zhou et al. (2019) protect it while removing social-gender bias, Omrani Sabbaghi & Caliskan
(2022) erase it to expose social-gender bias. Neither measures what happens to the component it
isn't targeting, and neither tests inanimate nouns against human judgments. On 2,459 Spanish and
3,505 French Wiktionary-labelled inanimate nouns outside every upstream list, we measure both
components before and after each mitigation. The grammatical gap survives every arm of Zhou's
mitigation (16/16 pre-registered verdicts) and, measured with English gender words, Hybrid
inflates it. A social-gender component exists beyond grammatical gender and frequency: the
embedding gender score tracks Glasgow human gender ratings of the objects' English translations
(β 0.17–0.34, all intervals above zero), a "gender ValNorm" for gendered languages. Where that
component lies depends on the yardstick: with native gender words it sits on Zhou's semantic axis,
with English words half to three quarters lies off it. Shift removes only the on-axis part, which
is close to built in; Hybrid removes the off-axis part too, and the effect persists with an
un-debiased English yardstick, so it is Bolukbasi's debiasing carried over by alignment; Omrani's
erasure cuts the grammatical gap by 72–95% and leaves the social component unchanged in raw units.
All measurements were pre-registered.

## 1. Introduction

- Two meanings of "gender" in a gendered language's vectors: *la mesa* (grammatical) and
  lipstick vs hammer (social). The 3 × 2 table (mitigation × component) is the paper.
- Why inanimates: occupations confound social and grammatical gender by design; things are where
  the two can be separated.
- Why static embeddings in 2026: reproducible testbed; the measurement lessons transfer to LLMs;
  link to the PI's LLM results (same split: grammatical strong, social small but real).
- Contributions: (i) the first inanimate-noun measurement under Zhou's mitigation (its stated aim,
  never evaluated); (ii) gender ValNorm for ES/FR; (iii) the yardstick dependence and what it
  reveals about Shift vs Hybrid; (iv) Omrani's erasure measured for the social half, plus a
  reproduction with held-out nouns and bug fixes; (v) frozen Wiktionary lists.

## 2. Background

- Grammatical gender in distributional space: Gonen et al. 2019, McCurdy & Serbetçi, Williams et
  al. 2021, Stańczak et al. 2023; human evidence weak (Mickan 2014, Elpers 2022) in one sentence.
- Bias scores: SC-WEAT / WEFAT (Caliskan 2017), ValNorm (Toney & Caliskan 2021), English gender-norm
  precedents (Lewis & Lupyan 2020 r 0.63/0.59; Valentini 2023 r 0.44–0.58).
- Mitigations: Bolukbasi hard debias; Zhou Shift/Hybrid (one paragraph, cite paper 1 for "the
  native-yardstick effect is partly algebraic, cos(a, d_s) 0.82/0.88"); Omrani iterative erasure
  (= INLP, Ravfogel 2020; contrast LEACE).
- Frequency confound in gender metrics (Valentini et al. 2022).

## 3. Materials

- MUSE aligned fastText (Zhou arms; 50k primary, 200k sensitivity); Common Crawl fastText
  (Omrani arm). Exact files + checksums (`scripts/fetch_embeddings.sh`).
- Noun lists: Wiktionary (kaikki) via `gbleed lexicon`; singular, inanimate, one gender, not
  another POS, outside every Zhou/Omrani list; C1 frozen `data/c1_nouns_{es,fr}_v1.csv`
  (2,459 / 3,505). Logged slips not edited (*eminencia*, *voyer*, *philharmonie*).
- Yardsticks: native gender words (Zhou's 8 + 8), English gender words in the aligned space,
  d_s-orthogonal yardstick (Addendum B), un-debiased English (Addendum C).
- Human ratings: Glasgow GEND of English translations; frequency: wordfreq.
- Statistics: split-half / bootstrap CIs, label permutation with continuous null (why: paper 1).

## 4. Grammatical gender under Zhou's mitigation (C1)

| § | result | numbers | file |
|---|---|---|---|
| 4.1 | Gap survives every arm, 16/16 | ES keeps 84–86% (native yardstick); FR 1.11–1.16× | `results/c1_{es,fr}/c1_verdicts.csv` |
| 4.2 | Hybrid inflates the English-yardstick gap | ES 1.4–1.8×, FR 2.5–2.9× (aligning to debiased English moves things toward *woman, mother*, left gendered by hard debias) | same |
| 4.3 | Not spelling | gap present for every ending: ES 0.038 regular vs 0.021 other; FR 0.027 suffixed vs 0.026 unsuffixed; exceptions (*problema, mano*) classified 93–100% | C1 spelling control; `results/label_audit_es/` |
| 4.4 | Zhou's Fig. 1 claim holds on that axis | masc −0.011, fem −0.012 on 2,000 Wiktionary nouns | `results/zhou_figures/` |

## 5. Social gender beyond grammar (gender ValNorm)

| § | result | numbers | file |
|---|---|---|---|
| 5.1 | Human rating predicts the gender score after grammatical gender + frequency | β_H ES 0.25 (native) / 0.34 (English); FR 0.17 / 0.28; all CIs > 0; grammatical β ≈ 1 | `results/gvalnorm_{es,fr}/gvalnorm_fits.csv` |
| 5.2 | English calibration | r 0.58 (fastText CC / MUSE wiki), in the 0.45–0.65 band from Lewis & Lupyan and Valentini | `results/gvalnorm_en/` |
| 5.3 | Where it lives (Addendum B, orthogonal yardstick) | native: entirely on d_s; English: off-axis ES 0.150, FR 0.206 (half / three quarters) | `gvalnorm_orth_*` |

## 6. What each mitigation does to the social part

| § | mitigation | grammatical | social | numbers |
|---|---|---|---|---|
| 6.1 | Shift | survives | removed on-axis only, built in (cos 0.82/0.88); off-axis untouched | Addendum B |
| 6.2 | Hybrid | survives (inflated vs English) | removes off-axis part; persists with un-debiased English yardstick (ES −0.043, FR −0.004) → Bolukbasi via rotation, invisible monolingually | Addendum C |
| 6.3 | Omrani erasure (bugs fixed, 14/28 iter ES, 27 FR) | −72% ES, −95% FR | unchanged in raw units (ES 0.0080 → 0.0074; FR 0.0064 → 0.0066); standardised β rises only because grammatical variance shrank | `gvalnorm_omrani_*` |

Figure: the 3 × 2 table with these cells; a second panel with per-noun scatter (Glasgow GEND vs
score) before/after per arm.

## 7. Reproducing Omrani Sabbaghi & Caliskan (supporting section or appendix)

- Spanish rerun reproduces; iteration count mismatch (text 15, tables = iter 28, notebook 14; O6).
- Held-out nouns (O1/O4): effect generalises, gap 1.516 → −0.026; training-noun gap ~13% larger.
- Minimal fix (dedup, both-gender words out, NFC, unpadded accuracy): conclusions unchanged;
  ValNorm 0.807/0.818 → 0.796/0.833 (slightly stronger).
- Interpretation: removal applied to attribute words too (O7), so raw means shift together
  (both negative); report m − f differences (O8).
- Lists: ~40% not singular nouns, 99.8% label accuracy on real nouns (O17); NFD bug (O16).
- Tone: none of this overturns a conclusion; several make their results look weaker than they are.

## 8. Discussion

- "Debiasing" names two operations; the table shows each paper's blind spot.
- Yardstick dependence as a general result: the social component you can see depends on whether
  your attribute words carry grammatical gender; English-in-aligned-space as a grammar-free
  yardstick, with the caveat that it may import English stereotype.
- Which component should a downstream user want gone? Translation/agreement needs grammatical
  gender; fairness audits need it out of the way (Omrani) but should then check the social part
  survived (it does).
- LLM link: same split in Qwen3/Gemma hidden states (cite the PI's main study); static embeddings
  as the baseline.
- ES vs FR differences (orthographic transparency, suffix structure).

## 9. Limitations

Glasgow = British raters, English translations; Hybrid is our rebuild; English yardstick in
aligned space may import English stereotype (Addendum C rules out the yardstick moving, not
that); effects small in cosine (ES gap ≈ 0.03) but consistent over thousands of nouns; MUSE 50k
primary; two languages; a few person/group nouns in the frozen lists (logged).

## 10. Possible additions before submission (PI decision)

- German arm (Omrani code covers DE; `gbleed` has the German lexicon; Gemma/Qwen link). Cost: new
  prereg, new frozen list, German review only via glosses.
- Native-language gender norms instead of Glasgow translations (none exist for ES/FR; Misersky
  2014 covers role nouns only). Could be a stated gap rather than an addition.
- LEACE arm as a one-step erasure comparison to Omrani's 14–28 iterations.

## Appendices

- A: O1–O18 table; C1 and gender-ValNorm preregs with addenda and commit hashes.
- B: list construction and counts; exclusion rules; logged slips.
- C: per-ending breakdowns; 200k sensitivity; 14 vs 28 iterations.

## Must not overstate

- Standardised β for the Omrani arm grows for an artefactual reason; always show raw units.
- Hybrid's effect is cross-lingual, on our rebuild.
- "Social gender" = Glasgow ratings of translations, not ES/FR speakers' judgments.

## Literature this paper leans on

From `papers-shared-plan.md` §B: B2 (grammatical gender in distributional space), B4 (human norms:
Lewis & Lupyan, Toney & Caliskan, Glasgow, Vankrunkelsven, Misersky), B1 for the three
mitigations and INLP/LEACE, B3 for Valentini frequency and Antoniak & Mimno seed lexicons, B7 for
the linguistics paragraph.
