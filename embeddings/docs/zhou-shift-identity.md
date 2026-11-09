# Zhou et al. (2019) Shift Ori: what the released code shifts, and why MWEAT falls (2026-11-08)

Sources: paper arXiv 1909.02224 §2, §3.1–3.2; notebook `bias_emnlp19.ipynb` at commit 6392702 of
https://github.com/shaoxia57/Bias_in_Gendered_Languages. Cell numbers are 0-based indices into the
notebook's `cells` array; line numbers are within the cell source.

## 1. What the paper says Shift Ori does

- §2: d_s = d_PCA − ⟨d_PCA, d_g⟩ d_g (semantic direction made orthogonal to the grammatical one).
- §3.1 Eq. (2): for **inanimate nouns**, minimise ⟨w, d_s⟩, i.e. project d_s out.
- §3.1 Eq. (4): for **two-form nouns** (occupations), minimise |⟨w_m, d_s⟩ + ⟨w_f, d_s⟩ − 2⟨w_a, d_s⟩|;
  Shift **Ori** uses the origin as anchor (w_a = 0).
- §3.2: "Effectively, for inanimate nouns, we remove the semantic gender component from the
  embeddings. For nouns with two gender forms … we shift the two forms along the semantic gender
  direction so that they have the same distance to the anchor point."

## 2. What the released notebook contains

| what | where | code |
|---|---|---|
| semantic direction | cell 78, lines 8–11 | `get_SG_component(v, ggd) = v - (v.dot(ggd)*ggd/ggd.dot(ggd))`; `ES_semantic_gender_direction = get_SG_component(ES_gender_direction, ES_grammar_gender_direction_3000)` |
| Eq. (4) offset, origin anchor | cell 275, lines 1–10 (also 273 ES-EN, 274 FR) | `offset = -0.5*(w_m·d_s + w_f·d_s)` per occupation pair |
| the shift itself, ES Hybrid Ori | cell 312, lines 2–7 | `for pair in ES_EN_combined_occupation_words: E[w_m] += offset*d_s/d_s.dot(d_s); E[w_f] += …` (offsets from cell 273) |
| same pattern, FR Hybrid Ori / other variants | cell 315, lines 7–12; cells 310, 320, lines 2–7 (these two loop over the hand list `combined_occupation_words`, cell 54) | as above |
| also shifted: definitional pairs | cell 292, lines 6–11 (FR: cells 293, 296; cell 296 line 8 divides by `d_DA·d_s` instead of `d_DA·d_DA`, a mixed-direction slip) | the same move applied to `ES_defs`, on top of `ori_offset_debiased_ES_embeddings` |
| **Spanish Shift Ori embedding** | **cell 14, lines 1–2** | **loaded from `./data/ori_offset_debiased_ES_embeddings.txt`, not built in the notebook; the file is not in the repo** (cell 4 lines 10–11: the commented-out `save_vec` that once wrote it) |
| the only `drop()` (projection) calls on word lists | cells 247–248 | applied to the career/family WEAT words, for a different test |
| the occupation words that are scored | cell 234, lines 1–2 | `ES_female/male_occupation_words` from `ES_EN_combined_occupation_words`: the same pairs the offsets loop over |
| Table-1-style p-values with saved outputs | cell 305 (ES Hybrid Ori: 0.309022, p 0.269); cell 306 (FR Shift Ori: 0.064891, p 0.019) | `_weat_pval(...)` |

**What can and can't be said about the code/equation gap:**
- **Eq. (2) (project d_s out of inanimate nouns) appears in no cell of the released notebook.** The
  only projections of a direction out of word lists are cells 247–248, on WEAT career/family words.
- The Spanish Shift Ori embedding was built **outside the released code** (cell 14 loads it), so
  whether *that* file applied Eq. (2) can't be determined. "The released code doesn't implement
  Eq. (2)" is accurate; "their Shift Ori didn't apply Eq. (2)" is **not established**.
- Eq. (4) with the origin anchor **is** implemented (cells 275 + 310/315/320), on the occupation
  pairs and, in cell 292, on the definitional pairs.
- **The words shifted by Eq. (4) are the same words MWEAT scores:** cell 312 shifts
  `ES_EN_combined_occupation_words`; cell 234 scores the same list. For the loaded Spanish Shift Ori
  file this can't be seen, but our rebuild (shift those pairs only) reproduces Table 1's 0.3090 to
  3 decimals and the notebook's saved FR value (0.0649 vs 0.0650).

**Correction to our own earlier labels.** Our C1/gender-ValNorm arm called "Shift Ori as
implemented" projects d_s out of Zhou et al.'s noun lists. That is **our reconstruction** (Eq. 2
applied to the nouns they name in §2), not something visible in the released code. It should read
"reconstruction A: Eq. 2 on their noun list"; "as specified" = Eq. 2 on all inanimate nouns.
Upstream Z12 ("their mitigation moves the attribute words") is likewise an inference from Eq. (2)
plus list membership, not an observation of their code.

## 3. The identity: why MWEAT falls under Shift Ori, whatever else is shifted

Notation. A = male attribute words, B = female attribute words (Spanish: the 8 + 8 `ES_WEAT_*`).
For a word w, MWEAT's per-word association is

  s(w) = mean_{x∈A} w·x − mean_{y∈B} w·y = w·a,  with a := mean_A x − mean_B y.

For the occupation pairs (m_i, f_i), i = 1…n: T_m = Σ_i m_i·a, T_f = Σ_i f_i·a, and the reported
statistic is S = | |T_m| − |T_f| |.

Let u = d_s/‖d_s‖ and split every vector into its u-part and the rest: a = αu + a⊥ (α = a·u),
w = (w·u)u + w⊥. Then w·a = α(w·u) + w⊥·a⊥.

**Step 1 (what Shift changes).** Shift Ori adds a multiple of u to m_i and f_i, chosen so that
m_i·u + f_i·u = 0 for every pair (that is Eq. (4) at its minimum, anchor 0). It leaves w⊥ unchanged.
Any other operation the code applies along u (shifting definitional pairs, projecting u out of
nouns or attribute words) changes only u-coordinates too, so it changes α but not a⊥.

**Step 2 (sum).** After Shift, write X = Σ_i m_i·u = −Σ_i f_i·u. Then
  T_m = αX + P_m,  T_f = −αX + P_f,  where P_m = Σ_i m_i⊥·a⊥, P_f = Σ_i f_i⊥·a⊥ (unchanged by Shift).
So T_m + T_f = P_m + P_f: **the u-parts cancel exactly, whatever α is.**

**Step 3 (sign).** If T_m and T_f have opposite signs (true whenever |αX| > |P_m|, |P_f|, i.e. the
u-part dominates), then | |T_m| − |T_f| | = |T_m + T_f|. Hence

  **S_after = |P_m + P_f| = | Σ over all 2n occupation words of w⊥·a⊥ |,**

a number fixed before any shifting: the occupation words' total association with the part of the
attribute axis orthogonal to d_s. It doesn't depend on which word of a pair is "masculine".

**Step 4 (consequences).**
- The drop 3.69 → 0.31 measures how much of a lies along d_s (ES: cos(a, u) = 0.82; FR 0.88), not
  how much bias was removed from anything else. Before Shift, S = |T_m| − |T_f| with T_m, T_f of the
  same sign carries the m/f asymmetry; after, only |P_m + P_f| is left.
- **Permutation ties (Z10):** a random re-split of the 2n words into two groups of n gives
  T'_1 + T'_2 = Σ_all w·a, and whenever T'_1, T'_2 have opposite signs the statistic equals
  |Σ_all w·a| = S_after exactly (after Shift, Σ_all w·a = Σ_all w⊥·a⊥, the u-parts cancelling over
  all pairs). Observed: 757/1,000 (ES), 945/1,000 (FR) permutations tie.
- Condition check (our numbers): after Shift ES T_m = +4.990, T_f = −4.681 (opposite signs), and
  |T_m + T_f| = 0.30875; original space, with a replaced by a⊥: 0.30875. FR: 0.065019 both ways.

## 4. Numbers to reproduce

`embeddings/scripts/zhou_construct.py` → `embeddings/results/zhou_construct.json` (MUSE
`wiki.multi.{es,fr}.vec`, first 50k words; LinearSVC grammatical direction, which reproduces the
paper's 0.389, see Z1).
