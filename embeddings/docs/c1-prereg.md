# C1: Zhou et al.'s inanimate-noun bias score, after Zhou et al.'s own mitigation (pre-registration)

*Status: **ADOPTED by the PI, 2026-11-08** (decisions below). Written by Claude. No C1 data
had been computed at adoption. Written after the reproduction work in `results/zhou_shift_es/` (which scores
occupations only) and the audits in `docs/upstream-issues.md`.*

## Question

Zhou et al. (2019) define a per-word bias score for inanimate nouns (§2.1, Eq. 1: b_w = |s(w, A, B)|)
and a mitigation meant to drive it to zero (Eq. 2), but report bias only for occupation pairs. **After
their mitigation, do grammatically masculine and feminine inanimate nouns still differ in their
association with male vs female words?** Their mitigation removes only the semantic gender direction
*orthogonalised* against the grammatical one (cos 0.389 before orthogonalisation), so the part of
"gender" shared by the two directions stays in every noun. Prediction (directional, stated before
any data): **the masculine-minus-feminine gap survives Shift Ori.**

## Materials

- **Embeddings:** MUSE aligned fastText Wikipedia vectors, `wiki.multi.es.vec` and `wiki.multi.en.vec`
  (store, checksummed in `SHA256SUMS`); the first 50,000 words, as Zhou et al. load them.
- **Directions and mitigation:** exactly as rebuilt in `scripts/zhou_shift.py` (PCA semantic
  direction from their 10 definitional pairs, LinearSVC grammatical direction on 3,000 + 3,000 of
  their nouns, orthogonalised semantic direction, Shift Ori), which reproduces Table 1's statistic
  (3.6918; 0.3087 vs 0.3090).
- **Test nouns** (from our Wiktionary lexicon, `data/lexicon/es_nouns.csv`, by rule):
  `gender` m or f (single gender); `animacy` = inanimate; no `multi_type` (no two-gender or
  meaning-split nouns); `also_form` empty (a lemma, not an inflected form of something else);
  `also_pos` empty (not also another part of speech; PI decision 3); in the MUSE vocabulary; **not** in Zhou et al.'s noun lists, definitional pairs, attribute
  words or occupation pairs (so nothing was used to build a direction or was shifted directly).
  Frozen before scoring as `embeddings/data/c1_nouns_es_v1.csv` (`scripts/c1_nouns.py`; sha256
  2c58b883…d4c0c2): **2,459 nouns in the 50k vocabulary (1,456 m, 1,003 f; primary)**, 8,126 in
  200k (sensitivity; column `in_50k`).
- **Attributes:** (a) Zhou et al.'s Spanish male/female words (8 + 8; they carry grammatical gender
  themselves); (b) their English definitional words (*man, woman, he, she…*) in the aligned space,
  which carry social gender without grammatical gender.

## Arms

1. Original embeddings.
2. Shift Ori **as implemented**: only Zhou et al.'s occupation pairs and their own inanimate-noun
   list are moved, so our held-out nouns are untouched by construction.
3. Shift Ori **as specified** (Eq. 2 applied to every inanimate noun, including ours): project the
   orthogonalised semantic direction out of each test noun.
4. Hybrid Ori, once rebuilt and checked against Table 1 (0.3090); until then, not run.

## Measures

- Per noun, signed s(w, A, B) = mean cos(w, male words) − mean cos(w, female words) (Zhou Eq. 1
  without the absolute value), and Zhou's b_w = |s|.
- **Primary:** the gap Δ = mean s(masculine nouns) − mean s(feminine nouns), per arm and attribute
  set. Inference: 95% bootstrap CI over nouns (2,000 resamples, seed fixed) and a label-permutation
  p-value (10,000 permutations).
- **Spelling control:** Δ separately for regular nouns (m-*o*, f-*a*) and nouns whose ending
  carries less gender information (`es_ending` = other); exceptions (m-*a*, f-*o*, -*ma*) reported
  descriptively (too few for inference).
- Distribution of b_w per arm (Zhou define it but never report it).

## Reading (fixed before data)

- **Gap survives mitigation** if, in arm 2 or 3 with Spanish attributes, Δ's 95% CI excludes 0 and
  Δ is at least half its arm-1 value.
- **Gap is removed** if the CI includes 0 and |Δ| < 0.25 × arm-1 Δ. Otherwise **"reduced"**.
- English attributes: same rule; if the arm-1 gap with English attributes is itself not
  distinguishable from 0, report "no social-gender lean to remove" rather than a mitigation verdict.

## PI decisions (2026-11-08)

1. Run both Shift Ori arms, as implemented and as specified ("both, because they ran both").
2. 50k vocabulary primary, 200k as sensitivity.
3. Drop nouns that are also another part of speech.

## Results (2026-11-08; script `scripts/c1_run.py` committed 1be8c7a before the run; outputs `results/c1_es/`)

Primary (50k, all 2,459 nouns). Δ = mean s(masculine) − mean s(feminine), in cosine units; all 95%
CIs exclude 0 and all permutation p < 0.0001.

| arm | Δ, Spanish attributes | verdict | Δ, English attributes | verdict |
|---|---|---|---|---|
| 1 original | 0.0305 [0.0286, 0.0326] | — | 0.0139 [0.0117, 0.0162] | — |
| 2 Shift Ori as implemented | 0.0261 (0.86×) | **survives** | 0.0139 (1.00×) | **survives** |
| 3 Shift Ori as specified | 0.0258 (0.84×) | **survives** | 0.0082 (0.59×) | **survives** |
| 4a Hybrid Ori as implemented | 0.0261 (0.86×) | **survives** | 0.0249 (1.79×) | **survives** |
| 4b Hybrid Ori as specified | 0.0258 (0.84×) | **survives** | 0.0195 (1.40×) | **survives** |

- **The prediction holds: the masculine/feminine gap survives every version of Zhou et al.'s
  mitigation** (8/8 pre-registered verdicts "survives"; 200k sensitivity the same direction, Δ
  0.007–0.026, all CIs exclude 0). The smallest remaining gap is arm 3 with English attributes, 59%
  of the original.
- **Hybrid with Spanish attributes is identical to Shift**, as expected (rotation invariance,
  `results/zhou_hybrid_es/`). With English attributes, aligning to hard-debiased English
  **increases** the gap (1.79× as implemented): Spanish things move toward the English gender words
  that hard debiasing deliberately keeps gendered.
- **Arm 2 changes the Spanish-attribute gap although our nouns are untouched**: 12 of the 16
  Spanish attribute words (*mujer, hombre, madre…*) are in Zhou et al.'s "inanimate" noun list, so
  their own mitigation projects the semantic direction out of the attribute words (upstream Z12).
- **Spelling control:** the gap is larger for regular *-o/-a* nouns (original, Spanish attributes
  0.038 vs 0.021 for other endings) but present in both and survives in both (all CIs exclude 0).
  Exceptions (45 nouns) descriptive only: `results/c1_es/c1_info.json`.
- Scale: Δ ≈ 0.03 against a mean |s| of 0.023 per noun; small in cosine terms, consistent across
  2,459 nouns.

## Addendum A: French (frozen 2026-11-08, before any French C1 score; PI decision: same design)

- Lexicon: `data/lexicon/fr_nouns.csv`, built by the same rules (`gbleed lexicon --langs fr`;
  French dump sha256 in `data/lexicon/metadata.json`). Same noun filter (m/f only, inanimate, no
  multi_type, no also_form, no also_pos), MUSE `wiki.multi.fr.vec`, excluding all of Zhou et
  al.'s French lists (nouns, definitional pairs, occupations).
- Frozen list `embeddings/data/c1_nouns_fr_v1.csv` (sha256 0e8687a3…0196e51c): **3,505 nouns in
  50k (2,022 m, 1,483 f; primary)**, 8,446 in 200k.
- Attributes: (a) Zhou et al.'s French attribute words = their 10 French definitional pairs
  (notebook cell 215); (b) the same English definitional words as Spanish.
- Hybrid alignment: Procrustes on the MUSE fr-en dictionary (5,000 most frequent French entries).
- Spelling control: nouns with a gender-predicting suffix (`fr_suffix`, a fixed list of 16
  textbook suffixes: -tion, -ité, -ment, -age…; 1,179 nouns) vs nouns without (2,326).
- Same arms, measures, prediction (gap survives) and reading rule.
