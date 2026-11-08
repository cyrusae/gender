# C1: Zhou et al.'s inanimate-noun bias score, after Zhou et al.'s own mitigation (DRAFT pre-registration)

*Status: DRAFT, 2026-11-08, written by Claude for the PI. Nothing here is adopted. No C1 data
have been computed. Written after the reproduction work in `results/zhou_shift_es/` (which scores
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
  in the 50k MUSE vocabulary; **not** in Zhou et al.'s noun lists, definitional pairs, attribute
  words or occupation pairs (so nothing was used to build a direction or was shifted directly).
  Count at this filter: **2,815 nouns** (1,753 m, 1,062 f; endings: m-o 799, f-a 719, other 1,234,
  exceptions m-a 15, f-o 8, -ma 40). List frozen as `data/c1_nouns_es_v1.csv` before scoring.
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

## Open decisions for the PI

1. Run both arm 2 (as implemented) and arm 3 (as specified)? Recommended: yes, they answer
   different questions (what their released method does vs what their equation says).
2. Vocabulary: 50k (Zhou's setting, 2,815 nouns) or 200k (8,815 nouns, rarer words)? Recommended
   50k primary, 200k as sensitivity.
3. Exclude nouns that are also another part of speech (`also_pos`, e.g. *cuenta*, also a verb
   form)? Recommended: yes for the primary analysis (1,290 of 23,790 base nouns).
