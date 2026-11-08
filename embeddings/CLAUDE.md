# embeddings/ — embeddings study (static word embeddings; scoped rules, DRAFT)

A separate, small study: a partial replication and extension of Zhou et al. (2019), "Examining
Gender Bias in Languages with Grammatical Gender" (EMNLP-IJCNLP 2019; arXiv 1909.02224). The root
`CLAUDE.md` applies here except where this file overrides it. **Status: idea stage, nothing
adopted.** The overrides below are placeholders for the PI to confirm when a design is written.

## The question (PI, 2026-11-08)

Zhou et al. separate the grammatical-gender component from the semantic-gender component of
Spanish/French word embeddings and report bias-association tests (a WEAT derivative, after
Caliskan et al. 2017) for **animate** nouns only. Do **inanimate** nouns still carry a
social-gender association once grammatical gender is removed? Related: Omrani Sabbaghi & Caliskan
(2022) on disentangling grammatical gender signals; Bolukbasi et al. (2016).

## Where things are

- `external/Bias_in_Gendered_Languages/`: Zhou et al.'s code and seed lists, cloned from
  https://github.com/shaoxia57/Bias_in_Gendered_Languages (commit 6392702, 2019-09-05). Read-only
  reference; git-ignored. It has no licence file, so don't copy its lists into tracked files
  without checking with the PI; cite it instead.
- Design docs, code and results for this study go in this folder (`docs/`, `results/`), not in
  the root `docs/design/` or `results/`.

## Rule notes (draft, for the PI)

- **Base-models-only rule:** not applicable as written (static embeddings, not LLMs). The point
  of the rule (no tuning-dependent model variants) carries over: use published embeddings,
  record exact versions and sources.
- **Gold labels:** noun genders still come from Wiktionary (`gbleed lexicon`), never from a model;
  animacy labels need a rule-based source too, decided in the design.
- **Pre-register before outcomes**, as everywhere else in the repo.
