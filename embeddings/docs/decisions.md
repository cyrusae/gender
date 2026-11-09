# Embeddings study: decisions log

| date | decision | rationale | status |
|---|---|---|---|
| 2026-11-08 | Study lives in `embeddings/` in the gender monorepo; upstream code cloned (git-ignored) in `external/`. | Reuses the Wiktionary lexicon; can be split out later with history. | done |
| 2026-11-08 | Reproductions first: GG_Disentangling Spanish rerun reproduces; Zhou Shift Ori rebuilt (Table 1 statistic reproduces, p-value doesn't). Upstream problems logged in `docs/upstream-issues.md`. | Know the baseline before changing it. | done |
| 2026-11-08 | **C1 pre-registration adopted** (`docs/c1-prereg.md`): both Shift Ori arms (as implemented / as specified), 50k primary + 200k sensitivity, drop nouns that are also another POS. Noun list frozen (`data/c1_nouns_es_v1.csv`, 2,459 / 8,126). | PI. | adopted |
| 2026-11-08 | **C1 run**: gap between masculine and feminine inanimate nouns survives all Zhou mitigations (8/8 'survives'); Hybrid aligned to debiased English increases the English-attribute gap 1.4–1.8×. Hybrid rebuilt (dad4469): equals Shift on Spanish-only measures; post-mitigation MWEAT p-values are tie artefacts. | Pre-registered rule. | done |
| 2026-11-08 | **Gender ValNorm prereg adopted** (`docs/gvalnorm-prereg.md`): 50k primary/200k check; Omrani erasure arm (CC fastText, bugs-fixed inputs, 14 and 28 iterations). French lexicon to be built before French C1/ValNorm. | PI. | adopted |
