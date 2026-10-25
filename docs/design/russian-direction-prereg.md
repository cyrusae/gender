# Russian gender direction: does it align with Spanish/German, and does it share German's frequency lean? Pre-registration (DRAFT for PI sign-off, 2026-10-25)

Written 2026-10-25, before any Russian-direction number exists. The Russian activations
(`phase4_ru`, Qwen3-8B/14B, pod CUDA bf16) were used so far only as test nouns for the Spanish and
German directions (Phase 4 T4); no direction has been fitted on them.

## Question (Stage 0, PI 2026-10-22)

Does a gender direction learned from Russian, which has no articles, align with Spanish and
German's? Is German's frequency entanglement shared?

## Training set

The reviewed Russian list `data/stimuli/phase4_ru_final_v1.csv`: 174 inanimate nouns in *-ь*
(100 f, 74 m), cells = last two letters × loan status (the list's own `cell` column). A list
across all endings adds nothing: outside the *-ь* class gender follows the ending (−а 3,428 f /
12 m, consonant 11 f / 4,540 m in the eligible lexicon), so within-cell variation exists only
inside *-ь* (`docs/decisions.md` 2026-10-25). Audit flags on the list (`results/lang_audit/ru/
phase4_ru_final_v1*`): feminine shorter (d −0.46), fewer tokens (−0.37), less concrete (−0.48);
frequency balanced (+0.09). Covariates handle these, as for German.

Covariates (the Phase 3 design): concreteness (Brysbaert norms via the English gloss, mean-imputed,
plus a missing flag; 11 of 174 missing), Zipf, token count. Models: Qwen3-8B-Base, Qwen3-14B-Base
(1.7B/4B: Russian dropped below 4B, decisions 2026-10-09), LAST position.

## R-A. Alignment (co-primary, four tests)

Per inner layer, the strat probe (Phase 4 `_fit`, residualised on cells + covariates):

- **RU→ES, RU→DE:** the Russian direction scores the Spanish `strat` and German `strat3` m/f
  training nouns; within-cell AUC.
- **ES→RU, DE→RU:** Phase 4 T4 (the Spanish/German directions on the Russian nouns), recomputed
  here with the same bootstrap.

Bootstrap: 1,000 rounds, training and test rows resampled within cell × gender (Phase 4).
Four co-primary tests, Bonferroni: a layer passes if the 98.75% interval's lower bound is above
0.5. A test passes at a majority of inner layers.

- **Aligns:** all four pass.
- **One-way:** both directions *into* Russian pass, or both *out of* Russian pass, but not all four.
- **Partial:** any other combination with at least one pass.
- **No alignment:** none pass.

**Descriptive:** the cross-fitted cosine between the Russian and each other language's f−m
vectors (Phase 4 T2 machinery, pooled design per language pair), reported only at layers where
it lies in [−1, 1].

## R-B. Frequency (the German confidence-control (c) test)

At the four Phase 5 layers: the cross-fitted cosine between the Russian f−m coefficient and the
Zipf coefficient, the within-cell × gender permutation null on the cross term, 200 permutations.
Overlap = cross term beats the null's 95th percentile and |cos| ≥ 0.3. With the outlier-dimension
validity gate (gemma-slim amendment): an invalid layer counts as no overlap; 2+ invalid layers =
no valid test; overlap / no overlap need 3 valid layers; else mixed. (No dimension qualifies as an
outlier on Qwen3, so the fix is a no-op; the gate is a safety check.) Reading: **shared** if
"overlap" on both models, **German-only** if "no overlap" on both, else mixed.

## Concreteness sensitivity (PI, 2026-10-25)

Concreteness is the largest imbalance. Matched f–m pairs within cells, |Δconcreteness| ≤ 0.25 SD
of the list (0.233 on the 1–5 scale), nouns with a missing rating excluded, greedy from the most
abstract feminine noun, nearest match, seeded. Counted before any activation was used: **36 pairs
in 9 cells, mean Δ −0.02**. R-A's RU→ES / RU→DE and R-B are rerun with the Russian direction
fitted on the 72 matched nouns only. Fewer than 20 pairs → "too few". **Robust** if every R-A
and R-B reading is unchanged; otherwise the change is reported, and the matched reading is the
one interpreted for concreteness. (ES→RU / DE→RU are also scored on the matched nouns.)

## Not done

No new list, no extraction, no steering. CPU only. Code `src/gbleed/russian_dir.py`, CLI
`gbleed russian-dir`; outputs `results/russian_dir/`.

## Known limits

- One spelling class (*-ь*): 174 nouns, as many as Spanish (165) or more than German (91), but
  all in one ending family, so the direction is learned from a narrower slice of Russian.
- Glosses for concreteness are English; concreteness is about the concept, not the Russian word.
