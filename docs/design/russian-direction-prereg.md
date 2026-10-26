# Russian gender direction: does it align with Spanish/German, and does it share German's frequency lean? Pre-registration (ADOPTED, PI 2026-10-25)

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

## Implementation note (before any run, 2026-10-25)

Concreteness is computed as in Phase 3 (`phase4._conc`: Brysbaert rating of the Wiktionary
gloss, falling back to the concept), for the covariate and the matching alike. The pair count
above was made with the concept only; with the Phase 3 rule there are **8 unrated nouns** (not
11) and **37 matched pairs in 8 cells, mean Δ −0.01** (not 36 in 9). Counted before any
activation was used; rule unchanged. Code `src/gbleed/russian_dir.py`, tests
`tests/test_russian_dir.py`. Seed 25.

## Results (2026-10-25)

`gbleed russian-dir` at commit ddcf1f8; `results/russian_dir/` (`readings.json`, `r_a_layers.csv`,
`r_b_layers.csv`). Matched set: 37 pairs (74 nouns), mean Δconcreteness −0.01.

**R-A, layers passing (Bonferroni 98.75% lower bound > 0.5; majority = 18/35 for 8B, 20/39 for 14B):**

| | RU→ES | RU→DE | ES→RU | DE→RU | reading |
|---|---|---|---|---|---|
| Qwen3-8B, full list | 29 | 25 | 30 | **9** | one-way (out of Russian) |
| Qwen3-8B, matched | 14 | 8 | 26 | 16 | partial |
| Qwen3-14B, full list | 33 | **18** | 34 | 28 | one-way (into Russian) |
| Qwen3-14B, matched | 21 | 12 | 26 | 13 | partial |

Median point AUCs (full list): 8B 0.88 / 0.84 / 0.84 / 0.66; 14B 0.91 / 0.75 / 0.86 / 0.74.
Descriptive cross-fitted cosines (all layers valid): RU–ES median 0.74 (8B), 0.65 (14B); RU–DE
0.53, 0.44.

**R-B (frequency):** full list **no overlap** on both models (cos −0.08 to −0.25) → **German-only**.
Matched: 8B mixed (overlap at 2/4 layers, cos −0.49, −0.33), 14B no overlap → mixed across models.

**Concreteness sensitivity: not robust** (R-A one-way → partial on both models; R-B German-only →
mixed). Per the prereg the matched reading is the one interpreted for concreteness. Note for
interpretation (not a reading): the drop is concentrated in the tests that *fit* a direction on
the 74 matched Russian nouns (RU→ES/DE); ES→RU, whose direction is fitted on the full Spanish
set and only scored on the matched nouns, holds (30→26, 34→26). Smaller-sample fitting and
concreteness confounding are not separated by this design.

**Summary.** On the full list, a Russian gender direction learned without articles scores
Spanish nouns by gender at most layers and is scored by the Spanish direction at most layers;
German is the weaker partner in both models (one direction misses the majority each time).
Spanish–Russian alignment is the strongest pair. Russian does not share German's frequency
lean. Neither statement survives the concreteness-matched refit cleanly, so both carry a
concreteness caveat.

## Follow-up S: size-matched control (DRAFT for PI sign-off, 2026-10-25; not run)

**Question.** The concreteness-matched refit (74 nouns) lost passing layers mainly in the tests
that fit a direction on Russian (RU→ES 29→14 / 33→21, RU→DE 25→8 / 18→12). Is that drop what any
74-noun Russian training set gives (sample size), or specific to concreteness matching?

**Design.** For each model (Qwen3-8B, 14B): draw 200 random subsets of the 174-noun list, each
with exactly the matched set's composition (the same number of f and m nouns in each of the same
8 cells: 37 f + 37 m), seed 26. For each subset refit the Russian direction (same probe,
covariates, LAST) and compute the **point** within-cell AUCs RU→ES and RU→DE at every inner layer
(no bootstrap per subset, for cost), and their mean over inner layers.

**Statistic.** For RU→ES and RU→DE separately: the matched set's layer-mean point AUC (from the
main run's `r_a_layers.csv`, `set = matched`) against the distribution of the 200 random subsets.

**Readings (per model, per test):**
- **Size explains it:** the matched AUC lies inside the random subsets' central 95%.
- **Concreteness matters:** the matched AUC is below the random subsets' 2.5th percentile.
- **Matched is better than random:** above the 97.5th percentile (reported; would mean matching helps).

Overall: "size explains the drop" if that holds for both tests on both models; "concreteness
matters" if it holds for at least one test on both models; otherwise mixed. ES→RU / DE→RU and
R-B are not repeated (their directions don't depend on the Russian sample size, or need the
permutation machinery per subset).

**Cost.** CPU only, about 200 × layers probe fits per model (minutes to an hour). Code would be
`russian_dir.size_control`; output `results/russian_dir/size_control.csv`.
