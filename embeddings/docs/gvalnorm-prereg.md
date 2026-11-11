# Gender ValNorm on Spanish inanimate nouns (pre-registration)

*Status: **ADOPTED by the PI, 2026-11-08** (decisions below). Written by Claude. The English pilot
(`results/gvalnorm_en/`) is a validity check and is done (r = 0.584 vs Lewis & Lupyan's 0.59).
C1's per-noun scores exist (`results/c1_es/c1_per_noun.csv`), but **no score has been compared
with Glasgow ratings**; only the coverage count below was computed.*

## Question

For Spanish inanimate nouns, does an embedding's gender lean follow (a) the noun's **grammatical**
gender, (b) how masculine or feminine **people rate the thing** (Glasgow GEND of its English
translation), or both? And does Zhou et al.'s mitigation remove the grammatical part, the
human-rated part, or neither? Prior art (`docs/gender-valnorm-prior-art.md`): the English
correlation is published (Lewis & Lupyan 2020; Valentini et al. 2023); no gendered-language or
inanimate-noun version was found.

## Materials

- **Nouns:** the frozen C1 list (`data/c1_nouns_es_v1.csv`, sha256 2c58b883…) restricted to nouns
  whose `concept_en` (Wiktionary English gloss, lowercased) is a Glasgow word (homonym senses
  averaged, as in the pilot). Coverage: **547 nouns in the 50k vocabulary (362 m, 185 f; primary)**,
  1,139 in 200k (sensitivity).
- **Human rating:** Glasgow GEND (1 feminine – 7 masculine), z-scored within the analysed set.
  Limitation stated up front: ratings are of the English word, from British raters; no Spanish
  word-gender norms exist (prior-art check).
- **Embedding scores:** s(w) = mean cos(w, male words) − mean cos(w, female words), exactly C1's
  per-noun score, for the same arms (original; Shift Ori as implemented / as specified; Hybrid Ori
  as implemented / as specified) and attribute sets (Spanish WEAT words; English definitional words
  in the aligned space).
- **English anchor:** the same concepts' English words scored in English MUSE with the pilot's
  attributes: how strongly the human rating shows up when no grammatical gender is involved.

## Model and measures

Per arm × attribute set, OLS on standardised variables:

  s_z = β_G · masc + β_H · GEND_z + β_F · zipf_z + ε  (masc = 1 if grammatically masculine)

- **β_H (primary):** does the embedding track human ratings of the thing beyond grammatical
  gender? 95% bootstrap CI over nouns (2,000 resamples, seed 0).
- **β_G:** the grammatical-gender part (C1's gap, now adjusted for GEND and frequency).
- Descriptive: Pearson r(s, GEND) overall and within each grammatical gender; the English anchor's
  r on the same concepts.

## Reading (fixed before data)

- **Tracks human ratings** if β_H's CI excludes 0 and β_H > 0 in the original space with English
  attributes (the attribute set without grammatical gender). Same test reported for Spanish
  attributes.
- **Mitigation removes the human-rated part** if β_H's CI includes 0 after mitigation and
  |β_H| < 0.25 × original; **survives** if the CI excludes 0 and β_H ≥ 0.5 × original;
  otherwise **reduced**. Same rule, separately, for β_G.
- If the English anchor's r on these concepts is itself < 0.2 (the human signal isn't in the
  English embeddings for this noun set), report "no valid test" for β_H.

## Arm added at adoption: Omrani Sabbaghi & Caliskan's erasure

- Space: fastText Common Crawl `cc.es.300.vec` (first 1M words, as their notebook), their
  iterative LinearSVC erasure with the **bugs-fixed** inputs (`scripts/gg_heldout.py --minfix`:
  deduplicated noun lists, two-gender words dropped), applied to every word as in their method.
- Stopping points: **14** iterations (their notebook) and **28** (their tables), both reported
  (upstream O6).
- Attributes: their (= Zhou's) Spanish 8 + 8 words, which are erased too, as in their method. CC
  spaces aren't aligned to English, so no English-attribute version and no English anchor here.
- Nouns: the same Glasgow-matched C1 nouns, restricted to the CC vocabulary; any C1 noun in their
  erasure training pairs is dropped from this arm (counted and reported).
- Same model, measures and reading rule; "original" for this arm = CC before erasure.

## PI decisions (2026-11-08)

1. 50k primary, 200k sensitivity, as C1. (50k = the slice of the MUSE file Zhou et al. load;
   200k = the whole MUSE file. Not a ratio, the two natural boundaries of these files.)
2. Bring in Omrani Sabbaghi & Caliskan's erasure (section above).
3. French: build the French Wiktionary lexicon first; French C1 and French gender ValNorm follow
   the same designs, frozen as addenda before they run.

## Addendum A: French (frozen 2026-11-08, before any French score; PI decision 3)

- Nouns: frozen `embeddings/data/c1_nouns_fr_v1.csv` ∩ Glasgow (concept_en): **813 in 50k
  (474 m, 339 f; primary)**, 1,276 in 200k.
- MUSE arms and attributes as French C1 (Addendum A of `c1-prereg.md`); English anchor as Spanish.
- Omrani Sabbaghi & Caliskan arm: `cc.fr.300.vec`, bugs-fixed inputs, their notebook's French
  setting of **27** iterations (no table-matching count is known for French); French attributes
  as in their notebook.
- Same model and reading rule.

## Results, MUSE arms (2026-11-08; `scripts/gvalnorm_run.py` committed b9cd402 before the run; `results/gvalnorm_{es,fr}/`)

Standardised coefficients, 50k primary. β_H = human rating (Glasgow GEND of the English
translation) beyond grammatical gender and frequency; β_G = grammatical gender.

| | Spanish, Spanish attr. | Spanish, English attr. | French, French attr. | French, English attr. |
|---|---|---|---|---|
| nouns (m/f) | 547 (362/185) | | 813 (474/339) | |
| English anchor r (validity; ≥ 0.2 needed) | 0.581 | | 0.513 | |
| **β_H original** | **0.250** [0.175, 0.321] | **0.337** [0.246, 0.423] | **0.172** [0.102, 0.238] | **0.279** [0.210, 0.353] |
| β_G original | 1.127 | 0.502 | 0.943 | 0.146 |
| β_H Shift Ori as implemented | 0.048 **removed** | 0.337 survives | 0.146 survives | 0.279 survives |
| β_H Shift Ori as specified | 0.023 **removed** | 0.150 reduced | 0.057 reduced | 0.206 survives |
| β_H Hybrid Ori as implemented | 0.048 **removed** | 0.171 survives | 0.146 survives | 0.077 reduced |
| β_H Hybrid Ori as specified | 0.023 **removed** | −0.036 **removed** | 0.057 reduced | −0.023 **removed** |
| β_G, all arms | survives (1.46–1.47) | survives (0.41–0.86) | survives (1.34–1.54) | survives (0.15–0.49) |

- **Embeddings of Spanish and French things track how people rate those things' gender, beyond
  grammatical gender** (β_H > 0 in all four original cells; pre-registered "tracks human ratings").
  The plain correlation r(s, GEND) is 0.20–0.36, below the English 0.58 on the same concepts.
- **Zhou et al.'s mitigation removes or reduces this human-rated (stereotype) component and keeps
  the grammatical one**, which is what their method intends. With the Spanish attributes it's
  removed in every arm; with English attributes it depends on the arm, and only Hybrid as
  specified removes it in both languages.
- **This qualifies C1:** C1's surviving masculine/feminine gap is mostly the grammatical component
  (β_G survives and grows once the human-rated part is gone; coefficients are standardised, so
  "grows" means it explains more of what's left).
- In French the human-rated component is smaller and more resistant: Shift as implemented leaves
  it at 85% (the French attribute words sit in Zhou's French noun list, Z12).
- Limitation: ratings are British judgments of the English translation.

## Addendum B: a yardstick independent of the shifted direction (PI, 2026-11-08; frozen before any B score)

**Why.** Zhou et al.'s Shift moves words only along the orthogonalised semantic direction ds, and
the Spanish/French attribute axis lies mostly along ds (cos 0.82 ES, 0.88 FR; upstream Z14). So
part of what the main analysis reads as "the stereotype component removed" is built in.

**Score.** For each arm and attribute set, a = mean(unit male attribute vectors) − mean(unit
female attribute vectors) is replaced by a⊥ = a − (a·ds)ds, with ds the arm's own orthogonalised
semantic direction (as computed in C1); s⊥(w) = unit(w) · a⊥. Same nouns (Glasgow-matched C1,
50k primary, 200k sensitivity), same model (s⊥_z = β_G·masc + β_H·GEND_z + β_F·zipf_z),
bootstrap and reading rules; MUSE arms only (Omrani's erasure doesn't use ds).

**Questions and readings, fixed now:**
1. **Primary: is there stereotype information off the semantic axis?** In the original space,
   "tracks human ratings off-axis" if β_H(s⊥) has a 95% CI excluding 0 and is > 0, separately for
   each language × attribute set.
2. Mitigation verdicts on s⊥ use the main rule. **Expected by construction, so not counted as
   evidence:** the Shift arms change s⊥ only through renormalising the shifted words, so they
   should leave β_H(s⊥) nearly unchanged. Hybrid arms are the informative ones here (different
   space, debiased-English attributes).
3. Reported descriptively: cos(a, ds) per arm × attribute set, and the share of the original
   β_H that remains on s⊥ (β_H(s⊥) / β_H(s)).

### Addendum B results (2026-11-08; runner committed 20a066e before the run; `results/gvalnorm_{es,fr}/gvalnorm_orth_*.csv`)

β_H on the ds-orthogonal yardstick s⊥ (50k), with the share of the main β_H it keeps:

| | Spanish, Spanish attr. | Spanish, English attr. | French, French attr. | French, English attr. |
|---|---|---|---|---|
| cos(attribute axis, ds), original | 0.82 | 0.77 | 0.88 | 0.64 |
| **β_H off-axis, original** | 0.009 [−0.058, 0.072] **no** | **0.150** [0.074, 0.228] **yes** | 0.038 [−0.010, 0.089] **no** | **0.206** [0.141, 0.274] **yes** |
| share of main β_H off-axis | 4% | 45% | 22% | 74% |
| Shift arms (expected unchanged) | 0.023 | 0.150 survives | 0.057 | 0.206 survives |
| Hybrid arms | 0.023 | −0.035 **removed** | 0.057 | −0.022 **removed** |

- **Primary question:** stereotype information off the semantic axis is present with the English
  yardstick (both languages) and absent with the native-language yardstick (both languages).
  With Spanish/French attribute words, the human-rated component lies entirely along the semantic
  direction, so Zhou's Spanish/French-attribute "removed" results are close to built in (Z14).
- With English attribute words, roughly half (ES) to three quarters (FR) of the stereotype signal
  is off the semantic axis; **Shift cannot touch it** and doesn't (as pre-stated).
- **Hybrid removes it.** Hybrid leaves our nouns' Spanish geometry unchanged (rotation), so the
  removal comes from the English side, i.e. Bolukbasi's hard debiasing of the English attribute
  space, not from Zhou's shift.
- Shift arms with native attributes: cos(a, ds) falls to 0.09 (ES) / 0.68 (FR) because Zhou's
  noun list contains the attribute words, which get projected (Z12); shares of 1.0 in the
  as-specified arms are trivial (ds already removed from the nouns).

**Correction (2026-11-08, after reading the notebook cell by cell, `docs/zhou-shift-identity.md`):**
arm 2 ("as implemented") is **our reconstruction**, not Zhou et al.'s released code: the released
notebook never applies Eq. (2) and loads its Spanish Shift Ori file from outside (Z15). Read arm 2
as "Eq. (2) on Zhou et al.'s own noun list only" and arm 3 as "Eq. (2) on every inanimate noun".
Results are unchanged; only the label's claim about their code is withdrawn.

## Addendum C: fixed English yardstick for the Hybrid arms (frozen 2026-11-08, before any C score)

**Why.** In Hybrid, the English attribute words come from the hard-debiased English space.
Bolukbasi's equalize step rebuilds 16 of our 20 English attribute words (woman/man, she/he, her/his,
mother/father…; all but gal/guy, mary/john) so each pair differs only along the English gender
direction, and neutralizes mary/john. So "Hybrid removes the off-axis signal" (Addendum B) may be
the yardstick moving (the O7/Z12 pattern on the English side), not the nouns changing.

**Test.** Hybrid arms (4a, 4b) re-scored with the **original, un-debiased English MUSE** attribute
vectors (same coordinate system: hard debiasing edits individual vectors, it doesn't rotate the
space). Both the full yardstick a and the ds-orthogonal a⊥ (Addendum B). Same nouns, model,
bootstrap and reading rule; reference = the original arm with original English attributes.

**Reading, fixed now:** if β_H (full, and off-axis) under Hybrid with the fixed yardstick
"survives" relative to the original arm, Hybrid's earlier "removed" was the yardstick moving;
if it is "removed" with the fixed yardstick too, Hybrid changed the nouns.

**Stated caveat (not tested here):** an English yardstick in an aligned space can carry English
stereotypes into the measurement; "present with the English yardstick" may be partly English
stereotype imported by the alignment. This test separates yardstick movement from noun movement,
not Spanish from English stereotype.

### Addendum C results (2026-11-08; runner committed before the run; `results/gvalnorm_{es,fr}/gvalnorm_fixed_en_*.csv`)

English attributes taken from the **original** English space for every arm (50k). Reference = original arm.

| β_H, English yardstick | Spanish off-axis | Spanish full | French off-axis | French full |
|---|---|---|---|---|
| original | 0.150 [0.074, 0.228] | 0.337 | 0.206 [0.141, 0.274] | 0.279 |
| Hybrid, debiased-English yardstick (Addendum B / main) | −0.035 removed | 0.171 | −0.022 removed | 0.077 |
| **Hybrid, fixed original yardstick** | **−0.043 [−0.126, 0.035] reduced*** | 0.182 survives | **−0.004 [−0.068, 0.061] removed** | 0.104 reduced |
| Hybrid as specified, fixed yardstick | −0.044 reduced* | −0.044 removed | −0.004 removed | −0.004 removed |

\* "reduced" only because |−0.043| is 0.29× the original, just above the 0.25 bar; the CI covers 0
and the sign has flipped.

- **The off-axis stereotype signal stays gone with the fixed yardstick.** Hybrid's removal was not
  the English yardstick being rebuilt by equalization; swapping the yardstick back changes the
  off-axis β_H by < 0.02.
- What Hybrid changes is the **cross-lingual** relation: Spanish/French nouns are rotated onto
  hard-debiased English, so their position relative to the (original) English gender words moves.
  Spanish-internal geometry is untouched (rotation), which is why no Spanish-only measure,
  including Table 1, can see it (Z11).
- Caveat from the addendum stands: an English yardstick can import English stereotype through the
  alignment; this test rules out yardstick movement, not that.

## Results, Omrani Sabbaghi & Caliskan erasure arm, Spanish (2026-11-08; `scripts/gvalnorm_omrani.py` committed 64b4b4e before results; `results/gvalnorm_es/gvalnorm_omrani_*`)

CC fastText (`cc.es.300.vec`), bugs-fixed inputs, their Spanish attributes (erased too). Nouns:
Glasgow-matched C1 nouns in the CC vocabulary (1,119), minus 114 in their training pairs →
**460 at 50k**, 1,005 at 200k. Held-out accuracy after erasure 0.519 (14) / 0.526 (28).

| 50k | before | after 14 iter | after 28 iter |
|---|---|---|---|
| β_H (standardised) | 0.133 [0.084, 0.178] | 0.261 **survives** | 0.255 **survives** |
| β_G (standardised) | 1.826 | 1.079 **survives** | 1.045 **survives** |
| raw m − f gap in s (cosine units) | 0.112 | 0.032 | 0.032 |
| SD of s | 0.060 | 0.028 | 0.030 |
| β_H in raw units (β_H × SD of s) | 0.0080 | 0.0072 | 0.0076 |

- Pre-registered verdicts: **both components "survive"**; 200k agrees.
- **Read with the raw numbers:** the erasure removes ~72% of the raw masculine/feminine gap on
  these held-out nouns but leaves a residual 0.03 (Omrani's own SC-WEAT effect sizes go to ≈ 0,
  a per-word normalised score; ours is the raw mean-cosine difference). The stereotype component
  is **unchanged in raw units** (0.0080 → 0.0072–0.0076); it only grows in standardised terms
  because the grammatical variance it sits beside shrinks.
- So the erasure targets grammatical gender and does not touch the human-rated (stereotype)
  component, as designed. The standardised rule's "survives" for β_G overstates what remains;
  note for the write-up that the standardised verdicts are relative to the remaining variance.

## Results, Omrani erasure arm, French (2026-11-08; same script; 27 iterations, Addendum A)

`cc.fr.300.vec`, bugs-fixed inputs, their French attributes. 1,270 Glasgow-matched C1 nouns in the
CC vocabulary minus 317 in their training pairs → **510 at 50k**, 953 at 200k. Held-out accuracy
after erasure 0.463.

| 50k | before | after 27 iter |
|---|---|---|
| β_H (standardised) | 0.122 [0.070, 0.171] | 0.272 **survives** |
| β_G (standardised) | 1.734 | 0.143 [−0.029, 0.327] **removed** |
| raw m − f gap in s | 0.092 | 0.004 |
| SD of s | 0.052 | 0.024 |
| β_H in raw units | 0.0064 | 0.0066 |

- French erasure removes the grammatical gap almost completely (−95% raw; standardised verdict
  "removed"; 200k: β_G 0.28, CI excludes 0, "reduced"), more than Spanish (−72%, a 0.03 residual).
- The stereotype component is again **unchanged in raw units** (0.0064 → 0.0066): Omrani
  Sabbaghi & Caliskan's erasure removes grammatical gender and leaves the human-rated part, in
  both languages.


## Deviation (2026-11-11): corrected attribute lists are now primary

*Post hoc, decided after both results were seen; reported as a deviation with both versions shown.*
Zhou et al.'s attribute words, which this study inherited, contain errors found by an open-ended
blind check (upstream issue Z19): the Spanish male list has *tio*, a rare misspelling (rank 35,247),
for *tío* (rank 3,243); the French list (their definitional pairs) has *femme* and *fille* twice and
the first-name pair *louise/gabriel*. **Corrected lists, primary from now on:** Spanish *tío*;
French de-duplicated with the names dropped (9 male, 7 female words), the same name rule used for the
word-pair cues. Nothing else changed. Reasons: the fix is objective and outcome-blind; the changes
it produces go in both directions across our studies. Original-list results stay in the frozen
result files; corrected results are in `results/attrfix_sensitivity/`.

**Effect here:** main verdicts unchanged. Addendum B/C, French attributes: the off-axis stereotype
part in the original space becomes detectable (β_H 0.038 [−0.010, 0.089] → 0.064 [0.007, 0.123]);
all mitigation verdicts unchanged.
