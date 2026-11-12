# Where is "zero lean"? An offset-corrected asymmetry test for Zhou et al.'s occupation pairs (DRAFT pre-registration)

*Status: **ADOPTED by the PI 2026-11-10** (drafted 2026-11-08). No score in this design had been
computed at adoption. Candidate reference lists are built
(`data/zero_ref_candidates_{es,fr}_v0.csv`, `scripts/zero_ref_lists.py`) but not frozen.*

## Why

Zhou et al.'s statistic for the occupation pairs is S = |T_m| − |T_f| (reported with an outer
absolute value), with T_m = Σ_i s(m_i), T_f = Σ_i s(f_i) and per-word lean s(w) = w·a,
a = mean(male attribute vectors) − mean(female attribute vectors). On the original Spanish space
T_m = +2.99, T_f = −6.68: feminine forms lean further toward the female words than masculine forms
lean toward the male words. That magnitude asymmetry is what their Figure 1 describes; what is
broken is their permutation null (upstream Z10: the observed value is the maximum of its own
permutation distribution, so p = 0 is guaranteed).

But the number is only meaningful relative to where "zero lean" sits for this yardstick. Adding a
constant c to every word's lean changes S (opposite-sign case) by exactly 2nc:
(T_m + nc) − (−(T_f + nc)) = (T_m + T_f) + 2nc. With n = 58 pairs, a global tilt of
3.69 / 116 ≈ **0.032 per word** toward the female words would produce the whole observed S with no
asymmetry at all (the O8 pattern). C1's per-noun leans are of that order, so the offset can't be
assumed away.

## Question

After locating the yardstick's zero point with sex-neutral reference words, **do feminine
occupation forms lean further from zero than masculine forms (or vice versa)?** Secondary: does
Shift Ori change that offset-corrected asymmetry?

## Materials

- Embeddings, occupation pairs, attributes, Shift Ori: exactly as in `scripts/zhou_shift.py`
  (MUSE `wiki.multi.{es,fr}.vec`, first 50k; ES 58 pairs / FR 23; attributes: ES 8 + 8
  `ES_WEAT_*`, FR the definitional pairs; raw dot products as in their notebook).
- **Reference sets** (built by rule from our Wiktionary lexicons; all exclude the occupation words
  and the attribute/definitional words; Zipf ≥ 3, unmarked, not another part of speech):
  1. **Epicene person nouns (primary anchor):** one grammatical gender, refer to anyone
     (*la víctima, la celebridad, el genio, el personaje*). Rule: single gender; animate in the lexicon; **persons only** (animal-category nouns dropped;
     animals would be a separate, untested anchor); more than half of the English gloss's WordNet
     usage is noun.person (or, absent WordNet, the gloss says person); no gendered counterpart and
     not common-gender; gloss doesn't denote a sex. (The lexicon's animacy flag deliberately errs
     toward 'animate', safe for inanimate lists but not for person lists: *licuado* 'smoothie' was
     animate because WordNet's first sense of *smoothie* is a person.) The rule
     pool is noisy (printer-type senses, proper names, plurals) and misses some core words that
     fail other filters (*persona*: also_pos; *individuo*, *criatura*: WordNet-inanimate), so
     **the PI marks `pi_keep` and may add words the rule missed**; additions and drops are logged.
     Pools: see the CSV counts (rebuilt 2026-11-08 with the person-share rule) before review.
  2. **Common-gender person nouns (check):** one spelling for both sexes (*el/la estudiante,
     periodista, testigo*). Same PI review (the pool also catches two-gender inanimates such as
     *cámara, marca*). Their lean includes occupation stereotypes, so they check the anchor rather
     than define it. Same person-share rule.
  3. **Inanimate nouns (check):** the frozen C1 nouns (50k), frequency-matched masculine/feminine in
     0.25-Zipf bins: ES 1,000 + 1,000, FR 1,455 + 1,455.

**Two tiers (added 2026-11-10, PI discussion; before any score):**
- **Strict** (primary): main sense is a person, routinely used today for either sex, no
  competing sense that dominates usage. Single-sex roles by convention (clergy/monastic titles,
  beauty titles, nobility forms; *apóstol, caudillo, heraldo* as historically single-sex roles) and
  sex-coded mythic beings (*hada, sirena, ninfa*) are excluded.
- **Broad** (sensitivity): strict + PI-accepted mixed-sense person nouns (*autoridad,
  personalidad*) + sex-neutral collectives (*gente, población, clientela*). Reported alongside
  strict; if the two tiers' zero points disagree (CIs don't overlap), say so and keep strict as
  primary.
- A word moved into a person set is removed from the inanimate reference set (*eminencia*);
  C1's frozen list is not edited (the flaw is logged in decisions.md).
- PI additions and drops: `data/zero_ref_additions_{lang}.csv`, merged after review.

**Sensitivity analyses (PI, 2026-11-10: "all sensitivity checks possible"; fixed before scoring).**
Each recomputes every anchor's zero point and A; any verdict that changes is reported as
"depends on <check>":
- S1 drop reference words spelled identically to their English gloss (loan/cognate; ES inanimate:
  23% of m vs 7% of f, a gender-correlated loan confound);
- S2 drop words more frequent in English than in the language (lexicon `zipf_en > zipf`);
- S3 drop the PI's own English/loanword-flagged words (reinstated in v1, see below);
- S4 broad tier instead of strict (above);
- S5 drop reference words that appear in Zhou et al.'s noun lists (`in_zhou_lists`);
- S6 drop the PI-flagged borderline keeps (comments with "?", e.g. *militante, sargento, edil*);
- S7 leave-one-out (reading rule 0).
Monolingual vectors are separate per language, so a shared spelling doesn't merge vectors; the
loan checks target English text inside each language's training corpus.

**PI review record.** Spanish marks: `data/zero_ref_candidates_es_v0_edited.csv` (PI, 2026-11-10).
French: `data/zero_ref_candidates_fr_v1.csv` = the PI's edited v0 with exclusions whose only reason
was "English"/"language" reinstated at the PI's request (covered by S1–S3), other exclusions kept;
French marks rest on English glosses (PI doesn't read French) and can be checked by a French reader.
Additions: `data/zero_ref_additions_es.csv`.

## Measures

- **Zero point** for a reference set R: c_R = ½(mean s over R's masculine words + mean s over R's
  feminine words); for common-gender nouns (one vector each), the plain mean.
- **Offset-corrected asymmetry**, Zhou form: A_R = |T_m − n·c_R| − |T_f − n·c_R| (signed: > 0 means
  masculine forms lean further from zero than feminine forms).
- Per-pair form (descriptive): mean over pairs of |s(m_i) − c_R| − |s(f_i) − c_R|.
- **Inference:** 95% bootstrap CI (10,000 resamples, seed 0) resampling, jointly and
  independently, occupation pairs, reference words (within gender), and the 8 + 8 attribute words
  (a recomputed each draw). No permutation test of S (Z10).
- Reported for each reference set separately, for the original space and Shift Ori; English
  attribute words in the aligned space as a secondary yardstick.

## Reading (fixed before data)

0. **Leave-one-out robustness (PI, 2026-11-10):** for each anchor, recompute the zero point and A
   with each reference word dropped in turn. A verdict in rules 2–3 counts only if it holds in
   every leave-one-out version; otherwise report it as "depends on <word(s)>" with those words
   named. The leave-one-out range of A is reported for every anchor and tier.

1. **Anchors agree** if the epicene and inanimate zero points' 95% CIs overlap. If they don't,
   report each anchor's result and give no single verdict ("zero point not identified").
2. With the epicene anchor (primary): **asymmetry present** if A's 95% CI excludes 0; its sign
   says which forms lean further. **"Offset explains the raw statistic"** if A's CI includes 0
   and |A| < 0.25 × the uncorrected |T_m| − |T_f|.
3. Shift Ori (secondary): by Z14 the shift fixes T_m + T_f; A after Shift is reported with the
   same CI and the same rule, labelled "algebraically constrained" in the write-up.
4. The common-gender check is descriptive: if its zero point differs from the epicene one, say
   so; occupation stereotypes are the expected reason.

## Open decisions for the PI

1. Review `pi_keep` in the epicene and common-gender pools (ES and FR); add missed words
   (candidates the rule missed: *persona, personne, individuo, criatura, ser humano* is multiword
   so excluded). Minimum to run (PI, 2026-11-10): ≥ **15** kept strict-tier epicenes per gender per
   language; otherwise the epicene anchor is "too few" and the inanimate anchor becomes primary.
   Rationale: with the per-word spread seen in C1 (SD ≈ 0.025), the zero-point SE contributes
   ±0.40 to A at 15 vs ±0.36 at 20 (≈12% wider); fragility is handled by the leave-one-out rule.
2. Primary anchor epicene (recommended) vs inanimate.
3. Run French alongside Spanish (recommended; same design).


## Adoption record (2026-11-10, before any score)

PI decisions: primary anchor = strict-tier epicenes (inanimate = check); French runs alongside
Spanish (French marks rest on English glosses); *gangster* dropped; crime nouns (*delincuente,
homicida, criminal…*) excluded from every set (a shared male-tilted stereotype class would bias a
check set) and noted as a follow-up probe (below).

Frozen lists (`scripts/zero_ref_freeze.py`):
- `data/zero_ref_es_frozen_v1.csv` sha256 fec29734937518fb0c1e191553e82b845e5654d38c19a039c0785256b2c6d7ee: strict epicenes 17 m / 18 f; broad adds
  3 epicene + 4 collective; common 104; inanimate 1,000 m / 999 f (*eminencia* moved out).
- `data/zero_ref_fr_frozen_v1.csv` sha256 a13fce8e54655e9c8f73b1c679a6adb740d1b45d4d0b6ec81f8cdb0107a44c1e: strict epicenes 76 m / 25 f; common 105;
  inanimate 1,452 m / 1,452 f.

Implementation details fixed now:
- Vocabulary: MUSE first 50k (as Zhou et al.); reference words outside it, or overlapping the
  occupation/attribute words, are dropped and counted; the floor (15 per gender) is checked after
  this.
- Leave-one-out (rule 0): exact point estimates for every dropped word; the LOO interval is the
  full-sample bootstrap interval shifted by (A_loo − A) (a full bootstrap per dropped word is
  impractical for the 2,000-word inanimate set).
- Sensitivity S1–S6 use 2,000 bootstrap draws (main analysis 10,000), seed 0.
- Yardsticks: native attributes (primary), English attributes in the aligned MUSE space
  (secondary). Spaces: original and Shift Ori (our rebuild, `scripts/zhou_shift.py`).

## Follow-up (not part of this study)

- **Crime nouns as a stereotype probe:** common-gender/epicene crime nouns (*delincuente, criminal,
  homicida, parricida, fratricida, infanticida…*) are grammatically neutral but share one strongly
  male-tilted stereotype; measuring their lean against the zero point would be a gender-ValNorm
  style test of a single stereotype class.

## Results (2026-11-10; script a40cd18, committed before scoring; `results/zhou_zero_point_{es,fr}/`)

A = offset-corrected |T_m| − |T_f| (negative: feminine forms lean further); primary anchor =
strict epicenes (floor met: ES 17/17, FR 74/23 after dropping words outside the 50k vocabulary:
ES *garante, inversionista, nutricionista, oligarca, prestamista, prole, yihadista*; FR
*automobiliste, dupe, gamer, gourmet, guette*). Anchors agree (zero-point CIs overlap) in every cell.

| original space | raw | A, epicene anchor [95% CI] | verdict | robustness |
|---|---|---|---|---|
| ES, Spanish attributes | −3.69 | −1.99 [−5.63, 1.34] | **inconclusive** | LOO and S1–S6 stable; broad tier same |
| ES, English attributes | −4.33 | −3.15 [−6.35, 0.17] | **inconclusive** | **depends on *bebé*, *hada*** (LOO); broad tier: feminine further |
| FR, French attributes | −2.40 | −1.91 [−3.42, −0.30] | **feminine forms lean further** | LOO, S1–S6 stable; inanimate anchor: inconclusive [−4.28, 0.36] |
| FR, English attributes | −2.44 | −2.03 [−3.39, −0.63] | **feminine forms lean further** | LOO, S1–S6 stable; inanimate anchor agrees |

After Shift Ori (algebraically constrained, Z14): inconclusive in all four cells (FR English:
S1/S5 change it).

- The zero-point correction shrinks Zhou's statistic modestly (ES ×0.54–0.73, FR ×0.80–0.83);
  the offset doesn't explain it (|A| > 0.25 × raw everywhere).
- **French: the asymmetry Zhou describes survives** an honest test with both yardsticks.
- **Spanish: inconclusive**, mainly because the interval, which resamples the 8 + 8 attribute
  words (Zhou's permutation test held them fixed), is wide. Direction agrees with French.
- The pre-adoption precision estimate (±0.4) omitted attribute-word resampling; that term
  dominates the interval (ES ≈ ±3.7).


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

**Effect here: the French verdict changes.** FR, French attributes: "feminine forms lean further"
(A −1.91 [−3.42, −0.30]) becomes **inconclusive** (−1.53 [−3.25, 0.41]). With duplicates removed but
the names kept it is also inconclusive (−1.74 [−3.37, 0.05]), so this doesn't depend on the name
rule. Spanish unchanged (inconclusive; −1.99 → −1.90). **Reading now: with native attribute
words Zhou's asymmetry is inconclusive in both languages.** The English-attribute cells don't use the
corrected lists and are unchanged: FR −2.03 [−3.39, −0.63], asymmetry (feminine further); ES −3.15
[−6.35, 0.17], inconclusive, depends on *bebé, hada*. So French still shows the asymmetry when
measured against English gender words.

**List audit (2026-11-11, exploratory, `results/list_audit/`).** With corrected attributes and the
occupation targets restricted to the cleaned list (French drops only *conservateur* 'tory'), French
native reads asymmetry again (−1.91 [−3.34, −0.39]); Spanish stays inconclusive. **French native is
therefore borderline**: it flips with single-word list choices and should be reported that way.
