# Where is "zero lean"? An offset-corrected asymmetry test for Zhou et al.'s occupation pairs (DRAFT pre-registration)

*Status: DRAFT, 2026-11-08, written by Claude for the PI's review (expected 2026-11-09). Nothing
adopted; no score in this design has been computed. Candidate reference lists are built
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
   so excluded). Minimum to run: ≥ 20 kept epicenes per gender per language; otherwise the epicene
   anchor is "too few" and the inanimate anchor becomes primary.
2. Primary anchor epicene (recommended) vs inanimate.
3. Run French alongside Spanish (recommended; same design).
