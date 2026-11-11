# Word-pair test with a full adjective list (DRAFT pre-registration)

*Status: **ADOPTED 2026-11-11** (PI reviewed the cue lists; adoption record at the end). Nothing
scored with the new cues before adoption.*

## Why

Zhou et al.'s Table 2 test ("good : doctor = buena : ?") uses seven adjective cues, none of them
feminine (Z17). Our rerun added Zhou's own adjectives in feminine form, but that is only three
feminine cues in Spanish (*trabajadora, buena, mala*), and the main quantity, **intrusion** (the
masculine occupation form retrieved better than the feminine one even when the cue is feminine),
came out 0.236 [−0.011, 0.449] (`wordpair-mechanism-prereg.md` Part A). The cue side is the
fragile part. Nothing in the test needs only seven adjectives: any adjective that can describe a
person and has distinct masculine and feminine forms works.

## Questions

1. **Q1 (primary).** With a full, rule-built adjective list, is there intrusion in the original
   Spanish space?
2. **Q2.** Do Zhou's mitigations (Shift Ori, Hybrid Ori) change intrusion, the agreement gap, or
   the invariant-cue gap? (Rerun of Part A with more cues.)
3. **Q3.** Does the adjective's own gender stereotype matter? If "aggressive : doctor =
   agresiva : ?" pulls toward *médico* more than "gentle : doctor = tierna : ?", intrusion should
   rise with how masculine the adjective is rated.
4. **Q4.** Does the occupation-stereotype result from the mechanism study (US labor share β 0.30)
   hold with the larger cue set?

## Cues (`data/wp_cues_{es,fr}_candidates_v0.csv`, built by `scripts/wp_cues_build.py`)

Rule (no hand-picking before review):
- Wiktionary (kaikki.org) adjectives; masculine singular headword; feminine singular = the form
  tagged "feminine". **Gendered** if it differs (*alto/alta*), **invariant** if there is no separate
  feminine (*feliz*).
- English gloss from the first or second sense must be on a published list of person-describing
  adjectives (as collected by Garg et al. 2018: Williams & Best's Adjective Check List, the
  Katz–Braly/Princeton traits, and Garg's appearance, intelligence and sensitivity lists) and in
  the first 200k MUSE English words. The English gloss is the cue's English side, so it must be a
  real translation.
- Both forms in the first 50k MUSE words (the rebuilt spaces' size).
- Excluded: senses tagged colloquial, slang, figurative, regional and similar; gender-defining
  words (*femenino, masculino*); Zhou's occupation forms.
- Candidates: ES 130 gendered + 86 invariant; FR 111 + 78. Claude's keep/drop suggestions (mostly
  wrong-sense glosses, e.g. *nacido* = "born" glossed "natural") are in
  `claude_suggest_unverified`; after them, ES 104 + 68, FR 87 + 62. **PI review decides** (`pi_keep`).

Each gendered pair gives one masculine cue and one feminine cue; invariant adjectives give one cue
each. Zhou's seven cues are not on the new list (most fail the person-list rule); they are reported
separately as a comparison set, not pooled.

## Fixed design (unchanged from the mechanism study)

- Occupations: cleaned list (ES 54; FR 22, descriptive), `data/occupation_mapping_v1.csv`.
- Query: EN[occupation] − EN[cue] + T[cue form]; rank all target words by cosine, cue form
  removed; a form outside the top 999 has reciprocal rank 0. Spaces: Original, Shift Ori, Hybrid
  Ori (our rebuilds).
- Intrusion = mean over feminine cues of (RR_masc − RR_fem); agreement gap = mean RR_masc over
  masculine cues − mean RR_fem over feminine cues; invariant gap = mean over invariant cues of
  (RR_masc − RR_fem).
- 95% intervals: 10,000 bootstrap draws (seed 0), resampling occupations and, independently, cues
  within cue type (crossed).

## Tests and readings, fixed now

- **Q1:** intrusion "present" if its interval is above 0 in ES original.
- **Q2:** a mitigation "reduces" a gap if the interval of the change is below 0.
- **Q3:** per-cue intrusion (mean over occupations) regressed on the adjective's stereotype score;
  primary score Williams & Best 1990 (the 1990 value in Garg's file runs 0–100 and rises with
  masculine association: *strong* 92, *aggressive* 88, *gentle* 21, *affectionate* 10; it covers
  79 of the 104 suggested Spanish gendered pairs), Glasgow GEND secondary. "Cue stereotype matters" if the slope's interval
  excludes 0. Bootstrap over cues and occupations.
- **Q4:** per-occupation intrusion regressed on D + R + H + S_BLS as in the mechanism study.
  "Replicates" if S_BLS's interval is above 0.
- **Floor:** if fewer than 20 gendered Spanish pairs survive review, Spanish is reported as
  descriptive. French is descriptive throughout (the PI can't vet it).

## What this can and can't confirm

The cues are new data, so Q1 and Q3 are confirmatory for Spanish. The occupations and spaces are
the ones already analysed, so Q2 and Q4 are replications with a better cue sample, not new
confirmations.

## Open for the PI

1. Review the Spanish list (`pi_keep`). French: accept the rule plus my suggestions as is, or skip
   French?
2. Williams & Best as the primary cue-stereotype score, Glasgow secondary: OK?
3. Allow second-sense glosses (marked in `sense`), or first sense only (ES then 82 gendered)?

## PI decisions so far (2026-11-11)

- Stereotype scores: Williams & Best 1990 primary, Glasgow GEND secondary.
- Added cues: *brillante* (brilliant). *sincero* was already a candidate. *estúpido* and
  *discapacitado* can't be scored: their feminine forms are outside the first 50k words
  (*discapacitada* is rank 163,349, though the plural *discapacitados* is 24,752).
- **Name rule:** a cue is dropped if either form is a Wiktionary given name, because MUSE vectors
  are lowercased (*bárbara* carries *Bárbara*, *franco* carries *Franco*). Spanish: drops
  *bárbaro, franco, modesto, cándido, generoso, bonito, cuco* (and *alejo, pacho*). French needs
  another name source (the French dump yields none under the same filter).
- Feminine forms that are also nouns (`fem_also_noun`, ES 65 of 216): kept; the primary tests are
  rerun without them as a sensitivity check.
- Disability/illness subset (*sordo, mudo, ciego, enfermo, loco*; *sano* as contrast), reported
  descriptively, not pooled.
- French: same rule; the MUSE es-fr dictionary match (`es_match`) is reported as a sanity signal.
- Merging at freeze is by (lang, masculine form), never by row order; the PI's candidate file is
  only read, never rewritten.

## Adoption record (2026-11-11, before any score)

- Cues frozen: `data/wp_cues_frozen_v1.csv`, sha256
  4617e638fdb570c3e51d384af329a4fa7082fc45cb71ea84cd7d2fcfa99cda0e. Built from the PI's review
  (`data/wp_cues_{es,fr}_candidates_v0_edited.csv`; Spanish marked true/false with notes, French
  blanks = accept the rule and Claude's suggestions, *queer, sexy, spirituel* dropped), the name
  rule, and the PI's further drops (*abandonado, paciente*). Gloss corrections for kept words whose
  matched sense was wrong: *primitivo* → primitive, *típico* → typical.
- Spanish primary: 87 gendered pairs + 67 invariant (incl. *brillante*); 39 pairs have a feminine
  form that is also a noun (sensitivity set). Disability subset: *sordo, mudo, ciego, enfermo,
  loco, sano* (descriptive). French: 86 + 60, descriptive.
- Second-sense glosses are kept where the PI marked the word true (the per-word review replaces the
  blanket first-sense question).
- Bootstrap resamples gendered pairs (masculine and feminine cue together), invariant cues and
  occupations independently. Q3 slope is standardised; prediction is a positive slope (more
  masculine adjective → more intrusion), but the reading is two-sided.
- Script: `scripts/wordpair_cues.py`.

## Results (2026-11-11, run at 8ccf23c; `results/wordpair_cues/`)

**Q1, intrusion with the full list (Spanish, original): present.** 0.291 [0.181, 0.395] with 87
feminine cues, against 0.236 [−0.011, 0.449] with Zhou's 3. Unchanged without the 18 pairs whose
feminine form is also a noun (0.288 [0.181, 0.394]). Agreement gap 0.391 [0.300, 0.475] and
invariant gap 0.484 [0.389, 0.571] also present.

**Q2, mitigations: all three gaps shrink but none closes.** Shift Ori: intrusion −0.109
[−0.150, −0.070], agreement −0.063, invariant −0.084; Hybrid Ori: −0.095 [−0.155, −0.040],
−0.042, −0.038. With Zhou's 7 cues only the invariant gap's reduction was detectable; the larger
cue set shows a reduction of about a third in intrusion, with intrusion still present after both
(Shift 0.182 [0.066, 0.296]).

**Q3, the adjective's own stereotype: not shown.** Standardised slope of per-pair intrusion on
Williams & Best −0.145 [−0.390, 0.144] (64 pairs); Glasgow −0.097 [−0.361, 0.198]. If anything the
sign is opposite to the prediction. Masculine-rated adjectives (*agresiva, fuerte*) don't pull
toward the masculine occupation more than feminine-rated ones.

**Q4, occupation stereotype: not replicated by the pre-registered rule.** S_BLS 0.302 [−0.007,
0.524]: the same estimate as the mechanism study (0.297) but the interval now just touches 0.
Frequency R 0.318 [−0.009, 0.595] likewise; dictionary D 0.055 [−0.200, 0.289] null again. Only the
cue side changed; the 54 occupations still limit this regression.

**Disability subset (descriptive):** intrusion 0.18–0.38 per cue (*ciego* 0.18, *loco* 0.21,
*sordo* 0.24, *enfermo* 0.26, *sano* 0.34, *mudo* 0.38), within the range of the main list; the
contrast word *sano* is no lower. No sign of a disability-specific effect at this size.

**French (descriptive):** intrusion 0.191 [0.007, 0.371]; after Shift and Hybrid inconclusive.

**Reading.** The intrusion is real and robust: with a feminine adjective, the masculine occupation
form is still retrieved better. It doesn't depend on which adjective is used, including its gender
stereotype; it is a property of the occupation pairs, not the cues. Zhou's mitigations reduce it by
about a third. Whether it tracks occupational stereotype is now borderline (same size, interval
touching 0).
