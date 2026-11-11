# Word-pair test with a full adjective list (DRAFT pre-registration)

*Status: DRAFT, 2026-11-11, written by Claude for the PI's review. Nothing has been scored with the
new cues.*

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
