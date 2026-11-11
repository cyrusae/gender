# Is masculine intrusion specific to occupations? (DRAFT pre-registration)

*Status: DRAFT, 2026-11-11, written by Claude for the PI's review. Nothing has been scored with the
new nouns.*

## Why

The expanded-cue study (`wordpair-cues-prereg.md`) showed that with a feminine adjective cue
("buena : ?"), the masculine occupation form is still retrieved better than the feminine one
(intrusion 0.29 [0.18, 0.39]), that the adjective's own stereotype doesn't matter, and that the link
to how male-dominated the job is is borderline (β 0.30, interval touching 0). Two explanations fit:

- **Occupational stereotype:** jobs are coded male, so the masculine form wins for jobs.
- **Generic masculine:** the masculine form is also the generic, citation and more frequent form for
  *every* gendered person noun, so it wins regardless of what the noun denotes.

They predict different things for person nouns that aren't occupations (*vecino/vecina* 'neighbour',
*amigo/amiga* 'friend'): the generic-masculine account predicts the same intrusion as for
occupations; the stereotype account predicts less. The non-blind Gemini walkthrough (section 4) tried
this and found the masculine form winning 23 of 30 times, but 3 of its 10 nouns are male-specific in
English (*brother, son, boy*: the masculine form is the correct answer), two feminine forms are
homographs (*perra* 'bitch', *prima* 'bonus'), and it used 3 cues with no interval. This study redoes
it with rule-built nouns and the 87 frozen feminine cues.

## Questions

1. **P1 (primary).** Is there intrusion for non-occupation person nouns (Spanish, original space)?
2. **P2 (primary).** Is it smaller than for occupations? Difference = intrusion(person) −
   intrusion(occupation), both groups scored with the same cues.
3. **P3.** Within all groups, does intrusion track (a) the frequency ratio of the two forms
   (log rank_f / rank_m) and (b) how male the English noun is rated (Glasgow GEND, where available)?
   The generic-masculine account predicts (a); the stereotype account predicts (b).
4. **P4 (descriptive).** Identity nouns (nationalities, religions, peoples: *mexicano/mexicana*), whose
   forms are also adjectives, reported separately.

## Nouns (`data/wp_persons_es_candidates_v0.csv`, built by `scripts/wp_persons_build.py`)

Rule (Spanish Wiktionary nouns; no hand-picking before review):
- masculine headword with a feminine form, both in the first 50k MUSE words; English gloss one word
  in the first 200k MUSE English words;
- English gloss **not sex-specific**: not in Bolukbasi et al.'s gender-specific list, first Wiktionary
  sense doesn't mention male/man/boy…, doesn't end in *-man* (removes *brother, duke, frenchman*);
- person: the gloss's Wiktionary sense refers to a person, or it's in Wiktionary's People category, or
  Bolukbasi's professions list, or the Spanish entry is in a people/occupation/demonym category;
- groups: *occupation* (Wiktionary occupation category or Zhou's list), *identity* (demonyms,
  nationalities, religions, peoples), *person* (all other person nouns);
- flags: name homograph (dropped, as for the cues), feminine form with its own non-person sense
  (`fem_also_noun`; sensitivity set).

444 candidates. Claude's suggestions (`claude_suggest_unverified`, unverified): occupation 121,
person 112, identity 120, drop 91 (animals, wrong glosses, male titles, slurs, colloquial, names).
**PI review decides** (`pi_keep`, and the group if a suggestion is wrong).

## Fixed design

- Cues: the 87 frozen feminine cues and their masculine partners (`data/wp_cues_frozen_v1.csv`);
  invariant cues reported descriptively.
- Query, scoring and spaces as `wordpair_cues.py`: EN[noun] − EN[cue] + ES[cue form], cosine over all
  50k Spanish words, cue form removed, reciprocal rank (0 outside the top 999). Original space
  primary; Shift Ori descriptive (it only shifts Zhou's occupations).
- Intrusion = mean over feminine cues of (RR_masc − RR_fem), per noun, then averaged per group.
- 95% intervals: 10,000 bootstrap draws (seed 0), resampling nouns within group and cue pairs
  (crossed; the same cue draw for both groups in P2).

## Readings, fixed now

- **P1:** "present" if the person-group interval is above 0.
- **P2:** "occupation-specific excess" if the difference's interval is below 0; "no difference
  shown" otherwise. If P1 is present and P2 shows no difference, the reading is "consistent with the
  generic masculine"; if P2 is below 0, "occupations carry extra intrusion".
- **P3:** standardised slopes of per-noun intrusion on log frequency ratio and on Glasgow GEND (nouns
  with a rating), in one regression with a group indicator; "tracks" if the slope's interval
  excludes 0.
- Sensitivity: P1–P2 without `fem_also_noun` nouns. Floor: fewer than 30 person nouns after review →
  descriptive only.

## What this can and can't show

Occupations and non-occupations differ in more than stereotype (frequency, register, how often the
feminine form occurs at all). P3 addresses frequency directly. A difference in P2 would show something
is specific to occupations, not that it is stereotype. French is not included (the PI can't vet it).

## Open for the PI

1. Review the list (`pi_keep`; correct `group` where wrong).
2. Should identity nouns stay descriptive (their forms are also adjectives, so frequency is driven by
   agreement with other nouns), or join the person group?
3. Occupation comparison group: the new rule-built occupations (larger, ~120) or Zhou's 54 cleaned
   pairs (already analysed)? Suggest the new list as primary, Zhou's as a check.
