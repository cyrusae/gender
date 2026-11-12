# Is masculine intrusion specific to occupations? (DRAFT pre-registration)

*Status: **ADOPTED 2026-11-11** (PI reviewed the noun list; adoption record at the end). Nothing
scored with the new nouns before adoption.*

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

## Adoption record (2026-11-11, before any score)

- Nouns frozen: `data/wp_persons_es_frozen_v1.csv`, sha256 75dd47c64beb9ce3b4526cb27a7d938a37e3fb984461324955a755ce4f8918ff, built by
  `scripts/wp_persons_freeze.py` from the PI's review (`data/wp_persons_es_candidates_v0_edited.csv`;
  blank = keep, PI), Claude's corrected group suggestions (`wp_persons_es_suggest_fix_v0.csv`), and
  the flags (`wp_persons_es_flags_v0.csv`).
- Groups: person 107, occupation 115, identity 102 (descriptive, PI). Group changes from the PI's
  comments: *arqueólogo, bioquímico* → occupation; *defendido, humano* (PI kept) → person; *empírico,
  judío* → identity. Gloss fix: *extraño* → stranger (PI).
- Drops beyond the PI's: name homographs; identity nouns where either form is a place name (PI);
  *germano* (historical Germanic peoples).
- Comparison group: the new rule-built occupation list (Q3 of the draft); Zhou's 54 not rerun here.
- Sensitivity set: drops every noun with any of `fem_also_noun`, `masc_also_noun`, `language_name`,
  `masc_identical_en`, `place_name` (all rule-built before scoring).
- P2 uses one cue resample per draw for both groups. Script: `scripts/wordpair_persons.py`.

## Results (2026-11-11, run at 5f887d7; `results/wordpair_persons/`)

**P1, intrusion for non-occupation person nouns: none.** −0.005 [−0.077, 0.070] (107 nouns, 87
feminine cues). With a feminine cue, *vecina, amiga, huérfana* are retrieved as well as their
masculine forms (mean RR 0.369 vs 0.364).

**P2, occupations vs persons: occupations carry extra intrusion.** Occupations 0.290 [0.221, 0.357]
(115 nouns, rule-built list; matches the expanded-cue study's 0.291 on Zhou's 54); difference
−0.294 [−0.389, −0.197]. Unchanged without any flagged noun: persons −0.020 [−0.103, 0.063],
occupations 0.321 [0.228, 0.409], difference −0.341 [−0.459, −0.224].

**P3, within nouns (101 with a Glasgow rating, group indicators included):** both track. Frequency
ratio of the forms (rarer feminine → more intrusion) β 0.350 [0.176, 0.500]; Glasgow GEND (more
masculine-rated English noun → more intrusion) β 0.215 [0.026, 0.389].

**Identity nouns (descriptive):** reversed, −0.144 [−0.222, −0.065]; the feminine form wins
(*mexicana, cubana*), plausibly because the feminine adjective form is common (agreement with
feminine nouns such as *la cocina mexicana*), the confound that kept this group descriptive.

**Reading.** The masculine intrusion is not a general property of gendered person nouns: it is
absent for ordinary person nouns and specific to occupations. That rules out the simple
generic-masculine explanation (and the non-blind Gemini notebook's §4 conclusion). Within nouns,
both a frequency imbalance between the forms and the English noun's gender stereotype contribute.
What makes occupations different is not settled here: occupations are the nouns whose feminine
forms are rarest relative to the masculine (as P3's frequency term shows), and that rarity itself
plausibly reflects who has held the jobs, so frequency and stereotype are entangled.

## Remaining conditions (2026-11-11, script e700dff; `results/wordpair_persons/extra_conditions.csv`)

Invariant cues were pre-registered as descriptive but omitted from the first script (Claude's
omission); masculine cues and plain translation (English noun alone, no cue) are exploratory additions
made after the main results. Gap = RR_masc − RR_fem, 95% intervals.

| Condition | Persons | Occupations | Difference |
|---|---|---|---|
| Plain translation (no cue) | 0.28 [0.21, 0.36] | 0.51 [0.45, 0.57] | −0.23 [−0.33, −0.13] |
| Invariant cues (descriptive) | 0.26 [0.20, 0.34] | 0.49 [0.43, 0.55] | −0.23 [−0.32, −0.13] |
| Masculine cues | 0.41 [0.34, 0.48] | 0.60 [0.54, 0.66] | −0.19 [−0.27, −0.10] |
| Feminine cues (primary, above) | −0.005 [−0.08, 0.07] | 0.29 [0.22, 0.36] | −0.29 [−0.39, −0.20] |

Same pattern without flagged nouns. **Reading (exploratory):** the masculine form is the default
translation for person nouns too (*neighbour* → *vecino* over *vecina*, gap 0.28), but about half as
strongly as for occupations (0.51). A feminine cue moves both groups by a similar amount (persons
0.28 → 0.00, occupations 0.51 → 0.29), so what is specific to occupations is the stronger masculine
default, not resistance to the cue: the cue cancels the person-noun default and only partly cancels
the occupational one. Identity nouns: plain 0.33, feminine cue −0.14.
