# French replication of the person-noun studies (pre-registration)

*Status: DRAFT, 2026-11-19, written by Claude for the PI's review. Nothing scored with French person
nouns.*

## Why

Three Spanish studies found or tested whether Zhou et al.'s effects are specific to occupations:
the word-pair baseline (`wordpair-persons-prereg.md`: no intrusion for ordinary person nouns, 0.29
for occupations), the person-noun zero point (`persons-zero-point-prereg.md`: inconclusive) and the
person-noun generic masculine (`persons-genmasc-prereg.md`: occupations supported, persons
inconclusive). French was the other language of Zhou et al.'s study and of our occupation results.
This repeats all three in French with the same measures and readings, so the occupation-specific
claim can be stated for both languages or qualified.

## Materials

**Nouns, built entirely by rule** (`scripts/wp_persons_fr_build.py`, output
`data/wp_persons_fr_v0.csv`, sha256 806660ae…; frozen unchanged as `data/wp_persons_fr_frozen_v1.csv`
at adoption). The PI doesn't read French, so no French list is reviewed by hand; instead:
- Candidates: French Wiktionary (kaikki.org) nouns with a masculine singular headword and a form tagged
  feminine, both among the first 50k MUSE French words; one-word English gloss in the first 200k MUSE
  English words (same gloss rule as Spanish).
- **Group labels inherited through English from the PI-reviewed Spanish list:** a French pair gets the
  group (person / occupation / identity) of the Spanish entry with the same English gloss. Glosses not
  in the Spanish list (146 of 238 candidates) are left out; none was ambiguous. So the French groups
  are the same concepts the PI already classified, where French has a distinct feminine form.
- Same drops as Spanish: name homographs (none occurred); identity nouns with a place-name form.
- Same flags, same rules (`fem_also_noun`, `masc_also_noun`, `language_name`, `masc_identical_en`,
  `place_name`), computed on the French dump.
- Result: **person 43, occupation 41, identity 8** (identity descriptive only, as in Spanish).
  Without flagged nouns: person ≈ 28, occupation ≈ 26 (exact counts reported with results).
- Plurals (for the generic-masculine study) from the Wiktionary entries; a feminine plural missing
  from both entries is formed by the regular +s rule. Pairs with all four forms in the first 200k
  words: person 40, occupation 29.

**Cues:** the frozen French cues (`data/wp_cues_frozen_v1.csv`, French primary: 86 gendered pairs,
60 invariant). **Zero point:** frozen French epicenes (`data/zero_ref_fr_frozen_v1.csv`, strict
tier). **Attribute words:** the corrected French lists (duplicates and the name pair removed; the
scripts' default). **Generic-masculine axis:** French definitional pairs without the name pair
*louise/gabriel* and without pairs that reuse a word (the Z19 rule; leaves *femme/homme*,
*fille/garçon*, *elle/il*, *mère/père*, *femelle/mâle*, *sœur/frère*, *tante/oncle*), orthogonalised to
the C1 grammatical-gender direction (`data/c1_nouns_fr_v1.csv`).

**Scripts:** the Spanish scripts, unchanged in method, run with `PERSONS_LANG=fr`
(`wordpair_persons.py`, `wordpair_persons_extra.py`, `persons_zero_point.py`, `persons_genmasc.py`;
outputs in `results/<name>_fr/`). With the default (`es`), `persons_zero_point.py` and
`persons_genmasc.py` reproduce the committed Spanish results byte for byte (checked 2026-11-19); the
word-pair scripts' changes are path substitutions only.

## Questions, measures and readings

Exactly as in the three Spanish preregs, applied to French:
1. **Word-pair baseline** (`wordpair-persons-prereg.md`): P1 intrusion for person nouns; **P2 (primary)
   persons − occupations**; P3 per-noun intrusion on the log frequency ratio of the forms and the
   Glasgow GEND rating of the English gloss; extra conditions (invariant cues pre-registered as
   descriptive; masculine cues and plain translation descriptive). Same readings ("occupations carry
   extra intrusion" if P2's interval is below 0, etc.).
2. **Zero point** (`persons-zero-point-prereg.md`): A per pair for persons, occupations and Zhou's 23
   French occupations; D = persons − occupations; native (primary) and English attribute words.
3. **Generic masculine** (`persons-genmasc-prereg.md`): Δρ for persons (G1, primary), masculine plural
   at neutral (G2), persons − occupations descriptive; **200k vocabulary** (as decided for Spanish).

All with 10,000 bootstrap draws, seed 0, and the same sensitivity set (no flagged nouns). Floors as in
Spanish (zero point 30 pairs per group; generic masculine 25 person pairs); below a floor → descriptive.

**Prediction** (from Spanish): P2 below 0 (occupation-specific); zero point inconclusive; generic
masculine positive Δρ in both groups. **French replicates** if P2's interval is below 0; that is
the one result the cross-language claim rests on.

## Limits

- Group labels are inherited from Spanish through English glosses, so French nouns whose gloss has no
  Spanish counterpart are missing, and a gloss can name a different sense in French (e.g. *dauphin*
  'successor', *mortel* 'human'; both are flagged non-person senses and leave the sensitivity set).
- The French lists are smaller than the Spanish ones (43 / 41 vs 107 / 115); intervals will be wider.
- French occupations have more common-gender forms (*médecin*, *ministre*), which aren't pairs and
  don't enter; the occupation group is the occupations French does inflect.

## Open questions at drafting

1. Adopt as written (inherited labels, no hand review)?
2. Any French words you'd like dropped from `data/wp_persons_fr_v0.csv` (the English glosses are in
   the file)?
