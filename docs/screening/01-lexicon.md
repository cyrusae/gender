# 01 · The lexicons: from Wiktionary to a table of nouns

*Code: `src/gbleed/lexicon.py` (`parse_dump`, `build_lexicon`, `judge_animacy`, `eligible`).
Command: `uv run gbleed lexicon [--langs de es ru]`. Output: `data/lexicon/{de,es,ru}_nouns.csv`
(gitignored; rebuilt in ~40 s per language), with the dump's URL, date and SHA-256 in
`data/lexicon/metadata.json`.*

## Source

The English Wiktionary, as machine-readable dumps from **kaikki.org** (one JSON record per entry;
German, Spanish, Russian ~0.9–1.1 GB each). Every gender label in the project comes from here.
This is the project's answer to "where do the gold labels come from": a human-edited dictionary,
not a model.

## Step 1: parsing (one pass over the dump)

For each **noun** entry whose headword passes a spelling pattern:

| language | pattern | effect |
|---|---|---|
| German | capitalised letters only (`^[A-ZÄÖÜ][a-zäöüß]+$`) | no hyphenated compounds, no abbreviations |
| Spanish | lowercase letters only | no proper names, no multi-word entries |
| Russian | lowercase Cyrillic | page titles are unstressed (*книга*, not *кни́га*) |

- **Senses that aren't real uses are skipped:** form-of, alternative spelling, abbreviation,
  misspelling, plural, plural-only, clipping. An entry with no real sense left is recorded as an
  inflected form of another noun (*Plane* = plural of *Plan* "plan").
- **Gender** comes from the sense tags (masculine/feminine/neuter), or, if the senses carry none,
  from the headword line (*Brücke f*). Russian puts the transliteration in parentheses *before*
  the gender (*соба́ка • (sobáka) f anim*), so its headword is parsed after it.
- **Other parts of speech** with the same spelling are recorded (*aber* "but" is a conjunction;
  given names too: *Charlotte*). Used below to drop homographs.

## Step 2: one row per noun

| column | meaning | how |
|---|---|---|
| `gender` | m / f / n, or `multi` | all genders across entries; more than one = `multi` |
| `concept_en` | short English concept | first gloss, parentheses and articles stripped: "bridge (structure…)" → "bridge" |
| `zipf` | frequency (log scale; 3 = once per million words, 4 = ten…) | `wordfreq`. Caveat: it lowercases, so *Aber* gets *aber*'s count |
| `zipf_en` | frequency of the same string in English | catches loans and false friends |
| `animacy` | animate / inanimate / uncertain | see below |
| `marked` | register marks | archaic, dated, rare, slang, vulgar, regional, colloquial, poetic, literary, childish… plus letter names and numbers |
| `regions` | regional tags | Austria, Mexico, Latin America… |
| `also_pos` | other words with this spelling | conjunction, verb, given name… |
| `also_form` | it's also an inflected form of another word | *camino* "path" = "I walk" (Phase 2's homograph pool) |
| `concrete` | dominant WordNet sense is a physical thing | usage-weighted: *table* is furniture, not a data table |
| `de_suffix` | German gender-predicting suffix | *-ung*, *-heit*, *-keit*, *-schaft*, *-ion*, *-tät*, *-enz*, *-anz*, *-ik*, *-ei*, *-ur* (f); *-ismus*, *-ling* (m) |
| `es_ending`, `es_regular`, `es_exception` | Spanish *-o/-a* pattern | regular (*-o* m, *-a* f) or exception (*el día*, *la mano*); Greek *-ma* and clippings (*la foto*) recognised |
| `initial_a_f` | feminine noun starting with stressed *a-*/*ha-* | these take *el* (*el agua* "water"), which breaks article tests |
| `ru_ending`, `ru_animacy`, `ru_indecl` | Russian ending class, grammatical animacy, indeclinable | see doc 06 |

## Animacy: the lexicon's judge

A noun is **animate** if *any* of these fire (`judge_animacy`):
1. **Gendered counterpart:** Wiktionary lists a different-gender form (*director/directora*), or a
   sense is "female/male equivalent of". (For Russian, the stressed headword form is ignored;
   it had made 79% of nouns look animate before that fix.)
2. Wiktionary tags it as an agent noun, or "by personal gender".
3. Its first sense is in a people category (occupations, family, nationalities, given names…) or
   an animal category (mammals, birds, fish…).
4. The gloss starts like a person or animal ("a person who…", "woman", "bird"…).
5. WordNet's main sense of the English concept is a person or an animal.

Otherwise it's **inanimate** if WordNet knows the concept and less than 25% of its tagged usage
is animate; **uncertain** if WordNet doesn't know it or the share is higher. Only *inanimate*
nouns are ever used. Known gap: the judge misses some animals and groups (*Wachtel* quail,
*Heer* army), which is why each word list adds a second animacy check (docs 03–06).

## `eligible`: the shared starting point

A noun is eligible for any stimulus list if it is: single-gender m or f; inanimate; no register
mark; no regional tag; not another part of speech; not an inflected form of another word; not
more frequent in English than in its own language; and, in Spanish, not a feminine
*a-*/*ha-* noun. Each phase then adds its own filters on top.

## Limits to state

- Wiktionary's coverage and tagging vary by entry; rare nouns are thinner.
- English glosses are the bridge to WordNet, so animacy and concreteness are judged on the
  English concept, not the German or Spanish word: senses can be mismatched (doc 05).
- Frequencies come from mixed web text and are corrupted by homographs.
