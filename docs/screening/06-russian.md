# 06 · Russian *-ь* pool (Phase 4, third language)

*Code: `src/gbleed/phase4_ru.py` and the Russian parts of `lexicon.py`, on branch `russian`
(worktree `../gender-ru`). Pool: `data/stimuli/phase4_ru_pool_v1.csv` (that branch). Review sheet:
`docs/reports/russian-pool-review.md` (PI review 2026-10-08). Status: built, not yet adopted or
known-checked.*

## Why only nouns ending in *-ь*

Russian has three genders, and outside one ending, gender follows spelling almost perfectly:

| ending | feminine | masculine | neuter |
|---|---|---|---|
| *-а/-я* | 1,183 | 1 | 4 |
| consonant | 2 | 1,857 | 3 |
| *-о/-е* | 0 | 2 | 1,040 |
| ***-ь* (soft sign)** | **399** | **97** | 0 |

(clean inanimate nouns, Zipf ≥ 3, before the filters below)

Only the soft sign *-ь* is shared by masculine and feminine (*день* "day" m / *ночь* "night" f),
so it's the only place a spelling-controlled masculine/feminine comparison exists. Neuter has no
spelling-matched partner, so Russian tests the masculine/feminine axis only.

## Lexicon specifics (`lexicon.py`, Russian)

- Headword lines put the transliteration before the gender, so gender is read after it.
- **Grammatical animacy:** Russian grammar marks whether a noun is animate (it changes the
  accusative). Wiktionary records it (*anim*/*inan*), giving an **independent** animacy check
  that German and Spanish lack. It agrees with our English-gloss animacy judge on 81% of nouns.
  Of the 15,152 nouns the gloss judge calls inanimate, 1.6% are grammatically animate and 1.7%
  are tagged both ways (about 3% in all): a useful error estimate for the gloss judge in all
  languages. (An earlier chat figure of "about 5%" was a miscalculation.)
- **Indeclinable** nouns (mostly loans: *кофе* "coffee", *пальто* "coat") are flagged and
  dropped: their gender isn't visible in their forms.
- The stressed headword form (*кни́га*) is not counted as a gendered counterpart (it had made 79%
  of nouns look animate).

## Filters (`build`)

Starting point: single-gender m/f nouns ending in *-ь*, Zipf ≥ 2.0. Excluded if any of:

1. **Grammatically animate**, or **not inanimate by the lexicon judge** (both must say inanimate).
2. Other part of speech; register mark; indeclinable.
3. **Gender-predicting suffixes inside *-ь*:** *-ость/-есть* (abstract nouns, always feminine:
   *радость* "joy"; 742 nouns), sibilant + *ь* (*ночь* "night", *речь* "speech": always
   feminine), *-тель* (instruments and agents, mostly masculine: *двигатель* "engine").
4. **Month names** (all masculine: *декабрь* "December"): a meaning class with one gender.
5. **Proper names:** place-name glosses, and glosses naming one specific thing ("The FIFA World
   Cup": *мундиаль*, PI review).
6. Chemicals; WordNet person/animal (two senses); WordNet groups (*рать* "army"); sex-typed
   garments.
7. **Animate gloss evidence** (species, bird, fish…), **but overridden when Russian grammar marks
   the noun inanimate** (PI review: vanilla, poplar, barley, wormwood and *сеть* "net", glossed
   with "fishing", had been dropped by the English gloss alone).
8. **Formed from another noun:** an etymology template builds it from a lexicon noun
   (*полутень* "penumbra" ← *тень* "shadow"; *взаимосвязь* "interconnection" ← *связь*), as
   German particle + noun formations. Affixes (written with a hyphen, *-ость*) don't count as
   base nouns.

**Loan status** from etymology templates: inherited or derived from Slavic sources = native;
borrowed = loan, **except borrowings from Old Church Slavonic** (*жизнь* "life"), which are old,
native-looking and core vocabulary, so they count as native. No etymology = unknown.

**Cells** = last two letters (consonant + *ь*: *-ль*, *-нь*, *-ть*, *-рь*, *-дь*) × loan status;
only cells with both genders kept. Masculine *-ь* nouns lean towards loans (*стиль* "style",
*шампунь* "shampoo"), which is why loan status is a cell key.

## Result

**245 nouns: 123 masculine, 122 feminine, in 10 cells.** PI review: one item removed by a new
rule (*мундиаль*), five restored by the grammar override.

## Still to do

- **Known check:** Russian has no articles, so frames must use agreement (a gender-marked
  adjective or past-tense verb after the noun). To be designed on grammatical grounds and checked
  on Qwen3 and EuroLLM, like the other frames.
- Limits to state: a script change (Cyrillic) as well as a language change, so a Russian null
  could reflect the script.
