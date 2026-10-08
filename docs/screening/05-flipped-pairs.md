# 05 · Flipped German–Spanish pairs and classics (Phases 4–5)

*Code: `lexicon.build_pairs`, `lexicon.build_classics`. Files: `data/lexicon/pairs_de_es.csv`,
`data/stimuli/classics.csv`. Review: `docs/reports/flipped-pairs-review.md` (PI review, all 519
rows read 2026-10-08), every drop and its reason: `docs/reports/flipped-pairs-flags.csv`.*

## What they're for

A **flipped pair** is one concept with opposite grammatical genders in German and Spanish:
*die Brücke* / *el puente* "bridge", *der Mond* / *la luna* "moon". They test the project's
null directly: a shared *grammatical* direction should follow each language's gender, while a
*meaning* direction should score both members alike. Phase 4 uses them to test sharing across
languages; Phase 5 uses them for the baseline behaviour test.

## How pairs are built (`build_pairs`)

1. Take each language's `eligible()` nouns with Zipf ≥ 2.5 (doc 01).
2. Match German and Spanish nouns on their **normalised English gloss** ("bridge"). For each
   gloss, the most frequent noun in each language is taken.
3. **Flipped** = the two genders differ (519 of 2,217 pairs).
4. Flags (not filters) computed per pair:
   - `strict`: needs no checking: each word is ≥ 10× more frequent than any other noun with that
     gloss in its language (*Zeit*/*vez* "time" fails: *tiempo* is also "time"), the two are within
     10× of each other in frequency, not cognates, not English words. 250 of the 519.
   - `cognate`: German and Spanish spellings ≥ 80% similar (4 of the 519).
   - `en_same`: a word ≈ the English gloss (≥ 90% similar: *Machete*, *Grill*).
   - `en_homograph`: a word is a common English word (Zipf ≥ 3: *Stein*).
   - `concrete`: the concept's dominant WordNet sense is a physical thing.

Gloss matching is ~10% wrong even under strict rules (CLAUDE.md), because English glosses are
ambiguous: *flat* = apartment (*apartamento*) or ice floe (*Scholle*). Hence the review.

## From 519 to 389 (2026-10-08)

| dropped for | pairs | why |
|---|---|---|
| **PI review** | 51 | glosses name different things or senses (temple building vs temple of the head; capital, volume, spring, drill, scooter…) |
| **training leak** | 37 | the Spanish noun is in Phase 2 *training* (doc 03); a direction would score it from memory. Also the classic *llave* |
| **English overlap** | 20 | both words *are* the English word (4: *Machete/machete*; ambiguous as bare nouns), or a word is also an unrelated English word (16: *Brief* "letter" / English "brief", *Welt*, *Angst*). The builder flagged these but never applied a drop. **English cognates** (63: *restaurante*, *púlpito*, *Veranda*) are **kept** in the primary analysis, with a pre-registered sensitivity analysis without them (PI, after a devil's-advocate review: dropping them cost power and balance, and their effect, likely a weaker gender signal, can be measured) |
| **animacy / groups** (rule v2) | 22 | army, troop, guild, synod, faction, entourage; *Betrügerin* "female fraudster" (a person, caught by the PI); some organisations (railway, theocracy) |
| **sex-associated** | 7 | diaper, miniskirt, underpants, clitoris, breast, pin (jewellery); same rule as training sets |
| **distinct total** | **130** | a pair dropped for several reasons is counted once |
| **remaining** | **389** | 239 German-f / Spanish-m, 150 German-m / Spanish-f (63 English cognates among them); before per-model known checks |

### Animacy rule v2 (for unused lists, 2026-10-08)

The Phase 3 rules over-excluded on the pairs (PI noticed *dishwasher*, *cucumber*). Two
documented failure modes were fixed on principle; the group rule was left as is:
1. **Plant taxonomy:** "species", "genus" or a Latin name counts as animate only if WordNet also
   classes the concept as a person or animal (plants are inanimate, as decided in Phase 3).
2. **WordNet sense order:** only the *first* sense counts, unless the gloss mentions meat or fur
   (the quail/sable case the two-sense check was built for). WordNet lists "a laborer who washes
   dishes" before the machine.

This restored pumpkin, cucumber, fir, valerian, dishwasher, slipper, world, talent, patronage,
cancer (and drill, which the PI's review drops anyway). A more aggressive version (WordNet groups
only when the glosses agree) was tried and rejected: it readmitted real groups (audience, guild,
synod) and tuning it further would have meant fitting the rule to the items it judges.

## Classics (`build_classics`)

19 hand-picked famous pairs (*Brücke/puente*, *Mond/luna*, *Sonne/sol*, *Schlüssel/llave*,
*Apfel/manzana*, *Löffel/cuchara*…) from
`classics_spec.csv`, with genders looked up in Wiktionary and a `check` column for anything
weakening an item (two genders, not actually flipped, an animate sense, a homograph). Kept either
way, as the one hand-picked exception to "drop rather than review"; reported with and without
flagged items (*luna*, *estrella* are also women's names).

## Still to do before Phase 4 uses them

- Per-model known checks on both members (shared set across models compared).
- The Phase 4–5 builder asserts no member appears in any Phase 2/3 training set.
- Reported limits: pairs differ in form, frequency and tokenisation as well as gender; the
  concept is controlled, those aren't.
