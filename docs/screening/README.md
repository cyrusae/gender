# How words are chosen and screened

*Reference docs, written 2026-10-08 from the code (not from memory). Each file describes one
system: what goes in, every filter in order, why it exists, what it costs, and known failure
modes. Foreign examples carry English glosses.*

| doc | system | code |
|---|---|---|
| [01-lexicon](01-lexicon.md) | Wiktionary → noun lexicons (German, Spanish, Russian): gender, animacy, register, homographs, frequency | `lexicon.py` |
| [02-known-checks](02-known-checks.md) | Behavioural screening: does the model know each noun's gender? | `phase0.py`, `phase3_known.py`, `framecheck.py` |
| [03-spanish-phase2](03-spanish-phase2.md) | Spanish training and test sets (Phase 2) | `phase2_stimuli.py`, `phase2.finalize` |
| [04-german-phase3](04-german-phase3.md) | German training and test sets, compounds (Phase 3) | `phase3_stimuli.py`, `phase3_known.py` |
| [05-flipped-pairs](05-flipped-pairs.md) | German–Spanish flipped pairs and classics (Phases 4–5) | `lexicon.build_pairs`, `build_classics` |
| [06-russian](06-russian.md) | Russian *-ь* pool (Phase 4, third language) | `phase4_ru.py` (branch `russian`) |
| [07-nonce-words](07-nonce-words.md) | Invented words (Phase 1; Phase 5 plan) | `phase1_stimuli.py`; Wuggy (planned) |

## Principles that apply everywhere

1. **Gold labels come from Wiktionary** (kaikki.org dumps), never from a language model,
   including Claude. Anything hand-written is marked `claude-unverified`.
2. **Drop rather than review.** A filter that over-excludes is preferred to one that needs a
   human to check its output. Small, clean, automatic sets beat large ones needing review. The
   cost is lost data, stated per filter below.
3. **Fix filters on principle, never on results.** A filter is changed only because it is wrong
   about language (a documented failure mode), not because a set came out small or a result
   came out unwelcome; and never by hand-rescuing individual items.
4. **Frozen once used.** A list with results is never edited; a new version is sampled.
   Unused lists can still be corrected (the flipped pairs and Russian pool were, 2026-10-08).
5. **Train/test separation.** Every list has a `split` column; test items are never trained
   on. The Phase 4–5 builders check against *every* Phase 2/3 training set (a gap here caused
   the 2026-10-08 leak; see doc 05).
6. **The PI doesn't read German.** Review sheets show English glosses; Spanish words are shown
   too (the PI reads Spanish).

## Every filter, and where it runs

| filter | what it removes | lexicon | ES Phase 2 | DE Phase 3 | pairs | RU |
|---|---|---|---|---|---|---|
| single gender m/f(/n) | nouns with two genders (*el/la mar*) | tagged | ✓ (test set "multi" separately) | ✓ | ✓ | ✓ |
| animacy (lexicon judge) | people, animals | tagged | ✓ | ✓ | ✓ | ✓ |
| animacy (WordNet 2 senses, gloss) | people/animals the judge missed (*Wachtel* quail) | — | ✓ | ✓ | ✓ (v2) | ✓ (+grammar) |
| groups (WordNet) | *Heer* army, *Rudel* pack | — | ✓ | ✓ | ✓ | ✓ |
| register marks | archaic, slang, regional… | tagged | ✓ | ✓ | ✓ | ✓ |
| other part of speech | *aber* "but", given names | tagged | ✓ | ✓ | ✓ | ✓ |
| inflected form of another word | *Plane* = plural of *Plan* | tagged | ✓ (pool for homographs) | ✓ | ✓ | — |
| English overlap | *Grill*, *machete*, *Stein* | — | ✓ training | ✓ training | ✓ (added 10-08) | — |
| sex-associated concept | garments, anatomy, cosmetics | — | ✓ training | ✓ training | ✓ (added 10-08) | garments |
| proper names | place names, "The FIFA World Cup" | — | ✓ | ✓ | — | ✓ |
| chemicals | *el sodio* (all masculine in Spanish) | — | ✓ | ✓ | — | ✓ |
| gender-predicting endings | Spanish *-ción*, *-dad*…; German *-ung*… | tagged | ✓ (training) | test set | — | ✓ |
| compounds / formations | *Herrenhaus* ← *Haus* | — | — | ✓ | — | ✓ |
| agent nouns | *Bohrer* ← *bohren* | — | (by suffix) | ✓ | — | *-тель* |
| Phase 0 example words | the few-shot examples | — | ✓ | ✓ | — | — |
| held-out test sets | flipped pairs, classics | — | ✗ (the leak) | ✓ | — | — |
| known by the models | nouns the model doesn't know | — | ✓ | ✓ | per model | to build |
