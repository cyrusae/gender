# 04 · German word lists (Phase 3)

*Code: `src/gbleed/phase3_stimuli.py` (pool, filters, compounds), `phase3_known.py` (known check,
final sets). Pool: `data/stimuli/phase3_pool_v4.csv`; frozen final list: `phase3_final_v3.csv`.
Explainer: `docs/explainers/03-phase3-german-three-genders.md`. PI review sheet:
`docs/reports/phase3-reference-set.md`.*

## The problems these lists solve

German has three genders and much less spelling regularity than Spanish, but it has its own
traps, each of which could fake a result:

| trap | example | handled by |
|---|---|---|
| compounds take the last part's gender | *das Herrenhaus* "manor house" ← *das Haus* | training on single-root nouns only; compounds become a test |
| some suffixes predict gender perfectly | *-ung* (f), *-chen* (n) | suffix nouns out of training; a separate test set |
| neuter is more often borrowed | 44% of neuter vs 28% of masculine nouns are loans | cells by loan status |
| suffixes shared by genders hide a spelling cue | *-nis*: *die Erlaubnis* / *das Ergebnis* | suffix-aware cells |
| derived nouns are masculine by rule, often people | *der Bohrer* "drill/driller" ← *bohren* | agent-noun filter |
| animals and groups slip past animacy tags | *die Wachtel* "quail", *das Heer* "army" | second animacy and group checks |
| place names are all neuter | *Simmerath* | place-name filter |

## Filters, in order (`exclusions`)

Starting point: German lexicon nouns of one gender (m/f/n), inanimate, unmarked, Zipf ≥ 2.0,
not another part of speech. Then a noun is excluded if any of these fire (reasons are kept in
the pool's `excluded` column):

1. **Compound:** any ending of ≥ 3 letters, after ≥ 3 letters, is itself a lexicon noun
   (over-inclusive on purpose: drops ~11,000). **Particle + noun** formations count too
   (*Abwasser* "waste water" ← *Wasser*; particles *ab-, an-, auf-, aus-, vor-, zu-*…).
2. **Proper name:** gloss mentions city, town, village, river, mountain, surname, "in Germany"…
3. **Diminutive** (*-chen*, *-lein*: always neuter) and **Ge-** nouns (*Gebirge*: mostly
   neuter): dropped for every gender, so the filter is symmetric.
4. **Chemical:** gloss (element, amino acid, enzyme, mineral…) or WordNet under chemical element,
   compound, alloy (a neuter-heavy class).
5. **Animate (WordNet):** the English concept's first **two** senses under person, animal or
   people. Two senses because the first sense of *quail* or *sable* is the meat or fur. (A first
   attempt with "organism" removed plants and *Kapitel* "chapter", so it was narrowed.)
6. **Group (WordNet):** first sense under social group, animal group, military unit (*Heer*).
7. **Animate (gloss):** "person who", "agent noun", species, genus, bird, mammal, insect, fish,
   "pack of", herd, flock, swarm, troop, army, crowd, or a Latin binomial "(Genus species)".
8. **Agent noun:** ends in *-er* and stem + *-en* is a German verb (*Bohr-er* ← *bohr-en*).
   Nouns that verbs were made *from* stay (*Leder* "leather" → *ledern*). Known false positives
   (*Faser* "fibre", *Weiher* "pond") only cost data.
9. **Nominalised verb:** ends in *-n* and contains a verb as its ending (*das Auswendiglernen*).
10. **Held out for Phases 4–5:** German nouns of all flipped pairs and classics.
11. **Phase 0 example words.**
12. **Sex-typed garment:** bodice, corset, girdle, petticoat, blouse… (not under WordNet's garment
    roots: *Mieder*).

Training sets additionally drop English-overlapping and sex-associated nouns (doc 03, rules 2–3).
Every noun the PI flagged on review (*Heer*, *Rudel* "pack", *Wachtel*, *Zobel* "sable",
*Bohrer*, *Kehrer* "sweeper", *Rasierer* "razor", *Widerstandskämpfer* "resistance fighter") is
asserted absent: the fixes are rules, so similar nouns are caught too.

## Cells and loan status

- **Cell** = last two letters, except shared suffixes, which get their own cell: *-nis*, *-sal*,
  *-tum*, *-ment*, *-sel* (so a neuter *-nis* noun isn't compared with masculine *Kreis* "circle"
  as if spelling were matched).
- **Loan status** from Wiktionary etymology templates: borrowed = loan; inherited or derived from
  earlier German/Germanic = native; word formations native; no etymology = unknown (its own
  category). Cells are ending × loan status.

## Compounds (a test, and a secondary training set)

- **Head** = the longest ending of ≥ 4 letters that is a single-gender lexicon noun; kept only if
  its gender equals the compound's (as German grammar says it must).
- **First part:** what's left, with a linking element allowed only where it fits the first
  part's gender (*Sonne-n-schein*: *-n-* after feminine *-e*; *Arbeit-s-platz*: *-s-* after m/n).
- **Conflict items:** first part and head of different genders (*der Herr* + *das Haus*). These
  are the sharp test: a gender direction should follow the head.
- One compound per head; heads split 50/50 (seed 0) into `compound_train` and `compound_test`.
- **Head rule:** any compound whose head is a single-root training noun is dropped (it would
  re-measure that noun).

## Known check and final sets

Known by Qwen3-1.7B and 4B on the three-way frames (doc 02). Final (`phase3_final_v3.csv`):

| set | role | count |
|---|---|---|
| `strat3` | **training (primary)**: every known noun in ending × loan cells containing all three genders | 64 m / 27 f / 31 n |
| `matched3` | training (strict): equal m/f/n per cell | 14 each |
| `matched3_end` | training (cells by ending only, loan as covariate) | 51 |
| `matched2` | training (m–f axis) | 30 each |
| `compound_train` | training (secondary) | 33 / 14 / 9 |
| `compound_test` | **test** | 720 (219 m/f conflict items) |
| `suffix` | test: feminine-suffix nouns (*-ung*, *-heit*, *-tät*…) | 78 f / 3 m |
| `multi` | test: *der/die See*-type words | 7 |

## Known issues

- Animacy needed three layers (lexicon, WordNet, gloss) plus a human read; a limitation worth
  stating.
- The animacy rules over-exclude (WordNet sense order; plant Latin names). Fixed for *unused*
  lists on 2026-10-08 (doc 05); this frozen list keeps the older rule.
- One compound-test item is a proper name: *Bundesfreiwilligendienst* ("the German Federal
  Volunteers Service"). Frozen; excluded in sensitivity analyses.
