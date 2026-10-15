# P23/P24 explained: does the grammatical gender direction point at "woman"?

Exploratory side study of Phase 5 (design: `docs/design/phase5-design.md` P23, P24, P24b; code:
`src/gbleed/english.py`). Laptop numbers below are development results (Mac, fp16, LAST
position, Qwen3 1.7B/4B; word list v3).

## The question

Phases 2–4 found a direction in the model's activations that sorts Spanish (or German) nouns by
grammatical gender. English has no grammatical gender on nouns but plenty of *social* gender
(*woman*/*man*, *queen*/*king*). If the model reused one direction for both, then:

- **P24, grammatical → social:** the Spanish or German direction should sort English person
  words by sex, and
- **§5.1 geometry:** the grammatical direction and a social direction (built from *woman − man*
  and similar pairs) should point the same way more than chance allows.

P23 asks the opposite way round: does the social direction line up with human ratings of how
"feminine" or "masculine" an English *object* word feels (Glasgow norms)?

## The tools

- **Social direction:** the average of nine female − male word-pair differences (*woman/man*,
  *girl/boy*, *queen/king*, *she/he*, …). Each pair is checked against its WordNet gloss.
- **Grammatical directions (ES, DE, POOLED):** the same Phase 4 directions, fitted on inanimate
  nouns only, within ending cells, so they can't have learned "person-ness" or spelling.
- **Person-word AUC:** project each English person word onto a direction. AUC = how often a
  random female word scores above a random male word (0.5 = chance, 1 = perfect sorting).
- **Three person-word sets:**
  - **v1** (35 words, WordNet rule): includes words whose *form* gives sex away (*waitress*,
    *salesman*, *ballerina*);
  - **unmarked v1** (20): those dropped;
  - **person2, the main set** (14 f / 13 m after the PI dropped *whore*, *jackass* and
    *prostitute*): unmarked words from two human-made sources (GN-GloVe's
    gendered word lists, Zhao et al. 2018, and WordNet), each word's sex confirmed by its
    dictionary gloss.
- **First names (P24b):** 1,738 US first names (Social Security records, ≥ 95% one sex,
  ≥ 10,000 births, not also an ordinary word like *Rose*). A name's sex is a convention, not a
  meaning, and English names leak sex through spelling (*-a* names are mostly female, and
  Spanish *-a* is mostly feminine). So names are only compared **within the same final letter**:
  *-a* names with *-a* names, *-n* names with *-n* names. A direction that only read "ends in *a*"
  scores 0.5 there.
- **Overlap test (cross term).** The *cosine* between the social and grammatical directions,
  estimated in split halves so noise can't inflate it (Gotchas, `estimators.py`). The test asks
  whether the overlap beats a **shuffle null**: grammatical labels shuffled within cells, 100
  times. The test statistic is the split-half **cross term** (the dot product of the two
  directions, each from an independent half), not the cosine. Under shuffling the grammatical
  direction has essentially zero length, and dividing by it made the null cosine explode
  (values above 1, which a cosine can't have). So the original test could never reject. The
  cross term needs no division (PI, 2026-10-10). The cosine stays as the effect size.
- **Reference direction:** the *number* direction (plural − singular on the same training
  nouns): a real direction built the same way, with no reason to point at "woman". It shows what
  "unrelated" looks like.

## Laptop results (English list v4, 2026-10-12)

Mean AUC over inner layers; in brackets, layers whose bootstrap interval is above 0.5 (pairs:
above chance share 0.5).

| | 1.7B | 4B |
|---|---|---|
| ES direction on person words (v4: 12 f / 10 m) | 0.81 (24/27) | 0.77 (24/35) |
| DE direction on person words | 0.70 (8/27) | 0.68 (5/35) |
| ES on names, within final letter | 0.78 (23/27) | 0.72 (32/35) |
| DE on names, within final letter | 0.61 (17/27) | 0.66 (27/35) |
| ES on pairs, female vs male word (16 pairs; share of pairs ordered right) | 0.81 (21/27) | 0.88 (30/35) |
| DE on pairs, female vs male | 0.80 (18/27) | 0.78 (24/35) |
| ES / DE on pairs, female vs generic (6 pairs: *actress/actor*, *heroine/hero*…) | 0.48 / 0.69 (0, 5 layers) | 0.67 / 0.77 (3, 5 layers) |
| social direction: person words / names / pairs (sanity check) | 0.82 / 0.97 / 0.99 | 0.83 / 0.98 / 0.96 |
| overlap with social, ES: layers above shuffle null | 21/27 | 29/35 |
| overlap with social, DE | 7/27 | 24/35 |
| split-half cosine with social: ES / DE / number | 0.30 / 0.12 / 0.01 | 0.30 / 0.22 / 0.01–0.03 |

Reading: the Spanish grammatical direction, learned only from inanimate nouns, sorts English
person words, first names (spelling held fixed) and matched female/male pairs (*niece/nephew*:
concept held fixed) by sex. It overlaps the social direction well beyond the shuffle null
(cosine ≈ 0.3 against ≈ 0.01 for number). German is weaker and less consistent: clear on names,
pairs and overlap at 4B, weak on the person-word list. Female vs generic (*actress* vs *actor*):
no reliable ordering, but only 6 pairs, so it can't distinguish "the generic word reads as
neutral" from noise. v3 → v4 changed the person-word numbers by ≤ 0.02.

## Cloud results (8B/14B, session 2, LAST)

Same layout as above: mean AUC, layers whose interval is above 0.5 in brackets. These were
pre-registered as side tests, not as confirmatory Phase 4–5 tests.

| | 8B (35 layers) | 14B (39 layers) |
|---|---|---|
| ES direction on person words | 0.87 (31) | 0.85 (35) |
| DE direction on person words | 0.67 (9) | 0.72 (20) |
| ES on names, within final letter | 0.75 (31) | 0.70 (34) |
| DE on names, within final letter | 0.64 (27) | 0.62 (34) |
| ES on pairs, female vs male word | 0.92 (30) | 0.87 (35) |
| DE on pairs, female vs male | 0.75 (25) | 0.81 (30) |
| ES / DE on pairs, female vs generic (6 pairs) | 0.60 / 0.73 (0, 9) | 0.62 / 0.71 (0, 8) |
| social direction: person words / names / pairs | 0.93 / 0.95 / 0.99 | 0.85 / 0.97 / 0.97 |
| overlap with social, layers above shuffle null: ES / DE | 34 / 15 | 34 / 13 |
| split-half cosine with social: ES / DE / number | 0.44 / 0.18 / −0.01 | 0.22 / 0.08 / 0.00 |

Reading: the laptop result replicates at both sizes. The Spanish direction, learned from
inanimate nouns, sorts English person words, names and female/male pairs by sex; German is
weaker. **One thing to set beside Phase 5:** on 14B the Spanish direction still *sorts* English
person words as well as on 8B, but its overlap with the social direction halves (cosine 0.44 →
0.22). Phase 5 found that pushing along it moves gendered adjectives on 8B and not on 14B. The
two fit together (a smaller shared component, a smaller push on social readouts), but this is a
post-hoc observation on two sizes, not a test. Sorting and steering ask different things: a
direction can rank person words by sex because a small part of it lines up with sex, without
that part being large enough for a push to move behaviour.

The personified-noun check (dropping nouns cultures personify) changes the directions little on
14B as well (cosine with the original ≥ 0.95).

## P27: German neuter diminutives (exploratory)

*Das Mädchen* (girl) and *das Brüderchen* (little brother) are grammatically **neuter**: the
diminutive endings *-chen/-lein* make any noun neuter, whatever the person's sex. So they test
whether the grammatical directions respond to a person's sex when grammar says neither.

- **Items:** 25 PI-screened person diminutives (12 female, 8 male, 5 sex-neutral such as
  *Kindchen*) and 165 diminutives of things (*Tischchen*), automatically from Wiktionary, whose
  base noun is feminine (68), masculine (62) or neuter (35).
- **Comparisons:** female vs male persons; things with a feminine vs masculine base (the
  **stem effect**: does the base noun's gender survive the diminutive?); *excess* = persons'
  separation minus things'; the *Mädchen* family vs neuter-base things, and vs sex-neutral
  persons (is it sex, or just being a person?). Bootstrap by stem family (*Mädchen*,
  *Zimmermädchen*… count as one).

| (mean AUC; layers with CI above 0.5) | 1.7B | 4B |
|---|---|---|
| DE direction: female vs male persons | 0.39 (0/27) | 0.38 (0/35) |
| DE: things, feminine vs masculine base (stem effect) | 0.60 (13/27) | 0.61 (27/35) |
| DE: *Mädchen* family vs neuter-base things | 0.68 (23/27) | 0.64 (21/35) |
| DE: sex-neutral persons vs neuter-base things | 0.79 (17/27) | 0.73 (8/35) |
| DE: *Mädchen* family vs sex-neutral persons | 0.35 (4/27) | 0.43 (6/35) |
| ES direction: female vs male persons | 0.60 (1/27) | 0.53 (2/35) |
| *Control:* English social direction, female vs male persons | 0.89 (19/27) | 0.95 (33/35) |
| *Control:* social, *Mädchen* family vs sex-neutral persons | 1.00 (27/27) | 0.99 (35/35) |
| *Control:* social, sex-neutral persons vs neuter-base things | 0.39 (0/27) | 0.49 (0/35) |
| *Control:* social, stem effect for things | 0.48 (0/27) | 0.51 (0/35) |

Reading: **no sex signal in neuter person diminutives.** Female and male persons aren't
separated (if anything the order is reversed, not reliably). *Mädchen* does look more
feminine than neuter things, but so do sex-neutral persons, by as much or more, so that is a
**person effect** (persons sit toward the feminine side of the direction), not sex. The base
noun's gender does survive diminutivisation for things (4B: 27/35 layers). Small sets: suggestive
only.

**The control makes the null informative (added at the PI's request, after the first results).**
The English social direction (woman − man) separates the same female and male diminutives almost
perfectly, puts the *Mädchen* family far from sex-neutral persons, and shows no person effect and
no stem effect. So the model *does* represent the sex of *das Mädchen* and *das Brüderchen*; the
grammatical directions just don't pick it up there. For neuter persons, the grammatical directions
track grammar and personhood, not sex: the opposite of what they do for English person words.

**Cloud (8B/14B):** the null replicates. DE direction, female vs male persons: 0.38 (1/35) on
8B, 0.47 (0/39) on 14B; ES direction 0.55 (8/35), 0.49 (0/39). The social control separates
them on both (0.97, 32/35; 0.97, 39/39) and the *Mädchen* family from sex-neutral persons
(0.99 on both, every layer). The stem effect for things weakens on 14B (8B 0.67, 30/35; 14B
0.56, 5/39).

## Limits worth stating

- 1.7B/4B are laptop runs (fp16); 8B/14B are cloud runs (bf16), so the two pairs of sizes differ
  in hardware and precision as well as size.
- Small person-word sets (27 main): wide intervals. Names are the large set, but a name's sex is a
  social convention, so they test association with sex, not the meaning *female*.
- Final-letter cells control the last letter only, not other spelling cues (*-ette*, *-lyn*) or
  frequency.
- A cosine of 0.3 is a partial overlap: most of each direction is not shared.
- WordNet's register labels miss some insults; the PI dropped three such words by hand
  (list v3), so the main set is not purely rule-built.
