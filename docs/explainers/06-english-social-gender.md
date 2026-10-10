# P23/P24 explained: does the grammatical gender direction point at "woman"?

Exploratory side study of Phase 5 (design: `docs/design/phase5-design.md` P23, P24, P24b; code:
`src/gbleed/english.py`). Laptop numbers below are development results (Mac, fp16, LAST
position, Qwen3 1.7B/4B).

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
  - **person2, the main set** (16 f / 14 m): unmarked words from two human-made sources (GN-GloVe's
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

## Laptop results

| | 1.7B | 4B |
|---|---|---|
| ES direction on person2 (mean AUC; layers with CI above 0.5) | 0.84 (25/27) | 0.80 (28/35) |
| DE direction on person2 | 0.66 (12/27) | 0.62 (3/35) |
| ES on names, within final letter | 0.78 (23/27) | 0.72 (32/35) |
| DE on names, within final letter | 0.61 (17/27) | 0.66 (27/35) |
| social direction on names (sanity check) | 0.97 | 0.98 |
| overlap with social, ES: layers above shuffle null | 23/27 | 29/35 |
| overlap with social, DE | 6/27 | 24/35 |
| split-half cosine with social: ES / DE / number | 0.30 / 0.12 / 0.01 | 0.30 / 0.22 / 0.01–0.03 |

Reading: the Spanish grammatical direction, learned only from inanimate nouns, sorts English
person words and first names by sex, with spelling held fixed for the names. It also overlaps the
social direction well beyond the shuffle null (cosine ≈ 0.3 against ≈ 0.01 for number). German
is weaker and less consistent: clear at 4B on names and overlap, weak on person words.

## Limits worth stating

- Exploratory and laptop-only so far; the RunPod session re-extracts on 8B/14B.
- Small person-word sets (30 main): wide intervals. Names are the large set, but a name's sex is a
  social convention, so they test association with sex, not the meaning *female*.
- Final-letter cells control the last letter only, not other spelling cues (*-ette*, *-lyn*) or
  frequency.
- A cosine of 0.3 is a partial overlap: most of each direction is not shared.
- The main person set includes a few insult and slur words (*whore*, *jackass*) that the
  dictionary's register check misses; flagged for the PI.
