# Label audit + ending-exception test, Spanish (exploratory, 2026-11-08)

`embeddings/scripts/label_audit.py`. Gold = our Wiktionary lexicon (`data/lexicon/es_nouns.csv`).
`word_labels.csv` has every distinct word with its category; `summary.json` the numbers.

## Label audit (distinct words in each paper's Spanish masculine + feminine lists)

| | Omrani Sabbaghi & Caliskan 2022 | Zhou et al. 2019 |
|---|---|---|
| distinct words | 10,056 | 12,233 |
| agree with Wiktionary | 5,709 | 6,244 |
| disagree | **14 (0.2%)** | **56 (0.9%)** |
| two genders in Wiktionary (*orden*, *capital*…) | 96 | 185 |
| listed under both genders | 96 | **860** |
| not a noun lemma in Wiktionary | 4,141 (**3,021 plurals**; rest adjectives/participles: *heridos, interesado, contrario*) | 4,888 (adjectives/determiners *próximo, mismo, nuevo, otro*; abbreviations *pib, fbi, iva*; numerals; tokens with quotes *«control»*) |
| animate among the agreeing nouns | 508 (9%) | 587 (9%) |

- Where a word *is* a Wiktionary noun, both papers' gender labels are almost always right. The
  problem is what else is in the lists: about 40% of entries aren't singular nouns at all, and ~9%
  of the nouns are animate in lists described as inanimate.
- **Zhou's disagreements point to labels read off agreement in text:** 18 of 56 are feminine nouns
  starting with stressed *a-* that take *el* (*el agua, el alma, el área, el aula, el ancla, el
  hambre*) labelled masculine. Combined with adjectives and determiners in the lists, this looks
  like a POS tagger's gender feature, not a dictionary.
- **Correction to an earlier guess:** *mano* is two-gendered in Wiktionary (Mexican *el mano*,
  "bro"), so it isn't a plain labelling error.

## Ending-exception test

Each paper's direction (LinearSVC, 3,000 + 3,000 deduplicated nouns from its own lists; Omrani on
cc.es with their column scaling, Zhou on MUSE) classifying our Phase 2 sets, words not in training:

| set | Omrani: n, accuracy | Zhou: n, accuracy |
|---|---|---|
| regular -o/-a | 150, 1.00 | 95, 0.94 |
| ending-matched (ending carries no gender) | 37, 1.00 | 14, 0.93 |
| *-ma* exceptions (*el problema, dilema, teorema*) | 9, 1.00 | 3, 1.00 |
| other exceptions (*la mano, el sofá, el tranvía*) | 3, 1.00 | 2, 1.00 |

**The guess was wrong:** neither direction is just "ends in -o vs -a". Exceptions and
ending-matched nouns are classified as well as regular ones. Small exception sets (n = 2–9), so
this rules out a pure-spelling direction, not a partial spelling component.
