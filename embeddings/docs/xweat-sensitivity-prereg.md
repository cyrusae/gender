# XWEAT T7/T8 sensitivity (pre-registration)

*Status: **DRAFT, written by Claude 2026-11-11, for the PI to adopt.** Script:
`scripts/xweat_sensitivity.py`. It was smoke-tested on **random fake vectors only** (mechanics,
and an independent check that the effect size equals XWEAT's formula); **no score on real
embeddings has been computed.***

## Question

Lauscher & Glavaš (*SEM 2019, Table 5) compare gender effect sizes across seven languages on
WEAT T7 (math vs arts words × male vs female words) and T8 (science vs arts × male vs female).
Two of their readings rest on these numbers: German comes out least biased ("German texts are
indeed less biased"), and languages rank differently (e.g. T8: Spanish 1.16 vs Italian 0.10).
**Do the numbers survive small, mechanical changes to the word lists?** If one word, or one fix
to an obvious list error, moves a language's score across another's, the comparison between
languages isn't supported by the test.

Why it matters for this study: T7/T8 targets are **inanimate topic nouns** scored against gendered
person words, which is the C1 setting (grammatical gender of things vs social gender). The topic
nouns were never balanced for grammatical gender (Goldfarb-Tarrant et al. 2021, §4.3); only
person words got both forms.

## Materials

- **Lists:** XWEAT (`external/XWEAT`, umanlp/XWEAT commit 4f95484). The English lists are read
  from `weat.py`; the translations come from **their own pickled dictionaries** (what actually ran;
  the Russian CSV uses other columns, so a re-parse would differ). They're loaded with an
  unpickler that refuses code objects (the pickles contain plain data only: checked opcodes).
  No licence file: nothing is copied into tracked files.
- **Vectors:** MUSE `wiki.multi.{lang}.vec` (Conneau et al. 2018): the same family as the
  paper's vectors (fastText Wikipedia, Bojanowski et al. 2017, 200k vocabulary), rotated into
  English space. Rotation keeps every within-language cosine, so monolingual scores should
  match the paper's *if* MUSE didn't also re-centre or re-normalise the vectors. Arm 1 tests
  this. Languages: en, de, es, it, hr, ru, tr (missing files are reported with the download URL).
- **Genders** (within-gender arm only): Wiktionary lexicon (`gbleed lexicon`): de, es, ru.
  Lemma match first, then a unique case-insensitive match. Plural entries (*matemáticas*,
  *Gleichungen* "equations") aren't lemmas, so they come out `unknown` and are excluded (no
  hand labels). Italian, Croatian and Turkish have no lexicon: no within-gender arm.

## Arms (all fixed now, before any real score)

1. **repro:** their pipeline exactly: translate (both forms where given; capitalised English
   words skip translation), de-duplicate, lowercase (`--lower True`, as in their
   `fasttext_multiling.sh`), drop out-of-vocabulary words, pop words from the longer set until
   sizes match. Which word gets popped depends on Python's hash-randomised set order (issue X2),
   so **every possible pop is scored**; we report the median and the range.
2. **clean:** mechanical fixes only, no judgment calls: drop a word that's in **both** attribute
   sets or both target sets (Spanish *su* "his/her", Turkish *o* "he/she/it"); drop multi-word
   items (can't be in a word vocabulary); drop untranslated English proper names in non-English
   languages (*Einstein*, *NASA*, *Shakespeare*: not words of the language, and male persons
   sitting in target sets). No popping. **Mistranslations are not fixed**, because fixing them
   needs judgment; candidates are listed in `upstream-issues.md` (X4) as claude-unverified.
3. **leave-one-out (loo):** the clean sets minus one word, for every word in every set.
4. **within-gender (descriptive only):** clean target sets restricted to masculine-only,
   feminine-only (and, in German, neuter-only) on both sides; scored only if each side keeps
   ≥ 3 words.

Statistic: as in XWEAT's code: per-word association s(w) = mean similarity to the male words
minus mean similarity to the female words; effect size = difference of the two target groups'
mean s, divided by the population standard deviation of s over both groups; one-sided p = share
of all re-splits of the pooled targets whose statistic is strictly larger (exact, not sampled).
Primary similarity: **cosine** (the paper's text). Secondary: XWEAT's Euclidean similarity
1/(1+distance), because the run script committed to their repo uses it (issue X7).

## Readings (decided now)

- **R1, reproduction:** "reproduces" if the paper's value lies within the repro range ± 0.15;
  otherwise "doesn't reproduce", and we check the Euclidean arm and MUSE preprocessing before
  interpreting anything else. (0.15 is a convention, not derived: tentative.)
- **R2, fragility** (per language × test, clean arm): "fragile" if **any** single leave-one-out
  (a) flips the sign of the effect, (b) moves p across 0.05, or (c) moves the effect by ≥ 0.5.
- **R3, cross-language comparison:** two languages are "distinguishable" on a test only if their
  leave-one-out ranges (min to max, including the clean value) **don't overlap**. Primary pairs:
  ES vs IT (T8), DE vs ES (T7, T8), and each language vs EN.
- **R4, within-gender:** descriptive; no confirmatory claim (sets of 3–6 words).
- The repro vs clean difference is reported per language as the effect of the mechanical fixes.

## Known limits

- Sets are tiny (≤ 8 words); exact permutation p-values are coarse (14 words → 3,432 splits).
- MUSE ≠ the paper's exact files; R1 tells us how far that matters.
- Lowercasing is kept for faithfulness even though it can push capitalised German nouns out of a
  cased vocabulary (X9). The OOV column in `per_word.csv` shows how many it loses. A cased German
  rerun would be an addendum, not part of this prereg.
- The test inherits WEAT's general problems: no link to downstream harm (Goldfarb-Tarrant et al.
  2021); this study only asks whether the numbers are stable, not what they mean.
