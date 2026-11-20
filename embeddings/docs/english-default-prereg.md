# Is the masculine default in Spanish and French inherited from English? (pre-registration)

*Status: DRAFT, 2026-11-20, written by Claude for the PI's review. Nothing scored.*

## Why

Caliskan et al. (2022, AIES; arXiv 2206.03390) report that in English static embeddings most frequent
words lean male (GloVe: 93% of the top 100, 77% of the top 1,000, 65% of the top 10,000, 55% of the top
100,000; fastText 83 / 65 / 58 / 57%), measured with SC-WEAT against eight male and eight female
attribute words (*male, he, him, his, man, boy, son, brother* / *female, she, her, hers, woman, girl,
daughter, sister*). SC-WEAT's zero is the midpoint between the attribute words, which nothing guarantees
is sex-neutral (the same problem we found in Zhou et al.); the decline with rarer words fits both a real
masculine default and an off-centre zero.

The PI's question: work on gendered languages, including Zhou et al.'s aligned spaces, treats English
as the neutral reference. If English person words already lean male, part of the "masculine bias" seen
in Spanish and French retrieval may be English's default made visible by morphology, rather than a
failure of gendered languages. Our person-noun results have a masculine default even for ordinary person
nouns under plain translation (Spanish 0.28, French 0.45) that we had attributed to the target side.

## Questions

1. **Does Caliskan et al.'s masculine default survive a sex-neutral English zero point?**
2. **Does an English word's own lean predict how strongly it lands on the masculine form** when
   translated into Spanish or French (plain translation, no cue)?
3. **Does the occupation-specific intrusion survive controlling for the English word's lean?**

## Materials

- **Vectors:** MUSE aligned fastText Wikipedia (`wiki.multi.{en,es,fr}`), as in all our studies; first
  200k English words. Not the Common Crawl GloVe/fastText Caliskan et al. used, so Q1 tests the claim on
  a different, widely used embedding rather than replicating their numbers exactly.
- **Attribute words:** Caliskan et al.'s sixteen, as listed above (all in the 200k vocabulary).
- **English zero point (new; PI review needed):** `data/en_zero_candidates_v0.csv`, built by
  `scripts/en_zero_build.py`. Two sets:
  - **Primary, generic person words** (17 after rule filters, `primary_set` = True): *person, persons,
    people, individual, individuals, someone, somebody, anyone, anybody, everyone, everybody, member,
    members, resident, residents, youth, folks* (seed list by Claude, `claude_suggest_unverified`; the
    PI keeps or drops each). Words for people that name no sex. Of the 6 with a Glasgow GEND rating,
    the mean is 4.06 (neutral = 4).
  - **Sensitivity, the other glosses of our frozen Spanish/French strict epicenes** (56 single words after
    dropping professions, gender-specific words and occupation glosses). Not primary because many are
    stereotyped concepts (*mermaid, monarch, archangel, sheriff, watchman*) or occupations Wiktionary
    doesn't mark (*tailor, locksmith, pilot*).
  - Rule filters (both sets): one word, in the 200k vocabulary, not in Bolukbasi et al.'s
    gender-specific list or professions list, not the gloss of a noun in our occupation groups.
    Two more (2026-11-20): **anchors and test nouns kept apart** (no anchor is the gloss of a noun
    tested in Q2/Q3, so *friend, adult, human, stranger, child, neighbour* aren't anchors); **no
    clearly gendered anchors** (PI: Glasgow GEND, of the word or its singular/plural partner, outside
    3.5–4.5 drops it: *child, kid, baby, infant, stranger* and their plurals). A friend-based zero
    point (*friend, friends, neighbour, neighbours*) is reported as a further sensitivity, with Q2/Q3
    rerun without the nouns glossed by those words.
  - Reported with the list: the anchors' mean Glasgow GEND rating where available (a check that the
    zero point isn't itself gender-rated).
- **Person nouns (Q2, Q3):** the frozen Spanish (`data/wp_persons_es_frozen_v1.csv`: person 107,
  occupation 115) and French (`data/wp_persons_fr_frozen_v1.csv`: person 40, occupation 41) lists.

## Measures

- **Lean** of an English word w: s(w) = mean cos(w, male attributes) − mean cos(w, female attributes)
  (positive = male), i.e. the numerator of SC-WEAT. **Fair lean:** s(w) − z, where z = mean s over the
  zero-point anchors.
- **Q1:** share of words with s > 0 (Caliskan's zero) and with s − z > 0 (fair zero) among the 100,
  1,000, 10,000 and 100,000 most frequent MUSE English words (vocabulary order). Also the same with
  SC-WEAT's effect size d instead of s, at Caliskan's zero, as the replication check.
  95% intervals by bootstrap (10,000 draws, seed 0) over attribute words (within gender) and anchors.
- **Q2:** per noun, plain-translation gap g = RR(masculine form) − RR(feminine form) for the query
  "English gloss" (scored as in `scripts/wordpair_persons_extra.py`). Regression, per language:
  z(g) ~ z(fair English lean of the gloss) + z(log rank_f/rank_m) + group (occupation vs person).
  Bootstrap over nouns (10,000, seed 0). Identity nouns excluded.
- **Q3:** per noun, feminine-cue intrusion (from `results/wordpair_persons{,_fr}/per_noun.csv`).
  Regression: z(intrusion) ~ occupation + z(fair English lean) + z(log rank ratio). Bootstrap over nouns.
  Also reported: mean fair English lean of person vs occupation glosses.

## Readings

- **Q1:** "English masculine default robust to the zero point" if, among the top 1,000 words, the
  fair-zero share's 95% interval is above 50%; "explained by the zero point" if it includes 50%;
  "reversed" if below. Same reading reported at 100 / 10,000 / 100,000 (secondary).
- **Q2:** "English lean contributes" if its coefficient's interval is above 0 in a language; reported
  per language, with the frequency coefficient beside it.
- **Q3:** "occupation-specific beyond English lean" if the occupation coefficient's interval is above 0
  with the English lean in the model. If it isn't, but the English lean is, the occupation excess is (at
  least partly) English occupations being more male-coded than English person words.
- **Prediction (Claude's, before any score):** Q1 share falls toward 50% but stays above it at 1,000;
  Q2 English lean contributes in both languages; Q3 occupation excess survives. None of these is
  required for the study to be informative.
  - **Prediction (PI):** Agreed on all, see conversation leading to this prereg.

## Sensitivity

- Epicene-gloss anchors instead of generic person words (Q1–Q3).
- Q1 with frequency-matched comparison: share at the fair zero within each frequency band uses the same
  anchors for all bands, so if the anchors are themselves frequent (they are), rarer bands may be
  mis-centred; report the anchors' rank distribution and, descriptively, the correlation of fair lean
  with log rank.
- Q2/Q3 without flagged nouns (the five homograph flags).

## Limits

- Wikipedia fastText, not Common Crawl; one embedding family.
- The English zero is set by people-words; if English generic person words themselves lean male
  (generic *he*), the fair zero inherits that and Q1 understates the default. This is the main
  interpretive limit and is stated with the result.
- Q2/Q3 are correlational across nouns.

## Open questions at drafting

1. Review the 17 primary anchors (`data/en_zero_candidates_v0.csv`, `primary_set` = True): keep or drop each.
   (Claude's own doubts: *folks* is informal; *member(s)* and *resident(s)* are institutional;
   *youth* leans young/formal.)
2. Primary reading at the top 1,000 (matches Caliskan et al.'s headline 77%)?
3. Adopt Q2/Q3 as specified?
