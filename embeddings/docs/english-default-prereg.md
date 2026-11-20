# Is the masculine default in Spanish and French inherited from English? (pre-registration)

*Status: **ADOPTED 2026-11-20** (PI; adoption record at the end). Nothing scored before adoption.*

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

## Adoption record (2026-11-20, before any score)

- PI: adopt as written; the PI concurs with Claude's predictions (stated in conversation; the PI's own
  note to this effect wasn't found in this file when the record was written).
- Anchors frozen in `data/en_zero_frozen_v1.csv`, sha256 5ed999c9c9abf1bee59c7fc553ddfbce00e559060b51b80a3b8e87ddf0ffce3f:
  **primary 22** (the 17 reviewed + life-stage words added by the PI: *adolescent, adolescents, teen,
  teens, youths*; *teenager(s)* dropped by the rule because Bolukbasi et al.'s lists contain
  *teenager*; *citizens* dropped with *citizen*, a listed profession), all kept by the PI;
  **epicene_gloss 56** (sensitivity); **glosses_in** = primary + *adult(s), human(s), friend(s),
  neighbour(s)* (PI's backup plan: anchors that are glosses of tested nouns, with those nouns dropped
  from Q2/Q3; sensitivity, replacing the friend-only variant).
- Rule changes before adoption: list filters apply to a word's singular/plural family.
- Script `scripts/english_default.py`, committed with this record.

## Results (2026-11-20, run at 944ffd2; `results/english_default/`)

**Q1, masculine default in English: not determinable with these attribute words.** Share of words leaning
male (MUSE English, top 1,000):

| Zero point | Share male [95% CI] | Pre-registered verdict |
|---|---|---|
| Caliskan et al.'s (attribute midpoint) | 60.1% [28.7, 80.3] | — |
| Primary anchors (22 generic person words) | 70.7% [16.6, 94.4] | "explained by the zero point" |
| Epicene-gloss anchors (sensitivity) | 35.6% [16.4, 68.9] | same |
| Glosses-in anchors (sensitivity) | 80.0% [29.6, 94.0] | same |

Same pattern at 100 / 10,000 / 100,000 (Caliskan's zero 71 / 57 / 52%). **Reading:** the pre-registered
label is "explained by the zero point", but it overstates what the data show: the interval spans 50%
*even at Caliskan et al.'s own zero*, because resampling the eight male and eight female attribute words
alone moves the share from ~29% to ~80%. With these sixteen words the share of male-leaning words isn't
pinned down in this embedding, and the point estimate swings from 36% to 80% with the choice of
sex-neutral anchors. (The primary anchors lean slightly female, z = −0.006, so the fair zero raises the
male share.) This doesn't refute Caliskan et al. (different vectors; they report counts without attribute
resampling) but shows the headline share is fragile to both the attribute words and the zero point.
Fair lean is uncorrelated with log rank (ρ −0.05): no frequency gradient in this embedding.

**Q2, does an English word's lean predict the plain-translation default?** Spanish: **English lean
contributes**, β 0.27 [0.15, 0.39] (222 nouns), beside the frequency ratio β 0.24 [0.12, 0.35] and
occupation 0.53 [0.30, 0.75]; robust without flagged nouns (0.34) and in the glosses-in set (0.28).
French: **not shown**, 0.24 [−0.01, 0.47] (81 nouns), frequency ratio 0.35 [0.12, 0.55].

**Q3, occupation-specific beyond English lean?** Spanish: **yes**, occupation β 0.68 [0.48, 0.88] with
English lean (0.30 [0.20, 0.41]) and frequency ratio (0.29 [0.18, 0.39]) in the model; robust in every
set. French: **not shown**, occupation 0.25 [−0.11, 0.61] (English lean 0.41 [0.24, 0.57], frequency
ratio 0.37 [0.21, 0.53]). English occupation glosses don't lean more male than person glosses (fair lean
difference ES 0.005 [−0.004, 0.015], FR −0.002 [−0.019, 0.015]).

**Reading.** (1) The English word's own male lean is part of where the masculine default and the
intrusion come from: a noun whose English gloss leans male lands more on the masculine form, in both
languages for intrusion and in Spanish for the plain default; so the PI's hypothesis holds in part,
noun by noun. (2) It doesn't explain the occupation effect in Spanish: English occupation words are not
more male-coded than English person words, and the occupation excess survives the control. (3) In French
the occupation excess is no longer separable once English lean and the frequency ratio are both in the
model (81 nouns); the frequency ratio and occupation overlap there. Claude's predictions: Q1 wrong in form
(not decidable rather than "robust"), Q2 right for Spanish, Q3 right for Spanish, not French.

## Addendum A: expanded attribute words (ADOPTED 2026-11-20; adoption record below)

**Why.** Q1 couldn't be decided: resampling Caliskan et al.'s eight male and eight female attribute words
alone moved the male share from ~29% to ~80%. Q2/Q3 used the same sixteen words. This addendum asks
whether a larger list of *sex-referring* words narrows the intervals and changes any answer.

**Attribute pairs** (`data/en_attr_candidates_v0.csv`, built by `scripts/en_attr_build.py`): Bolukbasi
et al.'s equalize and definitional pairs plus Caliskan et al.'s sixteen, as (male, female) pairs; rule
filters: one word each, no proper names, both in the 200k vocabulary, neither an English zero-point
anchor nor the gloss of a tested noun, and where Glasgow rates a word it must be clearly gendered
(male ≥ 5.5, female ≤ 2.5). **46 pairs pass the rules**; Claude marks 8 as doubtful
(`claude_suggest_unverified`: places *monastery/convent*, hormones, animals *colt/filly, gelding/mare*,
the mismatched *fella/granny*, informal *dudes/gals*, and the pronoun overlaps *him/her*, *his/hers*).
The PI keeps or drops each. Glasgow is used only as a check (13 pairs are rated), never as the source:
its extremes include stereotypes (*lipstick, beard, handbag*), which would make the lean partly a
stereotype measure.

**Contrast list (not attributes)** (`data/en_attr_contrast_glasgow_v0.csv`): the 40 most masculine and
40 most feminine Glasgow words in the vocabulary, stereotypes included, run as a deliberate contrast to
show how much a lean measure depends on choosing sex words vs stereotype words.

**What is rerun.** Q1 (shares at Caliskan's zero and the fair zero, all three anchor sets), Q2 and Q3,
with s(w) = mean cos with the male words − mean cos with the female words over the kept pairs; the
bootstrap resamples *pairs* (keeping each pair together), anchors, and nouns as before. Same for the
contrast list (resampling words within side). The original sixteen-word results stay as reported.

**Readings.** As in the main prereg, applied to the expanded list (primary for this addendum). Added:
**"attribute choice matters"** if the expanded and contrast lists' top-1,000 fair-zero shares differ by
more than the expanded list's 95% interval half-width.

**Prediction (Claude's, before any score):** the Q1 interval narrows a lot (resampling ~46 pairs instead
of 8 per side); the expanded list's fair-zero share lands within ±10 points of 50%; the contrast list
gives a clearly higher male share; Q2/Q3 Spanish conclusions unchanged.

**Open for the PI:** keep/drop the 46 pairs (especially the 8 marked); adopt.

### Addendum A adoption record (2026-11-20, before any score)

- PI: drop all 8 marked pairs; adopt as written.
- Frozen: `data/en_attr_frozen_v1.csv` (38 pairs), sha256
  6fbb63c443f980b564101726d85200d5e74498484a608b87663d17e667fcb76f; contrast list
  `data/en_attr_contrast_glasgow_v0.csv`, sha256 c6b0da0b2fff677358f592aa79d15ccf04b607a4f33358c8bf9ef6e13454b83f.
- Script `scripts/english_default_attr.py`, committed with this record.

### Addendum A results (2026-11-20, run at d2293e1; `results/english_default_attr/`)

**Q1 with 38 sex-referring pairs.** At Caliskan et al.'s zero the interval now excludes 50%: top 1,000
**64.6% [55.8, 71.2]** male (16 words: 60.1% [28.7, 80.3]); top 100: 77% [63, 85]; top 100,000: 54.6%
[49.9, 58.9]. With a sex-neutral zero it still depends on the anchors:

| Anchors | Top-1,000 share male [95% CI] | Verdict |
|---|---|---|
| Primary (22 generic person words) | 76.2% [44.6, 92.2] | explained by the zero point |
| Epicene glosses | 27.6% [12.4, 51.7] | explained by the zero point (reversed at 10k, 100k) |
| Glosses-in | 80.9% [51.8, 93.1] | robust to the zero point |

Contrast list (Glasgow extremes): Caliskan zero 50.8% [41.4, 61.0]; primary anchors 57.4% [20.6, 89.2].
"Attribute choice matters": **not shown** (contrast − expanded = −0.19, within the expanded list's
half-width 0.24).

**Q2/Q3 with the expanded list:** unchanged. Spanish: English lean contributes to the plain-translation
default, β 0.29 [0.17, 0.41]; occupation-specific beyond English lean, β 0.64 [0.44, 0.84]; both robust
without flagged nouns and in the glosses-in set. French: both not shown, as before. With the contrast list
Spanish conclusions also hold (occupation 0.49 [0.26, 0.71]). English occupation glosses lean slightly more
male than person glosses with the expanded list (ES +0.009 [0.000, 0.017]) and clearly with the contrast
list (+0.025 [0.016, 0.034]): the stereotype-laden list sees occupations as male-coded, the sex-word list
barely does.

**Reading.**
1. With enough attribute words, the masculine default *relative to the attribute midpoint* is real in this
   embedding too (65% of the top 1,000, falling with rarer words, the same gradient Caliskan et al.
   report), so the attribute-word noise was what made the 16-word result undecidable.
2. Whether that is a masculine default *relative to sex-neutral people-words* depends on which
   people-words set the zero: generic person words (*person, someone, people*) put it at 76%, glosses of
   Spanish/French epicenes (many stereotyped: *monster, genius, mermaid*) at 28%. The interval for the
   primary anchors still spans 50%. So "most frequent English words lean male" holds against the
   attribute words' midpoint; against a people-based zero, it isn't established here.
3. Q2/Q3 don't depend on the attribute list: the English word's own lean feeds into the Spanish default and
   intrusion, and Spanish occupations carry extra intrusion beyond it.
Claude's predictions: Q1 interval narrows (right at Caliskan's zero; only partly at the fair zero), fair
share within ±10 of 50% (wrong: 76%), contrast clearly higher (wrong: lower), Q2/Q3 Spanish unchanged
(right).
