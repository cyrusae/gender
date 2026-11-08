# 09: Does the German rarity lean generalise? And a third language without articles (Russian)

Two follow-ups from 2026-10-22 to 10-26, both pre-registered before any activations existed:
`docs/design/followups/rarity-lean-heldout-prereg.md` and `docs/design/followups/russian-direction-prereg.md`.

## 1. The rarity lean on new nouns

**The question.** On Qwen3 8B/14B the German feminine-minus-masculine direction points toward
"rarer word" (explainer 07). That direction was fitted on 91 training nouns. Is the lean a quirk
of those 91 words, or does the direction really carry frequency? New nouns the direction has
never seen separate the two.

**Why not just rerun the same cosine on new nouns.** That cosine compares masculine and feminine
nouns *within* the same ending, because German endings predict gender. Every usable noun in a
mixed-gender ending had already gone into training. The remaining nouns sit in endings that are
all one gender (*-ie*, *-te*, …), where a gender × frequency fit would mostly measure endings.

**The design instead: a within-gender ranking.** 888 fresh simple nouns (502 feminine, 190
masculine, 196 neuter), frozen with a checksum before extraction. Keep the training direction
fixed, project each new noun onto it, and ask, *separately inside each gender*: do rarer nouns
sit further toward "feminine"? The statistic is a **partial Spearman** correlation between that
position and word frequency, with token count and ending removed first. It passes if it beats a
shuffle null (frequency shuffled within endings, 2,000 times) *and* is at least −0.1.

- If the lean is a property of the model, rarer nouns of **both** genders should lean feminine.
- A control direction (singular → plural) should *not* show the same thing, or the result is
  about frequency in general, not gender.
- A sanity check: the direction must still tell masculine from feminine on the new nouns
  (AUC ≥ 0.60 at most layers), or the test is void.
- *-ie* loanwords are 211 of the 502 feminine nouns, so the feminine result must also hold
  without them.
- Stated in advance: with 190 masculine nouns the test can only detect a correlation of about
  −0.14 or stronger.

**Results.**

| | feminine | feminine without *-ie* | masculine | control direction | reading |
|---|---|---|---|---|---|
| Qwen3-8B | 3/4 layers | 4/4 | **0/4** | 1/4 | mixed |
| Qwen3-14B | (3/4) | (3/4) | – | strong | **no valid test** (AUC < 0.60 at 3/4 layers) |

**How to read it.** On 8B, rarer *feminine* nouns do lean further toward feminine, but masculine
nouns don't. A model-wide "rare = feminine" property would predict both. On 14B the training
direction doesn't even sort the new nouns by gender at most layers, so nothing can be read from
it. Together with Gemma (no lean at all, explainer 08), the German frequency alignment is best
described as a property of Qwen3's training sample, with at most a feminine-only echo on 8B.

**Limit.** The masculine test can't rule out a weak lean (smaller than about −0.14).

## 2. Russian: a gender direction learned without articles

**The question (PI).** Spanish and German mark gender on articles; Russian has none, so gender
shows up only on the noun's ending and on agreeing words. Does a direction learned from Russian
line up with the Spanish and German ones? And does Russian share German's frequency lean?

**The training set.** Outside one spelling class, Russian gender is almost fully predictable from
the ending (*-а* nouns 3,428 feminine vs 12 masculine; consonant-final 11 vs 4,540). Only nouns in
*-ь* (soft sign) come in both genders, so the direction is learned from 174 reviewed *-ь* nouns
(100 feminine, 74 masculine), compared within spelling cells. An audit found feminine nouns there
shorter and less concrete, so concreteness, token count and frequency are covariates, as for German.

**R-A, alignment (four tests).** Russian direction → Spanish nouns, → German nouns; Spanish
direction → Russian nouns; German direction → Russian nouns. A layer passes if the bootstrap
interval (Bonferroni-corrected for the four tests) stays above chance; a test passes at a majority
of layers.

| | RU→ES | RU→DE | ES→RU | DE→RU | reading |
|---|---|---|---|---|---|
| Qwen3-8B (of 35 layers) | 29 | 25 | 30 | **9** | one-way |
| Qwen3-14B (of 39) | 33 | **18** | 34 | 28 | one-way |

Spanish–Russian is the strongest pairing in both models; German is the weaker partner, failing
in one direction each time.

**R-B, frequency.** No overlap between the Russian gender direction and frequency on either model
(cos −0.08 to −0.25). So the frequency lean is **German-only** among the three languages.

**Concreteness check, and why it looked bad at first.** Refitting on 37 concreteness-matched
pairs (74 nouns) weakened everything (alignment → "partial"). But a smaller training set always
gives a noisier direction. Follow-up S compared the matched set with 200 random 74-noun subsets
of the same gender × cell makeup: the matched set scored right at the random median (e.g. 8B
RU→ES 0.781 vs 0.783). **Size, not concreteness, explains the drop**, so the full-list alignment
reading stands. The frequency result on the matched set wasn't part of that check and sits near
the overlap bar on 8B, so "German-only" keeps a small caveat.

## 3. A reproducibility bug found on the way

Re-running identical code gave cosines that differed by up to 0.03. The cause: the permutation
null grouped items by iterating over a Python `set`, whose order depends on a per-process hash
seed. Fixed by sorting (c4b0fd1). All verdicts in this explainer, and every earlier one checked,
were unchanged; Monte Carlo variation across split seeds is about 0.01 per layer.

## 4. Limits worth stating

- Both tests are on Qwen3 only; Gemma had no Russian or held-out rarity run.
- Russian is learned from one spelling class (*-ь*), a narrow slice of the language.
- Concreteness ratings come from English glosses, so they rate the concept, not the Russian or
  German word.
