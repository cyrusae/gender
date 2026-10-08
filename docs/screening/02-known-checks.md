# 02 · Known checks: does the model know each noun's gender?

*Code: `src/gbleed/phase0.py` (Spanish, and German m/f), `phase3_known.py` (German m/f/n),
`framecheck.py` (Spanish verb frame). Explainer: `docs/explainers/00-phase0-behavioural-check.md`.*

A noun whose gender the model gets wrong can't anchor a gender direction, so every noun is
**behaviourally checked** before it can enter a word list: the model is asked, in effect, which
article goes with the noun. Only nouns the model clearly knows are kept.

## How a noun is scored

The model never generates text. In a short sentence frame, we compare how probable the noun
(and the rest of the sentence) is after each possible article:

> *Esto tiene que ver con **el** puente.* vs *Esto tiene que ver con **la** puente.*

The **margin** is log P(noun + rest | correct article) − log P(… | wrong article), in nats
(natural-log units; 1 nat ≈ the correct version is 2.7× more probable). Scoring the noun *given*
the article, not the article itself, avoids measuring how much the model likes each article in
general (*die* is also the German plural article, for example).

Per frame: **right** if the margin is ≥ 1 nat the correct way, **wrong** if ≥ 1 nat the wrong
way, else **unsure**. Per noun:

| status | rule | used? |
|---|---|---|
| **known** | right in both frames | ✓ |
| wrong | a frame confidently wrong, the other not right | ✗ |
| conflict | one frame right, the other confidently wrong | ✗ |
| unsure | anything else (near-ties) | ✗ (not counted as wrong) |

## The frames

| check | language | frames | why these |
|---|---|---|---|
| Phase 0 | Spanish | *Esto tiene que ver con el/la X.* · *Aquí hay un/una X.* | definite and indefinite; no article has another reading; nothing after the noun agrees in gender |
| Phase 0 | German (m/f only) | *Das hat etwas mit dem/der X zu tun.* · *Hier ist ein/eine X.* | singular contexts (rules out plural *die*); but *dem* and *ein* serve neuter too, so they can't separate m from n |
| Phase 3 | German (m/f/n) | *Hier ist der/die/das X.* · *Dort gibt es einen/eine/ein X.* | all three articles differ; singular *ist* and indefinite *ein-* rule out plurals; margin = correct − best of the two wrong ones |

Rejected earlier versions, and why (on grammatical grounds, not scores): a sentence-initial
article (Qwen3 has no start token, so the first word has no context); *Ich weiß, dass die X…*
(*die* can be plural until the verb); Spanish *Sé que la X…* (*la* can mean "her"); frames where
a later word agrees in gender (*… es nuevo*); contracting articles (*a + el* → *al*). Every
input starts with `<|endoftext|>` + newline, so the scored word isn't on an attention-sink
position.

## Frame checks (the frames themselves are tested)

Frames are chosen on grammatical grounds and then **checked** on two model families (Qwen3 and
EuroLLM), never tuned on scores:
- **German three-way:** every model × gender must have ≥ 70% known among common nouns
  (Zipf ≥ 4). Passed on Qwen3-1.7B/4B and EuroLLM-1.7B (and on the pod: all sizes from 1.7B up).
- **Spanish verb frame** (for noun/verb homographs, doc 03): *Bueno, siempre ___* ("well,
  always ___") must make homographs (*camino* "path" / "I walk") more likely than regular nouns,
  relative to *Bueno, mi ___* ("well, my ___"), for both *-o* (1st person) and *-a* (3rd
  person) words, with the bootstrap lower bound > 0. Passed on every model (Qwen3 all sizes,
  EuroLLM-1.7B). An earlier frame used *yo*/*usted*, which gave away the person, hence gender.

## Which models decide

Word lists are selected with nouns **known by both Qwen3-1.7B and 4B** (Mac), then **frozen**.
Larger models are analysed on the frozen lists, with their own known rate reported and a
sensitivity analysis that drops nouns that model doesn't know. Larger models know *fewer* of the
frozen nouns (14B: 92% of the German training nouns), not because they know less (on the full
pools they know more), but because the lists were selected on the small models' near-threshold
luck (regression to the mean). The misses are "unsure", not "wrong".

## Precision note

bf16 (pod) margins differ from fp32 by a median 0.07 nats; the pre-registered bf16-vs-fp32 check
on Qwen3-4B came in at 98.6% same status (bar: 99%): 7 near-threshold flips, no bias. Pre-registered
fallback for future pod checks: fp32 output layer.
