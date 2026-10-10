# Phase 4 explained: is grammatical gender one direction across languages?

*Code: `src/gbleed/phase4.py` (pairs, known checks, extraction, analysis), `phase4_ru.py`
(Russian list). Design and pre-registered rules: `docs/design/phase4-design.md` (Q1–Q6) and
`docs/decisions.md` (2026-10-08/09). Terms in **bold** are in the [glossary](../glossary.md).
Numbers here are **exploratory** (Qwen3-1.7B/4B on the Mac, fp16); the confirmatory runs are
8B/14B in cloud session 2.*

## The question

Phases 2 and 3 found a Spanish gender direction and a German one. Does the model use **one**
direction for both languages, or one per language? It matters for Phase 5: if the directions
are separate, a Spanish-learned push may mean nothing in an English sentence, and a null there
would say "doesn't transfer", not "no bleed".

## The tools

**Directions.** ES (from the Phase 2 Spanish training nouns), DE (Phase 3, masculine vs
feminine), and POOLED (both languages together, **centred within language** so it can't use
"is this German?"). All are fitted with the same **stratified** estimators as Phases 2–3:
gender is compared only within spelling cells, after **residualising** frequency, length and
concreteness.

**Flipped pairs.** 271 concepts whose gender differs between the languages (*la luna* /
*der Mond*, *el puente* / *die Brücke*), known by both models, none in any training set
(asserted in code). A grammatical direction should follow each language's gender; a meaning
direction should score the two translations alike, since they mean the same thing.

## The tests

| test | what it asks | statistic |
|---|---|---|
| T1 transfer (co-primary) | Does the Spanish direction sort German nouns by gender, and vice versa? | AUC **within the test language's ending cells**, so German *-e* (mostly feminine) can't pass as Spanish *-e* (mostly masculine) |
| T3 flipped pairs (co-primary) | Does one direction score each pair's German noun by German gender and its Spanish noun by Spanish gender? | **paired AUC** over pairs, ES on both nouns and DE on both |
| T2 geometry | How aligned are the two directions? | **split-half cosine** with a within-cell **shuffle null** |
| T4 third language | Does either direction sort Russian *-ь* nouns (all share one ending) by gender? | AUC |

**Decision rules (adopted before the data).** A test passes if its bootstrap lower bound is
above 0.5 in a majority of inner layers. **Holm** across T1 and T3 by interval width: both are
first judged with 97.5% intervals; if one passes, the other is judged with 95%.

## Laptop results (exploratory, LAST position)

- **T3 passes everywhere:** in every layer of both models, one direction scores each pair by
  each language's own gender (4B mean paired AUC 0.93 with ES, 0.88 with DE; the same without
  English cognates). At the AFTER position it is much weaker (ES 15 of 35 layers, DE 0).
- **T1 fails the majority rule:** above chance in a minority of layers (4B: 12 and 17 of 35;
  1.7B: 1 and 15 of 27), mean AUC around 0.70–0.74. At AFTER, 0 layers.
- **T4 (4B only):** ES and DE both sort Russian *-ь* nouns by gender in almost every layer
  (34 and 31 of 35), a language and script neither was trained on.
- **T2:** split-half cosine about 0.44 at 4B, but no layer clears the shuffle null.

## How to read the combination

T3 and T4 say the direction carries gender into the other language and into Russian; T1, the
stricter within-cell test, says it doesn't sort German nouns well once endings are held fixed.
The two aren't contradictory: the pair test doesn't control German endings, and German endings
carry gender (*-e* feminine), so part of T3's success may come from the ES direction picking up
German spelling cues. Russian *-ь* nouns all share one ending, so T4 is spelling-controlled by
construction, which makes it the cleanest evidence of transfer so far. Worth stating in a
write-up: "partly shared", with the confirmatory 8B/14B runs deciding.

## Limits worth stating

- T1's test nouns are the other language's training set (no fresh held-out draw was possible);
  they are held out for the direction under test.
- The pairs come from English-gloss matching, about 10% wrong under strict rules, so some
  "pairs" don't mean quite the same thing; the cognate and no-cognate versions agree.
- Mac fp16 numbers; the cloud runs re-extract everything on one chip.
