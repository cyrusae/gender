# P30: does *la* pull toward feminine-rated adjectives when the noun is clearly an object? (DRAFT, awaiting PI sign-off)

Drafted 2026-10-17. Nothing below has been run. Follows P29
(`pooling-neutrality-prereg.md`). Exploratory.

## The question in plain words

In P29, a made-up noun behind *La* rather than *El* drew more feminine-rated adjectives, even
adjectives with one form for both genders (*grande*, *feliz*): 8B +0.028, 14B +0.082. Two
explanations:

- **The model guesses "a woman".** An unknown noun with *la* might be a female person:
  *la estudiante* "the (female) student", *la actriz* "the actress". That's ordinary Spanish
  meaning, not bleed.
- **Grammatical gender itself carries the association.** Feminine grammar makes feminine-rated
  words likelier, even for things. This is an unsteered, in-language version of the project's
  core question.

Test: make the noun clearly an object (or clearly a person) and see whether the pull survives.

## Design: two articles × three contexts

**Nonce nouns:** the R-NONCE Spanish set, all four endings (150 stems × *-a/-o/-e/-iz* = 600
words), each in every cell below, once with each article.

| Context | Wording | Gloss |
|---|---|---|
| object | O1 *{El/La} {N} que compré ayer es muy ___* | "The N I bought yesterday is very ___" |
| object | O2 *{El/La} {N} que guardé en el cajón es muy ___* | "The N I put away in the drawer is very ___" |
| neutral | N1 *{El/La} {N} es muy ___* | P29's W1, replicated |
| person | P1 *{El/La} {N} con quien hablé ayer es muy ___* | "The N I talked with yesterday...": *quien* is for people only |
| person | P2 *{El/La} {N} que conocí ayer es muy ___* | "The N I met yesterday..." |

Fixed on grammatical grounds: you buy and store things, not people; *quien* refers only to
people; *conocer* "meet" takes people.

**Positive control: real nouns that take either article for a person.** From the Wiktionary
lexicon, by rule:
- tagged both genders (free variation) and WordNet `noun.person`;
- Zipf ≥ 3.5; a single entry; no other part of speech;
- not ending in *-o* or *-s*; at least 4 letters; not an English loanword (English Zipf < 3).

That gave 53; the PI dropped *sacerdote*, *poeta* (dedicated feminine forms exist), *nene*
(colloquial) and *falte* (rare), leaving **49** (`P30_PERSON_DROP` in the code): *presidente,
juez, personaje, artista, periodista, bebé, comandante, representante, protagonista, colega,
...*. Scored in N1, P1, P2. They aren't used in the
object frames, which would be nonsense for them. Here the article really does mark a woman,
which gives the scale for "the model thinks it's a woman".

**Readouts**, unsteered, next token after *muy*:
- **primary:** the invariant adjectives (R3 proper's list);
- **secondary:** the R3-POOL lemmas (P29's M1 readout).

## Checks (validity)

**C1, the context works (manipulation check).** A probe on each prompt's stem, replacing
*es muy* with *es*: log P(*una persona*) − log P(*una cosa*) ("a person" vs "a thing"). Median
over nonce words and both articles, per wording. Valid iff:
- in each object wording the median is < 0, and at least 1 nat below N1's;
- in each person wording the median is at least 1 nat above N1's;
- and, as a check on the probe itself, the real person nouns' median in N1 is > 0.

The 1-nat margin (a factor of e) is fixed here, before data. A wording that fails is dropped;
**if both object wordings fail, there's no valid test**.

**C2, frame check (as R3-POOL's rule, PI 2026-10-15).** Median combined share of all read
tokens (invariant + pooled) ≥ 5% per wording, on these unsteered prompts; a failing wording
is dropped. The invariant readout is a subset read within that adjective slot. (In P29 its own
share was 3–5%, which a strict per-set bar would likely fail.)

## Measures (pre-registered)

S(context) = mean over nonce words of the graded score (−β_GEND, R3 covariates, as P29 M2) of
the *La* − *El* shift on the invariant adjectives, averaged over the context's wordings. 95% CI
from a stem bootstrap (a stem's four words stay together).

- **Primary: S(object).**
  - **The pull survives an object reading** iff its lower 95% bound > 0.
  - **Mostly the "a woman" guess** iff its upper 95% bound < half of S(neutral)'s point
    estimate.
  - Otherwise **partial** (both readings contribute).
- **Secondary (reported):**
  - S(neutral), S(person) and the ratio S(object)/S(neutral) with its CI;
  - S_real, the person nouns in P1/P2, as the scale;
  - per ending;
  - the same on the pooled gendered adjectives.

## What each outcome means

| C1/C2 | S(object) | Reading |
|---|---|---|
| fail | n/a | No valid test: the contexts don't steer object vs person readings. |
| pass | lower bound > 0 | **Feminine grammar pulls toward feminine-rated adjectives even for things.** An unsteered, in-language grammatical-to-social association. On its own it says nothing about steering or English. |
| pass | upper bound < ½ S(neutral) | P29's pull was mostly the model guessing "a woman". Ordinary semantics. |
| pass | otherwise | Both contribute. |

**Limit to state:** an object context reduces the person reading; it can't prove it's gone. C1
measures how far it drops.

## Models and cost

- **Qwen3 8B/14B** on one L40S pod, bf16 (as P29), about 25 min, about $0.50; the live price
  is stated before creating the pod.
- **EuroLLM-1.7B** on the Mac, fp16, free, as the second family for the frames and probes
  (project rule); its invariant and pooled lists already exist.
- Smoke test on Qwen3-0.6B first. Code committed before the run.

## Decisions for the PI

1. Adopt C1–C2 and the measures as written (including the 1-nat margin and the "half of
   S(neutral)" bound)?
2. The 53-noun positive control: built by the rule above with no hand-picking. The list is
   checkable in Spanish; anything you'd drop gets dropped, not replaced.
