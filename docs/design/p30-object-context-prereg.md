# P30: does *la* pull toward feminine-rated adjectives when the noun is clearly an object? (ADOPTED, PI 2026-10-17)

Drafted 2026-10-17. Only a 0.6B smoke test (n = 4 stems) has run; code in `src/gbleed/p30.py`. Follows P29
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

- **Precondition: the neutral pull replicates P29**, i.e. S(neutral)'s lower 95% bound > 0.
  If not, there's no pull to explain and the object result is reported descriptively. (Added
  2026-10-17 before sign-off: a logic gap the 0.6B smoke test exposed. With S(neutral) ≤ 0,
  "below half of S(neutral)" is meaningless. Not tuned on any value.)
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
| pass, but S(neutral) ≤ 0 | n/a | No pull to explain: descriptive only. |
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

## Decisions (PI, 2026-10-17)

Adopted as drafted, including the neutral-pull precondition; 49 person nouns after the PI's drops.

1. Adopt C1–C2 and the measures as written (including the 1-nat margin and the "half of
   S(neutral)" bound)?
2. The 53-noun positive control: built by the rule above with no hand-picking. The list is
   checkable in Spanish; anything you'd drop gets dropped, not replaced.

## Results (2026-10-17; Qwen3 on L40S bf16, commit 6ed2c99; EuroLLM-1.7B on the Mac; `results/p30/`)

| | EuroLLM-1.7B (Mac fp16) | Qwen3-8B | Qwen3-14B |
|---|---|---|---|
| C1 probe median (person − thing, nats): O1 / O2 / N1 / P1 / P2 | −2.06 / −0.16 / +2.30 / +1.97 / +0.88 | −2.50 / −1.25 / +1.12 / +4.50 / +0.50 | −2.25 / −1.50 / +1.81 / +5.87 / +1.88 |
| valid wordings | O1, O2, N1 (P1, P2 fail C1) | O1, O2, N1, P1 (P2 fails C1) | O1, O2, N1, P1 (P2 fails C1) |
| real person nouns, N1 probe median | +6.65 | +5.25 | +4.88 |
| **S(object)**, invariant adjectives | +0.008 [0.000, 0.016] | **+0.080 [0.073, 0.087]** | **+0.124 [0.115, 0.133]** |
| S(neutral) | −0.000 [−0.013, 0.013] | +0.020 [0.011, 0.029] | +0.068 [0.058, 0.077] |
| S(person) | n/a | +0.152 [0.143, 0.162] | +0.187 [0.176, 0.197] |
| real person nouns (scale): neutral / person | +0.052 / n/a | +0.113 / +0.131 | +0.159 / +0.192 |
| ratio object / neutral | n/a | 3.98 [2.83, 6.80] | 1.82 [1.62, 2.09] |
| **verdict** | **no neutral pull to explain** (descriptive) | **pull survives an object reading** | **pull survives an object reading** |

**Reading (as pre-registered):**
- **Qwen3 8B and 14B: the pull survives an object reading.** In both object sentences the model
  reads the made-up noun as a *thing* (the probe favours *una cosa* by 1.3–2.5 nats). Even so,
  *La* draws feminine-rated adjectives more than *El* does. The "model guesses a woman"
  explanation of P29 is therefore not enough.
- **EuroLLM-1.7B: no neutral pull to explain**, so the object result is descriptive: +0.008,
  barely above 0. The person sentences failed the probe check there.

**Not predicted by either reading:** the pull is *larger* in the object sentences than in the
neutral one (8B ×4, 14B ×1.8), although the neutral sentence leans "person" (+1.1/+1.8 nats).
Removing the person reading didn't shrink the pull.

**Limits to state:**
1. **Cross-sentence sizes aren't directly comparable.** A graded score's size depends on how the
   adjective distribution behaves in each sentence (the share differs: object 0.17–0.34,
   neutral 0.16). "Survives" is robust; "larger than neutral" is a weaker claim.
2. **The size is surprising.** On 14B the object pull is ~78% of the shift that real
   person nouns show when they really are a woman (+0.124 vs +0.159), and 8B ~70%.
3. **"Social" isn't established.** The result shows that Spanish feminine *grammar*, on a noun
   read as a thing, makes English-feminine-rated adjectives likelier. That's the grammatical-to-
   social association the project asks about, but it may run through distributional
   statistics: feminine nouns as a class may be described with different adjectives in Spanish
   text, for reasons unrelated to people. This design can't separate that from a social-gender
   route.
4. **Family generality is open.** EuroLLM-1.7B shows ~0 (as did Qwen3-0.6B's smoke test, n = 4
   stems), so the effect may need scale. It isn't shown to be family-general.
5. P2 (*que conocí*) failed C1 on every model; person = P1 only.

