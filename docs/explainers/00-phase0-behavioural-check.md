# Phase 0 explained: does the model know the genders?

*What the code in `src/gbleed/phase0.py` measures, why it's built this way, and what
the numbers mean. Terms in **bold** are in [the glossary](../glossary.md).*

## The question

Before looking *inside* a model for a representation of grammatical gender, check that
the model *behaves* as if it knows each noun's gender. If it can't pick *die Brücke* over
*der Brücke*, there's nothing to find, and that noun can't anchor a gender direction in
later phases.

## What a language model gives you

A language model reads text as **tokens** (word pieces: *Schlüssel* might be
`Schl|üss|el`) and, at every position, outputs a probability for every possible next
token. Multiplying those step by step gives the probability of a whole continuation:

> P("Brücke zu tun." | "Das hat etwas mit der") =
> P("Br" | …der) × P("ücke" | …der Br) × P(" zu" | …Brücke) × …

Those probabilities are tiny, so we work with their natural logarithms
(**log-probabilities**): products become sums, and the numbers stay readable.

## The measure: a margin between two articles

For each noun we build the same sentence twice, once with the masculine article and once
with the feminine one, and ask how well the noun (and the rest of the sentence) fits after
each:

> **margin** = log P(noun + rest | lead-in + *masculine article*)
>        − log P(noun + rest | lead-in + *feminine article*)

- Positive margin: the model prefers the masculine article for this noun. Negative:
  feminine.
- The unit is the **nat** (natural-log unit). A margin of +1 means the noun is
  e¹ ≈ 2.7 times more probable after the masculine article; +3 means ≈ 20 times;
  −5 means ≈ 150 times more probable after the feminine one.

**Why the article itself is left out.** The first version scored the whole sentence,
which includes P(article | lead-in). But German *ein* and *dem* also serve neuter nouns, so
they are more common than *eine* and *der* regardless of the noun. That tilted every
margin toward masculine (visible in two different models as feminine nouns having much
smaller margins). Conditioning on the article asks the right question: *given* this
article, how plausible is this noun?

## Two sentence frames

| | German | Spanish |
|---|---|---|
| frame 1 (dative) | *Das hat etwas mit dem/der X zu tun.* | *Esto tiene que ver con el/la X.* |
| frame 2 (indefinite) | *Hier ist ein/eine X.* | *Aquí hay un/una X.* |

Each frame was chosen so the **wrong article can't be read some other legitimate way**,
because then a model that knows the gender would still be pulled toward the wrong answer:

- **Plural readings.** German *die* is also the plural article: *die Schlüssel* is fine
  as "the keys". Frame 1 uses the dative, where the plural is *den Schlüsseln*; frame 2 uses
  the indefinite article, which has no plural. (An earlier frame, *Ich weiß, dass der/die X
  hier ist*, let the plural reading survive until the final *ist*, and masculine nouns
  whose plural looks like the singular, like *Stiefel*, *Besen*, got weak margins.)
- **Contractions.** Spanish *a el* → *al*, *de el* → *del*; *con el* doesn't contract.
- **Other readings of the article.** *Sé que la X…* lets *la* be the pronoun "her".
- **Giveaways later in the sentence.** *…es nuevo* agrees in gender, so it would leak
  the answer.
- **No context before the article.** A sentence-initial *Der/Die X ist hier.* scored only
  ~75% on Qwen3-0.6B (it adds no start-of-text token, so the first word is predicted from
  nothing). Every frame starts with a lead-in.

Using two frames that differ in case and article type (definite dative vs indefinite
nominative) means a noun has to pass two independent tests.

These frames were fixed on grammatical grounds and checked on two unrelated model families
(Qwen3 and EuroLLM), not chosen by picking whichever scored highest, which would tune the
test to one model.

## From margins to "known"

Each frame gets a verdict using a threshold, `--min-margin` (default 1.0 nat ≈ 2.7×):

| frame verdict | meaning |
|---|---|
| right | margin points to the correct article by ≥ 1 nat |
| wrong | margin points to the wrong article by ≥ 1 nat |
| unsure | \|margin\| < 1 nat: a near-tie |

A noun's **status** combines the two frames:

| status | rule | used downstream? |
|---|---|---|
| **known** | both frames right | yes (`passed`) |
| wrong | a frame confidently wrong, the other not right | no |
| conflict | one right, one confidently wrong | no |
| unsure | anything else (near-ties) | no |

The threshold is well above the rounding noise of half-precision arithmetic (~0.05 nat), so
"unsure" reflects the model, not the hardware. Near-ties are dropped, not counted as errors:
the goal is a clean set of nouns the model clearly knows, not a harsh grade.

## The headline number: balanced "known" rate

`known_bal` = (share of masculine nouns known + share of feminine nouns known) / 2.

Plain accuracy would reward a model that always answers *der* with whatever fraction of the
list happens to be masculine. **Balanced accuracy** gives such a model 50%.

Results are also broken down by **frequency bin** (`--by freq_bin`): low (Zipf 2.5–3.5),
mid (3.5–4.5), high (≥ 4.5). Common nouns are easy for every model; rare ones are where
models differ.

## The article quiz (diagnostic only)

The code also runs a few-shot quiz: `Teppich: der / Tasche: die / … / Brücke:` and reads
P(der) vs P(die) for the next token. It's reported (`quiz_bal_acc`) but doesn't decide
anything, because in small models it measured the format more than the knowledge: on
Qwen3-0.6B, reordering the four examples flipped 18% of German and 44% of Spanish answers,
and it failed about 1 in 5 *high-frequency* German nouns that sentence scoring got right. It's kept because whether a
model can *report* a gender, not just *use* it, may become interesting later.

## Where the nouns come from

Gold genders come from Wiktionary (via kaikki.org), never from a language model. The
lexicon builder then removes anything that would need a human check: animate nouns,
homographs (*de*, *camino*, *Tolle*), given names (*Charlotte*), nouns more common in English
(*arcade*), regional/archaic senses, and Spanish feminine nouns that take *el* (*el agua*).
See the README for the full list. The 19 classic pairs (*Brücke/puente*, *Mond/luna*, …) are
hand-picked and always included.

## Decision: compare sizes on the shared set

Each model gets its own Phase 0 result, so each ends up with a slightly different set of
known nouns. Cross-model comparisons in later phases use **the nouns known by every model
compared**, so differences can't come from different items. Per-model sets are a secondary
check. (Recorded in [decisions.md](../decisions.md).)

## Limits worth stating in a write-up

- Behaviour isn't representation: knowing which article to produce doesn't say *how* gender
  is stored. That's Phases 2–4.
- "Known" depends on the threshold and the frames; a noun can be known in other contexts.
- Two frames are a small sample of contexts. They're chosen to be clean, not representative.
- Wiktionary labels are reliable for gender, but the animacy and homograph filters are
  heuristics that err toward excluding.
