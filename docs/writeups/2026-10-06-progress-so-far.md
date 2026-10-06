# Does an LLM's "feminine" bridge leak into "feminine" people? Day one of finding out

*A progress write-up for the grammatical-gender bleedthrough project, covering Phases 0–2.
It assumes you know roughly what an LLM is and what a token is, but not interpretability
technique. Terms in bold are defined in the [glossary](../glossary.md); the reasoning behind each
choice is in the [decisions log](../decisions.md).*

---

## The question

In German, *bridge* is feminine (*die Brücke*); in Spanish it's masculine (*el puente*). For a
bridge, that's an accident of grammar: nothing about bridges is female or male. In 2003, Lera
Boroditsky and colleagues reported that this accident leaks into thought: German speakers
described bridges as *elegant* and *slender*, Spanish speakers as *strong* and *sturdy*. It's one
of the most-cited findings in linguistic relativity. It's also shaky: the study was never fully
published, and when Mickan, Schiefke and Stefanowitsch tried to replicate it in 2014, they found
nothing.

Language models give a new way to ask the question. A model trained on German and Spanish text
has to learn grammatical gender: it has to know that *Brücke* takes *die* and *puente* takes *el*.
The question is whether, *inside the model*, that grammatical "feminine" ends up sharing machinery
with social "feminine", the one that's about women and men. If you take the direction in the
model's internal space that separates feminine from masculine *inanimate* nouns and push a
sentence about a bridge along it, do the adjectives the model expects shift toward *elegant*?

The project is planned as six gated phases. Each produces a usable result even if the next never
runs:

| phase | question |
|---|---|
| 0 | Does the model even know the genders? |
| 1 | Can we remove spelling information, so "feminine" doesn't just mean "ends in -a"? |
| 2 | Is there a Spanish gender direction that tracks gender rather than spelling? |
| 3 | Same for German |
| 4 | Is that direction shared across the two languages? |
| 5 | Does it overlap with social gender? |

This post covers Phases 0 through 2. Phase 2 is set up and running as I write.

## Ground rules, decided early

A few principles shaped everything, so they're worth stating first.

- **Gold labels never come from a language model.** I don't read German, and I'm working with an
  AI assistant, which makes it tempting to ask it "what gender is *Kiefer*?". But that would
  quietly make the experiment depend on the very kind of system being studied. Every gender label
  comes from Wiktionary.
- **Drop rather than check.** Any item that would need a human to verify something (is this noun
  animate? is this a real translation?) gets filtered out automatically. Small, clean lists beat
  large ones that need review. When a person does need to look at something, it's shown with
  English glosses so it can be checked without knowing German.
- **Decide how to read results before seeing them.** For each test, the criteria were written into
  the decisions log *before* running ("pre-registration"). Analyses added afterwards are labelled
  exploratory.
- **Hold test sets out, and never tune on scores.** Sentence templates and thresholds were chosen
  on grammatical grounds, then checked on more than one model, not picked by whichever scored best.

## Building the word lists

Everything starts with nouns and their genders. Wiktionary publishes machine-readable dumps
(through a project called kaikki.org), about a gigabyte per language. A builder script turns each
dump into a table of about 50,000 nouns with their gender, first English gloss and word frequency,
plus a pile of flags.

Most of the work is in the flags, because most nouns are bad stimuli for one reason or another:

- **People and animals** have to go (the study is about *inanimate* nouns, where gender is
  arbitrary). There's no "animate" field in Wiktionary, so this is inferred from several signals:
  topic categories ("Occupations", "Mammals"), whether the word has a male/female counterpart
  (*director/directora*), and what WordNet says the English gloss means. One early bug: *Buch*
  ("book") was marked animate because of a rare second sense (a ruminant's stomach) filed under
  "Animal body parts". The fix was to judge only the first sense.
- **Homographs** are words spelled like another word. They're excluded because they corrupt word
  frequencies (Spanish *de* is listed as a noun, the name of the letter d, and is also the
  commonest preposition) and because they muddy what the model is representing.
- **English overlap.** *Kindergarten*, *Stagnation*, *Grill*, *Machete* are the same word in
  English. A model may represent them mostly through English, so they say little about how
  German or Spanish handles gender. Words more common in English than in the target language are
  dropped.
- **Sex-associated objects** (*falda* "skirt", *pintalabios* "lipstick", anatomical terms) carry
  *social* gender. Train a "grammatical gender" direction on them and the final phase becomes
  circular, so they're kept out of training sets.

Several of these rules came from me reading the generated files and spotting problems:
*Kindergarten* in the translation pairs, *pedestal* and *praxis* in a training list, *libido* and
*polio* among the test items. Each catch became a rule rather than a one-off deletion, so the
write-up can state *why* items were excluded.

Two sets were hand-picked: 19 **classic** flipped-gender pairs from the literature (*Brücke/puente*,
*Mond/luna*, *Schlüssel/llave*…), and a set of words with **one spelling and two genders**
(*der See* "lake" / *die See* "sea"; *el mar* / *la mar*), which I reviewed against German
Wiktionary and the Spanish Royal Academy's dictionary. Even hand-picked items get automatic checks:
for example, the check flagged that *luna* and *estrella* are also women's names, which matters a
lot for the social-gender phase.

## Phase 0: does the model know the genders?

There's no point looking for a representation of something the model gets wrong.

**How you ask a model.** A language model assigns a probability to every possible next token. Chain
those together and you get the probability of a whole stretch of text. So, to test whether a model
knows *Brücke* is feminine, compare how well the noun fits after each article:

> margin = log P("Brücke zu tun." | "Das hat etwas mit **dem**") − log P("Brücke zu tun." | "Das hat etwas mit **der**")

A negative margin means the noun fits better after the feminine article. The unit is the
**nat** (natural-log unit): a margin of −3 means about 20× more likely after *der*.

**The sentence templates turned out to matter a lot.** Each one is designed so that the wrong
article can't be read some other legitimate way:
- German *die* is also the plural article (*die Schlüssel* = "the keys"), so the first template
  uses the dative (*mit dem/der X*), where the plural would look different.
- The second template went through three versions:
  - Sentence-initial *Der/Die X ist hier* failed: with nothing before the article, the model scored
    only ~75%.
  - *Ich weiß, dass der/die X hier ist* still let the plural reading survive until the last word.
  - The final version uses the indefinite article (*Hier ist ein/eine X*), which has no plural at all.
- A subtler bug: the first version scored the whole sentence, *including the article itself*.
  German *ein* and *dem* also serve neuter nouns, so they're more common than *eine* and *der*
  whatever the noun is, and that tilted everything toward masculine. The fix scores the noun
  *given* the article.

A noun counts as **known** only if both templates favour the right article by at least 1 nat.
Anything closer is "unsure", not wrong. An early version also used a fill-in-the-blank quiz
("*Teppich: der / Tasche: die / Brücke:* ___"). It turned out to measure the format more than the
knowledge: reordering the four examples flipped up to 44% of answers. It's now a diagnostic only.

**Results.** On a balanced list of 257 nouns per language, Qwen3-4B knows 94% of German and 94% of
Spanish nouns, with almost no confident errors; its misses are near-ties on rare words.
EuroLLM-1.7B, a European-focused model, knows 97–98%. I chose the **Qwen3** family anyway: it comes
in sizes from 0.6B to 14B that share one tokenizer, so I can develop on a laptop and rerun the
identical experiment on rented GPUs at 8B and 14B, and compare sizes cleanly.

## Looking inside: what an "activation" is

Phases 1 onward look inside the model rather than at its outputs, so here's what that means
concretely (this is what `src/gbleed/activations.py` does).

A transformer processes text as a sequence of tokens. For each token it keeps a running vector,
the **residual stream**: a list of about 2,000 numbers for Qwen3-1.7B. Each of the model's ~28
**layers** reads that vector and adds something to it. The vector at a given layer is the model's
working summary of "what this token is, in context" at that depth.

The extraction code does four things:

1. **Formats the input.** Each word is fed as `<|endoftext|> brelda`: a document-separator token,
   then the word with a leading space, as it would appear mid-sentence.
2. **Runs the model once** and asks the Hugging Face library to return the residual stream at
   every layer (`output_hidden_states=True`). No special interpretability library is needed just
   to *read* these.
3. **Keeps the vector at the word's last token**, at every layer. A word like *Schlüssel* is
   several tokens (`Schl|üss|el`), and only at the last one has the model seen the whole word.
4. **Saves the result to disk** as an array of shape (words × layers × vector length), along with
   the tokens and the exact model, device and code version.

Why the separator token? Because of a surprise found while checking the code: **the first
position in any sequence is weird**. At layer 10 of the smallest Qwen, the vector there was about
180 times longer than anywhere else. Models use the first position as an **attention sink**, a
place to park attention they don't need. A one-token word placed there would be measured at a
broken spot.

## Phase 1: trying to erase spelling

**The problem.** Spanish nouns ending in -o are mostly masculine and -a mostly feminine. So a
"gender direction" learned from Spanish nouns might just be an "ends in -a" detector.

**The plan** (from the design doc): learn what "-o vs -a" looks like somewhere it has nothing to do
with gender, and erase it. Spanish verbs are perfect: *hablo* ("I speak") and *habla* ("she
speaks") differ only in -o/-a, which marks person, not gender. Then check that erasing this also
removes the ending from made-up words like *flitra/flitro*.

**Two tools.**
- A **probe** is a simple classifier (logistic regression) trained to guess a label from the
  activation vectors. If it can tell -a words from -o words it wasn't trained on, the information
  is there and linearly readable.
- **LEACE** (Belrose et al., 2023) is an eraser. Picture each word's vector as a point coloured -o
  or -a. The paper proves that no linear classifier can do better than guessing exactly when the
  two colours have the same average position, and LEACE is the formula that moves the points as
  little as possible to make those averages coincide.

**The result** (pre-registered outcome "(b)"): the verb eraser removed -o/-a from verbs, but left
the nonce words 99–100% readable, at every layer, in all three model sizes. The design doc's
fallback, mixing half the nonce words into the eraser's training, didn't help either.

**The mistake, and what it taught.** My first analysis said two variants of the eraser worked
perfectly: exactly 0.500 accuracy at every layer. That exactness was the giveaway. The probe I'd
used to *check* the eraser was trained on the same words the eraser had been fit on. LEACE makes
the class averages equal on its fit data, and at equal averages the best linear probe learns
nothing. So the check was guaranteed to say "erased", whether or not anything remained. Done
properly (probes trained only on words the eraser never saw), even an eraser fit *on nonce
words* left new nonce words fairly readable (AUC 0.82, where 0.5 means erased).

**Why erasure fails here.** For each nonce pair, draw an arrow from the -o form's vector to the
-a form's. If "-a vs -o" were one shared feature, all the arrows would point the same way. They
don't: the average arrow explains only 18–45% of a typical pair's change. Part of the reason is
tokenization. The ending is usually fused into a stem-specific final token (*it|ra* vs *it|ro*),
and there are 77 different final-token pairs among 300 nonce stems. LEACE removes what's shared;
most of the signal is word-specific.

That's a finding in itself, and it matters for the final phase too: one proposed Phase 5 test
erases grammatical gender and checks whether social gender breaks, and that only means something
if the erasure actually generalizes to the contexts being tested.

## Phase 2: a Spanish gender direction, set up to fail honestly

Since spelling can't be erased afterwards, Phase 2 controls it **by construction**. The main
training set is **ending-matched**: Spanish nouns that don't end in -o or -a, and don't have a
gender-predicting suffix, with equal numbers of masculine and feminine nouns *for each final two
letters* (*la nube / el aljibe*, *la señal / el pedernal*). In that set, the ending carries no
gender information at all. The gender direction is just the difference between the average
feminine vector and the average masculine vector.

It's then tested where spelling and gender disagree:
- **Exceptions:** *el problema*, *el día* (masculine but end in -a), *la mano*, *la foto* (feminine
  but end in -o). The main test compares the masculine *-a* exceptions with regular feminine -a
  nouns (same ending, different gender) and with regular masculine -o nouns (same gender,
  different ending). A gender direction separates the first pair and not the second; a spelling
  detector does the reverse.
- **Homographs:** the same string as a noun (*mi camino* "my path") and a verb (*yo camino* "I
  walk"). Gender should show up only in the noun reading.
- **One spelling, two genders:** *el mar* vs *la mar*. Does the direction follow the article?

Getting here took a surprising amount of stimulus repair, much of it from reading the files:
- Some exceptions turned out to be English words (*libido*, *polio*).
- The original test leaned on only five feminine exceptions, too few to carry it, so it was
  redesigned before any results came in.
- The training set was 27% English-like words and included sex-associated anatomy terms.
- The smallest model knew too few of the rare training nouns, so it was dropped from this phase.

Every one of these changes is logged, with its reason, as made *before* results.

The final training set is small (25 nouns per gender), so the confidence intervals resample the
training nouns as well as the test nouns. That way, uncertainty about the direction itself shows
up in the error bars.

## What I'd say I've learned so far

- **The measurement is most of the work.** More effort went into templates, filters and
  stimulus lists than into anything that looks like "interpretability".
- **Exact numbers are suspicious.** A clean 0.500 everywhere was a bug, not a result.
- **Guarantees have scopes.** "Provably removes all linear information" is true only for the data
  the eraser was fit on, and here that scope mattered.
- **Reading the data matters, even without the language.** Several of the most important fixes came
  from skimming English glosses and noticing that something looked off.

## What's next

Phase 2 results (Qwen3-1.7B and 4B). Then Phase 3 (German, where spelling predicts gender much
less, and where neuter nouns can serve as a neutral reference point for separate masculine and
feminine vectors), Phase 4 (cross-language), and Phase 5 (the actual question), on rented GPUs with
the larger Qwen3 models.
