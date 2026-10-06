# Phase 1 explained: a spelling eraser

*Code: `src/gbleed/phase1.py`, `phase1_stimuli.py`, `activations.py`. Terms in **bold** are in
[the glossary](../glossary.md). Paper: Belrose et al. 2023 (LEACE), in `sources/`.*

## The problem Phase 1 solves

In Spanish, nouns ending in -o are mostly masculine and nouns ending in -a mostly feminine. If
Phase 2 trains a "gender direction" on Spanish nouns, it could learn **"ends in -a"** instead of
**"is feminine"**, a spelling detector that happens to agree with gender on regular nouns. Phase 1
builds a tool to remove spelling information first, so whatever gender signal is left can't be
spelling.

The trick is to learn what "-o vs -a" looks like *somewhere it has nothing to do with gender*:
Spanish verbs. *hablo* ("I speak") and *habla* ("she speaks") differ only in -o/-a, and the ending
marks person (1st vs 3rd), not gender.

## Step 1: reading what the model represents

For each word we record the model's internal vector (**activation**) at the word's last
**token**, at every **layer**. Think of each layer's vector as the model's running summary of
"what this word is" at that depth: a list of 1,024 numbers for Qwen3-0.6B (2,560 for 4B).

Two practical details matter here:
- **The first position is anomalous.** Models use position 0 as an **attention sink**: on
  Qwen3-0.6B its vector is ~180× longer than any other. A bare one-token word placed there would
  be measured in a broken spot. So each word is fed as `<|endoftext|> brelda`: a
  document-separator token first, then the word with its usual leading space.
- **The last stored layer is different.** It already has the model's final normalisation applied
  (it feeds the output directly), so analyses report "inner layers" 1 to L−1.

## Step 2: probes, or "is the information there?"

A **probe** is a simple classifier, here **logistic regression**, trained to guess a label from
the activation vectors. If a probe can tell *-o* words from *-a* words with 100% accuracy on words
it wasn't trained on, the information is there and **linearly readable**. 50% = chance (the
classes are balanced).

**Train/test splits by stem.** *breld-a* and *breld-o* always land on the same side of the split.
Otherwise a probe could memorise "*breld*-something" and look clever without learning endings.
(For verbs: both forms of a verb stay in the same **cross-validation** fold.)

## Step 3: LEACE, or how to erase a concept

Picture each word's activation as a point in a 1,024-dimensional space, coloured by its label
(-o or -a). If the two colours sit in different regions, a linear classifier can draw a
flat boundary between them.

Belrose et al. prove a clean fact: **no linear classifier can beat guessing exactly when the
two classes have the same average position** (the same **class mean**). This property is called
**linear guardedness**. So to erase a concept linearly, you only need to make the class means
coincide.

**LEACE** ("LEAst-squares Concept Erasure") is the formula for doing that while moving every
point *as little as possible* ("as measured by a broad class of norms", in the paper's words). For
a two-way concept it changes the data along **one direction** (it's a **rank**-1 edit), using the
data's own spread to choose that direction so the rest of the representation is disturbed least.
It's closed-form: no training loop, just means and covariances.

The guarantee has a scope that Phase 1 is really about: **it holds for the data the eraser was fit
on.** An eraser fit on verbs makes *verbs'* -o/-a undetectable. Whether it also hides the -o/-a of
*other* words is an empirical question, and that's the test.

## Step 4: the test, nonce words

Nonce words like *flitra/flitro* have no meaning, no gender, no person: only the ending differs.
If the verb eraser removes "-a vs -o" *as a form*, a fresh probe on erased nonce words should
drop to chance. If it only removed "1st vs 3rd person", the nonce words stay separable.

Stimuli (all automatically checked; see `phase1_stimuli.py`):
- **299 regular -ar verb pairs** from Wiktionary's conjugation tables (*necesito/necesita* …
  *malgasto/malgasta*). Excluded: irregular/stem-changing verbs, and any form that is also a noun,
  adjective, pronoun or determiner, even ignoring accents (*canto* "song", *aquella* "that (f)",
  *termino* ≈ *término*).
- **300 nonce pairs** built from Spanish syllable patterns (*flitra/flitro*, *bercha/bercho*),
  absent from Spanish Wiktionary and with zero frequency in six languages; 150 stems train, 150
  test.
- **300 regular nouns** (*bolsa* f, …), a diagnostic only.

## How to read the results table (`results/phase1/<model>/layers.csv`)

**The rule that matters most:** a probe used to test an eraser must be trained on words *the eraser
never saw*. LEACE makes the class means equal on its fit data, and when class means are equal the
best logistic-regression probe has all-zero weights (that's the paper's Theorem 2.3 in action).
So a probe trained on the eraser's own fit data scores chance **by construction**, whether or not
information remains. The first version of this analysis made that mistake for three measures,
and they all came out at exactly 0.500, which is what gave it away (see "Two traps" below).

All measures report **ROC AUC** as well as accuracy. AUC asks "do the probe's scores rank -a words
above -o words?", regardless of where the yes/no cut-off falls; 0.5 = no information, 1.0 =
perfectly separable. Accuracy alone can sit at exactly 0.5 because a probe labels everything one
way, even when its scores still separate the classes.

| column | question |
|---|---|
| `verbs_before` → `verbs_after` | -o/-a on verbs. *after*: eraser fit on 4/5 of verbs, fresh probe cross-validated within the unseen 1/5 |
| `nonce_before` → `nonce_after` | **pre-registered test**: probe trained on train-split nonce stems, tested on test-split stems, before / after the verb eraser (which never saw nonce words) |
| `nonce_after_random` | control: erase a random direction instead |
| `nonce_cv_<eraser>` | probe cross-validated *within* the test-split stems, which no eraser ever sees. Erasers: `verb`; `pooled` (verbs + train stems as one concept: the design-doc fallback); `rank2` (verb ending and nonce ending as two concepts); `nonceonly` |
| `nouns_<eraser>` | gender probe on regular nouns (diagnostic only: for these nouns, ending = gender) |

**Pre-registered reading** (`decisions.md`, written before running): "near chance" = accuracy below
0.557 (95% upper bound for a coin-flipping probe on 300 items). Outcome (a) "removes -a/-o in
general" needs near-chance `nonce_after` in ≥ 2/3 of inner layers; outcome (b) "learned person, not
form" if it stays ≥ 0.60.

## Results (inner-layer averages, AUC)

| | Qwen3-0.6B | 1.7B | 4B |
|---|---|---|---|
| verbs: unseen verbs after verb eraser | 0.68 | 0.72 | 0.81 |
| **nonce, pre-registered (accuracy)**: after verb eraser | **0.99** | 1.00 | 1.00 |
| nonce (CV on unseen stems): verb eraser | 1.00 | 1.00 | 1.00 |
| … pooled (design fallback) | 0.99 | 1.00 | 1.00 |
| … rank-2 | 0.82 | 0.84 | 0.84 |
| … nonce-only | 0.82 | 0.84 | 0.83 |
| noun gender: none / after rank-2 | 1.00 / 0.97 | 1.00 / 0.97 | 0.99 / 0.98 |

**Pre-registered outcome: (b), at all three sizes.** The verb eraser leaves nonce endings fully readable at every inner layer (0 of 27 / 27 / 35 layers near chance). Erasure transfers to unseen *verbs* slightly *worse* in bigger models (AUC 0.68 → 0.72 → 0.81).

**What the corrected numbers say:**
- **Linear erasure of -a/-o generalises poorly to new words.** Even an eraser fit *on nonce words*
  leaves *other* nonce words at AUC ~0.8, and the verb eraser only partly hides the ending of verbs
  it wasn't fit on (~0.68). LEACE's guarantee is exact on the data it was fit on; it just doesn't
  carry over well to new items for this concept.
- A plausible reason: the ending is usually fused into a stem-specific final token (*it|ra* vs
  *it|ro*, *br|orda* vs *br|ordo*; only 8% of nonce words have the ending as its own token), so
  "ends in -a" is spread across many token-specific directions rather than one shared one.
- **Direct evidence for that** (0.6B): take each pair's difference vector, (*-a* form) − (*-o* form).
  If "-a vs -o" were one shared direction, these would all point the same way. They don't: the
  average cosine between pairs is only 0.19–0.47 across layers, and the *average* difference (the
  part an eraser can learn from other words) explains only 18–45% of a typical pair's difference.
  Pairs sharing the same final tokens (*…fa/…fo*, *…pa/…po*: 77 distinct token pairs among 300
  stems) are noticeably more alike (0.38–0.66), as the tokenization story predicts. Consistency
  rises in later layers, as the model abstracts away from the specific tokens.
- The design-doc fallback (pool half the nonce stems in) does essentially nothing: one direction
  can't serve both word types, and the 598 verb items dominate it.
- Noun gender stays readable after every eraser, but since none of them removes spelling well,
  that says little.
- Aside: in later layers the -a/-o directions line up more with the noun-gender direction
  (cosine ~0.04 at layer 4 → ~0.55 at layer 22, on 0.6B). Worth watching.

## Two traps (both hit, both now covered by tests in `tests/test_phase1_logic.py`)

1. **Probe trained on the eraser's fit data → chance by construction.** Explained above. This made
   the rank-2 and nonce-only erasers look like they worked (exactly 0.500).
2. **Erase on all data, then cross-validate → *below* chance.** Once class means are equal over the
   whole set, each training fold's mean difference points the *opposite* way to its held-out fold's,
   so the probe learns a reversed rule (the first version scored 0.15–0.20 on verbs).

## What this means for Phase 2 (a decision for the PI)

The eraser was meant to stop the Phase 2 gender vector from being a spelling detector. Since no
eraser removes spelling from new words reliably, the options are:

| option | how it controls spelling | cost |
|---|---|---|
| **A. Ending-matched training set** | train the gender direction on nouns whose ending carries no gender information *by construction*: for each final letter (-e, -z, -s, -l…) equal numbers of m and f nouns (*la llave / el puente*, *la luz / el lápiz*); no -o/-a, no gender-predicting suffixes | small: ~55 nouns per gender at Zipf ≥ 3 (more at a lower floor) |
| B. Erase anyway, then test | use the rank-2 eraser (best available, AUC ~0.8 residual) on regular nouns | spelling only partly removed; relies entirely on the exceptions test |
| C. Rely on the exceptions test | train on regular nouns with no control; Phase 2 test set 1 (*el problema*, *la mano*) shows whether the vector follows spelling or gender | the vector may well turn out to be a spelling detector, which is a finding but not a gender vector |

These aren't exclusive. A reasonable plan is **A as the main analysis, C on the same exception sets,
and B as a comparison**: if a direction trained on ending-matched nouns also sorts the exceptions by
gender, spelling can't be the explanation.

## Limits worth stating in a write-up

- Linear tools only: "erased" means linearly unreadable, not gone.
- LEACE's guarantee is in-distribution; generalisation to new items is empirical, and here it was weak.
- Nonce words are a stand-in for "form without meaning"; a model may still treat them as nouns
  with a (probably) inferred gender.
- Probe results depend on probe choice and regularisation; all layers use one fixed setting (C = 1).
