# 07: The German object test (P31), the confidence control, and the German frequency axis

Three pieces of work from 2026-10-21/22. Each was pre-registered before its data:
`docs/design/p31-german-object-context-prereg.md`, `confidence-ctl-prereg.md` and
`frequency-axis-prereg.md`. P29/P30, the Spanish version, are in explainer 05.

## 1. P31: does a feminine article on a *thing* pull toward feminine-rated adjectives, in German?

**What it measures.** As P30, in German: a made-up noun (*der/die/das Trart*) in sentences that
make it an object ("the Trart I bought yesterday is very ___"), a neutral sentence, or a person
sentence ("the Trart I talked with yesterday ..."). The score is how much *die* rather than
*der* shifts the next-word probabilities toward feminine-rated adjectives (Glasgow ratings via
English glosses), as a regression slope.

**Why German.** German adjectives after *ist* never change form for gender (*ist sehr groß*
for every noun), so no agreement form can leak into the readout. And *das* gives a third point:
is it "feminine" or just "not masculine"?

**Built by rule.** 300 made-up nouns whose endings don't predict gender in the German lexicon;
32 real person nouns that take *der* for a man and *die* for a woman with the same form
(*der/die Angestellte* "employee"), as the yardstick; the readout is the first token of each
German adjective translation.

**A probe that needed fixing before the real runs.** The check that the sentence makes the noun
read as a thing compares "a human/person/man/woman" with "a thing/object/item/thing". Summed
like that, it leans "person" everywhere, because *Mann* and *Frau* are much commoner words than
*Gegenstand* "item" or *Objekt*. So its zero point is meaningless. Fixed on principle before any
target-model data: the broad probe is judged only *relative to the neutral sentence*, and a
matched pair (*eine Person* vs *ein Ding*) carries the "it's a thing" sign.

**Results.**

| | EuroLLM-1.7B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
|---|---|---|---|---|---|
| object pull | **+0.081** | **+0.035** | **+0.043** | +0.002 (none) | not valid |
| neutral pull | +0.035 | +0.018 | +0.020 | +0.002 | +0.034 |
| Spanish object pull (P30) | +0.008 | +0.038 | +0.026 | +0.080 | +0.124 |

**How to read it.** Where the test works, the feminine article pulls toward feminine-rated
adjectives even when the noun is plainly a thing, and more in object sentences than neutral
ones, as in Spanish. But *which* models show it differs by language: the large Qwen3 models in
Spanish only, EuroLLM in German only, the small Qwen3 in both. So it isn't one effect that
simply grows with model size. Neuter *das* gives ≈ 0, so on Qwen it's feminine-specific.

**Limits to state.** Mac (fp16) and cloud (bf16) runs are mixed. The yardstick (real women)
barely moves on two of the models. 14B failed the check that its object sentences read as
objects. Descriptively, with that check's margin waived, it shows +0.025, below its neutral
+0.034; that is exploratory only.

## 2. The confidence control: are we measuring "less sure", not "more feminine"?

**The worry (Buttigieg et al. 2026, arXiv 2610.08559).** Directions built by contrasting
biased with unbiased answers turned out to be mostly *confidence* directions. Pushing along them
made the model less sure of everything, flattening the probabilities like a higher
temperature. If something similar happened to us, an article or a push could look like a gender
shift simply because it flattened the adjective distribution, if feminine-rated adjectives
happen to be the more (or less) likely ones.

**Three checks:**
- **(b) Is there room for it?** Yes, a little. More masculine-rated adjectives are slightly less
  likely before any push (rank correlation −0.10 to −0.32).
- **(a) Does it drive anything?** Each score was recomputed with one more control: how likely
  each adjective was in that sentence. That is the *midpoint* of the two conditions compared,
  because using the "before" value alone builds in regression to the mean.
  **Nothing changed.** The same Phase 5 layers pass. Every P29/P30/P31 verdict holds, with
  estimates within ~30% (most within 15%).
- **(c) Is our gender direction a frequency direction?** Spanish: no (|cos| ≤ 0.14). German:
  **yes, substantially** (−0.3 to −0.7). See part 3.

**How to read it.** The paper's mechanism doesn't explain our results. It would have shown up
in (a), and it didn't.

## 3. The German gender direction points toward "rarer word"

**What was found.** In Qwen3 8B/14B, the German feminine-minus-masculine direction has a cosine
of −0.3 to −0.7 with the direction in which activations move as words get rarer. This is despite
word frequency being a control when the direction was built.

**Why a control doesn't prevent it.** Controlling for frequency means the gender difference is
measured *at equal frequency*. It stops frequency differences between the noun groups from
being counted as gender. It says nothing about whether the gender direction and the frequency
direction *point the same way* in the model's space. The model can use overlapping dimensions
for both. A cosine asks about geometry, not confounding.

**What it isn't** (four pre-registered tests, `frequency-axis-prereg.md`):

| Explanation | Test | Result |
|---|---|---|
| The frequency control was too crude | curved frequency + token-count control | overlap stays |
| The regression produces it | frequency-matched feminine/masculine pairs, no regression | overlap stays (14 pairs) |
| Word structure: compounds, numerals | without the 12 complex nouns / with a control for them | overlap stays |
| Does it matter for Phase 4? | remove the frequency axis, rerun Phase 4 | **no**: verdicts unchanged; overlap cosine −0.01 (8B) / −0.05 (14B) |

**A stimulus problem found on the way.** Twelve of the 64 masculine German training nouns, and
none of the 27 feminine ones, are compounds or derived words: numeral nouns (*Vierer* "a four")
and instrument compounds (*Wasserkocher* "kettle", *Feuerlöscher* "fire extinguisher"). Phase 3's
"single-root words only" filter missed them; its rule had a gap. They turned out not to explain
the frequency overlap, but they're a flaw in every German direction and are logged as such.

**A side effect that turned out not to be what it looked like.** With the frequency axis removed,
the German direction sorts Spanish nouns by gender better (8B: 5 → 18 of 35 layers). That looked
like a cleaner shared gender signal. But splitting the comparisons by frequency shows otherwise.
On feminine/masculine Spanish pairs of similar frequency, nothing improves. The gain comes only
from pairs where the feminine noun is the commoner one, and those happen to dominate our Spanish
set. Removing the axis swapped one frequency lean for its opposite, and the opposite fit this
sample. Not reported as improved transfer.

**Register (age of acquisition) doesn't explain it either.** If the axis were "technical or
learned vocabulary", a control for when English speakers learn each noun's gloss should have
weakened the overlap. It didn't move it (e.g. 8B −0.62 → −0.61), and nouns' positions along the
axis don't track AoA. My earlier reading of the noun list ("compounding plus register") isn't
supported.

**What's left.** German grammatical femininity is partly encoded along the same axis as low
frequency in these models. Why is open. One possibility is that the "frequency" axis really
tracks something broader (how technical or dictionary-like a word is), with German feminine
nouns at one end of it.

**Limits to state.** 91 German training nouns in 8 ending groups, so cosine sizes are rough
(the permutation null is cleared by a wide margin at most layers). Two models of one family.
