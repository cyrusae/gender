# Phase 2 explained: a Spanish gender direction, tested where spelling and gender disagree

*Code: `src/gbleed/phase2.py`, `phase2_stimuli.py`, `norms.py`. Pre-registration and every
amendment: `docs/decisions.md` (2026-10-06). Terms in **bold** are in the [glossary](../glossary.md).*

## The question

Phase 1 showed spelling (-o/-a) can't be cleanly erased from new words. So Phase 2 asks the core
Spanish question directly: is there a direction in the model's representation of a bare noun that
tracks **grammatical gender** rather than **spelling**? The test cases are nouns where the two
disagree.

## What a "gender direction" is

Take each noun's activation vector at one layer. The **difference of means** direction is the
average feminine vector minus the average masculine vector. Projecting any new noun onto it gives
a single score: high = "feminine-looking", low = "masculine-looking". The **adjusted** version
(primary here) is the same difference *at equal concreteness and frequency*: a per-dimension
regression of the vector on gender + concreteness (Brysbaert ratings) + frequency, keeping the
gender coefficient.

## How it's trained: two kinds of training noun

| training set | what's in it | what it controls |
|---|---|---|
| **ending-matched** (primary) | 25 + 25 nouns that don't end in -o/-a and have no gender-predicting suffix, with equal m/f for every final two letters (*la nube / el aljibe*, *la señal / el pedernal*) | the ending carries no gender information *by construction* |
| **regular** (comparison) | 91 + 89 regular nouns (-o masculine, -a feminine) | nothing: spelling and gender coincide |
| regular **after erasure** (comparison) | the same regular nouns with the Phase 1 spelling directions erased (verb eraser, or the rank-2 verb + nonce eraser) | removes whatever spelling the Phase 1 erasers capture |

Only nouns all models in the comparison know (Phase 0 "known") are used; English-overlapping and
sex-associated nouns are excluded from training.

## How it's tested

| test | items | a gender direction… | a spelling detector… |
|---|---|---|---|
| **primary: masculine -a exceptions** | *el problema*, *el sistema*, *el día*, *el mapa*… (26) | separates them from regular *feminine -a* nouns (**A_f** high) but not from regular masculine -o nouns (**A_m** ≈ 0.5) | the reverse (A_m high, A_f ≈ 0.5) |
| homographs | the same string as noun (*mi camino*) vs verb (*yo camino*, *usted cuenta*) | sorts m/f homographs in the noun frame better than in the verb frame | sorts both frames equally |
| *mar*-type | *el mar* vs *la mar* (article present) | scores *la mar* as more feminine | nothing |
| feminine -o exceptions | *la mano*, *la foto*, *la moto* (only 3) | feminine-like | masculine-like |

Each layer gets a verdict from 95% bootstrap intervals that resample the training nouns (refitting
the direction) *and* the test nouns: **gender** (A_f's lower bound > 0.5, A_m's isn't), **spelling**
(the reverse), **mixed** (both), **neither**. The pre-registered overall reading is the verdict in a
majority of inner layers.

## A bug, found and fixed before these results

The first run measured single-token words on an attention-sink position, where one dimension took
values ~2,500 (vs ~0), so the "gender direction" mostly tracked "is this word one token?". Inputs
are now `<|endoftext|>` + newline + word, and extraction warns on outlier vectors. The results below
are from the corrected inputs; the first run is superseded (see decisions.md and the write-up).

## Results (Qwen3-1.7B and 4B; one letter per inner layer, G = gender, M = mixed, S = spelling, N = neither)

| direction | 1.7B | 4B |
|---|---|---|
| **adjusted, ending-matched (primary)** | `NNNNNNNNNNNNNNNNNGGGGGGGGNN` | `NNNNNNNNNNNNNNNNNNNNNNNNGGGGGGNNNNN` |
| probe, ending-matched | `NNNNNNNNNNNNNNNNGGGGGGGGGGG` | `NNNSSNNNNNNNNNNNNNNNNNGGGGGGGGGGGGG` |
| regular, raw | `MMMMMMMMMMMMMMMMMMMMMMMMMMM` | `MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM` |
| regular, after rank-2 erasure | `GGGGGGGGGGGGGGGGGGGGGGMMMMG` | `GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG` |

**Pre-registered primary outcome: "neither" in a majority of layers** for both models, which is
the design table's *"nothing sorts cleanly: can't find it with this instrument; not evidence of
absence"*. The point estimates lean toward gender (A_f ≈ 0.72, A_m ≈ 0.56–0.59 on average), and a
"gender" verdict appears in later-middle layers (1.7B layers 18–25, 4B layers 25–30). But the
direction is noisy: it barely sorts *held-out ending-matched nouns* (AUC 0.52 / 0.58), so the
intervals are wide. With 25 nouns per gender, rare words, and no spelling help, the instrument
is underpowered.

**Comparison directions (pre-registered as comparisons, read with the same rule):**
- The **raw regular-noun direction is "mixed" at every layer**: it tracks gender *and* spelling
  (A_m 0.91–0.98), as expected when the two coincide in training.
- **After the Phase 1 rank-2 erasure it becomes "gender" at every layer of 4B and 23 of 27 layers
  of 1.7B.** Masculine -a exceptions score clearly below regular feminine -a nouns (A_f ≈ 0.95), and
  are not reliably separable from regular masculine -o nouns (A_m ≈ 0.55).
- On **homographs**, that erased direction sorts m/f nouns better in the noun frame than in the verb
  frame in **26 of 35 layers of 4B** (noun AUC 0.88 vs verb 0.71): gender beyond spelling. In 1.7B,
  only 2 of 27 layers.
- Both hold when English-overlapping exceptions are dropped, and within the more-abstract and
  more-concrete halves (concreteness is not the explanation).
- ***mar*-type**: the direction reads the article (*la mar* above *el mar*) in roughly half to all
  layers, depending on the direction.

So the Phase 1 eraser, which failed its own test of removing -a/-o from *new* words, does strip
most of the spelling component from a direction learned on *regular nouns*, leaving one that
behaves like grammatical gender on exceptions and (in 4B) homographs.

## A result that cuts the other way

The three **feminine -o exceptions** (*la mano*, *la foto*, *la moto*) score **masculine-like** on
the erased direction (gender index −0.3 to 0.1, where 0 = typical masculine, 1 = typical feminine).
That looks like spelling for these items. With three words it can't be interpreted on its own, but
it means the clean story ("the direction follows gender") holds for *masculine* exceptions and not
yet for feminine ones. One possibility worth testing later, not a conclusion: the direction may
encode "prototypically feminine" with everything else falling to the masculine default, which is
what the pre-registered markedness hypothesis would predict.

## How to read this in a write-up

- The pre-registered primary test is inconclusive: the spelling-free instrument is too weak at this
  sample size to show gender or spelling.
- A pre-registered comparison (regular nouns + spelling erasure) shows gender-like behaviour on the
  masculine exceptions in every 4B layer, and on homographs in most 4B layers.
- The feminine exceptions don't follow, and are too few to say why.
- Everything is bare nouns, linear directions, two model sizes.

## Next steps the results point to

- **More ending-matched training data**, so the primary instrument has power. (German, Phase 3,
  should help: spelling predicts gender much less there.)
- **More feminine -o exceptions**, if any exist, or a different test of the feminine side.
- **Readout position** (the after-word position) as a pre-registered robustness check.
- The larger Qwen3 sizes on RunPod.
