# Phase 3 explained: German, three genders, and separate masculine and feminine vectors

*Code: `src/gbleed/phase3_stimuli.py`, `phase3_known.py`, `phase3.py`, `estimators.py`.
Pre-registration: `docs/design/phases-3-5-plan.md` ("Phase 2 (v4) and Phase 3 analysis").
Every decision: `docs/decisions.md` (2026-10-07/08). Terms in **bold** are in the
[glossary](../glossary.md). Results here are **exploratory** (Qwen3-1.7B/4B on the Mac); the
confirmatory runs are 8B/14B on RunPod.*

## The questions

1. Is there a German gender direction that carries over to nouns it wasn't trained on, in
   particular **compounds**, whose gender comes from their last element (*das Herrenhaus*,
   "manor house", from *das Haus*), even when the first part has another gender (*der Herr*)?
2. **Markedness** (pre-registered): masculine is the grammatical default in German (generic
   masculine), so with neuter as the reference point, is the masculine vector *shorter* than the feminine one?

## Why German needed its own word lists

Building the training set took longer than anything else in the project, and most of the
problems are worth stating in a write-up, because each could have faked a result:

| problem | example | fix |
|---|---|---|
| compounds give the gender away at the last token | *Herrenhaus* ← *Haus* | single-root nouns for training; compounds become a test set |
| particle + noun formations act like compounds | *Abwasser* ← *Wasser* | counted as compounds |
| place names are all neuter | *Simmerath* | gloss filter |
| neuter is more often borrowed | 44% of neuter vs 28% of masculine are loanwords | match on loan status |
| suffixes shared by genders hide a spelling cue | all neuter *-is* nouns end in *-nis*; the only masculine is *Kreis* | suffix-aware cells (*-nis*, *-sel*, *-tum*…) |
| agent nouns are masculine by derivation, and often people | *Bohrer* ← *bohren* | excluded |
| animacy tags miss animals, people, groups | *Wachtel* (quail), *Heer* (army) | WordNet (two senses) + gloss + group checks |
| the Phase 0 frames can't tell masculine from neuter | *dem*, *ein* serve both | new three-way frames (*der/die/das*, *einen/eine/ein*) |

The PI's own review of the word list caught four of these (agent nouns, quail, sable, groups).
Automatic animacy tagging needed three layers plus a human read: a limitation worth reporting.

## How it's measured

**Cells and stratification.** Nouns are sorted into **cells**: same ending (or shared suffix) and same loan status. Gender is only ever compared *within* a cell, so spelling and loan status can't drive the result. The primary set (`strat3`) uses every known noun in cells containing all three genders: 64 masculine, 27 feminine, 31 neuter. Before any direction is fitted, activations are **residualised** on the cells plus frequency, token count and concreteness.

**The compound test.** A masculine/feminine probe is trained on single-root nouns, then scores *conflict compounds* (first part and head of opposite gender). If the score sorts them by the head's gender (AUC above 0.5), the direction follows grammatical gender; if by the first part's, it follows the first word. No compound's head is a training noun (that would re-measure it).

**The markedness test, and why the obvious version is wrong.** The masculine vector is mean(masculine) − mean(neuter), the feminine vector mean(feminine) − mean(neuter). Their plain lengths are inflated by noise, and more so for smaller groups, so feminine (27 nouns) would look longer than masculine (64) even if they were equal: exactly the predicted result. The analysis therefore uses **split-half** lengths (vectors from two disjoint halves, dot product), which noise doesn't inflate, plus a **shuffle null** (gender shuffled within cells: what noise alone produces). Markedness passes in a layer only if feminine minus masculine is above zero (95% bootstrap interval) *and* above the null's 95th percentile.

**The cosine** between the two vectors is descriptive, not a test: −1 would mean one axis with neuter in the middle, +0.5 three unrelated categories. It is reported only if both vectors' directions reproduce across halves and the cosine is clearly outside the noise range.

## Results (exploratory; Qwen3-1.7B / 4B)

| | 1.7B | 4B |
|---|---|---|
| **compounds follow the head** (LAST) | **27 / 27 layers** | **35 / 35 layers** |
| head AUC on conflict compounds (mean) | 0.81 | 0.81 |
| same without suffix-headed compounds | 0.79 | 0.79 |
| feminine-suffix nouns (*-ung*…), 0 = masc, 1 = fem | 0.34 | 0.33 |
| **markedness (1)** | **0 layers** | **0 layers** |
| plain (biased) estimate "feminine longer" | 25 / 27 layers | 31 / 35 layers |
| cosine interpretable | 0 layers | 0 layers |
| compounds at AFTER | 1 / 27 | 1 / 35 |

- **Compounds follow their head**, strongly, in every layer, for compounds whose heads the direction never saw. Three-class version: the head's gender is predicted well above chance, the first part's below chance.
- **Markedness is not supported.** The split-half point estimates even have masculine slightly longer, though not reliably in any layer.
- **The plain estimator would have confirmed markedness in almost every layer.** This is the bias the split-half design was built to remove, observed in real data: the clearest methodological result of the phase.
- **Feminine-suffix nouns don't score feminine**: the direction learned on single-root nouns doesn't carry over to *-ung*/*-heit* words, which sit about a third of the way from masculine to feminine. (They never appear in training, by design.)
- **At the AFTER position nothing holds**, as in Spanish: the gender information is at the word's own last token.

## How to read this in a write-up

- "A German gender direction trained on single-root nouns predicts the gender of unseen compounds from their head, including when the first part has the other gender."
- "We find no support for masculine as the unmarked (shorter) vector; a naive length comparison would have shown it, because of a noise bias favouring the smaller class."
- Limits: two small models, bare nouns, linear directions, 27–64 training nouns per gender; the compound result is read at the head's own token, so it shows the direction recognises the head's gender inside a compound, not how the model composes the compound.

## Next

The same analysis on Qwen3-8B/14B (confirmatory), then Phase 4: does a direction trained on one
language predict the other, and do flipped pairs (*la luna* / *der Mond*) follow each language's gender?
