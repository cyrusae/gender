# Abstract, introduction and discussion: first draft (2026-11-07)

*Status: Claude's draft for the PI, written while the PI was away. It completes the sections the
methods/results draft (`2026-10-22-methods-results-draft.md`) leaves open. Claims follow that
draft's §3.11 table and `supplement-table.md`; no new numbers. Citations use only references
checked in `sources/Feynman-suggestions/review/` (corrected versions); each is tagged
[checked] or [check: …]. Everything here is for the PI to rewrite; nothing is final framing.*

---

## Abstract

Speakers of languages with grammatical gender have been reported to describe objects in ways
that follow the object's gender, a claim with mixed support in people. We asked a narrower
question about language models: does a model's representation of a noun's *grammatical* gender
overlap with what it represents about *social* (human) gender, so that pushing one moves the
other? Using pre-registered tests on Qwen3 base models (0.6B–14B) with a replication in Gemma 3
(4B, 12B), we find that (i) grammatical gender is linearly readable at the noun's own token and
tracks grammar rather than spelling; (ii) German compounds take their head's gender, with no
evidence that masculine is a "default"; (iii) Spanish, German and Russian gender directions are
partly shared; but (iv) steering a Spanish noun's gender direction moves English
social-gender adjective readouts in only one of four models (Qwen3-8B, small), and German
steering never had a valid test. Unsteered, a feminine article on an invented *thing*-noun pulls
toward feminine-rated adjectives in Spanish on every Qwen3 size tested (1.7B–14B); whether that reflects social
gender or how feminine nouns are described in text is not settled. A German-specific alignment of
the gender direction with word rarity in Qwen3 turned out to be a property of its training
sample: it did not appear in held-out nouns or in Gemma. Representation-level findings replicate
across families; the steering effect does not.

*(≈220 words. Open choices for the PI: lead with the human literature or with the model
question; whether to name the null as the headline.)*

---

## 1. Introduction

**The human question.** Whether grammatical gender shapes how people think about objects has a
long and contested history. Early studies reported that speakers describe objects with adjectives
matching the noun's gender [Boroditsky, Schmidt & Phillips 2003 — check: cite the 2003 chapter,
not the escholarship `0jt9w8zf` link, which is the 2000 CogSci paper]. A systematic review finds
the evidence mixed, with effects most reliable in tasks that make gender salient [Samuel, Cole &
Eacott 2019 — checked; primary figures unchecked]. Elpers, Jensen & Holmes (2022) separate a
correlational design, where the effect did not appear, from an induction design, where it did,
with a demand-characteristics caveat [checked].

**Why models.** Language models learn grammatical gender from text alone and expose their internal
states, so the question can be asked mechanistically: is there a direction for "feminine noun",
and is it the same direction the model uses for women? Prior work shows output-level adjective
associations that follow grammatical gender across languages, at accuracies modestly above
chance [Mihaylov & Shtedritski 2024, 54.3–69.0% within language — checked], and shared
grammatical features across languages inside models [Brinkmann et al. 2025 — checked; ablation
averaged over concepts]. Grammatical gender and meaning are confounded in the lexicon
[Stanczak et al. 2023 — checked; the quoted mutual-information values are word2vec only], which
is why we compare nouns within spelling cells. Steering results can also be explained by a model
simply becoming more or less confident [Buttigieg et al. 2026 — checked; layers 13–30], which we
control for.

**What we did.** A gated sequence of pre-registered phases: check the model knows each noun's
gender (Phase 0); find a Spanish and a German gender direction that survive spelling controls
(Phases 2–3); ask whether they are shared across languages, including article-free Russian
(Phase 4); then push along them and read English social-gender adjectives (Phase 5), with
unsteered article tests on invented nouns (P29–P31). Each rule was fixed before its data;
deviations are logged.

**Contributions.** (1) A spelling-controlled estimate of where and how strongly grammatical gender
is represented, with a split-half estimator that removes a noise bias the standard estimator
carries (it turned a "markedness" finding at 25 of 27 layers into 0). (2) Evidence that the
representation is partly shared across three languages and replicates in a second model family.
(3) A largely null causal answer: grammatical gender does not reliably route into social-gender
readouts. (4) Practical findings for others: an outlier dimension in Gemma 3 that breaks
geometric measures, and a training-sample property that mimicked a frequency effect.

---

## 4. Discussion

**Representation: robust.** Every representation-level verdict holds across Qwen3 sizes and in
Gemma 3: gender at the noun's own token, grammar not spelling, head-driven compounds, no
markedness, partial cross-language sharing. These are the paper's firmest results.

**Causal routing: mostly no.** If grammatical gender were represented *as* social gender, pushing
it should move social readouts in every model. It does so in one of four (Qwen3-8B), with a small
effect, and only in Spanish (German steering never had a valid test). One model out of four is
best read as a model-specific property. This sits comfortably beside the human picture: effects
in people appear mainly when gender is made salient (Elpers et al.'s induction design), and our
steering is a strong, artificial version of making it salient, yet it still mostly fails to reach
social meaning.

**The unsteered article pull.** A feminine article on an invented thing-noun pulls Qwen3 toward
feminine-rated adjectives in Spanish at every size tested (1.7B–14B), and survives the sentence making the noun a
*thing*. It is closest in spirit to Mihaylov & Shtedritski's output-level associations. Two
readings remain: the model treats grammatically feminine things as socially feminine, or it has
learned how feminine nouns are described in Spanish text (a distributional habit, not a social
one). The German version separates models (EuroLLM and small Qwen3 yes, Qwen3-8B no), so the
habit, whatever it is, is language- and model-specific.

**What the frequency episode teaches.** The German gender direction in Qwen3 pointed toward rarer
words, robustly within its 91 training nouns, through four controls. It did not appear in held-out
nouns, in Russian, or in Gemma. A within-sample result that survives every control can still be a
property of the sample. We report it because the path matters to anyone fitting directions on
small, curated word lists.

**Methods lessons.** Split-half estimation changed a headline. A pooling readout we relied on
failed its own neutrality check (P29) and was withdrawn. Gemma's outlier dimension produced
impossible cosines until fixed by a pre-registered rule. Each was caught because a check was
written down before the data.

**Limits.** Linear tools only; two model families; base models (instruct replication under way);
Glasgow norms are UK English ratings applied through translation; German steering untested;
small training sets. Full list in §5 of the methods/results draft.

**Open directions.** The instruct replication (stage A pilot: verdicts unchanged; an English
chat instruction weakens cross-language sharing); French and Dutch; what Qwen3-8B does
differently; and the character study (`character/`), which asks what chat models *say*.

---

## Citation notes for the PI

- Use **Mihaylov & Shtedritski 2024** (arXiv 2407.09704; Mihaylov is first author), numbers
  54.3–69.0%.
- Boroditsky et al.: cite the 2003 chapter; the linked escholarship item is the 2000 CogSci paper,
  which has no adjective-generation task.
- Not yet verified: Elpers Exp 2 n (189 recruited vs "~150"), Samuel 2019's primary figures.
- Literatures still missing a citation (no IDs added, to avoid unverified references): split-half /
  cross-fitting estimators, LEACE and erasure limits, frequency/anisotropy in embedding geometry,
  outlier dimensions in transformers.
