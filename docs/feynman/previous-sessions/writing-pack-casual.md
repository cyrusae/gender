# Writing pack: thinking about the project in easy doses

**Prepared:** 2026-10-21 · Purpose: short, digestible on-ramps so you can write from either of
your two states — deep in the weeds, or cold after a break. Nothing here is new science; it's
the same project re-arranged for humans, starting with the least technical.

---

## 1. Elevator pitches

**Version for a normal person (20 seconds):**
> German says *die Brücke* — "the bridge" is feminine. Spanish says *el puente* — masculine.
> For objects this gender is pure grammar; a bridge isn't female. A famous, never-quite-
> replicated psychology claim says this grammar rubs off on how we think: German speakers call
> bridges "elegant," Spanish speakers call them "strong." We asked the same question inside an
> AI language model, where you can do something you can't do with people: reach into its
> innards, find the little arrow that encodes "feminine," push a bridge along it, and watch
> whether the model starts describing bridges with words people rate as feminine. On an
> 8-billion-parameter model, it does — weakly. On the bigger 14-billion model, it doesn't.

**Version for a colleague (30 seconds):**
> We extract grammatical-gender directions from inanimate Spanish/German nouns in Qwen3 base
> models with spelling, frequency and concreteness controlled, then steer nouns along them and
> read English adjective probabilities rated for gender by humans. Headline: at 8B the push
> moves feminine-rated adjectives about a third as much as a woman-minus-man push; at 14B
> nothing moves although the same push still flips the article. Unsteered, a feminine article
> on a made-up noun read as a *thing* also pulls toward feminine-rated adjectives on every Qwen3
> size but not on EuroLLM. We also invalidated two of our own pooled readouts with a
> pre-registered control, and we document an estimator bias that would otherwise have
> "confirmed" a false hypothesis.

## 2. The whole project in 5 minutes (plain English)

**The question.** In some languages every noun has a gender. For people and animals that makes
sense (*la mujer* is feminine). For objects it's random — a bridge has no gender, yet German
made it feminine and Spanish made it masculine. A famous idea from psychology: maybe that
random grammar leaks into how speakers *think* about bridges (elegant vs. strong). It was
widely reported and never solidly replicated.

**Our version, aimed at the model instead of the person.** An LLM learned gender from texts,
so ask *it*: is the representation of grammatical gender (learned from objects, where gender
is pure grammar) entangled with *social* gender (men/women)? And unlike with humans, we can
check *causally*: find the internal arrow for "feminine," push a bridge along it, and see what
the model starts saying.

**How we make sure we're measuring gender and not junk:** gold labels from a dictionary, not
from the model; the model must demonstrably "know" each noun's gender before we use it; and we
control for the boring stuff — spelling (*-a*/*-o* = the biggest fake cue), frequency, length,
and how concrete the word is. Every test was written down *before* we looked at results, and
every decision (including mistakes) is logged with a date.

**The five steps.** (0) Does the model know the genders? Yes, ~90%+. (1) Can we erase the
spelling cue? No — erasers don't generalize, so we design around spelling instead. (2) Spanish
direction exists and reads gender, not spelling (at the noun's own token, not the next one).
(3) German: directions follow the *compound's head* — a train-direction knows *das Herrenhaus*
is neuter because *Haus* is. (4) Partly shared across languages: one direction follows each
language's gender on flipped pairs and even sorts Russian nouns. (5) The big test — push a noun
along the feminine direction: at 8B, English adjective choices move toward feminine-rated words
(about a third of a woman-push; a nonsense-number push does nothing); at 14B, nothing.

**Then we hurt our own feelings (the good part).** A pooled readout looked like it showed the
effect *everywhere*, even at 14B. We pre-registered a check: does just swapping the article on
a meaningless word do it? Yes — as much as steering did. Two results dead by our own hand.
That's the project: a small honest positive at one size, a null at a bigger size, and a
methods story about how easy it is to fool yourself.

## 3. Cold-start memo ("I took two days off — what was I doing?")

Read this first, in order (~2 minutes), then you're current:

1. **Status board (Oct 17):**
   - ✅ Phases 0–4 confirmed at 8B/14B. Phase 5 verdict: Spanish-only; **8B positive, 14B null**.
   - ❌ German Phase 5: no valid test (frame failed; we refused to switch post hoc).
   - ❌ R3-POOL & P20-POOL: positive → **invalidated by P29** (bare article flip reproduces them).
   - 🎯 **P30 (the current main finding):** unsteered, *La* on a noun read as a *thing* pulls
     toward feminine-rated adjectives — 8B +0.080, 14B +0.124, EuroLLM ≈ 0, all Qwen3 sizes.
     Mechanism (social vs. text statistics): open.
   - 🔜 Next: write-up. Pending checks: confidence-direction cosine + entropy (spec exists),
     paper skeleton (exists), blog series (outline exists).
2. **Files to open, in this order:** `docs/project-outline.md` §15–16 ("where things stand",
   "threads a blog post could pull on") → `docs/explainers/05-phase5-steering.md`
   (P29/P30 deep-dive) → `docs/explainers/06-english-social-gender.md` → `docs/decisions.md`
   (newest entries) → `outputs/gender-repo-exploration.md` (my map) →
   `outputs/paper-skeleton-exploratory.md` + `outputs/confidence-direction-control-spec.md`.
3. **The one-sentence progress log** (say these aloud, in order): *We found directions in
   Qwen3 that read grammatical gender; they're partly shared across languages; pushing one
   moves English adjective choice toward feminine-rated words at 8B but not 14B; a bare
   feminine article on a meaningless noun also pulls that way; two of our fancier readouts
   didn't survive our own checks; now we're writing it up.*

## 4. Mini-glossary (writing-terms edition)

| Term | Plain meaning |
|---|---|
| nat / log-prob / margin | How much likelier one option is, in log units (1 nat ≈ 2.7×; 2.3 nats = 10×). "Known" = right article puts the noun ≥ 1 nat ahead in both frames. |
| direction (vector) | The "arrow" through the model's internal space from average-masculine-noun to average-feminine-noun. |
| cosine | How parallel two arrows are: 1 = same, 0 = perpendicular. In 4096 dimensions random arrows are ~0. Our gender×social subspace cosine ≈ 0.3; the number×social cosine ≈ 0.01 ("what unrelated looks like"). |
| AUC | If you pick a random feminine noun and a random masculine noun, how often the feminine one scores "more feminine." 0.5 = chance, 1 = perfect. We quote AUC per layer. |
| bootstrap CI | Resample the nouns many times, recompute, look at the spread — "how much would this number wobble if we'd drawn a slightly different set of nouns?" |
| split-half estimate | Compute the thing twice on two disjoint halves and cross-multiply, to kill noise bias. Mandatory for lengths/cosines (see: the markedness trap). |
| pre-registration | Writing down the test, threshold, and what counts as a result *before* seeing data. We did this for everything; it's why P29 hurt and why the hurt is publishable. |
| confirmatory vs exploratory | Declared-in-advance tests = evidence. Found-after-the-fact = "interesting, but doesn't count until re-tested." |
| Holm correction | Adjusting p-values when you run several tests, so cherry-picking one lucky test doesn't count. |
| gate / dose / α\* | Steering's "on/off" test: the smallest push that flips the article (α\*), measured in units of a typical noun state. |
| Glasgow Norms | Big human rating dataset: how feminine/masculine/pleasant ~600 English adjectives feel. Our readout labels. |
| entropy (bits) | How spread out a probability distribution is: 0 = one word takes everything, ~1.6 bits over 3 words = all equally likely. Flattening = entropy goes up. |
| confidence direction | The internal arrow from "model is sure" to "model is guessing" — the impostor that can hide inside any contrast direction (see spec). |
| Kendall τ | How much two rankings agree (1 = identical order, 0 = unrelated). Confidence-flattening keeps rankings; a real shift reorders. |

## 5. Blog series — aligned with your outline, updated to Oct 17

Your `docs/blog/series-outline.md` (posts 1–9 drafted, 10–11 waiting on results) is good — the
results now exist. Suggested updates:

- **Post 10** (*The cloud run, and the 8B/14B split*): your draft outline's material + the
  headline table. Core tension: *the same push that flips the article at both sizes moves
  adjectives only at 8B.* One number per side: R1 +0.020 vs random 95% 0.012; 14B 0.000 while
  the woman-push still moves 0.047.
- **Post 11** (*Two results I killed myself*): P29 + P20-POOL reinterpretation + P30. Core
  tension: *the check we added specifically to be careful invalidated our best-looking result —
  and the thing that survived is stranger.* Numbers: article flip alone +0.096 (8B) vs
  steering effects 0.052–0.075; P30 object reading +0.080/+0.124.
- **New optional posts:**
  - *The estimator that would have fooled us* (split-half vs naive; markedness "confirmed" in
    25/27 layers → 0 after the fix).
  - *The test that couldn't fail* (shuffle-null cosines > 1).
  - *"Is it about women, or just about how Spanish talks about feminine things?"* (P30's two
    readings — great closer, invites reader participation).
- **Suggested writing order for you:** post 4 (noun funnel — your own note says it stands
  alone) and post 5 (frames), then 10 and 11 while they're fresh; 1–3 + 6–9 when you're back in
  the immersion state. Each post opens with its elevator-pitch line above and keeps one idea
  per paragraph; every number gets its **[glossary gloss]** as your outline already does.

## 6. A tiny writing routine for the "lost in the weeds" state

1. Five minutes with §3 (cold start) → re-anchor.
2. Pick ONE claim from the claims table in `outputs/paper-skeleton-exploratory.md`; write one
   paragraph about it in plain words (ignore structure); file it under `notes/` or the blog
   draft.
3. Only then open the corresponding explainer to verify numbers.
4. If you get stuck on the math, paste the sentence you want to say + the explainer section
   into whichever agent is helping; the spec files here were written so an agent can pick them
   up cold.
