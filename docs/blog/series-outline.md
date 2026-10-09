# Blog series: outlines

*Bullet outlines for a post series; the phrasing is the PI's. Stats terms get a one-line gloss
in brackets **[like this]**: that's the list to define or look up. Numbers are from the repo as
of 2026-10-09; sources in brackets point to where each story is documented in full.
Assumed known: tokens, parameters, vectors, steering.*

Suggested order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9, then 10–11 as results come in. Posts 4
(the noun funnel) and 5 (frames) stand alone well for a general audience.

---

## 1. The question: do bridges have a gender in a language model's head?

- The setup: *die Brücke* (f, German) vs *el puente* (m, Spanish). For objects, gender is pure
  grammar: nothing about a bridge is female.
- The human claim (Boroditsky and colleagues, early 2000s): German speakers call bridges
  "elegant", Spanish speakers "strong". Language shapes thought ("linguistic relativity",
  neo-Whorfianism).
- The catch: it never fully replicated.
  - The original was described in a book chapter without full statistics.
  - Mickan, Schiefke & Stefanowitsch 2014 tried twice and found nothing reliable.
  - Effects show up mostly in tasks that invite people to think about sex explicitly.
- Our version: not "do speakers think differently" but "inside a model trained on lots of
  German and Spanish text, does the *grammatical* gender signal (learned from objects) overlap
  with *social* gender (men/women)?"
- Why a model is a good place to ask:
  - You can read its internal state directly.
  - You can intervene: push a noun's representation and watch what changes.
  - There's no experimenter effect and no participant guessing the hypothesis.
- Closest prior work: Flint & Ivanova 2024 (CogSci), the same question in older word-embedding
  models (fastText, BERT). They found an effect in Spanish that was weaker in German, and named
  LLMs as the next step. Earlier: Kann 2019, Gonen et al. 2019 (embeddings) [sources/SUMMARIES.md].
- The plan in one line: five gated phases.
  - Does it know the genders?
  - Can we get spelling out of the way?
  - Find a Spanish direction, then a German one.
  - Is it shared across languages?
  - Does pushing it move social-gender content?
- Each phase produces a usable result even if the next never runs.

## 2. Ground rules before any data

- **Pre-registration** [writing down the test, the threshold and what counts as a result
  *before* seeing data, so you can't move the goalposts]. Every choice is logged with its date
  and reason (`docs/decisions.md`: dozens of entries).
- Gold labels never come from a language model, including the one helping write the code.
  Genders come from Wiktionary dumps.
- Drop rather than review: if a word would need a human to check it, drop it. Small, clean,
  automatically filtered sets beat big messy ones (sets up post 4's joke).
- **Held-out test sets** [items the method never sees while being fit, so a good score can't
  just be memory] are frozen before training, and code asserts they never leak (they leaked
  once anyway; see post 4).
- **Don't tune on the outcome**: frames and thresholds are fixed on linguistic grounds, then
  *checked* on two model families (Qwen3 + EuroLLM).
- Base models only; one device and precision per model; no comparing activations across
  hardware (the "same-chip rule").
- Honest-reporting habits:
  - **Confirmatory vs exploratory** [a confirmatory test was declared in advance; exploratory
    results are interesting but not evidence until re-tested].
  - Report findings that cut against what we built.

## 3. Does the model even know the genders? (Phase 0)

- **Scoring vs sampling** [we never let the model write; we read the probabilities it assigns
  to fixed choices; one pass is deterministic, so no repeats needed].
- Method: score *Aquí hay **un**/**una** mesa* both ways; the **margin** [log-probability
  difference between the two versions] says which gender it prefers.
  - **Log-probability / nats** [log of a probability; 1 nat = a factor of e ≈ 2.7; 2.3 nats =
    10×].
- Frame lessons (preview of post 5):
  - Score the noun *given* the article, not the article: German *ein/dem* also serve neuter,
    so they're likelier a priori.
  - Avoid frames where the article has another meaning (Spanish *la* = "her").
  - Qwen3 adds no start-of-text token, so the tested word can never come first.
- "Known" = a clear preference (≥ 1 nat) in two different frames.
- Results: ~90%+ known for small Qwen3 models.
- Fun twist: on the frozen lists, *bigger* models "know" fewer words.
  - The cause is **regression to the mean** [items picked for scoring high on a noisy measure
    score lower next time, by chance alone]: the lists were selected on the small models
    clearing a noisy threshold.
  - On the full pools, bigger models know more.

## 4. The great noun funnel: from 50,000 nouns to a few hundred

*The joke post. Every filter has a story; most were caught by the PI or by a result looking
too good.* [docs/screening/ has one doc per filter system]

- Start: Wiktionary has ~52,000 German and ~54,500 Spanish nouns with gender.
- End: Spanish training set 165 nouns; German 91 (m/f) + 31 neuter; 271 flipped pairs.
- **Why prune so hard:** every filter removes a way for a "gender direction" to secretly be
  something else (spelling, animacy, meaning, frequency, English).
- **Spelling is the big one:**
  - Spanish *-a* is mostly feminine and *-o* masculine, so a "gender detector" could just be
    an "-a detector".
  - The fix: compare genders only *within* the same ending (**stratification**
    [splitting data into groups and comparing only inside each group]).
- **Animacy** (we want objects, not people or animals):
  - Plants: a first version used WordNet's "organism", which threw out *Kartoffel* "potato",
    *Zwiebel* "onion" and *Kapitel* "chapter". It was narrowed to people and animals.
  - Meat and fur: WordNet lists *quail* the meat and *sable* the fur first, so animals slipped
    through. Fix: check the first two senses.
  - **Cucumbers and dishwashers** (the PI's catch, flipped pairs): the two-sense check then
    over-excluded.
    - Latin-name / "species" glosses made *cucumber*, *pumpkin*, *fir* look animal-like. Rule
      v2: plants are inanimate unless WordNet also says person/animal.
    - WordNet lists "a laborer who washes dishes" before the machine, so *dishwasher* was a
      person. Rule v2: first sense only, unless the gloss mentions meat or fur.
    - Restored 11 pairs: pumpkin, cucumber, fir, valerian, dishwasher, slipper, world, talent,
      patronage, cancer (+ drill).
    - A more aggressive fix was rejected: it readmitted real groups (audience, guild, synod),
      and tuning further would fit the rule to the items.
  - Groups: *das Heer* "army", *das Rudel* "pack" got a group filter.
- **German-specific traps:**
  - Compounds take the last part's gender (*das Herrenhaus* ← *das Haus*, though *der Herr*).
    An over-inclusive compound check drops ~11,000 words, and the compounds become a test.
  - Suffixes that predict gender perfectly (*-ung* f, *-chen* n) go to their own test set.
  - Agent nouns: *der Bohrer* is both "drill" and "driller", masculine by rule, often a person.
  - Place names are all neuter, chemicals mostly neuter, and neuter is more often borrowed
    (44% vs 28% of masculines are loans).
- **English overlap:**
  - Words that *are* English (*Machete/machete*) or also mean something in English (*Brief*
    "letter" vs "brief") are dropped.
  - The cognate saga: 63 English cognates (*restaurante*, *Veranda*) were dropped without
    sign-off, then a devil's-advocate review reversed it (326 → 389 pairs). Dropping them cost
    power and balance, and their effect can be measured instead.
    - **Statistical power** [the chance a real effect shows up as significant; more items =
      more power].
- **Sex-associated concepts:** garments like *bodice*, *corset*, plus jewellery and makeup,
  since their gender association is social, not grammatical.
- **The *el agua* artifact:**
  - Feminine nouns starting with stressed *a-* take *el* (*el agua*). They were excluded to
    keep article frames clean.
  - That left word-initial *a-* as a perfect *masculine* cue among the training nouns. A
    sparse-autoencoder feature exposed it, and it was fixed by amendment before the big run.
- **The leak:**
  - The Phase 2 builder never excluded the Phase 4–5 test words, so 37 flipped pairs (and the
    classic *llave* "key") were in training.
  - Lesson: held-out rules need code, not prose. Every builder now asserts against every
    earlier training set.
- **Russian:**
  - The code read gender from the wrong part of the headword.
  - Stressed spellings (accent marks) made 79% of nouns look animate.
  - After fixes: 245 → 174 nouns the model knows.
- **The flipped-pair funnel:**
  - Translation pairs: 2,217 → opposite genders: 519.
  - PI review in English and Spanish, about 10% sense mismatches (*Platte*/*apartamento*
    "flat"): → 389 after all drops.
  - Both nouns known by both laptop models: **271**.
- **Nonce words:** Wuggy (a pseudoword generator) made 1,389 candidate stems → 837 passing
  every check → 150 kept. *-s* was dropped as an ending (it's the plural marker); in English,
  *-s/-ing/-ed* look-alikes were dropped.
- **Adjectives:** Glasgow Norms 5,553 words → 803 adjective-dominant → 739 common → 656 a
  single Qwen3 token.
- Punchline: thousands in, a few hundred out, and each cut is one fewer way to fool ourselves.

## 5. Choosing the sentence: frame selection

*The friend's interest. How the wording of one prompt can make or break a measurement.*

- Why frames matter: a frame is the sentence around the word. A bad frame measures the frame
  (priors, ambiguities), not the word.
- **Phase 0 lessons** (post 3), plus: a frame is only "known" if *two* different frames agree.
- **The pronoun leak in the Spanish verb frame:**
  - Homographs like *camino* ("path" / "I walk") need a verb frame. The first used
    *yo*/*usted* ("I"/"you"), and that gave away person, and with it the -o/-a ending.
  - Replaced by *siempre ___* ("always ___") and checked on two model families.
- **Readout position:**
  - Read at the word's own last token (LAST) or at the next token (AFTER)?
  - Both were pre-registered. A result counts as "position-robust" only if it holds at both.
- **The steering gate** (Phase 5): after pushing a noun toward the other gender, does
  grammatical agreement follow?
  - **The constraints**, all of them German's fault:
    - The gendered word must come *after* the noun (we edit the noun's position; the model
      reads left to right).
    - Nothing before the noun may give its gender away, and every German singular article
      does. So zero-article contexts only: dictionary headwords, topic lines.
    - The read word must be unambiguous: *die* is also plural; *sie* is "she", "they" and
      formal "you".
  - Spanish is easy: *Mi mesa es muy ___*. *Mi* "my" is gender-neutral; predicate adjectives
    agree (*blanco/blanca*). It passes perfectly on every model.
  - **The German saga:**
    - Dictionary *Brücke, die*: masculine nouns lose to *die* because it's also plural. Fails
      on 4B (67% vs the 70% bar).
    - Pronoun *Thema: Brücke. Sie…*: sentence-initial *Sie* is ambiguous. Fails badly.
    - Label *Genus: feminin*: in real text "Genus:" is followed by abbreviations. Fails.
    - Five candidates declared in advance with a selection rule, all tested:
      - the *mit ihm/ihr* pronoun frames lean masculine (*dem/ihm* also serve neuter: a
        frequency prior);
      - the forced relative pronoun (*Brücke, mit der…*) works on EuroLLM only;
      - the winner is the dictionary frame with **contextual calibration** [subtract the answer
        the frame gives a content-free input, "N/A", to cancel its built-in bias toward one
        answer; Zhao et al. 2021].
  - Why testing many frames wasn't cheating:
    - fixed candidate list and rule;
    - selection on validity (does the frame measure gender at all?), on nouns we don't steer;
    - every failure reported.
- **Wording robustness** [does the result survive rephrasing?]:
  - Three wordings per main readout, averaged (*The bridge is very / was very / I think the
    bridge is very ___*).
  - Plus the PI's consensus contrast, *Everyone says…* vs *I think…*.
- Your naturalness check as a Spanish speaker (*El puente es muy ___* vs *Es un puente muy
  ___*). Both are fine. The pre-registered frame check decides whether a frame *works*.

## 6. Can we erase spelling? (Phase 1)

- Goal: remove the ending signal so whatever's left is gender.
- **LEACE** [a closed-form method that removes all *linearly* readable information about a
  label from vectors, by making the label's class means equal; Belrose et al. 2023]. It was
  trained on Spanish verbs, where *-o/-a* marks person, not gender (*hablo/habla*).
- Test: invented words (*flitra/flitro*). After erasure, can a fresh probe still read the ending?
- A bug we caught in our own test: testing the eraser with a probe trained on the eraser's own
  data gives chance *by construction* (LEACE makes class means equal → the optimal weights are
  zero). Erase-all-then-cross-validate even goes *below* chance.
  - **Cross-validation** [fit on part of the data, test on the rest, rotate].
  - **Probe** [a small classifier, here logistic regression, trained to read a property from
    vectors].
- Result: the eraser doesn't generalise to new words; their endings stay readable.
- So: control spelling by *design* (post 4's stratification), not by erasure.
- Later twist (post 7): the same "failed" eraser has a meaningful effect on homographs. Partial
  erasure is still erasure.
- Status: shown on 0.6B–4B (exploratory); the 8B/14B confirmatory analysis is still to run
  (the cloud data is saved).

## 7. Finding a Spanish gender direction (Phase 2)

- A "gender direction":
  - **Difference of means** [average feminine vector minus average masculine vector].
  - **Probe direction** [the weights of a logistic regression separating the genders].
    - **L2 regularisation** [a penalty on large weights; keeps the fit stable when there are
      more dimensions than examples].
- **Residualising** [removing what a set of nuisance variables (ending, frequency, token
  count, concreteness, loan status) can explain before fitting, so the direction can only use
  what's left; the Frisch–Waugh–Lovell idea].
- Tests where spelling and gender disagree:
  - masculine *-a* (*el día*, *el problema*, Greek *-ma* nouns), feminine *-o* (*la mano*),
    clippings (*la foto*);
  - noun/verb homographs: *camino* as "path" vs "I walk".
- **AUC** [probability a random feminine item scores above a random masculine one; 0.5 =
  chance, 1 = perfect].
- **Bootstrap confidence interval** [resample the data with replacement many times, refit,
  and take the middle 95% of results as the uncertainty range].
- The attention-sink bug: single-token words sat on a position with one giant activation
  value (~180× normal) that swamped everything. Fixed with a neutral prefix (`<|endoftext|>` +
  newline).
- First design underpowered (25 nouns per gender), so the **stratified design**: every noun in
  ending cells with both genders.
- **Confirmed (8B/14B, A100):** the direction reads gender, not spelling, in 28/35 and 26/39
  layers. No layer in any model reads spelling.
- **But not position-robust:** at the next token it vanishes in every dense model. The gender
  lives on the noun itself. Exception: the 30B mixture-of-experts model (exploratory).
- Homographs: the spelling-erased direction reads more gender when *camino* is a noun than a
  verb, in every model size.

## 8. German, three genders, and a statistical trap (Phase 3)

- **Compounds follow their head:** a direction trained on single-root nouns sorts unseen
  compounds by their last part's gender, even when the first part has the other gender. Every
  layer of every model, and never by the first part.
- **Markedness hypothesis:** is masculine the "default", i.e. its vector shorter?
- **The trap:**
  - The plug-in squared length of a mean difference is biased upward by noise, more so for the
    smaller group: E‖v̂‖² = ‖v‖² + noise × (1/n₁ + 1/n₂).
    - **Bias** [systematic error that doesn't shrink with more repeats].
    - **Plug-in estimate** [compute the statistic directly on the sample means].
  - The naive comparison "confirmed" markedness in almost every layer. It was noise.
- **The fix, cross-fitting / split-half** [split the items in two; noise in one half is
  independent of the other, so the dot product of the halves' estimates is unbiased].
  - Calibrated against a **within-cell shuffle null** [shuffle the labels inside each spelling
    cell to see what "no real difference" looks like, with everything else equal].
  - After the fix: 0 layers support markedness, in any model.
- A second subtlety: bootstrap resamples repeat items, and copies split across halves share
  noise. Caught because an interval didn't contain its own point estimate.
- The cleanest methods lesson of the project: a standard estimator would have handed us a
  false confirmation.
- Suffix puzzle (follow-up, pre-registered): *-ung*/*-heit* nouns sit only a third of the way
  toward feminine. It's partly a shared "derived noun" offset; confirmation is pending.

## 9. Research on a laptop plus $1.80 of cloud time

- The cloud run: one rented A100, six models (0.6B–30B), ~1.1 hours, **$1.80**.
- The precision check missed its pre-registered bar (98.6% vs 99%). That's rounding noise at
  a hard threshold, reported as a miss anyway.
  - **bf16/fp16** [half-precision number formats; fast, but they round small log-probability
    differences].
- The real bottleneck: laptop analysis estimated at ~60 hours.
  - Speedups: fitting in a smaller space with the same answer, exact rank-based AUC, batched
    matrix maths.
  - Two bugs found on the way (a crash on a degenerate layer; the bootstrap duplicates).
- Laptop logistics: eight parallel jobs filled 23 GB of swap and drained the battery faster
  than the 18 W charger could refill it. Memory, not cores, is the limit.
- Before the next cloud run: a steering engine 8–30× faster than the plain method, verified
  identical.
  - Compute the layers below the edit once.
  - Batch many conditions together.
  - Compute the vocabulary only where it's read.

## 10. Is gender shared across languages? (Phase 4, design; results pending)

- The sharpest test, **flipped pairs**:
  - A shared *grammatical* direction scores *Brücke* feminine and *puente* masculine.
  - A *meaning* direction scores both the same (it's one bridge).
- Tests:
  - **T1 transfer:** a Spanish-trained direction sorts German nouns, and vice versa, within
    spelling cells.
  - **T3 pairs:** one direction scores both nouns of each pair.
    - **Paired AUC** [compares score differences within pairs; a constant offset between the
      languages cancels].
  - Plus Russian as a third language, and where German neuter falls.
- **Multiple comparisons / Holm correction** [testing several things inflates false
  positives; Holm tightens the bar for the first test and relaxes it step by step].
- A deviation, logged: no fresh test nouns existed (all were already in training), so each
  direction is tested on the *other* language's training nouns.

## 11. Does pushing grammar move social gender? (Phase 5, design; results pending)

- **Steering at the noun only** (not every position: that would just prime the whole output).
  Every-position steering is a secondary test.
- **The gate:** agreement must follow the push before any social readout counts (post 5).
- Readouts:
  - English adjectives rated for gender by humans (**Glasgow Norms**).
  - Personification: "would it be a man or a woman?"
  - The PI's in-language comparison: *El puente es muy ___* vs *Die Brücke ist sehr ___*.
- The valence confound:
  - Feminine-rated adjectives are much more positive (gender–valence **correlation** r = −0.47
    [−1 to 1, how tightly two measures move together]).
  - So the main statistic is a **regression with covariates** [predict each adjective's shift
    from its gender rating while holding valence, arousal, size and frequency fixed].
- Controls:
  - 100 **random directions** [the null: how much does any push of this size move things?].
  - A singular/plural direction ("any grammatical push").
  - A social-gender direction as **positive control** [a push that *should* work; if it
    doesn't, the readouts are too insensitive to trust a null].
  - **Damage**: **KL divergence** [how far the steered next-word distribution drifts from the
    unsteered one; measures collateral damage].
- **Dose-response** [effect at several push sizes; a real effect grows with dose].
- Side quests:
  - **Gender wug test:** do models give invented *-a* words feminine articles? Yes, even 0.6B
    (~80% vs ~5% for *-o*). No model was too small, so the joke result didn't happen.
  - The model's own woman/man axis, validated against human ratings (**Spearman** [rank
    correlation]; **partial correlation** [with valence held fixed]).
  - English base case: does *bridge* already lean toward *puente*'s or *Brücke*'s gender?
    Flipped pairs give opposite predictions, so shared meaning can't fake it.

## Running thread: the mistakes ledger

*Could be its own post, or a box in each.*

- the circular eraser test (post 6)
- the attention-sink bug (7)
- the pronoun leak in the verb frame (5)
- the *el agua* initial-a artifact (4)
- the markedness noise bias (8)
- bootstrap duplicates across halves (8)
- the training leak (4)
- the cognate drop without sign-off (4)
- the over-eager animacy rule (cucumbers) (4)
- Russian stressed forms (4)
- the label frame that failed (5)
- Phase 1 never queued on the cloud data (6)
- Each one is a date-stamped entry in `docs/decisions.md`, with what was done about it.
