# Methods and results: first draft (2026-10-22)

*Status: draft for the PI. Framing, introduction and discussion wait on the literature review. Assembled from `docs/methods-summary.md`, `docs/explainers/00–07`, the preregistrations in `docs/design/` and `docs/decisions.md`. Numbers marked **[check]** were carried over from summaries and must be re-read from `results/` before anything leaves the repo.*

==This is partly my preference compared to how people actually write their methods sections in practice, but I'd prefer for it to be possible to understand the actual actions taken and stimuli used by reading the methods section--put cheekily, detailing the actual methods--as opposed to leaving it as an exercise for the reader. All of this information is available in other documents but if I were reading it in a paper I'd be going "okay, this is very 'and draw the rest of the owl', *how* did you do that". I want "methods" to actually be able to provide a view into "what did we *do*, in what order, and how" as opposed to being mostly lab notes for already-inculcated readers, even if that's not a standard other teams are holding themselves to as a rule. e.g., lexicon selection and stimulus design either should get their own breathing room or need an appendix, how elicitation happened wants more detail, what prompts are being used where, I want a reader to be able to read this and actually picture the broad strokes of "they did X and then they did Y" without needing to cross-reference my personal papers or take it on faith that the data Appeared Somehow. Think--I don't know, think "Labov needed to explain that he was going around to department stores performing an elicitation task with a tape recorder, not just go 'pronunciation of r was determined in the sample to...'".== 

---

## 2. Methods

### 2.1 What we did, in order

**The question.** In German *die Brücke* "the bridge" is feminine; in Spanish *el puente* "the bridge" is masculine. For an object like a bridge, the gender is pure grammar: nothing about bridges is female or male. Humans have been claimed (contestedly) to carry a little of the grammar over into how they think about objects, describing German bridges as "elegant" and Spanish ones as "strong". We ask the language-model version of the question. When a model learns which nouns are grammatically feminine or masculine, does that knowledge sit in the same place inside the model as its knowledge of *social* gender (women and men, and the words people associate with them)? If it does, pushing on the grammar should move the social side too. We call that leakage **bleed**.

**Two kinds of evidence.** We look at the model two ways throughout:

- **Inside the model ("representational").** A language model turns each word into a long list of numbers, its *hidden state*, at each of its layers. We look for a consistent direction in that space that separates feminine from masculine nouns, and ask what else it lines up with.
- **What the model does ("behavioural").** We give the model a prompt and read the probabilities it assigns to possible next words, for example whether it expects *la* or *el*, or "gentle" or "rugged". Behavioural tests are used both to check that the model knows what we assume it knows (§2.7) and to measure bleed (§2.8). We never ask the model to judge anything about itself; we only read its next-word probabilities.

**The phases.** The study ran in phases. Each one answers a question that the next phase depends on.

| Phase | Question | What it hands on |
|---|---|---|
| 0 | Does the model know each noun's gender? (an article-choice quiz, §2.7) | lists of nouns the model demonstrably knows |
| 1 | Can we just erase spelling cues (*-a* / *-o*) from the hidden states? | no: so spelling is controlled by how the nouns are chosen instead (§2.4) |
| 2 | Spanish: is there a direction that reads *gender* rather than *spelling*? | a Spanish gender direction |
| 3 | German: same question, plus compound nouns as a stress test | a German gender direction |
| 4 | Is the Spanish direction the same as the German one? (tests T1–T4) | evidence on whether there is one shared "grammatical gender" direction |
| 5 | If we push a noun along that direction, does the model's choice of socially gendered adjectives shift? (readouts R1–R3) | the main bleed result |
| P-numbered tests (P22b–P31) | follow-ups, each with its own pre-registration: English social gender, the pooled-readout check, unsteered article tests | refinements and robustness |
| late controls | does word frequency or the model's confidence explain any of the above? | caveats |

The labels T, R, P and C (and G for the Gemma replication) are just numbering schemes: T for Phase 4 tests, R for Phase 5 readouts, P for follow-up projects, C for the checks that a prompt really means what we intend (§2.7). Section 2.10 lists every test with its question and verdict.

### 2.2 Ground rules

- **Pre-registration.** Every confirmatory analysis was written down and committed to the repository before its data existed (`docs/design/*-prereg.md`). This covers the hypothesis, the test, and the rule that turns numbers into a verdict. Changes were logged with reasons in `docs/decisions.md`. Analyses decided after seeing data are labelled *exploratory* wherever they appear.
- **Gold labels never come from a model.** Every noun's gender comes from Wiktionary, the crowd-edited dictionary, via machine-readable dumps (kaikki.org). How the word lists were built is described step by step in §2.4.
- **Drop, don't review.** Nouns are kept or dropped by written, automatic rules. Nobody hand-picks words or rescues a word that a rule rejected, because hand-picking lets the experimenter's expectations into the stimuli. Once a list has produced results it is frozen. If a flaw turns up later, it is logged and its effect is tested rather than silently fixed (an example is in §2.9).
- **Held-out words stay held out.** Words used to test a direction are fixed before the analysis and never used to build it. The code checks this.
- **Same hardware for any comparison.** Numbers compared with each other come from the same type of chip, because tiny numerical differences between chips can otherwise masquerade as effects.

### 2.3 Models

- **Main family: Qwen3**, an open family released in several sizes: 0.6B, 1.7B, 4B, 8B and 14B parameters (roughly, the number of adjustable numbers in the model), plus Qwen3-30B-A3B, a *mixture of experts* model that routes each word through only part of its network.
- **Second family: EuroLLM-1.7B**, trained with more European-language text. It is used for the behavioural tests, as a check that a result isn't a Qwen quirk.
- **Third family, in progress: Gemma 3** (4B and 12B), a slimmed-down replication of the key results (the "slim" design, `docs/design/gemma-slim-prereg.md`).
- **Base models only.** These are models trained only to predict the next word, before any chat or instruction tuning. Instruction-tuned models are a planned follow-up.
- **Hardware and number format.** Confirmatory runs used bf16 numbers (a compact 16-bit format) on rented single-GPU servers (A100 or L40S). Small models were also run on an Apple-silicon Mac.

### 2.4 Stimuli

Choosing the words is most of the work in a study like this. Think of Labov's department-store study: the result depended on choosing stores and a question ("fourth floor") so that everything except the variable of interest was held fixed. Here, the variable of interest is a noun's grammatical gender. The danger is that some *other* property of the noun travels with its gender and gives the model a shortcut. The obvious shortcut is spelling: Spanish nouns ending in *-a* are usually feminine, and those ending in *-o* usually masculine. A direction that "reads gender" might really just read the last letter. Others are frequency, meaning (people versus things), being a loanword, and being a compound. This section walks through how each list was built, in the order it happened. Full step-by-step records, with counts at every filter, are in `docs/screening/01`–`07`.

#### 2.4.1 The dictionary everything starts from

1. **Source.** All nouns come from Wiktionary, via the machine-readable dumps at kaikki.org. Each entry gives the word, its grammatical gender, English translations ("glosses"), and often its etymology. Gender is taken *only* from the dictionary, never from a model.
2. **Eligibility.** A noun is kept only if:
   - it has exactly one gender (so not Spanish *el/la mar* "the sea", which takes either);
   - it is not marked archaic, slang or regional;
   - it is not mainly some other part of speech, or an inflected form of another word;
   - it is inanimate: not a person, animal, group of people, or other living thing (how that is judged is explained in step 4);
   - it is not more common as an English word. Otherwise its frequency would come from English text (*van*, *arcade*).
3. **Measures attached to every noun.** Each one is computed automatically from a published resource:
   - **Frequency** is the *Zipf* score from the `wordfreq` library: log₁₀ of how many times the word occurs per billion words of text. Zipf 3 is about one in a million words, and each extra point is ten times commoner. Frequency matters because models know common words better.
   - **Token count** is how many pieces the model's tokenizer splits the word into.
   - **Concreteness** asks whether the word's main dictionary sense is a physical thing, using the WordNet lexical database, reached through the English gloss. For Russian, where this check was too coarse, we also used published human ratings of how concrete the English translation is (Brysbaert norms).
   - **Loanword status** records whether Wiktionary's etymology says the word was borrowed (German *Büro* "office", from French). Loans differ from native words in spelling patterns and in which gender they get.
4. **Animacy, checked more than once.** Deciding "is this a living thing" from an English gloss misses cases. German *Wachtel* means "quail" but the gloss judge missed it; *Heer* "army" is a group of people. So every list gets a second, independent check of the meanings in WordNet (a person, animal or group in the first two senses means the noun is dropped). For Russian a third check uses the grammar itself: Russian marks animate nouns differently in some cases.

#### 2.4.2 Spanish (Phases 2 and 4–5)

The goal was nouns whose spelling gives away nothing about their gender.

1. **Drop every ending that predicts gender.** This removes nouns in *-o* and *-a*, and also in *-ción*, *-dad*, *-umbre*, *-ez*, *-itis*, *-or*, *-aje*, *-ón*, *-ista*, *-ante*, *-ud* and a few others. Each of these goes almost entirely one way. What remains are nouns like *el puente* "bridge", *la llave* "key", *el árbol* "tree" and *la cárcel* "prison", whose endings are shared by both genders.
2. **Drop nouns too close to English** (spelling nearly identical to the gloss, or common as English words). Otherwise the model might be reading an English word.
3. **Drop nouns whose meaning is tied to sex**, such as clothing, jewellery, cosmetics and body parts. Their gender could have a social reason rather than being arbitrary.
4. **Second meaning check:** drop people, animals, groups, places and chemicals. Chemical names are almost all masculine, so they would be a shortcut.
5. **Frequency floor:** Zipf ≥ 2, so the model has a fair chance to know the word.
6. **The model must know the gender** (the article quiz in §2.7).
7. **Group what is left into spelling cells.** A cell is the last two letters, plus loan status. *-te* nouns go together, *-al* nouns together, and so on. Feminine and masculine nouns are only ever compared *within* a cell. A difference that survives cannot be the ending, because the ending is the same on both sides.

The resulting sets:

| set | what it is | size |
|---|---|---|
| **stratified training set** ("strat") | the nouns from step 7, used to *find* the gender direction | 33 f, 132 m |
| **matched training set** | a stricter subset with equal numbers of each gender in each cell | 27 f, 27 m |
| **regular test nouns** | more nouns of the same kind, never used for training | 247 |
| **exceptions** | nouns whose gender goes *against* their ending: *el día* "day" (masculine despite *-a*), *la mano* "hand" (feminine despite *-o*), plus Greek-origin *el problema* "problem" and shortened words like *la foto* "photo". If a direction followed spelling, it would get these wrong. | 7 true exceptions, 20 *-ma*, 2 shortenings |
| **noun/verb homographs** | nouns that are also verb forms: *camino* is "path" (masculine noun) and "I walk". These test whether the direction follows the noun reading. | 72 |

Two weaknesses turned up and were handled by rule, not by hand:
- Words beginning with *a-* were slightly more often masculine in the training set (0 of 33 feminine, 17 of 132 masculine). This was added as a variable to remove (§2.6).
- 37 nouns that were later chosen as flipped-pair nouns (§2.4.4) had also been in training. They were removed from the later tests.

#### 2.4.3 German (Phases 3 and 4–5)

German has three genders, *der* (masculine), *die* (feminine) and *das* (neuter), and its spelling traps are different:

- **Compounds.** German writes compound nouns as one word. A compound always takes the gender of its *last* part, which is called its **head**, because the last part is what the thing is. *Die Tür* "door" + *der Rahmen* "frame" gives *der Türrahmen* "door frame", a kind of frame. So a compound's gender is visible from its last part, which is a shortcut. Compounds were removed from training, and saved for a test (below).
- **Suffixes that fix gender.** Every *-ung* noun is feminine, and every *-chen* noun is neuter. These were removed.
- **Loans are more often neuter** (44% versus 28% of native nouns). Loan status is part of the cells.
- **Agent nouns.** Nouns in *-er* made from verbs (*Bohrer* "drill", from *bohren* "to drill") are nearly all masculine. These were removed.

The filters, in order:
1. Drop compounds. Any noun whose last three or more letters, following at least three letters, form another noun is treated as a compound. This alone removes about 11,000 nouns.
2. Drop proper names, diminutives, chemicals, and *Ge-* collective nouns.
3. Drop living things and groups, as in §2.4.1.
4. Drop agent nouns, nouns made from verbs, and sex-typed clothing.
5. The model must know the gender (§2.7).
6. Group into cells: last two letters × loan status, with extra cells for shared suffixes such as *-nis* and *-tum*.

| set | what it is | size |
|---|---|---|
| **stratified training set** | single-root nouns from the steps above | 64 m, 27 f, 31 n |
| **matched training set** | equal numbers per gender within cells | 14 each |
| **compound test set** | compounds never seen in training. 219 of them are **conflict** items, where the first part and the head have different genders: *Türrahmen* has a feminine first part (*Tür*) and a masculine head (*Rahmen*). A direction that follows the head gets these right. One that follows the start of the word or the general look of the word does not. | 720 (219 conflict) |

A known flaw is described in §2.9: 12 of the 64 masculine training nouns turned out to be compounds or number-words the compound filter missed (*Wasserkocher* "kettle", *Vierer* "a four"). None of the feminine ones were.

#### 2.4.4 Flipped pairs: the "bridge" nouns (Phase 4–5)

These are the nouns that pose the original question: the same object has opposite genders in German and Spanish. *Die Brücke* "bridge" is feminine; *el puente* is masculine.

1. All eligible German and Spanish nouns with Zipf ≥ 2.5 were paired through their English gloss: German *Brücke* "bridge" with Spanish *puente* "bridge". Where several nouns shared a gloss, the most frequent was taken.
2. Pairs with opposite genders were kept (519 of 2,217 pairs).
3. Pairing by gloss is error-prone, because a gloss can name a different sense of the word. A spot check found that about 1 in 10 pairs do not really mean the same thing. Pairs were dropped by rule (people, animals, groups, sex-associated meanings, nouns too close to English, and nouns already used in training), and the PI dropped 51 more for wrong meanings. That left **389 pairs**: 239 feminine in German and masculine in Spanish, and 150 the other way round.
4. Separately, **19 classic examples** from the human literature (*Brücke/puente*, *Mond/luna* "moon", *Sonne/sol* "sun", …) were included. These are the one hand-picked set, chosen because readers will ask about them. They are always reported separately.

#### 2.4.5 Russian (prepared, not yet run)

Most Russian endings fix gender almost perfectly. The exception is the **soft sign** *-ь*, a letter that is not pronounced itself but softens the consonant before it. Nouns ending in *-ь* can be masculine (*день* "day") or feminine (*дверь* "door"). Our Russian test therefore uses only *-ь* nouns, so the ending cannot give the gender away. After the same kinds of filters as above (and a few Russian-specific ones, such as suffixes that do fix gender within *-ь*), 245 nouns remain (123 m, 122 f) in 10 spelling cells.

#### 2.4.6 Words read out in the behavioural tests (Phase 5 and later)

To measure bleed we need words with known *social* gender associations, scored by humans:

- **English adjectives** come from the **Glasgow Norms** (Scott et al., 2019). In that published study, UK participants rated thousands of English words on several 1–7 scales, including:
  - **gender association**: how strongly the word is associated with women versus men ("gentle" scores toward feminine, "rugged" toward masculine);
  - **valence**: how pleasant;
  - **arousal**: how exciting or calming;
  - **size**: how big the thing it describes is.

  We kept words that are mainly adjectives (at least half their WordNet senses), common (Zipf ≥ 3), and a single token for the model, so each one is a single next-word prediction. That leaves several hundred adjectives per model family [check: Qwen3 count] (721 for Gemma 3).
- **Why the other ratings matter.** In these ratings, feminine-rated words are also rated more pleasant (correlation −0.47). Any test that sees "more feminine adjectives" could therefore really be seeing "more pleasant adjectives". So valence, arousal, size and frequency are always included as variables to separate out (§2.8).
- **Spanish adjectives.** For one readout (R3) we used Spanish adjectives with the same form in both genders (*fuerte* "strong", *elegante* "elegant"), so that the grammar of the sentence cannot favour one of them.
- **Invented words** (*the robedo*, *the sianta*) were built from Spanish- or English-like syllables and checked to exist in no dictionary. They were used where we need nouns with no meaning at all (P30, P31; details in `docs/screening/07`).

### 2.5 Two kinds of measurement, and how the noun is shown to the model

The study measures two different things, and it helps to keep them apart:

1. **What the model represents (Phases 1–4).** No sentence and no adjectives are involved here. We show the model a single noun and record the numbers it computes internally for that noun, its **hidden state**. Then we ask whether those numbers contain a consistent "gender direction" (§2.6). This is like scanning a brain while a person reads one word, without asking them to do anything.
2. **What the model does (Phase 0, Phase 5 and P22–P31).** Here the model reads a sentence, and we look at which next word it predicts, and how strongly. This is like asking a person to finish a sentence. The adjectives from §2.4.6 come in only here, as the possible next words we score. The prompts are given in §2.7 and §2.8.

**How the noun is shown, for kind 1.** Each noun is shown on its own, with no sentence around it ("bare"). The exact input is three pieces:

```
<|endoftext|>   ⏎   ␣Brücke
```

That is the model's "start of a new document" marker, a line break, then the noun with a leading space, exactly as it would appear in running text. The marker and line break are always the same, so they cannot differ between feminine and masculine nouns. The line break is there because of a quirk of these models. The very first word after the document marker can become an **attention sink**: a position where the model parks attention it does not need, giving it huge, unrepresentative values. We found this the hard way (§2.9); the line break absorbs it. A check in the code stops the run if any noun's vector is more than 10 times the typical size.

**Where the hidden state is read.** A model splits text into pieces called *tokens*; *Brücke* might be one token or two (*Br* + *ücke*). It also computes one hidden state per token at every **layer**, its successive stages of processing (28 layers for Qwen3-1.7B, 40 for 14B). We read every layer, at one of two positions:

- **LAST**: the noun's last token. Models are known to assemble a whole-word meaning at a word's last token.
- **AFTER**: a line-break token added after the noun. This is the same token for every noun, so token-shape differences between nouns might matter less there. Because the model reads strictly left to right, adding it does not change the LAST reading.

We chose between them *before* looking at any gender result, by a pre-registered rule that used no gender labels. AFTER would win only if it both kept words distinguishable from each other (checked by whether a word's vector could be matched to the same word in another context, *Beispiel: Brücke* "Example: bridge") and depended less on how the word was split into tokens. On Qwen3-1.7B and 4B, AFTER met the rule in only 1 of 4 model × language combinations, so **LAST is the primary position** throughout. AFTER is reported as a robustness check.

### 2.6 Directions and the estimator

- A gender direction is a difference of class means or an L2 logistic probe, fitted after residualising nuisance variables (cell dummies, Zipf, token count, concreteness): X ↦ (I − Z Z⁺) X, so only within-cell variation is used. ==I'm going to need more detail on the math here (residualising nuisance variables? brief note on how we determined what nuisance variables were? "cell dummies" not clear. "difference of class means or an L2 logistic probe" is just me needing to understand the math more)==
- **Split-half (cross-fitted) estimator.** Plug-in squared lengths and cosines are biased upwards (E‖v̂‖² = ‖v‖² + tr Σ (1/n_g + 1/n_ref)), worst for the smaller class. We split the nouns into halves within strata and use E[v̂_A · v̂_B] = ‖v‖², averaged over 200 splits and computed exactly through the Gram matrix. Inference uses a **within-cell label-shuffle null** and a bootstrap in which copies of a resampled item stay in one half. ==Also here (the formula, shuffle null, Gram matrix, "plug-in squared lengths", bootstrap, lost track of how they're stratified due to density==
- This matters: the plug-in estimator supported "masculine is the shorter, default vector" in 25/27 and 31/35 layers; the split-half estimator supported it in 0.

### 2.7 Behavioural gates

Phase 0 "known" checks (balanced article-choice rate; cross-size comparisons only on nouns every model knows), frame checks on two families, and in Phase 5 an agreement-flip gate (a dose α* must flip article agreement beyond a random-direction limit) plus readout frame checks (C1: the
probe must read the intended referent type with a 1-nat margin). ==Include the actual prompts used/more explanation of what the action being taken here is. "Frame check" sounds like something the reader is expected to know by reading our mind, among other things. "the probe must read the intended referent type": how, when do we know it's doing so, why can people trust us when we say we know==

### 2.8 Steering and readouts (Phase 5, P29–P31)

- **Steering:** push the noun's state along the grammatical gender direction (dose in units of the state norm) and measure the shift in English (R1) or Spanish (R1-ES/R2/R3) adjective log-probabilities, regressed on the adjectives' gender ratings with the covariates above. ==Units of the state norm?==
- **Controls:** 100 random directions (rank test), a number direction, a social-gender direction, damage (KL) matching.
- **Unsteered article tests (P30 Spanish, P31 German):** nonce nouns given a feminine vs masculine article in a sentence that makes them a *thing* (object reading) vs neutral vs a *person*; S = slope of the die−der (la−el) shift on gender rating. ==Add detail here because it's neat!==
- **Verdict rules:** a majority of the pre-registered layers; Holm across co-primary tests. ==more context on Holm==

### 2.9 Controls added late

- **Confidence control** (prereg `confidence-ctl-prereg.md`): (a) midpoint log-prob covariate, (b) Spearman rank check, (c) cross-fitted cosine between the f−m coefficient and the Zipf coefficient with a within-cell × gender permutation null; "overlap" = cross term beyond the null's 95th percentile **and** |cos| ≥ 0.3. ==What are the covariate and rank check actually measuring here? Not clear at a glance as written==
- **Frequency axis** (prereg `frequency-axis-prereg.md`): A flexible frequency control, B frequency axis projected out and Phase 4 rerun, C frequency-matched pairs, D complex-noun drop/covariate. ==This is too dense.==
- **Register / age of acquisition** (prereg `register-axis-prereg.md`).
- **Stimulus flaw found late:** 12/64 masculine German training nouns are complex (numerals such as *Vierer*, compounds such as *Feuerlöscher*), 0/27 feminine; the simplex filter only checked whole-word *-er* agent nouns. Lists frozen; tested by frequency-axis D.

### 2.10 Table of experiments

==Some of my "...this means nothing out of context" notes presumably remedied by building this, but I feel like the structure should also be meaningfully introduced earlier.==

*(To build from `docs/glossary.md`: phase/P-number, question, prereg status, verdict.)*

---

## 3. Results

### 3.1 Phase 0–1: models know the genders; spelling can't be erased

- Known rates (balanced, 257 nouns per language; `results/phase0/comparison.csv`) are high and rise with Qwen3 size: German 0.89 / 0.90 / 0.94 and Spanish 0.89 / 0.92 / 0.94 at 0.6B / 1.7B / 4B; EuroLLM-1.7B 0.97 / 0.98. (Phase 0 was not run on 8B/14B.) The article quiz is diagnostic only: reordering the options changed 20% (DE) and 45% (ES) of 0.6B answers
  (`meta_order_agree` 0.80 / 0.55). ==Describe "the article quiz" somewhere in here as opposed to assuming the reader knows the stimulus. "balanced" isn't clear on what's being balanced==
- A verb-trained LEACE eraser leaves nonce endings readable at all sizes (≈ 0.99–1.00); unseen verbs AUC 0.68–0.81. Hence spelling control by design.

### 3.2 Phases 2–3: a gender direction exists, at the noun's own token

| Model | Spanish LAST | Spanish AFTER |
|---|---|---|
| 0.6B | 24/27 | 0/27 |
| 1.7B | 22/27 | 2/27 |
| 4B | 26/35 | 0/35 |
| 8B | 28/35 | 3/35 |
| 14B | 26/39 | 0/39 |
| 30B-A3B | 45/47 | 38/47 |

*Layers where the stratified direction reads gender (confirmatory, bf16).*

- German compounds follow their **head's** gender at LAST in every layer at every size (AUC 0.76–0.87); the first part's gender: 0 layers. Markedness: 0 layers once the estimator bias is removed. ==For my sake as the non-German speaker, note what the alternative would be/what it means? I don't actually know if German compounds *grammatically* follow their head anyway, or if the head is *overriding* something.==
- Both effects are **position-dependent**: at AFTER they largely vanish (except 30B-A3B).

### 3.3 Phase 4: partly shared across languages (8B, 14B)

- T1 (transfer in both directions): fails. T3 flipped pairs: passes, AUC 0.83–0.93. T4 Spanish → Russian soft-sign nouns (spelling-controlled): AUC 0.82 / 0.85. T2 overlap cosine 0.43 / 0.35, above the shuffle null. At AFTER, everything fails. **Reading: partly shared.** ==Explain flipped pairs as opposed to assuming the reader knows; in general T*x*, P*x*, R*x*, C*x* notation needs to be explained==
- Robust to removing the German frequency axis (§3.7, test B). A DE→ES transfer gain seen after removal was a sample-composition artefact and is retracted. 

### 3.4 Phase 5: steering grammar into social readouts

| | 8B | 14B |
|---|---|---|
| R1 grammatical push | +0.020 (3/4 layers) | 0.000 (0/3) |
| R1 random 95th pct | 0.012 | 0.008 |
| R1 social push (control) | 0.064 | 0.047 |

- German: **no valid test** (8B frame 62% < 70%; 14B gate failed); verdicts are Spanish only. ==Parenthetical reads like technobabble==
- 8B: small, language-specific bleed; 14B: none, with a sensitive readout (the social control moves). =="the social control moves"? Not parsing this==
- R2 moves both readouts but the noun is visible, so it can't separate routes. ==What==
- R3-POOL / P20-POOL looked positive, but **P29** shows an article-only flip moves the pooled readout by +0.096 on 8B, at least as much as steering: pooling is not agreement-neutral, so these readouts are not evidence. Reported in full as a cautionary result. ==This doesn't make sense out of context==
- Confidence control (a)–(c): every P29–P31 and Phase 5 verdict is unchanged.

### 3.5 English social gender (P22b, P23, P24, P27)

- Glasgow gender ratings predict model behaviour (EN ρ 0.51–0.59, DE 0.28–0.43), so the readout is meaningful. ==Note how we established this==
- The Spanish grammatical direction separates English person words by sex (0.87 / 0.85; German direction weaker, 0.67 / 0.72). Its overlap with a social direction falls with size (ES 0.44 → 0.22; DE 0.18 → 0.08). =="German direction weaker" needs its own sentence or bullet point. "Its overlap with a social direction" unclear whether Spanish direction or both.==
- German neuter diminutives show no sex signal: the person effect tracks persons, not sex. ==Articulate how we know this.==

### 3.6 Unsteered article pull (P30 Spanish, P31 German)

- **P30:** a feminine article on a nonce noun read as a *thing* pulls toward feminine-rated adjectives in every Qwen3 size (1.7B +0.038, 4B +0.026, 8B +0.080, 14B +0.124; verified against `results/p30/*/analysis.json`); EuroLLM's object pull is small (+0.008) and it shows no neutral pull (−0.000). Neutral and person wordings pull too, so a social reading of the object effect is not established. ==Needs more context; "object pull" and "neutral pull" aren't easily derived from reading==
- **P31 (German, *die* vs *der*, *das* reported alongside):**

  | Model | Verdict | S_object | object / neutral |
  |---|---|---|---|
  | Qwen3-1.7B | pull survives object reading | 0.035 | 1.95 |
  | Qwen3-4B | pull survives object reading | 0.043 | 2.14 |
  | Qwen3-8B | no object pull | 0.002 | (neutral 0.002; person 0.081) |
  | Qwen3-14B | no valid test (C1 margin) | 0.025 *(exploratory)* | 0.73 [0.63, 0.85] |
  | EuroLLM-1.7B | pull survives object reading | 0.081 | 2.32 |

  Verified against `results/p31/*/analysis.json`. The *das* object effect is ≈ 0 on Qwen3 (≤ 0.001) but not on EuroLLM (+0.045).

### 3.7 The German gender direction is aligned with word frequency

- Control (c): in German the f−m direction points toward *rarer* words (8B cos −0.34, −0.56, −0.62, −0.69 at the four Phase 5 layers; 14B −0.19 to −0.52); Spanish |cos| ≤ 0.14.
- This is not a list-frequency imbalance. In both training lists the feminine nouns are the *more* frequent (wordfreq Zipf, within spelling cells: Spanish d = +0.52, German d = +0.45; raw d ≈ +0.7), and frequency is a covariate in the fit; yet only German shows the alignment. Across the full Wiktionary lexicons the genders barely differ in frequency (Spanish f − m +0.17 Zipf; German −0.05), so the imbalance is a property of our samples, not of the
  languages.
- It survives a flexible frequency control (A), frequency-matched pairs (C, 14 German pairs) and dropping or covarying the complex nouns (D): the overlap disappears at only 0–1 of 3–4 layers, never the pre-registered majority.
- Projecting the frequency axis out (B) leaves the Phase 4 verdicts unchanged.
- Register / age of acquisition does not explain it (8B overlap −0.62 → −0.61).
- SAE-feature test (G2, Qwen3-8B, pre-registered in `gemma-slim-prereg.md`): **no valid test.** The held-out compound nouns (632) do not show the overlap (cos +0.11 to +0.26 at the four layers, opposite sign and under the 0.3 bar), so there was nothing to remove. Descriptively,
  this means the alignment does not carry over to a different German noun sample, which fits it being partly sample-specific. 
- **Open:** what the axis encodes, and why it appears in the training sample but not the held-out compounds. Gemma 3 is the cross-family check.

### 3.8 Claims, evidence and status

| Claim | Status |
|---|---|
| Directions read grammatical gender, not spelling | strong |
| Gender is read at the noun's own token | strong; must be stated |
| Partly shared across languages | strong-ish ("partly") |
| Steering bleeds grammar into social readouts at 8B | small; survives confidence control |
| No bleed at 14B | good null |
| Article pull on thing-nouns (P30/P31) | descriptive; not shown to be social |
| R3-POOL / P20-POOL | **not evidence** (P29) |
| German direction aligned with frequency | robust in training sample; absent in held-out compounds; mechanism open |

---

## 5. Limitations (to expand from the explainers' "Limits worth stating")

- One model family for the mechanistic results so far (Gemma 3 slim replication planned).
- Base models only.
- German Phase 5 had no valid test; Phase 5 verdicts are Spanish-only.
- Position dependence: the effects are read at the noun's own token.
- The German training list's complex-noun imbalance (tested, not fixed).
- Concreteness norms for German/Russian are English-gloss approximations.
