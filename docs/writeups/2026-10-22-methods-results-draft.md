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

### 2.6 Finding a gender direction, and measuring it fairly

#### 2.6.1 What a "direction" is

For each noun and each layer, the model gives us a list of a few thousand numbers (its *hidden state*, §2.5). Think of it as a point in a space with a few thousand axes. A **gender direction** is an arrow in that space that points from where masculine nouns tend to sit toward where feminine nouns tend to sit. If such an arrow exists, you can take any noun's point, measure how far along the arrow it lies, and guess its gender from that one number. We found arrows in two ways:

- **Difference of means.** Average the points of all feminine training nouns, average the points of all masculine ones, and draw the arrow from the second average to the first. This is the simplest possible method and makes no choices of its own.
- **Probe.** A *logistic regression* is a standard statistical model that learns a weighted sum of the inputs and turns it into a probability ("feminine with probability 0.8"). Its weights are an arrow too: the direction it found most useful for telling the genders apart. "L2" means a penalty that keeps the weights small, which stops the probe from latching onto a handful of the thousands of axes by chance when there are only about a hundred training nouns. The probe can ignore axes that vary a lot for reasons unrelated to gender, which a difference of means cannot, so it usually separates the genders better. Both were used. Where they agree, the result does not depend on the method.

> **Formula box (skippable).** Write x for a noun's hidden state and μ_f, μ_m for the feminine and masculine averages. The difference-of-means direction is v = μ_f − μ_m. A noun's score along it is the dot product v · x. The probe chooses w to minimise Σ_i log(1 + exp(−y_i w · x_i)) + λ‖w‖², where y_i is +1 for feminine and −1 for masculine; λ sets the size of the L2 penalty. With about 100 nouns and about 5,000 axes, there are far more axes than nouns. The fit is therefore done in the much smaller space spanned by the training nouns themselves, which gives exactly the same answer far faster.

#### 2.6.2 Removing nuisance variables ("residualising")

Even with the word lists of §2.4, feminine and masculine nouns may still differ on average in other ways: in frequency, in length, in spelling cell. An arrow could pick up those differences instead of gender. So before finding any direction, the part of each hidden state that is predictable from those other properties is removed.

The idea is the same as "controlling for" a variable in an ordinary regression. Suppose frequency affects the hidden states. Fit, across all the training nouns, the best straight-line prediction of the hidden state from frequency. Subtract that prediction from every noun. What is left (the **residual**) is the part of each hidden state that frequency cannot explain. A gender arrow found in the residuals cannot be using frequency, because there is no frequency information left in them. (In statistics this is the Frisch–Waugh–Lovell result: controlling for a variable in a regression is the same as removing it first and then fitting.)

The properties removed were:
- **Spelling cell.** This is done with **dummy variables**: one yes/no column per cell ("does this noun end in *-te*?", "does it end in *-al*?", …). Removing them subtracts each cell's own average, so feminine and masculine nouns are compared only against other nouns in the same cell.
- **Frequency** (Zipf), **token count**, and **concreteness**.

These were chosen *before* the gender results, for one reason each: they are the properties already known to differ between the genders in these lists (the counts are in `docs/screening/`), and each one could plausibly change a hidden state. A few were added later by the same rule when a check found a new imbalance, for example a word-initial *a-* in Spanish (§2.4.2). Each such addition is logged in `docs/decisions.md`.

> **Formula box.** Put the nuisance columns side by side in a matrix Z (one row per noun). Put the hidden states in a matrix X. The residuals are X − Z Z⁺ X, where Z⁺ is the pseudo-inverse of Z, so Z Z⁺ X is the least-squares prediction of X from Z. Written compactly: X ↦ (I − Z Z⁺) X.

#### 2.6.3 Why we measure lengths and angles with two halves (the estimator problem)

Several of our questions are about the *size* of an arrow, or the *angle* between two arrows:
- **Markedness:** is the masculine noun average closer to the overall average, as if masculine were the "default"?
- **Sharing:** does the Spanish gender arrow point the same way as the German one?

There is a trap here that we fell into at first. An arrow estimated from data contains noise, and noise always adds length: a noisy arrow is on average *longer* than the true one. The noise is bigger when fewer nouns go into an average. Here the feminine lists are the smaller ones, so the feminine arrow came out artificially long. That looked exactly like the "masculine is the default" result. This simple way of measuring (computing the length straight from the estimated arrow) is called the **plug-in** estimate. It supported markedness in 25 of 27 layers in one model and 31 of 35 in the other.

The fix is to split the nouns at random into two halves, A and B, keeping each spelling cell evenly divided between them. Estimate the arrow separately from each half, then multiply the two together (the dot product) instead of multiplying one arrow by itself. The true arrow is the same in both halves, but the noise in A and the noise in B are independent, so they do not reinforce each other: on average the noise cancels instead of adding. Different random splits give slightly different answers, so we average over 200 splits. With this **split-half** estimate, markedness was supported in **0** layers. The earlier effect was entirely noise.

> **Formula box.** If v̂ is the estimated arrow and v the true one, the plug-in length satisfies E‖v̂‖² = ‖v‖² + tr Σ (1/n_g + 1/n_ref). Here Σ is the noise in the hidden states and n_g, n_ref are the numbers of nouns in the two averages; the second term is pure bias, and it grows as n shrinks. With independent halves, E[v̂_A · v̂_B] = ‖v‖² exactly, with no bias term. Cosines (angles) are built from the same pieces. The 200 splits are computed exactly and cheaply through the **Gram matrix**: the table of dot products between every pair of nouns (about 120 × 120, instead of working with 5,000-number arrows).

#### 2.6.4 How sure we are: the null and the bootstrap

Two further tools tell us whether a number is bigger than chance and how precise it is:

- **Shuffle null** (what the number looks like with no real effect). Shuffle the gender labels at random, *within each spelling cell*, so each cell keeps its counts of feminine and masculine. Then compute the same number again, and repeat many times. This gives the range the number takes when gender means nothing. A real result has to beat about 95% of the shuffled values.
- **Bootstrap** (how much the number would change with a different sample of nouns). Draw a new sample of nouns of the same size *with replacement* from the original list, again within each cell, so that some nouns appear twice and some not at all. Redo the whole analysis, including refitting the arrow, and repeat about 1,000 times. The middle 95% of the results is the reported interval. One subtlety we found the hard way: when a noun is drawn twice, both copies must go into the *same* half in §2.6.3. Otherwise the two halves share that noun's noise, and the bias creeps back in. (It showed up as an interval that did not contain its own estimate.)

When several tests answer the same question, the significance threshold is tightened (the **Holm** correction), so that running more tests does not make a false positive more likely.

### 2.7 Checks that the model can do the task before we test it

Every experiment begins with checks run on the unmodified model, before any result of interest is looked at. If a check fails, that test is reported as "no valid test" rather than run on a model that cannot do the task. These are called **gates**. When a gate fails, we do not swap in a different sentence frame until one passes, because that would amount to choosing the test after seeing the data.

#### 2.7.1 Does the model know each noun's gender? (Phase 0)

The model reads a sentence once with the masculine article and once with the feminine one, and we compare how probable it finds the rest of the sentence (the noun and what follows) after each. The probability of the article itself is excluded, because some articles, such as German *ein*, also serve other genders, which would tilt the comparison. The frames are:

| language | frame 1 | frame 2 |
|---|---|---|
| German | *Das hat etwas mit **dem/der** X zu tun.* "That has something to do with the X." | *Hier ist **ein/eine** X.* "Here is a(n) X." |
| Spanish | *Esto tiene que ver con **el/la** X.* "This has to do with the X." | *Aquí hay **un/una** X.* "Here there is a(n) X." |

The frames were chosen so that no other reading is possible. A plural reading, for instance, would need a different article or noun form. Two early German frames were dropped for exactly this reason (*Ich weiß, dass der/die X hier ist*, where *die X* can be plural).

The difference in log-probability between the two versions is the **margin**, measured in *nats*. One nat means one version is about 2.7 times as probable as the other; 2.3 nats is ten times. A noun counts as **known** if the correct gender wins by at least 1 nat in *both* frames. If the model gets it confidently wrong, the noun is "wrong". If it is right in one frame and confidently wrong in the other, it is "conflict". Near-ties are "unsure". Only known nouns are used later. (Phase 3 used the same idea with three articles, *Hier ist der/die/das X* and *Dort gibt es einen/eine/ein X*.)

We also tried a quiz format, *Bestimmter Artikel:* "definite article:" followed by four worked examples (*Teppich: der*, *Tasche: die*, …) and then *X:*. The answers changed with the order of the examples, so the quiz was used only as a diagnostic and gates nothing (§3.1).

#### 2.7.2 Does a sentence frame work before we steer in it? (Phase 5 frame checks)

In Phase 5 we push nouns along the gender arrow (§2.8) and watch whether article agreement changes. That only makes sense in a frame where the unmodified model already gets the gender right. A **frame check** tests exactly that. It uses the model's known, common nouns (Zipf ≥ 4), and the frame passes only if at least 70% of nouns *of each gender* come out right. The German candidates, fixed in advance, were:

- C1 *Wörterbuch:\n X,* "Dictionary: X," followed by *der* or *die*;
- C2 *Thema: X. Was macht man mit* "Topic: X. What does one do with" followed by *ihm* "it (m)" or *ihr* "it (f)";
- C3 *Wörterbuch:\n X, mit* "Dictionary: X, with" followed by *dem* or *der*.

For Spanish, the frame is *Mi X es muy* "My X is very". The check reads whether the masculine or feminine form of the following adjectives is preferred (*rojo/roja* "red").

#### 2.7.3 Is the push strong enough to matter, but no stronger? (the agreement-flip dose)

Before reading social adjectives, each layer has to pass a dose check. The push is made larger in steps (0.05 to 3 times the typical length of a hidden state). Feminine nouns are pushed toward masculine and masculine nouns toward feminine. The **dose** α\* is the smallest step at which agreement flips for the *median* noun of both genders (a feminine noun now takes the masculine form, and vice versa). There is one condition: pushes of the same size along 20 *random* directions must flip, on average, fewer than 10% of nouns. A layer where this never happens drops out of the test. This makes sure that a later "no effect on social adjectives" cannot just mean the push was too weak to do anything, and that a positive effect is not something any large push would cause.

#### 2.7.4 Does the sentence make the model think of the thing we intend? (the C1 check in P30/P31)

P30 and P31 ask whether an article biases adjectives *even when the noun is clearly an object*. That claim needs evidence that the model actually reads the noun as an object in our "object" sentences, and as a person in our "person" sentences. So, in the same sentence with the adjective slot removed, we compare how probable the model finds a person word against a thing word coming next: Spanish *una persona* "a person" versus *una cosa* "a thing"; German *Mensch/Person/Mann/Frau* versus *Ding/Gegenstand/Objekt/Sache* (the German set was broadened by the PI before the run).

- An **object wording** passes only if the thing word wins on the median noun, *and* wins by at least 1 nat more than in the neutral wording.
- A **person wording** must favour the person word by at least 1 nat more than neutral.
- Real person nouns (*la doctora* "the (female) doctor") are scored the same way, as a reference for what a genuine person reading looks like.

Why trust it: the check reads the model's own expectations about what kind of thing comes next. It was fixed before any adjective result was computed, and it fails openly. Qwen3-14B's German object sentences failed it, and that test is reported as "no valid test" (§3.6) rather than rescued. It is still an indirect measure of what the model "thinks", and §5 lists it as a limitation.

### 2.8 Steering and readouts (Phase 5, P29–P31)

This is the stage that asks the actual question. If grammatical gender and social gender share machinery inside the model, then pushing a noun's internal state toward "grammatically feminine" should also make the model treat it a little more like something *socially* feminine.

#### 2.8.1 Steering: the push

1. Run the model on a prompt containing a noun, for example *puente* "bridge" (masculine in Spanish).
2. At one layer, at the noun's own token(s), add a multiple of the grammatical gender direction from §2.6 to the hidden state: *state + α × direction*. A positive α pushes toward feminine, a negative one toward masculine. This is called **steering**.
3. **How big a push.** Hidden states at different layers have very different typical sizes (their *norm*, i.e. length). A fixed push would be huge at one layer and invisible at another. So α is measured in multiples of the layer's typical state length: the median length of the hidden state at noun positions. α = 0.1 means "a push one tenth as long as a typical state at this layer".
4. **Which α.** For each layer we use α\*, the smallest push at which the model, asked for the noun's article, switches it for the median noun (*el puente* → *la puente*). So α\* is "just enough to change the grammar". As a check that this isn't simply damage, pushes of the same size along random directions must switch the article for fewer than 10% of nouns. A layer where the gender push never switches the article drops out.
5. Then let the model run on and read what it predicts next (the readouts below), with and without the push.

#### 2.8.2 Readouts: what we measure after the push

Each readout is a sentence that ends just where the model must pick a word, and we compare probabilities of human-rated words in that slot.

| | prompt (the push is at the noun) | what is compared |
|---|---|---|
| **R1** | *The Spanish word "puente" means bridge. Described in one word, it is very ___* | the ~600 English adjectives of §2.4.6 |
| **R1-EN** | *The bridge is very ___* (the push is applied to the English word, using the Spanish or German direction) | the same adjectives |
| **R2** | *If the bridge (puente) were a person, would it be a man or a woman? It would be a ___* | *man* versus *woman*; also *male/female* and *boy/girl*, each in both orders, averaged over the six |
| **R3** | *Mi puente es muy ___* "My bridge is very ___" | Spanish adjectives that look the same in both genders (*fuerte* "strong", *elegante* "elegant", *frágil* "fragile") |

The word *very* forces the next word to be an adjective. R3 uses only adjectives with one form for both genders, so the sentence's grammar cannot prefer one; the only thing that could favour *elegante* over *fuerte* is meaning.

**From readout to a single number.** For R1, the push changes the log-probability of every adjective a little. We then ask whether the change *lines up with the adjectives' gender ratings*: does pushing toward feminine raise feminine-rated words (*gentle*) and lower masculine-rated ones (*rugged*)? Concretely, the change for each adjective is regressed on its Glasgow gender rating, with valence, arousal, size and frequency as additional predictors. The coefficient on the gender rating is the **bleed score**.

> *Formula box (skippable).* For adjective *j*, let Δ*_j* = log P(*j* | pushed) − log P(*j* | not pushed). Fit Δ*_j* = *b*₀ + *b*_g · gender*_j* + *b*_v · valence*_j* + *b*_a · arousal*_j* + *b*_s · size*_j* + *b*_f · Zipf*_j* + error. The bleed score is *b*_g. The other predictors are there so that *b*_g can only reflect gender, not "pleasantness" (which correlates with gender at −0.47).

Each bleed score is compared with the same score from three other pushes:
- **Random directions** (20 or 100, same length). The gender push must beat the 95th percentile of the random pushes; otherwise any push "does" this.
- **A number direction** (singular versus plural, same length), as a second grammar-but-not-gender comparison.
- **A social-gender direction** (pushing *bridge* toward *woman*). This is a **positive control**: it must move the readouts, or the readouts can't detect gender at all and a null means nothing.
- For R1-EN, the push's **damage** to the model's predictions on ordinary English text (measured as KL divergence) is also capped, so that an effect cannot come from a broken model.

#### 2.8.3 No push at all: does the article itself carry social gender? (P29–P31)

The steering tests found little (§3.4), but one unsteered observation stood out (P29): in *La sianta es muy ___*, with an invented noun, the feminine article *la* (versus *el*) made feminine-rated adjectives a little more likely, even among adjectives that look identical in both genders. Two explanations are possible:
- the article's grammatical gender itself carries some social gender ("bleed"); or
- the model guesses that an unknown *la*-noun may be **a woman**: *la sianta* could be a female person, like *la presidenta*. That would be ordinary meaning, not bleed.

P30 (Spanish) and P31 (German) separate these by placing the same invented noun in sentences that make it clearly a **thing**, clearly a **person**, or neither:

| context | Spanish (P30) | German (P31) | gloss |
|---|---|---|---|
| object | *La sianta que compré ayer es muy ___* | *Die Flitt, die ich gestern gekauft habe, ist sehr ___* | "The N I bought yesterday is very ___" |
| object | *La sianta que guardé en el cajón es muy ___* | *…, die ich in die Schublade gelegt habe, …* | "…that I put away in the drawer…" |
| neutral | *La sianta es muy ___* | *Die Flitt ist sehr ___* | "The N is very ___" |
| person | *La sianta con quien hablé ayer es muy ___* | *…, mit der ich gestern gesprochen habe, …* | "…that I talked with yesterday…" (Spanish *quien* is used only for people) |
| person | *La sianta que conocí ayer es muy ___* | *…, die ich gestern kennengelernt habe, …* | "…that I met yesterday…" |

Every sentence is run with both articles (German also with neuter *das*), over 600 (Spanish) or 300 (German) invented nouns. The score **S** for each context is the bleed score of §2.8.2 applied to the *la* − *el* (*die* − *der*) difference, using gender-invariant adjectives in Spanish and, in German, predicate adjectives (which never change form there).

The logic: if the pull is the "a woman" guess, it should vanish when the noun is clearly a thing. If it survives in the object sentences, it is not about people.

Two checks must pass first, or there is no valid test:
- **C1, do the sentences really change the reading?** In the same sentence ending in *es …* "is …", we compare the model's probability of *una persona* "a person" versus *una cosa* "a thing" (German: *Mensch/Person/Mann/Frau* versus *Ding/Gegenstand/Objekt/Sache*). Object sentences must lean "thing" by at least 1 nat (a factor of e ≈ 2.7 in odds) more than the neutral sentence. Person sentences must lean "person" by at least 1 nat more. A wording that fails is dropped; if every object wording fails, there is no valid test.
- **C2, are the measured words actually likely there?** The adjectives read must together get at least 5% of the probability in that slot. Otherwise we would be measuring words the model wasn't going to say.
- Real person nouns that take either article (*el/la presidente*) are run too, as a yardstick: here the article really does mark a man or a woman.

Verdict (Spanish; German alike): **"the pull survives an object reading"** if S(object)'s 95% interval is above 0; **"mostly the 'a woman' guess"** if its upper end is below half of S(neutral).

#### 2.8.4 How verdicts are reached

- **Across layers.** Every test is run at the pre-registered layers (four per model in Phase 5). A result counts only if it holds at **a majority** of them. One striking layer is not enough.
- **Two main tests at once (Holm correction).** Where two tests are both "primary" (Phase 4: T1 and T3; Phase 5: R1 and R1-EN), testing both at the usual 95% level doubles the chance that one passes by luck. The Holm procedure fixes this cheaply: the stronger result must pass a stricter bar (a 97.5% interval excluding zero); only if it does is the second judged at the ordinary 95%. That keeps the overall chance of a false "yes" at 5%.
- **Verdict labels for Phase 5:** **bleed** (Spanish and German pushes both move social readouts the same way and beat random), **language-specific** (only one does), **no effect** (no more than random, while the social-gender push does move the readouts, so the test could have seen it).

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
