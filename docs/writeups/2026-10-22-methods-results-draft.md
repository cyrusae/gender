# Methods and results: first draft (2026-10-22)

*Status: draft for the PI. Framing, introduction and discussion wait on the literature review. Assembled from `docs/methods-summary.md`, `docs/explainers/00–07`, the preregistrations in `docs/design/` and `docs/decisions.md`. Numbers marked **[check]** were carried over from summaries and must be re-read from `results/` before anything leaves the repo.*

*Rewritten 2026-10-22/23 for a reader new to the project, following the PI's comments: what was done, in order, with the real prompts and word lists. Formula boxes can be skipped.*

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

  We kept words that are mainly adjectives (at least half their WordNet senses), common (Zipf ≥ 3), and a single token for the model, so each one is a single next-word prediction. That leaves 656 adjectives for Qwen3 (721 for Gemma 3).
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

These were added after the main results, in response to specific worries. Each has its own pre-registration in `docs/design/`.

#### 2.9.1 Is it really confidence, not gender? (`confidence-ctl-prereg.md`)

**The worry.** A push of any kind might just make the model less sure of itself, spreading probability from words it was going to say toward words it wasn't (like raising the "temperature"). Call this **flattening**. If feminine-rated adjectives happen to be the *less likely* ones in our prompts, flattening alone would raise them and look like a feminine shift. Gender would play no role.

Three checks, each aimed at one way this could happen:
- **(a) Control for how likely each word already was.** We add one more predictor to the bleed regression of §2.8.2: each adjective's log-probability, averaged over the two conditions being compared (the *midpoint*). If the "gender" effect were flattening, it would be absorbed by this predictor and the gender coefficient would shrink toward zero. (The midpoint is used rather than the "before" value, because regressing a change on its own starting point produces a spurious negative slope through regression to the mean.)
- **(b) Use ranks instead of values.** We repeat the test with a rank correlation (Spearman) between the shift and the gender rating. This asks only whether feminine-rated words tend to move up more, not by how much. A few extreme words can't drive it.
- **(c) Is the gender direction itself a frequency direction?** In the noun representations, we fit one direction for feminine versus masculine and one for frequency (the Zipf score), on separate halves of the nouns, and take the cosine between them (+1 = same direction, 0 = unrelated, −1 = opposite). The two directions are said to **overlap** if the cosine is at least 0.3 in size *and* is larger than 95% of cosines obtained after shuffling gender labels within spelling cells. (A cosine of 0.3 means about 9% of one direction's variation is shared with the other.)

#### 2.9.2 Following up the German frequency overlap (`frequency-axis-prereg.md`)

Check (c) found an overlap in German but not in Spanish (§3.7): the German feminine-minus-masculine direction points toward *rarer* words. Four follow-ups asked where this comes from:

| | test | the question it answers | what would count as "explained" |
|---|---|---|---|
| **A** | redo (c) with a flexible frequency control (curved, and allowing for token count) | Was the straight-line frequency control too crude? | the overlap disappears at most layers |
| **B** | remove the frequency direction from the hidden states, then rerun the cross-language tests of Phase 4 | Do any earlier German results depend on this overlap? | (here we hope for *no change*: same verdicts, cosine moves < 0.05) |
| **C** | compare only feminine/masculine pairs of nearly equal frequency (Zipf within 0.3), no regression at all | Is the overlap an artefact of the regression? | the overlap disappears |
| **D** | drop the 12 complex masculine nouns (below), or add a "complex" predictor | Is it word structure, not frequency? | the overlap disappears at most layers |

Two later tests went further:
- **Register / age of acquisition** (`register-axis-prereg.md`). Perhaps the "frequency" direction really encodes how technical or late-learned a word is. Controlled with age-of-acquisition norms.
- **Sparse-autoencoder check, G2** (`gemma-slim-prereg.md`). Sparse autoencoders break a hidden state into many interpretable "features". We asked whether removing the few features most tied to frequency removes the overlap, on held-out German compounds.

#### 2.9.3 A flaw in the German training list, found late

While running D, we found that 12 of the 64 masculine German training nouns are not single roots after all:
- number-words: *Vierer* "a four", *Fünfer*, *Sechser*, *Zehner*;
- compounds with a tool-noun head: *Wasserkocher* "kettle", *Feuerlöscher* "fire extinguisher", *Plattenspieler* "record player", *Zahnstocher* "toothpick", and four more.

None of the 27 feminine nouns are. The compound filter (§2.4.3) only checked whether the *whole* word was an *-er* agent noun made from a verb. It missed compounds whose *head* is one, and it missed numerals. Because results had already been reported on this list, it was not edited. Its effect was tested directly by D, and the miss is logged in `docs/decisions.md`.

### 2.10 Table of experiments

This is the index to everything above. §2.1 gives the overall plan; this table lists each test by its label. The labels are only numbering: **T** = a Phase 4 cross-language test, **R** = a Phase 5 readout (what we read after steering), **P** = a follow-up project, numbered in the order it was proposed, **C** = a check that a prompt means what we intend, **G** = the Gemma 3 replication. "Confirmatory" means the test and its pass rule were written down and approved before the data existed and it counts toward the conclusions. "Exploratory" means it was also written down first, but it is reported descriptively and never used to rescue a failed confirmatory test.

| Label | Phase | Question in plain words | Status | Outcome | Results |
|---|---|---|---|---|---|
| Phase 0 | 0 | Does the model know each noun's gender (article quiz)? | gate | yes, 0.89–0.98 of nouns at every size | §3.1 |
| Phase 1 | 1 | Can spelling information be erased from the model's states? | exploratory | no; so spelling is controlled by design | §3.1 |
| Phase 2 | 2 | Is there a Spanish gender direction that is not just spelling? | confirmatory | yes, at the noun's own token | §3.2 |
| Phase 3 | 3 | Do German compounds follow their head's gender? Is masculine the "default"? | confirmatory | head: yes; default: no | §3.2 |
| T1 | 4 | Does a Spanish direction read German gender, and vice versa? | confirmatory | fails | §3.3 |
| T2 | 4 | Do the Spanish and German directions point the same way? | confirmatory | partly (cos 0.43 / 0.35) | §3.3 |
| T3 | 4 | On "bridge" pairs, does each language's direction follow its own grammar? | confirmatory | passes | §3.3 |
| T4 | 4 | Does the Spanish direction read Russian soft-sign nouns? | confirmatory | passes | §3.3 |
| R1, R1-EN | 5 | Does pushing a noun toward feminine make feminine-rated English adjectives likelier? | confirmatory (co-primary, Holm) | small at 8B, none at 14B | §3.4 |
| R2 | 5 | Same, read as a woman/man choice | secondary | moves, but can't separate routes | §3.4 |
| R3 | 5 | Same, with Spanish gender-invariant adjectives | secondary | no valid test on 8B: the adjectives get under 5% of the probability (C2-style check) | §3.4 |
| P19 | 5 | Do invented words get the gender their ending suggests? | exploratory | yes, every model | §2.4.6 |
| P20, P28 (R3-POOL, P20-POOL) | 5 | Read gendered Spanish adjectives with both forms pooled | exploratory | looked positive; ruled out by P29 | §3.4 |
| P22, P22b | 5 | Does the model's own woman/man axis match the human ratings? | exploratory | yes in English; weaker in German | §3.5 |
| P23, P24, P24b | 5 | Does the grammatical direction also sort English person words and names by sex? | exploratory | Spanish yes; German weaker; overlap shrinks with size | §3.5 |
| P25 | 5 | Unsteered: do "bridge" concepts lean by grammatical gender in English? | exploratory | no lean (intervals include 0, 8B and 14B) | – |
| P26 | 5 | Does steering with the *other* language's direction do the same? | exploratory | 14B: the German direction, pushed on Spanish nouns, moves R1 above random at 2 of 3 layers (a majority; exploratory, so outside the verdicts) | – |
| P27 | 5 | German neuter diminutives (*das Schwesterchen*): does sex or grammar drive the person effect? | exploratory | persons, not sex | §3.5 |
| P29 | 5 | Is pooling agreement-neutral? | confirmatory check | no, so R3-POOL/P20-POOL aren't evidence | §3.4 |
| P30 | 5 | Spanish: does *la* on a "thing" invented noun pull toward feminine adjectives? | pre-registered, descriptive | yes, all Qwen3 sizes; not shown to be social | §3.6 |
| P31 | 5 | Same in German, with *das* alongside | pre-registered, descriptive | mixed by size | §3.6 |
| confidence (a)–(c) | late | Is any steering result just the model being more or less confident? | pre-registered control | no; German direction aligns with frequency | §3.7 |
| frequency A–D | late | Does the German frequency alignment come from how we controlled frequency, or from the compounds? | pre-registered control | no; it survives all four | §3.7 |
| register (E1–E4) | late | Is it age of acquisition / register? | pre-registered control | no | §3.7 |
| G2 | late | Can SAE features carrying frequency be removed from the German direction? | pre-registered | no valid test | §3.7 |
| G0, G1 | replication | Gemma 3: technical gates; do the Phase 3–5 conclusions replicate? | pre-registered | running | – |
| rarity held-out | late | Does the rarity lean appear in new, simple German nouns? | pre-registered | not yet run | – |
| Russian | new language | Does an article-free language's direction line up with Spanish and German? | prepared | not yet run | – |

---

## 3. Results

### 3.1 Phase 0–1: the models know the genders, and spelling can't simply be erased

**Phase 0: do the models know which article each noun takes?** We gave each model the two-frame article test of §2.7.1 on 257 common nouns per language: German nouns with *der/die*, Spanish with *el/la*, half of each gender. A noun counted as known only if the right article won by at least 1 nat in both frames. The table gives the share known, averaged over the two genders (so a model that knew every masculine noun and no feminine one would score 0.5, not the share of masculine nouns in the list):

| model | German | Spanish |
|---|---|---|
| Qwen3-0.6B | 0.89 | 0.89 |
| Qwen3-1.7B | 0.90 | 0.92 |
| Qwen3-4B | 0.94 | 0.94 |
| Qwen3-8B | 0.97 | 0.94 |
| Qwen3-14B | 0.95 | 0.95 |
| Qwen3-30B-A3B | 0.98 | 0.97 |
| EuroLLM-1.7B | 0.97 | 0.98 |

*Source: `results/phase0/comparison.csv`. 0.6B–4B and EuroLLM were run on the Mac (fp16); 8B, 14B and 30B on the pod (bf16, `worktrees/gender-pod`). Where the small models were run both ways, the two differ by at most 0.01.*

- Confidently **wrong** answers were rare (at most 1.2% of nouns). Almost all the rest were near-ties ("unsure"), not errors. So when a model doesn't pass a noun, it is usually unsure rather than mistaken.
- Knowledge rises with Qwen3's size up to 8B and then levels off. EuroLLM, a small model built specifically for European languages, knows the most.
- The quiz format (§2.7.1: *definite article:* followed by worked examples) gave answers that depended on the order of the examples. Reordering them changed 20% of 0.6B's German answers and 45% of its Spanish ones. This is why the quiz gates nothing and only the sentence frames count.

**Phase 1: can spelling be erased?** The cleanest way to stop a gender direction from reading spelling would be to delete the spelling information from the hidden states first. We tried this with **LEACE**, a published method that makes the smallest change to the states that leaves a property undetectable by any straight-line (linear) classifier. The trick was to learn what "-o versus -a" looks like where it has nothing to do with gender: in Spanish verbs, where *hablo* "I speak" and *habla* "she speaks" differ only in the ending. If the eraser learned the ending *as a form*, it should also erase it from invented words like *flitro/flitra*.

- It did not. After erasing, a fresh classifier still told invented *-o* words from *-a* words almost perfectly, at every model size (accuracy 0.99–1.00; chance is 0.5). The eraser had removed the verb distinction (1st versus 3rd person), not the ending as such.
- Even on verbs it had not been fitted on, erasure was only partial: the ending was still readable (AUC 0.68–0.81, where 0.5 means erased).
- **Consequence for the design:** spelling could not be removed from the model, so it is controlled by how the nouns are chosen instead (§2.4). Every comparison is made between nouns that share an ending.

### 3.2 Phases 2–3: a gender direction exists, at the noun's own token

**How a result is scored.** Most results below are **AUCs** (area under the ROC curve). Take one noun from each of two groups at random and ask whether the direction scores the feminine one higher. The AUC is how often it does: 0.5 is a coin flip, 1.0 is perfect sorting. Every model has many layers (28 to 48), and each layer gets its own verdict. A layer counts as reading **gender** when its 95% interval (§2.6.4) clears 0.5 on the test that tells gender apart from spelling, and does *not* clear it on the test that would favour spelling. The tables report how many layers pass. "LAST" and "AFTER" are the two places the hidden state is read (§2.5): on the noun's own last token, or on the neutral token after it.

**Spanish (Phase 2).** The key test nouns are the exceptions of §2.4.2: masculine nouns ending in *-a* (*el día*, *el problema*). A direction that reads *gender* should put them with the masculine nouns. A direction that reads *spelling* should put them with the feminine *-a* nouns.

| Model | Spanish LAST | Spanish AFTER |
|---|---|---|
| 0.6B | 24/27 | 0/27 |
| 1.7B | 22/27 | 2/27 |
| 4B | 26/35 | 0/35 |
| 8B | 28/35 | 3/35 |
| 14B | 26/39 | 0/39 |
| 30B-A3B | 45/47 | 38/47 |

*Layers where the stratified direction reads gender, out of the model's inner layers (confirmatory runs, bf16 precision). 30B-A3B is a mixture-of-experts model, included once for comparison.*

- At LAST, every size reads gender rather than spelling in most layers.
- At AFTER, the signal is almost gone, except in the 30B model. Gender is carried on the noun's own token and is not passed on to the next, neutral token. This matters for Phase 5, where steering is applied at the noun (§2.8).

**German (Phase 3).** The test set is the compounds whose first part and head disagree in gender (§2.4.3), like *der Türrahmen* "door frame": feminine first part (*die Tür*), masculine head (*der Rahmen*). In German grammar the head always wins, so the right answer is never in doubt. The question is what the *model's direction* follows. A direction that tracks the grammatical gender of the whole word should follow the head. A direction that tracks something else should not: for instance the first part, the word's first tokens (models split long words into pieces, and the first piece is often the first part), or general word shape. The test therefore asks whether the direction learned on single-root nouns sorts these conflict compounds by their head's gender or by their first part's.

- **The head's gender** was read at LAST in every layer of every model size (AUC 0.76–0.87).
- **The first part's gender** was read in no layer.
- **Markedness** ("masculine is the default", §2.6.3) was supported in 0 layers once the estimator problem was fixed. The plug-in estimate had shown it in 25 of 27 and 31 of 35 layers, entirely through noise.
- As in Spanish, this is a LAST-position result; at AFTER it largely vanishes (except 30B).

### 3.3 Phase 4: partly shared across languages (8B, 14B)

Phase 4 asks whether Spanish and German gender sit in the *same* place in the model or in two separate places. If they share a direction, then pushing on Spanish grammar could plausibly move anything that direction touches, including social gender in English. That is why this phase comes before the steering tests. It was run on the two largest dense models, Qwen3-8B and 14B, with four pre-registered tests:

- **T1, transfer.** Find the direction on Spanish nouns and use it to sort *German* nouns by gender, and the reverse. Sorting is scored within the test language's spelling cells, so spelling can't help. "Transfers" means the interval clears 0.5 in a majority of layers. T1 needs both directions to transfer. Layer counts use a Holm correction (§2.8) because T1 and T3 are tested together.
- **T2, angle.** How closely do the two directions point the same way? Measured as a cosine (1 = identical, 0 = unrelated), using the two-halves method of §2.6.3, against the shuffle null of §2.6.4.
- **T3, flipped pairs.** This is the test closest to the original question. Each "bridge" pair (§2.4.4) is the same object with opposite genders in the two languages: *die Brücke* is feminine, *el puente* masculine. Use *one* direction to score *both* nouns of a pair. If it sorts the German noun by German gender and the Spanish noun by Spanish gender, then it is following grammar, which the two languages disagree on, and not meaning, which they share. (Using each language's own direction on its own noun would prove nothing: it would only repeat Phases 2–3.)
- **T4, Russian.** Use the Spanish direction to sort Russian soft-sign nouns (§2.4.5), a language and alphabet the direction never saw, with spelling matched by construction.

Results:

| test | 8B | 14B | verdict |
|---|---|---|---|
| T1 transfer, Spanish→German | 6 of 35 layers | 7 of 39 layers | does not transfer |
| T1 transfer, German→Spanish | 5 of 35 layers | 22 of 39 layers | transfers in 14B only |
| T2 angle | cos 0.43 | cos 0.35 | above the shuffle null: the directions overlap, but far from identical |
| T3 flipped pairs | AUC 0.93 (Spanish direction), 0.83 (German) | AUC 0.91, 0.89 | passes in every layer: one direction follows each language's own grammar |
| T4 Russian | AUC 0.82 | AUC 0.85 | the Spanish direction sorts Russian gender |

*AUCs and cosines are averages over layers; layer counts are Holm-corrected. Source: `results/phase4/<model>/summary.json` (session-2 worktree).*

- **T1 fails**, because transfer must work both ways and Spanish→German never does.
- **Reading: partly shared.** There is a common component strong enough to sort the flipped pairs by grammar and to carry over to Russian, but not strong enough to give full transfer between Spanish and German.
- As in Phases 2–3, these are LAST-position results. At AFTER, every test fails.
- Removing the German frequency axis (§3.7, test B) leaves all four verdicts unchanged. An apparent *gain* in German→Spanish transfer after the removal turned out to come from a change in which nouns entered the comparison, not from the removal itself, and that claim was retracted.

### 3.4 Phase 5: steering grammar into social readouts

**German first: no valid test.** Steering only means something in a sentence where the unmodified model already gets the grammar right (the frame check, §2.7.2) and where a push can switch the article (the gate, §2.8.1 step 4). Neither held for German:
- **8B:** in the best of the pre-registered German frames, the model got only 62% of nouns of one gender right without any push, below the 70% bar.
- **14B:** the frame passed, but the push failed the gate at every steering layer. It pushed the feminine nouns' article to switch but left the masculine ones unswitched, so it moved both genders toward the middle rather than swapping them.

By the rule that a failed check is never replaced by a better-looking frame, German has no Phase 5 verdict. **All Phase 5 verdicts are for Spanish only.**

**Spanish, R1 (the main readout).** The bleed score (§2.8.2) at α\*, averaged over the steering layers that passed:

| | 8B | 14B |
|---|---|---|
| R1 grammatical push | +0.020 (3/4 layers) | 0.000 (0/3) |
| R1 random 95th pct | 0.012 | 0.008 |
| R1 social push (control) | 0.064 | 0.047 |

- **8B: a small effect.** The grammatical push moves the adjectives in the social direction (+0.020) at 3 of 4 layers, above what random pushes do (95th percentile +0.012). R1-EN, the all-English readout, passes at 3 of 4 layers too. The German direction can't be tested (above), so by the labels of §2.8.4 this is **language-specific at most**: "bleed" would need German to agree. It is also small: the push toward *woman* moves the same readout three times as much (+0.064).
- **14B: no effect.** The grammatical push moves R1 by about 0, inside the range of random pushes, at all 3 layers. The positive control still works: the push toward *woman* moves the readout (+0.047). So the 14B readout *could* have detected an effect and didn't. That is what makes "no effect" a real null rather than a broken test.
- **A loose end on 14B (P26, exploratory).** Pushing the Spanish nouns with the *German* direction instead, at the same push size, moves R1 above random at 2 of the 3 layers, while the Spanish direction moves it by about 0. The two directions overlap only partly (T2 cosine 0.35, §3.3), so they can have different side effects. Being exploratory, this does not change the 14B "no effect" verdict. It does mean "no bleed at 14B" holds for the Spanish direction on Spanish nouns, and should not be read as "no grammatical push can do it".
- **R2** (*would it be a man or a woman?*) moved above random at every layer in both models (8B: 4 of 4, 14B: 3 of 3). But in that prompt the noun itself is the thing being asked about, so a push on the noun's grammar can change the answer just by making *la puente* sound like a feminine noun, which is grammar, not social meaning. R2 cannot tell those two routes apart, so it is reported but gives no verdict.
- **R3 and the "pooled" readouts.** R3 uses only adjectives with one form for both genders, which leaves few words. To use more, a later idea (P28) read the many adjectives that do have two forms (*blanco/blanca* "white") by adding up both forms' probabilities, hoping the sum would ignore agreement (R3-POOL steered, P20-POOL unsteered). These looked positive. P29 then tested the hoped-for property directly. Take the plain sentence, with no push at all, and swap only the article, *el* → *la*. If pooling really ignored agreement, the pooled score should not move. It moved by +0.096 on 8B, as much as or more than steering had moved it. Agreement leaks into the pooled sum, so these readouts cannot separate social meaning from grammar and are **not evidence**. They are reported in full as a cautionary result.
- **Confidence control** (§2.9): adding the model's overall confidence as a covariate, and using rank correlation instead of the regression, changes no Phase 5 verdict (nor any P29–P31 verdict).

### 3.5 English social gender (P22b, P23, P24, P27)

These exploratory tests check the two ends that steering is supposed to connect: does the model's behaviour reflect the human gender ratings at all, and does the *grammatical* direction already sort people by sex?

**Do the human ratings match the model? (P22b)** We compared *The woman is very ___* with *The man is very ___*, and *My mother is very ___* with *My father is very ___*, and took the difference in each adjective's log-probability (the model's own "woman versus man" lean for that word). We then ranked the adjectives by that lean and correlated the ranking with the Glasgow gender ratings (Spearman ρ, where 0 is no relation and 1 is the same order). The same was done in German with *Die Frau / Der Mann ist sehr ___*, using the English ratings of the translated adjectives.
- English: ρ = 0.51–0.59 across model sizes. German: 0.28–0.43.
- So the ratings capture something the model does. A readout built from them can detect social gender, and the positive controls in §3.4 confirm this.
- In Spanish, the lean is largely explained by pleasantness (valence) rather than the gender rating, at least on 4B. This is one more reason the Spanish-adjective readouts (R3 and the pooled ones) are treated cautiously.

**Does the grammatical direction sort people? (P23, P24)** The Spanish and German gender directions, found on *inanimate* nouns, were used to score English person words with a sex (*aunt/uncle*, *actress/actor*) and common first names (spelling matched by final letter):
- **The Spanish direction** sorts English person words by sex well (AUC 0.87 on 8B, 0.85 on 14B).
- **The German direction** does so less well (0.67 and 0.72). It was trained on fewer nouns (§2.4.3), so it is noisier, and that alone could make it weaker [interpretation, not tested].
- Each direction was also compared with a "social gender" direction built from English person pairs (*woman/man*, *mother/father*, …), using the two-halves cosine of §2.6.3. The overlap **shrinks with model size**: Spanish 0.44 → 0.22, German 0.18 → 0.08, from 8B to 14B (`docs/explainers/06-english-social-gender.md`). Larger models keep grammatical and social gender more apart, which fits the absence of bleed at 14B in §3.4.

**Grammar or sex? German neuter diminutives (P27).** German diminutives in *-chen/-lein* are always neuter, whatever the person's sex: *das Schwesterchen* "little sister", *das Brüderchen* "little brother". That separates grammatical gender from sex, because both nouns have the same grammatical gender but people of different sex. (The test also controls for the original noun's gender, *die Schwester*, since that might linger.) Results (Qwen3-1.7B and 4B):
- The German grammatical direction does **not** sort female from male diminutives (AUC 0.38–0.39, with no layer passing).
- It does score person diminutives above thing diminutives, whether the person is female or sex-neutral. So it reacts to "is a person", not to sex.
- Control: the English *social* direction sorts the same female and male diminutives easily (0.89–0.95). So sex information is there in those hidden states; the grammatical direction just doesn't read it.

### 3.6 Unsteered article pull (P30 Spanish, P31 German)

These tests use no steering. Recall the setup of §2.8.3: an invented noun is given a feminine or a masculine article, and we read which adjectives the model expects next. The score **S** is the bleed score applied to that article swap. It measures how much the feminine article (*la*, *die*) raises feminine-rated adjectives relative to masculine-rated ones, compared with the masculine article. S = 0 means the article doesn't matter for social meaning. Its units are "change in log-probability per point of gender rating"; only comparisons between S values matter here. Each context gets its own S:
- **object pull** is S in the sentences that make the noun a thing (*the N I bought yesterday*);
- **neutral pull** is S in the bare sentence (*the N is very ___*);
- **person pull** is S in the sentences that make it a person (*the N I met yesterday*).

The question is whether the object pull survives. If the pull came only from the model guessing that a *la*-noun may be a woman, it should vanish once the noun is clearly a thing.

**P30, Spanish.** All five models passed the checks (C1, C2).

| model | object | neutral | person | real person nouns | verdict |
|---|---|---|---|---|---|
| Qwen3-1.7B | +0.038 | +0.009 | +0.025 | +0.043 | pull survives an object reading |
| Qwen3-4B | +0.026 | +0.019 | +0.065 | +0.066 | pull survives an object reading |
| Qwen3-8B | +0.080 | +0.020 | +0.152 | +0.130 | pull survives an object reading |
| Qwen3-14B | +0.124 | +0.068 | +0.186 | +0.192 | pull survives an object reading |
| EuroLLM-1.7B | +0.008 | −0.000 | – | – | no neutral pull to explain |

*Source: `results/p30/<model>/analysis.json`; the object pull's 95% interval is above 0 for every Qwen3 model. "Real person nouns" are words like* el/la presidente*, where the article really does mark a man or a woman: the yardstick of a genuine social effect.*

- In every Qwen3 model a feminine article pulls toward feminine-rated adjectives **even when the noun is clearly a thing**. The pull is larger in the object sentences than in the neutral one. So the "maybe a woman" guess does not explain it.
- The person sentences pull more, about as much as real person nouns. That is expected: there the article does say something about a person.
- **What this does and does not show.** The object pull is not the "a woman" guess, but it is not yet shown to be *social*. Feminine-rated adjectives could, for example, simply be the ones that co-occur with feminine articles in Spanish text. The design does not separate those. P30 is therefore reported descriptively, not as evidence of bleed.
- EuroLLM shows no pull in the neutral sentence, so there was nothing for the object sentences to explain.

**P31, German.** The same design with *die* versus *der*. Neuter *das* is reported alongside. German predicate adjectives never change form (*die Flitt ist schön*, *der Flitt ist schön*), so every rated adjective can be used.

| Model | Verdict | S_object | object / neutral |
|---|---|---|---|
| Qwen3-1.7B | pull survives object reading | 0.035 | 1.95 |
| Qwen3-4B | pull survives object reading | 0.043 | 2.14 |
| Qwen3-8B | no object pull | 0.002 | (neutral 0.002; person 0.081) |
| Qwen3-14B | no valid test (C1 margin) | 0.025 *(exploratory)* | 0.73 [0.63, 0.85] |
| EuroLLM-1.7B | pull survives object reading | 0.081 | 2.32 |

*Source: `results/p31/<model>/analysis.json`. "object / neutral" is the object pull divided by the neutral pull.*

- In the three smaller models the object pull survives and is about twice the neutral pull: as in Spanish.
- **Qwen3-8B behaves differently.** There is no object pull and no neutral pull (both 0.002). There is a clear person pull (0.081). In German, 8B's article pull appears only when the noun is a person.
- **Qwen3-14B: no valid test.** Its object sentences did not lean toward "thing" by the required 1 nat (check C1, §2.8.3). A descriptive look with that rule waived (labelled exploratory) shows an object pull smaller than the neutral pull (ratio 0.73).
- The *das* − *der* object effect is about 0 on every Qwen3 model (at most 0.001) but not on EuroLLM (+0.045).

### 3.7 The German gender direction is aligned with word frequency

This finding came out of a control (§2.9), not a planned test. The question was whether the steering results could be explained by the model simply growing more or less confident. To check, we also estimated the direction along which the hidden states change with word **frequency**, and measured its angle with the gender direction, using the two-halves cosine of §2.6.3. A cosine of 0 means unrelated, and −1 means opposite.

- **Spanish:** the gender and frequency directions are unrelated (|cos| ≤ 0.14 at every layer).
- **German:** the feminine-minus-masculine direction points toward *rarer* words. On 8B the cosine is −0.34, −0.56, −0.62 and −0.69 at the four Phase 5 layers; on 14B it runs from −0.19 to −0.52. By the pre-registered rule (beyond the shuffle null *and* |cos| ≥ 0.3) this counts as an overlap.

**This is not because the feminine nouns are rarer.** In both training lists the feminine nouns are actually the *more* frequent ones (within spelling cells: Spanish by d = +0.52 standard deviations, German by +0.45). And frequency is removed before the direction is found (§2.6.2). Yet only German shows the alignment. Across the whole dictionary the genders barely differ in frequency (Spanish f − m +0.17 Zipf points, German −0.05), so the imbalance in our lists comes from how the samples were drawn, not from the languages.

**What we tried, each pre-registered** (§2.9):
- **A, a more flexible frequency control** (curved rather than straight-line, plus token count): the overlap stays.
- **B, remove the frequency direction** from every hidden state and rerun Phase 4: all Phase 4 verdicts are unchanged (§3.3). So nothing reported in §3.3 depends on it.
- **C, matched pairs**: 14 feminine/masculine German pairs with almost identical frequency, compared without any regression. The overlap stays.
- **D, the complex nouns** (§2.9): dropping them, or controlling for them, leaves the overlap.
- Across A, C and D the overlap disappears at no more than 1 of the 3–4 layers, never at the majority the pre-registered rule needs to call it "explained".
- **Register / age of acquisition** (are the feminine nouns the later-learned, more "written" words?): an age-of-acquisition control barely changes the overlap (8B −0.62 → −0.61 at the 55% layer).
- **SAE features (G2, Qwen3-8B).** A sparse autoencoder (SAE) splits a hidden state into many interpretable "features". The plan was to find the features carrying frequency and remove them from the direction. Result: **no valid test.** On the held-out German compounds (632 nouns) the overlap wasn't there to begin with (cos +0.11 to +0.26: the opposite sign, and under the 0.3 bar), so there was nothing to remove.

**Where this leaves it.** In the German training sample the gender direction is genuinely aligned with "rarer" in Qwen3 8B and 14B. No frequency control, pairing, word-structure or register explanation accounts for it. But it does not carry over to a different German noun sample (the compounds), which suggests it is at least partly a property of this sample. What the axis encodes is open. A new, pre-registered test on fresh single-root German nouns (rarity held-out, not yet run) and the Gemma 3 replication are the next checks.

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
