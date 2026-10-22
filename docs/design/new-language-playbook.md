# Playbook: adding a new language

Written 2026-10-22, after two languages built in full (Spanish, German) and one in part (Russian:
lexicon, a *-ь* pool, known-check frames). Every step below exists because something went wrong,
or nearly did, the first time. The "**Learned from**" lines are those stories (and are blog
material). Steps marked **[PI gate]** need the PI's sign-off; everything shown to the PI is
in English (glosses) and built by rule, because the PI won't read most of these languages.

The order matters: cheap automatic audits come *before* anyone looks at a word list, so the
PI reviews a list whose quirks are already measured, not discovered later.

---

## Stage 0. Choose the language for a question **[PI gate]**

Write down which question the language answers *before* building anything. A new language is
useful for the contrast it creates with the ones we have.

| Property | Spanish | German | Why it matters |
|---|---|---|---|
| genders | 2 | 3 | does a third gender change the geometry? |
| articles mark gender | yes | yes | without them, frames and readouts need agreement (adjectives, verbs) |
| gender predictable from spelling | mostly (*-o/-a*) | weakly | how hard the spelling controls have to work |
| compounding | rare | heavy | word-structure confounds (Stage 3) |
| script / tokenisation | Latin | Latin | non-Latin scripts tokenise into more pieces: token count matters more |

Also check, per candidate:
- the Wiktionary (kaikki.org) dump's coverage of noun genders;
- `wordfreq` support (frequencies);
- whether the model families (Qwen3, EuroLLM, Gemma) were trained on the language;
- how many English glosses fall in the rating norms we use (Glasgow, concreteness, AoA).

**Current candidates:** Russian (3 genders, no articles: is German's frequency entanglement
about 3-gender systems?); French or Italian (is Spanish's cleanness Romance-wide?); Dutch
(masculine/feminine merged into "common": is the German entanglement tied to the feminine
category?).

**Learned from:** "German" turned out to bundle three genders, articles, heavy compounding and
unpredictable spelling, so a German-only result can't say which property drives it.

## Stage 1. Lexicon from Wiktionary

`gbleed lexicon` parses the kaikki.org dump into one row per noun: gender(s) from sense tags,
other parts of speech with the same spelling, glosses, frequency, WordNet class of the English
gloss, animacy. Per language you add:
- the dump, with **provenance** recorded (URL, date, SHA-256, counts);
- language-specific fields (Russian: grammatical animacy, indeclinable nouns);
- the ending-cell rule (German: last two letters, except shared suffixes, which get their own cell).

**Learned from:**
- **Form-of and alternative-spelling senses** must be skipped, or inflected forms enter as nouns.
- **Homographs corrupt frequencies:** `wordfreq` lowercases, so German *Aber* got the frequency of
  *aber* "but". Record other parts of speech and drop homographs.
- **English-gloss judgements can be overruled by the language.** Russian grammar marks animacy, and
  plants and objects the English gloss called "animate" are grammatically inanimate. Grammar wins
  (PI review).

## Stage 2. Spelling audit (automatic, before any list)

How much does the spelling give gender away? Computed on the lexicon, with no model:
- the **ending table**: gender shares for each final 1, 2 and 3 letters (with ≥ 50 nouns);
- **initial letters** by gender;
- **length and token count** by gender (in each model's tokenizer);
- **frequency by gender.**

These define the cells (spelling is held constant within a cell) and the covariates.

**Learned from:**
- **Phase 1:** linear "spelling erasure" didn't generalise, so spelling has to be controlled by
  design (cells), not removed afterwards.
- **Initial *a-* in Spanish:** excluding feminine *el agua*-type nouns left 17 masculine and 0
  feminine training nouns starting with *a-*, a perfect spelling cue our own filter created.
  An exploratory SAE look found it, not a planned check. Now it's a planned check.
- **Feminine nouns are the commoner ones in both our training sets** (German Zipf 3.77 vs 3.28),
  which later complicated a transfer result (frequency-axis threads 1–2).

## Stage 3. Word-structure audit (automatic)

Flag compounds, derived words, diminutives, agent/instrument nouns, nominalised verbs and
numeral nouns, then check how they're **distributed by gender**. An imbalance is a confound even
when each rule is sensible on its own.
- Start from `frequency_axis.complexity_rule` (German): numeral + suffix, noun + noun, verb +
  agent suffix, noun + verb + agent suffix. Port it per language: suffixes, linking elements,
  numeral words.
- Keep diminutives and pattern-assigned genders out (German *-chen*/*-lein* and *Ge-…-e* are
  neuter by pattern).

**Learned from:** German's "simplex only" filter tested only the whole word minus *-er*
(*Bohrer* ← *bohren*). It missed compounds with such a head (*Wasserkocher* "kettle") and numeral
nouns (*Vierer* "a four"). That left **12 of 64 masculine and 0 of 27 feminine** training nouns
complex, found three phases later. A rule that's right for each word can still be lopsided
across the set.

## Stage 4. Meaning filters

Inanimate nouns only, so there is no natural sex. Also out:
- proper names, including glosses naming one specific thing;
- sex-typed items (garments);
- semantic classes that take one gender by convention (German chemicals);
- the held-out sets of every phase.

**Learned from:**
- **"The FIFA World Cup"** is a gloss naming one thing: it counts as a proper name (PI review,
  Russian).
- **Glosses can encode a viewpoint.** Dictionaries written from one sex's perspective
  (*boyfriend* "the lover of a girl") are why gold labels never come from a model and glosses are
  checked.

## Stage 5. Frames and the known check

Does the model know each noun's gender? Two sentence frames per language, scored as a margin
(log-probability of the noun with the right article or agreement vs the wrong one); "known" means
right in both by ≥ 1 nat.
- **Without articles,** frames use agreement: Russian *наш/наша* "our (m/f)" and *новый/новая*
  "new (m/f)".
- **Frame check:** the frames themselves are tested on two model families (Qwen3 + EuroLLM)
  before they judge anything. A frame that fails isn't swapped for another after seeing results.
- **Formatting rules:** never put the scored word first (Qwen3 has no BOS token); prefix bare
  words with `<|endoftext|>` + newline (attention sinks).

**Learned from:**
- **German Phase 5's frame failed** on 8B and 14B, so that test was lost. Switching frames after
  seeing the failure would have been choosing on results.
- **The first token after a separator can be an attention sink** (one dimension ~2,500×
  typical).

## Stage 6. Sets, splits and held-out rules

- Train/test assigned once, by stem, with a fixed seed. Every stimulus file carries a `split`
  column.
- **Held-out rules are code, not prose:** each builder asserts it contains nothing from any
  earlier phase's test sets.
- Lists are versioned. A list with results is frozen; changes mean a new version.

**Learned from:** the Phase 2 builder never excluded the Phase 4–5 flipped pairs, and **37 leaked**
into training. Since then, every builder asserts.

## Stage 7. The balance report (automatic) **[PI gate]**

Before the PI sees a list, one script reports, by gender and cell:
- counts;
- Zipf frequency;
- token count;
- initial letters;
- the complexity flags;
- age of acquisition and concreteness of the English glosses;
- gloss coverage in each rating norm;
- the residual spelling cue (how well the last 3–4 letters still predict gender within cells).

The PI reviews counts, trade-offs and the drops each rule makes, all with English glosses.
Drops need sign-off even under an existing rule.

**Learned from:** every imbalance we found late (initial *a-*, the German complex nouns, the
frequency asymmetry) would have shown up in this one table.

## Stage 8. Translation pairs (for cross-language tests)

Flipped pairs (*la luna* / *der Mond*) are matched through English glosses. Even strict rules
are about **10% wrong** (*Platte* "flat [plate]" ↔ *apartamento* "flat [apartment]"). So pairs are
reviewed in English before use, and English cognates are flagged (with a sensitivity analysis
without them).

## Stage 9. Extraction

- Frozen lists.
- **Commit before running** (results record the commit).
- One device and precision per model; cross-size runs all on the pod (bf16).
- Batch every new language into **one pod session**.
- Bulky arrays go to the external store (`paths.py`).

## Stage 10. Analyses (already built, reused)

- **Gender direction:** stratified difference of means, plus a probe; split-half geometry with
  the cross-term null.
- **Cross-language:** transfer (T1), flipped pairs (T3), geometry (T2). *Needs one-time work:
  Phase 4 is written for exactly two languages.*
- **Controls:** the frequency-overlap check (`confidence_ctl` (c)), the baseline-probability
  covariate, the AoA check. Run them on every new language from the start, not as follow-ups.

## Stage 11. Write it down

- A `docs/screening/NN-<language>.md` (every filter, in order, with counts).
- `decisions.md` rows.
- An explainer section.
- Glossary terms.
- Results in the project outline.

---

## One language at a time, or several?

**Build lists one at a time; batch extraction and analysis.** Stages 1–7 are where each language's
quirks bite, and each needs PI review. Two in flight at once doubles the surprises and is harder
to track. Stages 9–10 batch well: one pod session, one analysis pass.

Suggested order:
1. **Russian:** its lexicon, pool and frames exist (branch `russian`). It needs a full training
   set (not just *-ь*), Stages 2–3 and 7, and an extraction run.
2. **French**, then **Dutch**: start the second when the first reaches its Stage 7 sign-off.
   French reuses Spanish's rules, and Dutch reuses German's.

## One-time code to make this fast

- `gbleed lang-audit LANG`: Stages 2, 3 and 7 in one report (spelling table, structure flags,
  balance by gender), generalising what's now spread over `phase3_stimuli`, `p31.ending_table`,
  `frequency_axis.complexity_rule` and `register_axis`.
- A per-language config (frames, agreement words, cell rule, structure rules, special filters) in
  one place, like `phase0.LANG_CONFIG`.
- Phase 4 for N languages (pairwise T1/T2 for every pair; T3 on pairs per language pair).

## Russian: where it stands

| Stage | Status |
|---|---|
| 0 question | **to write:** "is German's frequency entanglement about 3-gender systems / feminine nouns?" |
| 1 lexicon | done (provenance recorded) |
| 2 spelling audit | partial (the *-ь* class chosen because its gender is unpredictable from spelling) |
| 3 structure audit | to do |
| 4 meaning filters | done for the *-ь* pool (PI reviewed) |
| 5 frames | done (*наш/новый*, frame check passed on Qwen3-1.7B/4B and EuroLLM-1.7B) |
| 6 sets | *-ь* list only; a training set across endings is needed for a Russian direction |
| 7 balance report | to do |
| 8 pairs | not started |
| 9 extraction | *-ь* nouns extracted on the pod for 8B/14B (Phase 4 T4) |
