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
- **Any language, from the dictionary:** `wk_structure` (built 2026-10-22) reads Wiktionary's
  etymology templates (`compound`, `affix`/`af`, `suffix`, `prefix`, …). A noun counts as complex
  if some part of it is still a word of the language (*Wasser* + *Kocher*, *скорый* + *-ость*).
  Reconstructed or older-stage parts (*Kummer* ← *\*kom-*; *Wimper* ← Old High German) don't count.
  - On the German training nouns it finds all 12 of the hand rule's nouns, plus 8 transparent
    derivations the hand rule misses (*Benziner* ← *Benzin*, *Stößel* "pestle" ← *stoßen*): 27% of
    masculine vs 11% of feminine.
  - It also gives each language's suffix inventory (Russian: *-ка*, *-ость*, *-ение*, *-ник*…),
    which is the raw material for suffix cells.
  - Its limit: words Wiktionary lists only as inherited (*радость* "joy") count as simple even when
    historically derived.
- **Hand rule where needed:** `frequency_axis.complexity_rule` (German: numeral + suffix,
  noun + noun, verb + agent suffix, noun + verb + agent suffix) catches agent-noun structure
  Wiktionary may not template.
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
- semantic classes that take one gender by convention (German chemicals; Spanish and German
  months, weekdays and seasons; numbers used as nouns). `gbleed lang-audit` flags months,
  weekdays, seasons and numbers from the English gloss (`lang_audit.semantic_class`);
- the held-out sets of every phase.

**Learned from:**
- **"The FIFA World Cup"** is a gloss naming one thing: it counts as a proper name (PI review,
  Russian).
- **Months are masculine in Spanish by meaning, not form** (*marzo*, *noviembre*). Two slipped
  into the Spanish lists (one training noun of 132 masculine, one test/steering noun); found by
  the PI from the word-level export, 2026-10-25. Negligible for the results, but a cue spelling
  controls can't remove.
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

- **`gbleed lang-audit LANG [--list CSV]`** (built 2026-10-22, `src/gbleed/lang_audit.py`): Stages
  2, 3 and 7 in one report (`results/lang_audit/<lang>/<list>/report.md` + tables).
  - **Spelling:** leave-one-out gender accuracy from the last 1–3 letters; the residual cue within
    cells; initial letters.
  - **Balance:** length, tokens, Zipf, AoA and concreteness by gender within cells.
  - **Structure:** complex-noun share by gender. Fixed flag thresholds.
  - **Validated on our own training lists:** it flags the Spanish *a*-initial cue (17 m, 0 f), the
    German complex-noun imbalance (19% m, 0% f) and the feminine-commoner frequency asymmetry in
    both.
  - **It nearly didn't.** The first version missed the *a*-initial cue: at 20% feminine overall,
    "0% feminine" is only a 0.20 departure. Now a gender that is *missing* where ≥ 2 were expected
    is flagged too. Initial letters are accent-folded (*á* = *a*).
- **Per-language config** (built 2026-10-22, `src/gbleed/languages.py`): genders, `wordfreq`
  code, cell rule, hand structure rule, notes, whether the language has articles, and pointers
  to its known-check frames (they stay next to their scoring code, which frozen results depend on).
- **Wiktionary word structure** (built 2026-10-22, `src/gbleed/wk_structure.py`): see Stage 3.
  Reported by `lang-audit` for every language, next to any hand rule.
- **Phase 4 for N languages** (built 2026-10-22, `src/gbleed/phase4n.py`): T1 for every ordered pair, T2
  for every unordered pair, T3 where flipped pairs exist, pooled over all languages. With
  Spanish + German it reproduces `phase4.analyze` **exactly** (same random draws, every column).
  Verdict rules for new pairs are left to the pre-registration that adds the language.
- Phase 4 for N languages (pairwise T1/T2 for every pair; T3 on pairs per language pair).

## Phase 2 queue (PI, 2026-10-26)

Phase 1 base-model data collection is closed. Next, each with its own prereg:
1. **French**, then **Dutch** (start Dutch when French reaches Stage 7).
2. **Russian Stage 0 next steps:** masculine person nouns with feminine-type endings (*папа*
   "dad", *мужчина* "man"); common-gender person nouns (*сирота* "orphan", *коллега* "colleague")
   as a social-gender yardstick; grammatical animacy as a second category.

## Russian: where it stands

| Stage | Status |
|---|---|
| 0 question | **set (PI, 2026-10-22):** "Does a gender direction learned without articles align with Spanish/German's, and is German's frequency entanglement shared?" **Next step after that:** masculine person nouns with feminine-type endings (*папа*, *мужчина*: meaning overrides form); common-gender person nouns (*сирота*, *коллега*) as a social-gender yardstick; grammatical animacy as a second category alongside gender. |
| 1 lexicon | done (provenance recorded) |
| 2 spelling audit | lexicon-wide audit done (2026-10-22). The ending class predicts gender 97% of the time, but **within ending classes feminine nouns are far more abstract** (concreteness d = −1.38) and longer (+0.84), almost certainly the abstract-noun suffix *-ость* inside the *-ь* class. A training set needs suffix cells (as German's). |
| 3 structure audit | first pass (Wiktionary templates): **32% of feminine vs 13% of masculine eligible nouns are derived/compound**, the *-ость*/*-ка* pattern. Suffix cells needed. |
| 4 meaning filters | done for the *-ь* pool (PI reviewed) |
| 5 frames | done (*наш/новый*, frame check passed on Qwen3-1.7B/4B and EuroLLM-1.7B) |
| 6 sets | *-ь* list only; a training set across endings is needed for a Russian direction |
| 7 balance report | exploratory preview (2026-10-22, `lang-audit ru --cells suffix`). Wiktionary-suffix cells cut the within-cell gaps for length (+0.84 → +0.18) and AoA (+0.49 → +0.23), but concreteness stays large (−1.38 → −0.85). Feminine abstract nouns are likely untemplated (*радость* is listed as inherited) or Russian feminines really are more abstract. **For the PI's review:** a spelling-based *-ость* cell, concreteness strata, or concrete nouns only. |
| 8 pairs | not started |
| 9 extraction | *-ь* nouns extracted on the pod for 8B/14B (Phase 4 T4) |
