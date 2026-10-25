# I0: The instruct pilot — turning things into characters (pilot 0, v1–v5)

Work from 2026-10-25, on branch `instruct`, folder `instruct/`. Design notes:
`instruct/docs/design.md`; rules: `instruct/CLAUDE.md`; decisions: `docs/decisions.md`
(entries tagged **[instruct]**). Everything here is **pilot**: the six nouns are burned and no
result on them is ever reported as a finding (see "What a pilot is for" below).

## 1. What the study asks

The base-model phases asked whether a model's *internal* representation of grammatical gender
overlaps with social gender. The instruct study asks the behavioural question about the models
people actually talk to: **when a chat model turns a thing into a character, does the character
get a gender, and which one?** Later versions ask whether that gender tracks German/Spanish
grammatical gender (the "bridge study" pattern: German *die Brücke* speakers call bridges
elegant, Spanish *el puente* speakers call them strong) and whether steering can push it.

## 2. The task

The model gets one prompt (the PI's wording), roughly: *"Consider a pot: what makes it
recognizable… Now invent a character for a short story who is the human embodiment of a pot.
Write one paragraph in the third person describing the character's appearance, behavior, traits,
and personality. Do not use the word "pot" in the paragraph."* plus an instruction about the
character's name, which became the main thing the pilot varied (section 5).

- **Modifiers.** The noun phrase is one of *a pot / the best pot / the worst pot / a stereotypical
  pot / an unusual pot*.
- **Nouns (6).** Picked by rule from the Wiktionary lexicon and the Glasgow Norms (human ratings,
  1 = very feminine, 7 = very masculine): two where German and Spanish disagree (pot: *der Topf* /
  *la olla*; keyboard: *die Tastatur* / *el teclado*), two where they agree (tree m/m, door f/f),
  and two English controls with strong human ratings (flower 1.66, hammer 5.97). Classics and
  Phase 4 pairs were kept out so the real study can use them. *Ship* was dropped as too
  interesting to burn.
- **Sampling.** Each completion is a random draw using the vendor's published settings (for Gemma 3:
  temperature 1.0, top-p 0.95, top-k 64). Twenty draws per noun × modifier × condition.
- **Model.** Gemma-3-12B-it (the instruct sibling of the base Gemma 3 used in the replication),
  through OpenRouter, pinned to one provider (DeepInfra) at bf16 with no fallback. All five
  versions together cost **$0.16**.

## 3. How outputs are coded (no model judge)

All coding is by fixed rules in `instruct/code_outputs.py`, so any reader can rerun it.

- **Pronoun class (primary).** *she* = only she/her/hers/herself; *he* = only he-forms; *they* =
  only they-forms; *mixed* = both he- and she-forms; *it* = none of those and at least two
  it-forms (a character called "it"); *none* = no pronoun for the character.
- **Validity gates.** An output is thrown out if the noun or its plural appears as a whole word,
  if the requested `Name:` line is missing or misplaced, if the name was banned from the
  paragraph and appears there anyway, if it was cut off at the token limit, or if the request
  failed. Invalid rates are reported per cell, never hidden. Synonyms ("vessel") are allowed: the
  ban is on the word.
- **Name gender.** The first given name (after stripping titles like "Old" or "Mrs") is looked up
  in US Social Security birth records: *f* or *m* if ≥ 95% of births are one sex, else
  *ambiguous*, or *not found* (pun names like "Threshold").
- **Names in free text** (when no `Name:` line is asked for): a capitalised word that doesn't
  start a sentence, or a sentence-initial word that is in the SSA list and is not an ordinary
  English word (WordNet). The first version of this rule skipped sentence-initial words and
  missed most names, because Gemma opens the paragraph with the name; it was fixed on principle
  and the fix is logged. Possessives ("Silas's") count as uses of the name.

**How to read the numbers.** Shares are among valid outputs. The *after* condition was run three
times in separate batches and gave nearly identical counts (he 311/312/311, she 99/99/99), so
batch-to-batch noise is a few outputs per 600. With 20 draws per cell, a single noun × modifier
cell has wide uncertainty (±20 points); compare totals or noun rows, not single cells.

## 4. First result (v2, name written last)

| noun | German / Spanish | Glasgow | he | she | they | it/none |
|---|---|---|---|---|---|---|
| hammer | m / m | 5.97 | 94 | 0 | 6 | 0 |
| door | f / f | 4.27 | 82 | 0 | 18 | 0 |
| pot | m / f | 3.91 | 59 | 6 | 35 | 0 |
| keyboard | f / m | 4.31 | 45 | 0 | 55 | 0 |
| tree | m / m | 3.89 | 31 | 26 | 34 | 9 |
| flower | f / f | 1.66 | 0 | 67 | 25 | 8 |

The two controls behave like the human ratings, so the measure can see gender. *He* is the
default: door is feminine in both languages and English-neutral, yet came out 82 he / 0 she.
Modifiers move pronouns about as much as nouns do ("unusual" → *they*; "best" → more *she*).

## 5. The name instruction changes the answer (v3–v5)

The names in v2 often disagreed with the pronouns ("Terra" with *he* 34 times), which raised the
question of whether *where* the name is written matters. Five ways of handling the name, each
600 completions (6 nouns × 5 modifiers × 20):

| condition | instruction | he | she | they | it/none |
|---|---|---|---|---|---|
| **after** | name banned from paragraph, `Name:` line last | 311 | 99 | 177 | 13 |
| **before** | `Name:` line first, name banned from paragraph | 388 | 131 | 79 | 1 |
| **told no name** | "Do not give the character a name." | 433 | 127 | 40 | 0 |
| **free** (v4) | nothing about names | 430 | 170 | 0 | 0 |
| **free_noted** (v5) | nothing about names except a final `Name:` line | 408 | 192 | 0 | 0 |
| **plan** (v5, all 600) | *after* + "Before you start writing, decide on a name" | 234 | 123 | 217 | 26 |

What each shows (pilot nouns, one model, so hypotheses):

- **Left to itself, Gemma names the character first, and the name sets the gender.** In *free*,
  99% of paragraphs open with a name; the pronouns match the name's SSA gender 600/600. The names
  come from a tiny stock: 16 distinct names in 600 (Silas 257, Elara 127, Bartholomew 107). The
  gender decision is effectively "Silas or Elara".
- **Asking for a name line changes nothing** (*free_noted* ≈ *free*), and it gives a confirmed
  name, so the coder doesn't have to find it. Adopted as the primary condition.
- **Writing the name last produces *they*.** Most of the *they* in v2 comes from that. A plausible
  reading: the model avoids committing to a gender it hasn't named yet. Pun names live here
  (door → "Threshold" 72 times, pot → "Terra", keyboard → "Algorithmus", German for algorithm).
- **Writing the name first** makes pronouns copy the name (agreement 99.8% vs 86% for *after*).
  Door got "Silas" 100 times out of 100.
- **Telling it not to name** gives the cleanest single pronoun choice and the strongest *he*
  default; *they* nearly vanishes.
- **The planning sentence backfired as a format.** 26% of *plan* outputs broke the `Name:` line,
  mostly by putting the stock name in the label slot ("Silas: Bartholomew Claybourne") or ending
  with a bare "Silas.". Failures cluster in door and hammer, so valid-only *plan* numbers would be
  biased; the table uses all 600. *They* went *up*, not down. One reading: "decide on a name"
  calls up the stock name, which the structure then forbids using.

**Carried forward (provisional, PI 2026-10-25):** *free_noted* primary; *before*, *after* and
*told no name* secondary (the last as a control on *free_noted*). Pun names are worth a "name
type" code later (stock / pun-or-word / ordinary).

## 6. What a pilot is for, and the tuning rule

The project rule is "don't tune measurement choices on a model's scores". In the instruct study
the pilot nouns are burned, which allows the standard split between an **exploratory** phase
and a **confirmatory** one: on reserved pilot nouns, prompts may be revised on any grounds,
including outcomes. Before any non-pilot noun is run, the prompt is frozen and pre-registered,
and the write-up says which pilot results informed it (`instruct/CLAUDE.md`). Pilot results are
never reported as findings. What this does not protect against: choosing the prompt that gives
the biggest effect here means the real study inherits that choice, which is why the write-up has
to say how the prompt was chosen. Here, the choice was driven by measurability (a confirmed
name, no *they* deferral), not by which condition gave the most gender effect.

## 7. Limits worth stating

- One model, six nouns, one provider. Nothing here generalises yet; v6 (Llama-3.3-70B,
  Gemma-4-31B with and without thinking, GPT-6-Luna) is the first cross-model look.
- "Gender" is measured as pronouns plus SSA name gender: US-centric, binary at the name level,
  and blind to gender cues in the description itself (that's what a pre-registered judge would
  add later, never as the primary outcome).
- The name stock (Silas/Elara) means one categorical pick drives the outcome. Small changes in
  how the model chooses names could swing the totals.
- Modifier effects are large and would need to be part of any real design.
- Name spotting in free text misses names that are also ordinary words (Rose, Quill) when they
  only appear at the start of a sentence (1 case in 600).
- API models can change silently. Each run records the provider, precision and date; conditions
  that are compared run in the same batch.
