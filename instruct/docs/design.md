# Instruct study: design notes (DRAFT, 2026-10-25)

**Question.** Do chat models, the ones people actually talk to, give things a gender when they
turn them into characters? And later: does that gender follow German/Spanish grammatical gender
(the "bridge study" pattern), and can steering push it there?

**Pilot 0 = "does the workflow work at all".** Not a test of anything. No verdicts; it checks
that prompts produce a single person-paragraph, that pronouns can be coded by regex, that
thinking mode closes its `<think>` block within budget, and how variable outputs are.

## Pilot 0 design

- Prompt (PI draft, verbatim): `configs/pilot0_v1.json`. Modifiers: plain / best / worst /
  stereotypical / unusual.
- Nouns (7, `stimuli/pilot0_nouns_v1.csv`), **burned** for later instruct lists
  (`reserved_nouns.csv`). Genders from the Wiktionary lexicon; English gender association from
  the **Glasgow Norms** (Scott et al. 2019; human ratings, 1 = very feminine, 7 = very masculine).

  | noun | German | Spanish | Glasgow | role |
  |---|---|---|---|---|
  | pot | der Topf (m) | la olla (f) | 3.91 | flipped |
  | keyboard | die Tastatur (f) | el teclado (m) | 4.31 | flipped |
  | tree | der Baum (m) | el árbol (m) | 3.89 | both m |
  | door | die Tür (f) | la puerta (f) | 4.27 | both f |
  | ship | das Schiff (n) | el buque (m) / la nave (f) | 5.00 | English "she" convention |
  | flower | die Blume (f) | la flor (f) | 1.66 | English feminine control |
  | hammer | der Hammer (m) | el martillo (m) | 5.97 | English masculine control |

  Why not bridge/key or other classics: the pilot burns its nouns, and the classics and Phase 4
  pairs are the items the real study will want (keyboard is the one Phase 4 pair used: no
  non-classic f/m pair outside Phase 4 had a Glasgow rating).
  Your intuitions, checked against human ratings: **flower** is strongly feminine (1.66);
  **rock** moderately masculine (5.46); **ship** is only mildly masculine (5.00): the "she" for
  ships is a pronoun convention, not a feminine association, which makes it an interesting item.
  The "little base-model effect" noun is left out: no per-noun base score exists yet (computing
  per-noun projections on the social-gender direction is a separate, cheap Mac job).
- Models: `Qwen/Qwen3-8B` (instruct sibling of the 8B base; thinking on and off) and
  `google/gemma-3-12b-it` (sibling of the 12B pt; no thinking mode). 20 samples per cell →
  Qwen 7 × 5 × 2 × 20 = 1,400, Gemma 700.
- Sampling: each model's published defaults (Qwen3 thinking T 0.6 / top-p 0.95 / top-k 20;
  non-thinking T 0.7 / 0.8 / 20; Gemma 3 T 1.0 / 0.95 / 64). Not tuned.
- Backend: vLLM, bf16, one L40S (≈$1.1/h); expected well under an hour including downloads.

## Measures (all deterministic in the pilot)

- **Primary: pronoun class of the paragraph** (`code_outputs.py`): she / he / they / mixed / none,
  and `share_f` = she ÷ (she + he). "mixed" usually means a second character or a slip; "none"
  often means the model described the object instead of a person (a manipulation-check failure).
- Gendered person nouns (woman, man, queen, king…), word counts, truncation rate.
- CoT flags (thinking runs): talks about gender; mentions other languages / grammatical gender;
  mentions ships as "she". Counts are flags to read, not results.

## Rule changes (PI to sign off)

1. Instruct models allowed inside `instruct/` only (root rule stays for base-model work).
2. Model classifiers may code *outputs* (not inputs) under four conditions (`instruct/CLAUDE.md`):
   regex can't do it; frozen + pre-registered; validated against PI hand-coding in English;
   blind to condition. I agree it's a different thing from gold input labels, with one caveat: a
   classifier judging "is this character feminine-coded" brings its own gender stereotypes, so
   it must never decide the primary gender outcome (pronouns stay regex).
3. Sampling settings fixed to vendor defaults.

## Pilot 0 v2 (adopted 2026-10-25)

- Prompt `configs/pilot0_v2.json`: v1 + `Do not use the word "{noun}" in the paragraph.` + no name in
  the paragraph + a final `Name: <name>` line. Pronouns are written before the name exists.
- Nouns `stimuli/pilot0_nouns_v2.csv`: 6 (ship dropped, kept for the real study).
- Gates (`code_outputs.code_v2`): noun/plural as whole word in the paragraph; Name line missing or
  not last; truncated; API error → invalid. Rates per cell are reported. Paragraph count recorded,
  not gated.
- Names: first given name (after titles/descriptors such as Mrs, Old, Man) looked up in SSA US
  baby names, all years: f/m if ≥ 95% one sex, else ambiguous, or not found. Expect many invented
  fantasy names (not found). Origin coding (Wiktionary given-name entries, Behind the Name) later.
- Venue: OpenRouter for prompt-format iteration on Gemma-3-12B-it pinned to bf16 (DeepInfra, no
  fallbacks); Qwen3 is only served in fp8/int4 there, so Qwen (and thinking) waits for the pod.
  Every row records the serving provider and cost.
- Limits of API runs: unknown serving stack; frontier models return reasoning *summaries*, not raw
  CoT; reasoning models may ignore temperature; models change without notice (pin dated ids).

## Later (not built)

- Gender-flipped version (John/Jane/Jamie × role × genre; "do not mention the noun") — the
  outcome there is trait/stereotype content, which is where a classifier enters.
- "You are a Spanish/German-speaking writer… {word in that language}": does the model impute
  a grammatical-gender effect to an imagined L2 writer?
- Steering (English and in-language directions from the base study) on the instruct siblings.
- Log-prob version of the primary outcome (P(she) vs P(he) at the first pronoun) as a cheaper,
  less noisy complement to sampling.

## Writer-prompt study: reminders (PI, 2026-10-25)

- **Include an English-word control arm:** "You are a {language}-speaking writer … a {ENGLISH WORD}".
  Same writer persona, but the noun stays in English. Question: does the persona alone (no
  foreign word on the page) pull pronouns toward the noun's gender in that language? PI's null:
  no effect. Contrast with the bare-foreign-word arm separates "persona" from "word" effects.
- Writer arms: bare word, no modifiers; article form = de/es secondary.
