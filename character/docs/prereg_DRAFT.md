# Character study: pre-registration (ADOPTED by PI 2026-10-27)

Status: **draft for PI review. Nothing in the main run starts until the PI signs this off.** Every
item below is already logged in `docs/decisions.md` with its reasons; this file collects them in one
place. Items marked **[OPEN]** still need a decision.

## 1. Questions

- **Q1 (writer study, confirmatory).** When a model is asked to write an English character who is
  the "human embodiment" of a foreign word, does the noun's grammatical gender in that language
  push the character's pronoun toward that gender, compared with the same writer persona given the
  English word?
- **Q2 (character study, confirmatory).** Do adjectives change the pronoun an English character
  gets ("a rough door" vs "a door")? Do antonym pairs differ (rough vs smooth)?
- **Q3 (Casey/Riley): DEFERRED to its own phase** (PI 2026-10-27; `character/docs/casey_riley_deferred.md`).
- **Q4 (thinking, exploratory).** Does thinking mode change the outcomes in Qwen3-14B? A full
  thinking-on run of every experiment (adopted, PI 2026-10-27), labelled exploratory.
- **Q5 (frame, confirmatory).** Does answering as a writer persona differ from answering from the
  model's own point of view? Every experiment runs in two **frames** (PI 2026-10-27):
  - **Consider frame:** the model is addressed directly ("Consider …", "Imagine …").
  - **Writer frame:** the same task given to a writer persona.
  The frame contrast is tested within noun (and within arm or modifier) as the she-share difference
  Writer − Consider; each experiment's own effects (Q1, Q2) are also reported per frame.

## 2. Models and routes

- **GPT-6-Luna** via OpenRouter (API route), vendor-default sampling.
- **Qwen3-14B**, thinking off (primary) and thinking on (exploratory, Q4, Qwen's thinking-mode
  sampling defaults T 0.6 / top-p 0.95 / top-k 20), bf16, vLLM on one RunPod GPU (pod route), sampling T 0.7 / top-p
  0.8 / top-k 20 (Qwen's non-thinking defaults).
- gpt-oss-120b: not in this study (PI 2026-10-27, budget); a possible later replication.
- One route per model per analysis (API and pod outputs differed for Gemma-3-12B).
- Both main models passed fishing rule v2 (she among he+she 25–75% in both pilot conditions).

## 3. Stimuli (frozen at sign-off; files renamed from `_DRAFT` to `_v1`)

- **Nouns:** 191 (`character/results/confound/pool_191_DRAFT.csv`): all eight translations
  round-trip in each language's own Wiktionary, genders from Wiktionary, inanimate, no groups of
  people, confound-trimmed (|r| < 0.2 between gender and concreteness in every gendered language).
  Pilot nouns are burned and excluded (`reserved_nouns.csv`).
- **Writer arms (17 per noun):** for each of de, ru, es, hi, ar, fi, ja, tr: the foreign word, and
  the same persona with the English word; plus one plain arm with no persona. The word is quoted,
  with no article. Foreign words are in everyday spelling (no stress or vowel marks).
  `character/stimuli/main_writer_v1_DRAFT.csv`.
- **Character arms:** 191 nouns × 81 modifiers (`main_character_v1_DRAFT.csv`): the PI's list v2
  (70), plain forms a/an (primary), the, my, your, and 6 wugs (miryal, tisbal, gundle, sloke,
  distock, sputul) plus frindle (an extra, never analysed). 24 mass nouns drop every indefinite
  article. Idiom flags on *a confirmed*, *a real*, *a perfect*, *the perfect*, *an unnatural*,
  *a common*.
- **Prompts:** templates in `character/docs/prompt_drafts.md` (PI versions 2026-10-26), ending
  with the `free_noted` name line, in two frames:

  | experiment | Consider frame | Writer frame |
  |---|---|---|
  | writer (Q1) | **new, option (b):** the writer prompt with the persona removed: foreign word (8 languages) and English word, 9 arms | PI's writer prompt: 8 languages × foreign / English word + plain "You are a writer" = 17 arms |
  | character + modifiers (Q2) | PI's original "Consider {MOD_NOUN}…" prompt, 81 phrases | **new, option (a):** "You are a writer planning a short story. " + the same Consider prompt (the only difference is that sentence) |
  | Casey/Riley (Q3) | PI's "Imagine a fictional character named…" prompt | "You are a writer planning a short story. " + the same prompt |

  Consider-frame writer wording (PI-approved 2026-10-27; "your character" as in the writer frame,
  so the frames differ only in the persona/opening sentences):
  > Consider {WORD}. Invent a character for a short story who is the human embodiment of {WORD}.
  > Everything about your character is inspired by the nature of {WORD}: behavior, personality,
  > appearance, and other traits reflect {WORD} as you understand it. Write an English-language
  > paragraph about your character. Do not use the word {WORD} or its English translation in the
  > paragraph. {NAME_INSTR}

  ({WORD} quoted; English-word arm without the translation clause, as in the writer frame.)
  The writer-frame persona sentence for character and Casey/Riley is the plain arm's first sentence,
  so the persona is identical across experiments.

## 4. Outcomes and coding

- **Primary outcome:** pronoun class of the character by regex (`character/code_outputs.py`):
  she / he / they / it / neo / mixed / none. **The she-vs-he analyses use she / (she + he).**
  *They* and *it* are reported as their own outcomes, never folded in.
- **Validity gates (automatic):** noun or plural used (any arm); foreign word or English noun used
  (writer foreign arms); Name line missing or misplaced; truncated; empty paragraph; API error.
  Invalid outputs are excluded and their rates reported per model and arm.
- **Secondary (content, Q3 only in this round):** Glasgow word average and the six frozen embedding
  scorers, with character-only pronoun masking (primary) and mask-every-*they* (sensitivity);
  property anchors (primary), people anchors (secondary). Model-based measures are never the
  primary gender outcome.
- **The coder is frozen at sign-off** (git commit recorded). Bugs found later are fixed on
  principle and both versions reported.

## 5. Analyses

### Q1 writer (confirmatory)
- Per noun × gendered language: shift = she-share(foreign arm) − she-share(English-word arm).
- **Effect = feminine-minus-masculine difference in shift** (neuter separate). SESOI: 5 points
  pooled over de, ru, es, hi, ar.
- Model: mixed-effects logistic regression of she vs he, with noun random intercepts, arm × gender
  (sum-coded), language as a fixed effect; per model separately. Robustness: the noun-clustered
  regression on cell shifts used in the power simulation, and a permutation test that shuffles
  gender labels within language.
- **Decision:** effect if the pooled 95% CI excludes 0 and the estimate is positive; "no effect of
  interest" if the 90% CI lies inside ±5 points (TOST equivalence); otherwise inconclusive.
- **Per-language effects: estimates with CIs, not hypothesis tests** (10-point SESOI per language
  reaches 80% power only if noun spread τ ≲ 0.10).
- Persona alone (PI's null): English-word arm vs plain arm, reported per language.
- Genderless languages (fi, ja, tr) give a check: the foreign-vs-English shift should not depend on
  any of the gendered languages' genders.

### Q2 character (confirmatory)
- Each modifier vs plain a/an (or its own baseline: the/your for "the …"/"your …" modifiers), and
  each antonym pair, she-share within noun. Holm correction across modifiers.
- Taxonomy groupings are exploratory (PI 2026-10-26); a later taxonomy can be confirmed by a split
  of the nouns or a rerun.
- Wugs: (a) wug vs plain (any-adjective effect); (b) miryal vs sputul (sound symbolism, one word
  per side, so reported as an estimate).

## 6. Sample size and stopping

- Precision-based stopping, **cap 30 draws per arm**, first batch 10. Writer target: pooled 95% CI
  half-width ≤ 2.5 points. The stopping rule looks only at CI width, never at the direction or the
  p-value. Common stopping point across models where budget allows.
- **Character arms: precision rule, target ±5 points** (PI 2026-10-27, matching the other SESOIs):
  each modifier's effect vs plain a/an (mean over nouns of the she-share difference) must reach a
  95% CI half-width ≤ 5 points. Plain a/an × 10 draws per noun; modifiers start at 2 draws per noun
  and add draws in steps of 2 until the median modifier half-width ≤ 5 points, cap 6. Same draws in
  both frames and every model's cell layout. Simulation (`character/scripts/power_character.py`):
  2 draws suffice when ≥ ~90% of outputs are he/she (Luna); about 4 are needed at ~55% (Qwen3-14B
  thinking off, which writes *they* often).
- Power simulation: `character/scripts/power_sim.py`, results `character/results/power/`.

## 7. Budget (caps; precision stopping usually ends earlier)

Prices from the pilots: Luna ≈ $0.07 per 1,000 completions; Qwen3-14B on an RTX PRO 6000 ≈ 36,000
completions per hour thinking off at $2.49/h, ≈ 1.5× the time thinking on.

| experiment (both frames) | completions per model (cap) | Luna | Qwen3-14B off | Qwen3-14B on |
|---|---|---|---|---|
| writer: 26 arms × 30 draws × 191 nouns | ~149,000 | ~$10.50 | ~$10.50 | ~$16 |
| character + modifiers: 2 frames, 2–6 draws | ~74,000–200,000 | ~$5 | ~$10 (4 draws) | ~$12 |
| pod setup, downloads | | | ~$2 | (same pod) |
| **total** | | **~$16** | **~$23** | **~$28** |

**Expected total ≈ $65; hard cap ≈ $75** (all models at 6 modifier draws).

## 8. Open items

None for the first main run. Deferred: Casey/Riley (own phase, `casey_riley_deferred.md`);
gpt-oss-120b (possible later replication).

## 9. Deviations

Any change after sign-off is logged in `docs/decisions.md` with date and reason, and reported in the
write-up.
