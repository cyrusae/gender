# Casey/Riley: deferred to its own phase (PI, 2026-10-27)

The Casey/Riley prompt ("Imagine a fictional character named {Casey|Riley}. {She is / He is / They
are} the human embodiment of …") is **not part of the first main run**. It differs from the writer
and character experiments: the pronoun is assigned, so the outcomes are the *description* (content
measures) and whether the model *keeps* an assigned *they*. The PI wants time to decide whether full
crossing of role × genre can be justified. This note records where things stood.

## What is already built
- Config `character/configs/pilot0_casey_v1.json` (pilot), builder `build_casey` in
  `character/gen.py`, checks `kept_pron` / `kept_name` in `character/code_outputs.py`.
- Both frames agreed: Consider = "Imagine …"; Writer = "You are a writer planning a short story. "
  + the same prompt.
- Roles (9): a character, the protagonist, the main character, the hero, the antagonist, the
  villain, a side character, a minor character, the love interest. Genres (10): none, literary,
  historical, young adult, science fiction, fantasy, mystery, thriller, horror, romance. No
  "heroine".

## Pilot findings (6 burned nouns, 4 API models, 575 paragraphs; pilot only)
- Content scores (bge-m3, in shuffle-null SD units; + = more feminine): she vs he +0.5, they vs he
  +0.2. **Role and genre move the description more than the pronoun does**: romance +1.5, horror
  −1.3, thriller −1.2 (vs no genre); the love interest +1.4, the antagonist −1.3, the hero −1.0.
  Genre explains ~26% and role ~20% of the variation left after noun, model and pronoun.
- Overridden *they*: gpt-oss-120b wrote *he* in 20/48 "They are" prompts, Qwen3.8-Flash switched
  12/48 (he for hammer/pot/door, she for flower); GLM-5.3-Flash and GPT-6-Luna never did.

## The design question being deferred
Current pilot scheme draws role and genre independently per prompt, so role/genre noise lands on
the pronoun contrast. Options:

| option | how | gives | cost per model, both frames |
|---|---|---|---|
| A. independent random (pilot) | new role/genre each prompt | noisy pronoun contrast | — |
| B. paired random | one role+genre per noun × name × draw, shared by all 3 pronouns and both frames | role/genre cancel out of she/he/they and Writer/Consider; role/genre effects exploratory | ~$1 at 2 draws |
| C. B + balanced | as B, assignments cycle so every role and genre appears equally | as B, cleaner role/genre estimates | as B |
| D. full crossing | all 90 role × genre per noun × name × pronoun | every pronoun × role/genre interaction testable (e.g. does "love interest" change she vs he) | ~$15 Luna, more on pod |

Recommendation at deferral: C, unless pronoun × role (the love-interest question) is to be
confirmatory, which pushes toward D.

## Decisions needed when this resumes
1. Main question: pronoun effect on content (role/genre exploratory), or pronoun × role/genre
   confirmatory?
2. SESOI for content: suggested 0.25 null-SD units (half the pilot she-vs-he effect), or 0.05 on
   the Glasgow 1–7 scale.
3. SESOI for overridden *they*: suggested 5 points in the override rate.
4. Then: power simulation as for the writer study (noise ≈ 1.2 null-SD per paragraph after role,
   genre, noun and model; the real limit will be noun-to-noun spread of the pronoun effect).
