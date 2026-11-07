# Research brief: does reasoning ("thinking") mode change gendered choices in LLM text generation?

For an independent research agent (PI request, 2026-11-07). Purpose: a targeted literature review
to inform whether the character study's thinking-mode question (Q4) can be pre-registered with a
directional prediction, or stays exploratory.

## Background (what the study does)

Chat models are asked to invent a fictional character who is the "human embodiment" of an object
(e.g. a door, or a foreign word such as Spanish "puente"), write one English paragraph, and give a
name. The outcome is the pronoun the model uses for the character (she / he / they / it), coded by
regex. The same prompts run with the model's reasoning switched on and off:
- open-weight hybrid models with visible reasoning (Qwen3: `enable_thinking` on/off);
- a closed API model with hidden reasoning (OpenAI-style `reasoning_effort` none vs medium).

## Core question

When the same model generates text with and without an explicit reasoning phase (chain of thought,
"thinking" mode, reasoning effort), **does the gender of people it invents or refers to change**,
and if so, in which direction and by what mechanism?

## Sub-questions

1. **Direct evidence:** studies comparing reasoning on vs off (or chain-of-thought vs direct
   answering) on gender-related generation: pronoun choice, name choice, occupation/role
   stereotypes, creative writing, story generation, coreference.
2. **Bias benchmarks under reasoning:** results on BBQ, WinoBias, WinoGender, StereoSet, BOLD,
   or similar when reasoning/CoT is enabled. Does reasoning reduce, increase, or not change
   stereotyped answers? Does it increase *neutral* or *hedged* answers (e.g. "they", refusals,
   "cannot be determined")?
3. **Reasoning and diversity / mode collapse:** does reasoning make sampled outputs more or less
   varied (e.g. stock names, stock pronouns)? Evidence that reasoning pushes toward a "default"
   answer.
4. **Faithfulness:** does the visible reasoning mention the gender choice when the final output
   makes one? (Our pilot question: models rarely deliberate about gender but still choose one.)
5. **Model-size or family dependence:** any evidence that the effect of reasoning on bias differs
   by model size or training recipe (distilled reasoning models vs RL-trained ones).
6. **Hidden vs visible reasoning:** anything on API models with hidden reasoning tokens vs open
   models with visible chains.

## Scope and search guidance

- Years: 2022–2026 (chain-of-thought era onward); include preprints (arXiv, ACL Anthology,
  OpenReview, FAccT, NeurIPS/ICLR workshops).
- Terms to combine: "chain-of-thought" / "reasoning" / "thinking mode" / "reasoning effort" /
  "test-time compute" / "o1" / "R1" / "Qwen3 thinking" WITH "gender bias" / "stereotype" /
  "pronoun" / "fairness" / "BBQ" / "WinoBias" / "story generation" / "creative writing" /
  "diversity" / "mode collapse".
- Include null and contradictory findings; flag where results conflict and why (task, model,
  metric).
- Exclude: work on vision-only models; bias in training data without a reasoning manipulation.

## Deliverable

1. A table: citation, models, reasoning manipulation, task/metric, direction of effect on gender
   bias (more / less / no change / more neutral), sample size, notes on quality.
2. A short synthesis: is there a consistent direction? Does it depend on task type (QA
   benchmarks vs open generation)? Any plausible mechanism?
3. **An explicit verdict**: does the literature support a directional prediction for open
   character generation (reasoning → more "she" / more "he" / more "they" / more neutral / less
   variety), or is the honest prediction "no prior expectation"? State confidence.
4. Gaps: what has not been studied.

## Note on blinding

This brief deliberately does not include the study's own pilot results, so the review is not
anchored to them. Please do not try to find them.
