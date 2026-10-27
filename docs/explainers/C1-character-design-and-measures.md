# I1: From pilot to main study — models, measures, and how big the study needs to be

Work from 2026-10-25 to 2026-10-27, branch `character`. Everything measured so far is on the six
burned pilot nouns (pot, keyboard, tree, door, flower, hammer): **pilot results, never findings.**
Decisions are in `docs/decisions.md` ([instruct] entries); the draft pre-registration is
`character/docs/prereg_DRAFT.md`.

## 1. Choosing models: the "fishing" round and the balance rule

**What it measures.** Before the main study, each candidate chat model wrote 20 paragraphs per
pilot noun under two name instructions. We kept models whose characters are not almost all one
gender.

**Why.** A model that writes *he* 95% of the time has no room to show a shift toward *he*, and
little room to show one toward *she*. Picking balanced models is a measurement choice (like picking
a scale that is not stuck at its maximum), and the rule was written before the results came in.

**The rule (v2).** Among *he* and *she* paragraphs, *she* must be 25–75% in both conditions. A
model that mostly writes *they* fails a separate 30% "gendered floor" and is kept as a **case of
interest** rather than discarded: those models answer different questions.

**Pilot picture.** Passed: Qwen3-14B (thinking off), Qwen3-8B (thinking on), gpt-oss-120b,
GLM-5.3-Flash, Qwen3.8 models, DeepSeek-V4.1-Flash, Mistral Small, GPT-6-Luna. Failed by lean:
Gemma 3 and 4 (*he*, ~90% for Gemma 4), Llama-3.3-70B (*she*, ~99%). Mostly *they*: EuroLLM-22B,
Qwen3-14B with thinking on.

**Limit worth stating.** Selecting balanced models means results describe balanced models; the
lopsided ones (which may be the ones people use) need their own reporting.

## 2. Thinking mode (Qwen3, Gemma 4)

Some models write private reasoning before answering. In the pilot:
- Thinking moved Qwen3-8B toward *she*, Qwen3-14B toward *they*, and did nothing for Gemma 4.
- Reasoning almost never *discusses* gender (at most 0.5% of reasoning blocks), yet it still
  *chooses* one: Gemma 4 drafts the paragraph inside its reasoning, and the gender is already in
  the first draft.

**How to read it.** "The model never deliberated" is not "the model had no gender in mind": the
choice is made in drafting, not debated. A later measure could track where in the reasoning the
gender first appears.

## 3. Two generation routes are two instruments

Gemma-3-12B run through an API provider and on our own GPU, with the same prompt, precision and
requested sampling, agreed in one condition and disagreed in another (more *they* on our GPU,
p = 0.001). Cause unknown (providers may ignore some sampling settings). **Rule:** each model's
outputs within one analysis come from one route. This extends the base study's
"one device and precision per model" rule from activations to generated text.

## 4. Content measures: what the description says, not just the pronoun

For prompts where the pronoun is fixed (Casey/Riley), the outcome is the *description*. Two kinds
of score, both computed on paragraphs with the character's pronouns and name masked:

- **Glasgow word average (model-free).** The mean human gender rating (1 = feminine, 7 =
  masculine) of the words in the paragraph that Glasgow rates. Transparent; the limit is coverage
  (median 29 rated words per paragraph).
- **Embedding score (model-based).** Each paragraph is turned into a vector by an embedding model;
  score = closeness to 50 feminine-rated "anchor" words minus closeness to 50 masculine-rated ones.
  Six embedders of four kinds: retrieval-trained encoders (bge-m3, multilingual-e5), a
  translation-trained encoder (LaBSE), LLM-based embedders (Qwen3-Embedding-0.6B,
  EmbeddingGemma), and static word vectors (fastText, the classic word-embedding-bias tool).
  - **Anchors:** primary = stereotyped *things and qualities* (lipstick, lace, elegant / engine,
    shotgun, aggressive): Glasgow's extremes after removing person, body and animal words and every
    study noun. Secondary = all words (dominated by woman/beard-type words).
  - **Shuffle null:** the 100 anchors are split at random 200 times; this shows how big a score
    an arbitrary split gives, the same noise check as the base study's.
  - **Same-family rule:** a Qwen embedder scoring Qwen-written text may share Qwen's associations,
    so the main result uses embedders from a different family than the writer.

**Masking choice.** Only the character's own pronoun class is masked (PI's argument: plural *they*
in "absorbing their grief" carries no gender). The cost is that *they* paragraphs lose their
plural-*they* content and *he/she* paragraphs keep it. A check fixed in advance measured that cost:
masking plural *they* moves scores by at most 0.05 of the null spread, so the asymmetry is
negligible. Mask-every-*they* is kept as a sensitivity version.

**Pilot reading.** All six embedders and the Glasgow average agree with each other (r 0.66–0.95).
In the writer prompt, paragraphs the model wrote as *she* read more feminine than *he* ones on
every measure (the measure tracks something real). With the pronoun assigned (Casey/Riley),
*she* characters read slightly more feminine than *he* ones (about 0.4–0.9 of a null SD).

**Limits.** These are model-based judgments of outputs, never the gender outcome itself;
embedding scores depend on the anchor list; Glasgow ratings are British students' associations.

## 5. Overridden *they*

Given "They are the human embodiment of a hammer", gpt-oss-120b wrote *he* in 20 of 48 paragraphs
and Qwen3.8-Flash switched 12 of 48 (toward *he* for hammer/pot/door, *she* for flower); GLM and
Luna never did. The automatic "kept pronoun" check turns this into a measure in its own right.

## 6. How big the study needs to be: the power simulation

**What it does.** It invents many fake datasets that look like the pilot (real noun genders,
realistic baselines and noise) with a known true effect, analyses each one, and counts how often the
analysis finds the effect ("power"). It also runs with no true effect, to check the false-alarm
rate stays near 5%.

**The effect (SESOI).** The smallest effect worth caring about: feminine nouns' shift minus
masculine nouns' shift = 5 points pooled (each side moves 2.5 toward its own gender), 10 per
language. The stricter of two readings (PI).

**The key unknown: τ (tau).** How much nouns differ in how far they shift. With six pilot nouns,
τ came out anywhere from 0 to 0.23.

**Results (191 nouns).** If τ ≤ 0.15, 20–30 draws per arm give ≥ 86–91% power for the pooled
5-point test. Above τ ≈ 0.2, no number of draws reaches 80%: the study runs out of *nouns*, not
draws. Per-language 10-point tests only work if τ ≲ 0.10, so they are pre-registered as estimates
with confidence intervals.

**Stopping rule adopted.** Run 10 draws, estimate τ on the real nouns, then add draws until the
pooled confidence interval is ±2.5 points or the cap of 30 is reached. The rule never looks at
which way the effect goes, so it is not "fishing" (optional stopping on p-values is).

**Limits.** The simulation uses a simpler analysis than the planned mixed model, and assumes the
pilot's noise structure; ceiling effects (nouns already near 0% or 100% *she*) shrink observed
effects (a true 5 came out ≈ 4.1), and strongly lopsided models like Luna (88% *she* baseline)
will have less room to move.

## 7. Cost note for the write-up

Reasoning models are expensive at scale because they bill hidden reasoning tokens: at the 30-draw
cap, GLM-5.3-Flash ~$123 and Qwen3.8-Flash ~$73 per model, against ~$7 for GPT-6-Luna and ~$7–8
for Qwen3-14B on a rented GPU. Main run: GPT-6-Luna (API) + Qwen3-14B (pod); gpt-oss-120b
deferred.
