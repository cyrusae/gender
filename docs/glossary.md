# Glossary

Terms as they're used in this project. Grouped by area; *(tool)* marks a piece of
software, and says which one. Phase explainers live in [`explainers/`](explainers/).

## Models and text

- **LLM / language model.** A neural network trained to predict the next token of text. Every
  model here is a **transformer**.
- **Base vs instruct model.** A *base* (pretrained) model only predicts text; an *instruct*/chat
  model has been further trained to follow instructions. We use base models because we measure
  raw next-token probabilities. (Qwen names them `…-Base`.)
- **Parameters (0.6B, 4B, 14B).** The number of learned weights, in billions. A rough measure of
  model size; memory needed ≈ 2 bytes per parameter in half precision.
- **Token / tokenizer.** Models read text as tokens: whole words or word pieces
  (`Schlüssel` → `Schl|üss|el`). The *tokenizer* does the splitting. All Qwen3 sizes share one
  tokenizer, so they see identical tokens for the same text.
- **BOS token.** A "beginning of sequence" marker some tokenizers add. Qwen3 doesn't, so the first
  real word has no context. That's why every sentence frame has a lead-in.
- **Model family.** Models from one maker sharing an architecture and training recipe at different
  sizes (Qwen3 0.6B…14B).
- **Hugging Face (HF)** *(platform)*. Where open models are hosted; `transformers` is its Python
  library for loading them. **Gated** models (Llama, Gemma) require accepting terms on the website
  and logging in (`hf auth login`).
- **safetensors** *(file format)*. How model weights are stored on disk.

## Probabilities and scoring

- **Logits.** The model's raw scores for every possible next token, before conversion to
  probabilities.
- **Softmax.** The function that turns logits into probabilities that sum to 1.
- **Log-probability.** log P. Probabilities of text are tiny products; logs turn products into sums.
  Always ≤ 0; closer to 0 = more likely.
- **Nat.** The unit of natural-log quantities. A difference of 1 nat = a factor of e ≈ 2.7 in
  probability; 2.3 nats = 10×; 4.6 nats = 100×.
- **Margin.** In Phase 0: log P(noun… | masculine article) − log P(noun… | feminine article).
  Sign = which gender the model prefers; size = how strongly.
- **Balanced accuracy.** Average of per-class accuracies (masculine, feminine). Immune to a model
  that always guesses one class.
- **Zipf frequency** *(from the `wordfreq` library)*. log₁₀ of how often a word occurs per billion
  words. 3 = once per million words; 6 = once per thousand. Frequency bins: low 2.5–3.5,
  mid 3.5–4.5, high ≥ 4.5.

## Inside the model (interpretability)

- **Layer.** Transformers are stacks of identical blocks (Qwen3-0.6B: 28; Qwen3-14B: 40). Each refines the
  representation of every token.
- **Residual stream.** The running vector for each token that every layer reads from and adds to.
  "The representation of *Brücke* at layer 12" means the residual-stream vector at that token after
  layer 12.
- **Activation.** The numbers a model computes internally for an input, here residual-stream
  vectors. "Extracting activations" = recording them to disk.
- **Hidden size / dimension (d_model).** Length of each residual-stream vector (Qwen3-0.6B:
  1,024; Qwen3-14B: 5,120). A "direction" is a vector in this space.
- **Hook.** A function attached to a layer that reads (or edits) activations as the model runs.
- **TransformerLens (TL)** *(tool)*. Python library that wraps a model with named hook points
  for reading and editing activations. Version 4.0 (Sept 2026) replaced its classic
  `HookedTransformer` class with `TransformerBridge`, which wraps the Hugging Face model
  (per [model_candidates.md](model_candidates.md)).
- **nnsight** *(tool)*. An alternative to TransformerLens: works on almost any Hugging Face model
  directly, using a `with model.trace(...)` block to read/write activations. We'll pick one of the
  two when Phase 1 needs hooks.
- **Last-token position.** For a multi-token word, the activation at its *final* token is used,
  since that's where the model has seen the whole word (design doc rule).
- **Shared / middle layers.** Layers where representations are thought to be most
  language-independent. Which ones count is decided from Phase 4 data, not in advance.

## Methods

- **Linear probe.** A simple classifier (here **logistic regression**, from **scikit-learn**
  *(tool)*) trained to predict a label (e.g. masculine/feminine) from activations. High accuracy
  = the information is linearly readable at that layer. "Not found" only means not found
  *linearly*.
- **Difference of class means.** The simplest "gender direction": average activation of
  masculine nouns minus average of feminine nouns.
- **Train/test split by stem.** Words sharing a stem (*brelda*/*brelpo*) must all be in train or
  all in test, or the probe can memorise the stem instead of learning the concept.
- **Concept erasure.** Editing activations so a concept can no longer be read out.
- **LEACE** *(method; `concept-erasure` package by EleutherAI)*. "LEAst-squares Concept Erasure":
  the smallest linear edit that makes a concept undetectable by *any* linear classifier.
  Phase 1 uses it to erase spelling (-a/-o) so a gender probe can't cheat on word endings.
- **Steering.** Adding a direction to activations during a forward pass and seeing how outputs
  change (Phase 5: does adding the grammatical-gender direction shift "The bridge was ___"
  toward *elegant* or *sturdy*?).
- **Cosine similarity.** How aligned two directions are: 1 = same direction, 0 = unrelated,
  −1 = opposite. Needs a baseline (random/unrelated directions), because high-dimensional model
  activations are often not centred around zero.
- **Layer sweep.** Running the same probe/measurement at every layer and plotting the result.
- **Random-vector baseline.** Steering with a random direction of the same size, to show an effect
  isn't just from perturbing the model.

## Hardware and precision

- **fp16 / bf16 / fp32.** Number formats. fp32 = full precision (4 bytes/parameter); fp16 and bf16
  = half precision (2 bytes). bf16 has fp32's range but less precision; fp16 has more precision
  but overflows sooner (some models, e.g. Gemma, break in fp16). Apple M1 lacks native bf16, so
  the Mac uses fp16; RunPod (CUDA) uses bf16.
- **Quantization (4-bit, 8-bit).** Compressing weights further. Avoided: it distorts the
  activations being measured.
- **MPS / CUDA.** The GPU backends: MPS = Apple Silicon (the Mac), CUDA = NVIDIA (RunPod).
- **RunPod** *(service)*. Rented NVIDIA GPUs for the larger-model runs.
- **Device rule.** For any one model, extract all activations on one device and precision; never
  compare activations across hardware (design doc). Cross-size comparisons all run on RunPod.

## Linguistics and data

- **Grammatical gender.** Noun classes marked by agreement (articles, adjectives). Arbitrary for
  inanimate nouns: *die Brücke* (f) / *el puente* (m).
- **Social gender.** Gender as attributed to people (*he/she, king/queen*).
- **Flipped pair.** A concept whose noun has opposite gender in German and Spanish
  (*Mond* m / *luna* f). **Control pair**: same gender in both.
- **Lemma.** Dictionary form of a word (*Brücke*, not *Brücken*).
- **Homograph.** Same spelling, different word (*camino* "path" / *camino* "I walk").
- **Agreement / concord.** Words changing form to match a noun's gender (*la llave bonita*).
- **Case (German).** Nominative/accusative/dative/genitive; articles change by case
  (*der* = masculine nominative but also feminine dative).
- **Animate / inanimate.** People and animals vs things. The study uses inanimate nouns only, so
  grammatical gender isn't confounded with real-world sex.
- **Wiktionary / kaikki.org** *(data source)*. The free dictionary; kaikki.org publishes
  machine-readable extracts of it. Source of all gold gender labels.
- **WordNet** *(data source, via NLTK)*. English lexical database; its category labels
  (`noun.person`, `noun.artifact`) help judge animacy and concreteness from English glosses.
- **Nonce word.** An invented but plausible word (*brelda*), used in Phase 1.

## Project-specific terms

- **Frame.** A sentence template with slots for article and noun (Phase 0).
- **known / wrong / unsure / conflict.** Phase 0 noun statuses (see the
  [Phase 0 explainer](explainers/00-phase0-behavioural-check.md)).
- **Shared set.** Nouns known by every model in a comparison; the default basis for cross-size
  analyses.
- **Eligible.** A lexicon noun that passes every automatic filter for stimuli.
- **Strict pair.** An auto-matched translation pair that needs no checking: dominant gloss match,
  similar frequency, not a cognate.
- **Classic.** One of the 19 hand-picked flipped pairs from the literature
  (`data/stimuli/classics_spec.csv`).
- **multi_type.** For nouns with two genders: `meaning_split` (*der See* lake / *die See* sea) vs
  `free_variation` (*el/la mar*).
- **Outcome table.** The design doc's per-phase table of possible results → meaning → next step,
  written before running.
