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
- **Attention sink.** The first position of a sequence, where models park attention they don't
  need; its activations are huge (Qwen3-0.6B, layer 10: vector length ~6,700 vs ~37 elsewhere)
  and unrepresentative. The first token after a document separator can be one too (for some
  single-token words, one dimension jumps to ~2,500). So Phase 1+ inputs are `<|endoftext|>`, a
  newline as a buffer, then the word.
- **Massive activations.** A handful of dimensions with values orders of magnitude larger than the
  rest, tied to attention-sink positions (Sun et al. 2024, `docs/reading-list.md`). In Qwen3-1.7B,
  dimension 1793 (4B: dimension 4) jumped to ~2,500 for single-token words placed right after
  `<|endoftext|>`, which made raw difference-of-means directions track "is this word one token?".
- **Detokenization.** Models assemble a whole-word representation at the word's *last* token, mainly
  in early and middle layers (Kaplan et al. 2025). The justification for last-token readout, and
  the reason early layers may reflect token identity rather than the word.
- **Readout position.** Which token's vector represents a word: its last token (current), or a
  fixed token after it (e.g. a following newline: same token type for every word). To be chosen on
  gender-blind criteria before Phase 3 (`docs/design/phases-3-5-plan.md`).
- **`hidden_states`** *(transformers)*. What a Hugging Face model returns with
  `output_hidden_states=True`: one array per layer boundary. For Qwen3: index 0 = token
  embeddings, 1…L−1 = layer outputs, L = last layer *after* the final normalisation (different
  scale; excluded from "inner layers").
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
- **Cross-validation (k-fold).** Split the data into k parts; train on k−1, test on the held-out
  one; rotate; average. **Grouped** CV (`GroupKFold`) keeps all items of a group (both forms of a
  verb) in the same part.
- **In-sample vs out-of-sample.** Evaluated on the data a method was fit on vs on new data.
  In-sample erasure + cross-validation gives a misleading *below*-chance score (Phase 1 explainer).
- **Confidence interval (95%).** A range showing how much a result would move with a different but
  equally valid sample of items; if it excludes the "no effect" value (e.g. AUC 0.5), the result
  isn't a fluke of which items were picked.
- **Bootstrap.** Estimating that range by resampling the items you have: draw a same-size sample
  *with replacement* (some items twice, some not at all), recompute the result, repeat ~1,000 times,
  and take the middle 95% of the results. In Phase 2 each round resamples the **training** nouns
  (and refits the gender direction) *and* the **test** nouns, so the interval covers uncertainty in
  the direction itself as well as in the test. It can't correct a systematically biased item set.
- **Pre-registration.** Writing down how results will be read before running (the design doc's
  outcome tables; `decisions.md`). Analyses added after seeing results are labelled exploratory.
- **Concept erasure.** Editing activations so a concept can no longer be read out.
- **Linear guardedness.** A representation is linearly guarded against a concept if no linear
  classifier can predict the concept better than always guessing the majority class. Belrose et al.
  prove this holds exactly when the concept's classes have **equal average activation** (class
  means), which is what LEACE enforces.
- **Rank (of an erasure).** How many dimensions an edit changes. LEACE for a k-class concept has
  rank ≤ k−1: one direction for a binary concept like -o vs -a; two for the rank-2 eraser
  (verb ending and nonce ending as separate concepts).
- **Affine.** Linear plus a shift (x ↦ Ax + b). LEACE is affine: it recentres before projecting.
- **INLP** *(method)*. "Iterative nullspace projection" (Ravfogel et al. 2020): repeatedly train a
  probe and delete its direction. Used once in Phase 1 as a diagnostic; needs many more directions
  than LEACE because probe weights aren't the mean-difference direction.
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
  = half precision (2 bytes). bf16 has fp32's range but fewer significant digits (~3) than fp16
  (~3.3); fp16 overflows sooner (some models, e.g. Gemma, break in fp16). Apple M1 lacks native
  bf16, so the Mac uses fp16; RunPod (CUDA) uses bf16.
  - Qwen3 is *published and trained* in bf16, so fp32 adds no information to the weights; it only
    reduces rounding during computation, at 2× memory and much lower speed.
  - Rounding noise in Phase 0 margins: ~0.05 nat in fp16, up to a few tenths of a nat in bf16
    (bf16 steps are ~0.1 nat per token at typical logit sizes). Small next to the 1-nat
    threshold, but can flip borderline nouns. Plan: verify once by running Qwen3-4B in bf16 and
    fp32 on RunPod and comparing statuses (see `decisions.md`).
  - For probes/LEACE/cosine, activations are converted to fp32 for analysis; bf16 storage loses
    nothing beyond what the bf16 forward pass already had.
- **Quantization (4-bit, 8-bit).** Compressing weights further. Avoided: it distorts the
  activations being measured.
- **MPS / CUDA.** The GPU backends: MPS = Apple Silicon (the Mac), CUDA = NVIDIA (RunPod).
- **RunPod** *(service)*. Rented NVIDIA GPUs for the larger-model runs.
- **Global volume** *(RunPod)*. Storage not tied to one pod or data center (beta). Mounts at
  `/workspace` as GeeseFS, i.e. object storage: fast for large files, slow for many small files, so
  code, environments and the model cache stay on the pod's local disk.
- **Batched scoring.** Scoring many sentences per forward pass (padded to equal length, with an
  attention mask). Same numbers as one at a time, up to rounding; keeps the GPU busy.
- **Device rule.** For any one model, extract all activations on one device and precision; never
  compare activations across hardware (design doc). Cross-size comparisons all run on RunPod.

## Linguistics and data

- **Grammatical gender.** Noun classes marked by agreement (articles, adjectives). Arbitrary for
  inanimate nouns: *die Brücke* (f) / *el puente* (m).
- **Social gender.** Gender as attributed to people (*he/she, king/queen*).
- **Flipped pair.** A concept whose noun has opposite gender in German and Spanish
  (*Mond* m / *luna* f). **Control pair**: same gender in both.
- **Lemma / lexeme.** A *lexeme* is a word in the abstract sense: *hablar* together with *hablo*,
  *habla*, *hablaban*… The *lemma* is the conventional **citation form** chosen to stand for it:
  infinitive for Spanish verbs, singular for nouns, masculine singular for adjectives. Which form
  counts is a convention that varies by language (Latin verbs are cited by the 1sg, *amo*), and a
  lemma needn't be the form that occurs in a given text. **Lemmatising** text replaces each word
  by its lemma, so Italian *le case rosse* becomes roughly *il casa rosso*: agreement disappears,
  which is why Kann (2019) and Gonen et al. (2019) lemmatised context to strip gender cues. In this
  project's lexicon, "lemma" means the headword of a Wiktionary noun entry (the singular).
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
