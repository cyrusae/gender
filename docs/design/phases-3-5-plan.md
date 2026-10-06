# Plan for Phases 3–5 (working draft)

*Extends the design doc. Source: the methods conversation in
`docs/conversations/2026-10-06-methods.md`, triaged against the existing design. Items marked
**(new)** aren't in the design doc. Dated decisions and pre-registered hypotheses live in
`docs/decisions.md`; this file is the plan they point to. Nothing here is final until its phase
starts and the outcome tables are written.*

## Guiding principle: validate the instrument, then measure, then rule out, then corroborate

Run the **cheapest experiment that could kill the line of work first**. For Phase 5 that's the
agreement-flip gate (§5.2): if steering with the gender direction can't change grammatical
agreement, nothing else measured with that direction is interpretable.

---

## Phase 2 additions (now)

- **(new) Semantic and frequency balance of the training set.** Feminine nouns can skew abstract,
  so a direction might partly encode "abstract". The ending-matched set already excludes
  *-ción*/*-dad*, but check the masculine and feminine halves for differences in concreteness
  (WordNet category) and frequency, and report any gap. If it's large, a re-matched set or a
  covariate analysis follows.
- **(new) Selectivity control** (Hewitt & Liang 2019) for the logistic-probe direction: train the
  same probe on random but fixed per-noun labels; report real minus control. The main
  difference-of-means direction has no capacity to overfit, so this mainly guards the secondary probe.

## Before Phase 3: choose the readout position on gender-blind criteria

Phases 1–2 read each word at its **last token**. That's the standard choice (Kaplan et al. 2025:
models assemble whole words there), but the last token's identity varies (*problema* is one token,
*nube* ends in the fragment `ube`, German *Zeitung* likely in `ung`, a feminine suffix), and its
position varies with length. The alternative is a **fixed token after the word** (a following
newline): it has seen the whole word and is the same token type for every word. Because the model
reads left to right, appending it doesn't change the word's own vectors, so one forward pass records
both positions.

To avoid tuning on results, choose the primary readout **before looking at any gender result**,
using only:

| criterion | measure | better = |
|---|---|---|
| word identity preserved | can a probe tell words apart among words sharing a final token? | higher |
| independence from tokenisation | how well do token count and final-token identity predict the vector? | lower |
| clean position | sink guard; norm vs ordinary tokens | no warnings |

Then report Phase 2 at both positions as a robustness check, and use the chosen one as primary from
Phase 3 on. If the two positions disagree on gender, that's a finding to report, not to resolve by
picking one. (Mean-pooling over a word's tokens is a possible third comparison, not a primary: in a
left-to-right model the earlier tokens haven't seen the rest of the word.)

## Phase 3: German, and separate masculine/feminine vectors

- Training: masculine and feminine inanimate nouns, bare; suffix-marked nouns (*-ung*, *-heit*…)
  held out as their own test group; *der/die See*-type items as the spelling-constant test (with
  each model's lean recorded, see `decisions.md`).
- **Neuter as a reference point only** (decided in principle): masc = mean(masculine) −
  mean(neuter), fem = mean(feminine) − mean(neuter). The neuter set is built like every other:
  Wiktionary labels, inanimate, frequency-matched, and excluding word types that are neuter for
  other reasons (*-chen*/*-lein* diminutives, nominalised verbs like *das Essen*, *Ge-*
  collectives, chemical elements). Also compute the single masc–fem axis for comparison with the
  literature.
- Readouts: cosine(masc, fem) (−1 = one axis; ~0 = two features), and the **lengths** of the two
  vectors.
- **Token count is entangled with gender in German** (`docs/reports/tokenization-qwen3.md`: feminine
  nouns split into more tokens even within frequency bins, e.g. mid-frequency 3.01 vs 2.60), so the
  German training set is matched on, or adjusted for, token count, as Phase 2 adjusted for
  concreteness. Not needed for Spanish (no relationship).
- **(new) Pre-registered markedness hypothesis** (see `decisions.md`): masculine is the unmarked
  default in German and Spanish (generic masculine: *los niños*, *die Lehrer* for mixed groups),
  i.e. a *privative* opposition in Trubetzkoy's sense. Prediction: the masculine vector is shorter
  than the feminine one (masculine nouns sit nearer the neutral reference). Neuter is a third value,
  not "unmarked", so German may not mirror Spanish neatly. Spanish has no noun neuter, so its
  version needs an improvised reference (candidates: English translations, gender-invariant
  adjectives, *lo* + adjective), each anchor-dependent.

## Phase 4: shared across languages?

As in the design doc (Spanish-only, German-only and pooled vectors, tested on held-out nouns
from both languages and the flipped pairs). Additions:
- Compare **cognate vs non-cognate** pairs (`en_cognate` flag in `pairs_de_es.csv`): cognates may
  share representations more.
- Skim the final flipped pairs in English before use (~10% of auto-matched pairs are loose).

## Phase 5: does grammatical gender bleed into social gender?

### 5.0 (new) Baseline behaviour, no intervention

Does the model already show a Boroditsky-style effect? E.g. compare the log-probabilities of
stereotyped adjective sets after the same concept in German vs Spanish contexts, on the flipped
pairs and classics. This is a behavioural replication in the model, comparable to the human null
(Mickan et al. 2014), and it gives the erasure arm a concrete prediction (§5.6): erasure should
*shrink* any baseline asymmetry. Report with and without flagged classics (*luna*/*estrella* are
also women's names).

### 5.1 Geometry

Cosine similarity, layer by layer, between grammatical-gender vectors (single axis, and separate
masc/fem where a reference exists) and social-gender vectors. Baselines: random directions and
unrelated concept directions (model activations aren't centred, so "near zero" must be calibrated).

**(new) Social-gender directions from epicenes as well as English pairs.** Epicenes keep
grammatical gender fixed while the referent's sex varies: Spanish *la persona*, *la víctima*,
*el bebé*, *la criatura*, *el personaje*; German *die Person*, *das Opfer*, *das Kind*,
*der Mensch*, *die Geisel*, and *das Mädchen* (neuter noun, female referent). These separate
grammatical from social gender *within* each language, which the English-only plan can't. Exact
stimuli and contexts are to be designed, and must be checkable in English.

### 5.2 (new) Steering gate: agreement flip + random directions

Steer the noun's position with h′ = h + α·v (v = unit grammatical-gender direction; α scaled to
the typical residual-stream norm at that layer) and check that **agreement follows**:
- Spanish: gender-invariant determiner + inflecting adjective, e.g. *Compré mis mesas ___*,
  readout P(*blancas*) − P(*blancos*).
- German: predicate adjectives don't inflect, so use a different agreement target, e.g.
  pronoun reference (*Die Brücke ist alt. ___ ist …*: *er*/*sie*/*es*). To be designed.
- Same doses with random directions of the same norm (null for the agreement metric).

The dose at which agreement flips becomes the natural unit for everything after.

| | adjectives shift | adjectives don't shift |
|---|---|---|
| **agreement flips** | v is the gender feature *and* moves stereotyped associations: bleed-through evidence | v is the feature, no bleed at this layer: a clean null |
| **agreement doesn't flip** | suspicious: something in v other than grammatical gender moves the adjectives | dose too small or wrong layer; go back |

Diagnostic for the suspicious cell: if *erasing* v doesn't break agreement either, v isn't on the
agreement pathway (redundant encoding / self-repair).

### 5.3 (new) Dose-response with a damage metric

Sweep α over negative and positive values; at each dose record:
1. **Target**: log P(feminine-stereotyped adjectives) − log P(masculine-stereotyped adjectives).
2. **Damage**: KL divergence between steered and unsteered next-token distributions on neutral
   text (or loss on held-out text), steering the same positions.

Steer **only the noun's position**, not every token. The working window is the α range where
damage stays near the random-direction baseline. Patterns:
- **real effect**: target moves while damage stays flat
- **noise**: target moves only once damage spikes

The markedness hypothesis predicts that **+α (toward feminine) and −α (toward masculine) need
not be symmetric**.

### 5.4 (new) Specificity controls

- **Another grammatical direction**: singular/plural, at the same norm. If number also shifts the
  adjectives, it's "grammatical perturbation", not gender.
- **Off-target adjective pairs** matched on frequency and valence (*old/new*, *big/small*,
  *ornate/plain*): if they move too, the model just got "flowery".
- **Graded predictor**: the **Glasgow Norms** (Scott et al. 2019) give human gender-association
  ratings for ~5,500 English words. Test whether each adjective's shift scales with its rated
  gendered-ness, instead of relying on two hand-picked lists.

### 5.5 Readouts for social gender

English pronoun and occupation tasks (avoids the circularity that, in Spanish and German,
agreement on person nouns *is* social gender), plus the epicene contexts from §5.1.

### 5.6 Erasure arm (logged 2026-10-06 in `decisions.md`)

Erase the grammatical-gender subspace at every layer, then measure social-gender behaviour, with:
- a manipulation check (Phase 0 article frames must break *in the evaluation contexts*)
- random-subspace and LM-quality controls
- German m/f/n erasure (rank 2)

**(new) Convergent prediction**: erasure should shrink the §5.0 baseline asymmetry, if there is
one. Stretch goal: the reverse direction (erase social gender, test article agreement).

### Suggested order

1. Build v (Phases 2–3) with matched sets; selectivity for probes.
2. **Gate**: agreement flip + random directions (§5.2). Stop and fix v if it fails.
3. Baseline behaviour (§5.0) and dose-response with damage (§5.3).
4. Specificity (§5.4), which costs the most stimulus design.
5. Convergence: erasure (§5.6).

---

## After the Qwen RunPod session: Gemma 3 + Gemma Scope 2 (planned)

A cross-family replication with a sparse-autoencoder extension. Gemma Scope 2 has SAEs on every
layer of every Gemma 3 size (270M–27B, base and instruction-tuned); 270M and 1B are English-only, so
the German/Spanish work uses **4B, 12B and, budget allowing, 27B**.

**G0. Preconditions** (on the GPU; Gemma breaks in fp16, so not on the Mac):
- accept the Gemma licence on Hugging Face; pass the token as a pod-only secret (`HF_TOKEN`), not on
  the shared volume
- confirm the loading class for the multimodal 4B+ checkpoints gives the text model cleanly
- re-verify the input format: Gemma adds a start-of-text token; check the newline buffer keeps
  measured words off sink positions (sink guard quiet)
- map Gemma Scope 2's "residual stream after layer L" onto our `hidden_states` indexing, with a unit
  test (classic off-by-one)
- bf16 only

**G1. Frozen pipeline** on Gemma 3 4B/12B(/27B): Phase 0 known-check → Gemma shared set →
Phases 1–5 with the same pre-registered analyses. Compare *conclusions* with Qwen, never vectors.

**G2. SAE analyses** (own pre-registration, written before looking):
1. discovery on training nouns only: which SAE features separate fem/masc ending-matched nouns, per
   layer (selection out of tens of thousands of features happens on training data only)
2. the same held-out tests (exceptions, homographs, *mar*-type)
3. separate vs single features: does the SAE learn a "feminine" *and* a "masculine" feature? (the
   markedness hypothesis predicts a strong feminine feature with masculine nearer the default)
4. Phase 5 overlap: do those features fire on social-gender contexts (epicenes, *he/she*)?
5. causal: steer with a feature direction, agreement-flip gate first

Caveats to state: SAEs leave unexplained residual information; features split differently at
different widths; multilingual models may keep per-language features; Neuronpedia labels are hints.

**Compute.** SAE encoding of saved vectors is a small matrix multiply: run it on the Mac from saved
activations (download only the SAE weights for the layers used). The GPU is for running models
(extraction, steering). One chip for all sizes within a family: if 27B is included (needs 80 GB:
A100/H100), run all Gemma sizes on it; cross-family comparisons are conclusion-level, so Qwen on an
A40 and Gemma on an A100 is fine. RunPod stock on 2026-10-06 (CUDA ≥ 13.0): A40 Secure $0.49/h in
CA-MTL-1 / EU-SE-1; A100 SXM 80 GB $1.59 Secure / $1.39 community in EUR-IS-1 / US-KS-2 / US-MD-1
(no data center had both, which is why the global volume matters).

## Later / optional

- **Activation patching** (swap a vector from one run into another) to locate *where* an effect
  lives. Secondary for the main question.
- **Sparse autoencoders**: now planned via Gemma 3 + Gemma Scope 2 (section above).
- **Contextual vs bare comparison** (from the conversation): Phase 2 uses bare nouns; repeating
  with gender-invariant determiner frames (*mis X*) and comparing the two is itself a finding.
