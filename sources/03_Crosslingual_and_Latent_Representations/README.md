# Cross-Lingual and Latent Representations: Reading Guide & Triage

This directory contains 7 foundational papers exploring how multilingual Large Language Models (LLMs) organize linguistic and conceptual information across languages. The central questions uniting this collection are:
1. **The Shared Geometry Question:** Do multilingual models maintain siloed, language-specific representations, an "English-pivot" hub, or truly language-agnostic conceptual spaces?
2. **The Latent Transfer Question:** When a feature (such as grammatical gender or syntactic agreement) is represented in one language (e.g., Spanish or German), does that representation bleed into, align with, or modulate representations in another language (e.g., English or Russian)?
3. **The Layerwise Processing Question:** At what model depth do tokens combine into words, words translate into abstract concepts, and abstract concepts project back into target-language tokens?

These papers provide the theoretical scaffolding and empirical justification for the **Grammatical Gender Bleedthrough** project, specifically **Phase 4 (Cross-Lingual Direction Alignment & Flipped Pairs)** and **Phase 5 (Activation Steering Across Languages)**.

---

## 1. Executive Triage & Priority Matrix

| Paper | Focus & Core Mechanism | Project Role | Reading Priority |
|---|---|---|:---:|
| **[Wendler et al. 2024]**<br>*(Do Llamas Work in English?)* | Logit lens tracking of non-English queries; three-phase processing (input $\to$ English concept space $\to$ target output). | **Tier 1 Primary Anchor**<br>Provides the theoretical framework for cross-lingual bleedthrough into English. | **Must-Read First** |
| **[Dumas et al. 2024]**<br>*(Separating Tongue from Thought)* | Activation patching during translation; concept representation is dissociable from and follows language encoding. | **Tier 1 Primary Anchor**<br>Justifies language-agnostic concept vectors and middle-layer concept extraction. | **Must-Read First** |
| **[Kaplan et al. 2024]**<br>*(From Tokens to Words: Inner Lexicon)* | Intrinsic detokenization in early/middle layers; whole-word semantics consolidate at the **final subword token**. | **Tier 2 Structural Foundation**<br>Validates extraction and steering at the LAST token position rather than AFTER or pooled tokens. | **High Priority** |
| **[Huang et al. 2025]**<br>*(Platonic Representations)* | Concept steering vectors transfer across distinct LLM families via simple linear transformations. | **Tier 2 Architectural Transfer**<br>Supports cross-model generalization (Qwen3 vs. LLaMA vs. EuroLLM). | **High Priority** |
| **[Pokharel et al. 2026]**<br>*(Cross-Lingual Activation Steering)* | Steering non-dominant languages via activation offsets without degrading English capabilities; functional divergence. | **Tier 2 Methodological Analog**<br>Empirical precedent for cross-lingual vector arithmetic and steering gates. | **High Priority** |
| **[Kirtane & Huang 2026]**<br>*(Language Steering for Multilingual ICL)* | Mean difference vectors isolate language identity and steer cross-lingual transfer in context. | **Tier 3 Contextual Extension**<br>Validates difference-of-means vector extraction across language pairs. | **Medium Priority** |
| **[Flint et al. 2026]**<br>*(Failure of Contextual Invariance)* | LLMs fail to maintain contextual invariance under surrounding prompt variations and discourse embeddings. | **Tier 3 Critical Control / Risk**<br>Mandates minimal, tightly matched prompt templates (W1–W4, nonce frames). | **Medium Priority** |

---

## 2. Thematic Overlap & Cluster Map

```
                     ┌─────────────────────────────────────────────────────────┐
                     │          THE CROSS-LINGUAL LATENT LANDSCAPE            │
                     └─────────────────────────────────────────────────────────┘
                                                  │
         ┌────────────────────────────────────────┴────────────────────────────────────────┐
         ▼                                                                                 ▼
┌─────────────────────────────────┐                               ┌─────────────────────────────────┐
│     Macro Representation        │                               │     Micro Execution & Latent     │
│  "How are languages organized?" │                               │   "Where & how to intervene?"   │
└─────────────────────────────────┘                               └─────────────────────────────────┘
         │                                                                                 │
         ├─────────────────────────────────┐                                               ├─────────────────────────────────┐
         ▼                                 ▼                                               ▼                                 ▼
【The English Pivot】             【Language-Agnostic Concept】                    【Intrinsic Detokenization】      【Cross-Lingual Steering】
Wendler et al. (2024)             Dumas et al. (2024)                             Kaplan et al. (2024)              Pokharel et al. (2026)
• 3-stage pipeline                • Concept vs. Tongue disentanglement            • BPE merged into whole words     • CLAS selective offsets
• Middle layers decode to EN      • Mean concept patching improves translation    • Peak semantics at LAST token    • Functional divergence
• English as latent switchboard   • Language identity set before concept          • Validates probe token index     Kirtane & Huang (2026)
         │                                 │                                               │                        • Difference-of-means vectors
         └────────────────┬────────────────┘                                               │                                 │
                          ▼                                                                ▼                                 ▼
            ┌───────────────────────────┐                                    ┌───────────────────────────┐     ┌───────────────────────────┐
            │   Convergence / Platonic  │                                    │   Contextual Fragility    │     │   Project Application:    │
            │    Huang et al. (2025)    │                                    │    Flint et al. (2026)    │     │ Phase 4 (Shared Directions)│
            │ Cross-model linear maps   │                                    │ Context shifts break      │     │ Phase 5 (Bleed Steering)  │
            │ Universal latent geometry │                                    │ semantic invariance       │     │ LAST token, dose window   │
            └───────────────────────────┘                                    └───────────────────────────┘     └───────────────────────────┘
```

### 2.1 The "English Pivot" Hypothesis vs. Language-Agnostic Concept Spaces
A fundamental debate within mechanistic interpretability is whether multilingual LLMs translate through an English internal lingua franca or maintain abstract, language-neutral conceptual hubs:
* **The English Pivot Model (Wendler et al. 2024):** Multilingual models trained on heavily skewed corpora (e.g., LLaMA, Qwen) project non-English inputs into an English-dominated intermediate space in middle layers. When applying logit lenses at layers 40–70%, the model predicts the English translation of a non-English query *before* translating back into the target surface language.
* **The Language-Agnostic Space Model (Dumas et al. 2024):** Language identity and conceptual semantics are orthogonal. Activation patching demonstrates that the model chooses the output language in earlier layers than the actual conceptual content. Furthermore, substituting a concept representation with the *mean activation vector across four languages* preserves or improves downstream translation, proving that the deepest latent representations are genuinely shared and language-free.
* **Reconciliation for This Project:** These two perspectives are complementary rather than contradictory. The conceptual space is geometric and language-agnostic in structure (as Dumas et al. show), but because English represents the overwhelming majority of pre-training tokens, the geometry of that concept space is heavily aligned with English lexical semantic distributions (as Wendler et al. show). This explains precisely why injecting a grammatical gender feature into a Spanish concept (e.g., *el puente*) can project into English adjectival evaluations (e.g., *sturdy* vs. *elegant*).

### 2.2 Cross-Lingual Activation Transferability
* **Platonic Convergence (Huang et al. 2025):** The geometry of concept representations converges across different LLM architectures and families. Simple linear maps bridge steering vectors across disparate models, confirming that representation spaces reflect underlying conceptual invariants rather than idiosyncratic training noise.
* **Functional Divergence in Steering (Pokharel et al. 2026; Kirtane & Huang 2026):** Cross-lingual transfer does not require 100% cosine alignment across languages. Pokharel et al. prove that effective steering operates through *functional divergence*: modulating shared latent subspaces unlocks capabilities in non-dominant languages while keeping dominant (English) pathways stable. Kirtane & Huang show that simple activation offsets (difference-of-means) effectively steer task execution across languages in in-context learning.

---

## 3. Rhetorical Ammunition & Citation Guide

When drafting papers, pre-registrations, or rebuttal letters for Phase 4 and Phase 5, cite these papers to substantiate core architectural and methodological choices:

### 3.1 Justifying Why Spanish/German Gender Might Bleed into English Generations
* **The Reviewer Objection:** *"Why should grammatical gender—a purely formal morphological feature in Spanish or German—have any effect on English generations or adjectives, given that English lacks grammatical gender for inanimate objects?"*
* **The Counter-Argument / Citation:**
  > Citing **Wendler et al. (2024)** and **Dumas et al. (2024)**, we establish that multilingual autoregressive transformers do not process non-English inputs in isolated language silos. Rather, information passes through an intermediate, shared conceptual representation in middle layers (Dumas et al., 2024) that is strongly aligned with English distributional semantics (Wendler et al., 2024). If an inanimate noun's representation in Spanish (*el puente*) or German (*die Brücke*) carries grammatical gender features into this shared bottleneck, those features can interact with latent semantic associations before surface detokenization occurs.

### 3.2 Explaining Layerwise Dynamics: Why Probing and Steering Succeed at 40%–70% Depth
* **The Reviewer Objection:** *"Your probe accuracy and steering effects peak in the middle layers (layers 9–25 in 8B, layers 10–22 in 14B) and collapse in early or late layers. Does this indicate that your findings are an artifact of layer choice?"*
* **The Counter-Argument / Citation:**
  > This layerwise trajectory is directly predicted by the three-phase processing model of **Wendler et al. (2024)** and the activation patching timeline of **Dumas et al. (2024)**. In early layers (0–25%), representations operate in "input space", dominated by language-specific surface orthography and BPE tokenization artifacts. In middle layers (40–70%), representations transition into language-agnostic "concept space", where cross-lingual concept transfer and shared semantic directions emerge. In late layers (75–100%), representations shift into "output space" to resolve exact next-token surface forms in the target language. Intervening at middle depths targets the representations when they are most conceptually malleable and cross-lingually active.

### 3.3 Justifying Probe and Steering Extraction at the LAST Token Position
* **The Reviewer Objection:** *"Why extract noun representations and apply steering vectors exclusively at the final subword token of the noun, rather than pooling across all tokens or probing the AFTER token?"*
* **The Counter-Argument / Citation:**
  > We follow the findings of **Kaplan et al. (2024)**, who demonstrate that modern LLMs perform intrinsic detokenization in early-to-middle layers, synthesizing fragmented subword tokens into unified word-level semantic representations at the **last token** of the word sequence. Probing at the LAST token captures this consolidated lexical concept. Conversely, activations at the AFTER position are already heavily conditioned by attention heads anticipating subsequent syntactic agreement or punctuation, diluting the localized noun representation.

### 3.4 Defending Rigorous Lexical and Context Controls Against Invariance Failures
* **The Reviewer Objection:** *"Could the observed effects simply be driven by arbitrary variations in sentence framing or prompt syntax?"*
* **The Counter-Argument / Citation:**
  > As demonstrated by **Flint et al. (2026)**, LLM representations frequently suffer from a failure of contextual invariance, where extraneous discourse tokens alter latent representations. To prevent contextual artifacts, our Phase 5 evaluation relies on tightly controlled minimal frames (e.g., *Mi {noun} es muy ___* and *The {noun} is very ___*), systematic averaging across pre-registered wording variations (W1–W4), explicit nonce-noun controls (P29/P30), and residualization of lexical covariates (frequency, length, concreteness) as established in psycholinguistic literature.

---

## 4. Annotated Inventory of Papers

### 1. Wendler, Veselovsky, Monea & West (2024)
* **Title:** *Do Llamas Work in English? On the Latent Language of Multilingual Transformers*
* **ArXiv / Venue:** arXiv:2402.10588v4 [cs.CL], June 2024.
* **File:** [`[Wendler et al. 2024] Do Llamas Work in English.pdf`]([Wendler%20et%20al.%202024]%20Do%20Llamas%20Work%20in%20English.pdf)
* **Methodology:**
  * Analyzes intermediate representations of LLaMA-2 models (7B, 13B, 70B) on non-English prompts (Chinese, French, German, Russian, etc.) designed with unique, deterministic single-token continuations.
  * Projects intermediate hidden states into vocabulary logit space using the model's unembedding matrix (logit lens / tuned lens).
* **Key Findings:**
  * Identifies three distinct layerwise stages:
    1. *Input space* (early layers): Embeddings are distant from the output token vocabulary; dominated by surface language.
    2. *Concept space* (middle layers, ~40–75% depth): The logit lens can decode semantically correct next-token continuations, but assigns significantly higher probability to the **English translation** of the correct answer than to the prompt's native language.
    3. *Output space* (late layers, >80% depth): The latent representation moves into target-language-specific vocabulary regions to generate the correct surface token.
* **Limitations:**
  * Focuses primarily on LLaMA-2 architectures; studied primarily on single-token factual generation prompts.
* **Project Relevance:**
  * **Primary Theoretical Pillar.** Directly explains why a Spanish or German gender feature could bleed into English adjective evaluation: middle layers map concepts through an English-dominated conceptual bottleneck.

---

### 2. Dumas, Wendler, Veselovsky, Monea & West (2024)
* **Title:** *Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers*
* **ArXiv / Venue:** arXiv:2411.08745v4 [cs.CL], June 2025.
* **File:** [`[Dumas et al. 2024] Separating Tongue from Thought.pdf`]([Dumas%20et%20al.%202024]%20Separating%20Tongue%20from%20Thought.pdf)
* **Methodology:**
  * Causal activation patching on word-translation tasks across multilingual transformers (LLaMA-2/3, Mistral, Gemma).
  * Systematically patches latents from source translation prompts into target translation prompts, separating the encoding of the *target language* from the *concept* to be translated.
  * Patches activations with the *mean concept vector* averaged across multiple languages.
* **Key Findings:**
  * Language identity is determined earlier in the network than conceptual content.
  * Activation patching can alter the concept while keeping output language fixed, and alter output language while keeping the concept fixed.
  * Patching intermediate layers with the cross-lingual *mean concept representation* does not harm translation accuracy, but frequently improves it, demonstrating that transformers construct a unified, language-agnostic conceptual representation.
* **Limitations:**
  * Evaluated primarily on translation and concept generation; does not directly evaluate grammatical morphology or gender.
* **Project Relevance:**
  * **Core Justification for Phase 4.** Supports the hypothesis that cross-lingual transfer (T1/T3) operates on a shared concept subspace, providing the rationale for testing whether grammatical gender aligns across typologically distinct languages.

---

### 3. Kaplan, Oren, Reif & Schwartz (2024/2025)
* **Title:** *From Tokens to Words: On the Inner Lexicon of LLMs*
* **ArXiv / Venue:** ICLR 2025 / arXiv:2410.05864v4 [cs.CL], March 2025.
* **File:** [`[Kaplan et al. 2024] From Tokens to Words - Inner Lexicon of LLMs.pdf`]([Kaplan%20et%20al.%202024]%20From%20Tokens%20to%20Words%20-%20Inner%20Lexicon%20of%20LLMs.pdf)
* **Methodology:**
  * Investigates how subword tokenization (BPE) interacts with lexical semantics across diverse LLMs (Llama-3, Mistral, Gemma).
  * Probes intermediate states of words split into arbitrary subword tokens, typos, and out-of-vocabulary (OOV) compositions.
  * Evaluates latent word representations by feeding intermediate vectors back into the network.
* **Key Findings:**
  * LLMs perform *intrinsic detokenization*: subword sequences are integrated into coherent whole-word representations at their **final token**.
  * This integration occurs primarily in early-to-middle layers (layers 25%–50%).
  * The representation at the last token is robust to arbitrary token splits and typos, acting as an internal "lexicon" that represents the whole word meaning.
* **Limitations:**
  * Focuses on lexical understanding in English; does not address gender inflection across morphologically rich languages.
* **Project Relevance:**
  * **Critical Methodological Foundation.** Confirms our protocol of extracting activations and injecting steering vectors at the **LAST subword token** of each noun, explaining why probes at the AFTER position exhibit severe degradation.

---

### 4. Huang, Huang, Feng, Lei & Lv (2025)
* **Title:** *Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts*
* **ArXiv / Venue:** arXiv:2501.02009v2 [cs.CL], May 2025.
* **File:** [`[Huang et al. 2025] Cross-Model Transferability on Platonic Representations.pdf`]([Huang%20et%20al.%202025]%20Cross-Model%20Transferability%20on%20Platonic%20Representations.pdf)
* **Methodology:**
  * Evaluates whether concept steering vectors (SVs) extracted from one LLM family can be transferred to steer behavior in completely different LLM families (e.g., LLaMA, Vicuna, Mistral, Gemma).
  * Formalizes the alignment between model activation spaces using linear transformations (orthogonal Procrustes and affine mappings), motivated by the "Platonic Representation Hypothesis".
* **Key Findings:**
  * Concept representations across distinct LLM architectures share fundamental geometric structures.
  * Simple linear transformations allow steering vectors extracted from model $A$ to steer generation in model $B$ with high efficacy.
  * Concept representations become more linearly transferable in the middle and deeper layers of the networks.
* **Limitations:**
  * Cross-model mappings require a small set of anchor concepts to fit the linear projection; does not evaluate cross-lingual transfer within a single model.
* **Project Relevance:**
  * Supports the cross-architecture validity of our findings across Qwen3-1.7B, 4B, 8B, 14B, and EuroLLM, proving that learned gender and semantic directions reflect shared geometric manifolds rather than idiosyncratic single-model artifacts.

---

### 5. Pokharel, Agrawal & Nagar (2026)
* **Title:** *Cross-Lingual Activation Steering for Multilingual Language Models*
* **ArXiv / Venue:** arXiv:2601.16390v2 [cs.CL], October 2026.
* **File:** [`[Pokharel et al. 2026] Cross-Lingual Activation Steering for Multilingual LMs.pdf`]([Pokharel%20et%20al.%202026]%20Cross-Lingual%20Activation%20Steering%20for%20Multilingual%20LMs.pdf)
* **Methodology:**
  * Introduces Cross-Lingual Activation Steering (CLAS), a training-free inference-time intervention that selectively modulates neuron activations to improve performance on non-dominant languages.
  * Analyzes the separation and overlap between language-specific and language-shared neural subspaces across transformer layers.
* **Key Findings:**
  * Multilingual models suffer from representational imbalance between dominant and non-dominant languages in shared attention/MLP layers.
  * Effective cross-lingual transfer does not require forcing identical alignment; it operates via *functional divergence*, where steering shared directions unlocks latent capacity without disrupting high-resource (English) capabilities.
  * Performance improvements correlate with structured separation in intermediate activation clusters.
* **Limitations:**
  * Targets task accuracy and language generation benchmarks (XNLI, TyDiQA) rather than sociolinguistic or grammatical bias.
* **Project Relevance:**
  * Validates activation steering as a viable mechanism for modulating cross-lingual behaviors and demonstrates that latent steering in one language produces measurable downstream effects across languages.

---

### 6. Kirtane & Huang (2026)
* **Title:** *Language Steering for Multilingual In-Context Learning*
* **ArXiv / Venue:** arXiv:2602.02326v3 [cs.CL], October 2026.
* **File:** [`[Kirtane & Huang 2026] Language Steering for Multilingual In-Context Learning.pdf`]([Kirtane%20&%20Huang%202026]%20Language%20Steering%20for%20Multilingual%20In-Context%20Learning.pdf)
* **Methodology:**
  * Investigates cross-lingual in-context learning (ICL) gaps where prompts/demonstrations are in English but queries are in target non-English languages.
  * Computes language steering vectors using mean activation differences between parallel cross-lingual sentences and adds them during inference.
* **Key Findings:**
  * Language identity is encoded as clear, separable linear directions within intermediate transformer activations.
  * Adding a target-language offset vector during the forward pass steers the model to execute reasoning and formatting in the target language, closing cross-lingual performance gaps.
  * Demonstrates that mean-difference vector extraction is sufficient to capture language-level control features.
* **Limitations:**
  * Evaluated primarily on classification and reasoning tasks; does not probe fine-grained lexical or morphological features.
* **Project Relevance:**
  * Provides direct methodological precedent for extracting gender and language directions via difference-of-means vectors, reinforcing the validity of the Phase 4 and Phase 5 vector computation pipeline.

---

### 7. Flint, Kumar, Aiello & Baronchelli (2026)
* **Title:** *Failure of Contextual Invariance in Large Language Models*
* **ArXiv / Venue:** arXiv:2603.23485v2 [cs.CL], June 2026.
* **File:** [`[Flint et al. 2026] Failure of Contextual Invariance in Multilingual Representations.pdf`]([Flint%20et%20al.%202026]%20Failure%20of%20Contextual%20Invariance%20in%20Multilingual%20Representations.pdf)
* **Methodology:**
  * Tests the implicit assumption that LLM semantic representations remain stable under contextually equivalent prompt variations, conversational histories, and discourse embeddings.
  * Systematically varies surrounding context while holding task-relevant information constant across multiple model families and languages.
* **Key Findings:**
  * LLMs consistently fail the assumption of contextual invariance: introducing semantically irrelevant surrounding discourse triggers substantial drift in internal representations and output distributions.
  * Multilingual prompts are particularly vulnerable, as small changes in phrasing alter attention allocation across language-specific subspaces.
* **Limitations:**
  * Diagnostic and critical paper; does not propose an internal architecture to solve contextual instability.
* **Project Relevance:**
  * **Critical Methodological Safeguard.** Warns against unconstrained prompt formulations. Directly motivates our strict experimental controls: evaluating multiple balanced wordings (W1–W4), testing against nonce-word baselines, and holding syntax strictly invariant across comparisons.
