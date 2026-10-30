# Steering, Probing, and Interpretability: Reading Guide & Triage

This directory contains 10 foundational papers on mechanistic interpretability, representation probing, causal activation steering, concept erasure, and representational confounds in Large Language Models (LLMs). The central questions uniting this collection are:
1. **The Probe Validity Question:** When a linear classifier or probe predicts a linguistic property (such as grammatical gender) from internal activations, does the model genuinely use that representation, or has the probe merely memorized surface regularities or token orthography?
2. **The Causal Steering Question:** Does adding a directional activation offset ($\mathbf{h} \leftarrow \mathbf{h} + \alpha \mathbf{v}$) causally steer downstream semantic choices, or does it merely disrupt model confidence, induce fluency degradation, or act as an unspecific shock?
3. **The Activation Outlier & Readout Question:** How do massive activation outliers and attention sinks distort representation geometry, and why must model evaluations measure direct probability distributions rather than relying on metalinguistic prompting or first-token multiple-choice shortcuts?

These papers provide the direct methodological scaffolding, control baselines, and defense strategies for **Phase 4 (Cross-Lingual Probing Rigor)** and **Phase 5 (Causal Activation Steering & Readout Controls)**.

---

## 1. Executive Triage & Priority Matrix

| Paper | Focus & Core Mechanism | Project Role | Reading Priority |
|---|---|---|:---:|
| **[Belrose et al. 2023]**<br>*(LEACE: Closed-Form Concept Erasure)* | Closed-form affine projection that guarantees linear guarding with minimal representational damage. | **Tier 1 Methodological Anchor**<br>The theoretical standard for linear concept isolation and erasure (Phase 1 spelling eraser / covariate removal). | **Must-Read First** |
| **[Turner et al. 2023]**<br>*(Activation Engineering / ActAdd)* | Modifying forward pass activations at inference time via steering vectors ($\mathbf{h} + \alpha \mathbf{v}$). | **Tier 1 Methodological Anchor**<br>The architectural blueprint for Phase 5 activation steering (`steering.py`). | **Must-Read First** |
| **[Hewitt & Liang 2019]**<br>*(Control Tasks for Probing)* | Selectivity diagnostic: comparing probe performance on linguistic tasks vs. randomized control tasks. | **Tier 1 Methodological Anchor**<br>Justifies Phase 4 shuffle nulls, ending stratification, and Russian *-ь* invariant controls. | **Must-Read First** |
| **[Li et al. 2023]**<br>*(Inference-Time Intervention / ITI)* | Head-specific activation shifts; data-efficient vector extraction; quantifying truthfulness vs. helpfulness trade-offs. | **Tier 2 Causal Framework**<br>Informs Phase 5 dose calibration and the concept of a "working window" before damage. | **High Priority** |
| **[Buttigieg et al. 2026]**<br>*(Bias Directions Capture Confidence)* | Bias steering vectors often capture token likelihood/confidence rather than bias, driving abstention. | **Tier 2 Critical Confound Warning**<br>Mandates the Phase 5 grammatical agreement gate and P29 nonce controls to rule out confidence artifacts. | **High Priority** |
| **[Jorgensen et al. 2023]**<br>*(Mean-Centring Activation Steering)* | Subtracting dataset mean activations from steering vectors removes common unspecific offsets. | **Tier 2 Technical Refinement**<br>Directly justifies within-language mean-centering in Phase 4/5 POOLED directions. | **High Priority** |
| **[Sun et al. 2024]**<br>*(Massive Activations in LLMs)* | Tiny fractions of hidden dimensions exhibit massive activation magnitudes (>1000×) across layers. | **Tier 2 Geometric Confound**<br>Explains why unit-normalization and median-norm scaling ($\alpha \times \text{median}$) are required. | **High Priority** |
| **[Xiao et al. 2023]**<br>*(Attention Sinks in LLMs)* | Initial tokens (BOS) absorb disproportionate attention mass to stabilize softmax distributions. | **Tier 2 Attention Dynamics**<br>Explains why steering must isolate the noun tokens and avoid corrupting initial attention sinks. | **High Priority** |
| **[Hu & Lévy 2023]**<br>*(Prompting vs. Probing / Probabilities)* | Metalinguistic prompting fails to reflect internal knowledge; direct probability scoring is vastly superior. | **Tier 3 Evaluation Justification**<br>Validates Phase 5's "scoring, not sampling" protocol across fixed adjectival candidate distributions. | **Medium Priority** |
| **[Robinson et al. 2024]**<br>*(First-Token Probabilities vs. Text)* | Multiple-choice first-token probabilities decouple from full text answers in instruction-tuned LLMs. | **Tier 3 Readout Diagnostic**<br>Explains why forced-choice R2 (man/woman) diverges from continuous adjective scoring R1. | **Medium Priority** |

---

## 2. Thematic Overlap & Cluster Map

```
                     ┌─────────────────────────────────────────────────────────┐
                     │     STEERING, PROBING & INTERPRETABILITY LANDSCAPE      │
                     └─────────────────────────────────────────────────────────┘
                                                  │
         ┌────────────────────────────────────────┼────────────────────────────────────────┐
         ▼                                        ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│       Probing Validity &        │      │       Causal Steering &         │      │      Activation Geometry &      │
│      Representational Rigor     │      │        Concept Erasure          │      │        Evaluation Dynamics      │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
         │                                        │                                        │
         ├────────────────────────┐               ├────────────────────────┐               ├────────────────────────┐
         ▼                        ▼               ▼                        ▼               ▼                        ▼
【Probe Selectivity】     【Confidence Bias】   【Concept Erasure】      【Activation Add】    【Massive Outliers】     【Probability vs Prompt】
Hewitt & Liang (2019)     Buttigieg et al.(2026) Belrose et al. (2023)    Turner et al. (2023)  Sun et al. (2024)        Hu & Lévy (2023)
• Linguistic vs control   • Vectors encode       • LEACE closed-form      • Forward pass edit   • Channel-specific peaks • Metalinguistic prompts
• Probe memorization risk   model confidence     • Minimum distortion     • Linear activation   • Distorts raw norms       fail to reflect knowledge
• Requires shuffle nulls  • Induces abstention   • Oblique projection       addition (CAA)      • Mandates median scale  • Scoring > sampling
         │                        │               │                        │                       │                        │
         │                        │               ▼                        ▼                       ▼                        ▼
         │                        │      【Intervention Window】  【Mean-Centring】        【Attention Sinks】      【First-Token Bias】
         │                        │      Li et al. (2023 / ITI)   Jorgensen et al. (2023)  Xiao et al. (2023)       Robinson/Wang et al. (2024)
         │                        │      • Steering trade-offs    • Unspecific bias removal• BOS token sink         • MC logits decouple from
         │                        │      • Damage vs efficacy     • Functional vectors     • Contextual stability     generation trajectories
         │                        │               │                        │                       │                        │
         └────────────────────────┼───────────────┴────────────────────────┴───────────────────────┴────────────────────────┘
                                  ▼
         ┌─────────────────────────────────────────────────────────────────┐
         │                PROJECT APPLICATION & SYNTHESIS                  │
         │ • Phase 4: Stratified probes, shuffle cross-terms, Russian *-ь  │
         │ • Phase 5: Agreement gate, working damage window, noun-only push│
         │ • Readouts: Exact log-prob regression (Glasgow) over open text  │
         └─────────────────────────────────────────────────────────────────┘
```

### 2.1 Probing Validity & The Representation vs. Memorization Dilemma
* **The Probe Memorization Challenge (Hewitt & Liang 2019):** High classification accuracy on representation probes does not establish that the network uses or encodes that linguistic feature; probes with sufficient parameter capacity easily memorize superficial token regularities. Probes must be evaluated by *selectivity* against randomized control tasks.
* **The Steering Confound Challenge (Buttigieg et al. 2026):** Directional vectors derived from prompt contrasts (e.g., positive vs. negative, biased vs. unbiased) frequently isolate *model confidence* rather than the targeted semantic construct. Intervening along such vectors merely degrades token probabilities or forces the model into generic, hesitant states, masquerading as bias reduction.
* **Our Project Safeguards:**
  1. In Phase 4, probes are fitted strictly within morphological ending cells (ruling out surface spelling memorization) and tested against within-cell permutation nulls (T2 shuffle cross-term) and invariant morphological classes (Russian *-ь*).
  2. In Phase 5, before any semantic readout is evaluated, the steering direction must pass a functional **grammatical agreement gate** (flipping *el/la* or *der/die* without collapsing random directions), proving that the vector operates on the syntax-governing circuit rather than inducing unspecific entropy.

### 2.2 Steering & Erasure: From Theory to Intervention
* **Linear Erasure in Closed Form (Belrose et al. 2023):** LEACE establishes that any unwanted linear concept can be neutralized via an affine projection with minimal disturbance to the remaining representation space. This provides the mathematical benchmark for isolating grammatical gender from orthographic features (Phase 1) and lexical covariates.
* **Activation Engineering & Mean-Centring (Turner et al. 2023; Jorgensen et al. 2023):** Turner et al. demonstrated that linear activation offsets modify generation without fine-tuning. Jorgensen et al. proved that raw difference vectors are contaminated by global activation centroids; mean-centring activations before vector extraction isolates the true functional direction and prevents collateral damage.
* **Dose Calibration & Damage Windows (Li et al. 2023):** As shown in ITI, intervening in activations creates a trade-off between targeted behavioral shift and network damage. Phase 5 operationalizes this by defining an exact **working damage window** based on KL divergence on neutral sentences, ensuring that steering doses do not exceed baseline random disruption.

### 2.3 Activation Outliers & Rigorous Readout Design
* **Outlier Suppression (Sun et al. 2024; Xiao et al. 2023):** Transformer activations contain massive outlier dimensions and attention sinks that dwarf typical token vectors. Consequently, steering doses ($\alpha$) cannot be defined in arbitrary Euclidean norms; they must be calibrated against the **median layer norm** of noun tokens, and steering must be injected exclusively at the target noun to prevent destabilizing initial attention sinks.
* **Scoring vs. Sampling (Hu & Lévy 2023; Robinson et al. 2024):** Open-ended generation and metalinguistic prompting are noisy and subject to generation-order artifacts. Measuring the exact log-probability distribution across psycholinguistically calibrated candidate sets (Glasgow Norms) provides a direct, unconfounded readout of representational shift.

---

## 3. Rhetorical Ammunition & Citation Guide

When defending experimental rigor, probe selectivity, and steering mechanics in Phase 4 and Phase 5 against reviewer skepticism, cite these papers directly:

### 3.1 Defending Phase 4 Probing Rigor Against Probe Memorization
* **The Reviewer Objection:** *"Your probes achieve 70–90% AUC on predicting gender across languages. Isn't this just a high-capacity probe memorizing orthographic patterns or word frequencies rather than discovering a genuine latent gender direction?"*
* **The Counter-Argument / Citation:**
  > Following the methodological warnings of **Hewitt & Liang (2019)** regarding probe memorization, our probe design enforces strict selectivity controls:
  > 1. All gender contrasts are evaluated **within morphological ending cells** (stratified estimators), preventing the probe from exploiting language-specific noun endings (e.g., Spanish *-o/-a*, German *-e*).
  > 2. All candidate directions are residualized against word frequency, character length, and concreteness.
  > 3. Directional overlap (T2) is judged against a within-cell **split-half cross-term shuffle null**, ensuring that alignment reflects structured covariance rather than random high-dimensional dot products.
  > 4. Cleanest of all, the direction transfers zero-shot to **Russian inanimate nouns ending in *-ь*** (T4, AUC 0.82–0.85 in Qwen3-8B/14B), where orthography is identical across genders and the script (Cyrillic) was completely absent from training noun sets.

### 3.2 Defending Phase 5 Steering Vectors: Ruling Out Confidence and Abstention Artifacts
* **The Reviewer Objection:** *"Recent work shows that steering vectors often just perturb model confidence or induce token entropy (e.g., Buttigieg et al., 2026). How do you know your grammatical gender push isn't just making the model confused?"*
* **The Counter-Argument / Citation:**
  > We directly address the confound identified by **Buttigieg et al. (2026)** through a three-stage gating and control architecture:
  > 1. **Functional Grammatical Gate:** The steering vector must successfully flip grammatical agreement in a native syntactic frame (*Mi {noun} es muy ___* / *blanco* vs. *blanca*) at dose $\alpha^*$, while 20 random vectors of identical norm flip fewer than 10% of nouns. This proves the vector acts specifically on the syntactic gender pathway.
  > 2. **Damage Window (Li et al., 2023):** We measure whole-vocabulary KL divergence on neutral sentences to verify that the real vector induces no more disruption than random directions at the operating dose.
  > 3. **Non-Gender Orthogonal Controls:** Plural-minus-singular number vectors and random Gaussian vectors of identical norm yield near-zero adjectival shift ($\beta \approx 0.00$), demonstrating that the observed shift is feature-specific and not a general consequence of activation displacement.

### 3.3 Justifying Mean-Centring and Stratified Vector Extraction
* **The Reviewer Objection:** *"When pooling Spanish and German nouns to construct cross-lingual directions, how do you prevent the direction from encoding language identity rather than grammatical gender?"*
* **The Counter-Argument / Citation:**
  > As established by **Jorgensen et al. (2023)**, uncentered activation steering vectors are heavily corrupted by the shared centroid of the dataset. To ensure our cross-lingual directions isolate grammatical features rather than language markers, we apply **within-language mean-centring** prior to pooling. By de-meaning Spanish and German activations independently within their respective language distributions, the projection axis is mathematically orthogonal to the "is this Spanish vs. German?" language discrimination vector.

### 3.4 Justifying "Scoring, Not Sampling" and Rejecting First-Token Multiple-Choice Shortcuts
* **The Reviewer Objection:** *"Why didn't you evaluate bleedthrough by asking the model in free text: 'Describe a bridge' or asking it multiple-choice questions like 'A bridge is: (A) Strong, (B) Elegant'?"*
* **The Counter-Argument / Citation:**
  > Free-text sampling and multiple-choice prompting introduce severe methodological confounds. As **Hu & Lévy (2023)** demonstrate, metalinguistic prompting tests emergent task comprehension rather than direct linguistic representations, frequently producing false negatives. Furthermore, **Robinson et al. (2024)** show that first-token multiple-choice probabilities systematically decouple from underlying semantic preferences due to superficial token biases. By evaluating exact log-probabilities over a broad, pre-registered lexicon of human-normed adjectives (Glasgow Norms) with linear regression controlling for valence, arousal, and size, we obtain an unconfounded, continuous measurement of semantic drift.

---

## 4. Annotated Inventory of Papers

### 1. Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff & Biderman (2023)
* **Title:** *LEACE: Perfect Linear Concept Erasure in Closed Form*
* **ArXiv / Venue:** NeurIPS 2023 / arXiv:2306.03819v4 [cs.LG], April 2025.
* **File:** [`[Belrose et al. 2023] LEACE - Linear Concept Erasure in Closed Form.pdf`]([Belrose%20et%20al.%202023]%20LEACE%20-%20Linear%20Concept%20Erasure%20in%20Closed%20Form.pdf)
* **Methodology:**
  * Derives an exact, closed-form affine transformation that renders representations linearly guarded with respect to a target concept $Z$ (minimizing loss for all linear classifiers to chance).
  * Whitens the representation covariance, orthogonally projects out the cross-covariance with $Z$, and un-whitens, yielding an optimal oblique projection that minimizes geometric distortion under arbitrary inner product norms.
  * Extends concept erasure to deep networks via layer-by-layer *concept scrubbing*.
* **Key Findings:**
  * Outperforms iterative heuristic methods (INLP, RLACE) by reaching optimal theoretical erasure in a single closed-form pass, operating orders of magnitude faster.
  * Successfully scrubs social gender from BERT embeddings (Bias in Bios) and part-of-speech tags from LLaMA and Pythia, proving that linear concept removal can be achieved with minimal side-effect damage to adjacent representations.
* **Limitations:**
  * Erases strictly linear information; non-linear probe architectures or downstream attention layers may theoretically recover non-linear remnants.
* **Project Relevance:**
  * **Theoretical Gold Standard.** Formulates our mathematical understanding of linear concept spaces. Directly underpins our covariate residualization (frequency, length, concreteness) and provides the formal framework for the Phase 1 spelling eraser.

---

### 2. Turner, Udell, Thiergart, Vazquez, Leech, Mini & MacDiarmid (2023)
* **Title:** *Steering Language Models With Activation Engineering*
* **ArXiv / Venue:** arXiv:2308.10248v5 [cs.CL], October 2024.
* **File:** [`[Turner et al. 2023] Steering Language Models with Activation Engineering.pdf`]([Turner%20et%20al.%202023]%20Steering%20Language%20Models%20with%20Activation%20Engineering.pdf)
* **Methodology:**
  * Introduces Activation Addition (ActAdd), modifying forward-pass hidden states during inference without updating model weights or altering prompt prefixes.
  * Extracts steering vectors by taking the difference in intermediate activations between pairs of contrasting prompts (e.g., "Love" vs. "Hate") and adds them scaled by a steering coefficient $\alpha$ at target transformer layers.
* **Key Findings:**
  * Linear activation addition reliably controls sentiment, topic, high-level behavioral goals, and style in generative LLMs.
  * Demonstrates that steering vectors exhibit compositional properties and remain effective across varied prompt contexts.
  * Provides an efficient, training-free mechanism to explore latent causal representations.
* **Limitations:**
  * Applies vectors globally across all token positions, which can distort the generation of target tokens and cause grammar collapse at high doses.
* **Project Relevance:**
  * **Foundational Architecture for Phase 5.** Supplies the core mechanism of `src/gbleed/steering.py`. Our implementation refines Turner et al. by restricting the intervention strictly to **noun tokens** (preventing direct distortion of generated continuation tokens) and establishing an agreement-flipping gate.

---

### 3. Hewitt & Liang (2019)
* **Title:** *Designing and Interpreting Probes with Control Tasks*
* **ArXiv / Venue:** EMNLP 2019 / arXiv:1909.03368v1 [cs.CL], September 2019.
* **File:** [`[Hewitt & Liang 2019] Designing and Interpreting Probes with Control Tasks.pdf`]([Hewitt%20&%20Liang%202019]%20Designing%20and%20Interpreting%20Probes%20with%20Control%20Tasks.pdf)
* **Methodology:**
  * Formulates the concept of *probe selectivity* to resolve whether probes measure genuine representation quality or merely probe capacity to memorize.
  * Pairs linguistic probing tasks (e.g., part-of-speech tagging) with randomized control tasks that associate word types with arbitrary synthetic labels, forcing the probe to rely on memorization.
  * Selectivity is defined as the difference between linguistic task accuracy and control task accuracy.
* **Key Findings:**
  * High probe accuracy is insufficient evidence that a representation encodes linguistic structure; high-capacity linear and MLP probes achieve high accuracy on completely random control tasks.
  * Linear probes with restricted capacity achieve superior selectivity, proving that simpler probes provide more faithful diagnostic reflections of latent geometry.
* **Limitations:**
  * Focused on word-level static representations (ELMo, GloVe) and syntactic tagging tasks; does not evaluate generative transformer hidden states.
* **Project Relevance:**
  * **Core Probing Defense.** Justifies the methodological strictness of Phase 2, 3, and 4: utilizing minimal-parameter difference-of-means estimators, testing against within-cell shuffle nulls, and validating on held-out spelling-controlled classes.

---

### 4. Li, Patel, Viégas, Pfister & Wattenberg (2023)
* **Title:** *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*
* **ArXiv / Venue:** NeurIPS 2023 / arXiv:2306.03341v6 [cs.LG], June 2024.
* **File:** [`[Li et al. 2023] Inference-Time Intervention - Eliciting Truthful Answers.pdf`]([Li%20et%20al.%202023]%20Inference-Time%20Intervention%20-%20Eliciting%20Truthful%20Answers.pdf)
* **Methodology:**
  * Develops Inference-Time Intervention (ITI), identifying attention heads with high linear probe accuracy for truthfulness vs. untruthfulness on TruthfulQA.
  * Shifts activations during autoregressive inference along truthful directions across top attention heads.
  * Measures the empirical trade-off between truthfulness enhancement and model helpfulness/coherence across varying intervention strengths.
* **Key Findings:**
  * Shifts in a small subset of attention heads double truthfulness (32.5% to 65.1% in Alpaca) with negligible data requirements (a few hundred examples).
  * Confirms that LLMs maintain internal linear representations of truthfulness even when generating falsehoods on the surface.
  * Establishes that excessive steering magnitude collapses general linguistic coherence, requiring systematic tuning of the intervention budget.
* **Limitations:**
  * Focuses specifically on factual accuracy and truthfulness benchmarks; targets individual attention heads rather than residual streams.
* **Project Relevance:**
  * Direct precedent for dose sweeps and calibration. Informs our Phase 5 protocol of identifying the minimal gate-flipping dose ($\alpha^*$) and measuring language model damage via KL divergence.

---

### 5. Buttigieg, Madigan, Kamalaruban & Burrell (2026)
* **Title:** *Latent space bias directions in LLMs capture confidence, not fairness*
* **ArXiv / Venue:** arXiv:2610.08559v1 [cs.CL], October 2026.
* **File:** [`[Buttigieg et al. 2026] Bias Directions in LLMs Capture Confidence Not Fairness.pdf`]([Buttigieg%20et%20al.%202026]%20Bias%20Directions%20in%20LLMs%20Capture%20Confidence%20Not%20Fairness.pdf)
* **Methodology:**
  * Investigates what linear debiasing steering vectors—extracted by contrasting biased and anti-biased prompt activations—actually encode in LLM hidden states.
  * Evaluates steering interventions across both social bias benchmarks and general knowledge question-answering datasets.
* **Key Findings:**
  * The primary axis separating biased from anti-biased prompt activations corresponds to **model confidence and token probability**, not an abstract concept of fairness or social bias.
  * Steering along this axis reduces measured bias scores solely because it suppresses model confidence, inducing the model to abstain from answering or output generic tokens.
  * Shows that debiasing interventions can easily be misinterpreted when directional vectors conflate semantic features with certainty gradients.
* **Limitations:**
  * Evaluates contrastive debiasing on QA and classification benchmarks; does not study grammatical morphology.
* **Project Relevance:**
  * **Critical Methodological Warning.** Directly motivates the dual verification in Phase 5: steering vectors must flip functional grammar (the agreement gate) and must be tested on nonce nouns (P29) and orthogonal directions (number, social) to ensure shifts do not reflect confidence suppression.

---

### 6. Jorgensen, Cope, Schoots & Shanahan (2023)
* **Title:** *Improving Activation Steering in Language Models with Mean-Centring*
* **ArXiv / Venue:** arXiv:2312.03813v1 [cs.CL], December 2023.
* **File:** [`[Jorgensen et al. 2023] Improving Activation Steering with Mean-Centring.pdf`]([Jorgensen%20et%20al.%202023]%20Improving%20Activation%20Steering%20with%20Mean-Centring.pdf)
* **Methodology:**
  * Proposes mean-centring for activation steering: subtracting the mean activation vector of the general training/evaluation corpus from target concept activations before computing steering directions.
  * Benchmarks mean-centred steering vectors against standard activation addition on detoxification, story genre steering, and function vector elicitation.
* **Key Findings:**
  * Uncentered difference vectors are dominated by the global centroid of the representation space, introducing spurious offsets that degrade generation quality.
  * Mean-centring isolates the clean functional direction, dramatically improving steering efficacy and reducing off-target behavioral degradation across diverse natural language tasks.
* **Limitations:**
  * Requires estimating the activation mean across a representative corpus sample.
* **Project Relevance:**
  * **Direct Mathematical Justification.** Validates the within-language mean-centring implemented in Phase 4 for the POOLED direction (centering Spanish and German activations independently to eliminate language-origin confounds) and Phase 5 vector extraction.

---

### 7. Sun, Chen, Kolter & Liu (2024)
* **Title:** *Massive Activations in Large Language Models*
* **ArXiv / Venue:** arXiv:2402.17762v2 [cs.LG], February 2024.
* **File:** [`[Sun et al. 2024] Massive Activations in Large Language Models.pdf`]([Sun%20et%20al.%202024]%20Massive%20Activations%20in%20Large%20Language%20Models.pdf)
* **Methodology:**
  * Systematic empirical audit of activation distributions across hidden states and MLP intermediate layers in mainstream LLMs (LLaMA, Falcon, Mistral).
  * Tracks the emergence and behavior of extreme activation values across sequence positions and depth.
* **Key Findings:**
  * LLMs consistently exhibit "massive activations": a minute fraction of hidden dimensions (<0.1%) output values up to 1,000–10,000× larger than typical dimensions.
  * These massive activations emerge in specific middle layers, acting as persistent bias terms or memory registers for global context.
  * Unnormalized Euclidean operations or raw vector magnitudes are heavily skewed by these outlier channels, distorting standard cosine similarity metrics.
* **Limitations:**
  * Focuses on diagnosing the phenomenon and its pruning implications; does not propose an alternative probing framework.
* **Project Relevance:**
  * Dictates our normalization strategy: steering dose $\alpha$ is calibrated as a multiple of the **median token activation norm** rather than mean or max norms, preventing massive outlier channels from distorting the intervention scale.

---

### 8. Xiao, Tian, Chen, Han & Lewis (2023/2024)
* **Title:** *Efficient Streaming Language Models with Attention Sinks*
* **ArXiv / Venue:** ICLR 2024 / arXiv:2309.17453v4 [cs.CL], April 2024.
* **File:** [`[Xiao et al. 2023] Streaming Language Models with Attention Sinks.pdf`]([Xiao%20et%20al.%202023]%20Streaming%20Language%20Models%20with%20Attention%20Sinks.pdf)
* **Methodology:**
  * Investigates multi-round dialogue and streaming generation failure when caching limited KV windows.
  * Analyzes attention map distributions across autoregressive decoding steps.
* **Key Findings:**
  * Autoregressive transformers assign an extraordinarily large proportion of attention weights to the very first tokens of a sequence (the BOS token and initial positions), regardless of their semantic relevance.
  * These "attention sinks" are an architectural consequence of Softmax normalization requiring probabilities to sum to 1; the model parks unneeded attention mass on initial tokens.
  * Removing or perturbing these initial tokens causes catastrophic collapse in downstream attention distributions and output perplexity.
* **Limitations:**
  * Primarily focused on KV-cache compression and streaming inference architectures.
* **Project Relevance:**
  * Clarifies why Phase 5 steering must be injected **strictly at the noun's subword tokens** rather than all sequence positions. Perturbing initial prompt tokens alters attention sinks, destroying the contextual stability required for measuring next-token probabilities.

---

### 9. Hu & Lévy (2023)
* **Title:** *Prompting is not a substitute for probability measurements in large language models*
* **ArXiv / Venue:** EMNLP 2023, pp. 5040–5055.
* **File:** [`[Hu & Lévy 2023] Prompting vs Probing - Measuring LLM Knowledge.pdf`]([Hu%20&%20L%C3%A9vy%202023]%20Prompting%20vs%20Probing%20-%20Measuring%20LLM%20Knowledge.pdf)
* **Methodology:**
  * Compares metalinguistic prompting (asking models to explicitly judge linguistic acceptability or properties) against direct probability measurements over target strings across diverse linguistic phenomena.
* **Key Findings:**
  * Direct probability readouts derived from internal representations are vastly more consistent, robust, and sensitive than metalinguistic prompting.
  * LLMs frequently fail metalinguistic judgment tasks despite having perfect internal probability distributions that reflect the exact linguistic distinction.
  * Concludes that negative results obtained via prompting cannot be interpreted as evidence that the model lacks the targeted linguistic representation.
* **Limitations:**
  * Evaluated on syntax and linguistic acceptability benchmarks; does not evaluate latent steering.
* **Project Relevance:**
  * **Primary Methodological Justification for Readout Design.** Directly validates our Phase 5 "scoring, not sampling" paradigm: instead of prompting the model to generate text or select answers, we measure exact next-word log-probabilities over curated psycholinguistic adjectives.

---

### 10. Robinson et al. / Wang, Ma, Hu, Weber-Genzel, Röttger, Kreuter, Hovy & Plank (2024)
* **Title:** *“My Answer is C”: First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models*
* **ArXiv / Venue:** Findings of ACL 2024, pp. 7406–7419.
* **File:** [`[Robinson et al. 2024] My Answer is C - First-Token Probabilities vs Full Text.pdf`]([Robinson%20et%20al.%202024]%20My%20Answer%20is%20C%20-%20First-Token%20Probabilities%20vs%20Full%20Text.pdf)
* *(Note: Cataloged as Robinson et al. 2024; authored by Wang et al., 2024).*
* **Methodology:**
  * Evaluates instruction-tuned LLMs on multiple-choice question answering, comparing the first-token probability distribution over option labels (A, B, C, D) against the option selected when generating full text responses.
* **Key Findings:**
  * First-token probabilities exhibit severe misalignment with the actual content generated during autoregressive text rollout.
  * Instruction tuning induces strong label biases and formatting priors on the first token that do not reflect the model's latent preferences or semantic understanding.
  * Multiple-choice token scoring can drastically mischaracterize model capabilities and beliefs.
* **Limitations:**
  * Focuses on multiple-choice reasoning benchmarks (MMLU, QA).
* **Project Relevance:**
  * Explains our Phase 5 empirical findings: forced-choice R2 readouts ("man" vs. "woman") show superficial sensitivity to immediate token context, whereas continuous regression over the full adjectival distribution (R1) provides the true measure of whether semantic gender associations moved.
