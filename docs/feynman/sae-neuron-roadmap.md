# Neuron-level analysis & sparse autoencoders for your gender-vector project (Qwen3 → Gemma)

**Prepared:** 2026-10-21 · **Purpose:** what the monosemanticity/polysemanticity discussion means for "finding the gender neuron," what actually exists for Qwen3 and Gemma 3 SAE-wise, and a concrete three-phase plan that connects SAEs back to your gender-vector experiment.

---

## TL;DR

1. **Don't hunt for a single gender neuron.** The monosemanticity essays explain *why* (superposition), and a 2026 COLM paper — tested on your exact Qwen3 models — shows empirically that grammatical "grandmother neurons" are rare: grammatical distinctions live in ~2.5–4% of neurons per contrast, each with weak, narrow selectivity, and whole-vector probes decode what single neurons can't. SAEs are the modern answer to exactly this problem.
2. **You have first-class SAE tooling for both of your model families:**
   - **Qwen3:** Qwen-Scope (Alibaba's official suite — SAEs for all 36 layers of Qwen3-8B, plus Qwen3-1.7B/30B-A3B and Qwen3.5 variants), loadable in SAELens since v6.43.0 (May 2026). Community alternatives exist (BatchTopK and JumpReLU releases).
   - **Gemma:** Gemma Scope 2 (Dec 2025) = SAEs + transcoders + crosscoders for the whole Gemma 3 family (270M–27B), every layer, on Neuronpedia and HF, loadable in SAELens. Gemma Scope (original, 2024) covers Gemma 2 2B/9B/27B.
3. **The confidence-confound warning from 2610.08559 applies to SAE steering too** — build the control battery in from the start.

---

## 1. The monosemanticity ↔ polysemanticity discussion, in one page

| Source (chronological) | Claim you should absorb |
|---|---|
| Olah et al., **Zoom In: An Introduction to Circuits** (2020, Transformer Circuits Thread) | The original essay framing: neurons are not semantic atoms; look for circuits and features instead. |
| Elhage, Olah et al., **Toy Models of Superposition** (2022, arXiv 2209.10652) | Polysemanticity is not an accident — it's what networks *do* when they represent more features than dimensions: features live in superposition, so individual neurons reliably mix unrelated concepts. This is the canonical explanation for why "pin down a neuron" fails. |
| Bricken et al., **Towards Monosemanticity** (2023, Transformer Circuits Thread) | Dictionary learning (SAEs) recover ~interpretable features from residual-stream activations; the *feature*, not the neuron, is the unit. Male/female and other gendered-concept features are among the canonical examples. |
| Cunningham et al., **Sparse Autoencoders Find Highly Interpretable Features in Language Models** (2023, arXiv 2309.08600) | SAEs work on full-size LMs (Pythia, GPT-2); feature density/interpretability scaling laws. |
| Templeton et al., **Scaling Monosemanticity** (Anthropic, 2024; arXiv version 2605.29358) | SAEs scale to frontier models (Claude 3 Sonnet): tens-of-millions-of-feature dictionaries; explicitly documents gender-type features (e.g., opposing "female"/"male" feature groups, gendered-role clusters). |
| OpenAI, **Extracting Concepts from GPT-4** (2024) https://openai.com/index/extracting-concepts-from-gpt-4/ | 16M-feature dictionary on GPT-4 — demonstrates feature-level interpretability at production scale, including features tracking gender of the text's referent. |
| Lindsey, Gurnee, Ameisen, et al. (Anthropic), **On the Biology of a Large Language Model** (2025) https://transformer-circuits.pub/2025/attribution-graphs/biology.html | Feature-level circuit tracing on Claude 3.5 Haiku; the "microscope → biology" argument — features are the entry point; circuits are the next step. |
| **He & Mesgarani, Grammatical "grandmother neurons" are rare in LLMs** (COLM 2026, arXiv 2609.29328) | **The direct empirical answer to your question.** Probe-free, permutation-normalized Neuron Separability Index on 68 BLiMP/COMPS paradigms incl. gender-agreement contrasts, across Qwen3-0.6B/1.7B/4B/8B + Pythia/TinyLlama/Llama-3.1. Findings: (i) only **2.5–3.9%** of neurons show any grammatical sensitivity per paradigm; (ii) mean NSI ≈ 0.5 even for sensitive neurons (weak); (iii) only **3 of 68 paradigms** contain a single strong "grandmother" neuron (NSI>2) — gender agreement is not one of them; (iv) whole-vector probing (0.917 mean accuracy) ≫ single-neuron separability; (v) probe accuracy, neuron selectivity, and behavioral competence are **dissociated** (behavior↔probe r ≈ −0.03; probe↔mean-NSI r ≈ −0.56): a neuron showing high raw selectivity can be irrelevant to behavior, and ablating the rarest strong units does no more damage than random ablation. |
| Recent 2026 debate papers worth citing in related work | "Many Transcoder Features Are Neurons in Disguise" (Anthropic + Georgia Tech, arXiv 2610.24521) — even transcoders re-derive neuron-like polysemantic features; "Polysemanticity or Polysemy? Lexical Identity Confounds Superposition Metrics" (arXiv 2604.00443); "The Rate-Distortion-Polysemanticity Tradeoff in SAEs" (EPFL/ISTA, arXiv 2605.14694); "A mathematical theory of superposition" (arXiv 2608.27540). |

**Net: your two workable units are (a) *directions* in activation space (you already have this: the centroid gender vector), and (b) *SAE features* (sparse, mostly-monosemantic directions). Single neurons are the wrong abstraction for grammar.**

---

## 2. SAEs for Qwen3: you have an official suite

**Qwen-Scope** — Qwen team (Alibaba Cloud), May 2026. arXiv:2605.11887 · HF: `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (and `L0_100`) etc. · SAELens loader since v6.43.0.

- **Coverage:** 14 SAE groups, 7 backbones: **Qwen3-1.7B, Qwen3-8B** (dense), **Qwen3-30B-A3B** (MoE), Qwen3.5-2B/9B/27B/35B-A3B. **All layers**, residual-stream hook, TopK activations (exactly 50 or 100 features on).
- **Qwen3-8B specs:** d_model 4096, width 65,536 (16×); per-layer checkpoints as plain PyTorch dicts (`W_enc`, `W_dec`, `b_enc`, `b_dec`); Gradio demo included. Trained on in-house pretraining data with dead-feature aux loss and **outlier-activation filtering**.
- **Practical Qwen-specific caveat (from the SAE authors and community trainers):** Qwen models produce **random attention sinks** — activation norms 100–1000× the median on ~0.1% of tokens — that dominate feature attribution scores if unfiltered. Qwen-Scope filters them; community trainer Adam Karvonen's `qwen3-8b-saes` notes the same (and got a 3% MSE improvement from filtering). **You must apply the same filter when scoring features on your Spanish/German templates.**
- **Community Qwen3 SAEs as alternatives:**
  - `adamkarvonen/qwen3-8b-saes` — BatchTopK, 16k/65k widths, layers 25/50/75%; 500M tokens (90% FineWeb, 10% LMSYS-Chat); usable as JumpReLU at inference.
  - `juiceb0xc0de/qwen3-0.6b-SAE` — JumpReLU, all 28 layers, zero dead features (fast pilot work).
  - `Silen/cosine-scored-saes-qwen3-8b` — layer-18, 65k; cosine activation scoring improves SAEBench single-feature sparse-probing top-1 from 0.667 → 0.808 at matched reconstruction (ICML 2026 MI workshop); relevant if you care about feature *selectivity* vs. reconstruction.
- **Note on variants:** Qwen-Scope SAEs live on **base** backbones (Qwen3-8B-Base), so run your analysis on base or confirm the residual-stream hook placement on your chat/instruct variant (SAELens `HookedSAETransformer` / HF hooks handle this, but verify).

## 3. SAEs for Gemma: Gemma Scope and Gemma Scope 2

- **Gemma Scope (Gemma 2):** 2B/9B/27B; SAEs on residual, MLP, attention outputs; widths 16k–~1M. HF: `google/gemma-scope-*`; landing: https://huggingface.co/google/gemma-scope. Technical report with the JumpReLU SAE recipe (also: Rajamanoharan et al., *JumpReLU/Matryoshka SAEs*, 2024/2025).
- **Gemma Scope 2 (Gemma 3) — the one you want for a Gemma replication:** released Dec 2025 by Google DeepMind's interpretability team. SAEs at `resid_post`, `attn_out`, `mlp_out` **for every layer** of Gemma 3 PT and IT at 270M / 1B / 4B / 12B / 27B, plus **transcoders, cross-layer transcoders, and weak-causal crosscoders** (multi-layer). Matryoshka training; widths 16k–1M; recommended defaults: 64k–256k width, "medium" L0. Loads in SAELens (`SAE.from_pretrained(release="gemma-scope-2-27b-pt-resid_post", sae_id="layer_12_width_16k_l0_small")`); interactive dashboards on Neuronpedia (https://www.neuronpedia.org/gemma-scope-2); docs: https://ai.google.dev/gemma/docs/gemma_scope

## 4. Tooling stack (all verified current as of 2026-10-21)

| Tool | Role | Notes |
|---|---|---|
| **SAELens** (v6.54.12, Oct 2026) | Train/load/analyze SAEs; run evals; steering tutorial | GitHub `decoderesearch/SAELens`. Loaders for **Qwen Scope TopK** (v6.43.0+) and **Gemma Scope 2** (v6.24.0+) confirmed in changelog. Works with HF-transformers models directly; deep TransformerLens integration optional. Includes an "using an SAE as a steering vector" tutorial notebook. |
| **Neuronpedia** | Browse features, search by inference/explanation/token; steering API; UMAP | Hosts Gemma Scope, **Gemma Scope 2**, Llama Scope, gpt-oss, DeepSeek, and Qwen-family SAEs/transcoders (e.g., Qwen2.5-7B misaligned-persona SAEs; Qwen transcoders from mwhanna). Quick way to pre-explore "is there a German-feminine-bridge feature?" before any code. |
| **SAEBench** (Karvonen et al.) | Standard SAE quality evaluations (sparse probing, reconstruction, interpretability scores) | If you train SAEs yourself, report SAEBench metrics so results are comparable; the cosine-scored Qwen3 SAEs were evaluated this way. |
| **dictionary-learning / Sparsify / Overcomplete** | Training libraries (hackable, TopK-focused) | For training your own SAEs on Spanish/German-heavy corpora if the official ones underperform on your language material. |

## 5. Three-phase plan connecting SAEs to your gender-vector experiment

**Phase 0 — consolidate the exploratory write-up (no SAEs yet).**
Run the 2610.08559 control battery on whatever steering you already did (both-sign α sweeps; entropy/abstention; forced-choice evals; confidence-direction cosine). You can write the representational half now: layerwise gender-probe AUC on Qwen3; Spanish-vs-German direction cosine vs. a random-direction null; replication of 2407.09704's cross-language transfer at the hidden-state level.

**Phase 1 — the neuron question, cheap and honest.**
Rather than eyeballing neurons, run the He & Mesgarani-style **NSI** (permutation-normalized per-neuron separability) on *gender agreement* minimal pairs (e.g., German determiner/adjective agreement, Spanish article+adjective agreement) on your Qwen3 variant. Expect: vector-probe ≫ neuron separability; only a few weak neurons per contrast; ablating the "best" neurons ≈ random ablation. That is a publishable negative result that *motivates* Phase 2, and it directly engages the polysemanticity essays you've been reading. (Alternative: ITI-style head-level interventions, arXiv 2306.03341, or the neuron-level gendered-generation interventions in 2605.30717.)

**Phase 2 — SAE decomposition (Qwen3 → Gemma replication).**
1. Load Qwen-Scope SAEs for Qwen3-8B (all layers) via SAELens; apply attention-sink outlier filtering.
2. Collect activations on your Spanish/German inanimate-noun templates; for each SAE feature compute the contrast fire-rate `Δf = P(fire | feminine noun) − P(fire | masculine noun)` (the Qwen-Scope toxicity-classifier recipe, §5.3 — transferable to gender feature discovery).
3. Validate: top max-activating examples for the top-Δf features should be gender-coherent noun contexts across *both* languages; check cross-lingual feature sharing the way Brinkmann et al. (2501.06346) did (feature-overlap + steering flips grammatical gender in MT output).
4. Compare representations: cosine between your **centroid gender direction** and the top SAE feature directions; this tells you whether the linear gender axis is *one* feature or a *distributed bundle* (likely the latter; 2501.06346 found gender distributed over several shared features).
5. Steer the selected features in English completions (adjective generation à la 2407.09704 and pronoun completions) with the full control battery from Phase 0. Feature steering ≠ raw-vector steering: verify intervention effect with `Δ` magnitudes swept both signs, and check side-effect pre-intervention when possible (2606.08365).
6. **Replication on Gemma 3** with Gemma Scope 2 (same pipeline, same template sets — SAELens makes the swap ~config-only). Cross-model agreement in *which* features were gender-relevant and in the steering effects gives you a claims-about-the-phenomenon (vs. claims-about-one-model) lever, and you can additionally test whether the gender direction transfers *across models* (2501.02009).

## 6. Caveats to keep between you and an overclaim

- **Discovery vs. intervention:** features that *fire* with gender may not be the ones that *cause* gender-dependent output (2610.15850 — discovery scores and intervention effects diverge). Prefer causal tests (steering, ablation, patching) over "feature found" stories.
- **Steering side effects:** SAE feature steering is rarely clean across contexts (2606.08365); report side-effect scans, not just target flip rates.
- **The confidence confound survives the SAE step:** if your "gender features" were discovered by contrasting feminine vs. masculine templates, they can inherit likelihood differences; the 2610.08559 control battery belongs in Phase 2 as much as Phase 0.
- **Polysemanticity moves up a level:** SAEs reduce but don't eliminate it — rate–distortion tradeoffs (2605.14694), transcoders that re-derive neuron-like polysemy (2610.24521), and lexical-polysemy confounds in superposition metrics (2604.00443) are all live debates. Cite them; don't promise "atomic" features.
- **Corpus dependence:** Qwen-Scope SAEs were trained on English-heavy in-house pretraining data; features are a function of the SAE's training corpus ("zero dead features is not an interpretability statistic"). If Spanish/German gender features are under-represented, training your own SAE on a gendered-language corpus mix (via dictionary-learning/Sparsify + SAELens) is the fix — and is itself a contribution.

---

## Sources
- Elhage, N., et al. (2022). Toy Models of Superposition. arXiv:2209.10652 https://arxiv.org/abs/2209.10652
- Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread. https://transformer-circuits.pub/2023/monosemantic-features/
- Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600 https://arxiv.org/abs/2309.08600
- Templeton, A., et al. (2024). Scaling Monosemanticity. Anthropic; arXiv version: 2605.29358 https://arxiv.org/abs/2605.29358
- OpenAI (2024). Extracting Concepts from GPT-4. https://openai.com/index/extracting-concepts-from-gpt-4/
- Lindsey, J., Gurnee, W., Ameisen, E., et al. (2025). On the Biology of a Large Language Model. Anthropic. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- He, L., & Mesgarani, N. (2026). Grammatical "grandmother neurons" are rare in LLMs. COLM 2026. arXiv:2609.29328 https://arxiv.org/abs/2609.29328
- Qwen Team (2026). Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models. arXiv:2605.11887 https://arxiv.org/abs/2605.11887 · HF: https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50
- Google (2025). Gemma Scope 2 (SAEs/transcoders for Gemma 3). https://ai.google.dev/gemma/docs/gemma_scope · https://huggingface.co/google/gemma-scope-2-27b-pt · https://www.neuronpedia.org/gemma-scope-2
- Google (2024). Gemma Scope. https://huggingface.co/google/gemma-scope
- SAELens (v6.54.12, 2026). https://github.com/decoderesearch/SAELens (changelog: Qwen Scope TopK loader v6.43.0; Gemma Scope 2 loader v6.24.0)
- Neuronpedia docs: https://docs.neuronpedia.org/getting-started/find-resources
- Karvonen, A., et al. SAEBench. https://github.com/adamkarvonen/SAEBench
- Community Qwen3 SAEs: https://huggingface.co/adamkarvonen/qwen3-8b-saes · https://huggingface.co/datasets/juiceb0xc0de/qwen3-0.6b-SAE · https://huggingface.co/Silen/cosine-scored-saes-qwen3-8b
- Brinkmann, J., Wendler, C., Bartelt, C., & Mueller, A. (2025). LLMs Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages. arXiv:2501.06346 https://arxiv.org/abs/2501.06346
- Buttigieg, S., et al. (2026). Latent space bias directions in LLMs capture confidence, not fairness. arXiv:2610.08559 https://arxiv.org/abs/2610.08559
- Related 2026 debate papers: 2610.24521 (transcoder features as neurons in disguise), 2604.00443 (polysemy confounds), 2605.14694 (rate–distortion–polysemanticity), 2608.27540 (mathematical theory of superposition), 2610.15850 (discovery vs. intervention), 2606.08365 (steering side-effect prediction), 2501.02009 (cross-model concept transfer), 2605.30717 (neuron-level gendered generation interventions), 2306.03341 (Inference-Time Intervention)
