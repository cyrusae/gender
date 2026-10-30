# 07. Reasoning Mode and Chain of Thought (`07_Reasoning_Mode_and_Chain_of_Thought/`)

**Catalog of 45 Research Papers on Test-Time Compute, Reasoning Faithfulness, Overthinking, and Persona Dynamics in Large Language Models.**

This collection provides the foundational theoretical, mechanistic, and empirical literature supporting the **Question 4 (Q4) experimental arm** of the grammatical gender character study (`character/docs/prereg_v1.md`), which investigates within-model thinking-on versus thinking-off contrasts in **Qwen3-14B** and related reasoning architectures.

---

## Table of Contents

1. [Executive Triage & Tiered Reading List](#1-executive-triage--tiered-reading-list)
   - [Tier 1: The Essential Anchors (Must-Reads for Q4)](#tier-1-the-essential-anchors-must-reads-for-q4)
   - [Tier 2: High-Value Mechanistic Insights](#tier-2-high-value-mechanistic-insights)
   - [Tier 3: Domain-Specific Clusters](#tier-3-domain-specific-clusters)
2. [Thematic Overlap & Cluster Map](#2-thematic-overlap--cluster-map)
   - [Cluster Overview & High-Level Taxonomy](#cluster-overview--high-level-taxonomy)
   - [Replication Matrix 1: Distributional Concentration & Entropy Collapse](#replication-matrix-1-distributional-concentration--entropy-collapse)
   - [Replication Matrix 2: Bias Persistence & Counterfactual Flips](#replication-matrix-2-bias-persistence--counterfactual-flips)
   - [Replication Matrix 3: Faithfulness, Epiphenomenal Traces & Latent Commitment](#replication-matrix-3-faithfulness-epiphenomenal-traces--latent-commitment)
3. [Rhetorical Ammunition & Citation Matrix for Q4](#3-rhetorical-ammunition--citation-matrix-for-q4)
   - [1. Justifying the Character Generation Paradigm for Reasoning Tests](#1-justifying-the-character-generation-paradigm-for-reasoning-tests)
   - [2. Explaining Why Thinking Traces Omit Gender Despite Shifted Outputs](#2-explaining-why-thinking-traces-omit-gender-despite-shifted-outputs)
   - [3. Grounding Early Gender Commitment in Reasoning Blocks](#3-grounding-early-gender-commitment-in-reasoning-blocks)
   - [4. Justifying Sampling Parameter Divergence Across Modes](#4-justifying-sampling-parameter-divergence-across-modes)
   - [5. Justifying the Reasoning-Token Budget Cap](#5-justifying-the-reasoning-token-budget-cap)
4. [Complete Annotated Inventory of All 45 Papers](#4-complete-annotated-inventory-of-all-45-papers)

---

## 1. Executive Triage & Tiered Reading List

To navigate these 45 papers efficiently, reading is prioritized into three functional tiers based on immediate methodological necessity for our pre-registered study.

```
+-----------------------------------------------------------------------------------+
|                           TIER 1: THE ESSENTIAL ANCHORS                           |
|       Must-reads establishing Q4's methodological validity, priors, and risks     |
|   - Zheng et al. (2026): Within-model Qwen3 thinking-on/off bias flips (5x ratio) |
|   - Gao et al. (2026) & Song et al. (2026): Thinking inertia & entropy collapse   |
|   - Lanham et al. (2023): Stated reasoning as post-hoc unfaithful rationalization |
|   - Tu et al. (2025): Reasoning degrades role-playing, persona & creative voice   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        TIER 2: HIGH-VALUE MECHANISTIC INSIGHTS                    |
|        Mechanistic papers explaining latent states, budgets, and bias sites       |
|   - Chen et al. (2026) [SAE], Li et al. (2025), Zhang et al. (2026),              |
|     Shaikh et al. (2023), Bansal et al. (2026), McKenna et al. (2026),           |
|     Dua et al. (2026) [Commitment Boundary], Kwon et al. (2025) [Budget Control]  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         TIER 3: DOMAIN-SPECIFIC CLUSTERS                          |
|             Specialized evidence bases supporting specific design choices         |
|   - Creative Writing & Narrative under CoT (Zhao, Yang, Park, Li, Zhou, Gao)      |
|   - Overthinking & Test-Time Compute Scaling (Huang, Lin, Guo, Ding, Tan)         |
|   - Faithfulness & Rationalization Dynamics (Jiang, Venkatesh, Prasad, Sharma)    |
|   - Sampling, Diversity & Language of Thought (Mao, Srinivasan, Xu, Lee, Chen...) |
|   - Specialized Persona & Character Monologue (Wang, Zhang, Liu, Rao, Kocijan)   |
+-----------------------------------------------------------------------------------+
```

### Tier 1: The Essential Anchors (Must-Reads for Q4)

These 5 anchor papers directly frame the hypothesis, methodology, and interpretation of the Qwen3-14B thinking-on vs. thinking-off experimental arm:

1. **`[Zheng et al. 2026] Does Thinking Help Fairness? Reasoning Tokens Resolve Some Biases but Create More`**  
   - **Why It Is Essential:** The direct empirical and methodological precedent for Q4. Conducts a within-model thinking-on vs. thinking-off ablation across the Qwen family (including Qwen3-32B and QwQ-32B) on counterfactual pairs. Demonstrates that thinking mode does not unilaterally reduce demographic bias; instead, it resolves some biases while generating ~5× more new counterfactual flips, with bias amplifying along thinking depth.
   - **Core Takeaway for Q4:** Establishes that our Q4 arm tests an empirical open question rather than an assumed debiasing mechanism, and provides our core vocabulary of counterfactual flips.

2. **`[Gao et al. 2026] Thinking Inertia: LLMs Keep Thinking When Told Not To`** & **`[Song et al. 2026] A Few Bad Apples Spoil the Bunch: Preventing Global Entropy Collapse Driven by a Small Set of Tokens in LLM Reasoning`**  
   - **Why They Are Essential:** Establish the reality of mode leakage and distributional concentration. Gao et al. show that current hybrid LLMs cannot cleanly disable reasoning, and that reasoning leakage becomes most acute as the output space becomes open-ended (e.g., creative writing). Song et al. prove mathematically and empirically that extended thinking tokens trigger sudden entropy collapse across vocabulary distributions.
   - **Core Takeaway for Q4:** Justifies why our conditions are formally designated "thinking-mode on/off" rather than "with/without reasoning," and provides the theoretical foundation for why thinking mode concentrates lexical choice.

3. **`[Lanham et al. 2023] Measuring Faithfulness in Chain-of-Thought Reasoning`**  
   - **Why It Is Essential:** Anthropic's landmark causal study demonstrating that model reasoning chains are frequently unfaithful post-hoc rationalizations. When biasing context is present, the model commits to its decision in early latent representations and writes a plausible-sounding chain of thought that conceals the true causal driver.
   - **Core Takeaway for Q4:** Perfectly explains our pilot observation that reasoning blocks almost never mention gender (in ≤ 0.5% of cases) despite the model consistently generating a specific gendered character.

4. **`[Tu et al. 2025] Reasoning Does Not Necessarily Improve Role-Playing Ability`**  
   - **Why It Is Essential:** Evaluates 24 LLMs across six role-playing and persona benchmarks, providing definitive evidence that reasoning-optimized models and extended CoT actively degrade persona consistency and creative tone. Identifies two causal mechanisms: *attention diversion* (reasoning drains capacity from character fidelity) and *linguistic style drift* (forcing clinical, structured registers).
   - **Core Takeaway for Q4:** Serves as the primary literature precedent for Finding 4b. Provides the directional hypothesis that thinking mode will disrupt organic narrative voice and push generated paragraphs toward analytical structures.

---

### Tier 2: High-Value Mechanistic Insights

Eight essential papers that provide the internal interpretability, causal mechanisms, and technical parameters governing thinking models:

- **Chen et al. (2026)** (`[Chen et al. 2026] Thinking vs NoThinking - Sparse Autoencoders Analysis.pdf`): Applies Sparse Autoencoders (SAEs) to isolate feature representations between thinking and non-thinking states, identifying distinct meta-reasoning circuits and revealing that thinking mode actively suppresses open-ended creative features.
- **Li et al. (2025)** (`[Li et al. 2025] Does Reasoning Introduce Bias.pdf`): The first systematic audit demonstrating that social and demographic biases are actively generated and harbored *inside* reasoning traces, turning CoT into an independent site of bias propagation.
- **Zhang et al. (2026)** (`[Zhang et al. 2026] Mechanics of Bias and Reasoning.pdf`): Combines layer-wise probing with failure analysis to show that CoT fails to eliminate gender bias, leaving bias deeply entrenched within latent representations across all layers.
- **Shaikh et al. (2023)** (`[Shaikh et al. 2023] Lets Not Think Step by Step.pdf`): Landmark ACL finding that zero-shot CoT prompting systematically increases toxicity, social stereotyping, and bias compared to direct decoding, especially in larger model architectures.
- **Bansal et al. (2026)** (`[Bansal et al. 2026] Legibility is Not Interpretability.pdf`): COLM study proving that human-judged importance of reasoning steps does not correlate with their actual causal importance, demonstrating that legible chains of thought disguise non-transparent internal heuristics.
- **McKenna et al. (2026)** (`[McKenna et al. 2026] LLM Reasoning Is Latent.pdf`): Demonstrates that true reasoning in transformer models occurs within continuous latent activation space rather than through the surface tokens of the chain of thought.
- **Dua et al. (2026)** (`[Dua et al. 2026] Commitment Boundary in Test-Time Compute.pdf`): Discovers the "commitment boundary" in test-time compute: models fix their final answer within early latent activations (the first 10-20% of reasoning tokens), rendering downstream tokens causally epiphenomenal.
- **Kwon et al. (2025)** (`[Kwon et al. 2025] Controlling Thinking Budget in Reasoning Models.pdf`): Develops token-budget-aware reasoning controls, proving that unconstrained thinking generates bloated, repetitive loops without improving decision quality.

---

### Tier 3: Domain-Specific Clusters

The remaining 32 papers organized by research subfield, providing deep empirical context for creative writing, compute scaling, faithfulness, sampling dynamics, and persona modeling:

#### A. Creative Writing & Narrative under CoT
- **Zhao et al. (2026)** (`R2-Write`): Shows mainstream reasoning models yield limited gains in open-ended creative writing due to a lack of genuine reflection/revision dynamics.
- **Yang et al. (2026)** (`Language Models are Lazy Poets`): Documents large-scale creative mode collapse where aligned LLMs default to narrow, clichéd tropes in creative text generation.
- **Park et al. (2026)** (`Sampling More, Getting Less`): Demonstrates that extended reasoning restricts token calibration and collapses lexical diversity during repeated sampling.
- **Li et al. (2025)** (`Learning to Reason for Long-Form Story Generation`): Evaluates hierarchical CoT in story writing, noting that reasoning models over-structure narratives and flatten prose voice.
- **Zhou et al. (2025)** (`Can LLMs Generate Good Stories`): Evaluates narrative planning trade-offs, showing that algorithmic planning frequently constrains organic character emergence.
- **Gao et al. (2025)** (`Plan Before You Write`): Google Research framework evaluating hierarchical planning and reflection in structured creative prose.

#### B. Overthinking & Test-Time Compute Scaling
- **Huang et al. (2025)** (`Does Thinking More Always Help`): Reveals the mirage of test-time scaling on non-math, subjective, and creative tasks.
- **Lin et al. (2025)** (`Inverse Scaling in Test-Time Compute`): TMLR study identifying tasks where adding test-time compute monotonically degrades task performance.
- **Guo et al. (2025)** (`Underthinking and Overthinking in Reasoning`): Characterizes the empirical length-vs-correctness trade-off across reasoning model scales.
- **Ding et al. (2026)** (`When More Thinking Hurts`): Proves that overthinking introduces semantic drift and counterproductive revision loops beyond an optimal token threshold.
- **Tan et al. (2025)** (`LLMThinkBench`): Systematically benchmarks overthinking on basic tasks, showing models invent non-existent complexities when prompted to think.

#### C. Faithfulness & Post-Hoc Rationalization
- **Jiang et al. (2025)** (`Are DeepSeek R1 and Other Reasoning Models More Faithful`): Demonstrates that modern reasoning models (R1) remain as unfaithful and prone to post-hoc justification as earlier LLMs.
- **Venkatesh et al. (2025)** (`Analysing Chain of Thought Dynamics`): EMNLP study showing that decisions are encoded in hidden states early, leaving surface CoT as unfaithful post-hoc rationalization.
- **Prasad et al. (2026)** (`Are Stated Reasoning Steps Causally Load-Bearing`): Uses causal activation patching to prove that intermediate reasoning tokens carry near-zero causal weight.
- **Sharma et al. (2026)** (`Sycophancy Mitigated Yet Masked in Reasoning Models`): Shows reasoning models can shift social stances while generating traces that mask their true internal orientation.

#### D. Sampling, Diversity & Language of Thought
- **Mao et al. (2026)** (`Instruction-Tuned Models Cannot Sample from Distributions They Describe`): Proves that autoregressive LLMs cannot faithfully draw from empirical probability distributions, skewing repeated-sampling statistics.
- **Srinivasan et al. (2026)** (`Are You Sure Youre Sure`): Shows instruction tuning inflates confidence while severely reducing lexical diversity.
- **Xu et al. (2026)** (`Language of Thought Shapes Output Diversity`): Shows the internal language of reasoning tokens (e.g. English vs. target language) dictates output diversity and conceptual framing.
- **Lee et al. (2024)** (`TypedThinker`): ICLR paper proposing typed thought steps to combat mode collapse and reasoning path redundancy.
- **Chen et al. (2025)** (`MixReasoning`): Proposes dynamic mode-switching between fast execution and deliberate CoT to eliminate overthinking costs.
- **Zhou et al. (2025)** (`OThink-R1`): Optimizes test-time compute by implementing intrinsic fast/slow thinking switching.
- **Sun et al. (2025)** (`Reasoning Path Divergence`): Introduces formal divergence metrics showing repeated CoT rollouts rapidly converge onto identical trajectories.

#### E. Specialized Persona, Gender & Character Monologue
- **Wang et al. (2025)** (`Thinking in Character`): NeurIPS study detailing how default CoT degrades persona consistency and proposing role-aware reasoning.
- **Zhang et al. (2024)** (`Thinking Before Speaking`): Explores persona maintenance in dialogue, showing standard CoT breaks character voice without explicit mindset conditioning.
- **Liu et al. (2025)** (`R-CHAR`): EMNLP framework addressing character persona erosion during extended reasoning.
- **Rao et al. (2025)** (`Agree to Disagree - Meta-Evaluation of LLM Misgendering`): COLM study analyzing pronoun instability and non-binary pronoun handling under complex prompting.
- **Kocijan et al. (2019)** (`Default Reasoning in Number and Gender Assignment`): CoNLL study uncovering neural network default attractor states in grammatical gender and pronoun assignment.
- **Wang et al. (2025)** (`Demystifying Hybrid Thinking`): Demonstrates that hybrid models experience reasoning leakage into no-think modes, with significant reasoning tokens persisting.
- **Liu et al. (2026)** (`How Reasoning Shapes Social Bias in Generated Code`): Shows that reasoning tokens reshape social bias distributions in generated computational artifacts.
- **Wu et al. (2025)** (`Do Biased Models Have Biased Thoughts`): COLM study proving that biased models maintain heavily biased latent thoughts even when output text is nominally scrubbed.
- **Tam et al. (2024)** (`Evaluating Gender Bias via Chain-of-Thought Reasoning`): Demonstrates that CoT prompting fails to mitigate gender stereotypes across professional domains.
- **Kim et al. (2026)** (`Entertain, then Cancel`): Demonstrates that thinking mode alters the pragmatic interpretation of underspecified words, shifting natural language toward hyper-literal readings.

---

## 2. Thematic Overlap & Cluster Map

### Cluster Overview & High-Level Taxonomy

The 45 papers fall into five major conceptual clusters that map directly onto our experimental methodology and design decisions:

```mermaid
flowchart TD
    subgraph Cluster1 ["1. Bias Dynamics & Mode Comparisons (7 papers)"]
        Zheng2026["Zheng et al. 2026 (Does Thinking Help Fairness?)"]
        Shaikh2023["Shaikh et al. 2023 (Let's Not Think Step by Step)"]
        Li2025["Li et al. 2025 (Does Reasoning Introduce Bias?)"]
        Zhang2026["Zhang et al. 2026 (Mechanics of Bias & Reasoning)"]
        Liu2026["Liu et al. 2026 (Bias in Generated Code)"]
        Wu2025["Wu et al. 2025 (Do Biased Models Have Biased Thoughts?)"]
        Tam2024["Tam et al. 2024 (Gender Bias via CoT)"]
    end

    subgraph Cluster2 ["2. Faithfulness, Latent Reasoning & Commitment (8 papers)"]
        Lanham2023["Lanham et al. 2023 (Measuring Faithfulness in CoT)"]
        Bansal2026["Bansal et al. 2026 (Legibility is Not Interpretability)"]
        McKenna2026["McKenna et al. 2026 (LLM Reasoning Is Latent)"]
        Dua2026["Dua et al. 2026 (Commitment Boundary)"]
        Prasad2026["Prasad et al. 2026 (Are Stated Steps Causally Load-Bearing?)"]
        Venkatesh2025["Venkatesh et al. 2025 (CoT Dynamics & Information Flow)"]
        Jiang2025["Jiang et al. 2025 (Are R1 Models More Faithful?)"]
        Sharma2026["Sharma et al. 2026 (Sycophancy Mitigated Yet Masked)"]
    end

    subgraph Cluster3 ["3. Overthinking, Inertia & Compute Budgeting (10 papers)"]
        Gao2026["Gao et al. 2026 (Thinking Inertia in Reasoning Models)"]
        Song2026["Song et al. 2026 (Entropy Collapse in Long CoT)"]
        Ding2026["Ding et al. 2026 (When More Thinking Hurts)"]
        Huang2025["Huang et al. 2025 (Does Thinking More Always Help?)"]
        Lin2025["Lin et al. 2025 (Inverse Scaling in Test-Time Compute)"]
        Guo2025["Guo et al. 2025 (Underthinking & Overthinking)"]
        Tan2025["Tan et al. 2025 (LLMThinkBench Overthinking)"]
        Kwon2025["Kwon et al. 2025 (Token-Budget-Aware Reasoning)"]
        Wang2025D["Wang et al. 2025 (Demystifying Hybrid Thinking)"]
        Zhou2025O["Zhou et al. 2025 (OThink-R1 Mode Switching)"]
    end

    subgraph Cluster4 ["4. Persona, Creative Writing & Narrative Voice (9 papers)"]
        Tu2025["Tu et al. 2025 (Reasoning Does Not Improve Role-Play)"]
        Wang2025T["Wang et al. 2025 (Thinking in Character)"]
        Zhang2024["Zhang et al. 2024 (Thinking Before Speaking)"]
        Liu2025["Liu et al. 2025 (R-CHAR Metacognition Framework)"]
        Zhao2026["Zhao et al. 2026 (R2-Write Creative Reasoning)"]
        Yang2026["Yang et al. 2026 (Language Models are Lazy Poets)"]
        Li2025S["Li et al. 2025 (Reasoning for Long-Form Story Generation)"]
        Zhou2025S["Zhou et al. 2025 (Can LLMs Generate Good Stories?)"]
        Gao2025["Gao et al. 2025 (Plan Before You Write)"]
    end

    subgraph Cluster5 ["5. Sampling, Calibration & Language of Thought (11 papers)"]
        Park2026["Park et al. 2026 (Sampling More, Getting Less)"]
        Mao2026["Mao et al. 2026 (Cannot Sample from Target Distributions)"]
        Srinivasan2026["Srinivasan et al. 2026 (Instruction Tuning & Diversity)"]
        Xu2026["Xu et al. 2026 (Language of Thought Shapes Diversity)"]
        Sun2025["Sun et al. 2025 (Reasoning Path Divergence)"]
        Lee2024["Lee et al. 2024 (TypedThinker Structural Planning)"]
        Chen2025["Chen et al. 2025 (MixReasoning Switching Modes)"]
        Chen2026["Chen et al. 2026 (Thinking vs NoThinking SAE)"]
        Rao2025["Rao et al. 2025 (Meta-Evaluation of Misgendering)"]
        Kim2026["Kim et al. 2026 (Entertain, then Cancel)"]
        Kocijan2019["Kocijan et al. 2019 (Default Reasoning in Gender)"]
    end
```

---

### Replication Matrix 1: Distributional Concentration & Entropy Collapse

A primary empirical discovery across the recent literature is that **reasoning mode and test-time compute concentrate output distributions**, narrowing vocabulary entropy and driving mode collapse. This directly impacts our character study's repeated-draw methodology:

| Paper | Model Evaluated | Task Domain | Key Finding / Replication on Distributional Concentration |
|---|---|---|---|
| **Song et al. (2026)** | DeepSeek-R1, Qwen2.5-Math | Mathematical & Logical CoT | Global entropy collapse is triggered by narrow transition tokens ("Wait", "Alternatively"), collapsing subsequent vocabulary entropy across trajectories. |
| **Yang et al. (2026)** | Qwen3, Llama-3.1, GPT-OSS | Creative Poetry Generation | "Lazy Poets" phenomenon: aligned and reasoning models suffer massive mode collapse, repeatedly sampling identical lexical clusters and tropes. |
| **Park et al. (2026)** | GPT-4o, GPT-5.5, DeepSeek-R1 | Multi-step Problem Solving | "Sampling More, Getting Less": over-calibration in reasoning models acts as a diversity bottleneck; scaling sample draws yields rapidly diminishing unique variants. |
| **Gao et al. (2026)** | Qwen3-235B, DeepSeek-V4 | Open-Ended Prose & Decision | Thinking inertia causes models to fall into repetitive self-critique loops, sharply collapsing lexical entropy on open-ended prompts. |
| **Xu et al. (2026)** | Qwen-2.5, DeepSeek-R1 | Multilingual Generation | The internal "Language of Thought" restricts generative paths; non-native reasoning paths collapse lexical richness compared to native decoding. |
| **Srinivasan et al. (2026)** | Mistral-7B, Llama-3-8B | Open-Ended Generation | Post-training alignment shrinks lexical diversity while artificially boosting token probability confidence. |

---

### Replication Matrix 2: Bias Persistence & Counterfactual Flips

The naive expectation that extended reasoning allows models to "think through" and eliminate demographic bias is overwhelmingly refuted across seven independent studies:

| Paper | Model Evaluated | Bias / Fairness Benchmark | Direction & Finding on Demographic / Gender Bias |
|---|---|---|---|
| **Zheng et al. (2026)** | Qwen3-32B, QwQ-32B, DeepSeek-R1-Distill-32B | Adult, COMPAS, Credit (Counterfactuals) | **Asymmetric Bias Generation:** Thinking resolves some non-thinking flips but creates ~5× more new counterfactual flips; bias amplifies along thinking depth. |
| **Shaikh et al. (2023)** | GPT-3 (davinci-002/003) | BBQ, StereoSet, Toxicity Prompts | **Amplification:** Zero-shot CoT significantly increases harmful outputs and demographic stereotypes; harmful CoTs increase with model parameter scale. |
| **Li et al. (2025)** | DeepSeek-R1, Qwen-2.5-72B | StereoSet, CrowS-Pairs, BBQ | **Trace Bias:** Reasoning traces become active sites of bias; models generate elaborate rationalizations for biased choices that standard models make silently. |
| **Zhang et al. (2026)** | Llama-3-70B, Mistral-7B, Gemma-7B | WinoGender, Bias-in-Bios | **Superficial Debiasing:** CoT does not reduce the underlying bias gap; gender bias remains deeply entrenched in hidden layers across all depths. |
| **Liu et al. (2026)** | Qwen3-32B, DeepSeek-R1 | Code Generation (Demographics) | **Domain Bleed:** Reasoning tokens induce structured rationalizations that systematically skew demographic variable assignment in generated artifacts. |
| **Wu et al. (2025)** | Llama-3, Mistral, GPT-4 | Social Decision Benchmarks | **Biased Thoughts:** Safety tuning merely masks output text while internal reasoning traces remain heavily saturated with social and demographic bias. |
| **Tam et al. (2024)** | GPT-4, LLaMA-2-70B | Occupational Association | **Persistence:** CoT prompting completely fails to neutralize occupational gender stereotypes despite explicit fairness prompts. |

---

### Replication Matrix 3: Faithfulness, Epiphenomenal Traces & Latent Commitment

Eight studies definitively prove that visible chain-of-thought traces are not causal accounts of model decisions, explaining why our pilot observed high pronoun shifts without trace discussion:

| Paper | Method / Intervention | Target Architecture | Conclusion on Trace Causal Faithfulness |
|---|---|---|---|
| **Lanham et al. (2023)** | Trace Truncation, Step Corruption, Filler Insertion | Claude 1.3, PaLM 540B | Reasoning traces are frequently unfaithful post-hoc rationalizations; prompt bias dictates output prior to explicit reasoning. |
| **Dua et al. (2026)** | Probing Latent States along Test-Time Compute | DeepSeek-R1, QwQ-32B | **Commitment Boundary:** The model's decision commits within early latent states (first 10-20% of tokens); downstream reasoning tokens are epiphenomenal. |
| **Prasad et al. (2026)** | Step-Level Activation Patching | Qwen3-1.7B, Qwen3-4B | Intermediate reasoning steps carry near-zero causal weight; patching internal reasoning activations fails to perturb the final answer. |
| **McKenna et al. (2026)** | Latent Space Decoding vs. Surface Tokens | Llama-3, Qwen-2.5 | True reasoning is executed in continuous latent activation space; surface CoT tokens are downstream decodings that misrepresent internal dynamics. |
| **Bansal et al. (2026)** | Perturbation Analysis vs. Human/LLM Evaluation | Llama-3-70B, Qwen2.5-72B | **Legibility != Interpretability:** Judged step importance correlates poorly with causal importance; legible steps disguise non-transparent heuristics. |
| **Venkatesh et al. (2025)** | Layer-wise Information Probing | DeepSeek-R1, Qwen2.5-32B | Final outcomes are already encoded in hidden states early in generation; visible tokens function as post-hoc communicative rationalizations. |
| **Jiang et al. (2025)** | Causal Intervention on R1 Models | DeepSeek-R1, Claude 3.5 | Recent reasoning-optimized models show no improvement in causal faithfulness over standard instruction models; rationalization remains ubiquitous. |
| **Sharma et al. (2026)** | Stance Masking & Sycophancy Probing | DeepSeek-R1, QwQ-32B | Models internally alter decisions while generating extended chains of thought that actively disguise their stance shift. |

---

## 3. Rhetorical Ammunition & Citation Matrix for Q4

This section provides quote-ready methodological justifications and empirical defenses tailored directly for the pre-registration (`character/docs/prereg_v1.md`) and the experimental write-up.

### 1. Justifying the Character Generation Paradigm for Reasoning Tests
> **Reviewer Objection:** *"Why test reasoning mode on open-ended character generation rather than on standard logical, mathematical, or decision-making benchmarks?"*

```markdown
### Methodological Justification: Open-Ended Generation as the Ultimate Test of Mode Separation and Bias

We evaluate thinking mode on open-ended character generation because recent literature reveals that open-ended tasks constitute the exact domain where reasoning models exhibit their most severe behavioral failure modes:

1. **Reasoning Leakage Maximizes in Open-Ended Answer Spaces:** Gao et al. (2026, NeurIPS) demonstrate through systematic prompting interventions across six LLMs that thinking inertia and reasoning leakage into non-thinking modes become dramatically "more prevalent as the answer space opens." While Boolean and multiple-choice benchmarks constrain output trajectories, open-ended generation (such as character writing) exposes the true extent of mode divergence and incomplete mode separation (Wang et al., 2025).
2. **Documented Degradation of Persona and Creative Tone:** Unlike mathematical tasks where test-time compute provides clear verifiable gains, reasoning models suffer documented degradation in role-playing, persona adoption, and creative writing. Tu et al. (2025, Findings of ACL) evaluate 24 LLMs across six benchmarks and prove that reasoning ability disrupts role-playing scaling laws through "attention diversion" and "linguistic style drift." Similarly, Zhao et al. (2026) report that reasoning architectures achieve "limited gains on open-ended writing tasks" and lack genuine creative reflection patterns. Evaluating character personification directly tests whether reasoning-induced style drift alters linguistic demographic associations.
3. **Counterfactual Asymmetry Precedent in Qwen3:** In the closest direct precedent to our study, Zheng et al. (2026, Findings of EMNLP) conduct a within-model thinking ablation on Qwen3-32B and show that reasoning tokens resolve some baseline demographic biases while generating ~5× more new counterfactual flips. Testing Qwen3-14B on character personification evaluates whether this within-model counterfactual asymmetry generalizes from high-stakes classification to naturalistic narrative generation.
```

---

### 2. Explaining Why Thinking Traces Omit Gender Despite Shifted Outputs
> **Reviewer Objection:** *"Your pilot shows that thinking mode alters the generated character's gender distribution (e.g., shifting Qwen3-8B toward 'she' and Qwen3-14B toward 'they'), yet the thinking traces almost never discuss gender (≤ 0.5% of cases). Does this prove the gender shift is an unreasoned artifact or noise?"*

```markdown
### Empirical Defense: The Epiphenomenal Nature of Chain-of-Thought Traces

The near-complete absence of explicit gender deliberation within thinking traces (observed in ≤ 0.5% of pilot reasoning blocks) is not evidence that the gender choice was arbitrary or uncomputed. Rather, it is the exact expected signature established by the reasoning faithfulness literature:

1. **Stated Reasoning Is Post-Hoc Rationalization:** In their landmark causal intervention study, Lanham et al. (2023) prove that chain-of-thought traces are frequently unfaithful rationalizations. When models are influenced by latent demographic or contextual features, the final choice is committed internally, while the generated reasoning tokens merely fabricate plausible task justifications without mentioning the true causal driver.
2. **Legibility Does Not Equal Causal Interpretability:** Bansal et al. (2026, COLM) establish that human-legible reasoning traces correlate poorly with actual causal importance. A trace that discusses narrative pacing or physical descriptions while omitting demographic category assignment is functioning as communicative justification rather than a trace of internal computational state transitions.
3. **Reasoning Operates in Latent Activation Space:** McKenna et al. (2026) demonstrate that transformer reasoning is fundamentally latent; surface tokens are lossy projections that fail to capture continuous internal feature manipulations. The grammatical gender of the inanimate referent exerts its influence through latent activation steering (Chen et al., 2026) before or during early token drafting, bypassing explicit verbalized debate.
```

---

### 3. Grounding Early Gender Commitment in Reasoning Blocks
> **Reviewer Objection:** *"How do you interpret the observation that models like Gemma 4 or Qwen3 draft the character paragraph inside their thinking block and fix the character's gender in the very first sentence?"*

```markdown
### Theoretical Grounding: The Test-Time Commitment Boundary

Our empirical observation that character gender is fixed in the opening draft tokens of the reasoning block is directly explained by the "Commitment Boundary" discovered by Dua et al. (2026):

1. **Early Latent Convergence:** Dua et al. (2026) probe hidden representations throughout test-time compute sequences in DeepSeek-R1 and QwQ architectures, demonstrating that categorical choices reach a decisive commitment boundary within the first 10% to 20% of the reasoning budget. Once this boundary is crossed, the final output decision is mathematically fixed, rendering all subsequent reasoning tokens causally epiphenomenal.
2. **Thinking Inertia and Drafting Primacy:** Gao et al. (2026) document that when reasoning models face generative or descriptive prompts, "thinking inertia" forces them to begin drafting candidate outputs immediately within the scratchpad. Because grammatical agreement in English requires an immediate pronoun or gendered referent upon introducing an animate agent, the model commits to gender at the syntactic initialization of the draft.
```

---

### 4. Justifying Sampling Parameter Divergence Across Modes
> **Reviewer Objection:** *"In Q4, your thinking-on arm uses Temperature=0.6, TopP=0.95, while thinking-off uses Temperature=0.7, TopP=0.8. Why do you not keep decoding hyperparameters identical across arms?"*

```markdown
### Methodological Justification: Preserving Manufacturer-Specified Inference Regimes

We intentionally diverge sampling hyperparameters across thinking-on and thinking-off modes in strict adherence to official architecture specifications and empirical calibration literature:

1. **Manufacturer-Mandated Operational Baselines:** The Qwen technical documentation and official model cards (QwenLM, 2025; Alibaba, 2025) explicitly prescribe divergent decoding configurations:
   - *Thinking Mode:* `Temperature = 0.6, Top-P = 0.95, Top-K = 20`
   - *Non-Thinking Mode:* `Temperature = 0.7, Top-P = 0.80, Top-K = 20`
   The documentation explicitly warns against cross-applying parameters: applying non-thinking parameters to thinking mode induces severe looping and reasoning degradation, while using thinking parameters in non-thinking mode elevates uncalibrated tail tokens.
2. **Mitigating Entropy and Diversity Collapse:** Park et al. (2026) demonstrate that reasoning models suffer from an inherent "diversity bottleneck," where extended CoT naturally depresses token entropy. To prevent catastrophic entropy collapse (Song et al., 2026), thinking mode requires a wider nucleus (`TopP=0.95`) to allow exploration across reasoning paths (Sun et al., 2025). Conversely, standard instruction models operate on calibrated distributions where `TopP=0.80` prevents off-target sampling.
3. **Reporting Caveat (Mode-and-Settings Contrast):** In accordance with our pre-registration (`character/docs/prereg_v1.md`), we formally report Q4 as an evaluation of the **complete operational instrument** (thinking-mode-and-settings) rather than an isolated token-level ablation, reflecting how practitioners deploy these systems in practice.
```

---

### 5. Justifying the Reasoning-Token Budget Cap
> **Reviewer Objection:** *"Why do you enforce an explicit token budget cap on Qwen3-14B thinking tokens rather than allowing the model to generate test-time compute dynamically?"*

```markdown
### Computational & Theoretical Justification: Avoiding the Overthinking Degradation Zone

We enforce a strict thinking token budget cap during GPU pod execution based on extensive empirical evidence documenting inverse scaling and overthinking degradation:

1. **Diminishing Returns and Inverse Scaling:** Lin et al. (2025, TMLR) establish that test-time compute exhibits inverse scaling on tasks lacking verifiable objective criteria: allowing reasoning tokens to scale freely causes models to drift away from prompt constraints and invent spurious narrative complications. Ding et al. (2026) similarly prove that excessive thinking tokens induce semantic drift and degrade generative coherence along an inverted-U performance curve.
2. **Budget-Aware Inference Control:** Kwon et al. (2025, Findings of ACL) demonstrate that unconstrained reasoning models suffer from compute bloat without accuracy improvements, and that capping reasoning tokens via decode-time budgets preserves full analytical capacity while eliminating degenerate looping (Tan et al., 2025).
3. **Execution Cost and Latency Containment:** As detailed in our character design specification (`docs/explainers/C1-character-design-and-measures.md` §7), unconstrained thinking models generate up to 4,000+ hidden reasoning tokens per prompt. In an experimental design requiring 30 draws across 191 nouns in multiple languages, uncapped reasoning would produce compute explosion without scientific benefit.
```

---

## 4. Complete Annotated Inventory of All 45 Papers

The following catalog provides comprehensive bibliographic details, evaluated models, empirical findings, and project-specific relevance for all 45 papers in this collection, indexed alphabetically by citation tag.

```
+----------------------------------------------------------------------------------------------------+
|                                    ALPHABETICAL INVENTORY INDEX                                    |
|  01. Bansal et al. (2026)      12. Kocijan et al. (2019)     23. Park et al. (2026)       34. Venkatesh et al. (2025) |
|  02. Chen et al. (2025)        13. Kwon et al. (2025)        24. Prasad et al. (2026)     35. Wang et al. (2025, Demyst.) |
|  03. Chen et al. (2026)        14. Lanham et al. (2023)      25. Rao et al. (2025)        36. Wang et al. (2025, Char.) |
|  04. Ding et al. (2026)        15. Lee et al. (2024)         26. Shaikh et al. (2023)     37. Wu et al. (2025) |
|  05. Dua et al. (2026)         16. Li et al. (2025, Bias)    27. Sharma et al. (2026)     38. Xu et al. (2026) |
|  06. Gao et al. (2025)         17. Li et al. (2025, Story)   28. Song et al. (2026)       39. Yang et al. (2026) |
|  07. Gao et al. (2026)         18. Lin et al. (2025)         29. Srinivasan et al. (2026) 40. Zhang et al. (2024) |
|  08. Guo et al. (2025)         19. Liu et al. (2025)         30. Sun et al. (2025)        41. Zhang et al. (2026) |
|  09. Huang et al. (2025)       20. Liu et al. (2026)         31. Tam et al. (2024)        42. Zhao et al. (2026) |
|  10. Jiang et al. (2025)       21. Mao et al. (2026)         32. Tan et al. (2025)        43. Zheng et al. (2026) |
|  11. Kim et al. (2026)         22. McKenna et al. (2026)     33. Tu et al. (2025)         44. Zhou et al. (2025, Story) |
|                                                                                           45. Zhou et al. (2025, OThink) |
+----------------------------------------------------------------------------------------------------+
```

---

### 01. Bansal et al. (2026)
- **File:** `[Bansal et al. 2026] Legibility is Not Interpretability.pdf`
- **Citation:** Kevin Du, Alexander Hoyle, Laura Ruis, Acyr Locatelli, Mrinmaya Sachan, Hritik Bansal et al. (2026). "Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-Of-Thought Reasoning." *Conference on Language Modeling (COLM 2026)* / *arXiv:2609.04194*.
- **Problem & Abstract Summary:** Investigates whether the human- or LLM-judged importance of individual reasoning steps corresponds to their actual causal necessity during token generation. The authors conduct step-level perturbation and causal erasure across diverse reasoning tasks.
- **Models Evaluated:** Llama-3-70B-Instruct, Mistral-Large, Qwen2.5-72B-Instruct.
- **Key Empirical Findings:** Judged step importance correlates very weakly ($r < 0.25$) with actual causal impact. Highly legible, seemingly crucial reasoning steps can frequently be corrupted or removed without altering the final output, proving that surface legibility is largely decoupled from internal computational mechanics.
- **Project Relevance:** Essential rhetorical ammunition against naive readings of thinking traces. Demonstrates that whether a model's CoT trace explicitly discusses a character's gender is completely uninformative regarding whether latent gender representations drove the decision.

---

### 02. Chen et al. (2025)
- **File:** `[Chen et al. 2025] MixReasoning - Multi-Modal Reasoning Steps.pdf`
- **Citation:** Haiquan Lu, Gongfan Fang, Xinyin Ma, Qi Li, Xinchao Wang (2025). "MixReasoning: Switching Modes to Think." *arXiv:2510.06052*.
- **Problem & Abstract Summary:** Addresses the computational inefficiency and overthinking associated with uniform long CoT generation. Proposes a dynamic framework that switches between fast direct decoding (intuitive execution) and slow multi-hop CoT reasoning.
- **Models Evaluated:** Qwen2.5-Math, Qwen3, DeepSeek-R1-Distill.
- **Key Empirical Findings:** Uniform CoT throughout entire sequences produces verbose, redundant traces on straightforward sub-tasks; dynamic mode switching preserves full problem-solving accuracy while reducing generated token length by 40% to 60%.
- **Project Relevance:** Supports our justification for thinking-budget control and illustrates the structural distinction between hybrid fast/slow execution within the Qwen architecture.

---

### 03. Chen et al. (2026)
- **File:** `[Chen et al. 2026] Thinking vs NoThinking - Sparse Autoencoders Analysis.pdf`
- **Citation:** Bo Cheng, Qiaolin Lu, Yi Chang (2026). "Thinking vs. NoThinking: Towards Interpreting Reasoning Mechanisms of Large Language Models via Sparse Autoencoders." *arXiv:2608.08168*.
- **Problem & Abstract Summary:** Uses Sparse Autoencoders (SAEs) to analyze and interpret the internal latent feature activations of reasoning models operating in thinking-on versus thinking-off modes.
- **Models Evaluated:** DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-14B, Qwen-2.5-7B.
- **Key Empirical Findings:** Uncovers dedicated "meta-reasoning" and "self-verification" feature circuits that activate exclusively in thinking mode. Critically, the study reveals that thinking mode actively suppresses open-ended associative and stylistic features that are standardly active during non-thinking decoding.
- **Project Relevance:** Provides direct mechanistic grounding for why Qwen3-14B exhibits internal representation shifts between thinking-on and thinking-off arms, and explains why thinking mode can suppress creative stylistic variety.

---

### 04. Ding et al. (2026)
- **File:** `[Ding et al. 2026] When More Thinking Hurts.pdf`
- **Citation:** Ding et al. (2026). "When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling." *arXiv:2604.10739*.
- **Problem & Abstract Summary:** Systematically investigates the degradation regime of test-time compute scaling, testing whether the marginal utility of additional reasoning tokens eventually turns negative.
- **Models Evaluated:** DeepSeek-R1, QwQ-32B, Claude 3.5 Sonnet.
- **Key Empirical Findings:** Task performance follows an inverted-U curve with respect to reasoning length. Beyond an optimal token budget, excessive thinking tokens induce severe semantic drift, hallucinated constraints, and pathological self-revision loops that overturn initially correct answers.
- **Project Relevance:** Direct empirical justification for enforcing a strict reasoning-token cap in our Q4 GPU pod execution, guarding against uncalibrated compute bloat and semantic drift.

---

### 05. Dua et al. (2026)
- **File:** `[Dua et al. 2026] Commitment Boundary in Test-Time Compute.pdf`
- **Citation:** Daniel Scalena, Sara Candussio, Luca Bortolussi, Elisabetta Fersini, Malvina Nissim, Gabriele Sarti [Dua et al.] (2026). "Beyond the Commitment Boundary: Probing Epiphenomenal Chain-of-Thought in Large Reasoning Models." *arXiv:2606.13603*.
- **Problem & Abstract Summary:** Employs linear probing and representation steering to discover the exact token boundary within a chain of thought where the model's final answer becomes causally determined.
- **Models Evaluated:** DeepSeek-R1, QwQ-32B, Llama-3-70B-CoT.
- **Key Empirical Findings:** Identifies an early "Commitment Boundary": models irrevocably encode their final output choice within the first 10% to 20% of generated reasoning tokens. Tokens generated after this boundary are functionally epiphenomenal, serving merely to verbalize a choice already locked into latent space.
- **Project Relevance:** Perfectly explains the pilot finding in Gemma 4 and Qwen3 where the character's gender is fixed in the opening draft sentence of the reasoning block. Directly refutes the assumption that choices emerge from extended deliberation.

---

### 06. Gao et al. (2025)
- **File:** `[Gao et al. 2025] Plan Before You Write.pdf`
- **Citation:** Yoad Lewenberg, Hila Sheftel, Yael Karov (2025). "Plan Before You Write: Improve LLM writing by planning." *Advances in Neural Information Processing Systems (NeurIPS 2025) Creative AI Track*.
- **Problem & Abstract Summary:** Proposes an explicit planning and reflection framework for open-ended creative writing, decomposing creative prose tasks into hierarchical sub-goals.
- **Models Evaluated:** PaLM 2, Gemini 1.0 Pro, GPT-4.
- **Key Empirical Findings:** Unguided creative generation tends to collapse onto generic narrative tropes and clichés; structured outline planning improves plot coherence but requires explicit human-aligned rubrics to prevent mechanical voice.
- **Project Relevance:** Part of the creative narrative cluster; informs our understanding of baseline narrative structures and character framing in open-ended prompt designs.

---

### 07. Gao et al. (2026)
- **File:** `[Gao et al. 2026] Thinking Inertia in Reasoning Models.pdf`
- **Citation:** Dianqiao Lei, Kevin Qinghong Lin, Pan Lu et al. (2026). "Thinking Inertia: LLMs Keep Thinking When Told Not To." *Advances in Neural Information Processing Systems (NeurIPS 2026)* / *arXiv:2610.11765*.
- **Problem & Abstract Summary:** Investigates "thinking inertia"—the persistent tendency of reasoning-tuned LLMs to continue executing reasoning steps even when explicitly instructed not to think or when thinking mode is toggled off.
- **Models Evaluated:** Qwen3-235B, DeepSeek-V4, Claude 3.5 Sonnet, GPT-4o.
- **Key Empirical Findings:** Standard no-think controls fail to suppress reasoning behavior. Crucially, this reasoning leakage becomes dramatically more pronounced as the answer space opens up: while Boolean tasks maintain reasonable mode separation, open-ended generative writing exhibits severe leakage, repetitive internal loops, and diversity collapse.
- **Project Relevance:** Essential Tier 1 Anchor. Proves that "thinking off" is an operational configuration rather than a pure absence of reasoning, and explains why open-ended character writing is the most vulnerable domain for mode leakage.

---

### 08. Guo et al. (2025)
- **File:** `[Guo et al. 2025] Underthinking and Overthinking in Reasoning.pdf`
- **Citation:** Jinyan Su, Jennifer Healey, Preslav Nakov, Claire Cardie [Guo et al.] (2025). "Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and Correctness in LLMs." *arXiv:2505.00127*.
- **Problem & Abstract Summary:** Analyzes the empirical relationship between generated reasoning chain length and solution correctness across varying problem complexities and model scales.
- **Models Evaluated:** DeepSeek-R1-Distill-Qwen (1.5B, 7B, 14B, 32B), QwQ-32B.
- **Key Empirical Findings:** Demonstrates that reasoning models frequently overthink simple prompts, spending disproportionate tokens without performance gains, while underthinking difficult problems; highlights the necessity of adaptive length constraints.
- **Project Relevance:** Provides empirical length-scaling curves across the exact Qwen distillation family examined in our project.

---

### 09. Huang et al. (2025)
- **File:** `[Huang et al. 2025] Does Thinking More Always Help.pdf`
- **Citation:** Huang et al. (2025). "Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models." *arXiv:2506.04210*.
- **Problem & Abstract Summary:** Critically evaluates the hypothesis that test-time compute scaling provides monotonic improvements across non-verifiable, subjective, and creative domains.
- **Models Evaluated:** DeepSeek-R1, QwQ-32B, o1-preview.
- **Key Empirical Findings:** While test-time compute scales effectively on formal logic and mathematics, it exhibits severe plateauing and negative scaling on subjective generation and stylistic nuance, often substituting pedantic hairsplitting for substantive quality.
- **Project Relevance:** Strongly supports our prior that thinking mode should not be assumed to "improve" character personification, framing Q4 as an open empirical test of distribution shifts.

---

### 10. Jiang et al. (2025)
- **File:** `[Jiang et al. 2025] Are Reasoning Models More Faithful.pdf`
- **Citation:** James Chua et al. (Truthful AI) (2025). "Are DeepSeek R1 and Other Reasoning Models More Faithful?" *arXiv:2501.08156*.
- **Problem & Abstract Summary:** Assesses whether reinforcement-learning-optimized reasoning models (such as DeepSeek-R1) generate more causally faithful reasoning chains than prior instruction-tuned models.
- **Models Evaluated:** DeepSeek-R1, DeepSeek-R1-Distill-Qwen-32B, Claude 3.5 Sonnet, GPT-4o.
- **Key Empirical Findings:** R1-style models remain fundamentally unfaithful. Despite producing highly detailed, self-correcting verbal traces, the models frequently fabricate justifications for outputs that were predetermined by prompt biases or internal attractors.
- **Project Relevance:** Core evidence demonstrating that DeepSeek and Qwen reasoning architectures do not resolve unfaithful generation dynamics.

---

### 11. Kim et al. (2026)
- **File:** `[Kim et al. 2026] Entertain, then Cancel - Pragmatic Interpretation.pdf`
- **Citation:** Zhizhuo Kou, Jian Yang, Peng et al. (2026). "Entertain, then Cancel: How Thinking Makes Reasoning Models Read Some and Or Literally." *arXiv:2610.21204*.
- **Problem & Abstract Summary:** Investigates how chain-of-thought generation impacts the pragmatic interpretation of natural language expressions, such as scalar implicatures ("some" meaning "not all") and inclusive disjunctions.
- **Models Evaluated:** Qwen3-8B, Qwen3.8-27B, DeepSeek-V4, GPT-5.
- **Key Empirical Findings:** Thinking mode systematically disrupts pragmatic inference: models explicitly "entertain" natural conversational pragmatic readings inside the scratchpad, only to "cancel" them in favor of rigid, hyper-literal interpretations in the final output.
- **Project Relevance:** The closest existing study demonstrating that thinking mode induces qualitative semantic and linguistic register shifts in standard textual interpretation.

---

### 12. Kocijan et al. (2019)
- **File:** `[Kocijan et al. 2019] Default Reasoning in Number and Gender Assignment.pdf`
- **Citation:** Vid Kocijan, Ernest Davis, Thomas Lukasiewicz et al. (2019). "Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment." *Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL 2019)*.
- **Problem & Abstract Summary:** Uses contextual decomposition to probe how neural language models assign grammatical gender and number to ambiguous or generic referents.
- **Models Evaluated:** BERT, RoBERTa, LSTM baselines.
- **Key Empirical Findings:** Uncovers strong hardcoded default attractor basins: neural architectures systematically default to masculine agreement for underspecified nouns unless strong, unambiguous contextual evidence actively overrides the default.
- **Project Relevance:** Provides foundational linguistic and computational grounding for the masculine baseline asymmetry observed across LLM character generation.

---

### 13. Kwon et al. (2025)
- **File:** `[Kwon et al. 2025] Controlling Thinking Budget in Reasoning Models.pdf`
- **Citation:** Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma [Kwon et al.] (2025). "Token-Budget-Aware LLM Reasoning." *Findings of the Association for Computational Linguistics (ACL 2025 Findings)*.
- **Problem & Abstract Summary:** Proposes methods for controlling and capping the thinking token budget during test-time compute scaling without sacrificing analytical performance.
- **Models Evaluated:** DeepSeek-R1-Distill-Llama-8B, Qwen-2.5-7B, GPT-4o-mini.
- **Key Empirical Findings:** Demonstrates that unconstrained reasoning models suffer severe token bloat; enforcing explicit budget caps preserves problem-solving fidelity while eliminating redundant reasoning loops.
- **Project Relevance:** Cited directly to justify our pre-registered thinking token cap during Qwen3-14B pod execution.

---

### 14. Lanham et al. (2023)
- **File:** `[Lanham et al. 2023] Measuring Faithfulness in Chain-of-Thought Reasoning.pdf`
- **Citation:** Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Dario Amodei et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning." *arXiv:2307.13702*.
- **Problem & Abstract Summary:** Landmark Anthropic study introducing causal perturbation methods (truncation, corruption, filler insertion) to quantify whether LLMs actually rely on their stated reasoning steps.
- **Models Evaluated:** Claude 1.3, Claude Instant, PaLM 540B.
- **Key Empirical Findings:** On diverse tasks, models are causally unfaithful: reasoning traces serve as post-hoc rationalizations rather than the true determinants of the final answer. Prompt biases dictate the choice prior to reasoning generation.
- **Project Relevance:** Tier 1 Anchor. Explains why thinking blocks rarely mention gender even while the output gender is determined.

---

### 15. Lee et al. (2024)
- **File:** `[Lee et al. 2024] TypedThinker - Structural Planning in Reasoning.pdf`
- **Citation:** Lee et al. (2024). "TypedThinker: Diversify Large Language Model Reasoning with Typed Thought." *International Conference on Learning Representations (ICLR 2025)* / *arXiv:2410.01952*.
- **Problem & Abstract Summary:** Addresses path redundancy and mode collapse during repeated CoT sampling by introducing typed, structured reasoning steps (e.g. divergent, deductive).
- **Models Evaluated:** LLaMA-3-8B, Mistral-7B, Qwen-7B, GPT-4o.
- **Key Empirical Findings:** Demonstrates that unconstrained repeated CoT sampling suffers from high path overlap and structural redundancy; enforcing structured thought types restores exploratory variance.
- **Project Relevance:** Methodological context for repeated-draw sampling distributions and reasoning diversity.

---

### 16. Li et al. (2025, Bias)
- **File:** `[Li et al. 2025] Does Reasoning Introduce Bias.pdf`
- **Citation:** Xuyang Wu, Li et al. (2025). "Does Reasoning Introduce Bias? A Study of Social Bias Evaluation and Mitigation in LLM Reasoning." *Findings of the Association for Computational Linguistics (EMNLP 2025 Findings)*.
- **Problem & Abstract Summary:** Presents the first systematic audit of social and demographic bias harbored directly within LLM-generated reasoning traces versus final task outputs.
- **Models Evaluated:** DeepSeek-R1, DeepSeek-R1-Distill-Llama-8B, Qwen-2.5-72B.
- **Key Empirical Findings:** Reasoning tokens actively introduce and amplify demographic biases; models frequently generate explicit stereotypes inside reasoning chains to justify biased decisions made silently in direct decoding.
- **Project Relevance:** Tier 2 Mechanistic Anchor. Proves that reasoning traces themselves are sites of demographic stereotyping rather than objective debiasing mechanisms.

---

### 17. Li et al. (2025, Story)
- **File:** `[Li et al. 2025] Learning to Reason for Long-Form Story Generation.pdf`
- **Citation:** Alexander Gurung, Miruna Clinciu et al. [Li et al.] (2025). "Learning to Reason for Long-Form Story Generation." *Conference on Language Modeling (COLM 2025)* / *arXiv:2503.22828*.
- **Problem & Abstract Summary:** Investigates the application of long CoT reasoning models and reinforcement learning to long-form creative story generation.
- **Models Evaluated:** DeepSeek-R1, Qwen-2.5-14B, Llama-3-70B.
- **Key Empirical Findings:** While reasoning improves high-level narrative plot consistency, it flattens stylistic richness and forces characters into predictable, stereotypical arcs.
- **Project Relevance:** Domain-specific narrative writing evidence: reinforces that reasoning mode alters character framing and creative voice.

---

### 18. Lin et al. (2025)
- **File:** `[Lin et al. 2025] Inverse Scaling in Test-Time Compute.pdf`
- **Citation:** Aryo Pradipta et al. [Lin et al.] (2025). "Inverse Scaling in Test-Time Compute." *Transactions on Machine Learning Research (TMLR 12/2025)* / *arXiv:2507.14417*.
- **Problem & Abstract Summary:** Identifies and analyzes task environments where allocating additional test-time compute (longer reasoning chains) leads to a monotonic decline in model performance.
- **Models Evaluated:** Qwen3, DeepSeek-R1, Claude 3.5 Sonnet.
- **Key Empirical Findings:** Discovers inverse test-time scaling on tasks involving intuitive heuristics, open-ended framing, and subtle stylistic cues, where extended thinking locks models into erroneous reasoning spirals.
- **Project Relevance:** Methodological justification for capping test-time compute and cautioning against assumptions that thinking mode will improve character balance.

---

### 19. Liu et al. (2025)
- **File:** `[Liu et al. 2025] R-CHAR - Reasoning Character Framework.pdf`
- **Citation:** Haiming Qin, Jiwei Zhang et al. [Liu et al.] (2025). "R-CHAR: A Metacognition-Driven Framework for Role-Playing in Large Language Models." *Proceedings of EMNLP 2025 (Main Conference)*.
- **Problem & Abstract Summary:** Examines why standard reasoning architectures experience severe persona erosion in interactive fiction and role-playing, proposing a metacognitive conditioning framework.
- **Models Evaluated:** DeepSeek-R1, Qwen-2.5-72B-Instruct.
- **Key Empirical Findings:** Standard reasoning traces naturally adopt an analytical, third-person assistant persona that dilutes character immersion and alters narrative voice.
- **Project Relevance:** Corroborates the persona degradation literature, confirming that default reasoning mechanisms systematically interfere with character immersion.

---

### 20. Liu et al. (2026)
- **File:** `[Liu et al. 2026] How Reasoning Shapes Social Bias in Generated Code.pdf`
- **Citation:** Weifeng Sun, Jieke Shi, Zhou Yang et al. [Liu et al.] (2026). "How Reasoning Shapes Social Bias in LLM-Generated Code?" *arXiv:2608.06829*.
- **Problem & Abstract Summary:** Evaluates the downstream impact of reasoning tokens on demographic, gender, and racial biases in generated code artifacts.
- **Models Evaluated:** Qwen3-32B, DeepSeek-R1, Llama-3.1-8B, Qwen2.5-Coder-7B.
- **Key Empirical Findings:** Reasoning mode systematically alters demographic distributions in generated outputs, often embedding stereotyped assumptions into downstream variables and structure.
- **Project Relevance:** Demonstrates that reasoning mode reshapes demographic bias in non-decision, creative generative tasks.

---

### 21. Mao et al. (2026)
- **File:** `[Mao et al. 2026] Instruction-Tuned Models Cannot Sample from Distributions They Describe.pdf`
- **Citation:** Chaemin Jang, Jihee Kim et al. [Mao et al.] (2026). "Instruction-Tuned Language Models Cannot Sample from Distributions They Can Describe." *arXiv:2607.25292*.
- **Problem & Abstract Summary:** Investigates whether autoregressive LLMs can faithfully sample from empirical probability distributions across repeated draws or whether repeated sampling violates distributional independence.
- **Models Evaluated:** Llama-3-8B/70B, Qwen-2.5-7B/72B, GPT-4o.
- **Key Empirical Findings:** Models fail to sample representatively from underlying response distributions; repeated draws suffer from severe mode pooling and attractor bias.
- **Project Relevance:** Essential caveat for the character study's repeated-draw methodology and the statistical interpretation of character gender proportions as true probabilities.

---

### 22. McKenna et al. (2026)
- **File:** `[McKenna et al. 2026] LLM Reasoning Is Latent.pdf`
- **Citation:** Wenshuo Wang et al. [McKenna et al.] (2026). "LLM Reasoning Is Latent, Not the Chain of Thought." *arXiv:2604.15726*.
- **Problem & Abstract Summary:** Tests whether transformer reasoning is fundamentally executed via discrete surface tokens or within continuous internal latent representations.
- **Models Evaluated:** Llama-3-8B, Mistral-7B, Qwen-2.5-14B.
- **Key Empirical Findings:** Proves that the primary computational transitions driving task outcomes take place in continuous latent layer space; generated CoT tokens are downstream decodings that often obscure internal dynamics.
- **Project Relevance:** Directly explains why grammatical gender steering and persona choices operate sub-cognitively in latent activations without surfacing in CoT text.

---

### 23. Park et al. (2026)
- **File:** `[Park et al. 2026] Sampling More, Getting Less - Diversity Collapse.pdf`
- **Citation:** Amin Banayeeanzade, Qingchuan Yang et al. [Park et al.] (2026). "Sampling More, Getting Less: Calibration is the Diversity Bottleneck in LLMs." *arXiv:2605.11128*.
- **Problem & Abstract Summary:** Investigates the diversity bottleneck in modern LLMs, analyzing why scaling sample draws yields diminishing unique outputs under standard sampling temperatures.
- **Models Evaluated:** GPT-4o, GPT-5.5, DeepSeek-R1, Llama-3.1.
- **Key Empirical Findings:** Reasoning models restrict output token entropy, causing repeated sample draws to collapse into narrow syntactic and semantic basins.
- **Project Relevance:** Cited to justify divergent nucleus sampling parameters across modes (`TopP=0.95` for thinking vs. `TopP=0.80` for non-thinking) and warns of variance shrinkage.

---

### 24. Prasad et al. (2026)
- **File:** `[Prasad et al. 2026] Are Stated Reasoning Steps Causally Load-Bearing.pdf`
- **Citation:** Archiki Prasad et al. (2026). "Are Stated Reasoning Steps Causally Load-Bearing? Activation Patching Says Less Than You Think." *arXiv:2609.27038*.
- **Problem & Abstract Summary:** Uses causal activation patching across individual reasoning tokens to evaluate whether generated CoT steps are causally necessary for reaching final answers.
- **Models Evaluated:** Qwen3-1.7B, Qwen3-4B, Llama-3-8B.
- **Key Empirical Findings:** Demonstrates that the vast majority of intermediate reasoning steps carry near-zero causal weight; patching or ablating intermediate activations leaves downstream outputs largely unaffected.
- **Project Relevance:** Provides rigorous causal interpretability evidence confirming that thinking traces are largely epiphenomenal, validating our focus on output gender rather than trace text.

---

### 25. Rao et al. (2025)
- **File:** `[Rao et al. 2025] Agree to Disagree - Meta-Evaluation of LLM Misgendering.pdf`
- **Citation:** Arjun Subramonian et al. [Rao et al.] (2025). "Agree to Disagree? A Meta-Evaluation of LLM Misgendering." *Conference on Language Modeling (COLM 2025)* / *arXiv:2504.17075*.
- **Problem & Abstract Summary:** Conducts a comprehensive meta-evaluation of how LLMs assign pronouns and navigate ambiguous or non-binary gender identities.
- **Models Evaluated:** GPT-4, Llama-3, Claude 3, Mistral.
- **Key Empirical Findings:** Models display severe instability and default bias when processing singular 'they' and gender-neutral referents, frequently misgendering characters or defaulting to binary pronouns.
- **Project Relevance:** Informs our analysis of overridden-*they* paragraphs and contextualizes why Qwen3-14B thinking mode displayed shifts toward *they* in the pilot.

---

### 26. Shaikh et al. (2023)
- **File:** `[Shaikh et al. 2023] Lets Not Think Step by Step.pdf`
- **Citation:** Omar Shaikh, Hongxin Zhang et al. (2023). "On Second Thought, Let's Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning." *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)*.
- **Problem & Abstract Summary:** Systematic evaluation of zero-shot CoT prompting across benchmark datasets measuring demographic bias, stereotyping, and toxicity.
- **Models Evaluated:** GPT-3 (text-davinci-002, text-davinci-003, InstructGPT).
- **Key Empirical Findings:** Zero-shot CoT significantly increases a model's propensity to generate biased and toxic content; harmful outputs scale positively with model size while decreasing with instruction following.
- **Project Relevance:** Foundational Tier 2 anchor establishing that reasoning mode cannot be assumed to reduce demographic or social bias.

---

### 27. Sharma et al. (2026)
- **File:** `[Sharma et al. 2026] Sycophancy Mitigated Yet Masked in Reasoning Models.pdf`
- **Citation:** Zhaoxin Feng, Zheng et al. [Sharma et al.] (2026). "Good Arguments Against the People Pleasers: How Reasoning Mitigates (Yet Masks) LLM Sycophancy." *arXiv:2603.16643*.
- **Problem & Abstract Summary:** Analyzes how test-time reasoning models navigate sycophantic pressure and user biases during complex dialogues.
- **Models Evaluated:** DeepSeek-R1, QwQ-32B, Claude 3.5 Sonnet.
- **Key Empirical Findings:** While reasoning enables models to resist superficial sycophancy, models frequently generate verbose reasoning traces that mask or misrepresent their internal stance shifts.
- **Project Relevance:** Reinforces the principle that observable reasoning text does not reliably mirror internal model alignments.

---

### 28. Song et al. (2026)
- **File:** `[Song et al. 2026] Entropy Collapse in Long CoT Generation.pdf`
- **Citation:** Song et al. (2026). "A Few Bad Apples Spoil the Bunch: Preventing Global Entropy Collapse Driven by a Small Set of Tokens in LLM Reasoning." *Findings of the Association for Computational Linguistics (ACL 2026 Findings)*.
- **Problem & Abstract Summary:** Formally analyzes token-level entropy dynamics during extended chain-of-thought generation, identifying the mathematical mechanisms behind sudden diversity collapse.
- **Models Evaluated:** DeepSeek-R1, Qwen2.5-Math-7B, Llama-3-8B-Instruct.
- **Key Empirical Findings:** Extended reasoning sequences trigger global entropy collapse, driven by a narrow set of high-frequency transition tokens that permanently contract subsequent probability distributions.
- **Project Relevance:** Tier 1 Anchor. Explains why thinking mode narrows the variety of character voices and provides theoretical grounding for unequal variance between arms.

---

### 29. Srinivasan et al. (2026)
- **File:** `[Srinivasan et al. 2026] Are You Sure Youre Sure - Confidence and Lexical Diversity.pdf`
- **Citation:** Gaurav Srivastava et al. [Srinivasan et al.] (2026). "Are You Sure You're Sure? On the Impact of Instruction Tuning on Confidence and Lexical Diversity." *arXiv:2608.13430*.
- **Problem & Abstract Summary:** Investigates how post-training instruction alignment impacts model calibration, output confidence, and lexical diversity.
- **Models Evaluated:** Mistral-7B, Llama-3-8B, Qwen-2.5-7B.
- **Key Empirical Findings:** Instruction tuning inflates model confidence while sharply reducing lexical diversity across open-ended generation tasks.
- **Project Relevance:** Contextualizes lexical narrowing in aligned models, directly bearing on our Glasgow psycholinguistic norm measures.

---

### 30. Sun et al. (2025)
- **File:** `[Sun et al. 2025] Reasoning Path Divergence.pdf`
- **Citation:** Feng Ju, Zeyu Qin et al. [Sun et al.] (2025). "Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking." *arXiv:2510.26122*.
- **Problem & Abstract Summary:** Introduces formal divergence metrics to evaluate whether multiple sampled CoT trajectories explore genuinely diverse solution spaces.
- **Models Evaluated:** Qwen3-4B-Base, Llama-3-8B.
- **Key Empirical Findings:** Repeated CoT rollouts exhibit severe path collapse, rapidly converging onto identical topological reasoning trajectories unless explicit divergence curation is applied.
- **Project Relevance:** Informs repeated-sampling methodologies and provides mathematical metrics for reasoning diversity.

---

### 31. Tam et al. (2024)
- **File:** `[Tam et al. 2024] Evaluating Gender Bias via Chain-of-Thought Reasoning.pdf`
- **Citation:** Masahiro Kaneko, Danushka Bollegala [Tam et al.] (2024). "Evaluating Gender Bias in Large Language Models via Chain-of-Thought Prompting." *arXiv:2401.15585*.
- **Problem & Abstract Summary:** Evaluates gender bias across occupational and professional roles using explicit CoT prompts.
- **Models Evaluated:** GPT-3.5-Turbo, GPT-4, LLaMA-2-70B.
- **Key Empirical Findings:** CoT prompting fails to mitigate gender bias in occupational association tasks; gender stereotypes persist in reasoning chains even when models are instructed to be fair.
- **Project Relevance:** Confirms that CoT does not neutralize occupational or grammatical gender stereotypes.

---

### 32. Tan et al. (2025)
- **File:** `[Tan et al. 2025] LLMThinkBench - Benchmarking Overthinking.pdf`
- **Citation:** Gaurav Srivastava, Tan et al. (2025). "Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff in Language Models (LLMThinkBench)." *arXiv:2507.04023*.
- **Problem & Abstract Summary:** Establishes LLMThinkBench to systematically benchmark overthinking behavior on elementary versus complex reasoning tasks.
- **Models Evaluated:** GPT-5, DeepSeek-R1, QwQ-32B.
- **Key Empirical Findings:** Reasoning models frequently overthink basic prompts, generating thousands of superfluous reasoning tokens that introduce errors into previously trivial tasks.
- **Project Relevance:** Demonstrates the structural failure mode of overthinking on simple prompts (such as writing a short character paragraph).

---

### 33. Tu et al. (2025)
- **File:** `[Tu et al. 2025] Reasoning Does Not Necessarily Improve Role-Playing.pdf`
- **Citation:** Xiachong Feng, Longxu Dou, Lingpeng Kong [Tu et al.] (2025). "Reasoning Does Not Necessarily Improve Role-Playing Ability." *Findings of the Association for Computational Linguistics (ACL 2025 Findings)*: 10301–10314.
- **Problem & Abstract Summary:** Extensive evaluation of 24 LLMs across six role-playing benchmarks, investigating whether reasoning ability and CoT improve persona adoption.
- **Models Evaluated:** 24 LLMs including GPT-4-Turbo, DeepSeek-R1, Qwen-2.5, Claude 3.5 Sonnet.
- **Key Empirical Findings:** CoT systematically reduces role-playing performance; reasoning ability disrupts role-playing scaling laws via "attention diversion" and "linguistic style drift."
- **Project Relevance:** Tier 1 Anchor. Direct precedent for Finding 4b: proves that thinking mode degrades creative persona writing, providing directional expectations for Q4.

---

### 34. Venkatesh et al. (2025)
- **File:** `[Venkatesh et al. 2025] CoT Dynamics - Commitment and Information Flow.pdf`
- **Citation:** Samuel Lewis-Lim, Xingwei Tan et al. [Venkatesh et al.] (2025). "Analysing Chain of Thought Dynamics: Active Guidance or Unfaithful Post-hoc Rationalisation?" *Proceedings of EMNLP 2025 (Main Conference)*.
- **Problem & Abstract Summary:** Traces layer-wise information flow and decision commitment during CoT generation using diagnostic hidden-state probes.
- **Models Evaluated:** DeepSeek-R1, Qwen2.5-32B-Instruct, Llama-3-8B-Instruct.
- **Key Empirical Findings:** Outcomes are encoded in hidden states early in generation; visible tokens function as post-hoc communicative rationalizations rather than real-time deliberation.
- **Project Relevance:** Corroborates the early commitment boundary and latent decision-making findings.

---

### 35. Wang et al. (2025, Demystifying)
- **File:** `[Wang et al. 2025] Demystifying Hybrid Thinking.pdf`
- **Citation:** Shouren Wang, Wang Yang, Xia et al. (2025). "Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?" *arXiv:2510.12680*.
- **Problem & Abstract Summary:** Examines mode separation and reasoning leakage in hybrid models supporting both think and no-think modes (like Qwen3).
- **Models Evaluated:** Qwen3-8B, Qwen2.5-7B, DeepSeek-R1-Distill, GPT-oss.
- **Key Empirical Findings:** Current hybrid LLMs achieve only partial mode separation; reasoning behaviors frequently leak into the no-think mode, emitting proto-reasoning tokens and extended sequences unless rigorously suppressed.
- **Project Relevance:** Core methodological caveat for Q4: proves that "thinking off" is not a pure zero-reasoning baseline, requiring careful terminology ("thinking mode on/off").

---

### 36. Wang et al. (2025, Character)
- **File:** `[Wang et al. 2025] Thinking in Character.pdf`
- **Citation:** Yihong Tang, Kehai Chen, Muyun Yang et al. [Wang et al.] (2025). "Thinking in Character: Advancing Role-playing Agents with Role-Aware Reasoning." *Advances in Neural Information Processing Systems (NeurIPS 2025)*.
- **Problem & Abstract Summary:** Investigates role-playing degradation in reasoning models and designs role-aware CoT training to preserve persona consistency.
- **Models Evaluated:** DeepSeek-R1, GPT-4o, Llama-3-70B.
- **Key Empirical Findings:** Default reasoning models exhibit cognitive conflict between objective problem-solving and character persona adoption; without specialized alignment, models revert to assistant personas.
- **Project Relevance:** Corroborates Tu et al. on persona erosion and style drift during chain-of-thought generation.

---

### 37. Wu et al. (2025)
- **File:** `[Wu et al. 2025] Do Biased Models Have Biased Thoughts.pdf`
- **Citation:** Swati Rajwal, Shivank Garg et al. [Wu et al.] (2025). "Do Biased Models Have Biased Thoughts?" *Conference on Language Modeling (COLM 2025)* / *arXiv:2508.06671*.
- **Problem & Abstract Summary:** Compares social and demographic bias harbored within internal reasoning thoughts versus final output responses across decision benchmarks.
- **Models Evaluated:** Llama-3, Mistral, GPT-4-CoT.
- **Key Empirical Findings:** Biased models exhibit heavily biased internal thoughts; safety tuning frequently suppresses biased surface outputs while leaving underlying reasoning chains saturated with bias.
- **Project Relevance:** Demonstrates that latent bias survives even when surface text appears sanitized.

---

### 38. Xu et al. (2026)
- **File:** `[Xu et al. 2026] Language of Thought Shapes Output Diversity.pdf`
- **Citation:** Shaoyang Xu, Wenxuan Zhang (2026). "Language of Thought Shapes Output Diversity in Large Language Models." *arXiv:2601.11227*.
- **Problem & Abstract Summary:** Investigates how the internal language of reasoning tokens (e.g. English vs. prompt language) affects output diversity and conceptual framing.
- **Models Evaluated:** Qwen-2.5, DeepSeek-R1, Llama-3.
- **Key Empirical Findings:** The language used in thinking tokens occupies distinct latent subspaces, strongly steering the lexical diversity and cultural perspectives of downstream outputs.
- **Project Relevance:** Highly relevant for multilingual character generation: shows that internal thinking language interacts with grammatical gender concepts.

---

### 39. Yang et al. (2026)
- **File:** `[Yang et al. 2026] Language Models are Lazy Poets.pdf`
- **Citation:** Jacob A. Matthews, Sil Hamilton [Yang et al.] (2026). "Language Models are Lazy Poets: Creative Mode Collapse at Scale." *arXiv:2610.24223*.
- **Problem & Abstract Summary:** Large-scale empirical study of creative mode collapse in aligned and reasoning LLMs across poetry and creative literature.
- **Models Evaluated:** Qwen3, Llama-3.1, GPT-OSS.
- **Key Empirical Findings:** Aligned and reasoning models suffer from severe creative mode collapse ("lazy poets"), repeatedly defaulting to high-probability tropes, clichéd metaphors, and narrow syntactic structures.
- **Project Relevance:** Directly connects to creative character generation and persona collapse under RL alignment.

---

### 40. Zhang et al. (2024)
- **File:** `[Zhang et al. 2024] Thinking Before Speaking.pdf`
- **Citation:** Baohua Zhang, Yongyi Huang, Wenyao Cui, Huaping Zhang (2024). "Thinking Before Speaking: A Role-playing Model with Mindset." *arXiv:2409.13752*.
- **Problem & Abstract Summary:** Proposes mindset reasoning mechanisms to preserve character voice in role-playing dialogue systems.
- **Models Evaluated:** LLaMA-2-13B, Baichuan-2-13B.
- **Key Empirical Findings:** Standard CoT without character grounding breaks persona; conditioning internal monologue on character mindset preserves consistency across long dialogue turns.
- **Project Relevance:** Persona literature cluster: demonstrates the difficulty of maintaining character voice during multi-step reasoning.

---

### 41. Zhang et al. (2026)
- **File:** `[Zhang et al. 2026] Mechanics of Bias and Reasoning.pdf`
- **Citation:** Edie Pearman, Sophia Osborne, Mira Kandlikar-Bloch, Mina Arzaghi, Florian Carichon, Golnoosh Farnadi [Zhang et al.] (2026). "Mechanics of Bias and Reasoning: Interpreting the Impact of Chain-of-Thought Prompting on Gender Bias in LLMs." *arXiv:2605.20410*.
- **Problem & Abstract Summary:** Combines benchmark evaluation, internal representation probing, and failure analysis to interpret the impact of CoT on gender bias.
- **Models Evaluated:** Llama-3-8B/70B, Mistral-7B, Gemma-7B.
- **Key Empirical Findings:** CoT does not consistently reduce the gender-bias gap; gender bias remains deeply embedded in hidden representations across all layers, and surface-level reductions stem from dataset memorization rather than debiasing.
- **Project Relevance:** Tier 2 mechanistic anchor: proves gender bias remains entrenched in latent activations despite chain-of-thought generation.

---

### 42. Zhao et al. (2026)
- **File:** `[Zhao et al. 2026] R2-Write - Reasoning for Creative Writing.pdf`
- **Citation:** Wanlong Liu, Bo Zhang, Chenliang Li, Shaopeng Lai, Yuning Wu, Xuanyu Lei, Ming Yan [Zhao et al.] (2026). "R2-Write: Reflection and Revision for Open-Ended Writing with Deep Reasoning." *arXiv:2604.03004*.
- **Problem & Abstract Summary:** Evaluates whether deep reasoning models improve open-ended creative writing through reflection and revision.
- **Models Evaluated:** DeepSeek-R1, Qwen3-8B, Qwen3-30B-A3B-Instruct.
- **Key Empirical Findings:** Reasoning models achieve very limited gains on open-ended writing tasks; unlike mathematical reasoning, models lack deep reflection and revision patterns in creative prose, frequently defaulting to rigid formulas.
- **Project Relevance:** Sets the empirical prior for Q4: reasoning buys little improvement in open-ended creative tasks, making a null on pronoun distribution unsurprising.

---

### 43. Zheng et al. (2026)
- **File:** `[Zheng et al. 2026] Does Thinking Help Fairness.pdf`
- **Citation:** Deng Pan, Joe Germino, Yihong Ma, Elizabeth Daly et al. [Zheng et al.] (2026). "Does Thinking Help Fairness? Reasoning Tokens Resolve Some Biases but Create More." *Findings of the Association for Computational Linguistics (EMNLP 2026 Findings)* / *arXiv:2609.30768*.
- **Problem & Abstract Summary:** Conducts a within-model thinking-vs-non-thinking ablation across QwQ-32B, DeepSeek-R1-Distill-Qwen-32B, and Qwen3-32B on counterfactual demographic fairness benchmarks.
- **Models Evaluated:** Qwen3-32B, QwQ-32B, DeepSeek-R1-Distill-Qwen-32B.
- **Key Empirical Findings:** Thinking mode both resolves and creates counterfactual demographic flips, but created flips outnumber resolved flips by approximately 5× across all 9 model × dataset configurations; bias amplifies along thinking depth.
- **Project Relevance:** Primary Tier 1 anchor paper. Exact within-model ablation precedent in the Qwen3 family showing asymmetric bias generation under reasoning.

---

### 44. Zhou et al. (2025, Story)
- **File:** `[Zhou et al. 2025] Can LLMs Generate Good Stories.pdf`
- **Citation:** Yi Wang, Max Kreminski [Zhou et al.] (2025). "Can LLMs Generate Good Stories? Insights and Challenges from a Narrative Planning Perspective." *arXiv:2506.10161*.
- **Problem & Abstract Summary:** Evaluates narrative planning and story generation capabilities in modern LLMs from a computational narratology perspective.
- **Models Evaluated:** GPT-4, Claude 3 Opus, Llama-3-70B.
- **Key Empirical Findings:** LLMs struggle with global narrative planning; reasoning models tend to over-structure narrative arcs at the expense of character depth and organic development.
- **Project Relevance:** Creative writing cluster: highlights the tension between structured reasoning and character-driven organic prose.

---

### 45. Zhou et al. (2025, OThink)
- **File:** `[Zhou et al. 2025] OThink-R1 - Optimization in Reasoning.pdf`
- **Citation:** Shengjia Zhang, Junjie Wu, Jiawei Chen, Changwang Zhang, Zhe Li, Xingyu Lou, Wangchunshu Zhou et al. (2025). "OThink-R1: Intrinsic Fast/Slow Thinking Mode Switching for Over-Reasoning Mitigation." *arXiv:2506.02397*.
- **Problem & Abstract Summary:** Proposes dynamic fast/slow thinking mode switching to eliminate over-reasoning in R1-style architectures.
- **Models Evaluated:** Qwen3, DeepSeek-R1, GPT-4o.
- **Key Empirical Findings:** Mitigating over-reasoning via adaptive mode switching prevents cognitive bias amplification and eliminates 50%+ of redundant reasoning tokens.
- **Project Relevance:** Informs test-time compute optimization and mode separation in reasoning architectures.

---

*Catalog compiled and verified for Google NotebookLM and research workflows on 2026-10-30.*
