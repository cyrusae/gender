# Reading Guide & Triage: 07A CoT Faithfulness, Latent Dynamics & Trace Mechanics (NB5A)

This collection contains **22 research papers** investigating what actually happens inside language model residual streams and token distributions during test-time reasoning / Chain-of-Thought (CoT).

---

## 1. Guiding Theoretical Questions
- **Are Stated Steps Causally Load-Bearing?** Does the model rely on its visible reasoning tokens, or are outputs determined by early latent representations with CoT acting as post-hoc rationalization?
- **Commitment Boundaries:** At what token index in a reasoning trajectory does an LLM fix its final answer?
- **Mechanistic Legibility:** Do sparse autoencoders (SAEs) and probing reveal meaningful internal deliberation vs. superficial fluency?

---

## 2. Executive Triage & Priority Reading List

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: DEFINITIVE ANCHORS                      │
│  Lanham et al. (2023)  │  Prasad et al. (2026)  │  McKenna et al. (2026)│
│  [Trace Faithfulness]  │ [Causal Load-Bearing]  │  [Latent Reasoning]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    TIER 2: SPECIFIC DYNAMICS & PLANNING                │
│  Dua et al. (2026)     │  Venkatesh et al. (2025) │ Chen et al. (2026) │
│  [Commitment Boundary] │ [Information Flow Dynamics]│ [SAEs in Thinking]│
│  Gao et al. (2025)     │  Lee et al. (2024)       │ Wang et al. (2025) │
│  [Plan Before Write]   │ [TypedThinker Planning]  │ [Thinking Character]│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                   TIER 3: EVALUATION & BEHAVIORAL AUDITS               │
│  Bansal (2026)   │ Chen (2025)    │ Jiang (2025)   │ Kim (2026)        │
│  Li (2025)       │ Mao (2026)     │ Srinivasan (2026) │ Sun (2025)     │
│  Wang (2025)     │ Xu (2026)      │ Zhang (2024)   │ Zhao (2026)       │
│  Zhou (2025)     │                │                │                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Essential Anchors
1. **Lanham et al. (2023)** — *Measuring Faithfulness in Chain-of-Thought Reasoning*
   - Introduces truncation, filling, and perturbation tests to measure whether CoT reasoning causally drives model predictions. Shows that models often arrive at answers regardless of flawed reasoning steps.
2. **Prasad et al. (2026)** — *Are Stated Reasoning Steps Causally Load-Bearing?*
   - Uses activation patching and causal interventions to demonstrate that many intermediate CoT steps can be altered without changing the final output, proving significant unfaithful post-hoc rationalization.
3. **McKenna et al. (2026)** — *LLM Reasoning Is Latent*
   - Proves that models form semantic commitments in latent activation space prior to emitting overt verbal reasoning tokens.

---

## 3. Complete Paper Index (22 Papers)

1. `[Bansal et al. 2026] Legibility is Not Interpretability.pdf`
2. `[Chen et al. 2025] MixReasoning - Multi-Modal Reasoning Steps.pdf`
3. `[Chen et al. 2026] Thinking vs NoThinking - Sparse Autoencoders Analysis.pdf`
4. `[Dua et al. 2026] Commitment Boundary in Test-Time Compute.pdf`
5. `[Gao et al. 2025] Plan Before You Write.pdf`
6. `[Jiang et al. 2025] Are Reasoning Models More Faithful.pdf`
7. `[Kim et al. 2026] Entertain, then Cancel - Pragmatic Interpretation.pdf`
8. `[Lanham et al. 2023] Measuring Faithfulness in Chain-of-Thought Reasoning.pdf`
9. `[Lee et al. 2024] TypedThinker - Structural Planning in Reasoning.pdf`
10. `[Li et al. 2025] Learning to Reason for Long-Form Story Generation.pdf`
11. `[Mao et al. 2026] Instruction-Tuned Models Cannot Sample from Distributions They Describe.pdf`
12. `[McKenna et al. 2026] LLM Reasoning Is Latent.pdf`
13. `[Prasad et al. 2026] Are Stated Reasoning Steps Causally Load-Bearing.pdf`
14. `[Srinivasan et al. 2026] Are You Sure Youre Sure - Confidence and Lexical Diversity.pdf`
15. `[Sun et al. 2025] Reasoning Path Divergence.pdf`
16. `[Venkatesh et al. 2025] CoT Dynamics - Commitment and Information Flow.pdf`
17. `[Wang et al. 2025] Demystifying Hybrid Thinking.pdf`
18. `[Wang et al. 2025] Thinking in Character.pdf`
19. `[Xu et al. 2026] Language of Thought Shapes Output Diversity.pdf`
20. `[Zhang et al. 2024] Thinking Before Speaking.pdf`
21. `[Zhao et al. 2026] R2-Write - Reasoning for Creative Writing.pdf`
22. `[Zhou et al. 2025] Can LLMs Generate Good Stories.pdf`
