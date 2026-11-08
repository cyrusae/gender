# Reading Guide & Triage: 07A CoT Faithfulness, Latent Dynamics & Trace Mechanics (NB5A)

This collection contains **22 research papers** investigating what actually happens inside language model residual streams and token distributions during test-time reasoning / Chain-of-Thought (CoT).

---

## 1. Guiding Theoretical Questions
- **Are Stated Steps Causally Load-Bearing?** Does the model rely on its visible reasoning tokens, or are outputs determined earlier, with CoT acting as post-hoc rationalization? The papers here disagree.
- **Commitment Boundaries:** At what token index in a reasoning trajectory does an LLM fix its final answer?
- **Mechanistic Legibility:** What do sparse autoencoders (SAEs), probing and patching show about internal computation during thinking?

---

## 2. Executive Triage & Priority Reading List

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: DEFINITIVE ANCHORS                      │
│  Lanham et al. (2023)  │ Bhupatiraju (2026)     │  Wang (2026)          │
│  [Trace Faithfulness]  │ [Causal Load-Bearing]  │  [Latent Reasoning]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    TIER 2: SPECIFIC DYNAMICS & PLANNING                │
│  Scalena (2026)        │  Lewis-Lim (2025)        │ Cheng (2026)       │
│  [Commitment Boundary] │ [Information Flow Dynamics]│ [SAEs in Thinking]│
│  Lewenberg (2025)      │  Wang (2024, TypedThinker)│ Tang (2025)       │
│  [Plan Before Write]   │ [TypedThinker Planning]  │ [Thinking Character]│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                   TIER 3: EVALUATION & BEHAVIORAL AUDITS               │
│  Du (2026)       │ Lu (2025)      │ Chua (2025)    │ Kou (2026)        │
│  Gurung (2025)   │ Jang (2026)    │ Srinivasan (2026) │ Ju (2025)      │
│  Wang (2025)     │ Xu (2026)      │ Zhang (2024)   │ Liu (2026)        │
│  Wang (2025)     │                │                │                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Essential Anchors
1. **Lanham et al. (2023)** — *Measuring Faithfulness in Chain-of-Thought Reasoning*
   - Anthropic study (arXiv 2307.13702). Intervenes on the CoT (truncation, adding mistakes, paraphrasing, filler tokens). Reliance on CoT varies across tasks, and larger models give less faithful reasoning on most tasks.
2. **Bhupatiraju & Nyaupane (2026; file name says Prasad)** — *Are Stated Reasoning Steps Causally Load-Bearing? Activation Patching Says Less Often Than Text Editing Does*
   - Activation patching on synthetic multi-hop tasks: 76.9% of stated steps are causally load-bearing in Qwen3-4B (54.8% in Qwen3-1.7B); the text-editing test overstates faithfulness by 11.4 points.
3. **Wang (2026; file name says McKenna)** — *LLM Reasoning Is Latent, Not the Chain of Thought*
   - Single-author position paper arguing latent-state trajectories should be the default object of study; it presents no new proof that commitments precede verbal tokens.

---

## 3. Complete Paper Index (22 Papers)

Note: tier-box names above use the real first authors. Many PDF file names below carry a different author (e.g. Dua = Scalena et al., Prasad = Bhupatiraju & Nyaupane, McKenna = Wang, Bansal = Du et al., Chen 2025 = Lu et al., Chen 2026 = Cheng et al., Gao 2025 = Lewenberg et al., Jiang = Chua & Evans, Kim = Kou et al., Lee = Wang et al., Li = Gurung & Lapata, Mao = Jang et al., Sun = Ju et al., Venkatesh = Lewis-Lim et al., Zhang 2024 = Zhang et al. (same), Zhao 2026 = Liu et al., Zhou 2025 = Wang & Kreminski). File names are left unchanged so links keep working. See the master 07 README for per-paper details.

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
