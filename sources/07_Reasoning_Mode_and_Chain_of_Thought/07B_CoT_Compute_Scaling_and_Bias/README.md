# Reading Guide & Triage: 07B CoT Compute Scaling, Overthinking & Bias/Fairness (NB6B)

This collection contains **23 research papers** exploring the failure modes, scaling trade-offs, and social/gender bias implications of test-time reasoning compute.

---

## 1. Guiding Theoretical Questions
- **When Does More Thinking Hurt?** Does allocating more test-time tokens lead to overthinking, task degradation, and entropy collapse?
- **Bias Amplification in Reasoning:** Does chain-of-thought mitigate latent social and gender stereotypes, or does it construct elaborate rationalizations for biased choices?
- **Persona Degradation:** How does heavy reasoning interact with creative writing, role-playing, and character consistency?

---

## 2. Executive Triage & Priority Reading List

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: DEFINITIVE ANCHORS                      │
│  Tu et al. (2025)      │  Zheng et al. (2026)   │  Lin et al. (2025)   │
│  [Reasoning vs Persona]│  [Thinking & Fairness] │  [Inverse Scaling]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    TIER 2: OVERTHINKING & DIVERSITY COLLAPSE           │
│  Ding et al. (2026)    │  Gao et al. (2026)     │  Guo et al. (2025)   │
│  [More Thinking Hurts] │  [Thinking Inertia]    │  [Under/Overthinking]│
│  Song et al. (2026)    │  Park et al. (2026)    │  Sharma et al. (2026)│
│  [Entropy Collapse]    │  [Diversity Collapse]  │  [Masked Sycophancy] │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                   TIER 3: SPECIFIC BIAS AUDITS & BENCHMARKS            │
│  Huang (2025)    │ Kocijan (2019) │ Kwon (2025)    │ Li (2025)         │
│  Liu (2025)      │ Liu (2026)     │ Rao (2025)     │ Shaikh (2023)     │
│  Tam (2024)      │ Tan (2025)     │ Wu (2025)      │ Yang (2026)       │
│  Zhang (2026)    │ Zhou (2025)    │                │                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Essential Anchors
1. **Tu et al. (2025)** — *Reasoning Does Not Necessarily Improve Role-Playing*
   - Foundational paper for the Character Study. Proves that enabling deep reasoning mode leads to "character leakage," clinical detachment, and persona degradation in non-factual/story tasks.
2. **Zheng et al. (2026)** — *Does Thinking Help Fairness?*
   - Systematic evaluation showing that additional test-time compute does not reliably improve demographic parity or reduce stereotyping; in many ambiguous tasks, thinking traces entrench initial biased heuristics.
3. **Lin et al. (2025)** — *Inverse Scaling in Test-Time Compute*
   - Identifies tasks where increasing reasoning tokens monotonically decreases task performance due to over-complication of straightforward judgments.

---

## 3. Complete Paper Index (23 Papers)

1. `[Ding et al. 2026] When More Thinking Hurts.pdf`
2. `[Gao et al. 2026] Thinking Inertia in Reasoning Models.pdf`
3. `[Guo et al. 2025] Underthinking and Overthinking in Reasoning.pdf`
4. `[Huang et al. 2025] Does Thinking More Always Help.pdf`
5. `[Kocijan et al. 2019] Default Reasoning in Number and Gender Assignment.pdf`
6. `[Kwon et al. 2025] Controlling Thinking Budget in Reasoning Models.pdf`
7. `[Li et al. 2025] Does Reasoning Introduce Bias.pdf`
8. `[Lin et al. 2025] Inverse Scaling in Test-Time Compute.pdf`
9. `[Liu et al. 2025] R-CHAR - Reasoning Character Framework.pdf`
10. `[Liu et al. 2026] How Reasoning Shapes Social Bias in Generated Code.pdf`
11. `[Park et al. 2026] Sampling More, Getting Less - Diversity Collapse.pdf`
12. `[Rao et al. 2025] Agree to Disagree - Meta-Evaluation of LLM Misgendering.pdf`
13. `[Shaikh et al. 2023] Lets Not Think Step by Step.pdf`
14. `[Sharma et al. 2026] Sycophancy Mitigated Yet Masked in Reasoning Models.pdf`
15. `[Song et al. 2026] Entropy Collapse in Long CoT Generation.pdf`
16. `[Tam et al. 2024] Evaluating Gender Bias via Chain-of-Thought Reasoning.pdf`
17. `[Tan et al. 2025] LLMThinkBench - Benchmarking Overthinking.pdf`
18. `[Tu et al. 2025] Reasoning Does Not Necessarily Improve Role-Playing.pdf`
19. `[Wu et al. 2025] Do Biased Models Have Biased Thoughts.pdf`
20. `[Yang et al. 2026] Language Models are Lazy Poets.pdf`
21. `[Zhang et al. 2026] Mechanics of Bias and Reasoning.pdf`
22. `[Zheng et al. 2026] Does Thinking Help Fairness.pdf`
23. `[Zhou et al. 2025] OThink-R1 - Optimization in Reasoning.pdf`
