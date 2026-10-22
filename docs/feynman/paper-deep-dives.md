# Deep dives: Elpers et al. (2022), Buttigieg et al. (2026, "confidence, not fairness"), and Shtedritski & Mihaylov (2024, "What an Elegant Bridge")

**Prepared:** 2026-10-21 · **Purpose:** three close readings relevant to writing up a grammatical-gender-vector × English-completion experiment on Qwen3/Gemma. Every number below traces to the paper or its open materials (OSF/RBI archive), linked in Sources.

---

## 1. Elpers, Jensen & Holmes (2022) — *Does grammatical gender affect object concepts? Registered replication of Phillips and Boroditsky (2003)*

**Journal of Memory and Language 127:104357.** Reed College. Open-access PDF at reed.edu (linked below); Stage-1 protocol + materials + code on OSF: https://doi.org/10.17605/osf.io/9c2ak

### What they did
- Registered replication of **two** of Phillips & Boroditsky's (2003) experiments, total **N = 375**.
- **Experiment 1 (correlational test):** native **Spanish and German** speakers (proficient in English) plus an English monolingual control group rated the similarity of picture pairs, each pair = one person (male/female) + one inanimate object/animal, and judged on a 9-point scale. Two design upgrades over the original: (a) the object's gender was coded from **each participant's own native-language label** of the picture rather than the experimenter's presumptive dominant label; (b) prespecified mixed-effects models with random slopes/intercepts for participants, person items, and object items (P&B had only by-participant and by-item *t*-tests).
- **Experiment 2 (causal/induction test):** monolingual English speakers (~n=150) learned two "fictional language" gender-like categories (all females + some objects are *oosative*; all males + other objects are *soupative*), reaching ≥90% classification accuracy, then did the same picture-similarity task.

### What they found
- **Exp 1: null.** No gender-congruency effect for Spanish or German speakers in the registered mixed-effects analysis, and no difference from the English control group. Registered Bayesian analyses "overwhelmingly supported the null." Importantly, they argue the null can't be blamed on weak power or sloppy measurement: high-powered design, verified native-language comprehension, and gender coding per-participant — if anything, these make the design *more* sensitive than the original.
- **Exp 2: replicated, and bigger than the original.** Same-gender pairs rated more similar (M = 3.71, SD = 2.04) than different-gender pairs (M = 2.56, SD = 1.21); mixed-effects χ²(1) = 11.80, p = .001; effect size d = 0.66 (by participants), BF₁₀ = 5.01×10¹⁰. The 1.15-point difference is **descriptively larger** than P&B's original 0.66-point difference and the CIs don't overlap.
- Their interpretation: the *correlational* "native-language gender leaks into English-language judgments" finding did **not** replicate; the *causal induction* finding **did** — with an important caveat: several participants later reported they assumed they were *supposed* to use the newly learned categories, i.e., the effect may partly be demand characteristics / strategic use. This is why they emphasize that even positive grammatical-gender effects in this task class may reflect recruiting the sex↔gender *statistical association* as a similarity heuristic rather than a changed object concept.

### What this means for your write-up
1. **Your pipeline is the Experiment-2 analog, not Experiment 1.** You induce a gender structure in one language ("train the model on Spanish templates" → extract a direction → apply it in English). The behavioral precedent says this *causal* version is the one that replicates — that's a strong framing hook and a legitimate defense of the exploratory design.
2. **The demand-characteristics finding is your mechanistic control.** If your vector "works," the human literature forces the question: does the direction encode *grammatical gender* or a *he/she or masculine/feminine lexical statistics axis* the model can recruit whenever the task makes sex salient? Your control should be a **social-gender direction from animate/occupation nouns** (2606.30152) and a **meaning-matched inanimate control**, not just a random vector.
3. **Reporting template:** they registered everything (OSF Stage-1 protocol), pre-specified the mixed-effects model, and reported registered vs. exploratory analyses separately. You can borrow this even post-hoc: write up which analyses were preplanned, which were added after seeing results, and show nulls. Given you flew blind initially, this discipline is what makes an exploratory paper credible.

---

## 2. Buttigieg, Madigan, Kamalaruban & Burrell (2026) — *Latent space bias directions in LLMs capture confidence, not fairness* (arXiv 2610.08559)

**Institute of Astronomy, Cambridge + Visa Risk & Security AI Lab.** arXiv preprint, Oct 2026.

### What they did
- **Setup:** 8 open-weight models — Llama-3.1-8B (base + instruct), Falcon3-7B (±instruct), Ministral-3-8B (±instruct), Qwen3.5-9B (±instruct). Bias probes/dishes: BBQ, StereoSet, CrowS-Pairs. Unbiased QA: MMLU, OpenBookQA.
- **Claim tested:** the standard "debiasing direction" used in activation steering — a linear direction computed by *contrasting activations on biased vs. anti-biased prompts* (CAA-style, `v_b = mean(h_anti-biased − h_biased)`, Eq. 4 in their paper) — is widely assumed to encode "bias." They ask: what does it *actually* encode?
- **Two measurements:** (1) a logistic-regression classifier trained on biased vs. anti-biased hidden states; (2) cosine similarity between the CAA direction `v_b` and a **confidence direction** `v_c` computed from highest-vs-lowest-probability answers on *unrelated* QA datasets (MMLU, OpenBookQA).

### What they found
1. **Baseline bias is real in all 8 models** (CrowS-Pairs stereotype rate 69.0–75.8%; StereoSet 62.7–72.3%; BBQ ambiguous-context bias scores 0.07–0.18).
2. **A "bias" classifier transfers to confidence without any bias content:** a classifier trained on biased/anti-biased BBQ prompts sorts *high-probability vs. low-probability* MMLU/OpenBookQA answers with AUROC **0.52–0.95 across models — including datasets that contain no societal bias at all** (Table 2). I.e., the linear structure it found is largely likelihood/confidence.
3. **The CAA debiasing direction ≈ confidence direction:** cosine similarity between `v_b` and `v_c` ranged **0.29–0.91**, mean ≈ 0.68 (BBQ & StereoSet) / 0.50 (CrowS-Pairs) — strikingly aligned in high-dimensional space (Table 3).
4. **Steering along `v_b` degrades confidence, not bias:** where an abstain option exists (BBQ ambiguous), steering increases abstention; where it doesn't, it flattens answer probabilities (higher entropy) while rankings — and therefore greedy accuracy — barely change. E.g., the StereoSet direction more than doubled OpenBookQA answer entropy (0.24 → 0.50) in Qwen3.5-9B. Net effect: "bias score" goes down partly because the model stops committing to answers — "the effect of temperature scaling rather than a targeted correction."
5. **Bottom line (their words):** reported steering-based debiasing should be interpreted with care; the observed "debiasing" is largely a confidence artifact.

### What this means for your write-up — this is your #1 artifact-killer
- Any **contrast-pair-derived gender direction** (Spanish-masculine/noun vs Spanish-feminine/noun template pairs) may be partially aligned with a *confidence* direction: feminine vs. masculine templates can differ systematically in predictability/rarity.
- Mandatory controls, cheap to run:
  1. Compute the confidence direction `v_c` on your own prompts (highest vs. lowest probability completions on *gender-irrelevant* QA/wikipedia text) and report `cos(v_gender, v_c)`.
  2. Steer both signs over a magnitude sweep (α ∈ [−3, +3]) and require **direction-specific, monotone** effects.
  3. Report **answer entropy and abstention rates** alongside flip rates.
  4. Prefer **forced-choice completions with no abstain option** (WinoGender-style) so a "gender effect" can't be an abstention effect.
  5. Include a **random-vector** steering control and, ideally, a **frozen-random-permutation of the contrast pair** control.
- If you already ran without these, the exploratory write-up should *report* that and mark steering-effect claims `unverified pending confidence-control analysis`. That is a legitimate and honorable move, and reviewers will respect the explicit flag.

---

## 3. Shtedritski & Mihaylov (2024) — *What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages* (arXiv 2407.09704)

**Oxford AI Society, University of Oxford.** Jul 2024. (Second author's name order: Aleksandar Shtedritski & Viktor Mihaylov in the PDF header.)

### What they did
- The **LLM remix of Boroditsky's bridge study**, at the *output/prompting* level: for each of 10 gendered languages (Bulgarian, Czech, French, German, Greek, Hindi, Italian, Latvian, Portuguese, Spanish), they collected common inanimate nouns + grammatical gender (hundreds–thousands per language; German 2031, Spanish 1758), prompting Mistral-7B (and Llama2-7B) **50× per noun** to "Describe the word X using comma-separated adjectives", keeping the top-50 most frequent adjectives per noun; translated the adjectives to English (kills morphological gender cues: *bonito/bonita*); represented them as frequency-weighted GloVe embeddings; and trained a **2-layer MLP to predict grammatical gender from the adjectives**.
- Then the key experiment: **leave-one-out cross-language transfer** — train the gender classifier on 9 languages, test on the held-out 10th.

### What they found
1. **Within-language gender is predictable from adjectives:** overall accuracy 54.3–69.0% (F1 0.40–0.68). Spanish 63.3%, German 60.0% (Table 1). Stable between Mistral-7B and Llama2-7B (Table 3).
2. **The signal transfers across languages:** leave-one-out accuracy 48.8–62.8% (Spanish 58.8%, German 58.6%; Table 2), far above chance. **"While LLMs may describe words differently in different languages, they are biased similarly."**
3. **Structure:** gendered-adjective similarity scores cluster by language family (Romance, Slavic, Germanic internally high; Greek/Hindi/Latvian low vs. others) — yet the classifier still transfers, implying the *semantic* stereotype (strength/elegance dimensions) is what generalizes, not lexical overlap.
4. **Bonus:** a classifier trained on all 10 gendered languages agrees with **native English speakers' intuitions** about object gender 55% (Mistral) / 65% (Llama2) of the time — i.e., LLM "gender of bridges" patterns are partly shared with humans' own folk notions.
5. **Exclusions matter:** animate nouns removed (jobs/kin terms), robustness check shows inclusion doesn't change conclusions; sample adjective examples mirror the folklore: feminine nouns → *intricate, desolate*; masculine nouns → *dedicated, brave*.

### Limitations they themselves flag (quote-worthy for your related work)
- Only open 7B models (Mistral-7B, Llama2-7B); unknown for larger/commercial models.
- Only Indo-European languages; no neuter handling.
- **Prompting-level, correlational** — they explicitly note it can't tell whether LLM biases predict human biases, and call for causal tests.
- Contrast: Mukherjee et al. 2023 (*Global Voices, Local Biases*, arXiv 2310.17586) found per-language *divergent* biases in multilingual LLMs; this paper shows grammatical-gender adjective bias is a notable *convergent* exception.

### What this means for your write-up
- This is your **closest prior art and your launchpad**. Your experiment is the *causal, hidden-state* upgrade of exactly this study: instead of asking the model to produce adjectives and reading gender off the distribution, you (a) extract a direction from hidden states, (b) steer, (c) measure English completions. That's a clean, defensible contribution statement: *"we verify at the mechanism level what [2407.09704] established at the output level."*
- It also **predicts your headline result**: if the same latent "gender" axis drives adjective generation across languages, a Spanish-derived direction *should* modulate English completions in your model — so a null finding would be genuinely surprising and interesting (and consistent with Elpers' human nulls; a wash-out on Gemma but not Qwen3 would be a great model-contrast result).
- Reuse their exclusions (animate nouns) and their translation-to-English trick (it's an English-completion experiment anyway, but the principle — remove morphological gender leakage — applies to any Spanish/German template design).

---

## Synthesis: what your write-up can honestly claim

| Claim | Status | Requires |
|---|---|---|
| Qwen3's hidden states contain a direction that separates Spanish/German grammatical gender of inanimate nouns (layerwise profile, e.g. probe AUC by layer) | **Defensible now** — descriptive/representational claim | Probe-AUC along with the confidence-direction check (2610.08559) |
| The Spanish and German gender directions align within one model (cross-lingual sharing at the mechanism level, cf. 2501.06346) | **Defensible now** — but report cosine against null distribution of random directions | Random-direction cosine baseline |
| Steering the direction changes English completions (adjectives/pronouns) in a gender-consistent way | **Needs the full control battery** (both-sign sweep, entropy/abstention, random vector, confidence direction, forced-choice eval) | 2610.08559 controls |
| This says something about *linguistic relativity / Whorf in machines* | **Framing choice, not a finding** — humans show little/no correlational effect (Elpers; Samuel et al. 2019); the claim must be scoped to "property of models trained on gendered text," unless you also run the behavioral comparison | Explicit scoping sentence in the paper |
| The effect transfers across model families (Qwen3 → Gemma) | **A planned replication**, not yet a result | Be clear in abstract: "single-family results unless replicated" |

**On the exploratory framing:** yes, it's the right call — name it "an exploratory, mechanism-first map," report preplanned vs. post-hoc analyses, publish every null you ran (flip-rate tables at all α values, all layers probed, both languages), and register a Stage-1-style analysis plan *now* for the Gemma replication so that half your project is confirmatory. That mirrors exactly how Elpers et al. made a replication of an unpublished study credible — and it converts "flying blind" from a weakness into a documented exploratory phase.

---

## Sources
- Elpers, N., Jensen, G., & Holmes, K. J. (2022). Does grammatical gender affect object concepts? Registered replication of Phillips and Boroditsky (2003). *Journal of Memory and Language*, 127, 104357. DOI https://doi.org/10.1016/j.jml.2022.104357 · open PDF https://www.reed.edu/psychology/thought-lab/assets/publications/Elpers,%20Jensen,%20_%20Holmes%202022%20-%20JML.pdf · OSF Stage-1 protocol/materials https://doi.org/10.17605/osf.io/9c2ak
- Buttigieg, S., Madigan, M., Kamalaruban, P., & Burrell, S. (2026). Latent space bias directions in LLMs capture confidence, not fairness. arXiv:2610.08559 https://arxiv.org/abs/2610.08559
- Shtedritski, A., & Mihaylov, V. (2024). What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages. arXiv:2407.09704 https://arxiv.org/abs/2407.09704
- Samuel, S., Cole, G. G., & Eacott, M. J. (2019). Grammatical gender and linguistic relativity: A systematic review. *Psychonomic Bulletin & Review*, 26, 1767–1786. https://doi.org/10.3758/s13423-019-01652-3
- Mukherjee, A., Raj, C., Zhu, Z., & Anastasopoulos, A. (2023). Global voices, local biases: Socio-cultural prejudices across languages. arXiv:2310.17586
- Estimating Grammatical Gender Directions in Contextual Embeddings (2026). arXiv:2606.30152 https://arxiv.org/abs/2606.30152
- Brinkmann, J., Wendler, C., Bartelt, C., & Mueller, A. (2025). LLMs Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages. arXiv:2501.06346 https://arxiv.org/abs/2501.06346
