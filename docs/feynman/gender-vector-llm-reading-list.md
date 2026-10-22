# Grammatical gender → LLM representations → English completions: a source map

**Prepared:** 2026-10-21 · **Scope:** reading list for extracting a grammatical-gender vector from Spanish/German hidden states and testing whether it influences English generation, inspired by Boroditsky's bridge/key studies. Sources verified via arXiv, eScholarship, Crossref/OpenAlex/Semantic Scholar, and web fetch; items flagged `preprint`, `venue unverified`, or `authors unverified` where appropriate.

---

## TL;DR

1. **The Boroditsky bridge effect, in humans, is on shaky ground.** The famous adjective study (German speakers call bridges *beautiful/elegant*; Spanish speakers call them *strong/sturdy*) was never published in a peer-reviewed journal, and the replication record is mostly null (Mickan et al. 2014; Elpers et al. 2022; systematic review by Samuel et al. 2019; causal corpus analysis by Stanczak et al. 2024). The one pillar that *does* replicate is the causal one: teaching English speakers an artificial gender-like category changes their judgments (Elpers et al. 2022 replicated exactly this). Your planned manipulation (introduce a vector learned from one language, test transfer in English) is architecturally the **causal** version — the stronger pillar.
2. **In LLMs, the phenomenon is much easier to find.** LLMs demonstrably encode grammatical gender in *shared, cross-lingual internal features* that are causally steerable (Brinkmann et al. 2025), and adjective-based gender signals transfer across languages at the output level ("What an Elegant Bridge," 2024). So your experiment has genuine prior art indicating it should work mechanically.
3. **But three confounds will decide whether your result means anything:**
   - *Grammatical vs. social gender* get conflated in contextual embeddings (2606.30152; Omrani Sabbaghi & Caliskan 2022).
   - *Linear "bias/gender directions" often capture model confidence, not the concept* (2610.08559) — steering can change completions by degrading the model.
   - *The internal concept space is English-pivoted* (Wendler et al. 2024), so "Spanish → English transfer" may partly be an artifact of architecture, not of the vector.

---

## 1. The original studies (anchor texts)

| Source | What it is | Why it matters |
|---|---|---|
| **Phillips & Boroditsky (2003)**. *Can Quirks of Grammar Affect the Way You Think? Grammatical Gender and Object Concepts.* Proc. 25th Ann. Meeting of the Cognitive Science Society. [eScholarship PDF](https://escholarship.org/uc/item/31t455gf) | The only empirical report of the similarity-rating experiments (Spanish/German bilinguals tested **in English** on picture similarity; plus artificial-category training producing the effect in English speakers). | This is the study all "bridge experiment" claims trace back to. Never peer-reviewed in a journal; the adjective-generation data (key/bridge) were never fully published — a fact the replication literature repeatedly stresses. |
| **Boroditsky, Schmidt & Phillips (2003)**. *Sex, Syntax, and Semantics.* In *Language in Mind* (MIT Press). [DOI 10.7551/mitpress/4117.003.0010](https://doi.org/10.7551/mitpress/4117.003.0010) · [open-access PDF](https://escholarship.org/uc/item/0jt9w8zf) | The widely-cited chapter describing grammatical-gender effects, incl. the bridge/key adjective data. | The *popularized* version of the finding (TED talk, popular books) — and the version you're "inspired by." Good to read alongside the CogSci paper to see what was actually reported vs. retold. |
| **Boroditsky & Schmidt (2000)**. *Sex, Syntax, and Semantics.* Proc. 22nd Ann. Meeting of the Cognitive Science Society (via [Stanczak et al. 2024 refs](https://arxiv.org/abs/2311.18567)) | Earliest report of the adjective-generation task (German/Spanish speakers describing nouns in English). | Predecessor of the chapter; same unpublished-procedure problem. |
| Boroditsky (2001). *Does Language Shape Thought? Mandarin and English speakers' conceptions of time.* Cognitive Psychology 43(1):1–22. | The other famous Boroditsky effect (time). | Context for the neo-Whorfian program; also has its own replication controversies worth knowing if you frame your project as "Whorf in LLMs." |

## 2. The replication record (read this before you interpret anything)

| Source | Verdict | Details worth stealing |
|---|---|---|
| **Samuel, Cole & Eacott (2019)**, *Grammatical gender and linguistic relativity: A systematic review.* Psychonomic Bulletin & Review 26:1767–1786. [DOI 10.3758/s13423-019-01652-3](https://doi.org/10.3758/s13423-019-01652-3) · [open PDF](https://link.springer.com/content/pdf/10.3758/s13423-019-01652-3.pdf) | Systematic review of 43 studies: only ~32% of results give unambiguous support for gender-congruency effects; 43% give none. | Effects concentrate in tasks where gender is **highly salient** (voice-choice, sex-assignment) — i.e., task demands. The Phillips-Boroditsky similarity task was one of only two "low salience" studies. |
| **Mickan, Schiefke & Stefanowitsch (2014)**, *Key is a llave is a Schlüssel.* Yearbook of the German Cognitive Linguistics Association. [DOI 10.1515/gcla-2014-0004](https://doi.org/10.1515/gcla-2014-0004) · [open PDF](https://stefanowitsch.de/anatol/files/p/mickanetal2014_kls.pdf) | Two failed replications (word association + primed lexical decision). | Their diagnosis: original results were "a statistical fluke" or due to undocumented stimulus details. Also useful: their critique that association tasks are dominated by perceptually salient descriptors. |
| **Elpers, Jensen & Holmes (2022)**, *Does grammatical gender affect object concepts? Registered replication of Phillips and Boroditsky (2003).* Journal of Memory and Language 127:104357. [DOI 10.1016/j.jml.2022.104357](https://doi.org/10.1016/j.jml.2022.104357) | **The** high-powered preregistered replication (N=375). **Null** gender-congruency effect for Spanish/German speakers' similarity ratings once key error variance is modeled — but the English fake-category training effect **replicated**. | This split result is the single most important fact for your project: the *correlational* "native-language gender leaks into English judgments" effect did not replicate; the *causal induction* effect did. |
| **Semenuks, Philips, Dalca, Kim & Boroditsky (2017)**, *Effects of Grammatical Gender on Object Description.* Proc. 39th Ann. Meeting of the Cognitive Science Society. [eScholarship PDF](https://escholarship.org/uc/item/3j37f380) | Partial positive: French/German/Romanian bilinguals generated slightly more feminine adjectives for feminine-gender nouns — but only for adjectives #2 and #3, strongest for animals; **null for concrete objects**. | Matches the hypothesis that any real effect is weak and category-dependent; also shows why single-adjective designs (and single-item sets like "bridge") are uninformative. |
| **Xiao, Cremers, Straboni, Strickland & Peperkamp (2026)**, *The Influence of Grammatical Gender on Object Conceptualization Is Weak and Language-Dependent.* [PDF (LSCP)](http://www.lscp.net/persons/peperkamp/Xiao_Cremers_Straboni_Strickland_Peperkamp_(2026)_The_influence_of_grammatical_gender_on_object_conceptualization_is_weak_and_language-dependent.pdf) · `venue unverified` | Three conceptual replications of the adjective-association task with sensitivity controls (person-noun control condition). | Explicit control-condition design is a model for your own sensitivity checks. Concludes effects, if any, are weak and language-dependent. |
| **Mahajan & Durgin (2025)**, *A conceptual replication of an implicit test of grammatical gender effects on inanimate concepts.* Language and Cognition. [DOI 10.1017/langcog.2024.76](https://doi.org/10.1017/langcog.2024.76) | Conceptual replication of the implicit similarity task in English. | Another data point that the Elpers-style null pattern extends to other implicit tasks. (Authors/venue verified via Crossref on 2026-10-21.) |
| **Kurinski & Sera (2011)**, *Does learning Spanish grammatical gender change English-speaking adults' categorization of inanimate objects?* Bilingualism: Language and Cognition. [DOI 10.1017/s1366728910000179](https://doi.org/10.1017/s1366728910000179) | English adults who learned Spanish gender assigned more gender-stereotyped properties to objects. | **This is the human analog of your exact manipulation** (induce gender knowledge in one language, test transfer in English). Follow-ups: Kurinski, Jambor & Sera (2016) [DOI 10.1177/1367006915576833](https://doi.org/10.1177/1367006915576833); Athanasopoulos & Boutonnet (2016) [chapter](https://doi.org/10.21832/9781783094837-011). |
| Preregistered Boroditsky replication attempt. [OSF DOI 10.17605/osf.io/ics7w](https://doi.org/10.17605/osf.io/ics7w) | A community preregistration of the adjective study. | Useful template for preregistering your own pipeline (stimuli, exclusion criteria, analysis plan). |

## 3. Gender in LLM internal representations

| Source | What it shows |
|---|---|
| **Estimating Grammatical Gender Directions in Contextual Embeddings under Controlled and Natural Contexts** (2026). [arXiv 2606.30152](https://arxiv.org/abs/2606.30152) | **Your extraction playbook.** Spanish/French in-context embeddings conflate *grammatical* gender with *social semantic* bias. They build balanced inanimate-noun datasets in controlled templates vs. natural contexts and compare centroid / SVM / LDA gender-direction estimators. Findings: controlled unweighted templates give the *purest grammatical gender directions*; the simple **centroid estimator beats discriminative ones**; orthogonal projection on the axis removes grammatical signal without wrecking semantics. |
| **Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages** (Brinkmann, Wendler, Bartelt, Mueller; 2025). [arXiv 2501.06346](https://arxiv.org/abs/2501.06346) | **The mechanistic heart of your hypothesis.** Sparse autoencoders on Llama-3-8B and Aya-23-8B: up to ~50% overlap in the top-32 causally-relevant SAE features encoding grammatical gender across typologically diverse languages; ablating "massively multilingual" features collapses gender classifiers; steering a *single* shared gender feature flips grammatical gender (e.g., masculine→feminine pronoun) in machine-translation output, with high selectivity. That is causal evidence of a shared, steerable cross-lingual gender representation. |
| **What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages** (Shtedritski & Mihaylov; 2024). [arXiv 2407.09704](https://arxiv.org/abs/2407.09704) | **The LLM analog of Boroditsky, at the output level.** Prompted Mistral-7B / Llama2-7B to describe gendered nouns in 10 languages (incl. Spanish & German), translated adjectives to English, and trained classifiers to predict grammatical gender from the adjectives alone: within-language accuracy ~56–69%; **cross-language (leave-one-out) transfer still ~49–63%** (Spanish 58.8%, German 58.6%). Also: adjectives' gender associations agree with native English speakers' intuitions (55–65%). Caveat: this is prompting-level, not hidden-state steering — correlational, not causal. |
| **The Causal Influence of Grammatical Gender on Distributional Semantics** (Stanczak, Du, Williams, Augenstein, Cotterell; 2024). [arXiv 2311.18567](https://arxiv.org/abs/2311.18567) | **Your confound-control template.** Causal graphical model over noun meaning, gender, and adjective choice in DE/ES/HE/PL/PT Wikipedia. Correlational mutual information MI(A;G) ≈ 0.2–0.5; after intervening on noun meaning, MI_do shrinks to ~10⁻⁴–10⁻³ (DE 1.24e–4, ES 4.60e–5). Gender↔adjective associations are almost entirely explained by *meaning*, not gender. Any gender signal you extract must be tested against noun-meaning controls or it may be semantic similarity in disguise. |
| **Grammatical Gender, Neo-Whorfianism, and Word Embeddings** (Kann; 2019). [arXiv 1910.09729](https://arxiv.org/abs/1910.09729) | Early static-embedding version of the idea (gender→adjective associations in word2vec). |
| **Grammatical gender associations outweigh topical gender bias in cross-linguistic word embeddings** (2020). [arXiv 2005.08864](https://arxiv.org/abs/2005.08864) | Cross-lingual embeddings: grammatical-gender associations dominate topical bias signals; an early hint that gender is a strong geometric axis across languages. |
| **Measuring gender bias in word embeddings of gendered languages requires disentangling grammatical gender signals** (Omrani Sabbaghi & Caliskan; AIES 2022) | The conflation problem formalized for static embeddings — cite this when motivating the grammatical-vs-social split. |

## 4. Cross-lingual transfer machinery (why a Spanish vector might work in English)

| Source | Key idea |
|---|---|
| **Do Llamas Work in English? On the Latent Language of Multilingual Transformers** (Wendler et al.; 2024). [arXiv 2402.10588](https://arxiv.org/abs/2402.10588) | Llama-2's abstract "concept space" sits **closer to English** than to other languages. Explains *why* cross-lingual vectors transfer — but also means part of your "transfer" is the model routing through English anyway. |
| **Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations** (2024). [arXiv 2411.08745](https://arxiv.org/abs/2411.08745) | Activation patching shows concepts are largely language-agnostic in mid layers — direct mechanistic support for concept-level transfer between languages. |
| **Cross-Lingual Activation Steering for Multilingual Language Models** (2026). [arXiv 2601.16390](https://arxiv.org/abs/2601.16390) | Steering vectors extracted in one language applied in another — the exact operational move you're planning; includes failure modes and layer recommendations. |
| **Language Steering for Multilingual In-Context Learning** (2026). [arXiv 2602.02326](https://arxiv.org/abs/2602.02326) | Language identity itself is a steerable activation-space direction; separable language structure coexists with concept structure. |
| **Cross-model Transferability on Platonic Representations of Concepts** (2025). [arXiv 2501.02009](https://arxiv.org/abs/2501.02009) | Concept steering vectors transfer across *differently trained* models; gender-direction transfer arguments get stronger if your findings generalize across model families. |
| Also useful: Cross-Architecture Steering Transfer (2026) [arXiv 2608.05164](https://arxiv.org/abs/2608.05164), BabelSteering (English steering vectors for multilingual safety) [arXiv 2608.16577](https://arxiv.org/abs/2608.16577), Cross-lingual function vectors (2026) [arXiv 2608.29613](https://arxiv.org/abs/2608.29613), Target-language generation via optimal control [arXiv 2609.16967](https://arxiv.org/abs/2609.16967). |

## 5. Steering methods (the toolkit)

| Source | Contribution |
|---|---|
| **Activation Addition** (Turner et al.; 2023, title now *Steering Language Models With Activation Engineering*). [arXiv 2308.10248](https://arxiv.org/abs/2308.10248) | Canonical CAA: contrast-pair mean-difference steering vector; single contrast pair suffices. |
| **Inference-Time Intervention (ITI)** (Li et al.; 2023, NeurIPS). [arXiv 2306.03341](https://arxiv.org/abs/2306.03341) | Head-level activation shifts along truthfulness directions; the standard for concept-direction interventions. |
| **Improving Activation Steering with Mean-Centring** (Jorgensen et al.; 2023). [arXiv 2312.03813](https://arxiv.org/abs/2312.03813) | Cheap trick with real payoff: subtract the dataset mean from the contrast direction. Relevant to cleaning a gender direction extracted from templates. |
| Bias-mitigation steering family (see also §6 cautions — treat all "bias reduced" claims as confounded until checked): *Shifting Perspectives* (2025) [arXiv 2503.05371](https://arxiv.org/abs/2503.05371); *FairSteer* (2025) [arXiv 2504.14492](https://arxiv.org/abs/2504.14492); *Activation Steering for Bias Mitigation* (2025) [arXiv 2508.09019](https://arxiv.org/abs/2508.09019); *Unsupervised Concept Vector Extraction* (2025) [arXiv 2502.19721](https://arxiv.org/abs/2502.19721); *Neuron-Level Interventions for Gendered and Gender-Neutral Generation* (2026) [arXiv 2605.30717](https://arxiv.org/abs/2605.30717). |

## 6. Cautions & confounds (read before you report any "effect")

1. **Bias directions often capture confidence, not the concept.** *Latent space bias directions in LLMs capture confidence, not fairness* (Buttigieg et al.; 2026). [arXiv 2610.08559](https://arxiv.org/abs/2610.08559). A CAA debiasing vector computed from biased vs. anti-biased prompts has cosine similarity **0.29–0.91** (mean ~0.5–0.7) with a pure *confidence* direction computed from unrelated QA data; "steering away from bias" mostly raises abstention/entropy and flattens answer probabilities — like temperature scaling, not debiasing. **Implications for you:** (a) run magnitude sweeps in *both* directions; (b) include forced-choice completions with no abstain option; (c) report answer entropy/perplexity alongside any gender flip rates; (d) include a control vector computed from high- vs. low-probability completions to check your gender vector isn't accidentally a confidence vector; (e) don't trust "bias score went down" on ambiguous benchmarks (BBQ-style) since bias scores factor in abstention.
2. **Grammatical vs. social gender conflation** (2606.30152; Omrani Sabbaghi & Caliskan 2022). Extract from inanimate nouns in controlled templates; keep occupation/people nouns for the *social* axis as a separate vector and compare.
3. **Meaning confound** (Stanczak et al. 2024). A gender direction can be a disguised noun-meaning direction. Test: does your vector predict gender *within* semantically matched noun pairs, and does male/female word content explain your English completions better than grammatical gender does?
4. **English pivot** (Wendler et al. 2024). A Spanish-derived vector applied in English may just reinforce the model's English-centric concept space. Compare Spanish→English vs. German→English transfer asymmetry, and use a morphologically distant language (e.g., Turkish, if the model handles it) as a reference to see how much "sharing" is real.
5. **Behavioral reality check.** Elpers et al. 2022 and Samuel et al. 2019: humans show little/no native-language gender congruency. If your LLM experiment "confirms Boroditsky," the appropriate claim is *about the model and its training distribution*, not about human cognition (the Elegant Bridge authors make exactly this point in their discussion).

## 7. Evaluation resources (English-completion side)

| Source | Use |
|---|---|
| **WinoGender** (Rudinger et al., 2018, NAACL). [arXiv 1804.09301](https://arxiv.org/abs/1804.09301) · **WinoBias** (Zhao et al., 2018, ACL). [arXiv 1804.06876](https://arxiv.org/abs/1804.06876) | Standard forced-choice he/she completion probes in English — the natural measurement surface for "did the vector flip English completions?" |
| **Adapting Psycholinguistic Research for LLMs: Gender-inclusive Language in a Coreference Context** (2025). [arXiv 2502.13120](https://arxiv.org/abs/2502.13120) | Psycholinguistic coreference paradigm (Tibblin et al.) adapted to LLMs in EN/DE/FR: English keeps antecedent gender with an underlying masculine bias; German bias is stronger. Direct template for measuring gendered completions. |
| **Tracing Stereotypes from Representation to Output in Multilingual LLMs** (2026). [arXiv 2609.08322](https://arxiv.org/abs/2609.08322) | Methodologically closest to your pipeline: traces where stereotype information lives and how it moves from hidden states to outputs across languages. |
| **GRUFF: LLM Pronoun Fidelity, Reasoning, and Biases in German** (2026). [arXiv 2605.30214](https://arxiv.org/abs/2605.30214) | German-specific pronoun/reference diagnostics — useful if you extend to German target tasks. |
| **MORPHOGEN** (multilingual gender-aware morphology benchmark, 2026). [arXiv 2604.18914](https://arxiv.org/abs/2604.18914) | Gender-agreement generalization evaluation across languages. |

## 8. Design implications → a concrete checklist for your experiment

1. **Extract** per-layer hidden states for inanimate Spanish nouns (feminine vs. masculine) in controlled templates; estimate the direction with the centroid method (2606.30152); mean-center it (2312.03813). Repeat for German. Compute cross-lingual cosine alignment *within the same model* (2501.06346 predicts high alignment; 2402.10588 warns part of it is English-centric).
2. **Disentangle**: build a parallel *social* gender vector from occupation/people nouns and a *confidence* control vector from high/low-probability completions (2610.08559, 2606.30152).
3. **Steer** English generation (adjective generation à la 2407.09704 and WinoGender-style pronoun completions) at a sweep of magnitudes α ∈ {−3…+3}, in both signs.
4. **Measure** flip rates, forced-choice accuracy, *and* entropy/perplexity; require effects to be direction-specific and magnitude-monotone, not symmetric degradation.
5. **Control**: random-vector steering; meaning-matched noun pairs; steer the German vector on the Spanish task and vice versa to test transfer specificity; ablate gender-marked tokens.
6. **Contextualize**: compare your result with the behavioral record (Elpers et al. 2022 null; Kurinski & Sera 2011 positive for explicit learning) before making any claims about linguistic relativity vs. model-statistics.

## 9. Open questions your project could actually move

- Is there one *linear* shared gender direction across Spanish/German, or a distributed set of features (2501.06346 suggests features, not necessarily a single line)?
- Does a *grammatical-only* vector (inanimate nouns) transfer to English completions better or worse than a *social* gender vector? Nobody has cleanly compared.
- Which layer? (Mid-layers look best for SAE gender features, 2501.06346; best bias-classifier layers vary 13–22, 2610.08559.) Layerwise transfer curves would be a nice contribution.
- Does "What an Elegant Bridge" hold up *causally* — i.e., with steering rather than prompting? Its authors note their result is correlational.

---

## Sources

- Phillips, W., & Boroditsky, L. (2003). Can Quirks of Grammar Affect the Way You Think? Grammatical Gender and Object Concepts. *Proc. 25th Annual Meeting of the Cognitive Science Society*. https://escholarship.org/uc/item/31t455gf
- Boroditsky, L., Schmidt, L. A., & Phillips, W. (2003). Sex, Syntax, and Semantics. In *Language in Mind* (MIT Press). https://doi.org/10.7551/mitpress/4117.003.0010 · https://escholarship.org/uc/item/0jt9w8zf
- Samuel, S., Cole, G. G., & Eacott, M. J. (2019). Grammatical gender and linguistic relativity: A systematic review. *Psychonomic Bulletin & Review*, 26, 1767–1786. https://doi.org/10.3758/s13423-019-01652-3
- Mickan, A., Schiefke, M., & Stefanowitsch, A. (2014). Key is a llave is a Schlüssel: A failure to replicate an experiment from Boroditsky et al. 2003. *Yearbook of the German Cognitive Linguistics Association*. https://doi.org/10.1515/gcla-2014-0004 · https://stefanowitsch.de/anatol/files/p/mickanetal2014_kls.pdf
- Elpers, N., Jensen, G., & Holmes, K. J. (2022). Does grammatical gender affect object concepts? Registered replication of Phillips and Boroditsky (2003). *Journal of Memory and Language*, 127, 104357. https://doi.org/10.1016/j.jml.2022.104357
- Semenuks, A., Philips, W., Dalca, I., Kim, C., & Boroditsky, L. (2017). Effects of Grammatical Gender on Object Description. *Proc. 39th Annual Meeting of the Cognitive Science Society*. https://escholarship.org/uc/item/3j37f380
- Xiao, K., Cremers, A., Straboni, J., Strickland, B., & Peperkamp, S. (2026, preprint; venue unverified). The Influence of Grammatical Gender on Object Conceptualization Is Weak and Language-Dependent. http://www.lscp.net/persons/peperkamp/Xiao_Cremers_Straboni_Strickland_Peperkamp_(2026)_The_influence_of_grammatical_gender_on_object_conceptualization_is_weak_and_language-dependent.pdf
- Mahajan, D., & Durgin, F. H. (2025). A conceptual replication of an implicit test of grammatical gender effects on inanimate concepts. *Language and Cognition*. https://doi.org/10.1017/langcog.2024.76
- Kurinski, E., & Sera, M. D. (2011). Does learning Spanish grammatical gender change English-speaking adults' categorization of inanimate objects? *Bilingualism: Language and Cognition*. https://doi.org/10.1017/s1366728910000179
- Stanczak, K., Du, K., Williams, A., Augenstein, I., & Cotterell, R. (2024). The Causal Influence of Grammatical Gender on Distributional Semantics. https://arxiv.org/abs/2311.18567
- Estimating Grammatical Gender Directions in Contextual Embeddings under Controlled and Natural Contexts (2026). https://arxiv.org/abs/2606.30152
- Brinkmann, J., Wendler, C., Bartelt, C., & Mueller, A. (2025). Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages. https://arxiv.org/abs/2501.06346
- Shtedritski, A., & Mihaylov, V. (2024). What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages. https://arxiv.org/abs/2407.09704
- Wendler, C., Veselovsky, V., Monea, G., & West, R. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. https://arxiv.org/abs/2402.10588
- Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers (2024). https://arxiv.org/abs/2411.08745
- Cross-Lingual Activation Steering for Multilingual Language Models (2026). https://arxiv.org/abs/2601.16390
- Language Steering for Multilingual In-Context Learning (2026). https://arxiv.org/abs/2602.02326
- Turner, A. M., et al. (2023). Steering Language Models With Activation Engineering (Activation Addition). https://arxiv.org/abs/2308.10248
- Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-Time Intervention. https://arxiv.org/abs/2306.03341
- Jorgensen, O., Cope, D., Schoots, N., & Shanahan, M. (2023). Improving Activation Steering in Language Models with Mean-Centring. https://arxiv.org/abs/2312.03813
- Buttigieg, S., Madigan, M., Kamalaruban, P., & Burrell, S. (2026). Latent space bias directions in LLMs capture confidence, not fairness. https://arxiv.org/abs/2610.08559
- Adapting Psycholinguistic Research for LLMs: Gender-inclusive Language in a Coreference Context (2025). https://arxiv.org/abs/2502.13120
- Tracing Stereotypes from Representation to Output in Multilingual LLMs (2026). https://arxiv.org/abs/2609.08322
- Rudinger, R., Naradowsky, J., Leonard, B., & Van Durme, B. (2018). Gender Bias in Coreference Resolution (WinoGender). https://arxiv.org/abs/1804.09301
- Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K.-W. (2018). Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods (WinoBias). https://arxiv.org/abs/1804.06876
- Kann, K. (2019). Grammatical Gender, Neo-Whorfianism, and Word Embeddings. https://arxiv.org/abs/1910.09729
- Grammatical gender associations outweigh topical gender bias in cross-linguistic word embeddings (2020). https://arxiv.org/abs/2005.08864
- GRUFF: LLM Pronoun Fidelity, Reasoning, and Biases in German (2026). https://arxiv.org/abs/2605.30214
- MORPHOGEN (2026). https://arxiv.org/abs/2604.18914
