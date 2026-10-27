# Sources & Reading List Catalog

Organized repository of 38 research papers on grammatical gender, linguistic relativity, representation learning, and activation steering in multilingual large language models.

Standardized for upload to Google NotebookLM in `[Author year] Short title.pdf` format across 5 thematic folders.

---

## Folder Organization Overview

| Folder | Focus | Papers |
|---|---|:---:|
| [`01_Human_Psycholinguistics_and_Replications/`](01_Human_Psycholinguistics_and_Replications/) | Human behavioral baselines, original Whorfian studies, and replication attempts | 10 |
| [`02_Gender_in_Embeddings_and_LLMs/`](02_Gender_in_Embeddings_and_LLMs/) | Grammatical vs. semantic gender in static word vectors, SAEs, and LLM representations | 9 |
| [`03_Crosslingual_and_Latent_Representations/`](03_Crosslingual_and_Latent_Representations/) | Shared cross-lingual geometry, English-pivot hypothesis, detokenization, and latent transfer | 6 |
| [`04_Steering_Probing_and_Interpretability/`](04_Steering_Probing_and_Interpretability/) | Activation addition (CAA), ITI, LEACE concept erasure, probe controls, and confounds | 8 |
| [`05_Lexical_Norms_and_Evaluation/`](05_Lexical_Norms_and_Evaluation/) | Psycholinguistic norm datasets (Glasgow, Dutch), concreteness covariates, and coreference benchmarks | 5 |
| `_archive_duplicates/` | Byte-for-byte identical redundant downloads safely archived | 5 |

Total unique papers: **38** (well within NotebookLM's 50-source limit for a single notebook).

---

## Complete Paper Index

### 1. Human Psycholinguistics & Replications (`01_Human_Psycholinguistics_and_Replications/`)

1. **`[Boroditsky 2001] Does Language Shape Thought.pdf`**  
   *Original:* `boroditsky_2001.pdf`  
   *Citation:* Lera Boroditsky (2001). "Does Language Shape Thought?: Mandarin and English Speakers' Conceptions of Time." *Cognitive Psychology* 43(1): 1–22.
2. **`[Phillips & Boroditsky 2003] Can Quirks of Grammar Affect the Way You Think.pdf`**  
   *Original:* `Can_quirks_of_grammar_affect_the_way_you_think_Gra.pdf`  
   *Citation:* Webb Phillips & Lera Boroditsky (2003). "Can Quirks of Grammar Affect the Way You Think? Grammatical Gender and Object Concepts." *Proc. 25th Annual Meeting of the Cognitive Science Society*.
3. **`[Boroditsky et al. 2003] Sex, Syntax, and Semantics.pdf`**  
   *Original:* `Boroditsky_ea_2003.pdf`  
   *Citation:* Lera Boroditsky, Lauren A. Schmidt & Webb Phillips (2003). "Sex, Syntax, and Semantics." In *Language in Mind*, MIT Press, pp. 61–79.
4. **`[Kurinski & Sera 2011] Spanish Grammatical Gender and Object Categorization.pdf`**  
   *Original:* `kurinski2010.pdf` (duplicate `kurinski2010-2.pdf` archived)  
   *Citation:* Elena Kurinski & Maria D. Sera (2011). "Does learning Spanish grammatical gender change English-speaking adults' categorization of inanimate objects?" *Bilingualism: Language and Cognition* 14(2): 203–220.
5. **`[Mickan et al. 2014] Key is a llave is a Schlüssel.pdf`**  
   *Original:* `Mickan_2014_Key-Is-A-Llave-Is-A-Schluessel.pdf` (duplicate `mickanetal2014_kls.pdf` archived)  
   *Citation:* Anne Mickan, Maren Schiefke & Anatol Stefanowitsch (2014). "Key is a llave is a Schlüssel: A failure to replicate an experiment from Boroditsky et al. 2003." *Yearbook of the German Cognitive Linguistics Association* 2: 39–50.
6. **`[Semenuks et al. 2017] Effects of Grammatical Gender on Object Description.pdf`**  
   *Original:* `eScholarship UC item 3j37f380.pdf`  
   *Citation:* Arturs Semenuks, Webb Philips, Ioana Dalca, Cora Kim & Lera Boroditsky (2017). "Effects of Grammatical Gender on Object Description." *Proc. 39th Annual Meeting of the Cognitive Science Society*.
7. **`[Samuel et al. 2019] Grammatical Gender and Linguistic Relativity.pdf`**  
   *Original:* `s13423-019-01652-3.pdf`  
   *Citation:* Steven Samuel, Geoff Cole & Madeline J. Eacott (2019). "Grammatical gender and linguistic relativity: A systematic review." *Psychonomic Bulletin & Review* 26: 1767–1786.
8. **`[Elpers et al. 2022] Registered Replication of Phillips and Boroditsky.pdf`**  
   *Original:* `Elpers, Jensen, _ Holmes 2022 - JML.pdf`  
   *Citation:* Nan Elpers, Greg Jensen & Kevin J. Holmes (2022). "Does grammatical gender affect object concepts? Registered replication of Phillips and Boroditsky (2003)." *Journal of Memory and Language* 127: 104357.
9. **`[Mahajan & Durgin 2025] Implicit Test of Grammatical Gender Effects.pdf`**  
   *Original:* `A Conceptual Replication Of An Implicit Test Of Grammatical Gende.pdf`  
   *Citation:* D. Mahajan & Frank H. Durgin (2025). "A Conceptual Replication Of An Implicit Test Of Grammatical Gender Effects On Inanimate Concepts." *Language and Cognition* 17: e28, 1–19.
10. **`[Xiao et al. 2026] Grammatical Gender on Object Conceptualization is Weak.pdf`**  
    *Original:* `Xiao_Cremers_Straboni_Strickland_Peperkamp_(2026)_The_influence_of_grammatical_gender_on_object_conceptualization_is_weak_and_language-dependent.pdf`  
    *Citation:* Hualin Xiao, Alexandre Cremers, Camille Straboni, Brent Strickland & Sharon Peperkamp (2026). "The Influence of Grammatical Gender on Object Conceptualization Is Weak and Language-Dependent." *Cognitive Science* 50: e70176.

---

### 2. Gender in Embeddings and LLMs (`02_Gender_in_Embeddings_and_LLMs/`)

11. **`[Kann 2019] Grammatical Gender, Neo-Whorfianism, and Word Embeddings.pdf`**  
    *Original:* `Kann_2019_Grammatical-Gender-Neo-Whorfianism-And-Word-Embeddings.pdf` (duplicate `1910.09729v1.pdf` archived)  
    *Citation:* Katharina Kann (2019). "Grammatical Gender, Neo-Whorfianism, and Word Embeddings: A Data-Driven Approach to Linguistic Relativity." *arXiv:1910.09729*.
12. **`[Gonen et al. 2019] How Grammatical Gender Affects Noun Representations.pdf`**  
    *Original:* `Gonen_2019_How-Does-Grammatical-Gender-Affect-Noun-Representations.pdf`  
    *Citation:* Hila Gonen, Yova Kementchedjhieva & Yoav Goldberg (2019). "How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?" *Proc. CoNLL 2019*, pp. 463–471.
13. **`[McCurdy & Serbetçi 2020] Grammatical Gender Outweighs Topical Gender Bias.pdf`**  
    *Original:* `2005.08864v1.pdf`  
    *Citation:* Katherine McCurdy & Oğuz Serbetçi (2020). "Grammatical gender associations outweigh topical gender bias in crosslinguistic word embeddings." *arXiv:2005.08864* (WiNLP extended abstract).
14. **`[Omrani Sabbaghi & Caliskan 2022] Disentangling Grammatical Gender Signals in Embeddings.pdf`**  
    *Original:* `2206.01691v1.pdf`  
    *Citation:* Shiva Omrani Sabbaghi & Aylin Caliskan (2022). "Measuring Gender Bias in Word Embeddings of Gendered Languages Requires Disentangling Grammatical Gender Signals." *Proc. ACM AIES 2022*.
15. **`[Stańczak et al. 2023] Causal Influence of Grammatical Gender on Distributional Semantics.pdf`**  
    *Original:* `2311.18567v2.pdf` (duplicate `2311.18567v2-2.pdf` archived)  
    *Citation:* Karolina Stańczak, Kevin Du, Adina Williams, Isabelle Augenstein & Ryan Cotterell (2023/2024). "The Causal Influence of Grammatical Gender on Distributional Semantics." *arXiv:2311.18567* / *TACL 2024*.
16. **`[Flint & Ivanova 2024] Distributional Account of Grammatical Gender Effects.pdf`**  
    *Original:* `Flint_2024_Testing-A-Distributional-Account-Of-Grammatical-Gender-Effects-On-Semantic-Gender-Perception.pdf`  
    *Citation:* George Rocco Flint & Anna A. Ivanova (2024). "Testing a Distributional Account of Grammatical Gender Effects on Semantic Gender Perception." *Proc. 46th Annual Meeting of the Cognitive Science Society*, pp. 2847–2853.
17. **`[Mihaylov & Shtedritski 2024] What an Elegant Bridge - Multilingual LLMs Bias.pdf`**  
    *Original:* `2407.09704v1.pdf`  
    *Citation:* Viktor Mihaylov & Aleksandar Shtedritski (2024). "What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages." *arXiv:2407.09704*.
18. **`[Brinkmann et al. 2025] Latent Grammatical Concepts in Multilingual LLMs.pdf`**  
    *Original:* `2501.06346v2.pdf` (duplicate `2501.06346v2-2.pdf` archived)  
    *Citation:* Jannik Brinkmann, Chris Wendler, Christian Bartelt & Aaron Mueller (2025). "Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages." *arXiv:2501.06346*.
19. **`[Xiao & Li 2026] Estimating Grammatical Gender Directions in Contextual Embeddings.pdf`**  
    *Original:* `2606.30152v1.pdf`  
    *Citation:* Huanping Xiao & Yingji Li (2026). "Estimating Grammatical Gender Directions in Contextual Embeddings under Controlled and Natural Contexts." *arXiv:2606.30152*.

---

### 3. Crosslingual & Latent Representations (`03_Crosslingual_and_Latent_Representations/`)

20. **`[Wendler et al. 2024] Do Llamas Work in English.pdf`**  
    *Original:* `2402.10588v4.pdf`  
    *Citation:* Chris Wendler, Veniamin Veselovsky, Giovanni Monea & Robert West (2024). "Do Llamas Work in English? On the Latent Language of Multilingual Transformers." *arXiv:2402.10588*.
21. **`[Kaplan et al. 2024] From Tokens to Words - Inner Lexicon of LLMs.pdf`**  
    *Original:* `2410.05864v4.pdf`  
    *Citation:* Guy Kaplan, Matanel Oren, Yuval Reif & Roy Schwartz (2024/2025). "From Tokens to Words: On the Inner Lexicon of LLMs." *ICLR 2025* / *arXiv:2410.05864*.
22. **`[Dumas et al. 2024] Separating Tongue from Thought.pdf`**  
    *Original:* `2411.08745v4.pdf`  
    *Citation:* Clément Dumas, Chris Wendler, Veniamin Veselovsky, Giovanni Monea & Robert West (2024). "Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers." *arXiv:2411.08745*.
23. **`[Huang et al. 2025] Cross-Model Transferability on Platonic Representations.pdf`**  
    *Original:* `2501.02009v2.pdf`  
    *Citation:* Youcheng Huang, Chen Huang, Duanyu Feng, Wenqiang Lei & Jiancheng Lv (2025). "Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts." *arXiv:2501.02009*.
24. **`[Pokharel et al. 2026] Cross-Lingual Activation Steering for Multilingual LMs.pdf`**  
    *Original:* `2601.16390v2.pdf`  
    *Citation:* Rhitabrat Pokharel, Ameeta Agrawal & Tanay Nagar (2026). "Cross-Lingual Activation Steering for Multilingual Language Models." *arXiv:2601.16390*.
25. **`[Kirtane & Huang 2026] Language Steering for Multilingual In-Context Learning.pdf`**  
    *Original:* `2602.02326v3.pdf`  
    *Citation:* Neeraja Kirtane & Kuan-Hao Huang (2026). "Language Steering for Multilingual In-Context Learning." *arXiv:2602.02326*.

---

### 4. Steering, Probing & Interpretability (`04_Steering_Probing_and_Interpretability/`)

26. **`[Hewitt & Liang 2019] Designing and Interpreting Probes with Control Tasks.pdf`**  
    *Original:* `1909.03368v1.pdf`  
    *Citation:* John Hewitt & Percy Liang (2019). "Designing and Interpreting Probes with Control Tasks." *Proc. EMNLP 2019* / *arXiv:1909.03368*.
27. **`[Belrose et al. 2023] LEACE - Linear Concept Erasure in Closed Form.pdf`**  
    *Original:* `Belrose_2023_LEACE-Perfect-Linear-Concept-Erasure-In-Closed-Form.pdf`  
    *Citation:* Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff & Stella Biderman (2023). "LEACE: Perfect Linear Concept Erasure in Closed Form." *Proc. NeurIPS 2023* / *arXiv:2306.03819*.
28. **`[Li et al. 2023] Inference-Time Intervention - Eliciting Truthful Answers.pdf`**  
    *Original:* `2306.03341v6.pdf`  
    *Citation:* Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister & Martin Wattenberg (2023). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." *Proc. NeurIPS 2023* / *arXiv:2306.03341*.
29. **`[Turner et al. 2023] Steering Language Models with Activation Engineering.pdf`**  
    *Original:* `2308.10248v5.pdf`  
    *Citation:* Alexander Matt Turner, David Udell, Lisa Thiergart, Juan J. Vazquez, Gavin Leech & Monte MacDiarmid (2023). "Steering Language Models With Activation Engineering." *arXiv:2308.10248*.
30. **`[Xiao et al. 2023] Streaming Language Models with Attention Sinks.pdf`**  
    *Original:* `2309.17453v4.pdf`  
    *Citation:* Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han & Mike Lewis (2023/2024). "Efficient Streaming Language Models with Attention Sinks." *Proc. ICLR 2024* / *arXiv:2309.17453*.
31. **`[Jorgensen et al. 2023] Improving Activation Steering with Mean-Centring.pdf`**  
    *Original:* `2312.03813v1.pdf`  
    *Citation:* Ole Jorgensen, Dylan Cope, Nandi Schoots & Murray Shanahan (2023). "Improving Activation Steering in Language Models with Mean-Centring." *arXiv:2312.03813*.
32. **`[Sun et al. 2024] Massive Activations in Large Language Models.pdf`**  
    *Original:* `2402.17762v2.pdf`  
    *Citation:* Mingjie Sun, Xinlei Chen, J. Zico Kolter & Zhuang Liu (2024). "Massive Activations in Large Language Models." *arXiv:2402.17762*.
33. **`[Buttigieg et al. 2026] Bias Directions in LLMs Capture Confidence Not Fairness.pdf`**  
    *Original:* `2610.08559v1.pdf`  
    *Citation:* Stephanie Buttigieg, Maeve Madigan, Parameswaran Kamalaruban & Stuart Burrell (2026). "Latent space bias directions in LLMs capture confidence, not fairness." *arXiv:2610.08559*.

---

### 5. Lexical Norms & Evaluation (`05_Lexical_Norms_and_Evaluation/`)

34. **`[Brysbaert et al. 2014] Concreteness Ratings for 40 Thousand English Lemmas.pdf`**  
    *Original:* `s13428-013-0403-5.pdf`  
    *Citation:* Marc Brysbaert, Amy Beth Warriner & Victor Kuperman (2014). "Concreteness ratings for 40 thousand generally known English word lemmas." *Behavior Research Methods* 46: 904–911.
35. **`[Scott et al. 2019] The Glasgow Norms - Ratings on Nine Scales.pdf`**  
    *Original:* `s13428-018-1099-3.pdf`  
    *Citation:* Graham G. Scott, Anne Keitel, Marc Becirspahic, Bo Yao & Sara C. Sereno (2019). "The Glasgow Norms: Ratings of 5,500 words on nine scales." *Behavior Research Methods* 51: 1258–1270.
36. **`[Vankrunkelsven et al. 2024] Semantic Gender Norms for 24000 Dutch Words.pdf`**  
    *Original:* `Vankrunkelsven_2024_Semantic-Gender-Norms-For-24000-Dutch-Words.pdf`  
    *Citation:* Hendrik Vankrunkelsven, Yang Yang, Marc Brysbaert, Simon De Deyne & Gert Storms (2024). "Semantic gender: Norms for 24,000 Dutch words and its role in word meaning." *Behavior Research Methods* 56: 113–125.
37. **`[Bartl et al. 2025] Gender-Inclusive Language in Coreference Context.pdf`**  
    *Original:* `2502.13120v1.pdf`  
    *Citation:* Marion Bartl, Thomas Brendan Murphy & Susan Leavy (2025). "Adapting Psycholinguistic Research for LLMs: Gender-inclusive Language in a Coreference Context." *arXiv:2502.13120*.
38. **`[Tumurchuluun et al. 2026] Tracing Stereotypes from Representation to Output.pdf`**  
    *Original:* `2609.08322v1.pdf`  
    *Citation:* Ariun-Erdene Tumurchuluun, Yusser Al Ghussin, Pinzhen Chen, Josef van Genabith & Koel Dutta Chowdhury (2026). "Tracing Stereotypes from Representation to Output in Multilingual LLMs." *arXiv:2609.08322*.

---

## Duplicate Archive (`_archive_duplicates/`)

The following 5 redundant downloads were bit-for-bit identical to active papers and have been moved here:

| Archived File | Retained File | MD5 Hash |
|---|---|---|
| `1910.09729v1.pdf` | `[Kann 2019] Grammatical Gender, Neo-Whorfianism, and Word Embeddings.pdf` | `2223305dd36ac1966834b9aa9151c162` |
| `2311.18567v2-2.pdf` | `[Stańczak et al. 2023] Causal Influence of Grammatical Gender on Distributional Semantics.pdf` | `575cf11a979c7238b1130553529d4f78` |
| `2501.06346v2-2.pdf` | `[Brinkmann et al. 2025] Latent Grammatical Concepts in Multilingual LLMs.pdf` | `d766c7889f92286ed000f996d48f683b` |
| `kurinski2010-2.pdf` | `[Kurinski & Sera 2011] Spanish Grammatical Gender and Object Categorization.pdf` | `40033c4d4a65663f68406dcd39da86e5` |
| `mickanetal2014_kls.pdf` | `[Mickan et al. 2014] Key is a llave is a Schlüssel.pdf` | `ac8633941830084983bcf1faec67e16b` |
