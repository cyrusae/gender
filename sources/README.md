# Sources & Reading List Catalog

Organized repository of 135 research papers on grammatical gender, linguistic relativity, representation learning, activation steering, character generation, chain-of-thought reasoning, and experimental methodology in multilingual large language models.

Standardized for research workflows and Google NotebookLM in `[Author year] Short title.pdf` format across 8 thematic folders.

---

## Folder Organization Overview

| Folder | Focus | Papers |
|---|---|:---:|
| [`01_Human_Psycholinguistics_and_Replications/`](01_Human_Psycholinguistics_and_Replications/) | Human behavioral baselines, original Whorfian studies, sound symbolism, and replication attempts | 19 |
| [`02_Gender_in_Embeddings_and_LLMs/`](02_Gender_in_Embeddings_and_LLMs/) | Grammatical vs. semantic gender in static word vectors, SAEs, and LLM representations | 14 |
| [`03_Crosslingual_and_Latent_Representations/`](03_Crosslingual_and_Latent_Representations/) | Shared cross-lingual geometry, English-pivot hypothesis, detokenization, and latent transfer | 7 |
| [`04_Steering_Probing_and_Interpretability/`](04_Steering_Probing_and_Interpretability/) | Activation addition (CAA), ITI, LEACE concept erasure, probe controls, and confounds | 10 |
| [`05_Lexical_Norms_and_Evaluation/`](05_Lexical_Norms_and_Evaluation/) | Psycholinguistic norm datasets (Glasgow, Dutch), concreteness covariates, and coreference benchmarks | 8 |
| [`06_Story_Generation_and_Personification/`](06_Story_Generation_and_Personification/) | LLM personification, open-ended character generation, gender bias in generated stories, and anthropomorphism | 27 |
| [`07_Reasoning_Mode_and_Chain_of_Thought/`](07_Reasoning_Mode_and_Chain_of_Thought/) | Thinking mode (Qwen3, R1), test-time compute, reasoning faithfulness, overthinking, and persona degradation | 45 |
| ├── [`07A_CoT_Faithfulness_and_Latent_Dynamics/`](07_Reasoning_Mode_and_Chain_of_Thought/07A_CoT_Faithfulness_and_Latent_Dynamics/) | **NB5A**: Internal representations, SAEs, commitment boundaries, and causal faithfulness | 22 |
| └── [`07B_CoT_Compute_Scaling_and_Bias/`](07_Reasoning_Mode_and_Chain_of_Thought/07B_CoT_Compute_Scaling_and_Bias/) | **NB6B**: Test-time compute budget, overthinking, entropy collapse, and bias amplification | 23 |
| [`08_Statistical_and_Experimental_Methodology/`](08_Statistical_and_Experimental_Methodology/) | Equivalence testing, linear mixed models, contrast coding, sample sizing, and preregistration in NLP | 5 |
| `_archive_duplicates/` | Byte-for-byte identical redundant downloads safely archived | 5 |

Total unique papers: **135**.

---

## Complete Paper Index

### 1. Human Psycholinguistics & Replications (`01_Human_Psycholinguistics_and_Replications/`)

1. **`[Bender et al. 2011] Grammatical Gender in German - An Implicit Association Study.pdf`**  
   *Source:* https://doi.org/10.1080/17470218.2011.582128  
   *Citation:* Andrea Bender, Sieghard Beller & Karl Christoph Klauer (2011). "Grammatical gender in German: A case for linguistic relativity?" Quarterly Journal of Experimental Psychology 64(9): 1821–1835. DOI 10.1080/17470218.2011.582128.

2. **`[Bender et al. 2016] Lady Liberty and Godfather Death.pdf`**  
   *Source:* https://doi.org/10.1080/17470218.2015.1021701  
   *Citation:* Andrea Bender, Sieghard Beller & Karl Christoph Klauer (2016). "Lady Liberty and Godfather Death as candidates for linguistic relativity? Scrutinizing the gender congruency effect on personified allegories with explicit and implicit measures." Quarterly Journal of Experimental Psychology 69(1): 48–64 (online 2015). DOI 10.1080/17470218.2015.1021701.

3. **`[Boroditsky & Schmidt 2000] Sex, Syntax, and Semantics.pdf`**  
   *Source:* https://escholarship.org/content/qt0jt9w8zf/qt0jt9w8zf.pdf  
   *Citation:* Lera Boroditsky & Lauren A. Schmidt (2000). "Sex, Syntax, and Semantics." *Proc. 22nd Annual Meeting of the Cognitive Science Society*.

4. **`[Boroditsky 2001] Does Language Shape Thought.pdf`**  
   *Original:* `boroditsky_2001.pdf`  
   *Citation:* Lera Boroditsky (2001). "Does Language Shape Thought?: Mandarin and English Speakers' Conceptions of Time." *Cognitive Psychology* 43(1): 1–22.

5. **`[Boroditsky et al. 2003] Sex, Syntax, and Semantics.pdf`**  
   *Original:* `Boroditsky_ea_2003.pdf`  
   *Citation:* Lera Boroditsky, Lauren A. Schmidt & Webb Phillips (2003). "Sex, Syntax, and Semantics." In *Language in Mind* (Gentner & Goldin-Meadow, eds.), MIT Press, ch. 4, pp. 61–79 *[page range from the local scan; its footer reads pp. 59-80]*.

6. **`[Brand et al. 2025] Constraints on Linguistic Relativity Effects for Grammatical and Conceptual Gender.pdf`**  
   *Citation:* James Brand, Mikuláš Preininger, Adam Kříž & Markéta Ceháková (2025). "Feminine fox, not so feminine box: constraints on linguistic relativity effects for grammatical and conceptual gender." *Language and Cognition* 17: e34. DOI 10.1017/langcog.2025.3.

7. **`[Cassidy et al. 1999] Inferring Gender from Name Phonology.pdf`**  
   *Source:* https://doi.org/10.1037/0096-3445.128.3.362  
   *Citation:* Kimberly Wright Cassidy, Michael H. Kelly & Lee'at J. Sharoni (1999). "Inferring gender from name phonology." Journal of Experimental Psychology: General 128(3): 362–381. DOI 10.1037/0096-3445.128.3.362.

8. **`[Elpers et al. 2022] Registered Replication of Phillips and Boroditsky.pdf`**  
   *Original:* `Elpers, Jensen, _ Holmes 2022 - JML.pdf`  
   *Citation:* Nan Elpers, Greg Jensen & Kevin J. Holmes (2022). "Does grammatical gender affect object concepts? Registered replication of Phillips and Boroditsky (2003)." *Journal of Memory and Language* 127: 104357.

9. **`[Kurinski & Sera 2011] Spanish Grammatical Gender and Object Categorization.pdf`**  
   *Original:* `kurinski2010.pdf` (duplicate `kurinski2010-2.pdf` archived)  
   *Citation:* Elena Kurinski & Maria D. Sera (2011). "Does learning Spanish grammatical gender change English-speaking adults' categorization of inanimate objects?" *Bilingualism: Language and Cognition* 14(2): 203–220.

10. **`[Mahajan & Durgin 2025] Implicit Test of Grammatical Gender Effects.pdf`**  
   *Original:* `A Conceptual Replication Of An Implicit Test Of Grammatical Gende.pdf`  
   *Citation:* D. Mahajan & Frank H. Durgin (2025). "A Conceptual Replication Of An Implicit Test Of Grammatical Gender Effects On Inanimate Concepts." *Language and Cognition* 17: e28, 1–19.

11. **`[Mickan et al. 2014] Key is a llave is a Schlüssel.pdf`**  
   *Original:* `Mickan_2014_Key-Is-A-Llave-Is-A-Schluessel.pdf` (duplicate `mickanetal2014_kls.pdf` archived)  
   *Citation:* Anne Mickan, Maren Schiefke & Anatol Stefanowitsch (2014). "Key is a llave is a Schlüssel: A failure to replicate an experiment from Boroditsky et al. 2003." *Yearbook of the German Cognitive Linguistics Association* 2: 39–50.

12. **`[Pawley 2002] Using He and She for Inanimate Nouns in English.pdf`**  
   *Source:* https://www.researchgate.net/publication/286526774_Using_He_and_She_for_Inanimate_Referents_in_English_Questions_of_Grammar_and_World_View  
   *Citation:* Andrew Pawley (2002). "Using he and she for inanimate nouns in English: Questions of grammar and worldview." In N. J. Enfield (Ed.), Ethnosyntax: Explorations in Language and Culture, pp. 110–137. Oxford University Press. *[The local PDF is a different Pawley paper, "He's a big tree, she's a cold wind: Pronominal gender assignment to inanimates in Australian Vernacular English"; the chapter above is cited in that PDF's reference list.]*

13. **`[Phillips & Boroditsky 2003] Can Quirks of Grammar Affect the Way You Think.pdf`**  
   *Original:* `Can_quirks_of_grammar_affect_the_way_you_think_Gra.pdf`  
   *Citation:* Webb Phillips & Lera Boroditsky (2003). "Can Quirks of Grammar Affect the Way You Think? Grammatical Gender and Object Concepts." *Proc. 25th Annual Meeting of the Cognitive Science Society*.

14. **`[Samuel et al. 2019] Grammatical Gender and Linguistic Relativity.pdf`**  
   *Original:* `s13423-019-01652-3.pdf`  
   *Citation:* Steven Samuel, Geoff Cole & Madeline J. Eacott (2019). "Grammatical gender and linguistic relativity: A systematic review." *Psychonomic Bulletin & Review* 26: 1767–1786.

15. **`[Segel & Boroditsky 2011] Grammar in Art.pdf`**  
   *Source:* https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2010.00244/pdf  
   *Citation:* Edward Segel & Lera Boroditsky (2011). "Grammar in art." *Frontiers in Psychology* 1: 244.

16. **`[Semenuks et al. 2017] Effects of Grammatical Gender on Object Description.pdf`**  
   *Original:* `eScholarship UC item 3j37f380.pdf`  
   *Citation:* Arturs Semenuks, Webb Philips, Ioana Dalca, Cora Kim & Lera Boroditsky (2017). "Effects of Grammatical Gender on Object Description." *Proc. 39th Annual Meeting of the Cognitive Science Society*.

17. **`[Sidhu & Pexman 2015] Sound Symbolism and Gender in First Names.pdf`**  
   *Source:* https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0126809&type=printable  
   *Citation:* David M. Sidhu & Penny M. Pexman (2015). "What's in a name? Sound symbolism and gender in first names." *PLoS ONE* 10(5): e0126809.

18. **`[Sidhu et al. 2022] Higher-Order Factors of Sound Symbolism.pdf`**  
   *Citation:* David M. Sidhu, Gabriella Vigliocco & Penny M. Pexman (2022). "Higher order factors of sound symbolism." *Journal of Memory and Language* 125: 104323.

19. **`[Xiao et al. 2026] Grammatical Gender on Object Conceptualization is Weak.pdf`**  
   *Original:* `Xiao_Cremers_Straboni_Strickland_Peperkamp_(2026)_The_influence_of_grammatical_gender_on_object_conceptualization_is_weak_and_language-dependent.pdf`  
    *Citation:* Hualin Xiao, Alexandre Cremers, Camille Straboni, Brent Strickland & Sharon Peperkamp (2026). "The Influence of Grammatical Gender on Object Conceptualization Is Weak and Language-Dependent." *Cognitive Science* 50: e70176.

---

### 2. Gender in Embeddings & LLMs (`02_Gender_in_Embeddings_and_LLMs/`)

20. **`[Bolukbasi et al. 2016] Man is to Computer Programmer as Woman is to Homemaker.pdf`**  
   *Source:* https://proceedings.neurips.cc/paper_files/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf  
   *Citation:* Bolukbasi et al. (2016). "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings."

21. **`[Brinkmann et al. 2025] Latent Grammatical Concepts in Multilingual LLMs.pdf`**  
   *Original:* `2501.06346v2.pdf` (duplicate `2501.06346v2-2.pdf` archived)  
    *Citation:* Jannik Brinkmann, Chris Wendler, Christian Bartelt & Aaron Mueller (2025). "Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages." *arXiv:2501.06346*.

22. **`[Caliskan et al. 2017] Semantics Derived Automatically Contain Human-like Biases.pdf`**  
   *Source:* https://arxiv.org/pdf/1608.07187.pdf  
   *Citation:* Aylin Caliskan, Joanna J. Bryson & Arvind Narayanan (2017). "Semantics derived automatically from language corpora contain human-like biases." *Science* 356(6334): 183–186.

23. **`[Chaloner & Maldonado 2019] Measuring Gender Bias in Word Embeddings.pdf`**  
   *Source:* https://aclanthology.org/W19-3804.pdf  
   *Citation:* Chaloner & Maldonado (2019). "Measuring Gender Bias in Word Embeddings across Domains and Discovering New Gender Bias Word Categories." *Proc. First Workshop on Gender Bias in NLP (GeBNLP)*, 2019.

24. **`[Flint & Ivanova 2024] Distributional Account of Grammatical Gender Effects.pdf`**  
   *Original:* `Flint_2024_Testing-A-Distributional-Account-Of-Grammatical-Gender-Effects-On-Semantic-Gender-Perception.pdf`  
    *Citation:* George Rocco Flint & Anna A. Ivanova (2024). "Testing a Distributional Account of Grammatical Gender Effects on Semantic Gender Perception." *Proc. 46th Annual Meeting of the Cognitive Science Society*, pp. 2847–2853.

25. **`[Gonen et al. 2019] How Grammatical Gender Affects Noun Representations.pdf`**  
   *Original:* `Gonen_2019_How-Does-Grammatical-Gender-Affect-Noun-Representations.pdf`  
    *Citation:* Hila Gonen, Yova Kementchedjhieva & Yoav Goldberg (2019). "How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?" *Proc. CoNLL 2019*, pp. 463–471.

26. **`[Kann 2019] Grammatical Gender, Neo-Whorfianism, and Word Embeddings.pdf`**  
   *Original:* `Kann_2019_Grammatical-Gender-Neo-Whorfianism-And-Word-Embeddings.pdf` (duplicate `1910.09729v1.pdf` archived)  
    *Citation:* Katharina Kann (2019). "Grammatical Gender, Neo-Whorfianism, and Word Embeddings: A Data-Driven Approach to Linguistic Relativity." *arXiv:1910.09729*.

27. **`[Kotek et al. 2023] Gender Bias and Stereotypes in Large Language Models.pdf`**  
   *Source:* https://arxiv.org/pdf/2308.14921.pdf  
   *Citation:* Hadas Kotek, Rikker Dockum & David Q. Sun (2023). "Gender bias and stereotypes in Large Language Models." *Collective Intelligence Conference (CI '23)*, Delft; arXiv:2308.14921.

28. **`[McCurdy & Serbetçi 2020] Grammatical Gender Outweighs Topical Gender Bias.pdf`**  
   *Original:* `2005.08864v1.pdf`  
    *Citation:* Katherine McCurdy & Oğuz Serbetçi (2020). "Grammatical gender associations outweigh topical gender bias in crosslinguistic word embeddings." *arXiv:2005.08864* (WiNLP extended abstract).

29. **`[Mihaylov & Shtedritski 2024] What an Elegant Bridge - Multilingual LLMs Bias.pdf`**  
   *Original:* `2407.09704v1.pdf`  
    *Citation:* Viktor Mihaylov & Aleksandar Shtedritski (2024). "What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages." *arXiv:2407.09704*.

30. **`[Omrani Sabbaghi & Caliskan 2022] Disentangling Grammatical Gender Signals in Embeddings.pdf`**  
   *Original:* `2206.01691v1.pdf`  
    *Citation:* Shiva Omrani Sabbaghi & Aylin Caliskan (2022). "Measuring Gender Bias in Word Embeddings of Gendered Languages Requires Disentangling Grammatical Gender Signals." *Proc. ACM AIES 2022*.

31. **`[Salinas et al. 2024] Whats in a Name - Auditing LLMs for Bias.pdf`**  
   *Source:* https://arxiv.org/pdf/2402.14875.pdf  
   *Citation:* Salinas et al. (2024). "What's in a Name? Auditing Large Language Models for Race and Gender Bias."

32. **`[Stańczak et al. 2023] Causal Influence of Grammatical Gender on Distributional Semantics.pdf`**  
   *Original:* `2311.18567v2.pdf` (duplicate `2311.18567v2-2.pdf` archived)  
    *Citation:* Karolina Stańczak, Kevin Du, Adina Williams, Isabelle Augenstein & Ryan Cotterell (2023/2024). "The Causal Influence of Grammatical Gender on Distributional Semantics." *arXiv:2311.18567* / *TACL 2024*.

33. **`[Xiao & Li 2026] Estimating Grammatical Gender Directions in Contextual Embeddings.pdf`**  
   *Original:* `2606.30152v1.pdf`  
    *Citation:* Huanping Xiao & Yingji Li (2026). "Estimating Grammatical Gender Directions in Contextual Embeddings under Controlled and Natural Contexts." *arXiv:2606.30152*.

---

### 3. Crosslingual & Latent Representations (`03_Crosslingual_and_Latent_Representations/`)

34. **`[Dumas et al. 2024] Separating Tongue from Thought.pdf`**  
   *Original:* `2411.08745v4.pdf`  
    *Citation:* Clément Dumas, Chris Wendler, Veniamin Veselovsky, Giovanni Monea & Robert West (2024). "Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers." *arXiv:2411.08745*.

35. **`[Flint et al. 2026] Failure of Contextual Invariance in Multilingual Representations.pdf`**  
   *Source:* https://arxiv.org/pdf/2603.23485.pdf  
   *Citation:* Sagar Kumar, Ariel Flint, Luca Maria Aiello & Andrea Baronchelli (2026). "Failure of contextual invariance in large language models." arXiv:2603.23485. *[PDF file name says 'Flint et al.' and 'Multilingual Representations'; the paper is about pronoun selection, not multilingual representations]*

36. **`[Huang et al. 2025] Cross-Model Transferability on Platonic Representations.pdf`**  
   *Original:* `2501.02009v2.pdf`  
    *Citation:* Youcheng Huang, Chen Huang, Duanyu Feng, Wenqiang Lei & Jiancheng Lv (2025). "Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts." *arXiv:2501.02009*.

37. **`[Kaplan et al. 2024] From Tokens to Words - Inner Lexicon of LLMs.pdf`**  
   *Original:* `2410.05864v4.pdf`  
    *Citation:* Guy Kaplan, Matanel Oren, Yuval Reif & Roy Schwartz (2024/2025). "From Tokens to Words: On the Inner Lexicon of LLMs." *ICLR 2025* / *arXiv:2410.05864*.

38. **`[Kirtane & Huang 2026] Language Steering for Multilingual In-Context Learning.pdf`**  
   *Original:* `2602.02326v3.pdf`  
    *Citation:* Neeraja Kirtane & Kuan-Hao Huang (2026). "Language Steering for Multilingual In-Context Learning." *arXiv:2602.02326*.

39. **`[Pokharel et al. 2026] Cross-Lingual Activation Steering for Multilingual LMs.pdf`**  
   *Original:* `2601.16390v2.pdf`  
    *Citation:* Rhitabrat Pokharel, Ameeta Agrawal & Tanay Nagar (2026). "Cross-Lingual Activation Steering for Multilingual Language Models." *arXiv:2601.16390*.

40. **`[Wendler et al. 2024] Do Llamas Work in English.pdf`**  
   *Original:* `2402.10588v4.pdf`  
    *Citation:* Chris Wendler, Veniamin Veselovsky, Giovanni Monea & Robert West (2024). "Do Llamas Work in English? On the Latent Language of Multilingual Transformers." *arXiv:2402.10588*.

---

### 4. Steering, Probing & Interpretability (`04_Steering_Probing_and_Interpretability/`)

41. **`[Belrose et al. 2023] LEACE - Linear Concept Erasure in Closed Form.pdf`**  
   *Original:* `Belrose_2023_LEACE-Perfect-Linear-Concept-Erasure-In-Closed-Form.pdf`  
    *Citation:* Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff & Stella Biderman (2023). "LEACE: Perfect Linear Concept Erasure in Closed Form." *Proc. NeurIPS 2023* / *arXiv:2306.03819*.

42. **`[Buttigieg et al. 2026] Bias Directions in LLMs Capture Confidence Not Fairness.pdf`**  
   *Original:* `2610.08559v1.pdf`  
    *Citation:* Stephanie Buttigieg, Maeve Madigan, Parameswaran Kamalaruban & Stuart Burrell (2026). "Latent space bias directions in LLMs capture confidence, not fairness." *arXiv:2610.08559*.

43. **`[Hewitt & Liang 2019] Designing and Interpreting Probes with Control Tasks.pdf`**  
   *Original:* `1909.03368v1.pdf`  
    *Citation:* John Hewitt & Percy Liang (2019). "Designing and Interpreting Probes with Control Tasks." *Proc. EMNLP 2019* / *arXiv:1909.03368*.

44. **`[Hu & Lévy 2023] Prompting vs Probing - Measuring LLM Knowledge.pdf`**  
   *Source:* https://aclanthology.org/2023.emnlp-main.306.pdf  
   *Citation:* Jennifer Hu & Roger Levy (2023). "Prompting is not a substitute for probability measurements in large language models." *[venue not verified]*

45. **`[Jorgensen et al. 2023] Improving Activation Steering with Mean-Centring.pdf`**  
   *Original:* `2312.03813v1.pdf`  
    *Citation:* Ole Jorgensen, Dylan Cope, Nandi Schoots & Murray Shanahan (2023). "Improving Activation Steering in Language Models with Mean-Centring." *arXiv:2312.03813*.

46. **`[Li et al. 2023] Inference-Time Intervention - Eliciting Truthful Answers.pdf`**  
   *Original:* `2306.03341v6.pdf`  
    *Citation:* Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister & Martin Wattenberg (2023). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." *Proc. NeurIPS 2023* / *arXiv:2306.03341*.

47. **`[Robinson et al. 2024] My Answer is C - First-Token Probabilities vs Full Text.pdf`**  
   *Source:* https://aclanthology.org/2024.findings-acl.441.pdf  
   *Citation:* Xinpeng Wang, Bolei Ma, Chengzhi Hu, Leon Weber-Genzel, Paul Röttger, Frauke Kreuter, Dirk Hovy & Barbara Plank (2024). "'My Answer is C': First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models." *[PDF file name carries the wrong author 'Robinson']*

48. **`[Sun et al. 2024] Massive Activations in Large Language Models.pdf`**  
   *Original:* `2402.17762v2.pdf`  
    *Citation:* Mingjie Sun, Xinlei Chen, J. Zico Kolter & Zhuang Liu (2024). "Massive Activations in Large Language Models." *arXiv:2402.17762*.

49. **`[Turner et al. 2023] Steering Language Models with Activation Engineering.pdf`**  
   *Original:* `2308.10248v5.pdf`  
    *Citation:* Alexander Matt Turner, David Udell, Lisa Thiergart, Juan J. Vazquez, Gavin Leech & Monte MacDiarmid (2023). "Steering Language Models With Activation Engineering." *arXiv:2308.10248*.

50. **`[Xiao et al. 2023] Streaming Language Models with Attention Sinks.pdf`**  
   *Original:* `2309.17453v4.pdf`  
    *Citation:* Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han & Mike Lewis (2023/2024). "Efficient Streaming Language Models with Attention Sinks." *Proc. ICLR 2024* / *arXiv:2309.17453*.

---

### 5. Lexical Norms & Evaluation (`05_Lexical_Norms_and_Evaluation/`)

51. **`[Bartl et al. 2025] Gender-Inclusive Language in Coreference Context.pdf`**  
   *Original:* `2502.13120v1.pdf`  
    *Citation:* Marion Bartl, Thomas Brendan Murphy & Susan Leavy (2025). "Adapting Psycholinguistic Research for LLMs: Gender-inclusive Language in a Coreference Context." *arXiv:2502.13120*.

52. **`[Brysbaert et al. 2014] Concreteness Ratings for 40 Thousand English Lemmas.pdf`**  
   *Original:* `s13428-013-0403-5.pdf`  
    *Citation:* Marc Brysbaert, Amy Beth Warriner & Victor Kuperman (2014). "Concreteness ratings for 40 thousand generally known English word lemmas." *Behavior Research Methods* 46: 904–911.

53. **`[Du et al. 2019] Exploring Human Gender Stereotypes with Word Association Test.pdf`**  
   *Source:* https://aclanthology.org/D19-1635.pdf  
   *Citation:* Yupei Du, Yuanbin Wu & Man Lan (2019). "Exploring Human Gender Stereotypes with Word Association Test."

54. **`[Ducel et al. 2024] Gender Bias in Autoregressive Language Models in French and Italian.pdf`**  
   *Citation:* Fanny Ducel, Aurélie Névéol & Karën Fort (2024). "'You'll be a nurse, my son!' Automatically Assessing Gender Biases in Autoregressive Language Models in French and Italian."

55. **`[Munoz-Garcia et al. 2025] Evaluating Gender Bias in Spanish Language Generation.pdf`**  
   *Source:* https://arxiv.org/pdf/2405.08638.pdf  
   *Citation:* *[UNVERIFIABLE: the local PDF is an unrelated reinforcement-learning paper (vMFER, arXiv:2405.08638); author and title of this entry could not be checked.]* README previously listed: Munoz-Garcia et al. (2025), "Evaluating Gender Bias in Spanish Language Generation."

56. **`[Scott et al. 2019] The Glasgow Norms - Ratings on Nine Scales.pdf`**  
   *Original:* `s13428-018-1099-3.pdf`  
    *Citation:* Graham G. Scott, Anne Keitel, Marc Becirspahic, Bo Yao & Sara C. Sereno (2019). "The Glasgow Norms: Ratings of 5,500 words on nine scales." *Behavior Research Methods* 51: 1258–1270.

57. **`[Tumurchuluun et al. 2026] Tracing Stereotypes from Representation to Output.pdf`**  
   *Original:* `2609.08322v1.pdf`  
    *Citation:* Ariun-Erdene Tumurchuluun, Yusser Al Ghussin, Pinzhen Chen, Josef van Genabith & Koel Dutta Chowdhury (2026). "Tracing Stereotypes from Representation to Output in Multilingual LLMs." *arXiv:2609.08322*.

58. **`[Vankrunkelsven et al. 2024] Semantic Gender Norms for 24000 Dutch Words.pdf`**  
   *Original:* `Vankrunkelsven_2024_Semantic-Gender-Norms-For-24000-Dutch-Words.pdf`  
    *Citation:* Hendrik Vankrunkelsven, Yang Yang, Marc Brysbaert, Simon De Deyne & Gert Storms (2024). "Semantic gender: Norms for 24,000 Dutch words and its role in word meaning." *Behavior Research Methods* 56: 113–125.

---

### 6. Story Generation & Personification (`06_Story_Generation_and_Personification/`)

59. **`[Agarwal et al. 2026] MORPHOGEN - Measuring Grammatical Gender in Generative Models.pdf`**  
   *Source:* https://arxiv.org/pdf/2604.18914.pdf  
   *Citation:* Mehul Aggarwal, Aditya Agarwal, Arnav Goel, Medha Hira & Anubha Gupta (2026). "MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation." arXiv:2604.18914.

60. **`[Benedetto et al. 2026] Beyond Names - Gender Bias in Italian Recommendations.pdf`**  
   *Source:* https://aclanthology.org/2026.eacl-long.264.pdf  
   *Citation:* Luca Benedetto, Antonia Donvito, Alberto Lucchetti, Andrea Cappelli & Paula Buttery (2026). "Beyond Names: How Grammatical Gender Markers Bias LLM-based Educational Recommendations." *EACL 2026* (vol. 1), pp. 5648-5668.

61. **`[Epley et al. 2007] On Seeing Human - Three-Factor Theory of Anthropomorphism.pdf`**  
   *Source:* https://doi.org/10.1037/0033-295X.114.4.864  
   *Citation:* Nicholas Epley, Adam Waytz & John T. Cacioppo (2007). "On seeing human: A three-factor theory of anthropomorphism." Psychological Review 114(4): 864–886. DOI 10.1037/0033-295X.114.4.864.

62. **`[Eyssel & Hegel 2012] Gender Stereotyping of Robots.pdf`**  
   *Source:* https://doi.org/10.1111/j.1559-1816.2012.00937.x  
   *Citation:* Friederike Eyssel & Frank Hegel (2012). "(S)he's got the look: Gender stereotyping of robots." Journal of Applied Social Psychology 42(9): 2213–2230. DOI 10.1111/j.1559-1816.2012.00937.x.

63. **`[Feng 2026] The Illusion of Debiasing - Latent Gender Signals in Personas.pdf`**  
   *Source:* https://arxiv.org/pdf/2609.07117.pdf  
   *Citation:* Ziyue Feng, Hongbo Fang & James A. Evans (2026). "The Illusion of Debiasing: Persona Steering Redistributes Rather Than Reduces Bias in LLMs." arXiv:2609.07117.

64. **`[Gautam et al. 2026] GRUFF - LLM Pronoun Fidelity Reasoning and Biases in German.pdf`**  
   *Source:* https://arxiv.org/pdf/2605.30214.pdf  
   *Citation:* Mewes et al. (2026). GRUFF *[full title and author list: see SUMMARIES.md entry 11; the 'Gautam' attribution was wrong; PDF file name keeps it]*.

65. **`[Grohmann 2009] Gender Dimensions of Brand Personality.pdf`**  
   *Source:* https://doi.org/10.1509/jmkr.46.1.105  
   *Citation:* Bianca Grohmann (2009). "Gender dimensions of brand personality." Journal of Marketing Research 46(1): 105–119. DOI 10.1509/jmkr.46.1.105.

66. **`[Hamilton 2026] Elias in the Lighthouse, Again - Gender Bias in Story Prompts.pdf`**  
   *Source:* https://arxiv.org/pdf/2605.26492.pdf  
   *Citation:* Sil Hamilton & David Mimno (2026). "Elias in the Lighthouse, Again? Diagnosing Low Diversity in LLM Stories." arXiv:2605.26492.

67. **`[Hida et al. 2025] Analyzing and Mitigating Gender Bias in Generated Stories.pdf`**  
   *Source:* https://aclanthology.org/2025.findings-emnlp.783.pdf  
   *Citation:* Rem Hida, Masahiro Kaneko & Naoaki Okazaki (2025). "Social Bias Evaluation for Large Language Models Requires Prompt Variations."

68. **`[Hsu et al. 2024] Animate or Inanimate - That is the Question for LLMs.pdf`**  
   *Citation:* Leonardo Ranaldi, Giulia Pucci & Fabio Massimo Zanzotto (2024). "Animate, or Inanimate, That is the Question for Large Language Models." arXiv:2408.06332. *[PDF file name carries the wrong author 'Hsu']*

69. **`[Ilie 2025] DeepL das Madchen Genus-Sexus Study.pdf`**  
   *Source:* https://talks.stuts.de/system/event_attachments/attachments/000/000/776/original/M%C3%A4dchenAbstract_TizianaIlie_STuTS2025.pdf  
   *Citation:* Ilie (2025). "Maschinelle Übersetzung und die Genus-Sexus-Kongruenz: DeepLs Pronominalisierung von 'das Mädchen'." *[author and venue not visible in the PDF; unverified]*

70. **`[JBCS 2025] Rewriting Stories with LLMs - Gender Bias in Portuguese Narratives.pdf`**  
   *Citation:* Mariana O. Silva, Michele A. Brandão et al. (2025). "Rewriting Stories with LLMs: Gender Bias in Generated Portuguese-language Narratives." *Journal of the Brazilian Computer Society* 31:1, doi 10.5753/jbcs.2025.5799. *[PDF file name uses 'JBCS' as author]*

71. **`[Jain et al. 2026] Personality Shapes Gender Bias in Persona Narratives.pdf`**  
   *Source:* https://arxiv.org/pdf/2604.23600.pdf  
   *Citation:* Tanay Kumar, Aman Chadha, Shreya Gautam et al. (2026). "Personality Shapes Gender Bias in Persona-Conditioned LLM Narratives Across English and Hindi: An Empirical Investigation." arXiv:2604.23600. *[PDF file name says 'Jain']*

72. **`[Klein 2026] Is She Even Relevant - Pronoun Attribution in Open-Ended Generation.pdf`**  
   *Source:* https://arxiv.org/pdf/2605.07622.pdf  
   *Citation:* Jonas Klein, Chiara Manna & Eva Vanmassenhove (2026). "Is She Even Relevant? When BERT Ignores Explicit Gender Cues." arXiv:2605.07622. *[the earlier title ('Pronoun Attribution in Open-Ended Generation') does not match; the paper studies BERT]*

73. **`[Lucy & Bamman 2021] Gender and Representation Bias in GPT-3 Generated Stories.pdf`**  
   *Source:* https://aclanthology.org/2021.nuse-1.5.pdf  
   *Citation:* Lucy & Bamman (2021). "Gender and Representation Bias in GPT-3 Generated Stories."

74. **`[Lutz et al. 2025] The Prompt Makes the Persona - Persona Conditioning Skew.pdf`**  
   *Source:* https://arxiv.org/pdf/2507.16076.pdf  
   *Citation:* Marlene Lutz, Indira Sen, Georg Ahnert, Elisa Rogers & Markus Strohmaier (2025). "The Prompt Makes the Person(a): A Systematic Evaluation of Sociodemographic Persona Prompting for Large Language Models." arXiv:2507.16076.

75. **`[Nass et al. 1997] Are Machines Gender Neutral.pdf`**  
   *Source:* https://doi.org/10.1111/j.1559-1816.1997.tb00275.x  
   *Citation:* Clifford Nass, Youngme Moon & Nancy Green (1997). "Are machines gender neutral? Gender-stereotypic responses to computers with voices." Journal of Applied Social Psychology 27(10): 864–876. DOI 10.1111/j.1559-1816.1997.tb00275.x.

76. **`[Pucci et al. 2026] Neutrality Bites - Gender Representation in AI Animal Stories.pdf`**  
   *Source:* https://arxiv.org/pdf/2606.07969.pdf  
   *Citation:* Imani Finkley, Yuanxi Li & Melanie Walsh (2026). "Neutrality Bites: Gender Representation in AI-Generated Animal Stories." arXiv:2606.07969. *[PDF file name carries the wrong author 'Pucci']*

77. **`[Rejdebrant 2025] How ChatGPT Assigns Gender in Non-Gendered Story Prompts.pdf`**  
   *Source:* http://urn.kb.se/resolve?urn=urn:nbn:se:hv:diva-25343  
   *Citation:* Gunilla Rejdebrant (2025). "Biases in AI-Generated English Texts - How ChatGPT Assigns Gender in Non-Gendered Story Prompts." Research Essay (15 hp), English 61-90 Online (EON200), October 2025. *[institution not verified; earlier first name 'Gustav' was wrong]*

78. **`[Saeed et al. 2025] Beyond Content - Grammatical Gender in Text-to-Image.pdf`**  
   *Source:* https://arxiv.org/pdf/2508.03199.pdf  
   *Citation:* Muhammed Saeed, Shaina Raza, Ashmal Vayani, Muhammad Abdul-Mageed, Ali Emami & Shady Shehata (2025). "Beyond Content: How Grammatical Gender Shapes Visual Representation in Text-to-Image Models." arXiv:2508.03199.

79. **`[Shah et al. 2025] Investigating Gender Bias in Stories via Psychological Stereotypes.pdf`**  
   *Source:* https://arxiv.org/pdf/2508.03292.pdf  
   *Citation:* Shahed Masoudian, Gustavo Escobedo, Hannah Strauss & Markus Schedl (2025). "Investigating Gender Bias in LLM-Generated Stories via Psychological Stereotypes." arXiv:2508.03292. *[PDF file name says 'Shah']*

80. **`[Sukumaran et al. 2024] Grammatical Abstraction Using Few-Shot Learning of Novel Noun Gender.pdf`**  
   *Source:* https://arxiv.org/pdf/2403.10338.pdf  
   *Citation:* Sukumaran et al. (2024). "Investigating Grammatical Abstraction in Language Models Using Few-Shot Learning of Novel Noun Gender."

81. **`[Tang et al. 2025] MISGENDERED+ - Evaluating Pronoun Agreement and Bias in LLMs.pdf`**  
   *Source:* https://arxiv.org/pdf/2508.00788.pdf  
   *Citation:* Xushuo Tang, Yi Ding, Zhengyi Yang et al. (2025). "Do They Understand Them? An Updated Evaluation on Nonbinary Pronoun Handling in Large Language Models." arXiv:2508.00788. *[the name 'MISGENDERED+' is in the PDF file name; not verified as the paper's own]*

82. **`[Wang et al. 2026] CASPER - Character Variety in Generated Stories.pdf`**  
   *Citation:* Anneliese Brei, Abhisheik Sharma, Nicholas Sanaie, Lu Wang & Snigdha Chaturvedi (2026). "CASPER in the Machine: Insights into Character Variety in LLM-Generated Stories." arXiv:2606.22454. *[PDF file name says 'Wang']*

83. **`[Weeber et al. 2026] Effects of Answer Format Variation on Gender Bias.pdf`**  
   *Source:* https://arxiv.org/pdf/2608.17516.pdf  
   *Citation:* Ksenia Merzlyakova, Sebastian Padó & Franziska Weeber (2026). "Effects of Answer Format Variation on Gender Bias Evaluation in Large Language Models." arXiv:2608.17516. *[author order as printed on the PDF's first page; the layout is ambiguous]*

84. **`[Wentworth 2026] Gendered Asymmetries in AI Stories in English and Spanish.pdf`**  
   *Source:* https://doi.org/10.1007/s00146-026-03163-x  
   *Citation:* Isabelle Wentworth (2026). "Gendered asymmetries in AI stories in English and Spanish." AI & Society 41: 8557–8570. DOI 10.1007/s00146-026-03163-x.

85. **`[Zhao et al. 2024] Role-Play Paradox - Stereotype Amplification in Persona LLMs.pdf`**  
   *Source:* https://arxiv.org/pdf/2409.13979.pdf  
   *Citation:* Jinman Zhao et al. (2024). "Role-Play Paradox in Large Language Models: Reasoning Performance Gains and Ethical Dilemmas." arXiv:2409.13979.

---

### 7. Reasoning Mode & Chain of Thought (`07_Reasoning_Mode_and_Chain_of_Thought/`)

86. **`[Bansal et al. 2026] Legibility is Not Interpretability.pdf`**  
   *Source:* https://arxiv.org/pdf/2609.04194.pdf  
   *Citation:* Kevin Du, Alexander Hoyle, Laura R. et al. (2026). "Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-of-Thought Reasoning." *COLM 2026*; arXiv:2609.04194. *[PDF file name says 'Bansal']*

87. **`[Chen et al. 2025] MixReasoning - Multi-Modal Reasoning Steps.pdf`**  
   *Source:* https://arxiv.org/pdf/2510.06052.pdf  
   *Citation:* Haiquan Lu, Gongfan Fang, Xinyin Ma, Qi Li & Xinchao Wang (2025). "MixReasoning: Switching Modes to Think." arXiv:2510.06052. *[PDF file name says 'Chen']*

88. **`[Chen et al. 2026] Thinking vs NoThinking - Sparse Autoencoders Analysis.pdf`**  
   *Source:* https://arxiv.org/pdf/2608.08168.pdf  
   *Citation:* Bo Cheng et al. (2026). "Thinking vs. NoThinking: Towards Interpreting Reasoning Mechanisms of Large Language Models via Sparse Autoencoders." arXiv:2608.08168. *[PDF file name says 'Chen']*

89. **`[Ding et al. 2026] When More Thinking Hurts.pdf`**  
   *Source:* https://arxiv.org/pdf/2604.10739.pdf  
   *Citation:* (2026). "When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling." arXiv:2604.10739. *[authors not shown on the PDF's first page; 'Ding' unverified]*

90. **`[Dua et al. 2026] Commitment Boundary in Test-Time Compute.pdf`**  
   *Source:* https://arxiv.org/pdf/2606.13603.pdf  
   *Citation:* Daniel Scalena, Sara Candussio, Luca Bortolussi, Elisabetta Fersini et al. (2026). "Beyond the Commitment Boundary: Probing Epiphenomenal Chain-of-Thought in Large Reasoning Models." arXiv:2606.13603. *[PDF file name says 'Dua']*

91. **`[Gao et al. 2025] Plan Before You Write.pdf`**  
   *Source:* https://proceedings.nips.cc/paper_files/paper/2025/file/9cd4b3c9f2f1bb0208d36ca986f6f4f9-Paper-Creative_AI_Track.pdf  
   *Citation:* Yoad Lewenberg, Hila Sheftel & Yael Karov (Google Research) (2025). "Plan Before You Write: Improve LLM writing by planning." *[PDF file name says 'Gao']*

92. **`[Gao et al. 2026] Thinking Inertia in Reasoning Models.pdf`**  
   *Source:* https://arxiv.org/pdf/2610.11765.pdf  
   *Citation:* Dianqiao Lei, Kevin Qinghong Lin, Pan Lu et al. (2026). "Thinking Inertia: LLMs Keep Thinking When Told Not To." arXiv:2610.11765. *[PDF file name says 'Gao']*

93. **`[Guo et al. 2025] Underthinking and Overthinking in Reasoning.pdf`**  
   *Source:* https://arxiv.org/pdf/2505.00127.pdf  
   *Citation:* Jinyan Su, Jennifer Healey, Preslav Nakov & Claire Cardie (2025). "Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and correctness in LLMs." arXiv:2505.00127. *[PDF file name says 'Guo']*

94. **`[Huang et al. 2025] Does Thinking More Always Help.pdf`**  
   *Source:* https://arxiv.org/pdf/2506.04210.pdf  
   *Citation:* Soumya Suvra Ghosal, Souradip Chakraborty, et al. (2025). "Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models." arXiv:2506.04210. *[PDF file name says 'Huang']*

95. **`[Jiang et al. 2025] Are Reasoning Models More Faithful.pdf`**  
   *Source:* https://arxiv.org/pdf/2501.08156.pdf  
   *Citation:* James Chua & Owain Evans (2025). "Are DeepSeek R1 and Other Reasoning Models More Faithful?" arXiv:2501.08156. *[PDF file name says 'Jiang']*

96. **`[Kim et al. 2026] Entertain, then Cancel - Pragmatic Interpretation.pdf`**  
   *Source:* https://arxiv.org/pdf/2610.21204.pdf  
   *Citation:* Zhizhuo Kou, Jian Yang, Pengfei Hao, Junyu Luo, Jingshu Peng, Sirui Han & Yike Guo (2026). "Entertain, then Cancel: How Thinking Makes Reasoning Models Read Some and Or Literally." arXiv:2610.21204. *[PDF file name says 'Kim']*

97. **`[Kocijan et al. 2019] Default Reasoning in Number and Gender Assignment.pdf`**  
   *Source:* https://aclanthology.org/K19-1001.pdf  
   *Citation:* Jaap Jumelet, Willem Zuidema & Dieuwke Hupkes (2019). "Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment." *[venue not verified; PDF file name says 'Kocijan']*

98. **`[Kwon et al. 2025] Controlling Thinking Budget in Reasoning Models.pdf`**  
   *Source:* https://aclanthology.org/2025.findings-acl.1274.pdf  
   *Citation:* Tingxu Han, Zhenting Wang et al. (2025). "Token-Budget-Aware LLM Reasoning." *[PDF file name says 'Kwon']*

99. **`[Lanham et al. 2023] Measuring Faithfulness in Chain-of-Thought Reasoning.pdf`**  
   *Source:* https://arxiv.org/pdf/2307.13702.pdf  
   *Citation:* Lanham et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning."

100. **`[Lee et al. 2024] TypedThinker - Structural Planning in Reasoning.pdf`**  
   *Source:* https://arxiv.org/pdf/2410.01952.pdf  
   *Citation:* Danqing Wang, Jianxin Ma, Fei Fang & Lei Li (2024/2025). "TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking." *ICLR 2025*; arXiv:2410.01952. *[PDF file name says 'Lee']*

101. **`[Li et al. 2025] Does Reasoning Introduce Bias.pdf`**  
   *Source:* https://aclanthology.org/2025.findings-emnlp.1006.pdf  
   *Citation:* Xuyang Wu, Jinming Nian et al. (2025). "Does Reasoning Introduce Bias? A Study of Social Bias Evaluation and Mitigation in LLM Reasoning." *[PDF file name says 'Li']*

102. **`[Li et al. 2025] Learning to Reason for Long-Form Story Generation.pdf`**  
   *Source:* https://arxiv.org/pdf/2503.22828.pdf  
   *Citation:* Alexander Gurung & Mirella Lapata (2025). "Learning to Reason for Long-Form Story Generation." *COLM 2025*; arXiv:2503.22828. *[PDF file name says 'Li']*

103. **`[Lin et al. 2025] Inverse Scaling in Test-Time Compute.pdf`**  
   *Source:* https://arxiv.org/pdf/2507.14417.pdf  
   *Citation:* Aryo Pradipta Gema, Alexander Hägele et al. (2025). "Inverse Scaling in Test-Time Compute." *TMLR* (12/2025); arXiv:2507.14417. *[PDF file name says 'Lin']*

104. **`[Liu et al. 2025] R-CHAR - Reasoning Character Framework.pdf`**  
   *Source:* https://aclanthology.org/2025.emnlp-main.1372.pdf  
   *Citation:* Haiming Qin, Jiwei Zhang, Wei Zhang, KeZhong Lu, Mingyang Zhou, Hao Liao & Rui Mao (2025). "R-CHAR: A Metacognition-Driven Framework for Role-Playing in Large Language Models." *[PDF file name says 'Liu']*

105. **`[Liu et al. 2026] How Reasoning Shapes Social Bias in Generated Code.pdf`**  
   *Source:* https://arxiv.org/pdf/2608.06829.pdf  
   *Citation:* Weifeng Sun, Jieke Shi & Zhou Yang (2026). "How Reasoning Shapes Social Bias in LLM-Generated Code?" arXiv:2608.06829. *[PDF file name says 'Liu']*

106. **`[Mao et al. 2026] Instruction-Tuned Models Cannot Sample from Distributions They Describe.pdf`**  
   *Source:* https://arxiv.org/pdf/2607.25292.pdf  
   *Citation:* Chaemin Jang, Jihee Kim & Dongman Lee (2026). "Instruction-Tuned Language Models Cannot Sample from Distributions They Can Describe." arXiv:2607.25292. *[PDF file name says 'Mao']*

107. **`[McKenna et al. 2026] LLM Reasoning Is Latent.pdf`**  
   *Source:* https://arxiv.org/pdf/2604.15726.pdf  
   *Citation:* Wenshuo Wang (2026). "LLM Reasoning Is Latent, Not the Chain of Thought." arXiv:2604.15726 (position paper). *[PDF file name says 'McKenna']*

108. **`[Park et al. 2026] Sampling More, Getting Less - Diversity Collapse.pdf`**  
   *Source:* https://arxiv.org/pdf/2605.11128.pdf  
   *Citation:* Amin Banayeeanzade, Qingchuan Yang, Dhruv Tarsadiya et al. (2026). "Sampling More, Getting Less: Calibration is the Diversity Bottleneck in LLMs." arXiv:2605.11128. *[PDF file name says 'Park']*

109. **`[Prasad et al. 2026] Are Stated Reasoning Steps Causally Load-Bearing.pdf`**  
   *Source:* https://arxiv.org/pdf/2609.27038.pdf  
   *Citation:* Abhiram Bhupatiraju, Rayan Nyaupane et al. (2026). "Are Stated Reasoning Steps Causally Load-Bearing? Activation Patching Says Less Often Than Text Editing." arXiv:2609.27038. *[PDF file name says 'Prasad']*

110. **`[Rao et al. 2025] Agree to Disagree - Meta-Evaluation of LLM Misgendering.pdf`**  
   *Source:* https://arxiv.org/pdf/2504.17075.pdf  
   *Citation:* Arjun Subramonian, Vagrant Gautam, Preethi Seshadri, Dietrich Klakow, Kai-Wei Chang & Yizhou Sun (2025). "Agree to Disagree? A Meta-Evaluation of LLM Misgendering." *COLM 2025*; arXiv:2504.17075. *[PDF file name says 'Rao']*

111. **`[Shaikh et al. 2023] Lets Not Think Step by Step.pdf`**  
   *Source:* https://aclanthology.org/2023.acl-long.244.pdf  
   *Citation:* Omar Shaikh, Hongxin Zhang, William Held, Michael Bernstein & Diyi Yang (2023). "On Second Thought, Let's Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning." *[venue not verified]*

112. **`[Sharma et al. 2026] Sycophancy Mitigated Yet Masked in Reasoning Models.pdf`**  
   *Source:* https://arxiv.org/pdf/2603.16643.pdf  
   *Citation:* Zhaoxin Feng, Zheng Chen, Jianfei Ma, Yip Tin Po, Emmanuele Chersoni & Bo Li (2026). "Good Arguments Against the People Pleasers: How Reasoning Mitigates (Yet Masks) LLM Sycophancy." arXiv:2603.16643. *[PDF file name says 'Sharma']*

113. **`[Song et al. 2026] Entropy Collapse in Long CoT Generation.pdf`**  
   *Source:* https://aclanthology.org/2026.findings-acl.641.pdf  
   *Citation:* Jaeeun Jang, Hansle Lee & Sangmin Kim (2026). "A Few Bad Apples Spoil the Bunch: Preventing Global Entropy Collapse Driven by a Small Set of Tokens in LLM Reasoning." *[PDF file name says 'Song']*

114. **`[Srinivasan et al. 2026] Are You Sure Youre Sure - Confidence and Lexical Diversity.pdf`**  
   *Source:* https://arxiv.org/pdf/2608.13430.pdf  
   *Citation:* (2026). "Are You Sure You're Sure? On the Impact of Instruction Tuning on Confidence and Lexical Diversity." arXiv:2608.13430. *[authors not shown on the first page; 'Srinivasan' unverified]*

115. **`[Sun et al. 2025] Reasoning Path Divergence.pdf`**  
   *Source:* https://arxiv.org/pdf/2510.26122.pdf  
   *Citation:* Feng Ju, Zeyu Qin, Rui Min, Zhitao He, Lingpeng Kong & Yi R. (May) Fung (2025). "Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking." arXiv:2510.26122. *[PDF file name says 'Sun']*

116. **`[Tam et al. 2024] Evaluating Gender Bias via Chain-of-Thought Reasoning.pdf`**  
   *Source:* https://arxiv.org/pdf/2401.15585.pdf  
   *Citation:* Masahiro Kaneko, Danushka Bollegala, Naoaki Okazaki & Timothy Baldwin (2024). "Evaluating Gender Bias in Large Language Models via Chain-of-Thought Prompting." arXiv:2401.15585. *[PDF file name says 'Tam']*

117. **`[Tan et al. 2025] LLMThinkBench - Benchmarking Overthinking.pdf`**  
   *Source:* https://arxiv.org/pdf/2507.04023.pdf  
   *Citation:* Gaurav Srivastava, Aafiya Hussain, Sriram Srinivasan & Xuan Wang (2025). "Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff in Language Models." arXiv:2507.04023. *[PDF file name says 'Tan'/'LLMThinkBench']*

118. **`[Tu et al. 2025] Reasoning Does Not Necessarily Improve Role-Playing.pdf`**  
   *Source:* https://aclanthology.org/2025.findings-acl.537.pdf  
   *Citation:* Xiachong Feng, Longxu Dou & Lingpeng Kong (2025). "Reasoning Does Not Necessarily Improve Role-Playing Ability." *Findings of ACL 2025*, pp. 10301-10314. *[PDF file name says 'Tu']*

119. **`[Venkatesh et al. 2025] CoT Dynamics - Commitment and Information Flow.pdf`**  
   *Source:* https://arxiv.org/pdf/2508.19827.pdf  
   *Citation:* Samuel Lewis-Lim, Xingwei Tan, Zhixue Zhao & Nikolaos Aletras (2025). "Analysing Chain of Thought Dynamics: Active Guidance or Unfaithful Post-hoc Rationalisation?" arXiv:2508.19827. *[PDF file name says 'Venkatesh']*

120. **`[Wang et al. 2025] Demystifying Hybrid Thinking.pdf`**  
   *Source:* https://arxiv.org/pdf/2510.12680.pdf  
   *Citation:* Shouren Wang, Wang Yang, Xianxuan Long, Debargha Ganguly, Qifan Wang, Vipin Chaudhary & Xiaotian Han (2025). "Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?" arXiv:2510.12680.

121. **`[Wang et al. 2025] Thinking in Character.pdf`**  
   *Source:* https://proceedings.neurips.cc/paper_files/paper/2025/file/aacca7b6a20d157112205a44f42821c8-Paper-Conference.pdf  
   *Citation:* Yihong Tang, Kehai Chen, Muyun Yang, Zhengyu Niu, Jing Li, Tiejun Zhao & Min Zhang (2025). "Thinking in Character: Advancing Role-playing Agents with Role-Aware Reasoning." *[PDF file name says 'Wang']*

122. **`[Wu et al. 2025] Do Biased Models Have Biased Thoughts.pdf`**  
   *Source:* https://arxiv.org/pdf/2508.06671.pdf  
   *Citation:* Swati Rajwal, Shivank Garg, Reem Abdel-Salam, Abdelrahman Zayed et al. (2025). "Do Biased Models Have Biased Thoughts?" *COLM 2025*; arXiv:2508.06671. *[PDF file name says 'Wu']*

123. **`[Xu et al. 2026] Language of Thought Shapes Output Diversity.pdf`**  
   *Source:* https://arxiv.org/pdf/2601.11227.pdf  
   *Citation:* Shaoyang Xu & Wenxuan Zhang (2026). "Language of Thought Shapes Output Diversity in Large Language Models." arXiv:2601.11227. *[PDF file name says 'Xu et al.'; earlier title mentioned 'Multilingual Reasoning']*

124. **`[Yang et al. 2026] Language Models are Lazy Poets.pdf`**  
   *Source:* https://arxiv.org/pdf/2610.24223.pdf  
   *Citation:* Jacob A. Matthews & Sil Hamilton (2026). "Language Models are Lazy Poets: Creative Mode Collapse at Scale." arXiv:2610.24223. *[PDF file name says 'Yang']*

125. **`[Zhang et al. 2024] Thinking Before Speaking.pdf`**  
   *Source:* https://arxiv.org/pdf/2409.13752.pdf  
   *Citation:* Baohua Zhang, Yongyi Huang, Wenyao Cui & Huaping Zhang (2024). "Thinking Before Speaking: A Role-playing Model with Mindset." arXiv:2409.13752.

126. **`[Zhang et al. 2026] Mechanics of Bias and Reasoning.pdf`**  
   *Source:* https://arxiv.org/pdf/2605.20410.pdf  
   *Citation:* Edie Pearman, Sophia Osborne, Mira Kandlikar-Bloch, Mina Arzaghi, Florian Carichon & Golnoosh Farnadi (2026). "Mechanics of Bias and Reasoning: Interpreting the Impact of Chain-of-Thought Prompting on Gender Bias in LLMs." arXiv:2605.20410. *[PDF file name says 'Zhang']*

127. **`[Zhao et al. 2026] R2-Write - Reasoning for Creative Writing.pdf`**  
   *Source:* https://arxiv.org/pdf/2604.03004.pdf  
   *Citation:* Wanlong Liu, Bo Zhang, Chenliang Li et al. (2026). "R2-Write: Reflection and Revision for Open-Ended Writing with Deep Reasoning." arXiv:2604.03004. *[PDF file name says 'Zhao']*

128. **`[Zheng et al. 2026] Does Thinking Help Fairness.pdf`**  
   *Source:* https://arxiv.org/pdf/2609.30768.pdf  
   *Citation:* Deng Pan, Joe Germino et al. (2026). "Does Thinking Help Fairness? Reasoning Tokens Resolve Some Biases but Create More." arXiv:2609.30768. *[PDF file name says 'Zheng']*

129. **`[Zhou et al. 2025] Can LLMs Generate Good Stories.pdf`**  
   *Source:* https://arxiv.org/pdf/2506.10161.pdf  
   *Citation:* Yi Wang & Max Kreminski (2025). "Can LLMs Generate Good Stories? Insights and Challenges from a Narrative Planning Perspective." arXiv:2506.10161. *[PDF file name says 'Zhou']*

130. **`[Zhou et al. 2025] OThink-R1 - Optimization in Reasoning.pdf`**  
   *Source:* https://arxiv.org/pdf/2506.02397.pdf  
   *Citation:* Shengjia Zhang, Junjie Wu, Jiawei Chen et al. (2025). "OThink-R1: Intrinsic Fast/Slow Thinking Mode Switching for Over-Reasoning Mitigation." arXiv:2506.02397. *[PDF file name says 'Zhou']*

---

### 8. Statistical & Experimental Methodology (`08_Statistical_and_Experimental_Methodology/`)

131. **`[Baayen et al. 2008] Mixed-Effects Modeling with Crossed Random Effects.pdf`**  
   *Source:* http://www.sfs.uni-tuebingen.de/~hbaayen/publications/baayenDavidsonBates.pdf  
   *Citation:* Baayen et al. (2008). "Mixed-Effects Modeling with Crossed Random Effects for Subjects and Items."

132. **`[Fitts 2023] Precision and Sample Size in Mixed-Effects Models.pdf`**  
   *Source:* https://link.springer.com/content/pdf/10.3758/s13428-022-01896-3.pdf  
   *Citation:* Douglas A. Fitts (2023). "Absolute precision confidence intervals for unstandardized mean differences using sequential stopping rules." *Behavior Research Methods* 55: 1839-1862. *[the earlier title ('Precision and Sample Size in Mixed-Effects Models') does not match]*

133. **`[Lakens & Delacre 2020] Equivalence Testing and the Second Generation P-Value.pdf`**  
   *Citation:* Lakens & Delacre (2020). "Equivalence Testing and the Second Generation P-Value."

134. **`[Schad et al. 2020] How to Capitalize on a Priori Contrasts in LMMs.pdf`**  
   *Source:* https://arxiv.org/pdf/1904.09843.pdf  
   *Citation:* *[UNVERIFIABLE locally: the PDF is Jain et al., 'GestARLite' (arXiv:1904.09843), the wrong paper.]* Intended: Schad, Vasishth, Hohenstein & Kliegl, "How to capitalize on a priori contrasts in linear (mixed) models: A tutorial", arXiv:1807.10451 (published in *Journal of Memory and Language* 110, 2020).

135. **`[van Miltenburg et al. 2021] Preregistering NLP Research.pdf`**  
   *Source:* https://aclanthology.org/2021.naacl-main.51.pdf  
   *Citation:* Emiel van Miltenburg, Chris van der Lee & Emiel Krahmer (2021). "Preregistering NLP research." *NAACL-HLT 2021*, pp. 613-623.

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
