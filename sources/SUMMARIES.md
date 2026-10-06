# Source summaries: grammatical gender bleedthrough in LLMs

Summaries are based on a full read of the extracted PDF text. Numbers are quoted from the papers; where something is unclear it is flagged.

---

## 1. Mickan, Schiefke & Stefanowitsch (2014)

**Citation:** Mickan, Anne; Schiefke, Maren; Stefanowitsch, Anatol (2014). "Key is a llave is a Schlüssel: A failure to replicate an experiment from Boroditsky et al. 2003." *Yearbook of the German Cognitive Linguistics Association (GCLA)* 2: 39-50. DOI 10.1515/gcla-2014-0004.

**File:** `mickanetal2014_kls.pdf` -> `Mickan_2014_Key-Is-A-Llave-Is-A-Schluessel.pdf`

**Summary.** Boroditsky, Schmidt & Phillips (2003) reported, in a four-paragraph book-chapter description with no quantitative results, that German and Spanish speakers described inanimate objects with adjectives rated more masculine when the noun was grammatically masculine in their language (the "gender-as-sex" hypothesis). The study was never fully published and its stimulus list was unavailable. This paper asks whether the effect replicates.

Experiment 1 is a direct replication of the adjective-association task. It used a new list of 10 concrete nouns with opposite genders in German and Spanish (e.g. luna/Mond, reloj/Uhr, sol/Sonne). 15 native speakers per language wrote the first three adjectives that came to mind, in their native language (the original used English). A new set of 10 speakers per language rated each adjective female, male or neutral. The adjectives were scored on a 0-20 male/female scale, giving a "total response value" (mean of three responses) and a "primary response value" (first response). Differences went in the predicted direction for total scores (e.g. German feminine 8.89 vs masculine 9.67) but were not significant (by-subject F(1,28)=3.19, p=.085; by-item p=.457). Primary-response differences were smaller and reversed for Spanish.

Experiment 2 is a primed lexical decision task, run in English. Line-drawing primes were 16 objects with opposite gender in German and Spanish. Targets were 8 stereotypically male and 8 female adjectives, selected from semantic-differential norms and a 5-point rating by 91 speakers. Congruent vs incongruent gender-adjective pairs were compared. Response times were not faster for congruent pairs (German 667.6 vs 652.4 ms; Spanish 677.6 vs 670.5 ms; no significant effect). The number of participants is not stated directly; the degrees of freedom suggest about 56 after exclusions.

**Findings.** Both attempts failed to replicate. The authors suggest the original was a statistical fluke or depended on an unreported procedural detail. They do not claim gender-as-sex effects never exist. They note that studies finding effects tend to use designs that encourage explicit strategic use of sex, while implicit priming designs (e.g. Bender et al. 2011) find none.

**Limitations.** Small item sets (10 and 16 objects). Adjective denotations may swamp gender connotations in association tasks. Native-language vs English testing differs from the original. The stimulus lists differ from the original, so the replication is conceptual.

**Relevance to bleedthrough in LLMs.** This is the human baseline and the source of the paradigm that Kann (2019) turns into a computational test. Reusable material: the noun lists with cross-lingual gender mismatches (key, moon, sun, clock, fork, bridge, etc.), the stereotypically male/female adjective list (rough, hard, strong, heavy / beautiful, soft, elegant, tiny, light, etc.), and the male-female rating-score scheme. For an LLM study, prompts could elicit adjectives for these nouns in German/Spanish/English and be scored with the same ratings. The null result also argues for careful controls: any LLM effect should be compared against weak or absent human effects.

---

## 2. Kann (2019)

**Citation:** Kann, Katharina (2019). "Grammatical Gender, Neo-Whorfianism, and Word Embeddings: A Data-Driven Approach to Linguistic Relativity." arXiv:1910.09729v1 [cs.CL], 22 Oct 2019. The PDF shows no conference or journal venue; the acknowledgments say it is essentially unchanged from a manuscript written in 2018. The paper was developed with Wallach, Wolf-Sonkin and Cotterell, who are thanked in the acknowledgments.

**File:** `1910.09729v1.pdf` -> `Kann_2019_Grammatical-Gender-Neo-Whorfianism-And-Word-Embeddings.pdf`

**Summary.** The paper builds a computational analogue of the Boroditsky et al. (2003) experiment. If grammatical gender of inanimate nouns shapes how speakers describe them, the effect should appear in corpus co-occurrence counts and hence in word embeddings, once the overt agreement morphology is removed.

Data: Wikipedia (March 2018) for 9 languages chosen from Universal Dependencies (Bulgarian, French, Hebrew, Italian, Polish, Romanian, Russian, Slovak, Spanish). Each had at least 80% out-of-vocabulary lemmatization accuracy, and neuter nouns were dropped. Corpora were lemmatized with LEMMING, trained on UD treebanks. Token-level accuracy was 95.0-98.5%. 100-dimensional skip-gram word2vec embeddings were trained (negative sampling 10, window sizes 2/5/10 compared, 2 worked best). Four conditions were compared: (i) forms, (ii) lemmata, (iii) nouns lemmatized only, (iv) everything except nouns lemmatized. Conditions (i) and (iii) retain concord cues and act as skylines.

Experiment 1 is a binary masculine/feminine MLP classifier on noun embeddings, trained on a lemma-gender lexicon derived from the tagged Wikipedias (lemmas with frequency above 50). Evaluation used inanimate concepts from NorthEuraLex, with animate concepts manually excluded (the concept list is in the paper's appendix A). Experiment 2 learns an "ultra-dense" orthogonal rotation (Rothe et al. 2016) that places gender in one dimension, and reports Spearman's rho between that dimension and true gender.

**Findings.** With unlemmatized context (forms, nouns), accuracy beats the majority baseline by up to about 30 points. With lemmatized context (lemmata, not-nouns), accuracy is rarely above the majority baseline and never significantly so (p<0.05). The Spearman correlations show the same pattern. The author concludes that inanimate noun gender is not recoverable from lemmatized context, which is evidence against neo-Whorfianism.

**Limitations.** The author stresses caution. The training lexicon includes animate nouns. The test set is small and depends on which NorthEuraLex concepts are present per language. The test uses static embeddings with an assumed-arbitrary gender. Lemmatization errors could leak gender, a risk Gonen et al. (2019) show is real. A corpus-usage result is only an indirect proxy for the psycholinguistic claim. The contrast with Gonen et al. (below) suggests the "null" may depend on how thoroughly gender signals were removed. This is my inference, not stated in the paper.

**Relevance to bleedthrough in LLMs.** The probe design transfers directly: predict a noun's grammatical gender from representations in which overt agreement cues are removed. For LLMs, the analogue is probing hidden states of inanimate nouns, with and without agreement-marked context. Reusable material: the NorthEuraLex-based inanimate concept list, the Boroditsky/Schmidt stimulus list (Table 2), the lemma-gender lexicon recipe from UD-tagged Wikipedia, the LEMMING/UD pipeline, and the ultra-dense gender-dimension method.

---

## 3. Gonen, Kementchedjhieva & Goldberg (2019)

**Citation:** Gonen, Hila; Kementchedjhieva, Yova; Goldberg, Yoav (2019). "How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?" *Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)*, pp. 463-471, Hong Kong, 3-4 November 2019. ACL. Anthology ID K19-1043.

**File:** `K19-1043.pdf` -> `Gonen_2019_How-Does-Grammatical-Gender-Affect-Noun-Representations.pdf`

**Summary.** The paper asks whether grammatical gender of inanimate nouns, transmitted by agreement morphology in their contexts, leaves a trace in distributional word embeddings, and whether it can be removed.

Data and measures: the inanimate-noun subset of SimLex-999 (529 English pairs) with the German and Italian versions from Leviant & Reichart (2015). The authors manually annotated grammatical gender, then split pairs into same-gender and different-gender sets. English serves as a gender-free reference. Measures are average cosine similarity per set and the average rank of one word in the other's nearest-neighbour list. Embeddings are word2vec on Wikipedia.

Findings: same-gender pairs are more similar than different-gender pairs, more so than in English (Italian gap 0.057 vs English 0.009; German 0.076 vs English 0.043). Bolukbasi et al. (2016) hard-debiasing, zeroing the projection on a gender direction, zeroes the projection but barely changes pairwise similarities (Italian gap 0.057 to 0.049; a 16.67% reduction). The authors therefore conclude gender information is not confined to the gender direction. Neutralizing gender in the training context works far better, using either context-word lemmatization (target words kept) or changing all context words to one gender. In German, lemmatization worked better. In Italian, gender change worked better, and changing to feminine beat masculine. The gap reduction is 91.67% for Italian and 100% for German. Getting this right required fixing analyzer quirks, such as inconsistent lemmas for gender pairs, ambiguous forms, multiple opposite-gender forms, and frequency mismatch. Debiased embeddings also improved SimLex-999 and WordSim-353 scores slightly (e.g. Italian SimLex 0.280 to 0.288) and improved cross-lingual alignment with English (MUSE bilingual dictionary induction precision; e.g. German to English 47.58 to 50.48).

**Limitations.** Only Italian and German. Static word2vec embeddings, not contextual models or LLMs. Small SimLex-derived noun set. Method is language-specific and relies on analyzers whose behavior matters. The effect is on similarity structure, not on stereotyped semantic associations.

**Relevance to bleedthrough in LLMs.** This is the clearest evidence that grammatical gender is encoded in noun representations because of agreement, not because of meaning. It is a direct contrast to Kann (2019): when agreement cues are properly neutralized the effect vanishes. Reusable material: the SimLex-999 inanimate pair set with German/Italian gender annotations (the annotation may be in their repo), the same-gender vs different-gender similarity and rank metrics, and the context-neutralization recipe. The latter could be adapted to build counterfactual agreement-free text for LLM probing or for training-data ablations. Their repo is listed below.

---

## 4. Belrose et al. (2023)

**Citation:** Belrose, Nora; Schneider-Joseph, David; Ravfogel, Shauli; Cotterell, Ryan; Raff, Edward; Biderman, Stella (2023). "LEACE: Perfect linear concept erasure in closed form." *37th Conference on Neural Information Processing Systems (NeurIPS 2023)*. The PDF is arXiv:2306.03819v4 [cs.LG], dated 3 Apr 2025.

**File:** `2306.03819v4.pdf` -> `Belrose_2023_LEACE-Perfect-Linear-Concept-Erasure-In-Closed-Form.pdf`

**Summary.** The paper introduces LEAst-squares Concept Erasure (LEACE). Given representations X and concept labels Z, it computes in closed form an affine edit that makes X linearly guarded with respect to Z: no linear classifier can beat a constant predictor at recovering Z. Among such edits it changes the representation as little as possible, in a broad class of norms. The method de-means and whitens, projects out the subspace carrying the X-Z cross-covariance, then un-whitens. The result is an oblique rather than orthogonal projection. The paper treats the categorical case and notes an extension to continuous Z with a least-squares loss. It also proposes concept scrubbing, which applies LEACE sequentially at every layer of a deep network, fitting layer by layer so that earlier edits are accounted for.

Experiments: (1) Intrinsic and fairness evaluation on BERT [CLS] embeddings of the Bias in Bios dataset (De-Arteaga et al.), following Ravfogel et al. LEACE is the only method to reach chance gender accuracy with a small edit, compared with INLP and RLACE, and it is about two orders of magnitude faster than RLACE. Profession accuracy falls from 79.3% to 77.3%, the TPR gap falls from 0.198 to 0.084, and the correlation between per-profession TPR gap and percentage of women falls from 0.867 to 0.392. (2) Amnesic probing: erasing part-of-speech from bert-base-uncased layers, where LEACE removes about 17 dimensions versus about 360 for INLP. (3) Concept scrubbing of POS in LLaMA (7B/13B/30B) and Pythia (160M-12B). Perplexity rises sharply (e.g. LLaMA 7B 0.69 to 1.73 bits per byte), compared with little change for random erasure. The baseline SAL damages the model more. POS tags came from spaCy.

**Limitations.** The authors state that concept scrubbing needs more validation, especially for concepts much narrower than POS and with behavioral metrics. Only linear information is erased, so a nonlinear adversary or the model itself might still use the concept. Erasure from the whole network was not tested for gender. Fitting needs a lot of memory or disk (up to 500GB for the scrubbing experiments). No grammatical-gender experiment appears in the paper; the gender experiment is social gender in English.

**Relevance to bleedthrough in LLMs.** It supplies a causal-intervention tool. One could fit a LEACE eraser for grammatical gender (labels from gendered nouns, or from agreement-marked tokens) and test whether erasing it changes downstream gendered associations, e.g. profession or adjective choice in German/Spanish/French. This would separate "gender as an agreement feature" from "gender as a semantic association". The POS-scrubbing recipe on Pythia/LLaMA is a template for scrubbing a grammatical feature at every layer, and the repo is public (URL below). A concern is that grammatical gender and biological/social gender may be linearly entangled, so erasing one may affect the other. That is a research question to test, not a result of this paper.

---

## Cross-cutting themes

- **Agreement vs. semantics.** Kann and Gonen et al. converge on the central distinction: gender in noun representations is largely a distributional echo of agreement morphology. Kann finds no recoverable gender in lemmatized context. Gonen et al. show that gender signal remains until context is carefully neutralized. Mickan et al. find no human behavioral evidence of a strong gender-as-sex effect. For LLMs, the task is to separate surface/agreement leakage from semantic association.
- **Controlling leakage.** Both embedding papers show how fragile controls are: lemmatizer quirks, ambiguous forms, frequency mismatches. Any LLM test needs equally careful controls, such as stimuli with agreement cues removed or balanced.
- **Debiasing limits.** Projection-based debiasing (Bolukbasi) fails to remove grammatical-gender structure (Gonen et al.). LEACE offers a stronger linear alternative, but Belrose et al. only test it on social gender in English and on POS.
- **Inanimate noun stimuli.** Several lists exist that cross gender between languages: Boroditsky/Schmidt (via Kann, Table 2), Mickan et al.'s two lists, SimLex-999 inanimate pairs, and NorthEuraLex concepts.
- **Languages covered.** Human studies: German, Spanish (English for the second Mickan experiment). Kann: 9 languages (bg, es, fr, he, it, pl, ro, ru, sk). Gonen: German, Italian. Belrose: English only. None of the four studies an LLM for grammatical gender directly, so there is an open gap.

## Datasets, word lists and code mentioned

| Resource | Where mentioned | URL (if given in the papers) |
|---|---|---|
| LEACE code | Belrose et al. | https://github.com/EleutherAI/concept-erasure |
| Gonen et al. code and debiased embeddings | Gonen et al. | https://github.com/gonenhila/grammatical_gender |
| NorthEuraLex (multi-way concept-aligned dictionary, v0.9) | Kann | http://www.northeuralex.org/ |
| LEMMING lemmatizer / morphological tagger | Kann | http://cistern.cis.lmu.de/lemming/ |
| Universal Dependencies treebanks | Kann, Belrose | no URL given in the papers |
| SimLex-999 with German and Italian versions (Hill et al. 2015; Leviant & Reichart 2015); WordSim-353 | Gonen et al. | no URL given |
| Boroditsky & Schmidt (2000) stimulus list, Appendix A (18-word subset shown in Kann Table 2) | Kann | no URL given; the original 24-item list is not published in Boroditsky et al. 2003 |
| Mickan et al. stimulus and adjective lists | Mickan et al. | in the paper text; the underlying theses (Mickan 2013; Schiefke 2011) are available from the authors on request |
| Bias in Bios (De-Arteaga et al.) | Belrose et al. | no URL given |
| MUSE alignment system (Conneau et al. 2018) | Gonen et al. | no URL given |
| Pythia and LLaMA models; Pile validation split and RedPajama | Belrose et al. | no URL given |
| spaCy POS tagger | Belrose et al. | no URL given |
| Bolukbasi et al. (2016) gender pairs and debiasing | Gonen et al., Kann | no URL given |
