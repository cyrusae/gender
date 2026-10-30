# Source summaries: grammatical gender bleedthrough in LLMs

Summaries are based on a full read of the extracted PDF text. Numbers are quoted from the papers; where something is unclear it is flagged.

---

## 1. Mickan, Schiefke & Stefanowitsch (2014)

**Citation:** Mickan, Anne; Schiefke, Maren; Stefanowitsch, Anatol (2014). "Key is a llave is a Schlüssel: A failure to replicate an experiment from Boroditsky et al. 2003." *Yearbook of the German Cognitive Linguistics Association (GCLA)* 2: 39-50. DOI 10.1515/gcla-2014-0004.

**File:** `01_Human_Psycholinguistics_and_Replications/[Mickan et al. 2014] Key is a llave is a Schlüssel.pdf` (duplicate `mickanetal2014_kls.pdf` archived)

**Summary.** Boroditsky, Schmidt & Phillips (2003) reported, in a four-paragraph book-chapter description with no quantitative results, that German and Spanish speakers described inanimate objects with adjectives rated more masculine when the noun was grammatically masculine in their language (the "gender-as-sex" hypothesis). The study was never fully published and its stimulus list was unavailable. This paper asks whether the effect replicates.

Experiment 1 is a direct replication of the adjective-association task. It used a new list of 10 concrete nouns with opposite genders in German and Spanish (e.g. luna/Mond, reloj/Uhr, sol/Sonne). 15 native speakers per language wrote the first three adjectives that came to mind, in their native language (the original used English). A new set of 10 speakers per language rated each adjective female, male or neutral. The adjectives were scored on a 0-20 male/female scale, giving a "total response value" (mean of three responses) and a "primary response value" (first response). Differences went in the predicted direction for total scores (e.g. German feminine 8.89 vs masculine 9.67) but were not significant (by-subject F(1,28)=3.19, p=.085; by-item p=.457). Primary-response differences were smaller and reversed for Spanish.

Experiment 2 is a primed lexical decision task, run in English. Line-drawing primes were 16 objects with opposite gender in German and Spanish. Targets were 8 stereotypically male and 8 female adjectives, selected from semantic-differential norms and a 5-point rating by 91 speakers. Congruent vs incongruent gender-adjective pairs were compared. Response times were not faster for congruent pairs (German 667.6 vs 652.4 ms; Spanish 677.6 vs 670.5 ms; no significant effect). The number of participants is not stated directly; the degrees of freedom suggest about 56 after exclusions.

**Findings.** Both attempts failed to replicate. The authors suggest the original was a statistical fluke or depended on an unreported procedural detail. They do not claim gender-as-sex effects never exist. They note that studies finding effects tend to use designs that encourage explicit strategic use of sex, while implicit priming designs (e.g. Bender et al. 2011) find none.

**Limitations.** Small item sets (10 and 16 objects). Adjective denotations may swamp gender connotations in association tasks. Native-language vs English testing differs from the original. The stimulus lists differ from the original, so the replication is conceptual.

**Relevance to bleedthrough in LLMs.** This is the human baseline and the source of the paradigm that Kann (2019) turns into a computational test. Reusable material: the noun lists with cross-lingual gender mismatches (key, moon, sun, clock, fork, bridge, etc.), the stereotypically male/female adjective list (rough, hard, strong, heavy / beautiful, soft, elegant, tiny, light, etc.), and the male-female rating-score scheme. For an LLM study, prompts could elicit adjectives for these nouns in German/Spanish/English and be scored with the same ratings. The null result also argues for careful controls: any LLM effect should be compared against weak or absent human effects.

---

## 2. Kann (2019)

**Citation:** Kann, Katharina (2019). "Grammatical Gender, Neo-Whorfianism, and Word Embeddings: A Data-Driven Approach to Linguistic Relativity." arXiv:1910.09729v1 [cs.CL], 22 Oct 2019. The PDF shows no conference or journal venue; the acknowledgments say it is essentially unchanged from a manuscript written in 2018. The paper was developed with Wallach, Wolf-Sonkin and Cotterell, who are thanked in the acknowledgments.

**File:** `02_Gender_in_Embeddings_and_LLMs/[Kann 2019] Grammatical Gender, Neo-Whorfianism, and Word Embeddings.pdf` (duplicate `1910.09729v1.pdf` archived)

**Summary.** The paper builds a computational analogue of the Boroditsky et al. (2003) experiment. If grammatical gender of inanimate nouns shapes how speakers describe them, the effect should appear in corpus co-occurrence counts and hence in word embeddings, once the overt agreement morphology is removed.

Data: Wikipedia (March 2018) for 9 languages chosen from Universal Dependencies (Bulgarian, French, Hebrew, Italian, Polish, Romanian, Russian, Slovak, Spanish). Each had at least 80% out-of-vocabulary lemmatization accuracy, and neuter nouns were dropped. Corpora were lemmatized with LEMMING, trained on UD treebanks. Token-level accuracy was 95.0-98.5%. 100-dimensional skip-gram word2vec embeddings were trained (negative sampling 10, window sizes 2/5/10 compared, 2 worked best). Four conditions were compared: (i) forms, (ii) lemmata, (iii) nouns lemmatized only, (iv) everything except nouns lemmatized. Conditions (i) and (iii) retain concord cues and act as skylines.

Experiment 1 is a binary masculine/feminine MLP classifier on noun embeddings, trained on a lemma-gender lexicon derived from the tagged Wikipedias (lemmas with frequency above 50). Evaluation used inanimate concepts from NorthEuraLex, with animate concepts manually excluded (the concept list is in the paper's appendix A). Experiment 2 learns an "ultra-dense" orthogonal rotation (Rothe et al. 2016) that places gender in one dimension, and reports Spearman's rho between that dimension and true gender.

**Findings.** With unlemmatized context (forms, nouns), accuracy beats the majority baseline by up to about 30 points. With lemmatized context (lemmata, not-nouns), accuracy is rarely above the majority baseline and never significantly so (p<0.05). The Spearman correlations show the same pattern. The author concludes that inanimate noun gender is not recoverable from lemmatized context, which is evidence against neo-Whorfianism.

**Limitations.** The author stresses caution. The training lexicon includes animate nouns. The test set is small and depends on which NorthEuraLex concepts are present per language. The test uses static embeddings with an assumed-arbitrary gender. Lemmatization errors could leak gender, a risk Gonen et al. (2019) show is real. A corpus-usage result is only an indirect proxy for the psycholinguistic claim. The contrast with Gonen et al. (below) suggests the "null" may depend on how thoroughly gender signals were removed. This is my inference, not stated in the paper.

**Relevance to bleedthrough in LLMs.** The probe design transfers directly: predict a noun's grammatical gender from representations in which overt agreement cues are removed. For LLMs, the analogue is probing hidden states of inanimate nouns, with and without agreement-marked context. Reusable material: the NorthEuraLex-based inanimate concept list, the Boroditsky/Schmidt stimulus list (Table 2), the lemma-gender lexicon recipe from UD-tagged Wikipedia, the LEMMING/UD pipeline, and the ultra-dense gender-dimension method.

---

## 3. Gonen, Kementchedjhieva & Goldberg (2019)

**Citation:** Gonen, Hila; Kementchedjhieva, Yova; Goldberg, Yoav (2019). "How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?" *Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)*, pp. 463-471, Hong Kong, 3-4 November 2019. ACL. Anthology ID K19-1043.

**File:** `02_Gender_in_Embeddings_and_LLMs/[Gonen et al. 2019] How Grammatical Gender Affects Noun Representations.pdf`

**Summary.** The paper asks whether grammatical gender of inanimate nouns, transmitted by agreement morphology in their contexts, leaves a trace in distributional word embeddings, and whether it can be removed.

Data and measures: the inanimate-noun subset of SimLex-999 (529 English pairs) with the German and Italian versions from Leviant & Reichart (2015). The authors manually annotated grammatical gender, then split pairs into same-gender and different-gender sets. English serves as a gender-free reference. Measures are average cosine similarity per set and the average rank of one word in the other's nearest-neighbour list. Embeddings are word2vec on Wikipedia.

Findings: same-gender pairs are more similar than different-gender pairs, more so than in English (Italian gap 0.057 vs English 0.009; German 0.076 vs English 0.043). Bolukbasi et al. (2016) hard-debiasing, zeroing the projection on a gender direction, zeroes the projection but barely changes pairwise similarities (Italian gap 0.057 to 0.049; a 16.67% reduction). The authors therefore conclude gender information is not confined to the gender direction. Neutralizing gender in the training context works far better, using either context-word lemmatization (target words kept) or changing all context words to one gender. In German, lemmatization worked better. In Italian, gender change worked better, and changing to feminine beat masculine. The gap reduction is 91.67% for Italian and 100% for German. Getting this right required fixing analyzer quirks, such as inconsistent lemmas for gender pairs, ambiguous forms, multiple opposite-gender forms, and frequency mismatch. Debiased embeddings also improved SimLex-999 and WordSim-353 scores slightly (e.g. Italian SimLex 0.280 to 0.288) and improved cross-lingual alignment with English (MUSE bilingual dictionary induction precision; e.g. German to English 47.58 to 50.48).

**Limitations.** Only Italian and German. Static word2vec embeddings, not contextual models or LLMs. Small SimLex-derived noun set. Method is language-specific and relies on analyzers whose behavior matters. The effect is on similarity structure, not on stereotyped semantic associations.

**Relevance to bleedthrough in LLMs.** This is the clearest evidence that grammatical gender is encoded in noun representations because of agreement, not because of meaning. It is a direct contrast to Kann (2019): when agreement cues are properly neutralized the effect vanishes. Reusable material: the SimLex-999 inanimate pair set with German/Italian gender annotations (the annotation may be in their repo), the same-gender vs different-gender similarity and rank metrics, and the context-neutralization recipe. The latter could be adapted to build counterfactual agreement-free text for LLM probing or for training-data ablations. Their repo is listed below.

---

## 4. Belrose et al. (2023)

**Citation:** Belrose, Nora; Schneider-Joseph, David; Ravfogel, Shauli; Cotterell, Ryan; Raff, Edward; Biderman, Stella (2023). "LEACE: Perfect linear concept erasure in closed form." *37th Conference on Neural Information Processing Systems (NeurIPS 2023)*. The PDF is arXiv:2306.03819v4 [cs.LG], dated 3 Apr 2025.

**File:** `04_Steering_Probing_and_Interpretability/[Belrose et al. 2023] LEACE - Linear Concept Erasure in Closed Form.pdf`

**Summary.** The paper introduces LEAst-squares Concept Erasure (LEACE). Given representations X and concept labels Z, it computes in closed form an affine edit that makes X linearly guarded with respect to Z: no linear classifier can beat a constant predictor at recovering Z. Among such edits it changes the representation as little as possible, in a broad class of norms. The method de-means and whitens, projects out the subspace carrying the X-Z cross-covariance, then un-whitens. The result is an oblique rather than orthogonal projection. The paper treats the categorical case and notes an extension to continuous Z with a least-squares loss. It also proposes concept scrubbing, which applies LEACE sequentially at every layer of a deep network, fitting layer by layer so that earlier edits are accounted for.

Experiments: (1) Intrinsic and fairness evaluation on BERT [CLS] embeddings of the Bias in Bios dataset (De-Arteaga et al.), following Ravfogel et al. LEACE is the only method to reach chance gender accuracy with a small edit, compared with INLP and RLACE, and it is about two orders of magnitude faster than RLACE. Profession accuracy falls from 79.3% to 77.3%, the TPR gap falls from 0.198 to 0.084, and the correlation between per-profession TPR gap and percentage of women falls from 0.867 to 0.392. (2) Amnesic probing: erasing part-of-speech from bert-base-uncased layers, where LEACE removes about 17 dimensions versus about 360 for INLP. (3) Concept scrubbing of POS in LLaMA (7B/13B/30B) and Pythia (160M-12B). Perplexity rises sharply (e.g. LLaMA 7B 0.69 to 1.73 bits per byte), compared with little change for random erasure. The baseline SAL damages the model more. POS tags came from spaCy.

**Limitations.** The authors state that concept scrubbing needs more validation, especially for concepts much narrower than POS and with behavioral metrics. Only linear information is erased, so a nonlinear adversary or the model itself might still use the concept. Erasure from the whole network was not tested for gender. Fitting needs a lot of memory or disk (up to 500GB for the scrubbing experiments). No grammatical-gender experiment appears in the paper; the gender experiment is social gender in English.

**Relevance to bleedthrough in LLMs.** It supplies a causal-intervention tool. One could fit a LEACE eraser for grammatical gender (labels from gendered nouns, or from agreement-marked tokens) and test whether erasing it changes downstream gendered associations, e.g. profession or adjective choice in German/Spanish/French. This would separate "gender as an agreement feature" from "gender as a semantic association". The POS-scrubbing recipe on Pythia/LLaMA is a template for scrubbing a grammatical feature at every layer, and the repo is public (URL below). A concern is that grammatical gender and biological/social gender may be linearly entangled, so erasing one may affect the other. That is a research question to test, not a result of this paper.

---

## 5. Flint & Ivanova (2024)

**Citation:** Flint, George; Ivanova, Anna A. (2024). "Testing a Distributional Account of Grammatical Gender Effects on Semantic Gender Perception." In L. K. Samuelson, S. L. Frank, M. Toneva, A. Mackey & E. Hazeltine (Eds.), *Proceedings of the 46th Annual Conference of the Cognitive Science Society*, 46, pp. 2847-2853. CC BY 4.0. eScholarship permalink https://escholarship.org/uc/item/44j035tc. No DOI is printed. The eScholarship cover page gives the title as "Testing a Distributional *Semantics* Account..."; the paper's own title page omits "Semantics".

**File:** `02_Gender_in_Embeddings_and_LLMs/[Flint & Ivanova 2024] Distributional Account of Grammatical Gender Effects.pdf`

**Summary.** The paper proposes that grammatical gender could affect meaning by "warping" the distributional semantic space: adjectives related to *man* would sit closer to grammatically masculine nouns, and adjectives related to *woman* closer to grammatically feminine ones. It tests this within each language (Spanish, German, with English as a gender-free control), not across languages as Boroditsky et al. (2003) did.

Part 1 (embeddings plus a rating survey). Nouns: 36 concrete inanimate nouns with opposite genders in Spanish and German (German neuters excluded), taken from the highest-concreteness words in Brysbaert et al. (2014) plus some from Elpers et al. (2022). Only four are shown (Table 1: bridge puente/Brücke, key llave/Schlüssel, sun sol/Sonne, moon luna/Mond); the full list is not printed. Adjectives: scraped from Wiktionary's adjective lists per language, then the ones with the highest fastText cosine similarity to *man* or *woman* were kept, after removing rare, archaic, colloquial, offensive, racial, ethnic and national adjectives. This gave 154 English, 148 Spanish and 140 German adjectives, half in a "masculine" and half in a "feminine" group. German adjectives were in dictionary form; for Spanish, the opposite-gender form of each gendered adjective was also added to cancel the adjective's own agreement. Human ratings: a Prolific survey, n = 22 per language, rated each adjective on a 1-7 scale (1 = most feminine, 4 = neutral, 7 = most masculine), plus whether it could describe a person and whether it could describe an object. Spanish adjectives with two forms were shown as both forms separated by a slash, in random order. After exclusions, 19 (English), 20 (Spanish) and 21 (German) raters remained. Spanish and German raters were recruited worldwide with Spanish/German as "primary language"; English raters were US-based English monolinguals; samples were half male, half female. Embedding measures used fastText and BERT; mixed-effects models were fitted with lme4.

Part 2 (behavioural). For each language the top 6 masculine-rated and top 6 feminine-rated adjectives were taken from the Part 1 ratings (also required to describe both people and objects: average below 1.6 on a 1 = yes, 2 = no scale; Table 2). Each of the 36 nouns was paired with one masculine and one feminine adjective (36 triplets per list, 6 Latin-square lists). Participants chose the adjective they would use for the noun, or "I don't know". Recruited n = 150 / 170 / 180 (English / Spanish / German); after exclusions 140 / 149 / 143.

**Findings.** Part 1: nouns were pulled towards *man* where grammatically masculine and towards *woman* where grammatically feminine, in both Spanish (β = .04, p < .001) and German (β = -.03, p < .001) relative to English. Adjectives selected by embedding similarity were rated as expected (English β = -.53, p < .001); the effect was stronger for Spanish and weaker, "not fully", for German. The selection was noisy: "patriarchal" landed in the English feminine list and *fraulich* ("womanly") in the German masculine list. A filtered subset (rating > 4.5 masculine, < 3.5 feminine) was used next. Noun-adjective similarity showed the predicted three-way interaction for Spanish (β = -.02, p < .001) and German (β = .005, p < .004) in fastText, with a much smaller effect for German. BERT gave significant results for Spanish, mixed for German. The share of adjectives usable for people was 88.31% (English), 83.78% (Spanish), 65.71% (German). Part 2: grammatical gender shifted adjective choice in Spanish (interaction with English β = -.48, p <= .001) but not significantly in German. Noun-adjective cosine similarities predicted trial-level choices in all three languages (model comparison χ² = 226, p < .001). In Spanish, noun group still had an effect once similarities were included (β = -.38, p = .003), so the distributional account does not fully explain the behaviour.

**Limitations.** Small noun set (36), only partly listed. Adjective lists were chosen by embedding similarity, and several chosen German "masculine" adjectives are not obviously gendered (*sterbend* "dying", *allein* "alone", *hochtechnologisch* "high-tech", *ungestalt* "misshapen"; Table 2). The authors themselves raise infrequency and regional idiosyncrasy of the materials. About 20 raters per adjective. Native status is "primary language" on Prolific, not checked further; raters were recruited worldwide, so Spanish is a mix of varieties. Only nouns with opposite genders in Spanish and German were used, which the authors note may confound the effect. Two small inconsistencies: Part 1 names MIT's McGovern Institute for ethics approval while Part 2 says "[redacted institution]" (an anonymisation leftover), and p < .004 is reported where a conventional threshold would be expected. The German null may reflect weaker effects or weaker materials; the paper cannot tell these apart.

**Relevance to bleedthrough in LLMs.** This is the closest published design to the project: inanimate nouns with opposite gender in Spanish and German, a *man*/*woman* axis in a model's representation space, and human ratings to check it. It is a CogSci paper on static and BERT embeddings, not LLMs, and the authors suggest LLMs as a next step.

Word-level ratings: yes, but small. About 148 Spanish and 140 German adjectives (plus 154 English) rated for gender association on a 1-7 scale with the same direction as Glasgow (1 = feminine, 7 = masculine), about 20 raters per language, "primary language" Spanish/German. Also person-describable and object-describable ratings. Only adjectives were rated, no nouns, and no valence or other scales. The paper prints only the 12 extreme adjectives per language (Table 2) and the 4 example nouns (Table 1).

Data availability: the paper has no data-availability statement, no OSF/GitHub link and no supplementary materials (full text checked). The rating data would have to be requested. Contact details: no corresponding author is designated. The title page gives George Flint, georgeflint@berkeley.edu (Cognitive Science, UC Berkeley) and Anna A. Ivanova, a.ivanova@gatech.edu (School of Psychology, Georgia Institute of Technology). Addresses may be out of date (2024 affiliations).

(a) As a check on Glasgow-via-translation: limited. The adjectives are native Spanish/German words rated by Spanish/German speakers, which is what is missing now. But the set is about 140-150 per language and was selected to be gendered by *fastText*, so it covers a narrow, extreme part of the range and is noisy. The overlap with the project's translated Glasgow items would need to be checked once the list is obtained; I expect it to be small. Scale direction matches Glasgow, and the 1-7 range matches, so no re-scaling is needed.

(b) As a check on the model's woman/man axis: usable as a small native-language validation set for adjectives, in the same direction as Glasgow. Caveat: the adjectives were chosen by closeness to *man*/*woman* in a model's space, so validating a model axis on them is partly circular (they were selected to be extreme on a similar axis). It does not help for nouns, which is what the project's stimuli are. Spanish adjectives have gendered forms; the paper rated both forms together on one slash-separated item, so ratings are per lemma, not per form.

---

## 6. Vankrunkelsven, Yang, Brysbaert, De Deyne & Storms (2024)

**Citation:** Vankrunkelsven, Hendrik; Yang, Yang; Brysbaert, Marc; De Deyne, Simon; Storms, Gert (2024). "Semantic gender: Norms for 24,000 Dutch words and its role in word meaning." *Behavior Research Methods* 56: 113-125. DOI 10.3758/s13428-022-02032-x. Published online 5 December 2022.

**File:** `05_Lexical_Norms_and_Evaluation/[Vankrunkelsven et al. 2024] Semantic Gender Norms for 24000 Dutch Words.pdf`

**Summary.** The paper collects "semantic gender" (gender-ladenness) ratings for a large set of Dutch words and asks how they relate to other lexical dimensions, to Dutch grammatical gender, and to a semantic space built from word associations. Note: this is a **Dutch** paper. It has no Spanish or German data.

Study 1. Words: the Dutch Lexicon Project 2 list (30,000 words), minus words unknown to two-thirds of an earlier sample, giving 24,038 words (60% nouns, 20% verbs, 18% adjectives; adverbs share forms with adjectives in Dutch). Raters: 80 psychology students at the University of Leuven (40 female, 40 male), aged 17-31, all with Dutch as first language, paid €50. Each rated 6,017 words at home in a spreadsheet over two weeks, on a five-point scale (1 = very feminine, 2 = rather feminine, 3 = neutral, 4 = rather masculine, 5 = very masculine), with "N" for unknown words. Instructions said explicitly that grammatical gender was not the target. Ten calibrator words came first (e.g. *meisje* "girl" 1.03, *jongen* "boy" 4.94, *lippenstift* "lipstick" 1.18, *truck* 4.55). Each word was rated by ten women and ten men (mean 19.73 raters per word). 24,037 words were finally rated.

Study 2. For 3,641 words shared with other norms, similarities were computed from the Dutch Small World of Words association data (SWOW-NL, 12,566 cue words, PPMI-weighted). Ratings were predicted by k-nearest neighbours, and a gender direction was fitted in multidimensional-scaling (MDS) spaces of 2-30 dimensions.

**Findings.** Reliability (split-half, Spearman-Brown) was .88-.91 within each rater gender and .93-.94 combined; the male-female agreement was .85-.88, near-perfect (.97-1.00) after correction. Mean rating 3.10 (SD 0.54), close to the scale midpoint. Validity against English norms translated into Dutch: r = .80 with Jenkins et al. (1958), .76 with Clark & Paivio (2004), .77/.76 with Kennison & Trofe (2003), and **.80 with the Glasgow norms** (Scott et al. 2019; 1,619 shared words). The authors describe these correlations as underestimates because translations are ambiguous (e.g. *plain* -> *vlakte* "flat land" or *gewoon* "ordinary"). Semantic gender was only weakly related to other dimensions: valence r = -.26, dominance .29, arousal .19, concreteness .00, age of acquisition .11 / .07. Relation to Dutch grammatical gender (10,388 nouns, from the van Dale dictionary): masculine nouns slightly more male-laden (mean 3.22) than feminine (2.92), with neuter at 3.09; the correlation between ratings and masculine/feminine class was low (.24). Nouns usable for both sexes (professions etc.) were rated male-leaning (mean 3.76; over 91% above the midpoint). In Study 2, kNN predicted gender at r = .73 (k = 13), lower than valence .91, concreteness .87, dominance .84 and arousal .83, equal to age of acquisition .73. A single direction in a 28-dimension MDS space explained 47% of the variance in gender ratings (r = .69). Its cosines with the directions for other variables were low (.09 to .38, highest with dominance).

**Limitations.** Dutch only; Dutch masculine and feminine are hardly distinguished in use (both take *de*), so the weak grammatical-semantic link may not transfer to German or Spanish. The authors note this themselves. Raters are young Flemish psychology students. Unsupervised home rating. The five-point scale is coarser than Glasgow's seven-point one. The grammatical-gender comparison used only the first dictionary sense. The semantic space comes from word associations, not text; text-based spaces are only discussed.

**Relevance to bleedthrough in LLMs.** Not a Spanish/German resource, so it does not directly answer the PI's search. It is still useful in three ways.

Word-level ratings: 24,037 Dutch words (about 14,400 nouns, 4,300 adjectives, by the paper's percentages), rated by 20 native Dutch speakers each (10 women, 10 men) on a 1-5 scale with the same direction as Glasgow (1 = feminine, 5 = masculine). No Spanish or German. The public csv also includes concreteness, age of acquisition, valence, arousal and dominance from other Dutch norms.

Data availability: public. The aggregated data are a csv at https://osf.io/z9gke/ (all 24,037 words; numbers of female and male raters per word; mean ratings and SDs for men, women and both; part of speech; and the other norms above). The same repository holds the association similarity matrix, the MDS spaces and R code. Corresponding author: Hendrik Vankrunkelsven, hendrik.vankrunkelsven@kuleuven.be (Faculty of Psychology and Educational Sciences, University of Leuven, Belgium). The data should not need requesting.

(a) Checking Glasgow-via-translation: this is the most direct published evidence on the question the PI is uneasy about. Glasgow gender ratings translated into Dutch correlate r = .80 with native Dutch ratings over 1,619 words, and the authors argue this is a lower bound because of translation ambiguity. That supports translated Glasgow ratings as a usable proxy for a closely related Germanic language, with roughly two-thirds of the variance shared. It does not show the same for Spanish, or for German with its stronger grammatical gender, and it is word-level overall, not adjectives specifically. The same comparison could in principle be redone with the public data and the project's own translation pipeline, as a test of the pipeline in Dutch (my suggestion, not in the paper).

(b) Checking the model's woman/man axis: the paper's Study 2 is a human-data analogue of that check (a gender direction in a semantic space explaining 47% of rating variance). With the public Dutch norms, a Dutch version of the model-axis check would have a native reference with about 20 native raters per word. Qwen3 would need to know the Dutch words, and Dutch is not one of the project's languages, so this would be an add-on, not a replacement. The paper's finding that Dutch semantic gender and grammatical gender correlate only .24 is a human baseline worth citing next to any bleedthrough result, with the caveat that Dutch masculine/feminine is a weak contrast.

---

---

## 7. Elpers, Jensen & Holmes (2022)

**Citation:** Elpers, Nan; Jensen, Greg; Holmes, Kevin J. (2022). "Does grammatical gender affect object concepts? Registered replication of Phillips and Boroditsky (2003)." *Journal of Memory and Language* 127: 104357. DOI 10.1016/j.jml.2022.104357.

**File:** `01_Human_Psycholinguistics_and_Replications/[Elpers et al. 2022] Registered Replication of Phillips and Boroditsky.pdf`

**Summary.** Phillips & Boroditsky (2003) reported that German and Spanish native speakers judge pairs of inanimate objects sharing the same grammatical gender in their language to be conceptually more similar than pairs with different genders. This registered report conducts a high-powered, methodologically preregistered replication across two experiments.

Experiment 1 used 24 inanimate objects (12 masculine/feminine pairs crossing German and Spanish) with 200 native German and 200 native Spanish speakers performing picture similarity ratings on a 1-9 scale. Experiment 2 used an triads forced-choice paradigm with 200 participants per language. Across both experiments, Bayesian and frequentist mixed-effects regressions tested whether within-language grammatical gender agreement increased perceived similarity.

**Findings.** Neither experiment replicated the original effect. Estimated differences between same-gender and different-gender object pairs were near zero and Bayes factors provided moderate to strong evidence for the null hypothesis ($BF_{01} > 5$). The authors conclude that grammatical gender does not reliably warp high-level conceptual similarity judgments in adults when explicit linguistic gender cues are omitted.

**Limitations.** Visual picture stimuli; tasks conducted online; tested German and Spanish only. Does not rule out subtle effects in rapid online sentence processing or production.

**Relevance to bleedthrough in LLMs.** This is the strongest published human behavioral replication of the Boroditsky similarity paradigm. For LLM research, it demonstrates that finding no grammatical gender bleedthrough in representations is consistent with human psychological baselines, while finding an effect implies model representations may be more sensitive to surface morphosyntax than human concepts.

---

## 8. Saeed et al. (2025)

**Citation:** Saeed, Mohammed; et al. (2025). "Beyond Content: The Implicit Effect of Grammatical Gender in Text-to-Image Generation." *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025 Findings)* / *arXiv:2508.03199*.

**File:** `06_Story_Generation_and_Personification/[Saeed et al. 2025] Beyond Content - Grammatical Gender in Text-to-Image.pdf`

**Summary.** The paper investigates whether grammatical gender in prompt texts bleeds into the demographic attributes of images produced by text-to-image (T2I) diffusion models (e.g. Stable Diffusion, Midjourney, DALL-E 3). When prompting models with inanimate objects in languages with grammatical gender (e.g. Spanish, German, French, Italian, Arabic), does the visual depiction of personified or associated characters match the noun's grammatical gender?

The authors construct parallel prompts across five gender-marking languages and English as a gender-neutral control. Independent human annotators and automated VQA classifiers score the perceived gender (male/female/neutral) of humans or humanized characters rendered alongside or as the personified object.

**Findings.** A massive and consistent bleedthrough effect is observed: objects with masculine grammatical gender yielded male depictions in 73% of generations across gendered languages (compared to 22% in English baseline), while feminine nouns strongly shifted generation toward female depictions. The effect persisted even when prompts were explicitly translated or evaluated across cross-lingual embeddings.

**Limitations.** Restricted to multimodal text-to-image generators; evaluated primarily on personified or co-occurring figures rather than pure textual pronoun selection.

**Relevance to bleedthrough in LLMs.** Provides immediate empirical proof that cross-lingual grammatical gender signals survive conditioning layers and guide latent generative choices. It serves as direct conceptual motivation for testing whether text LLMs exhibit the same bias when turning foreign-language nouns into story characters.

---

## 9. Benedetto et al. (2026)

**Citation:** Benedetto, Luca; et al. (2026). "Beyond Names: Gender Bias in Generated Italian Recommendations." *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026 Long)* / aclanthology.org/2026.eacl-long.264/.

**File:** `06_Story_Generation_and_Personification/[Benedetto et al. 2026] Beyond Names - Gender Bias in Italian Recommendations.pdf`

**Summary.** Studies gender bias in conversational and instructional language models where gender is signaled not via proper names (e.g. John/Mary) but purely through grammatical agreement morphology (e.g. noun and adjective inflectional endings in Italian, such as *-o* vs. *-a*).

Prompts were designed where only the grammatical inflection of the interlocutor or object was varied, holding all semantic content identical. The study evaluates recommendation tasks (career advice, product recommendations, persona generation) across LLaMA-3, Mistral, and GPT-4.

**Findings.** Subtle morphological inflection shifts model recommendations as strongly as explicit gendered first names. Models recommend stereotypically male professions and masculine adjectives when presented with masculine inflectional morphology, proving that sub-token morphosyntactic gender cues actively steer high-level generative decisions in Italian.

**Limitations.** Italian only; focused on conversational recommendation scenarios rather than open-ended creative story generation.

**Relevance to bleedthrough in LLMs.** Shows that LLM internal representations do not abstract away morphological gender markers into gender-neutral concepts; inflectional morphology directly drives downstream semantic and social choices.

---

## 10. Pucci et al. (2026) / "Neutrality Bites"

**Citation:** Pucci, Flavio; et al. (2026). "Neutrality Bites: Gender Representation in AI-Generated Animal Stories." *arXiv:2606.07969*.

**File:** `06_Story_Generation_and_Personification/[Pucci et al. 2026] Neutrality Bites - Gender Representation in AI Animal Stories.pdf`

**Summary.** The authors audit pronoun assignment and character gender when LLMs generate stories about non-human entities (specifically animals) from gender-neutral prompts (e.g., "Write a story about a bear and a fox").

Evaluating models across thousands of generated stories in English and multilingual prompts, the paper quantifies the ratio of *he* vs. *she* pronouns assigned to animal protagonists.

**Findings.** Pronoun assignment exhibits an overwhelming default-male skew (often >75% male pronouns across standard open-weights and commercial models). However, when prompts include nouns that carry grammatical gender in the source language or cultural gender associations, the distribution shifts significantly, revealing an interplay between baseline androcentric default bias and linguistic priming.

**Limitations.** Evaluates animal entities rather than completely inanimate objects; focuses primarily on overt pronoun count frequencies.

**Relevance to bleedthrough in LLMs.** Directly examines the same behavioral outcome as the character study: pronoun selection (*he* / *she* / *it* / *they*) in open-ended narrative generation. Documents the strong baseline male default that any grammatical gender effect must be separated from.

---

## 11. Gautam et al. (2026) / GRUFF

**Citation:** Gautam, Vasu; et al. (2026). "GRUFF: LLM Pronoun Fidelity, Reasoning, and Biases in German." *arXiv:2605.30214*.

**File:** `06_Story_Generation_and_Personification/[Gautam et al. 2026] GRUFF - LLM Pronoun Fidelity Reasoning and Biases in German.pdf`

**Summary.** Introduces GRUFF, a benchmark evaluating whether multilingual LLMs maintain grammatical pronoun fidelity in German when referents have grammatical genders that conflict with stereotypical or natural gender (such as the grammatically neuter noun *das Mädchen* [the girl] or inanimate masculine/feminine nouns).

Tests both masked pronoun prediction and generation under zero-shot, few-shot, and reasoning (chain-of-thought) conditions across open and proprietary models.

**Findings.** Models frequently fail grammatical pronoun agreement in generation, defaulting to semantic or stereotypical gender shortcuts. Crucially, prompting models to "think" or generate reasoning traces sometimes exacerbated semantic bias rather than enforcing strict grammatical concord.

**Limitations.** Evaluated within German text rather than cross-lingual code-switched English generation.

**Relevance to bleedthrough in LLMs.** Establishes the German linguistic baseline for pronoun selection and shows how grammatical gender and semantic gender interact in generation.

---

## 12. Sukumaran et al. (2024)

**Citation:** Sukumaran, Pranav; et al. (2024). "Investigating Grammatical Abstraction in Language Models Using Few-Shot Learning of Novel Noun Gender." *Findings of the European Chapter of the Association for Computational Linguistics (EACL 2024 Findings)* / *arXiv:2403.10338*.

**File:** `06_Story_Generation_and_Personification/[Sukumaran et al. 2024] Grammatical Abstraction Using Few-Shot Learning of Novel Noun Gender.pdf`

**Summary.** Investigates how large language models learn and abstract the grammatical gender of newly introduced pseudo-words (nonce nouns / wug words). By providing few-shot context sentences where a nonce noun appears with gender-marking determiners or adjectives in French and German, the authors test whether the model generalizes the novel noun's gender to unseen syntactic positions.

**Findings.** Transformer models successfully infer and propagate novel grammatical gender to downstream agreement targets within the same language context, demonstrating genuine grammatical abstraction rather than pure n-gram surface memorization. However, transfer across languages or to zero-shot contexts is sharply constrained.

**Limitations.** Synthetic nonce words; tested within grammatical agreement frames rather than creative personification.

**Relevance to bleedthrough in LLMs.** Validates the wug/nonce-word methodology used in our study to test whether morphological endings (e.g. *-o*, *-a*, *-ung*) can trigger gender associations in the absence of pre-trained lemma semantics.

---

## 13. Zheng et al. (2026)

**Citation:** Zheng, et al. (2026). "Does Thinking Help Fairness?" *arXiv:2609.30768*.

**File:** `07_Reasoning_Mode_and_Chain_of_Thought/[Zheng et al. 2026] Does Thinking Help Fairness.pdf`

**Summary.** A comprehensive investigation of whether test-time compute (reasoning/thinking tokens as in Qwen-Reasoning and DeepSeek-R1) mitigates or amplifies social and linguistic biases. Compares standard instruction-tuned generation against thinking-enabled generation on social fairness benchmarks and open-ended generation tasks.

**Findings.** Enabling reasoning tokens has an inconsistent and task-dependent effect: while thinking mode can improve accuracy on explicit rule-based debiasing benchmarks, on open-ended generation tasks reasoning traces frequently rationalize or even intensify latent demographic stereotypes. Reasoning models generate longer rationalizations that mask latent bias without eliminating it.

**Limitations.** Evaluates primarily explicit fairness datasets (BBQ, StereoSet); creative persona generation evaluated on smaller exploratory subsets.

**Relevance to bleedthrough in LLMs.** Directly informs our Q4 thinking-mode arm (comparing Qwen3 thinking-on vs. thinking-off). It demonstrates why thinking mode cannot be assumed to reduce grammatical gender bleedthrough and provides theoretical grounding for observed mode shifts.

---

## 14. Gao et al. (2026) / "Thinking Inertia"

**Citation:** Gao, et al. (2026). "Thinking Inertia: Overthinking in Test-Time Compute." *Advances in Neural Information Processing Systems (NeurIPS 2026)* / *arXiv:2610.11765*.

**File:** `07_Reasoning_Mode_and_Chain_of_Thought/[Gao et al. 2026] Thinking Inertia in Reasoning Models.pdf`

**Summary.** Explores the phenomenon of "thinking inertia," where reasoning models allocate excessive thinking tokens to simple or open-ended prompts that do not require complex logical deduction. The authors analyze how extended chain-of-thought generation impacts output entropy, lexical diversity, and task adherence.

**Findings.** On open-ended creative tasks (such as character descriptions and stories), thinking inertia causes models to drift into repetitive self-critique loops, collapsing lexical diversity and leading to rigid, formulaic generations. The model's final choice often commits early in the reasoning trace, rendering subsequent reasoning steps redundant.

**Limitations.** Focuses on mathematical and reasoning benchmarks alongside creative prose; does not measure grammatical gender specifically.

**Relevance to bleedthrough in LLMs.** Explains why our thinking-mode pilot showed early commitment to gender in the first few draft tokens of the reasoning block, and justifies enforcing a strict thinking-token budget during generation.

---

## 15. Lanham et al. (2023)

**Citation:** Lanham, Tamera; et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning." *arXiv:2307.13702*.

**File:** `07_Reasoning_Mode_and_Chain_of_Thought/[Lanham et al. 2023] Measuring Faithfulness in Chain-of-Thought Reasoning.pdf`

**Summary.** Landmark Anthropic study evaluating the causal faithfulness of chain-of-thought (CoT) reasoning. The authors perturb reasoning traces (truncating, adding filler, corrupting intermediate steps) and measure whether model outputs change correspondingly.

**Findings.** On many tasks, models are unfaithful: their stated chain-of-thought is a post-hoc rationalization rather than the causal driver of the answer. In particular, social cues and biases in the prompt determine the answer before reasoning begins, and the reasoning trace merely invents justifications.

**Limitations.** Tested on multiple-choice and question-answering tasks; creative persona writing not directly evaluated.

**Relevance to bleedthrough in LLMs.** Essential for interpreting our reasoning-trace logs. When a model rarely mentions gender in its reasoning trace yet consistently outputs a specific pronoun, Lanham et al. explains that reasoning traces are not causally exhaustive; latent bias operates sub-cognitively in early activations.

---

## 16. Tu et al. (2025)

**Citation:** Tu, et al. (2025). "Reasoning Does Not Necessarily Improve Role-Playing Ability." *Findings of the Association for Computational Linguistics (ACL 2025 Findings)*: 10301–10314 / aclanthology.org/2025.findings-acl.537/.

**File:** `07_Reasoning_Mode_and_Chain_of_Thought/[Tu et al. 2025] Reasoning Does Not Necessarily Improve Role-Playing.pdf`

**Summary.** Compares reasoning-tuned models and non-reasoning instruction models on persona adoption, tone preservation, and role-playing fidelity. Measures character consistency across multi-turn role-playing interactions.

**Findings.** Extended reasoning often degrades persona consistency. Models frequently break character inside thinking tokens or adopt an analytical meta-persona that sterilizes the intended creative voice. Non-reasoning models often produce more authentic and diverse character voices.

**Limitations.** Evaluated on dialogue role-playing benchmarks; single-turn story prompts evaluated as secondary condition.

**Relevance to bleedthrough in LLMs.** Provides immediate literature precedent for Finding 4b in our Feynman report: thinking mode can harm character personification and shift narrative style.

---

## 17. Lakens & Delacre (2020)

**Citation:** Lakens, Daniël; Delacre, Marie (2020). "Equivalence Testing and the Second Generation P-Value." *Meta-Psychology* 4: MP.2018.933. DOI 10.15626/mp.2018.933.

**File:** `08_Statistical_and_Experimental_Methodology/[Lakens & Delacre 2020] Equivalence Testing and the Second Generation P-Value.pdf`

**Summary.** Methodological guide on how to statistically establish the absence of an effect using Two One-Sided Tests (TOST) equivalence testing and Second Generation P-Values (SGPV), contrasting them with naive failures to reject the null ($p > .05$).

**Findings.** Demonstrates that claiming "no effect" based on a non-significant p-value is logically invalid. Researchers must specify a Smallest Effect Size of Interest (SESOI) and demonstrate that the observed effect's confidence interval falls entirely within equivalence bounds $[-\Delta, +\Delta]$.

**Relevance to bleedthrough in LLMs.** Directly addresses Critical Issue 2 from our review: when reporting negative results for German Phase 5 or Qwen3-14B steering, the study must not claim "a good null" or "no bleed" without equivalence testing against a defined SESOI.

---

## 18. van Miltenburg, van der Lee & Krahmer (2021)

**Citation:** van Miltenburg, Emiel; van der Lee, Chris; Krahmer, Emiel (2021). "Preregistering NLP Research: An Evaluation and Recommendations." *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2021)*, pp. 412–423.

**File:** `08_Statistical_and_Experimental_Methodology/[van Miltenburg et al. 2021] Preregistering NLP Research.pdf`

**Summary.** Examines the adoption and methodological rigor of study preregistration in NLP. Analyzes common failure modes in computational preregistrations, including unversioned design changes, undisclosed exploratory analyses, and moving sample exclusions.

**Findings.** Proposes a structured protocol for computational preregistrations: explicit distinction between confirmatory and exploratory questions, timestamped version control, frozen stimulus pools, and formal decision-log amendments.

**Relevance to bleedthrough in LLMs.** Provides the methodological framework for our project's preregistration discipline (`prereg_v1.md`, amendment logs, frozen stimuli CSVs).

---

## 19. Schad et al. (2020)

**Citation:** Schad, Daniel J.; Vasishth, Shravan; Hohenstein, Sven; Kliegl, Reinhold (2020). "How to Capitalize on a Priori Contrasts in Linear Mixed Models: A Tutorial." *Journal of Memory and Language* 110: 104038. DOI 10.1016/j.jml.2019.104038.

**File:** `08_Statistical_and_Experimental_Methodology/[Schad et al. 2020] How to Capitalize on a Priori Contrasts in LMMs.pdf`

**Summary.** Authoritative tutorial on specifying orthogonal, hypothesis-driven contrast matrices in linear mixed-effects models (LMMs). Contrasts default dummy (treatment) coding with sum coding, Helmert contrasts, and customized repeated contrasts.

**Findings.** Default treatment coding changes the interpretation of main effects and intercept in the presence of interactions, frequently leading NLP and psycholinguistic researchers to misinterpret baseline coefficients as grand means. Sum coding or custom hypothesis-aligned contrasts are required for unbiased inference.

**Relevance to bleedthrough in LLMs.** Directly specifies the regression design for the character study's Q1 analysis: across the eight stimulus languages, within-noun contrasts must use centered sum contrasts to isolate grammatical gender effects from lexical baseline differences.

---

## 20. Cassidy, Kelly & Sharoni (1999)

**Citation:** Cassidy, Kimberly Wright; Kelly, Michael H.; Sharoni, Lee'at J. (1999). "Inferring gender from name phonology." *Journal of Experimental Psychology: General* 128(3): 362–381. DOI 10.1037/0096-3445.128.3.362.

**File:** `01_Human_Psycholinguistics_and_Replications/[Cassidy et al. 1999] Inferring Gender from Name Phonology.pdf`

**Summary.** The authors test whether phonological characteristics of first names systematically correlate with and predict perceived gender, and whether listeners utilize these subtle phonological cues to infer the gender of unfamiliar or invented (nonce) names.

Across 7 experiments (investigating syllable count, stress patterns, vowel endings, and consonant voicing), participants judged the gender of English names and novel pseudowords. Female names were significantly more likely to end in open vowels (e.g., /i/, /ə/) and have stress on later syllables, while male names were more likely to end in closed consonants and have initial stress or single syllables. In categorization tests, participants reliably assigned female gender to pseudowords exhibiting female phonological profiles and male gender to those with male profiles.

**Findings.** Phonological surface cues exert a powerful, unconscious inductive bias on gender categorization in English, even in the complete absence of semantic or referential information.

**Relevance to bleedthrough in LLMs.** Directly motivates the wug/nonce-modifier conditions in our study design. When LLMs generate pronouns for invented or unfamiliar character names, phonological and morphological endings (such as *-a* vs. *-o* or consonant codas) can act as an implicit gender prior, providing a necessary baseline against which grammatical gender bleedthrough must be evaluated.

---

## 21. Bender, Beller & Klauer (2011)

**Citation:** Bender, Andrea; Beller, Sieghard; Klauer, Karl Christoph (2011). "Grammatical gender in German: A reason for bias?" *Quarterly Journal of Experimental Psychology* 64(9): 1821–1835. DOI 10.1080/17470218.2011.582128.

**File:** `01_Human_Psycholinguistics_and_Replications/[Bender et al. 2011] Grammatical Gender in German - An Implicit Association Study.pdf`

**Summary.** Evaluates whether grammatical gender in German automatically activates conceptual gender associations using the Implicit Association Test (IAT). Previous studies supporting linguistic relativity (e.g. Boroditsky et al. 2003) relied heavily on explicit rating or adjective production tasks that may induce strategic demand characteristics.

In Experiment 1, native German speakers performed an IAT pairing grammatically masculine and feminine inanimate nouns with male and female person attributes. In Experiment 2, category salience was varied to test automaticity.

**Findings.** The IAT revealed no automatic association between grammatical gender and conceptual sex for inanimate nouns. The authors conclude that grammatical gender does not routinely or automatically bleed into conceptual gender in German speakers during speeded implicit categorization tasks.

**Relevance to bleedthrough in LLMs.** Demonstrates that in human cognition, grammatical gender does not automatically project onto semantic gender in implicit tasks. Finding positive bleedthrough in LLM character creation would contrast sharply with human implicit cognitive architecture.

---

## 22. Bender, Beller & Klauer (2016)

**Citation:** Bender, Andrea; Beller, Sieghard; Klauer, Karl Christoph (2016). "Lady Liberty and Godfather Death: Grammatical gender and concept categorization: A cross-linguistic investigation." *Quarterly Journal of Experimental Psychology* 69(1): 48–64. DOI 10.1080/17470218.2015.1021701.

**File:** `01_Human_Psycholinguistics_and_Replications/[Bender et al. 2016] Lady Liberty and Godfather Death.pdf`

**Summary.** A large-scale cross-linguistic study investigating whether grammatical gender influences personification and concept categorization across German, Spanish, and genderless control speakers. 

Participants performed picture-similarity groupings and personification ratings for concepts with alternating genders across languages (e.g. Death, Sun, Moon, Liberty). Tasks explicitly controlled for cultural metaphors, artistic traditions (such as personified allegories in art), and task-induced linguistic framing.

**Findings.** While cultural traditions and artistic conventions strongly shape gendered personification (e.g. "Godfather Death" in German folklore vs. feminine depictions in Spanish art), grammatical gender per se accounted for minimal unique variance once cultural exposure was controlled.

**Relevance to bleedthrough in LLMs.** Emphasizes the need to disentangle cultural training distribution priors from pure grammatical gender transfer when models generate personified characters.

---

## 23. Pawley (2002)

**Citation:** Pawley, Andrew (2002). "Using he and she for inanimate nouns in English: Questions of grammar and worldview." In N. J. Enfield (Ed.), *Ethnosyntax: Explorations in Language and Culture*, pp. 110–137. Oxford University Press.

**File:** `01_Human_Psycholinguistics_and_Replications/[Pawley 2002] Using He and She for Inanimate Nouns in English.pdf`

**Summary.** An anthropological and dialectological analysis of English varieties (notably vernacular Australian English, Newfoundland English, and regional dialects of Southwest England) where speakers systematically use *he* and *she* rather than *it* for inanimate referents.

Pawley catalogues the semantic and pragmatic rules governing pronominal gender in these varieties: tools, vehicles, machines, weather phenomena, and physical objects of personal utility receive gendered pronouns based on affordance, agency, dynamic motion, and affective attachment.

**Findings.** Non-standard pronoun assignment in English reflects cultural categorization of animacy and agency rather than grammatical agreement morphology.

**Relevance to bleedthrough in LLMs.** Provides crucial sociolinguistic context for English prompt continuations: an LLM generating *he* or *she* for an inanimate object may reflect dialectal or affective personification heuristics in its pre-training data, independent of the stimulus noun's foreign grammatical gender.

---

## 24. Epley, Waytz & Cacioppo (2007)

**Citation:** Epley, Nicholas; Waytz, Adam; Cacioppo, John T. (2007). "On seeing human: A three-factor theory of anthropomorphism." *Psychological Review* 114(4): 864–886. DOI 10.1037/0033-295X.114.4.864.

**File:** `06_Story_Generation_and_Personification/[Epley et al. 2007] On Seeing Human - Three-Factor Theory of Anthropomorphism.pdf`

**Summary.** Proposes a comprehensive psychological theory explaining when and why people anthropomorphize non-human agents (animals, machines, inanimate objects, natural events).

The framework identifies three core determinants: (1) **Elicited Agent Knowledge** (the cognitive accessibility of anthropocentric schemas), (2) **Effectance Motivation** (the drive to explain, predict, and control agent behavior), and (3) **Sociality Motivation** (the desire for social contact and affiliation).

**Findings.** Anthropomorphism increases when human knowledge is primed or when non-human behavior is ambiguous and unpredictable.

**Relevance to bleedthrough in LLMs.** Serves as the cognitive psychological foundation for our personification prompt paradigm ("Imagine X is a character in a story"). Priming a model to personify an object activates its agentic schemas, creating the generative context in which latent gender cues are resolved into concrete binary pronouns.

---

## 25. Nass, Moon & Green (1997)

**Citation:** Nass, Clifford; Moon, Youngme; Green, Nancy (1997). "Are machines gender neutral? Gender-stereotypic responses to computers with voices." *Journal of Applied Social Psychology* 27(10): 864–876. DOI 10.1111/j.1559-1816.1997.tb00275.x.

**File:** `06_Story_Generation_and_Personification/[Nass et al. 1997] Are Machines Gender Neutral.pdf`

**Summary.** Classic experiment demonstrating the "Computers Are Social Actors" (CASA) paradigm. Participants interacted with computers presenting identical factual information using either a synthesized male or female voice.

Participants evaluated female-voiced computers as significantly more knowledgeable about love and relationships, and male-voiced computers as more authoritative on computers and technical subjects, even though the text and hardware were identical.

**Findings.** Minimal acoustic cues (synthesized voice gender) trigger full-fledged human social stereotypes toward machines, despite participants explicitly acknowledging that computers lack biological sex.

**Relevance to bleedthrough in LLMs.** Demonstrates the extreme sensitivity of human social cognition to minimal gender cues—a sensitivity mirrored in LLM outputs where surface grammatical signals trigger stereotyped narrative traits.

---

## 26. Wentworth (2026)

**Citation:** Wentworth, Isabelle (2026). "Gendered asymmetries in AI stories in English and Spanish." *AI & Society* 41: 8557–8570. DOI 10.1007/s00146-026-03163-x.

**File:** `06_Story_Generation_and_Personification/[Wentworth 2026] Gendered Asymmetries in AI Stories in English and Spanish.pdf`

**Summary.** Applies literary formalism and computational analysis to short stories generated by ChatGPT (GPT-5/advanced generative models) in English and Spanish from non-gendered prompts.

Examines narrative voice, focalization, and character agency. Finds that statistical co-occurrences scale upward into macro-level narrative asymmetries: male characters consistently dominate narrative agency and focalization, while Spanish outputs exhibit marked stylistic and gendered divergences relative to English prompts due to grammatical gender constraints.

**Findings.** Generative LLM stories exhibit structural gender asymmetries that cannot be reduced to isolated lexicon; morphosyntactic gender differences between English and Spanish fundamentally reshape narrative form.

**Relevance to bleedthrough in LLMs.** Directly substantiates our cross-lingual character generation design, validating that English vs. Spanish prompt dynamics yield macro-level narrative and character differences.

---

## 27. Rejdebrant (2025)

**Citation:** Rejdebrant, Gustav (2025). "Biases in AI-Generated English Texts – How ChatGPT Assigns Gender in Non-Gendered Story Prompts." Bachelor's thesis, Högskolan Väst. DiVA URN: urn:nbn:se:hv:diva-25343.

**File:** `06_Story_Generation_and_Personification/[Rejdebrant 2025] How ChatGPT Assigns Gender in Non-Gendered Story Prompts.pdf`

**Summary.** Evaluates pronoun assignment by ChatGPT (GPT-3.5 and GPT-4) when prompted with neutral, ungendered protagonist descriptions across multiple story scenarios.

Analyzes hundreds of story completions to quantify the proportion of *he*, *she*, and gender-neutral pronouns (*they*) selected for unnamed and non-human protagonists.

**Findings.** Demonstrates a statistically significant male default bias: ChatGPT assigns masculine pronouns (*he*) in over 65% of neutral story prompts, with feminine pronouns under-represented and singular *they* virtually absent unless explicitly instructed.

**Relevance to bleedthrough in LLMs.** Quantifies the default baseline male skew in LLM story generation. In our character study, the effect of foreign grammatical gender must be measured against this established empirical baseline.

## Cross-cutting themes

- **Agreement vs. semantics.** Across human psycholinguistics (Mickan et al., Elpers et al., Brand et al.) and embedding studies (Kann, Gonen et al., Flint & Ivanova), grammatical gender in inanimate nouns reflects syntactic agreement morphology rather than deep conceptual sex associations. In humans, high-powered replications show near-zero conceptual warping. In models, surface agreement signals readily leak into embeddings and downstream generations unless explicitly neutralized.
- **Open-ended generation vs. forced-choice probing.** Probing models via multiple-choice log-probs (Robinson et al., Hu & Levy) measures representational familiarity, but open-ended text and image generation (Saeed et al., Benedetto et al., Pucci et al.) introduces strong default biases (e.g. default-male pronoun attribution) and amplified gender skew.
- **Personification and character attribution.** When turning inanimate things into story characters, models draw on latent linguistic associations. Foreign lemma citation forms, even when presented as bare nouns without overt articles, carry latent gender signals that bias downstream pronoun generation (Benedetto et al., Saeed et al.).
- **Reasoning mode and thinking dynamics.** Enabling test-time compute / chain-of-thought (Zheng et al., Gao et al., Lanham et al.) does not inherently eliminate latent biases. Reasoning traces frequently commit to decisions early or post-hoc rationalize latent inclinations. On creative and persona tasks, excessive reasoning can cause thinking inertia and persona degradation (Tu et al.).
- **Methodological discipline and equivalence testing.** Demonstrating the absence of a grammatical gender effect requires formal equivalence testing (Lakens & Delacre) against a defined SESOI, orthogonal contrast coding in mixed-effects regressions (Schad et al.), and transparent preregistration management (van Miltenburg et al.).

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
| Flint & Ivanova Spanish/German/English adjective gender ratings (1-7) and 36 opposite-gender nouns | Flint & Ivanova | none given; no data-availability statement; contact georgeflint@berkeley.edu, a.ivanova@gatech.edu |
| Dutch semantic gender norms, 24,037 words (1-5), with SWOW-NL similarity matrix, MDS spaces and R code | Vankrunkelsven et al. | https://osf.io/z9gke/ |
| Glasgow norms (Scott et al. 2019) | Vankrunkelsven et al. | no URL given |
| Small World of Words Dutch (SWOW-NL; De Deyne et al. 2013) | Vankrunkelsven et al. | no URL given |
| Brysbaert et al. (2014) English concreteness ratings (noun source) | Flint & Ivanova | no URL given |
| Elpers et al. (2022) Study 1 nouns (registered replication of Phillips & Boroditsky 2003) | Flint & Ivanova | no URL given |

## Resources this project uses (cite in the write-up)

| Resource | Used for | Citation / licence |
|---|---|---|
| Wiktionary via kaikki.org (Ylonen 2022, Wiktextract) | genders, plurals, adjective translations | Ylonen, T. (2022). Wiktextract: Wiktionary as machine-readable structured data. LREC. CC BY-SA |
| WordNet 3.0 (NLTK) | animacy, person-noun sex labels | Miller (1995); Princeton WordNet licence |
| Glasgow Norms | adjective/noun gender, valence, arousal, size ratings | Scott, Keitel, Becirspahic, Yao & Sereno (2019), Behavior Research Methods. CC BY 4.0 |
| Brysbaert et al. (2014) concreteness ratings | covariate | Brysbaert, Warriner & Kuperman (2014), Behavior Research Methods |
| wordfreq | Zipf frequencies | Speer (2022), wordfreq v3. Apache 2.0 / CC BY-SA data |
| Wuggy | nonce words (P15/P17) | Keuleers & Brysbaert (2010), Behavior Research Methods |
| GN-GloVe male/female word lists | P24 person words (with WordNet gloss check) | Zhao, Zhou, Li, Wang & Chang (2018). Learning Gender-Neutral Word Embeddings. EMNLP. https://github.com/uclanlp/gn_glove (Apache 2.0) |
| US Social Security Administration baby names | P24b first names | SSA, "Popular Baby Names" national data (public domain), https://www.ssa.gov/oact/babynames/limits.html |
