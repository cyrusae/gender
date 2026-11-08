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

**Summary.** Phillips & Boroditsky (2003, "P&B") reported that Spanish and German speakers judged person-object picture pairs as more similar when the biological gender of the person matched the grammatical gender of the object's label, and that English speakers showed the same pattern after learning gender-like categories. This registered replication reproduces two of P&B's experiments (total N = 375).

Experiment 1 (P&B's Experiment 1): native Spanish, German and English (monolingual control) speakers recruited on Prolific (333 sampled, 225 analysed; target 75 per group) rated the similarity of person-object picture pairs (112 unique pairs, 56 same-gender, 56 different-gender; pair type coded from the gender of the participant's own label for each picture). Experiment 2 (P&B's Experiment 4): 150 English monolinguals (189 sampled) learned an invented two-category "gender-like" distinction (people and objects assigned to each category) and then rated person-object pair similarity (96 pairs, 48 per type). Analyses were frequentist and Bayesian mixed-effects models that include variance sources (participants and items) that P&B's by-participant and by-item t-tests did not model.

**Findings.** Experiment 1 did not replicate: Spanish and German speakers showed no effect of grammatical gender in the registered mixed-effects analysis, and the effect did not differ from the English control group. Simpler analyses that disregard item variance, closer to P&B's, gave some support for an effect (e.g. a Bayes factor of 37.42 by participants, but 2.97, anecdotal, by person-items), whereas the Bayesian analyses that include both variance sources favoured the null. Experiment 2 replicated: category-trained English speakers rated same-gender pairs as more similar than different-gender pairs. The authors read the overall pattern as mixed.

**Limitations (stated or evident).** Online Prolific samples; picture-pair similarity ratings in one task; Spanish and German only (Spanish sample mostly from Mexico/Spain/Chile). The authors note the null for native speakers does not show that no gender effect exists in other tasks; they suggest first establishing which contexts give reliable effects and then testing mechanisms.

**Relevance to bleedthrough in LLMs.** A high-powered, preregistered human null for the Spanish/German similarity-judgment effect, alongside a positive result for explicitly trained gender-like categories in English. It is a reminder that human behavioural evidence for grammatical-gender effects on object concepts is weak, so any LLM effect cannot be assumed to mirror a robust human one. (The earlier claim that the paper implies models may be "more sensitive to surface morphosyntax than human concepts" is not in the paper.)

---

## 8. Saeed et al. (2025)

**Citation:** Saeed, Muhammed; Raza, Shaina; Vayani, Ashmal; Abdul-Mageed, Muhammad; Emami, Ali; Shehata, Shady (2025). "Beyond Content: How Grammatical Gender Shapes Visual Representation in Text-to-Image Models." *arXiv:2508.03199* (v4, Feb 2026; the arXiv page lists no venue, so the EMNLP 2025 Findings attribution is unverified).

**File:** `06_Story_Generation_and_Personification/[Saeed et al. 2025] Beyond Content - Grammatical Gender in Text-to-Image.pdf`

*(Entry corrected 2026-11-07 against the arXiv record: the earlier citation and summary did not match the paper. Summary below is from the arXiv abstract only; the PDF has not been read. The PDF file name still carries the old attribution.)*

**Summary.** A multilingual benchmark of words whose grammatical gender conflicts with the gender stereotypically linked to the concept (e.g. French *une sentinelle* "guard", feminine). Five gendered languages (French, Spanish, German, Italian, Russian) and two controls without grammatical gender (English, Chinese); 800 prompts, 28,800 images, three text-to-image models.

**Findings.** Masculine grammatical markers raised male depictions to 73% on average (22% for neutral English prompts); feminine markers raised female depictions to 38% (28% for English). Effects were stronger in high-resource languages and varied by model.

**Limitations.** Text-to-image models only; the prompts are about people-like referents (role nouns), not inanimate objects. (Abstract-level; check the paper for annotation method.)

**Relevance to bleedthrough in LLMs.** Evidence that grammatical gender in a prompt shifts a generative model's depiction of gender, asymmetrically (the masculine effect is much larger). Closest to the character study; it does not test inanimate nouns or hidden states.

---

## 9. Benedetto et al. (2026)

**Citation:** Benedetto, Luca; Donvito, Antonia; Lucchetti, Alberto; Cappelli, Andrea; Buttery, Paula (2026). "Beyond Names: How Grammatical Gender Markers Bias LLM-based Educational Recommendations." *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026), Volume 1: Long Papers*, pp. 5648-5668, 24-29 March 2026. (aclanthology.org/2026.eacl-long.264 was not checked.)

**File:** `06_Story_Generation_and_Personification/[Benedetto et al. 2026] Beyond Names - Gender Bias in Italian Recommendations.pdf` (the file name's "Italian Recommendations" shorthand does not match the paper's title)

*(Entry corrected 2026-11-07 against the PDF first page: the earlier citation, models and tasks did not match the paper.)*

**Summary.** LLM-based virtual assistants are asked for university degree-course recommendations in Italian, with gender signalled only by minimal grammatical markers (noun and adjective endings, e.g. masculine *-o* vs feminine *-a*), no proper names. The paper compares the recommendations across the two gender conditions.

**Findings (from the abstract).** Changing noun and adjective endings significantly shifts recommendations: LLMs recommend STEM disciplines less for feminine-marked prompts and narrow the set of disciplines recommended for masculine-marked prompts. The effects persist across multiple commercial LLMs (from OpenAI, Anthropic and Google). Up to 76% of the bias seen with proper-name prompts is already present with grammatical gender markers alone.

**Limitations.** Italian only; educational (degree-course) recommendations only; commercial models via API, no analysis of internal representations. (Further limitations not checked beyond the abstract and introduction.)

**Relevance to bleedthrough in LLMs.** Behavioural evidence that agreement morphology on words about a person (the user) alone changes a model's gendered recommendations. It concerns people, not inanimate nouns, and says nothing about hidden states. (The earlier summary's LLaMA-3/Mistral/GPT-4, career/product/persona tasks, and claims about "internal representations" are not in the paper.)

---

## 10. Finkley et al. (2026) / "Neutrality Bites"

**Citation:** Finkley, Imani; Li, Yuanxi; Walsh, Melanie (2026). "Neutrality Bites: Gender Representation in AI-Generated Animal Stories." FAccT 2026. *arXiv:2606.07969*. (Earlier listed as "Pucci, Flavio et al.", which is not an author.)

**File:** `06_Story_Generation_and_Personification/[Pucci et al. 2026] Neutrality Bites - Gender Representation in AI Animal Stories.pdf`

*(Entry corrected 2026-11-07 against the arXiv record: the earlier citation and summary did not match the paper. Summary below is from the arXiv abstract only; the PDF has not been read. The PDF file name still carries the old attribution.)*

**Summary.** Six LLMs finish English-language stories about seven anthropomorphic animals whose gender is never stated, across four narrative settings and several temperatures (about 23.8K stories).

**Findings.** Models often leave the character ungendered (19% on average) or use *it/its* (38.2%); when they do gender it, the skew is heavily male (40.6% male vs 2.2% female). The authors call this "neutrality bites": neutrality-first debiasing can erase marginalised identities.

**Limitations.** English prompts only; animals, not inanimate objects; no grammatical-gender manipulation. (The earlier summary's multilingual prompts and grammatical-gender shift are not in the abstract.)

**Relevance to bleedthrough in LLMs.** The same outcome as the character study (pronoun choice for a non-human character in open-ended stories) and a baseline: a strong male default plus frequent *it*, which any grammatical-gender effect must be separated from.

---

## 11. Mewes et al. (2026) / GRUFF

**Citation:** Mewes, Fabian; Lauscher, Anne; Gautam, Vagrant (2026). "GRUFF: LLM Pronoun Fidelity, Reasoning, and Biases in German." *arXiv:2605.30214*. (Earlier listed as "Gautam, Vasu et al."; Vagrant Gautam is the third author.)

**File:** `06_Story_Generation_and_Personification/[Gautam et al. 2026] GRUFF - LLM Pronoun Fidelity Reasoning and Biases in German.pdf`

*(Entry corrected 2026-11-07 against the arXiv record: the earlier citation and summary did not match the paper. Summary below is from the arXiv abstract only; the PDF has not been read. The PDF file name still carries the old attribution.)*

**Summary.** A large German dataset for pronoun fidelity (does a model keep using the pronoun introduced for a person when other people are mentioned in between?), covering four gender-agreement systems for nouns and four pronoun sets, including the neopronouns *xier* and *en*.

**Findings.** Without intervening context, LLMs agree well for masculine and feminine entities but not for the neopronouns. Intervening mentions generally throw models off; encoder-only models hold up better in German than in English, which the authors link to grammatical gender. Occupational stereotypes correlate poorly across grammatical cases and across most models.

**Limitations.** About people and pronouns, not inanimate nouns. (The earlier summary's *das Mädchen* framing, chain-of-thought conditions and "thinking worsens bias" finding are not in the abstract.)

**Relevance to bleedthrough in LLMs.** German baseline for how models track grammatical vs referential gender on people; useful context for German pronoun outcomes, not direct evidence about objects.

---

## 12. Sukumaran et al. (2024)

**Citation:** Sukumaran, Priyanka; Houghton, Conor; Kazanina, Nina (2024). "Investigating grammatical abstraction in language models using few-shot learning of novel noun gender." *Findings of the Association for Computational Linguistics: EACL 2024* (per the arXiv listing). *arXiv:2403.10338* (v1, 15 Mar 2024).

**File:** `06_Story_Generation_and_Personification/[Sukumaran et al. 2024] Grammatical Abstraction Using Few-Shot Learning of Novel Noun Gender.pdf` (the file name's title is shortened/reworded)

*(Entry corrected 2026-11-07 against the PDF: the earlier authors, languages, models and findings did not match the paper.)*

**Summary.** A word-learning experiment in French modelled on child-language studies. An LSTM and a decoder-only transformer, both small and word-tokenised, are trained from scratch on 80 million tokens of French Wikipedia. A novel noun's embedding is introduced and updated from one to a few examples (updates restricted to the embedding layer), in one grammatical agreement context, and the model is then tested on agreement in another, unseen context.

**Findings.** Both models generalise the learned gender from one or two examples to unseen agreement contexts, with a bias toward masculine (feminine learning was less efficient). Because only the embedding layer was updated, the authors conclude the embedding space holds enough gender information for this; they say the generalisation suggests an abstract gender category, with details left for further work. A one-shot experiment with 25 native French speakers also showed a masculine bias and imperfect one-shot learning.

**Limitations (stated).** The novel-noun embeddings are built to carry an initial gender categorisation (though learning still worked for nouns starting at 49-51% for either gender); test sentences may contain unintended gender cues; small noun set (20 novel-noun combinations); two small models and four agreement contexts only. French only (no German).

**Relevance to bleedthrough in LLMs.** A precedent for testing grammatical gender with nonce nouns and agreement tests, in small models and French. It is not about pre-trained LLMs, German, or morphological endings such as *-o*, *-a*, *-ung*. (The earlier claim that it "validates the wug/nonce-word methodology used in our study" is unsupported.)

---

## 13. Pan et al. (2026)

**Citation:** Pan, Deng; Germino, Joe; Ma, Yihong; Daly, Elizabeth; Moniz, Nuno; Hua, Ting; Chawla, Nitesh (2026). "Does Thinking Help Fairness? Reasoning Tokens Resolve Some Biases but Create More." Findings of ACL: EMNLP 2026. *arXiv:2609.30768*. (Earlier listed as "Zheng et al."; no Zheng is an author.)

**File:** `07_Reasoning_Mode_and_Chain_of_Thought/[Zheng et al. 2026] Does Thinking Help Fairness.pdf`

*(Entry corrected 2026-11-07 against the arXiv record: the earlier citation and summary did not match the paper. Summary below is from the arXiv abstract only; the PDF has not been read. The PDF file name still carries the old attribution.)*

**Summary.** Each model is run with thinking on and off to test counterfactual fairness: three 32B-scale models (QwQ-32B, DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B) on three high-stakes tabular decision datasets (Adult, COMPAS, Credit).

**Findings.** Thinking fixes some counterfactual flips but introduces more new ones, often at near-full confidence: new flips outnumber fixed ones about 5 to 1 across all nine model–dataset pairs. Two analysis tools (Counterfactual Depth Probability Gap; Bias Transition Matrix) trace the bias growing with thinking depth.

**Limitations.** Tabular decision tasks, not open-ended generation or language-specific gender. (The earlier summary's BBQ/StereoSet and open-ended generation are not in the abstract.)

**Relevance to bleedthrough in LLMs.** Supports not assuming that thinking mode reduces bias, for the character study's thinking-on/off comparison; the tasks are far from ours, so it is motivation, not a prediction.

---

## 14. Lei et al. (2026) / "Thinking Inertia"

**Citation:** Lei, Dianqiao; Lin, Kevin Qinghong; Lu, Pan; Torr, Philip; Zou, James (2026). "Thinking Inertia: LLMs Keep Thinking When Told Not To." NeurIPS 2026 (accepted). *arXiv:2610.11765*. (Earlier listed as "Gao et al." with the subtitle "Overthinking in Test-Time Compute"; neither matches the record.)

**File:** `07_Reasoning_Mode_and_Chain_of_Thought/[Gao et al. 2026] Thinking Inertia in Reasoning Models.pdf`

*(Entry corrected 2026-11-07 against the arXiv record: the earlier citation and summary did not match the paper. Summary below is from the arXiv abstract only; the PDF has not been read. The PDF file name still carries the old attribution.)*

**Summary.** Common stand-ins for "no thinking" (a switched-off thinking mode, short outputs) are unreliable. Responses are split into a pre-answer trace and a final answer and scored at three levels (empty-thinking rate, question–pre-answer relevance, an LLM-judge explicit-inference rate), for six prompting interventions on six LLMs over Boolean, multiple-choice and open-ended questions.

**Findings.** Explicit no-think instructions don't reliably stop visible reasoning ("thinking inertia"), more so as the answer space widens. Accuracy holds on Boolean and multiple-choice questions; on open-ended ones, answering without reasoning costs accuracy.

**Limitations.** QA tasks, not creative generation or gender. (The earlier summary's creative-prose self-critique loops and lexical-diversity collapse are not in the abstract.)

**Relevance to bleedthrough in LLMs.** A caution for the character study's thinking-off condition: switching thinking off may not remove reasoning, so "thinking off" should be checked in the outputs rather than assumed. (The earlier claim that this paper explains our pilot's early gender commitment is unsupported.)

---

## 15. Lanham et al. (2023)

**Citation:** Lanham, Tamera; Chen, Anna; Radhakrishnan, Ansh; Steiner, Benoit; Denison, Carson; Hernandez, Danny; Li, Dustin; Durmus, Esin; Hubinger, Evan; Kernion, Jackson; Lukošiūtė, Kamilė; Nguyen, Karina; Cheng, Newton; Joseph, Nicholas; Schiefer, Nicholas; Rausch, Oliver; Larson, Robin; McCandlish, Sam; Kundu, Sandipan; Kadavath, Saurav; Yang, Shannon; Henighan, Thomas; Maxwell, Timothy; Telleen-Lawton, Timothy; Hume, Tristan; Hatfield-Dodds, Zac; Kaplan, Jared; Brauner, Jan; Bowman, Samuel R.; Perez, Ethan (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning." *arXiv:2307.13702* (v1, 17 Jul 2023; Anthropic). [Author list as on the PDF title page; short form "Lanham et al."]

**File:** `07_Reasoning_Mode_and_Chain_of_Thought/07A_CoT_Faithfulness_and_Latent_Dynamics/[Lanham et al. 2023] Measuring Faithfulness in Chain-of-Thought Reasoning.pdf` (folder path as it now exists; the earlier entry omitted the 07A subfolder)

*(Entry corrected 2026-11-07 against the PDF abstract and task list: the earlier findings about social cues and invented justifications are not in the paper.)*

**Summary.** The paper asks whether a model's stated chain-of-thought (CoT) is a faithful explanation of how it reaches its answer. It intervenes on the CoT and sees how predictions change: truncating it (early answering), adding mistakes, paraphrasing it, and replacing it with filler tokens (ellipses). Tasks include ARC (Easy and Challenge), AQuA, HellaSwag, LogiQA, MMLU, OpenBookQA and TruthfulQA.

**Findings.** Models differ widely across tasks in how strongly they condition on the CoT: sometimes heavily (e.g. AQuA, where the CoT changes the final answer more than 60% of the time), sometimes mostly ignoring it. The CoT's performance boost does not seem to come from extra test-time compute alone (filler tokens) or from information encoded in particular phrasing (paraphrasing). As models become larger and more capable they produce less faithful reasoning on most tasks studied. The authors conclude CoT can be faithful if model size and task are chosen carefully.

**Limitations.** Multiple-choice and QA tasks; one model family; the paper does not test social bias, gender, or persona writing.

**Relevance to bleedthrough in LLMs.** A caution for reading reasoning traces: a stated chain of thought need not be the cause of the answer, so a trace that does not mention gender does not show gender played no role. (The earlier claim that the paper shows "latent bias operates sub-cognitively in early activations" is not in the paper.)

---

## 16. Feng, Dou & Kong (2025)

**Citation:** Feng, Xiachong; Dou, Longxu; Kong, Lingpeng (2025). "Reasoning Does Not Necessarily Improve Role-Playing Ability." *Findings of the Association for Computational Linguistics: ACL 2025*, pp. 10301-10314. (Earlier listed as "Tu et al."; no Tu is an author. aclanthology ID 2025.findings-acl.537 was not checked.)

**File:** `07_Reasoning_Mode_and_Chain_of_Thought/07B_CoT_Compute_Scaling_and_Bias/[Tu et al. 2025] Reasoning Does Not Necessarily Improve Role-Playing.pdf` (file name carries the old, wrong first author; the earlier entry omitted the 07B subfolder)

*(Entry corrected 2026-11-07 against the PDF: authors, scope and findings did not match.)*

**Summary.** A study of whether reasoning techniques improve role-playing in LLMs: 6 role-playing benchmarks, 24 LLMs and 3 strategies (direct zero-shot role-playing, role-playing with chain-of-thought, and role-playing with reasoning-optimised LLMs).

**Findings (from the abstract).** CoT may reduce role-playing performance; reasoning-optimised LLMs are unsuitable for role-playing; reasoning ability disrupts the role-playing scaling law; large models still lack proficiency in advanced role-playing; Chinese role-playing performance surpasses English. The authors propose role-aware CoT and reinforcement learning for role-playing as directions.

**Limitations.** Role-playing benchmarks, not story or pronoun generation. (The earlier summary's multi-turn persona consistency, breaking character inside thinking tokens and single-turn story prompts are not in the abstract.)

**Relevance to bleedthrough in LLMs.** Suggests that adding reasoning does not by default help character role-play, which is a caution for the character study's thinking-mode comparison. It does not study gender or personification. (The earlier claim that it is precedent for "Finding 4b" of the Feynman report was not checked and is removed.)

---

## 17. Lakens & Delacre (2020)

**Citation:** Lakens, Daniël; Delacre, Marie (2020). "Equivalence Testing and the Second Generation P-Value." *Meta-Psychology* 4: MP.2018.933. DOI 10.15626/mp.2018.933.

**File:** `08_Statistical_and_Experimental_Methodology/[Lakens & Delacre 2020] Equivalence Testing and the Second Generation P-Value.pdf`

**Summary.** The paper compares the second generation p-value (SGPV; Blume et al. 2018) with the established Two One-Sided Tests (TOST) equivalence procedure (Schuirmann 1987). Both compare data against a range of values deemed equivalent to no meaningful effect (the equivalence range), rather than testing against exactly zero.

**Findings.** The two approaches give almost identical results under optimal conditions. Under suboptimal conditions (e.g. a confidence interval wider than the equivalence range, or asymmetric intervals) the SGPV becomes difficult to interpret. The SGPV is interpretable dichotomously (equal to 0 or 1 when the confidence interval lies completely outside or inside the equivalence range), but that reading needs no calculation. The authors conclude equivalence tests yield more consistent p-values, distinguish between datasets that give the same SGPV, and allow easier control of Type I and II error rates.

**Relevance to bleedthrough in LLMs.** Supports a stated equivalence range (smallest effect size of interest) before calling a result a null, rather than treating a non-significant test as evidence of no effect. (The earlier text attributing the phrase "naive failures to reject the null" to the paper's findings and linking it to Phase 5 / Qwen3-14B steering is the project's application, not the paper's content.)

---

## 18. van Miltenburg, van der Lee & Krahmer (2021)

**Citation:** van Miltenburg, Emiel; van der Lee, Chris; Krahmer, Emiel (2021). "Preregistering NLP research." *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2021)*, pp. 613-623, June 6-11, 2021. (Earlier given as pp. 412-423 with the subtitle "An Evaluation and Recommendations"; neither matches the PDF.)

**File:** `08_Statistical_and_Experimental_Methodology/[van Miltenburg et al. 2021] Preregistering NLP Research.pdf`

*(Entry corrected 2026-11-07 against the PDF: the earlier summary described an evaluation of NLP preregistrations that the paper does not contain.)*

**Summary.** A position paper. Preregistration means specifying what you will do and expect to find before carrying out the study, a practice common in medicine and psychology but rarely discussed in NLP. The paper discusses preregistration, explores how NLP researchers could preregister their work, presents several preregistration questions for different kinds of studies, and argues in favour of registered reports. The authors say the goal is to elicit discussion in the NLP community, aiming to synthesise it into a general NLP preregistration form later.

**Relevance to bleedthrough in LLMs.** A citable rationale and question list for the project's preregistration practice. (The earlier statement that it supplies the framework for `prereg_v1.md`, amendment logs and frozen stimuli CSVs is the project's own mapping, not the paper's content.)

---

## 19. Schad et al. (2020)

**Citation:** Schad, Daniel J.; Vasishth, Shravan; Hohenstein, Sven; Kliegl, Reinhold (2020). "How to Capitalize on a Priori Contrasts in Linear Mixed Models: A Tutorial." *Journal of Memory and Language* 110: 104038. DOI 10.1016/j.jml.2019.104038.

**File:** `08_Statistical_and_Experimental_Methodology/[Schad et al. 2020] How to Capitalize on a Priori Contrasts in LMMs.pdf`

*(Entry corrected 2026-11-07: the local PDF with this file name is a different paper (Jain et al., "GestARLite", arXiv:1904.09843), so it cannot be used. The summary below is from the arXiv abstract of the Schad et al. tutorial (arXiv:1807.10451); the journal citation above was not re-checked.)*

**Summary.** A tutorial on contrast coding for linear (mixed) models. The authors argue that contrasts planned before seeing the data are the right way to build specific predictions into the model, instead of an omnibus F-test followed by follow-ups. It covers the mathematics and properties of treatment, sum, repeated, polynomial, custom, nested and interaction contrasts, how to apply them in R, and the generalized inverse needed to get coefficients for hypotheses the default contrast sets do not cover. Code is on OSF (osf.io/7ukf6, per arXiv).

**Relevance to bleedthrough in LLMs.** A reference for specifying contrasts in the project's mixed-effects models. (The earlier text about NLP researchers misreading baseline coefficients and about the character study's Q1 design across "eight stimulus languages" is the project's own application, not the paper's content; the earlier claims that sum coding is "required for unbiased inference" go beyond the abstract and are removed.)

---

## 20. Cassidy, Kelly & Sharoni (1999)

**Citation:** Cassidy, Kimberly Wright; Kelly, Michael H.; Sharoni, Lee'at J. (1999). "Inferring gender from name phonology." *Journal of Experimental Psychology: General* 128(3): 362–381. DOI 10.1037/0096-3445.128.3.362.

**File:** `01_Human_Psycholinguistics_and_Replications/[Cassidy et al. 1999] Inferring Gender from Name Phonology.pdf`

*(Entry corrected 2026-11-07 against the PDF: the author order was wrong, and the "7 experiments", syllable/voicing/stress experiment list and several findings were not accurate.)*

**Summary.** English male and female first names differ in phonology. The article asks three questions: how informative phonology is about gender, whether English speakers have learned this, and whether the knowledge affects name usage. It reports eight studies: pseudoname matching to male and female dolls with adults and children (Study 1); pronoun use in sentence completions (Study 2); a connectionist model of phonological cues (Study 3); a name-gender classification task on familiar names (Study 4); name evolution over the century (Studies 5 and 6); and brand-name preferences and cues in brand names (Studies 7 and 8).

**Findings (from the abstract).** A connectionist model finds that English phonology predicts name gender quite well. English speakers have learned these cues: for example, names were classified as male or female more quickly and accurately when phonologically typical for their gender. The evolution of names this century was affected by how male or female they sounded, and knowledge of the cues influences the perception and structure of brand names. The introduction lists cues such as more syllables, later stress and vowel endings for female names, and fewer syllables and stop-consonant endings (/k/, /t/) for male names.

**Relevance to bleedthrough in LLMs.** A human baseline showing that sound form alone can carry gender expectations for English names. It is about names, not nouns for objects. (The earlier claim that it "directly motivates the wug/nonce-modifier conditions in our study design" rests on project design the paper cannot know; it is a background reference for form-based gender cues only.)

---

## 21. Bender, Beller & Klauer (2011)

**Citation:** Bender, Andrea; Beller, Sieghard; Klauer, Karl Christoph (2011). "Grammatical gender in German: A case for linguistic relativity?" *Quarterly Journal of Experimental Psychology* 64(9): 1821-1835. DOI 10.1080/17470218.2011.582128.

**File:** `01_Human_Psycholinguistics_and_Replications/[Bender et al. 2011] Grammatical Gender in German - An Implicit Association Study.pdf` (file name's subtitle is not the paper's title; the paper uses a priming design, not an IAT)

*(Entry corrected 2026-11-07 against the PDF: the title was wrong and the paper is a priming study, not an Implicit Association Test.)*

**Summary.** The paper proposes a priming design that avoids earlier problems by preventing participants from using gender information strategically. Four experiments with German native speakers test priming effects of a prime noun's grammatical gender and biological gender on animate and nonanimate targets.

**Findings.** There were priming effects of the prime's grammatical gender on animate and nonanimate targets, an effect of the prime's biological gender on animate targets, but no effect of the prime's biological gender on nonanimate targets. The authors conclude this speaks against an effect of language on thought for German gender.

**Relevance to bleedthrough in LLMs.** A human implicit-measure study in which grammatical gender affects speeded responses but conceptual (biological) gender of inanimate primes does not. (The earlier claim that it contrasts with what LLM character creation would show is the project's inference, not the paper's.)

---

## 22. Bender, Beller & Klauer (2016)

**Citation:** Bender, Andrea; Beller, Sieghard; Klauer, Karl Christoph (2016). "Lady Liberty and Godfather Death as candidates for linguistic relativity? Scrutinizing the gender congruency effect on personified allegories with explicit and implicit measures." *Quarterly Journal of Experimental Psychology* 69(1): 48-64. DOI 10.1080/17470218.2015.1021701. (Published online 7 April 2015; the volume/pages 69(1): 48-64 are not on the PDF title page and were left as in the earlier entry, *[unverified]*.)

**File:** `01_Human_Psycholinguistics_and_Replications/[Bender et al. 2016] Lady Liberty and Godfather Death.pdf` (file name uses only the first part of the title)

*(Entry corrected 2026-11-07 against the PDF: the title was incomplete, and the earlier description (German, Spanish and genderless speakers, picture-similarity groupings, variance partitioning) did not match the paper, which tested native speakers of German in three experiments.)*

**Summary.** Personified allegories (Lady Liberty, Godfather Death) are human personifications of otherwise intangible entities. Three experiments with native German speakers test whether a gender congruency effect on allegorically used nouns reflects grammatical gender or the allegorical association, using an explicit measure (assignment of biological sex) and an implicit measure (Extrinsic Affective Simon Task).

**Findings.** There is a gender congruency effect on allegorical nouns, stronger with the explicit than the implicit measure, and it disappeared in the implicit measure when grammatical gender and allegorical associations were set into contrast. The authors conclude the effect is driven by the association of nouns with personifications rather than by their grammatical gender.

**Relevance to bleedthrough in LLMs.** Human evidence that a gender effect on personified nouns can come from the cultural personification tradition rather than the grammatical gender itself, so a model's output on personified nouns could also reflect that tradition. (This is an inference for the project; the paper studies humans only.)

---

## 23. Pawley (2002)

**Citation:** Pawley, Andrew (2002). "Using he and she for inanimate objects in English: Questions of grammar and world view." In N. J. Enfield (Ed.), *Ethnosyntax: Explorations in Language and Culture*, pp. 110-137. Oxford University Press. (Title as given in a later Pawley paper's reference list, which also confirms the pages; the publisher was not checked.)

**File:** `01_Human_Psycholinguistics_and_Replications/[Pawley 2002] Using He and She for Inanimate Nouns in English.pdf` (**this PDF is a different Pawley paper**: "He's a big tree, she's a cold wind: Pronominal gender assignment to inanimates in Australian Vernacular English", written for a volume honouring Viktor Krupa; its journal details are not on the PDF)

*(Entry corrected 2026-11-07: the earlier summary described Newfoundland and Southwest England dialects as a focus, which the PDF does not support. The summary below is of the PDF actually on file, not of the 2002 chapter, which could not be read.)*

**Summary.** The PDF describes pronominal gender assignment to inanimates in "animated style" in Australian Vernacular English (AVE). A few classes are consistently masculine (trees and other plants, vehicles under a driver's control, lower animals, higher animals of unknown sex). Most inanimate things, concrete or abstract, are feminine (roads, rivers, wind, buildings, machines, situations, almost all body parts). Some portable objects are most often feminine but may be masculine or feminine, possibly reflecting emotional attachment (feminine) or detachment (masculine). Fragmentary historical data suggest a similar system in 17th and 18th century middle-class English speech, perhaps back to Middle English; the same system occurs in New Zealand Vernacular English, with parallels in parts of North America and Britain.

**Relevance to bleedthrough in LLMs.** Background: in some English varieties *he* and *she* are used for inanimate things by regular rules, so an English-language model could have seen such usage in training text. It does not bear on foreign grammatical gender. (The earlier comparisons to a "foreign grammatical gender" are the project's, not the paper's.)

---

## 24. Epley, Waytz & Cacioppo (2007)

**Citation:** Epley, Nicholas; Waytz, Adam; Cacioppo, John T. (2007). "On seeing human: A three-factor theory of anthropomorphism." *Psychological Review* 114(4): 864–886. DOI 10.1037/0033-295X.114.4.864.

**File:** `06_Story_Generation_and_Personification/[Epley et al. 2007] On Seeing Human - Three-Factor Theory of Anthropomorphism.pdf`

**Summary.** A theory paper (it reports no new experiments). It proposes three psychological determinants of anthropomorphism: (1) elicited agent knowledge (the accessibility and applicability of knowledge about humans), (2) effectance motivation (the motivation to be an effective social agent, i.e. to understand and predict), and (3) sociality motivation (the desire for social connection). It also discusses implications for robotics and human-computer interaction and for dehumanization.

**Findings (theory predictions, per the abstract).** Anthropomorphism should be stronger when anthropocentric knowledge is accessible and applicable to the target, when people are motivated to be effective social agents, and when they lack social connection.

**Relevance to bleedthrough in LLMs.** General background on why people see non-human things as agents. It says nothing about language models or grammatical gender. (The earlier claim that it is the "cognitive psychological foundation" for the project's personification prompt, and that priming a model activates its "agentic schemas", is not in the paper.)

---

## 25. Nass, Moon & Green (1997)

**Citation:** Nass, Clifford; Moon, Youngme; Green, Nancy (1997). "Are machines gender neutral? Gender-stereotypic responses to computers with voices." *Journal of Applied Social Psychology* 27(10): 864–876. DOI 10.1111/j.1559-1816.1997.tb00275.x.

**File:** `06_Story_Generation_and_Personification/[Nass et al. 1997] Are Machines Gender Neutral.pdf`

**Summary.** An experiment with 40 undergraduates (mixed design: subject gender x tutor voice x evaluator voice between subjects; topic, computers vs love and relationships, within subjects). Computers with synthesized male or female voices ran a tutoring session (10 facts per topic), then a separate computer evaluated the tutor's performance, and participants rated tutor and evaluator. The aim was to test whether minimal gender cues (voice) produce gender-stereotypic responses.

**Findings.** Three stereotypes were tested. (1) Evaluation from males is more valid: with a male-voiced evaluator the tutoring session was rated more friendly, F(1,32) = 5.74, p < .05, and more competent, F(1,32) = 5.10, p < .05. (2) Dominance in females is unbecoming: the female-voiced evaluator was rated less friendly, F(1,32) = 6.03, p < .05. (3) Women know more about "feminine" topics: supported by an interaction on tutor informativeness (F = 7.85, p < .01) but only for the male-voiced-evaluator condition; not supported with a female-voiced evaluator. The authors say Stereotype 3 appears less powerful than 1 and 2. No differences by subject gender.

**Relevance to bleedthrough in LLMs.** Human evidence that a minimal gender cue (a synthesized voice) shapes social judgments of machines. It is not about language models or grammatical gender; the earlier statement that LLM outputs "mirror" this is not in the paper. (The earlier summary's single-sentence version of the topic-knowledge result was incomplete, and the paper's own framing in terms of "Computers Are Social Actors" was not checked.)

---

## 26. Wentworth (2026)

**Citation:** Wentworth, Isabelle (2026). "Gendered asymmetries in AI stories in English and Spanish." *AI & Society* 41: 8557–8570. DOI 10.1007/s00146-026-03163-x.

**File:** `06_Story_Generation_and_Personification/[Wentworth 2026] Gendered Asymmetries in AI Stories in English and Spanish.pdf`

**Summary.** A close reading, in the New Formalist tradition, of short stories generated by ChatGPT (the paper says ChatGPT-5) in English and Spanish. In English, the author asked ChatGPT four times, in separate chats, to write one short character-driven story with a male protagonist and one with a female protagonist (prompt: "Write two character-driven short stories: one story with a female protagonist and one story with a male protagonist"); the first pair is analysed in detail. A Spanish prompt ("Escriba dos relatos breves centrados en el...") is analysed the same way. The prompts do specify protagonist gender.

**Findings.** Across the four English pairs the women became an office worker, gardener, seamstress and housewife and the men an engineer, novelist, boxer and soldier. In the analysed stories the female-protagonist story is set in domestic/interior space and differs in focalisation, adjectives and interiority from the male-protagonist story; the Spanish stories show similar patterning, and the author notes Spanish is more morphologically pronounced for gender.

**Limitations (stated).** "Necessarily limited in scale", diagnostic rather than exhaustive; very small sample; qualitative reading, not a statistical or computational analysis; one model.

**Relevance to bleedthrough in LLMs.** A qualitative example of gendered patterning in generated stories in English and Spanish. It is not a controlled test of grammatical gender or of non-gendered prompts. (The earlier claims of "non-gendered prompts", "computational analysis", "male characters consistently dominate agency" and that it "directly substantiates our cross-lingual design" are not supported by the paper.)

---

## 27. Rejdebrant (2025)

**Citation:** Rejdebrant, Gunilla (2025). "Biases in AI-Generated English Texts – How ChatGPT Assigns Gender in Non-Gendered Story Prompts." Research essay, 15 hp (English), Department of Social and Behavioural Studies, October 2025. *[unverified: the earlier citation gave the first name as Gustav, called it a bachelor's thesis at Högskolan Väst, and gave DiVA URN urn:nbn:se:hv:diva-25343; the PDF shows "Gunilla Rejdebrant" and "Research Essay 15 hp" and I did not find the institution or the URN in it.]*

**File:** `06_Story_Generation_and_Personification/[Rejdebrant 2025] How ChatGPT Assigns Gender in Non-Gendered Story Prompts.pdf`

*(Entry corrected 2026-11-07 against the PDF: the earlier model list, sample and numbers were not in the paper.)*

**Summary.** A small study of how ChatGPT (GPT-3.5, one model) assigns gender when the prompt does not specify it. Four prompts, 40 short stories in total, about parents dropping a child off at an activity (preschool, cheerleading, soccer, football). The author looks at names, pronouns and kinship terms for parents and children.

**Findings.** Parents were predominantly assigned female names and pronouns: all mothers for the preschool and cheerleading prompts; male parents appeared first with the soccer prompt and more with football. The children's gender also depended on the activity. The author links this to caregiving being associated with women.

**Limitations (stated).** Small datasets; only one model of one LLM (GPT-3.5); small prompt-wording changes changed the output.

**Relevance to bleedthrough in LLMs.** A small illustration of an activity-dependent default in story generation, with a student-essay level of evidence. It does not quantify a general male default. (The earlier "GPT-4", "hundreds of completions", "unnamed and non-human protagonists", "he in over 65%" and "singular they" claims are not in the paper.)

---

## 28. Mihaylov & Shtedritski (2024)

**Citation:** Mihaylov, Viktor; Shtedritski, Aleksandar (2024). "What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages." *Proceedings of the 1st Workshop on NLP for Science (NLP4Science)*, pages 16–23, November 16, 2024. Association for Computational Linguistics.

**File:** `02_Gender_in_Embeddings_and_LLMs/[Mihaylov & Shtedritski 2024] What an Elegant Bridge - Multilingual LLMs are Biased Similarly in Different Languages.pdf`

**Summary.** Revisits Boroditsky et al.'s (2003) bridge example in open-weight LLMs. The LLMs (Mistral-7B as the main model, Llama2-7B repeated as a check) are prompted to describe nouns with adjectives in 10 languages with grammatical gender: Bulgarian, Czech, French, German, Greek, Hindi, Italian, Latvian, Portuguese and Spanish. The generated adjectives are translated into English (with Google Translate) so that gender-marked adjective forms cannot give the answer away. The authors compute a gendered-adjective similarity score between language pairs and train a classifier per language to predict a noun's grammatical gender from its adjectives.

**Findings.** The adjectives predict grammatical gender above chance in every language (Table 1). Gendered-adjective patterns are similar across many language pairs (e.g. among Romance and Slavic languages) and lower for Greek, Hindi and Latvian. A classifier trained on adjectives from 9 languages transfers to the held-out tenth. The authors conclude that the LLMs "are biased similarly in different languages" (the arXiv v1 abstract) / find a strong social influence of language (the ACL version abstract).

**Limitations.** Generated adjectives only (behavioural output, not hidden states); adjectives pass through machine translation; two 7B models; translated nouns/word lists not checked here.

**Relevance to bleedthrough in LLMs.** Closely related behavioural prior work: it shows grammatical gender of nouns is recoverable from the adjectives an LLM uses, with transfer across languages. It does not test hidden representations or steering. (The earlier entry listed Polish and Russian among the languages, which is wrong, and described "shared latent representations", which the paper's behavioural design does not show.)

**Note.** The folder holds two PDFs of this paper: the ACL version (`...Biased Similarly in Different Languages.pdf`) and the arXiv v1 (`...Multilingual LLMs Bias.pdf`).

## Cross-cutting themes

- **Agreement vs. semantics.** In humans, the evidence that grammatical gender of inanimate nouns shapes object concepts is weak or mixed: Mickan et al. (2014) found no effect in either of two replications, and Elpers et al. (2022) found no effect for Spanish and German speakers in the registered mixed-effects analysis, though category-trained English speakers did show one. In embeddings, Gonen et al. (2019) show that nouns of the same grammatical gender sit closer together, and that removing this signal at training time improved embedding quality. (The earlier text cited Brand et al. here and claimed "near-zero conceptual warping"; Brand et al. is not summarised in this file and the claim overstates the replications.)
- **Open-ended generation vs. forced-choice probing.** Behavioural studies of generation (Saeed et al. on text-to-image models; Benedetto et al. on Italian recommendations; Finkley et al. on animal stories) show gender defaults or shifts in outputs. The earlier sentence about log-probability probing (Robinson et al., Hu & Levy) is not based on a summary in this file; see folder 04.
- **Personification and character attribution.** Generation studies find that gender markers or defaults shape what models write about people or animals (Benedetto et al., Saeed et al., Finkley et al.; Wentworth and Rejdebrant as small qualitative or student studies). None of them tests inanimate nouns turned into characters. (The earlier claim about "foreign lemma citation forms" carrying latent gender signals is not from these papers.)
- **Reasoning mode and thinking dynamics.** Enabling thinking does not automatically reduce bias: Pan et al. find thinking fixes some counterfactual flips but creates more new ones (tabular decisions); Lei et al. find that "no-think" instructions do not reliably stop visible reasoning; Lanham et al. find that a stated chain of thought is not always what drives the answer; Feng, Dou & Kong find CoT may reduce role-playing performance. These are different tasks from ours, so they are cautions, not predictions.
- **Methodological discipline and equivalence testing.** Lakens & Delacre compare equivalence tests (TOST) with the second generation p-value; van Miltenburg et al. argue for preregistration and registered reports in NLP; Schad et al. is a tutorial on contrast coding in linear mixed models (read from its arXiv abstract, as the local PDF is the wrong paper).

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
