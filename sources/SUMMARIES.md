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

## 5. Flint & Ivanova (2024)

**Citation:** Flint, George; Ivanova, Anna A. (2024). "Testing a Distributional Account of Grammatical Gender Effects on Semantic Gender Perception." In L. K. Samuelson, S. L. Frank, M. Toneva, A. Mackey & E. Hazeltine (Eds.), *Proceedings of the 46th Annual Conference of the Cognitive Science Society*, 46, pp. 2847-2853. CC BY 4.0. eScholarship permalink https://escholarship.org/uc/item/44j035tc. No DOI is printed. The eScholarship cover page gives the title as "Testing a Distributional *Semantics* Account..."; the paper's own title page omits "Semantics".

**File:** `qt44j035tc.pdf` -> `Flint_2024_Testing-A-Distributional-Account-Of-Grammatical-Gender-Effects-On-Semantic-Gender-Perception.pdf`

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

**File:** `s13428-022-02032-x.pdf` -> `Vankrunkelsven_2024_Semantic-Gender-Norms-For-24000-Dutch-Words.pdf`

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

## Cross-cutting themes

- **Agreement vs. semantics.** Kann and Gonen et al. converge on the central distinction: gender in noun representations is largely a distributional echo of agreement morphology. Kann finds no recoverable gender in lemmatized context. Gonen et al. show that gender signal remains until context is carefully neutralized. Mickan et al. find no human behavioral evidence of a strong gender-as-sex effect. For LLMs, the task is to separate surface/agreement leakage from semantic association.
- **Controlling leakage.** Both embedding papers show how fragile controls are: lemmatizer quirks, ambiguous forms, frequency mismatches. Any LLM test needs equally careful controls, such as stimuli with agreement cues removed or balanced.
- **Debiasing limits.** Projection-based debiasing (Bolukbasi) fails to remove grammatical-gender structure (Gonen et al.). LEACE offers a stronger linear alternative, but Belrose et al. only test it on social gender in English and on POS.
- **Inanimate noun stimuli.** Several lists exist that cross gender between languages: Boroditsky/Schmidt (via Kann, Table 2), Mickan et al.'s two lists, SimLex-999 inanimate pairs, and NorthEuraLex concepts.
- **Languages covered.** Human studies: German, Spanish (English for the second Mickan experiment). Kann: 9 languages (bg, es, fr, he, it, pl, ro, ru, sk). Gonen: German, Italian. Belrose: English only. None of the four studies an LLM for grammatical gender directly, so there is an open gap.
- **Human gender-association norms outside English.** Flint & Ivanova (2024) collected 1-7 gender ratings for about 140-150 native Spanish and German adjectives (about 20 raters per language; data not public). Vankrunkelsven et al. (2024) give 1-5 ratings for 24,037 Dutch words (public, OSF). Both scales run feminine to masculine, like Glasgow. Vankrunkelsven et al. report r = .80 between native Dutch ratings and Glasgow ratings translated into Dutch.
- **Grammatical vs. semantic gender in humans.** Flint & Ivanova find grammatical-gender effects on adjective choice in Spanish but not German, and only partly explained by embedding similarity. Vankrunkelsven et al. find a low correlation (.24) between Dutch semantic gender and masculine/feminine class. Both support treating any LLM bleedthrough as an effect to compare against weak and language-dependent human effects.

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
