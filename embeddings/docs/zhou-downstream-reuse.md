# Downstream reuse of Zhou et al. 2019

Question: which papers citing Zhou et al. 2019, "Examining Gender Bias in Languages with Grammatical Gender" (EMNLP-IJCNLP 2019, ACL Anthology D19-1531, arXiv 1909.02224), reuse its methods or numbers, rather than only citing it?

## 1. Screening

- OpenAlex ID for Zhou 2019: `https://openalex.org/W2970408399` (verified: title, 2019, DOI 10.18653/v1/d19-1531).
- Query: `https://api.openalex.org/works?filter=cites:W2970408399&per-page=200`. Returned 84 works (OpenAlex's `cited_by_count` is 87; the gap is OpenAlex coverage).
- Screening: all 84 titles and reconstructed abstracts were read. Keyword pass for Zhou, grammatical, MWEAT, Shift, Hybrid, De-Align, WEAT, debias, gendered language, and the language names. Full texts were opened only for the candidates in section 2.
- Limits: the other 80 were judged on title and abstract only. A paper that reuses Zhou without saying so in its abstract could be missed.

## 2. Papers that go beyond related-work citation

| paper (authors, year, venue, link) | how it uses Zhou 2019 | quoted sentence | evidence URL |
|---|---|---|---|
| Xiao & Li, 2026, arXiv:2606.30152, "Estimating Grammatical Gender Directions in Contextual Embeddings under Controlled and Natural Contexts" (https://arxiv.org/abs/2606.30152, DOI 10.48550/arxiv.2606.30152) | REUSES METHOD: the LDA estimator for the grammatical gender direction, one of three estimators compared. Not Shift, Hybrid or De-Align. | "The second estimator is based on linear discriminant analysis (LDA), following Zhou et al. [5], which models grammatical gender as a supervised separation problem and identifies the direction that maximizes class separability." Also: "Zhou et al. [5] pioneered a disentanglement paradigm that separately estimates grammatical and semantic gender directions in embedding space, and further eliminates grammatical gender components from semantic gender vectors to isolate social gender bias." | https://arxiv.org/pdf/2606.30152 (read via pdftotext of the PDF, Section 2.2 and Section 3) |
| Omrani Sabbaghi & Caliskan, 2022, "Measuring Gender Bias in Word Embeddings of Gendered Languages Requires Disentangling Grammatical Gender Signals," AIES 2022 (https://doi.org/10.1145/3514094.3534176; arXiv:2206.01691) | REUSES METHOD, then argues against it: adapts Zhou's LDA step to find the grammatical gender direction, and reports that a single LDA pass does not remove the signal. Compares its method to Zhou in Table 1, saying the two are "not directly comparable." | "Our method of disentangling grammatical gender from embeddings is inspired by Zhou et al. [62]'s use of LDA for identifying grammatical gender." Also: "While Zhou et al. [62] apply LDA once to obtain the representation of grammatical gender, our method suggests that the hyperplane obtained using this technique (d_g) does not fully capture grammatical gender signals." Table 1 caption: "Our methodology is not directly comparable to Zhou et al. [62] since grammatical gender is not disentangled from word embeddings in their study." | https://arxiv.org/pdf/2206.01691 (read via pdftotext of the arXiv PDF; the ACM PDF returned 403) |
| Xiao & Li, 2026 (as above), comparison | USES AS BASELINE (inferred): Zhou's LDA estimator is one of the "discriminative baselines" the new centroid estimator is compared against. The text names "both baselines," so the two baselines are LDA and SVM. This identification is my inference, not stated in those words. | "The centroid-based estimator significantly outperforms both baselines in terms of overall trade-off." | https://arxiv.org/pdf/2606.30152 |
| Machine Translation and Gender biases in video game localisation: a corpus-based analysis, 2022, Journal of Data Mining & Digital Humanities (https://doi.org/10.46298/jdmdh.9065) | CITES MEASURE CONCEPT ONLY (borderline): cites Zhou for the idea of visualising semantic and grammatical directions of embeddings. No Zhou numbers, Shift, Hybrid, De-Align or MWEAT. | "...and to visualise the semantic and grammatical directions of the words embeddings [Zhou; Shi; Zhao and al, 2019]..." | https://jdmdh.episciences.org/10245/pdf (read via pdftotext; the JDMDH PDF is partly readable) |

Count of papers that cite Zhou but only in related work, from the abstract screen: not counted individually (about 80, judged on abstract only).

### UNVERIFIED

- Gender Bias in Spanish Distributional Language Models, "On the Limits of Debiasing: A Multimetric Evaluation of Post-Processing Strategies for Gender Bias in Spanish Distributional Language Models," 2025, IEEE WI-IAT (https://doi.org/10.1109/wi-iat67162.2025.00050; IEEE document 11676915). The abstract describes "an approach that projects out grammatical gender directions from the bias subspace" and "a method that builds upon existing grammatical disentanglement". That could be Zhou-derived, but I could not open the full text: DOI and IEEE return 302 and 403, and no open-access copy was found. Status: UNVERIFIED, whether it reuses Zhou's Shift, Hybrid, De-Align or MWEAT is unknown.

## 3. The specific questions

**Does anyone report post-mitigation p-values from Zhou's MWEAT test?**
No, among the papers I could read. Omrani Sabbaghi & Caliskan (arXiv:2206.01691) has no "p-value" or "p <" in the text, and no MWEAT or "Hybrid". Xiao & Li (arXiv:2606.30152) also has no p-value reporting or MWEAT. The Spanish 2025 paper is UNVERIFIED, so this is not established for it. Note that Omrani Sabbaghi & Caliskan use WEAT effect sizes on their own post-processing. They do not report Zhou's MWEAT p-values.

**Does anyone rely on Hybrid being better than Shift?**
No, among the papers I could read. "Hybrid" and "Shift" do not appear as Zhou's mitigation variants in Omrani Sabbaghi & Caliskan. Xiao & Li use "shift" only as a description of a distribution ("grammatical gender manifests as a global shift between two distributions"), not as Zhou's Shift method. The Spanish 2025 paper is UNVERIFIED.

## 4. Caveats

- Only 4 papers were opened in full, and one of them (the Spanish 2025 paper) was not opened at all.
- The PDF extracts were read with pdftotext; the quoted sentences come from those extracts, and line breaks were removed.
- Zhou's own MWEAT and Hybrid/Shift/De-Align results are not restated here; nothing in the readable downstream papers reuses them.
