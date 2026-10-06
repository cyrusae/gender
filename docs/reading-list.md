# Reading list: references that came up (not in `sources/`)

*Papers and resources mentioned while building the project, with why they're relevant and
where they came up. Bibliographic details were checked against publisher/arXiv/ACL pages on
2026-10-06 unless marked otherwise. Nothing here has been read for the project yet; these are
pointers, not summaries.*

## Model internals that affected the measurements

| reference | why it matters here | came up |
|---|---|---|
| Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. arXiv:2309.17453 | Names and explains **attention sinks**: models put large attention on initial tokens regardless of meaning. Background for why position 0 and the first token after `<|endoftext|>` gave anomalous vectors | Phase 1 position-0 finding; Phase 2 sink bug |
| Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024). *Massive Activations in Large Language Models.* arXiv:2402.17762 (COLM 2024) | Documents **massive activations**: a few dimensions with values orders of magnitude larger than the rest, tied to sink tokens. Explains the single dimension (1.7B dim 1793; 4B dim 4) holding 72–93% of variance for single-token words. Cite when describing the bug and the newline-buffer fix | Phase 2 sink bug |
| Kaplan, G., Oren, M., Reif, Y., & Schwartz, R. (2025). *From Tokens to Words: On the Inner Lexicon of LLMs.* ICLR 2025. arXiv:2410.05864 | Shows **detokenization**: models assemble whole-word representations at a word's *last* token, mainly in early and middle layers. The main justification for last-token readout, and the reason early-layer results may reflect token identity rather than the word | readout-position question |

## Probing and erasure methods

| reference | why it matters here | came up |
|---|---|---|
| Hewitt, J., & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP-IJCNLP 2019, 2733–2743. doi:10.18653/v1/D19-1275 | **Control tasks and selectivity**: train the same probe on random but consistent labels to see what the probe can "learn" from nothing. Planned for the Phase 2 secondary probe | methods conversation; Phases 3–5 plan |
| Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL 2020, 7237–7256 | **INLP**, the iterative erasure method that preceded LEACE. Used once as a Phase 1 diagnostic (needed many more directions than LEACE). Gonen and Goldberg are also authors of the Gonen et al. 2019 paper in `sources/` | Phase 1 diagnostics |

## Norms and stimulus resources

| reference | why it matters here | came up |
|---|---|---|
| Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). *Concreteness ratings for 40 thousand generally known English word lemmas.* Behavior Research Methods, 46, 904–911 | **Used in the pipeline**: concreteness covariate for the Phase 2 adjusted direction (37,058 words + 2,896 two-word expressions). Data from a third-party mirror (ArtsEngine on GitHub), verified against the paper's counts, checksum in `src/gbleed/norms.py` | Phase 2 concreteness confound |
| Scott, G. G., Keitel, A., Becirspahic, M., Yao, B., & Sereno, S. C. (2019). *The Glasgow Norms: Ratings of 5,500 words on nine scales.* Behavior Research Methods, 51(3), 1258–1270 | Ratings for 5,553 English words on nine scales, including **gender association**. Planned graded predictor for Phase 5 (does an adjective's shift scale with how gendered people rate it?) | methods conversation; Phases 3–5 plan |

## Theory and later tools (not checked here)

| reference | why it matters here | came up |
|---|---|---|
| Trubetzkoy, N. S. (1939). *Grundzüge der Phonologie.* | Source of the **privative vs equipollent** opposition behind the pre-registered markedness hypothesis (masculine as unmarked default). Check the edition/translation before citing | methods conversation |
| Gemma Scope (open sparse autoencoders for Gemma models; browsable on Neuronpedia) | Possible later tool for finding gender features without supervision; exists for Gemma, not Qwen | methods conversation; model research |
