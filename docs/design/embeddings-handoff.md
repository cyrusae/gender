# Handoff: static-embedding debiasing study (standalone, spun off from `cyrusae/gender`)

Context from a planning chat with Cyrus, 2026-11-08. Purpose: get the three key papers' repos running, reproduce a headline number from each, then build the extensions below. **Figures quoted from the papers were read off the full texts during the chat; verify them against the PDFs/repos before relying on them. Items marked (verify) came from memory.**

## Scope

A small standalone paper using **static embeddings only** (fastText / MUSE), separate from the main repo's phases. It should live in its own directory or repo and **reuse the `gbleed` lexicon** (Wiktionary gold labels, inanimate filter, ending-matched / regular / exception sets, stem-split held-out sets). Follow the main repo's conventions: preregister measurement choices in a `decisions.md`, freeze held-out sets, version stimulus lists, never edit results into them.

Working one-sentence claim (Cyrus to refine):
> Prior work treats grammatical gender in embeddings either as a confound to erase (Omrani Sabbaghi & Caliskan 2022) or a signal to protect (Zhou et al. 2019), but neither measures what it carries. We show that [___].

## Papers and repos

| Paper | arXiv | Code | Role |
|---|---|---|---|
| Bolukbasi et al. 2016, *Man is to Computer Programmer…* | 1607.06520 | `tolga-b/debiaswe` (verify) | Original hard debiasing (neutralize/equalize) |
| Zhou et al. 2019, *Examining Gender Bias in Languages with Grammatical Gender* (EMNLP) | 1909.02224 | `shaoxia57/Bias_in_Gendered_Languages` | Semantic vs. grammatical gender directions; Shift / Hybrid mitigation |
| Omrani Sabbaghi & Caliskan 2022, *Measuring Gender Bias in Word Embeddings of Gendered Languages Requires Disentangling Grammatical Gender Signals* (AIES) | 2206.01691 | `shivaomrani/GG_Disentangling` | Iterative erasure of grammatical gender; GG-WEAT |

Also relevant: Gonen, Kementchedjhieva & Goldberg 2019 (CoNLL, 1910.14161); Gonen & Goldberg 2019 "Lipstick on a Pig"; Williams, Cotterell et al. 2021 (TACL, 2005.01204); McCurdy & Serbetci 2020 (2005.08864).

## Key facts from the papers

**Bolukbasi 2016**
- Gender direction = top principal component of 10 definitional pair differences (she−he, mother−father, …).
- Hard debias: *neutralize* removes the gender component from every word in the neutral set N, then renormalizes to unit length. *Equalize* makes equality-set pairs differ only along the gender direction.
- Scope is broad: N = whole vocabulary minus the gender-specific set S (218 hand-picked, extended to 6,449 by an SVM). They note that S is subjective and should be adapted to the application.
- §6 concedes that Equalize can remove legitimate distinctions ("to grandfather a regulation").

**Zhou 2019**
- Semantic direction: PCA over Spanish definitional pairs. Grammatical direction: LDA on about 3,000 masculine and 3,000 feminine nouns. Their noun list is *not* purely inanimate (it includes "woman").
- Cosine between the two directions = **0.389**. They then orthogonalize: semantic direction minus its projection onto the grammatical direction.
- Mitigation removes only the orthogonalized semantic component from inanimates, so **the overlapping component stays in every inanimate noun**.
- §2.1 *defines* an inanimate-noun bias score (WEAT association with gender words). §4 measures bias **only on animate nouns**, so the inanimate score is never reported.
- Utility checks are word similarity and translation only; nothing grammatical.
- They use fastText, which builds vectors from character n-grams, so noun endings are part of every vector. There is no spelling control.
- Spanish attribute words are translations of English gender words and carry grammatical gender themselves.

**Omrani Sabbaghi & Caliskan 2022**
- Iteratively train a linear SVC on about 6,000 inanimate nouns (from Universal Dependencies treebanks; Spanish from AnCora), project out its direction, and repeat (15 rounds) until the classifier is at chance. Essentially INLP (Ravfogel et al. 2020), which they appear not to cite.
- **Spanish resisted erasure:** GG-WEAT 1.83 → 1.23, still significant. Polish behaved the same. The Discussion speculates that fastText's subword features are the cause.
- The single-category GG-WEAT on 2,000 nouns per language is essentially Zhou's inanimate score, but with *Spanish* attribute words (madre/padre…), which carry grammatical gender.
- All "utility preserved" checks are semantic (analogy family + capitals subsets, ValNorm, baseline WEATs). Spanish analogy 82 → 82; ValNorm .81 → .82. The improvement claim is mostly Polish.
- English is used as the reference point (pairwise-distance baseline; congruence with IAT sign).

## Planned studies

### C1: Zhou's inanimate metric, run after Zhou's own mitigation
- Spanish inanimate nouns from the `gbleed` lexicon, scored before and after Shift Ori / Hybrid.
- **Attribute words in English**, in the aligned MUSE ES–EN space (social gender without grammatical gender). Also run Zhou's Spanish attributes for comparison.
- Report **ending-matched vs. regular** nouns separately as the spelling control.
- Optional: decompose the semantic direction into its overlap piece (the part along the grammatical direction) and the orthogonal remainder, and test each.
- Prediction to preregister before running: does Hybrid barely move the inanimate scores?

### C2: What erasure costs
- Train probes for semantic contrasts, run Omrani Sabbaghi & Caliskan's iterative erasure, and test whether the probes collapse while their benchmarks stay flat.
  1. **Abstract vs. concrete, balanced across gender** (confirmatory). Feminine *-ción, -dad, -tud, -eza, -ura* vs. masculine *-miento, -aje, -ismo*, half of each gender. Train on some suffixes and test on held-out suffixes so the probe can't just read the suffix.
  2. **Tree vs. fruit** (demonstration). manzano/manzana, naranjo/naranja, cerezo/cereza, ciruelo/ciruela, almendro/almendra; reversed exception higuera/higo. About 15–20 pairs, carried entirely by the ending.
  3. **Size alternations** (exploratory). barco/barca, cesto/cesta, huerto/huerta, jarro/jarra, bolso/bolsa. The size direction is inconsistent.
- Static embeddings have one vector per spelling, so **homographs (el/la capital, orden, frente) can't be used here.** Save them for the transformer study.
- Controls: random-direction erasure baseline (many draws); stem-split held-out sets.

## First tasks for the local session
1. Clone the three repos into a `vendor/` (or similar) directory. Record each one's dependencies, embedding downloads (fastText Spanish Wikipedia/CC, MUSE ES–EN aligned) and their sizes.
2. Reproduce one headline number from each: Zhou Table 1 (Spanish MWEAT-Diff 3.69 → Hybrid Ori 0.31); Omrani Sabbaghi & Caliskan Table 5 (Spanish GG-WEAT 1.83 → 1.23). Bolukbasi is optional, since it's English w2v.
3. Map their word lists onto the `gbleed` lexicon: overlap, and what our inanimate / ending filters would drop.
4. Scaffold C1, then C2, with prereg stubs in `decisions.md`.

## Parked: transformer study (later, larger)
Apply Bolukbasi-style debiasing to a multilingual LLM (Qwen3) and measure Spanish agreement. Notes for when it starts:
- Intervene through **forward hooks on the embedding output, keyed by token ID** (the faithful match to Bolukbasi's table rewrite). Hooks avoid untying problems; Qwen3 0.6B/1.7B/4B probably tie embeddings, so check `tie_word_embeddings`. For output-side edits, clone and edit the output matrix. Add a unit test that hook and weight edit give matching logits.
- Measure agreement by surprisal margins on minimal pairs using gender-invariant possessive frames (*mi mesa es muy bonit_*), not accuracy alone. Arms: regular nouns, masculine *-a* exceptions (*el problema*), ending-matched feminine nouns (the critical arm), *agua*-type nouns (watch for *el agua frío*), homographs (meaning cost), nonce wug test (does a form→gender rule exist?).
- Social side: common-gender occupations (*mi electricista es muy simpátic_*) to show the intended effect; epicene appositives (*La víctima, un/una…*) to separate referential from grammatical gender.
- Arms: the raw Bolukbasi direction vs. Zhou's orthogonalized direction; removal and steering with the person-derived direction.
- Error direction (masculine default vs. form cue) is preregistered. Cyrus's current guess is that everything trends masculine, based on Phase 1/2 results.

## Reading list for the bibliography agent
Irvine & Gal 2000 (language ideology; "erasure"); Wierzbicka 2014 *Imprisoned in English*; Bender 2019 (#BenderRule); Joshi et al. 2020 (linguistic diversity in NLP); Samuel, Cole & Eacott 2019 (systematic review, grammatical gender and relativity; verify venue); Corbett 1991 *Gender*; Zubin & Köpcke 1986; Ravfogel et al. 2020 (INLP); Belrose et al. 2023 (LEACE); MultiBLiMP (verify); plus papers citing Zhou 2019, to check whether anyone has reported the inanimate metric.
