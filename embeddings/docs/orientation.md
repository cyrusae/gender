# Orientation: the two upstream repos (2026-11-08)

Read from the cloned code and the notebooks' saved outputs (`embeddings/external/`), not from
the papers. Paper claims still need checking against the PDFs.

## Embeddings (downloaded to the store, linked into each repo)

`embeddings/scripts/fetch_embeddings.sh` puts everything in
`/Volumes/Lexar/gbleed/shared/embeddings/` (checksums in `SHA256SUMS`) and symlinks it where
each notebook looks.

| repo | vectors | other files |
|---|---|---|
| Zhou et al. 2019 | MUSE aligned fastText Wikipedia `wiki.multi.{en,es,fr,de}.vec` (300d, 200k words; notebook loads the first 50k) | MUSE dictionaries `en-es`, `es-en`, `es-fr` (+ `en-fr`, `fr-en`) |
| Omrani Sabbaghi & Caliskan 2022 | fastText Common Crawl `cc.{es,fr,de,it,pl,en}.300.vec` (2M words; notebook loads the first 1M) | none |

Not obtainable from the repos: Zhou's *derived* files (`debiased_aligned_*`, `hybrid_*`,
`*_offset_debiased_*`, the hard-debiased English `.bin`), `WEFAT_occupations.txt`,
`ES_gender_neutral_words.txt` and the `genbiasmt-master` occupation lists. The notebook writes
some of the derived files itself (`save_vec`); the others would have to be rebuilt.

## Zhou et al. 2019 (`Bias_in_Gendered_Languages/bias_emnlp19.ipynb`, 395 cells)

- One exploratory notebook, run top to bottom by hand; many cells commented out or repeated
  (two definitions of `modified_WEAT`).
- **Two directions.** A *semantic* gender direction from definitional pairs (`ES_definitional_pairs.json`
  etc.) and a *grammatical* one from inanimate feminine/masculine nouns (`feminine_nouns_es`,
  5,000 per gender; a classifier, then PCA/LDA variants, cells 63–125).
- **Bias test** (`modified_WEAT`, cells 215–232): occupation word *pairs* (feminine form,
  masculine form, e.g. *doctora/doctor*) scored against 8 male / 8 female definitional words
  (*hombre, niño, padre…* / *niña, mujer, madre…*); permutation p-value (`_weat_pval`).
  Career/family WEAT in cells 243–250. **Only animate occupation words are tested**, which is
  the gap the PI noticed.

## Omrani Sabbaghi & Caliskan 2022 (`GG_Disentangling/main.ipynb`, 81 cells)

Clean, linear notebook; saved outputs are for Spanish.

1. Load 5,000 inanimate nouns per gender (`data/nouns/<lang>-{masc,fem}-v2.txt`); keep those in
   the vocabulary (Spanish: 5,381 m / 4,837 f), equalise, pair them up.
2. **Grammatical-gender direction** = coefficients of a LinearSVC trained on 3,000 + 3,000 nouns
   (train acc 0.99; next 6,000 nouns held out, 0.985).
3. **Iterative removal** (`num_iter`, set per language, e.g. French 27): project the direction out
   of **every word in the vocabulary**, refit the SVC, repeat. Tracked each round: SVC accuracy,
   WEATs (science/arts, career/family, a grammatical-gender WEAT), Gonen et al.'s same- vs
   different-gender noun similarity.
4. **Single-category GG-WEAT on inanimate nouns** (cells 46–58): for each of 1,000 masculine and
   1,000 feminine nouns, an SC-WEAT effect size against *man* words (*hombre, padre…*) vs
   *woman* words, before and after removal. Saved Spanish output:
   - masculine nouns: mean d **+0.84 → −0.24** (998/1000 move toward feminine);
   - feminine nouns: mean d **−0.90 → −0.29** (969/1000 move toward masculine).
5. ValNorm (valence norms) before/after, analogy test, plots.

## What this means for the PI's idea

- **The 2022 paper already runs a gender-association test on inanimate nouns** (step 4). So the
  gap is narrower than "nobody tested inanimate nouns": Zhou et al. didn't, Omrani Sabbaghi &
  Caliskan did, at least for Spanish in the notebook (the paper may report other languages; check).
- **The question their numbers leave open is the one closest to ours:** after removal, do
  grammatically masculine and feminine *things* still differ in social-gender association? In the
  saved Spanish run they end up close (−0.24 vs −0.29). Whether that difference is reliable,
  and whether it tracks the *human* gender ratings of the objects (Glasgow GEND), isn't tested.
- **Design issues to keep in mind (from reading the code, unverified against the paper):**
  - The 1,000 nouns scored in step 4 come from the 3,000 the direction was **trained on**; a
    held-out set (the next 3,000 exist) would be cleaner.
  - Removal is applied to **every** word, including the *man/woman* attribute words, so the
    attributes change too.
  - After removal both groups lean "feminine" (negative d). That could be a baseline shift in the
    attribute words rather than anything about the nouns; a within-pair difference (masculine
    minus feminine nouns) is more interpretable than the raw means.
  - The noun lists' gender labels and animacy filtering need checking against Wiktionary
    (our rule: gold labels never come from a model or unverified lists).
- **Natural link to the base-model study:** our Glasgow-norm adjective machinery and the
  German/Spanish lexicons would give an object-level social-gender rating to test against, and the
  2022 code already covers German.

## Not yet done

- Reading both papers (the PI is doing this); checking the notebook numbers against the papers.
- Running anything. The Common Crawl vectors are ~4.2 GB each as text; loading 1M × 300 in
  float64 takes ~2.4 GB RAM per language, fine on the Mac one at a time.
