# Orientation: start here (updated 2026-11-11)

Paths are relative to `embeddings/` in the monorepo (`/Users/watcher/GitHere/gender`) and to the
repo root in the standalone `gender-embeddings` repo (same files; see "Two repos" below).

## Where things stand (2026-11-11)

**The project.** Two papers grow out of one question (does Zhou et al. 2019's bias mitigation keep the
grammatical gender of inanimate nouns?): (1) a reanalysis of Zhou et al. 2019, (2) the inanimate-noun
findings. Skeletons and drafts in `drafts/` (git-ignored, local only).

**One-paragraph state.** Zhou's cross-lingual bias holds and is stronger than their test showed; it is
specific to occupations (ordinary person nouns show a weaker masculine default that a feminine cue
cancels). Most of their quantitative support doesn't hold: Table 1 p-values are guaranteed or tie
artefacts, Shift's drop is algebraic, Hybrid = Shift on Table 1, Table 2 has no feminine cues, the
French numbers come from an unreleased list, and the attribute lists have errors (Z19, Z23). With a
fair zero point their monolingual asymmetry is inconclusive in Spanish and borderline in French.
Grammatical gender of inanimates survives the mitigations; human gender ratings of objects show up in
the vectors. Full claim table: `docs/zhou-summary.md`; inanimates: `docs/inanimates-summary.md`.

**Authors.** Emailed 2026-11-11; Ryan Cotterell replied wanting details for a marked arXiv/Anthology
correction and asked Jieyu Zhao about unreleased code. PI replied with a summary and a technical
attachment (`drafts/issues-for-authors-v2.md`). Appendix draft: `drafts/zhou-appendix-issues.md`
(paper labels B1–B21 with a Repo ID column; never renumber Z IDs).

**Open next.**
- French: a French person-noun list (by rule; PI can't review French) and whether to rerun the two
  person-noun follow-ups in French (PI: "tomorrow", i.e. 2026-11-12).
- `gender-embeddings` is synced but **not pushed**: waiting for the authors to say they want it public.
- If the authors send the unreleased files (mitigated vectors, hard-debiased English, alignment
  details, French Table 1 list), check B3/B13 (Hybrid, De-Align, translation) against them.
- Paper drafting (PI), with a separate Claude; point it at `zhou-summary.md`, today's rows of
  `decisions.md`, the preregs and `upstream-issues.md`.

## Conventions that bite

- **Pre-register, commit the scoring script, then score.** Every study in `docs/*-prereg.md` has an
  adoption record; deviations are dated sections at the end of the prereg with both versions shown.
- **Attribute words:** the corrected lists (Spanish *tío*; French de-duplicated, names dropped) are
  primary and the scripts' **default** (`scripts/zhou_shift.py`, `attributes()`).
  `ZHOU_ATTR=published` reproduces Zhou's lists and the per-study result folders frozen before
  2026-11-11; the corrected outputs are in `results/attrfix_sensitivity/`. Rerunning a study with the
  default overwrites its frozen folder (git shows it).
- **Issue IDs:** Z1–Z23 (Zhou), O1–… (Omrani), X1–X13 (XWEAT, monorepo only), in
  `docs/upstream-issues.md`. Every decision is a row in `docs/decisions.md`.
- **Labels:** gold labels never come from a model; Claude suggestions are marked
  `claude_suggest_unverified` and the PI reviews lists (`.numbers` → `*_edited.csv`).
- **Exploratory vs pre-registered:** sandbox sensitivity runs go to their own result folder with a
  README (`results/attrfix_sensitivity/`, `results/list_audit/`, `results/z23_check/`).

## Studies

| Study | Prereg | Result (short) |
|---|---|---|
| C1: inanimates keep grammatical gender after mitigation | `c1-prereg.md` | survives, 16/16 (ES 84%, FR 1.09–1.13×) |
| Gender ValNorm: object gender ratings in vectors | `gvalnorm-prereg.md` | yes in both languages (β 0.18–0.34) |
| Zero point for Zhou's asymmetry | `zhou-zero-point-prereg.md` | ES inconclusive; FR borderline; FR with English words: asymmetry |
| Crime-noun probe | `crime-noun-probe-prereg.md` | ES male-leaning; FR borderline |
| Generic masculine (occupations) | `generic-masculine-prereg.md` | ES supported; FR inconclusive |
| Word-pair mechanism | `wordpair-mechanism-prereg.md` | not the dictionary; stereotype β 0.30 |
| Expanded feminine cues (87 ES / 86 FR) | `wordpair-cues-prereg.md` | intrusion ES 0.29, FR 0.19; mitigations cut ~⅓ |
| Person-noun baseline | `wordpair-persons-prereg.md` | persons none, occupations 0.29; masculine default 0.28 vs 0.51 |
| Person-noun zero point (ES) | `persons-zero-point-prereg.md` | inconclusive, groups not distinguishable |
| Person-noun generic masculine (ES) | `persons-genmasc-prereg.md` | occupations supported, persons inconclusive (unstable with vocabulary) |
| XWEAT sensitivity (monorepo only) | `xweat-sensitivity-prereg.md` | see `results/xweat_sensitivity/README.md` |

Reproductions without a prereg: `scripts/zhou_shift.py`, `zhou_hybrid.py`, `zhou_figures.py`,
`zhou_sec43_44.py`, `zhou_wordpair.py`, `repro_es.py`; Shift identity write-up in
`docs/zhou-shift-identity.md`.

## Blind checks

Two rounds, each with a Claude and a Gemini agent given only the paper, code and vectors: the first
with a claim list (`~/zhou-blind-check*`), the second open-ended (`~/zhou-blind-open-{claude,gemini}`,
TASK.md 806b032). What each found independently is in `docs/upstream-issues.md` (notes after Z22) and
the appendix preamble. The non-blind Gemini walkthrough is `blind-checks/stress_test_zhou2019.ipynb`
in `gender-embeddings` (its §4 conclusion was wrong; see the person-noun prereg).

## Two repos

- **Monorepo** `gender/embeddings/`: where work happens. Data on the external drive
  (`/Volumes/Lexar/gbleed/shared/`: `embeddings/muse`, `kaikki`, `person_adjectives`).
- **`gender-embeddings`** (standalone, for sharing): made by `git subtree split`; `docs/HISTORY-MAP.md`
  maps old commit hashes, including a post-split table. Scripts use `scripts/paths.py`
  (`EMB_STORE`, default `./store`); `external/`, `store/`, `drafts/` are git-ignored. Sync by applying
  the monorepo diff with `--relative=embeddings`, excluding XWEAT and `.numbers` files, appending
  `decisions.md` rows, and rewriting `python embeddings/scripts/` → `python scripts/` in docstrings.

---

# Original orientation: the two upstream repos (2026-11-08)

*Kept as written on 2026-11-08 except where marked; later findings supersede parts of it (see above
and `docs/upstream-issues.md`).*

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
  (*hombre, niño, padre…* / *niña, mujer, madre…*); permutation p-value (`_weat_pval`; 1,000 random re-partitions, see Z10).
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

## Not yet done (as of 2026-11-08; all since done)

- Reading both papers (the PI is doing this); checking the notebook numbers against the papers.
- Running anything. The Common Crawl vectors are ~4.2 GB each as text; loading 1M × 300 in
  float64 takes ~2.4 GB RAM per language, fine on the Mac one at a time.
