# Shared plan for the two embeddings papers

*2026-11-10. Decision (PI, after discussion): **two papers**, not one.*

- **Paper 1, the Zhou reanalysis:** `paper1-zhou-reanalysis-skeleton.md`. Reproduction + audit of
  Zhou et al. (2019), then corrected tests showing the asymmetry is real and why. Numbers frozen.
  Needs the author-contact decision before submission.
- **Paper 2, inanimate nouns:** `paper2-inanimates-skeleton.md`. Grammatical vs social gender on
  inanimate nouns under three mitigations (Zhou Shift/Hybrid, Omrani erasure, Bolukbasi via
  alignment). May still grow (German arm, LLM link). Cites paper 1 for one fact only: Shift's
  native-yardstick effect is partly algebraic, which motivates the orthogonal yardstick.
- Why two: different readers and review temperaments; no result appears in both; different
  timelines; the dependency runs one way. Risk accepted: two manuscripts to track, so paper 1 goes
  first and paper 2 is next, not parallel.

Below: the literature map (B) and the repo split plan (C), shared by both papers.

---

## B. Literature map

Legend: **Z** = in Zotero (key), **S** = PDF under `sources/` only, **—** = not held. "Role" says
what the paper needs it for. Add new items under the Zotero `Gender` project with the tag
`review:embeddings-study` (already used on the two Valentini papers).

### B1. The two targets and their lineage

| paper | held | role |
|---|---|---|
| Zhou et al. 2019 | Z UWAPUV7H | target |
| Omrani Sabbaghi & Caliskan 2022 | Z VM62ICFA | target (check the arXiv ID in the record: lit-verification found 2210.13451 mis-recorded) |
| Bolukbasi et al. 2016 | Z 693JWKEJ | Hybrid's English half |
| Gonen & Goldberg 2019 (lipstick) | Z QIB2CVLH | "removed from the direction, not from the space" is exactly our Shift finding |
| Zhao et al. 2018 GN-GloVe | Z 4NVJBQRY | same group, alternative mitigation |
| Zhao et al. 2020 multilingual bias | Z V6FEYK4A | same group, aligned-space bias |
| Ravfogel et al. 2020 INLP | — | Omrani = INLP (O12) |
| Belrose et al. 2023 LEACE | Z XABMMT3S | erasure that reaches chance in one step; contrast to 14–28 iterations |
| Xiao & Li 2026 | Z XAPHS4QE | only paper reusing Zhou's estimator (`zhou-downstream-reuse.md`) |
| Dev & Phillips 2019; Ethayarajh et al. 2019 | Z JU6XSVCL, PCKTBQTT | background on projection-based debiasing and its measurement |

### B2. Grammatical gender in distributional space

| paper | held | role |
|---|---|---|
| Gonen, Kementchedjhieva & Goldberg 2019 | Z 4HCPF8JT / 9RM4XRNV | same-gender nouns closer; debiasing at training time |
| McCurdy & Serbetçi 2020 | Z C6ZHWFU9 | grammatical outweighs topical gender |
| Williams et al. 2021 | Z 22T8FCVJ | MI between inanimate gender and adjectives/verbs |
| Stańczak et al. 2023 | Z CQYGBENU | causal framing of the same question |
| Kann 2019; Flint & Ivanova 2024 | Z AFDEHHXN, 7XG47XXR | computational Boroditsky analogues |
| Brinkmann et al. 2025 | Z PFIGSTYW | grammatical concepts in LLMs (for the discussion link) |
| Zmigrod et al. 2019 (counterfactual data augmentation; source of Omrani's animacy list) | — | Omrani's animate filter |
| Lauscher & Glavaš 2019 (XWEAT); Lauscher et al. 2020 | — | cross-lingual WEAT, attribute translation problems |
| Matthews, Grasso, et al. 2021 "Gender bias in NLP for languages with grammatical gender" | — (verify title) | survey of the problem space |
| Kurpicz-Briki 2020 (German/French WEAT) | — | French attribute-word precedent |

### B3. Validity of embedding bias metrics (backs §3 and §6)

| paper | held | role |
|---|---|---|
| Caliskan, Bryson & Narayanan 2017 | Z D3LKABP5 | WEAT definition, permutation test as specified |
| Valentini et al. 2022 (frequency dependence) | Z 4P3XBKTE | frequency control in gender ValNorm |
| Valentini et al. 2023 (PMI; Glasgow r 0.44–0.58) | Z VR9TMA3W | English precedent for §5.2 |
| Chaloner & Maldonado 2019 | Z CDZFCADD | WEAT across embeddings |
| Goldfarb-Tarrant et al. 2021 (intrinsic vs extrinsic) | — | why intrinsic numbers need validity checks |
| Antoniak & Mimno 2021 (seed lexicons) | — | attribute-word choice; our resampling |
| Du, Fang & Nguyen 2021 "Assessing the reliability of WEAT" | — | WEAT instability |
| Schröder et al. 2021; Delobelle et al. 2022 | — | metric comparability |
| Zhang, Sneyd & Stevenson 2020 (robustness of WEAT) | — | |
| Badilla et al. 2020 WEFE | — | tooling; shows how WEAT p-values are normally computed |
| Blodgett et al. 2020 ("Language (technology) is power"), 2021 (benchmark pitfalls) | — | framing of what "bias" claims need |
| Phipson & Smyth 2010 "Permutation p-values should never be zero" | — | the Z10 point in one citation |

### B4. Human gender norms and external criteria (backs §4.4, §5.2)

| paper | held | role |
|---|---|---|
| Scott et al. 2019 Glasgow Norms | Z VGLFA4XI | GEND rating used |
| Lewis & Lupyan 2020 | — | closest English precedent (r 0.63/0.59); must be cited |
| Toney & Caliskan 2021 ValNorm | — | naming precedent; Omrani's semantic check |
| Vankrunkelsven et al. 2024 Dutch norms | Z QTIIR79E | only non-English word-gender norm |
| Du et al. 2019 word-association stereotypes | Z ZR2F4SSV | |
| Garg et al. 2018 PNAS | — | occupation share as criterion |
| Caliskan et al. 2022 (frequency, syntax, semantics) | — | SC-WEAT on whole vocabularies |
| Misersky et al. 2014 (role-noun gender norms, 7 languages) | — | Spanish/French occupation stereotype norms, alternative to BLS/Eurostat |
| BLS 2025, Eurostat 2024 | data | mapping v1 sources |

### B5. Generic masculine (backs §4.3)

| paper | held | role |
|---|---|---|
| Gygax et al. 2008 (FR/DE/EN role nouns) | — | masculine plural read as male-biased |
| Stahlberg, Braun, Irmen & Sczesny 2007 (review) | — | overview |
| Brauer & Landry 2008 (French) | — | |
| Gabriel & Gygax 2008; Gabriel et al. 2018 | — | |
| Spanish: Nissen 2002/2013; RAE 2020 report on inclusive language | — | |
| Bartl, Nissim & Gatt 2020 (German BERT generic masculine) | — | embeddings/LM precedent |

### B6. Reproducibility and reporting (backs §3 tone and §8)

| paper | held | role |
|---|---|---|
| van Miltenburg et al. 2021 "Preregistering NLP research" | Z UGIPE5XP | our preregs |
| Belz et al. 2021 (systematic review of reproduction in NLP) | — | |
| Cohen et al. 2018 (three dimensions of reproducibility) | — | |
| Pineau et al. 2021 (ML reproducibility checklist) | — | |
| Lipton & Steinhardt 2018 ("Troubling trends") | — | optional |
| ACL Rolling Review reproducibility / responsible-NLP checklist | — | for the venue |

### B7. Linguistics background (one paragraph in §2)

| paper | held | role |
|---|---|---|
| Corbett 1991 *Gender* | — (cited by Zhou) | definitions; epicene nouns |
| Harris 1991 (Spanish gender exponence) | — | *el agua*, -a masculines |
| RAE *NGLE* on epicenes and common gender | — | zero-point anchor set |
| Boroditsky et al. 2003; Mickan et al. 2014; Elpers et al. 2022 | S / Z (01 collection) | one sentence on human evidence being weak |

### Search plan

1. Zotero first: `zotero_semantic_search` for each "—" row; several may already be in `Topics/`.
2. alphaXiv / Semantic Scholar for the "—" rows in B3 and B5; save to Zotero with
   `review:embeddings-study`.
3. Forward citations of Zhou (done: `zhou-downstream-reuse.md`, 84 works) and of Omrani (not
   done): anyone building on GG-WEAT or the erasure since 2022.
4. Run `feynman-to-sources` on the final list to pull PDFs into `sources/`.

---

## C. Splitting into a standalone repo

### Why split

The study depends on the monorepo for one import (`gbleed.lexicon._wordnet`), two data files
(the ES/FR Wiktionary lexicons, Glasgow norms) and a drive layout. Everything else is already
inside `embeddings/`. A standalone repo is what reviewers and the Zhou/Omrani authors should be
linked to; the monorepo's history and LLM machinery would only confuse them.

### What moves (with history)

Use `git subtree split -P embeddings -b embeddings-split` (or `git filter-repo --subdirectory-filter
embeddings` on a fresh clone), then push to the new repo. Both keep every commit that touched
`embeddings/`, so pre-registration hashes stay verifiable. **Record the mapping** old → new
commit for the hashes quoted in the preregs (e928283, 52cc9ed, 0421290, dad4469, …) in the new
`README`, or keep a pointer to the monorepo commit.

| item | action |
|---|---|
| `embeddings/{docs,scripts,data,results}` | move as-is |
| `docs/design/embeddings-handoff.md`, `embeddings-lit-verification.md` | move into `docs/` of the new repo |
| `docs/feynman/outputs/zhou2019-inanimate-nouns.md` (if still useful) | move or drop |
| `~/zhou-blind-check/{TASK.md,REPORT.md,page*.txt}` and `~/zhou-blind-check-claude/{TASK.md,REPORT.md}` | add as `blind-checks/gemini/` and `blind-checks/claude/`; exclude vectors and the `zhou_repo` copy (point at commit 6392702) |
| `data/lexicon/es_nouns.csv`, `fr_nouns.csv` (frozen copies) | copy into `data/lexicon/` with the kaikki dump date and the `gbleed lexicon` commit in a `PROVENANCE.md`; these are derived from Wiktionary (CC BY-SA), so say so |
| `data/raw/glasgow/` | do **not** redistribute; add a fetch note (OSF) and the checksum |
| `gbleed.lexicon._wordnet` | vendor the one function into `scripts/_wordnet.py` (check what it needs: OMW via NLTK?) |
| `embeddings/external/` | stays git-ignored; `README` gives both repos' URLs and commits, and `scripts/fetch_embeddings.sh` covers vectors with `SHA256SUMS` |
| drive paths (`/Volumes/Lexar/gbleed/shared/embeddings/`) | replace with one env var `EMB_STORE` defaulting to `./store/`, set in a tiny `paths.py` |
| root `CLAUDE.md` rules that still apply | fold the relevant ones (gold labels from Wiktionary, preregister, frozen lists, no manual-check items) into the new `CLAUDE.md`/`CONTRIBUTING.md` |

### What stays in the monorepo

- A row in the root `README.md` table pointing at the new repo, and `embeddings/README.md` saying
  it moved (or keep `embeddings/` as a git submodule, which is less confusing for the PI's own
  workflow but adds friction for readers; recommendation: pointer, not submodule).
- `docs/decisions.md` rows already logged there.
- The Glasgow norms and lexicon builders.

### Repo layout to aim for

```
zhou-reanalysis/                 (name TBD)
  README.md                      claims in 10 lines, how to reproduce each table, licence
  CITATION.cff, LICENSE          code MIT/Apache; data CC BY-SA 4.0 where Wiktionary-derived
  paper/                         the write-up (this skeleton → draft → LaTeX/Markdown)
  docs/                          preregs, upstream-issues.md, zhou-shift-identity.md, summaries
  blind-checks/                  the two independent reports + TASK.md
  scripts/                       as now; add `make tables` that regenerates every number in the paper
  data/                          frozen lists, lexicon copies, PROVENANCE.md
  results/                       as now (35 MB; fine), each folder with its README
  external/                      git-ignored clones; README with URLs + commits
  tests/                         the held-out/disjointness assertions, NFC/coverage checks
```

### Before splitting

1. Finish any open prereg so the split doesn't straddle a frozen list.
2. `grep -rn 'Volumes/Lexar\|data/lexicon\|gbleed' embeddings/scripts` and fix every path (12 hits
   today, 10 files).
3. Decide the licence question for the upstream lists (neither upstream repo has one): we cite,
   never copy; check that no tracked file under `embeddings/data/` contains their lists verbatim.
4. Decide whether the author contact happens before or after the repo goes public (a public issue
   log naming Z1–Z18 is itself a form of contact).

### Zotero fill, 2026-11-10 (Haiku agent, by DOI/arXiv; no PDFs attached; tag `review:embeddings-study`)

Added (key, collection): Ravfogel 2020 T9VPBIMF (02); Zmigrod 2019 PZXBJ9JB (02); Lauscher &
Glavaš 2019 XL9VX288 (02); Goldfarb-Tarrant 2021 EIE4PJ9E (02); Antoniak & Mimno 2021 PXTSBEV6
(02); Du, Fang & Nguyen 2021 TLJ5TGPR (02); Delobelle 2022 NTA5M2L5 (02); Zhang, Sneyd &
Stevenson 2020 M5T4K6A6 (02); Badilla 2020 WEFE GX8ID365 (02); Blodgett 2020 UMQHUYXJ, 2021
ENUJIPMD (02); Phipson & Smyth 2010 QMQSH6S5 (02); Lewis & Lupyan 2020 RYS9WYXE (05); Toney &
Caliskan 2021 BT9SSSQU (05); Garg 2018 PHP8UZSB (05); Caliskan 2022 INRNQH4T (05); Misersky 2014
UJI887PY (05); Gabriel, Gygax & Kuhn 2018 FHZ5YC6L (01); Nissen 2013 KLW3YEPB (01); Corbett 1991
4HTK7RD7 (01, Open Library record, check fields); Bartl, Nissim & Gatt 2020 88LNGXQ2 (01); Belz
2021 ELJNYTW8 (08); Cohen 2018 9NWBM8Z9 (08); Pineau 2021 H9RCDR7I (08); Lipton & Steinhardt
2018/19 8RDXV99D (08).

Unresolved (not added, nothing guessed): Lauscher 2020 (no title in row); Matthews et al. 2021
(closest: "Gender Bias in NLP Across Human Languages", TrustNLP 2021, 10.18653/v1/2021.trustnlp-1.6);
Kurpicz-Briki 2020 (only 2021 Frontiers paper, 10.3389/fdata.2021.625290); Schröder 2021 (only
2024); Gabriel & Gygax 2008 and Gygax et al. 2008 (candidates 10.3758/brm.40.1.206,
10.1080/01690960701702035, PI to pick); Stahlberg et al. 2007 (chapter, no DOI); Brauer & Landry
2008; Harris 1991; Nissen 2002; RAE NGLE; RAE 2020 report; ARR checklist.

Flags: VM62ICFA (Omrani) has DOI field `10.48550/arXiv.2210.13451`, wrong; correct arXiv is
2206.01691, AIES DOI 10.1145/3514094.3534176 already in Extra. Not changed. New items are not in
the semantic-search index until `zotero_update_search_database` runs.
