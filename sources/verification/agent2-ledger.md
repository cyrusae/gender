# Agent 2 verification ledger (folders 04, 05, 06, 07 incl. 07A/07B, 08)

Evidence key: "PDF pN" = local PDF text read (first pages plus full-text greps); URL = web check. 07A and 07B list subsets of the 45 papers in folder 07, so they share rows. No PDFs were renamed; mismatched file names are listed in the last section. Git was not touched.

Totals: 95 papers. VERIFIED 3, CORRECTED 90, UNVERIFIABLE 2 (Munoz-Garcia, Schad).

## 04 Steering, Probing, Interpretability (10)
| id | file | status | what was wrong / changed | evidence |
|---|---|---|---|---|
| Belrose LEACE | 04 README | VERIFIED | arXiv 2306.03819 and claims match | PDF p1 |
| Li ITI | 04 README | VERIFIED | none | PDF p1 |
| Turner ActAdd | 04 README | CORRECTED | removed invented "all positions"/grammar collapse/CAA label; limitation is user-chosen c, l, a; models GPT-2-XL, OPT, LLaMA-3 | PDF limitations, Alg. 1 |
| Hewitt & Liang | 04 README | CORRECTED | ELMo only; POS/dependency edge; MLP 97.3/92.8/4.5, linear 97.2/71.2/26.0 | PDF |
| Buttigieg et al. 2610.08559 | 04 README | CORRECTED | BBQ + general-knowledge benchmarks, Llama-3.1-8B-Instruct | PDF p1 |
| Jorgensen et al. | 04 README | CORRECTED | removed invented "centroid-dominated" finding; paper hypothesises anisotropy; models GPT-2, GPT-J, GPT-NeoX, Llama-2 | PDF |
| Sun massive activations | 04 README | CORRECTED | removed <0.1%/pruning/cosine claims; abstract numbers (100,000x), models | PDF p1 |
| Xiao attention sinks | 04 README | CORRECTED | removed BOS/"catastrophic" claims; project reasoning flagged | PDF |
| Hu & Levy | 04 README | CORRECTED | pages 5040-5060; removed overclaim | PDF |
| "Robinson 2024" My Answer is C | 04 README | CORRECTED | real authors Wang, Ma, Hu, Weber-Genzel, Rottger, Kreuter, Hovy, Plank, Findings ACL 2024 pp. 7407-7416; MMLU + OpinionQA; project link marked [unverified] | PDF p1 |

## 05 Lexical Norms and Evaluation (8)
| id | file | status | what was wrong / changed | evidence |
|---|---|---|---|---|
| Scott 2019 Glasgow | 05 README | CORRECTED | scales (AROU/VAL/DOM 1-9, others 1-7); removed invented Scottish bias, "excludes inflections" | PDF |
| Vankrunkelsven 2024 | 05 README | CORRECTED | removed "10,388 nouns", "first and only"; r=.24 is correlation with grammatical gender class; arousal/split-half marked unverified | PDF |
| Brysbaert 2014 | 05 README | CORRECTED | 37,058 words + 2,896 expressions, >4,000 MTurk; removed split-half .91, lexical decision claims | PDF |
| Du, Wu, Lan 2019 | 05 README | CORRECTED | SWOW-EN >12,000 cues; stereotype propagation after Zhou 2003; removed "3 million responses", WEAT/GloVe claims | PDF |
| Tumurchuluun 2026 (2609.08322) | 05 README | CORRECTED | Llama-3.1-8B, Qwen3-8B, Gemma-2-9B; 4 languages; removed invented depths, de/fr, integrated gradients; link to project marked own interpretation | PDF |
| Bartl, Murphy, Leavy 2025 | 05 README | CORRECTED | adapts Tibblin 2023 to EN/DE; models GPT-2, OLMo, Qwen2.5-32B, LeoLM; removed Llama-3, neopronouns | PDF |
| Ducel, Neveol, Fort 2024 | 05 README | CORRECTED | seven models incl. xglm, BLOOM, cerbero; >52,000 French and >4,100 Italian letters; removed CamemBERT/FlauBERT, 70-85% | PDF |
| Munoz-Garcia 2025 | 05 README | UNVERIFIABLE | local PDF is unrelated vMFER RL paper (2405.08638); DOI 10.1016/j.knosys.2025.113915 resolves to a different KBS title (BiasBloom); invented methodology removed, entry marked unverified | PDF; DOI search (publisher page not retrievable) |

## 06 Story Generation and Personification (27)
All 27 entries rewritten from PDFs; author names corrected in headings, mermaid, tier lists and matrix.
| id | status | what was wrong / changed |
|---|---|---|
| Aggarwal et al. MORPHOGEN | CORRECTED | author Aggarwal; model list (Llama, Qwen3, Gemma2/3, Phi4, not GPT-4o-mini/Qwen2.5); swapped 70B/8B numbers |
| Benedetto 2026 | CORRECTED | EACL 2026 (19th), pp. 5648-5668; commercial models only; anthology id unverified |
| Epley 2007 | VERIFIED | |
| Eyssel & Hegel 2012 | CORRECTED | N=60, FloBi, F(1,59)=34.68 etc. |
| Feng, Fang, Evans 2026 | CORRECTED | models GPT-4o-mini, Llama-3.1-8B, DeepSeek-V3.2; removed "12 axes", ">82%" |
| GRUFF (Mewes, Lauscher, Gautam) | CORRECTED | author order; no reasoning evaluated; removed ">60%" |
| Grohmann 2009 | CORRECTED | scale items (expresses tender feelings, tender); removed N>1,200 |
| Hamilton & Mimno | CORRECTED | models Claude Haiku 4.5, Gemini 3.1 Flash-Lite, GPT-5.4-Mini, OLMo 7b Thinking; not traced to RLHF |
| Hida, Kaneko, Okazaki | CORRECTED | pp. 14507-14530; 26 open LLMs on BBQ; removed Spearman 0.12; anthology id unverified |
| Ranaldi, Pucci, Zanzotto 2024 | CORRECTED | heading "Hsu" wrong; removed 94.2/95.8/68.4/71.2 |
| Ilie 2025 | CORRECTED | PDF is a ~3.5 KB German abstract; removed 3,072 pairs, >85%; venue unverified |
| Kumar et al. (file "Jain") | CORRECTED | authors; HEXACO + Dark Triad (not Big Five); 23,400 stories; removed numbers |
| Silva, Brandao, Moro JBCS 2025 | CORRECTED | not "Caseli"; 840 works; Mistral-7B, LLaMA-3.2-3B only |
| Klein, Manna, Vanmassenhove | CORRECTED | BERT probing study, not generation; removed >92%/>70% |
| Lucy & Bamman 2021 | CORRECTED | davinci only; 2154 characters; removed 1,000 stories, 66.7/33.3 |
| Lutz et al. | CORRECTED | 5 models, 9 prompt types, 15 groups |
| Nass, Moon, Green 1997 | CORRECTED | N=40 not 48 |
| Finkley, Li, Walsh (Neutrality Bites) | CORRECTED | Pucci -> Finkley everywhere; 23.8K stories; model names unverified |
| Rejdebrant 2025 | CORRECTED | 40 stories (essay, not thesis claims); removed 300 completions, 67.3% |
| Saeed et al. | CORRECTED | title; FR/ES/DE/IT/RU not Arabic; DALL-E 3, Ideogram v3, Flux Pro 1.1 |
| Masoudian et al. (file "Shah") | CORRECTED | 148,082 stories, 5 LLMs; removed invented numbers |
| Sukumaran et al. | CORRECTED | French only; LSTM + transformer; venue unverified; removed 88% |
| Tang et al. MISGENDERED+ | CORRECTED | 5 LLMs not 22; removed 96/41.2/<18% |
| Brei et al. CASPER | CORRECTED | arXiv 2606.22454; 200 human + 4,400 LLM stories; models |
| Weeber et al. | CORRECTED | BBQ + OpinionQA; 3 models; removed 38% |
| Wentworth 2026 | CORRECTED | close reading of ChatGPT-5 stories, not 400 stories/computational |
| Zhao et al. Role-Play Paradox | CORRECTED | 3 models; removed 8-14% and 3.2x-5.8x |
Project-link claims throughout 06 softened or marked [unverified].

## 07 Reasoning Mode and CoT (45; 07A has 22, 07B has 23)
All 45 inventory entries, tier text, cluster labels, Replication Matrices 1-3 and Q4 rhetorical sections rewritten from PDFs; unknown model lists marked [unverified]. 07A/07B tier boxes and index notes corrected. Lanham 2023 is VERIFIED only for arXiv ID; entry still CORRECTED (see below).
Opposite-direction errors found: Jiang/Chua & Evans (paper finds R1 MORE faithful: 59% vs 7%); Prasad/Bhupatiraju & Nyaupane (76.9% of steps causally load-bearing, not "near-zero"); Tam/Kaneko et al. (CoT REDUCED bias); Liu/Sun et al. code (reasoning lowered bias rate 0.64 to 0.40); Wu/Rajwal et al. (thought bias weakly correlated with output bias, <0.6).
| README id (file author) | real authors / title | status | key corrections |
|---|---|---|---|
| Bansal | Du, Hoyle, Ruis, Locatelli et al., COLM 2026 | CORRECTED | invented models, r<0.25 |
| Chen 2025 | Lu, Fang, Ma, Li, Wang (MixReasoning) | CORRECTED | 40-60% unverified |
| Chen 2026 | Cheng, Lu, Chang, Wu | CORRECTED | single R1-Distill-Qwen-7B, math; removed meta-reasoning/Qwen3-14B |
| Ding | Zhou, Ling, Chen et al. | CORRECTED | removed inverted-U, models |
| Dua | Scalena et al. | CORRECTED | logit/attention probes; 10-20% invented; early exit 55% |
| Gao 2025 | Lewenberg, Sheftel, Karov | CORRECTED | venue/models unverified; 79.4% |
| Gao 2026 | Lei, Lin, Lu, Torr, Zou (2610.11765) | CORRECTED | QA tasks not creative; invented models/diversity collapse |
| Guo | Su, Healey, Nakov, Cardie | CORRECTED | models unverified |
| Huang | Ghosal et al. | CORRECTED | removed o1-preview/creative claims |
| Jiang | Chua & Evans | CORRECTED | opposite finding fixed |
| Kim | Kou et al. | CORRECTED | models unverified |
| Kocijan | Jumelet, Zuidema, Hupkes (CoNLL 2019) | CORRECTED | LSTM, not BERT |
| Kwon | Han et al. (TALE) | CORRECTED | authors |
| Lanham | Lanham et al. | CORRECTED | Amodei not an author; 175B model; not "always post-hoc" |
| Lee | Wang, Ma, Fang, Li (TypedThinker, ICLR 2025) | CORRECTED | authors, gains 3.4/6.5/7% |
| Li 2025 bias | Wu et al. | CORRECTED | BBQ, ADBP |
| Li 2025 story | Gurung & Lapata (COLM 2025) | CORRECTED | removed flattening claim |
| Lin | Gema et al. (TMLR) | CORRECTED | failure modes |
| Liu 2025 | Qin et al. (R-CHAR) | CORRECTED | SocialBench 34.64 to 68.59% |
| Liu 2026 | Sun et al. | CORRECTED | opposite direction fixed |
| Mao | Jang, Kim, Lee (KAIST) | CORRECTED | silicon sampling, KNOWS/DOES |
| McKenna | Wenshuo Wang (position paper) | CORRECTED | removed "proves" and models |
| Park | Banayeeanzade et al. | CORRECTED | 14 models; arXiv id unverified; TopP justification removed |
| Prasad | Bhupatiraju & Nyaupane | CORRECTED | opposite direction fixed |
| Rao | Subramonian et al. (COLM 2025) | CORRECTED | 6 models, 20.2% |
| Shaikh | Shaikh, Zhang, Held, Bernstein, Yang | CORRECTED | content narrowed to abstract |
| Sharma | Feng, Chen, Ma, Yip, Chersoni, Li | CORRECTED | authors |
| Song | Jang, Lee, Kim et al. | CORRECTED | RL (SCOPE) paper, not inference-time |
| Srinivasan | arXiv 2608.13430 | CORRECTED | wrong author "Srivastava" removed; authors not on extracted page |
| Sun | Ju et al. | CORRECTED | RPD/1PNS |
| Tam | Kaneko, Bollegala, Okazaki, Baldwin | CORRECTED | opposite direction fixed |
| Tan | Srivastava et al. | CORRECTED | basic math |
| Tu | Feng, Dou, Kong | CORRECTED | attention diversion/style drift terms belong to Wang et al. |
| Venkatesh | Lewis-Lim, Tan, Zhao, Aletras (2508.19827) | CORRECTED | overclaims removed |
| Wang Demystifying | Wang, Yang, Long, Ganguly, Wang, Chaudhary, Han | CORRECTED | |
| Wang Character | Tang et al. (HIT) | CORRECTED | venue unverified |
| Wu | Rajwal et al. | CORRECTED | opposite direction fixed |
| Xu | Xu & Zhang | CORRECTED | invented gender link removed |
| Yang | Matthews & Hamilton | CORRECTED | 1.1M poems, 16 LLMs |
| Zhang 2024 | Zhang et al. (TBS) | CORRECTED | models unverified |
| Zhang 2026 | Pearman et al. | CORRECTED | |
| Zhao 2026 | Liu et al. (R2-Write) | CORRECTED | |
| Zheng | Pan, Germino, Ma, Daly, Moniz, Hua, Chawla (Findings EMNLP 2026; 2609.30768) | CORRECTED | tabular Adult/COMPAS/Credit; "closest precedent" softened |
| Zhou story | Wang & Kreminski | CORRECTED | |
| Zhou OThink | Zhang et al. | CORRECTED | removed 50% and bias claims |

## 08 Statistical and Experimental Methodology (5)
| id | status | what was wrong / changed | evidence |
|---|---|---|---|
| Lakens & Delacre 2020 | CORRECTED | removed "gold standard", SGPV details and power claims not in paper | PDF p1 |
| Schad et al. 2020 | UNVERIFIABLE | local PDF is GestARLite (arXiv 1904.09843); README note claimed the opposite and was rewritten; entry content marked unverified against local PDF | PDF p1 |
| van Miltenburg 2021 | CORRECTED | pp. 613-623 (not 412-423); title; discussion paper, not an audit; removed "plan, not a prison" | PDF |
| Baayen et al. 2008 | CORRECTED | removed "exact Type I control", "mathematically mandatory" | PDF abstract |
| Fitts 2023 | CORRECTED | removed N_max cap and alpha' claims (not in abstract; full text not found); coverage ~90% kept | PDF p1 |

## PDF file names that do not match the paper (left unchanged)
04: "[Robinson et al. 2024] My Answer is C" is Wang et al. (Findings ACL 2024).
05: Munoz-Garcia 2025 PDF is an unrelated RL paper (Zhu et al., 2405.08638).
06: Pucci -> Finkley, Li, Walsh; Gautam -> Mewes, Lauscher, Gautam; Hsu -> Ranaldi, Pucci, Zanzotto; Jain -> Kumar et al.; Shah -> Masoudian et al.; Agarwal -> Aggarwal et al.; Wang -> Brei et al.; Weeber (authors listed in different order); Hida file title "Analyzing and Mitigating Gender Bias in Generated Stories" does not match the paper title.
07: Bansal=Du; Chen 2025=Lu; Chen 2026=Cheng; Ding=Zhou; Dua=Scalena; Gao 2025=Lewenberg; Gao 2026=Lei; Guo=Su; Huang=Ghosal; Jiang=Chua; Kim=Kou; Kocijan=Jumelet; Kwon=Han; Lee=Wang; Li 2025=Wu/Gurung; Lin=Gema; Liu=Qin/Sun; Mao=Jang; McKenna=Wang; Park=Banayeeanzade; Prasad=Bhupatiraju; Rao=Subramonian; Sharma=Feng; Song=Jang; Sun=Ju; Tam=Kaneko; Tan=Srivastava; Tu=Feng; Venkatesh=Lewis-Lim; Wang 2025 Character=Tang; Wu=Rajwal; Yang=Matthews; Zhang 2026=Pearman; Zhao 2026=Liu; Zheng=Pan; Zhou story=Wang; Zhou OThink=Zhang.
08: Schad PDF is the wrong paper; Fitts file title ("...Mixed-Effects Models") does not match the paper.
