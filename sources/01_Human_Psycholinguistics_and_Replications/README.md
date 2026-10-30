# Human Psycholinguistics and Replications: Literature Guide & Triage

This directory contains 19 foundational, empirical, and replication papers spanning over 25 years of research into whether grammatical gender shapes human perception, memory, concept formation, and personification (the Linguistic Relativity / Neo-Whorfian hypothesis), alongside related psycholinguistic baselines (name phonology, sound symbolism, and English dialectal pronominalization).

This guide provides an executive triage of these works, maps their thematic clusters and replication trajectories, provides a ready-to-use rhetorical citation matrix for manuscripts and rebuttal letters, and catalogs an exhaustive annotated inventory of all 19 sources.

---

## 1. Executive Triage & Tiered Reading List

To navigate these 19 papers efficiently when designing experiments, writing pre-registrations, or drafting manuscript discussion sections, use the following priority tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 1: Essential Anchors (Must-Read Methodological & Meta Baselines)  │
│  • Samuel et al. (2019) — Systematic review (43 studies, N=5,895)       │
│  • Elpers et al. (2022) — Registered replication of Phillips & Boroditsky│
│  • Xiao et al. (2026)   — Modern adjective-generation benchmark         │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
┌────────────────────────────────────▼───────────────────────────────────┐
│  TIER 2: Critical / Distinct Angles (Mechanisms, Modalities & Controls) │
│  • Brand et al. (2025)   — Megastudy (N=4,621): Animals vs Inanimates   │
│  • Bender et al. (2011)  — Implicit Association Test (IAT) null         │
│  • Mahajan & Durgin (2025) — High-powered implicit potency null        │
│  • Cassidy et al. (1999) — Name phonology & wug gender baseline         │
│  • Sidhu & Pexman (2015) / Sidhu et al. (2022) — Sound symbolism       │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
┌────────────────────────────────────▼───────────────────────────────────┐
│  TIER 3: Contextual, Historical & Exploratory Studies                   │
│  • Boroditsky et al. (2003) / Phillips & Boroditsky (2003) — Claims    │
│  • Boroditsky & Schmidt (2000) / Boroditsky (2001) — Genesis & paradigms│
│  • Mickan et al. (2014)  — First adjective-generation failure          │
│  • Bender et al. (2016)  — Cultural iconography vs grammar             │
│  • Segel & Boroditsky (2011) — Art history personifications            │
│  • Semenuks et al. (2017) — Serial position in adjective generation    │
│  • Kurinski & Sera (2011) — L2 acquisition timecourse                  │
│  • Pawley (2002)         — English vernacular inanimate "he/she"       │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: The Essential Anchors
*Read before finalizing design, preregistration, effect size expectations, or methods.*

1. **[Samuel, Cole & Eacott (2019)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSamuel%20et%20al.%202019%5D%20Grammatical%20Gender%20and%20Linguistic%20Relativity.pdf)** — *Psychonomic Bulletin & Review*
   - **Why it is essential:** The definitive systematic review and meta-evaluation of the field (43 empirical studies, 5,895 participants). Demonstrates that evidence for linguistic relativity from grammatical gender is weak, fragile, and concentrated almost exclusively in high-salience, explicit tasks with substantial demand characteristics. When high-salience tasks are excluded, support collapses from 32% to 11%, while null findings rise to 64%.
   - **Primary utility:** Sets the empirical baseline: human grammatical gender effects on object conceptualization are not settled fact; they are highly fragile.

2. **[Elpers, Jensen & Holmes (2022)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BElpers%20et%20al.%202022%5D%20Registered%20Replication%20of%20Phillips%20and%20Boroditsky.pdf)** — *Journal of Memory and Language*
   - **Why it is essential:** The gold-standard registered replication of the seminal Phillips & Boroditsky (2003) picture-similarity experiment. In a high-powered, preregistered sample ($N = 375$), Experiment 1 found a complete null for native Spanish and German speakers ($BF_{01} = 47.6$, $p = .16$). Crucially, Experiment 2 successfully replicated gender-like category effects when English speakers were explicitly *trained* on artificial categories ($BF_{10} > 10^{10}$).
   - **Primary utility:** Provides the methodological template for distinguishing native linguistic relativity from inductive statistical learning of category systems.

3. **[Xiao, Cremers, Straboni, Strickland & Peperkamp (2026)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BXiao%20et%20al.%202026%5D%20Grammatical%20Gender%20on%20Object%20Conceptualization%20is%20Weak.pdf)** — *Cognitive Science*
   - **Why it is essential:** The modern, high-powered test of the exact Boroditsky (2003) adjective-generation paradigm. Tests native French and German speakers across three preregistered experiments. Finds a complete null in French, a tiny and statistically inconclusive effect in German whose credible intervals cross zero ($\beta = 0.23, 95\%\text{ CrI } [-0.09, 0.684]$), and shows that intrinsic semantic features (natural entity vs. artifact) exert an effect twice as large ($\beta = -0.50$).
   - **Primary utility:** The closest published human benchmark to our LLM story/character adjective generation; establishes that even when tested in the native language, the human effect size is tiny or zero.

---

### Tier 2: Critical & Distinct Angles
*Read when constructing specific controls, testing implicit vs. explicit mechanisms, evaluating sound symbolism, or establishing wug baselines.*

4. **[Brand, Preininger, Kříž & Ceháková (2025)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBrand%20et%20al.%202025%5D%20Constraints%20on%20Linguistic%20Relativity%20Effects%20for%20Grammatical%20and%20Conceptual%20Gender.pdf)** — *Language and Cognition*
   - **Angle:** Megastudy methodology ($N = 4,621$ participants, $1,208$ items; Czech vs. English vs. Czech L1/English L2). 
   - **Key insight:** Grammatical gender affects conceptual gender reliably *only* for animals (biological animacy); for inanimate artifacts, the effect is substantially weaker and disappears entirely in implicit tasks.

5. **[Bender, Beller & Klauer (2011)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBender%20et%20al.%202011%5D%20Grammatical%20Gender%20in%20German%20-%20An%20Implicit%20Association%20Study.pdf)** — *Quarterly Journal of Experimental Psychology*
   - **Angle:** Implicit cognitive processing via the Implicit Association Test (IAT).
   - **Key insight:** When native German speakers perform speeded sorting without conscious reflection, inanimate nouns do not elicit automatic gender associations. Proves that earlier positive results were driven by explicit strategy or task demand.

6. **[Mahajan & Durgin (2025)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BMahajan%20&%20Durgin%202025%5D%20Implicit%20Test%20of%20Grammatical%20Gender%20Effects.pdf)** — *Language and Cognition*
   - **Angle:** High-powered preregistered conceptual replication of Konishi's (1993) implicit semantic-differential "potency" metric ($N = 240$ German/Spanish bilinguals, $N = 120$ English controls).
   - **Key insight:** Bayes odds of $24.5:1$ and $8.0:1$ against the grammatical gender hypothesis. Inanimate nouns do not systematically shift along the masculine-potency / feminine-softness axis.

7. **[Cassidy, Kelly & Sharoni (1999)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BCassidy%20et%20al.%201999%5D%20Inferring%20Gender%20from%20Name%20Phonology.pdf)** — *Journal of Experimental Psychology: General*
   - **Angle:** Surface phonological cues in names and nonce words (wugs).
   - **Key insight:** Across 7 experiments, phonological markers (final open vowels /i, ə/ vs. closed consonant codas, stress position, syllable count) strongly and automatically guide human gender inferences for unknown names and pseudowords. Crucial for calibrating LLM wug/nonce modifier arms.

8. **[Sidhu & Pexman (2015)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSidhu%20&%20Pexman%202015%5D%20Sound%20Symbolism%20and%20Gender%20in%20First%20Names.pdf)** & **[Sidhu, Vigliocco & Pexman (2022)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSidhu%20et%20al.%202022%5D%20Higher-Order%20Factors%20of%20Sound%20Symbolism.pdf)** — *PLoS ONE* / *Journal of Memory and Language*
   - **Angle:** Sound symbolism (maluma/takete) and higher-order semantic dimensions (Activity, Potency, Valence, Novelty).
   - **Key insight:** Sonorants and rounded back vowels map to round silhouettes, figurative "round" personalities, and feminine gender, whereas voiceless stops map to spiky shapes, aggressive traits, and male gender. Provides the psycholinguistic mechanism for why certain foreign noun surface forms trigger gendered character traits independent of grammar.

---

### Tier 3: Contextual, Historical & Exploratory Studies
*Consult for specific historical claims, art history baselines, second-language acquisition dynamics, or English dialectal pronominalization.*

9. **[Boroditsky, Schmidt & Phillips (2003)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBoroditsky%20et%20al.%202003%5D%20Sex%2C%20Syntax%2C%20and%20Semantics.pdf)** — Book Chapter in *Language in Mind* (MIT Press)
   - The origin chapter reporting the key/bridge adjective study, movie character voice assignment, and Gumbuzi learning. Note: Reports no statistics, no sample size, and no effect sizes for the adjective task; cites an unpublished manuscript.

10. **[Phillips & Boroditsky (2003)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BPhillips%20&%20Boroditsky%202003%5D%20Can%20Quirks%20of%20Grammar%20Affect%20the%20Way%20You%20Think.pdf)** — *CogSci 2003*
    - The peer-reviewed report of picture-similarity ratings, verbal interference, and Gumbuzi learning. Does *not* contain the adjective generation task.

11. **[Boroditsky & Schmidt (2000)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBoroditsky%20&%20Schmidt%202000%5D%20Sex%2C%20Syntax%2C%20and%20Semantics.pdf)** — *CogSci 2000*
    - Earliest conference report: object-proper name memory congruency (apple–Patricia vs apple–Patrick) and animal vs. artifact gender intuition correlations.

12. **[Boroditsky (2001)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBoroditsky%202001%5D%20Does%20Language%20Shape%20Thought.pdf)** — *Cognitive Psychology*
    - Foundational paper on Mandarin vs. English spatio-temporal metaphors (vertical vs. horizontal time). Establishes the cross-linguistic priming and training paradigm later ported to grammatical gender.

13. **[Mickan, Schiefke & Stefanowitsch (2014)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BMickan%20et%20al.%202014%5D%20Key%20is%20a%20llave%20is%20a%20Schl%C3%BCssel.pdf)** — *Yearbook of the GCLA*
    - The first published direct failure to replicate the Boroditsky et al. (2003) adjective association task and primed lexical decision in German and Spanish speakers.

14. **[Bender, Beller & Klauer (2016)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBender%20et%20al.%202016%5D%20Lady%20Liberty%20and%20Godfather%20Death.pdf)** — *QJEP*
    - Cross-linguistic test of allegorical personifications ("Lady Liberty", "Godfather Death"). Shows that cultural/artistic conventions account for personification gender; grammatical gender per se explains minimal unique variance.

15. **[Segel & Boroditsky (2011)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSegel%20&%20Boroditsky%202011%5D%20Grammar%20in%20Art.pdf)** — *Frontiers in Psychology*
    - Analyzes 765 personifications in Western art from the ARTstor database. Reports a 78% overall correspondence between grammatical gender and depicted sex (Italian 82%, French 80%, German 62%, Spanish 54% [N=24]).

16. **[Semenuks, Philips, Dalca, Kim & Boroditsky (2017)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSemenuks%20et%20al.%202017%5D%20Effects%20of%20Grammatical%20Gender%20on%20Object%20Description.pdf)** — *CogSci 2017*
    - Adjective generation in French, German, and Romanian speakers tested in English. Demonstrates that the effect is null for the 1st adjective and emerges only at the 2nd and 3rd adjectives, and is absent for concrete objects / body parts.

17. **[Kurinski & Sera (2011)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BKurinski%20&%20Sera%202011%5D%20Spanish%20Grammatical%20Gender%20and%20Object%20Categorization.pdf)** — *Bilingualism: Language and Cognition*
    - Longitudinal study of adult English speakers learning Spanish. Categorization shifts appear around 10 weeks, but are heavily constrained by pre-existing conceptual biases and natural/artificial object divisions.

18. **[Pawley (2002)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BPawley%202002%5D%20Using%20He%20and%20She%20for%20Inanimate%20Nouns%20in%20English.pdf)** — Book Chapter in *Ethnosyntax* (Oxford)
    - Anthropological and dialectological analysis of English varieties (Australian, Newfoundland, West Country) using *he* and *she* for inanimate tools, vehicles, and weather based on agency, affordance, and affective connection.

---

## 2. Thematic Overlap & Cluster Map

The 19 papers naturally organize into five thematic clusters representing the intellectual evolution of the field from early Neo-Whorfian assertions through replication failures, implicit psychometric scrutiny, and orthogonal phonological baselines:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CLUSTER 1: The Neo-Whorfian Origin Claims (2000–2011)                                  │
│ Boroditsky & Schmidt (2000) ──► Phillips & Boroditsky (2003) ──► Boroditsky et al.(2003)│
│                                           │                                            │
│                                           ▼                                            │
│                                 Segel & Boroditsky (2011)                              │
│                                 [Corpus Art Personifications]                          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ challenged by
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CLUSTER 2: The Replication Crisis & Methodological Demolition (2014–2026)              │
│ Mickan et al. (2014)     ──► Direct adjective generation & priming failure             │
│ Samuel et al. (2019)     ──► Meta-review: 64% null in implicit tasks                   │
│ Elpers et al. (2022)     ──► Registered replication failure of picture similarity      │
│ Mahajan & Durgin (2025)  ──► High-powered replication failure of implicit potency      │
│ Xiao et al. (2026)       ──► Adjective generation null in French, inconclusive in German│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ refined by
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CLUSTER 3: Implicit Tasks vs Demand Characteristics & Semantic Boundaries             │
│ Bender et al. (2011)     ──► Null on Implicit Association Test (speeded IAT)           │
│ Bender et al. (2016)     ──► Cultural allegories trump grammatical gender in art       │
│ Semenuks et al. (2017)   ──► Serial order effect: null on 1st word, emerges on 2nd/3rd │
│ Brand et al. (2025)      ──► Megastudy boundary: Animals replicate, inanimates fail    │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CLUSTER 4: Sound Symbolism, Name Phonology & Nonce Word Priors (Wugs)                 │
│ Cassidy et al. (1999)    ──► Open vowels / codas / stress predict name & nonce gender  │
│ Sidhu & Pexman (2015)    ──► Bouba/Kiki maps sonorants to female, voiceless to male    │
│ Sidhu et al. (2022)      ──► Higher-order dimensions: Potency, Activity, Valence       │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CLUSTER 5: Second-Language Transfer & Natural English Dialectal Baselines              │
│ Kurinski & Sera (2011)   ──► L2 Spanish learning alters categorization after 10 weeks  │
│ Pawley (2002)            ──► Dialectal English uses he/she for inanimates via agency   │
│ Boroditsky (2001)        ──► Cross-linguistic spatio-temporal metaphorical transfer    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Cluster Inter-Relationships and Dynamic Dialogues

1. **The Origin vs. Replication Battle (Cluster 1 vs. Cluster 2):**
   - [Boroditsky et al. (2003)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBoroditsky%20et%20al.%202003%5D%20Sex%2C%20Syntax%2C%20and%20Semantics.pdf) popularized the vivid idea that Germans think keys (*der Schlüssel*) are heavy, hard, and jagged, while Spaniards think keys (*la llave*) are intricate, little, and lovely.
   - [Mickan et al. (2014)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BMickan%20et%20al.%202014%5D%20Key%20is%20a%20llave%20is%20a%20Schl%C3%BCssel.pdf) attempted the exact adjective-generation task with German and Spanish speakers and found no significant differences, concluding the original result was a statistical fluke or an artifact of unstated procedures.
   - [Elpers et al. (2022)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BElpers%20et%20al.%202022%5D%20Registered%20Replication%20of%20Phillips%20and%20Boroditsky.pdf) registered an official replication of [Phillips & Boroditsky (2003)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BPhillips%20&%20Boroditsky%202003%5D%20Can%20Quirks%20of%20Grammar%20Affect%20the%20Way%20You%20Think.pdf) and yielded an unambiguous null ($BF_{01} = 47.6$).
   - [Xiao et al. (2026)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BXiao%20et%20al.%202026%5D%20Grammatical%20Gender%20on%20Object%20Conceptualization%20is%20Weak.pdf) tested adjective generation across French and German, finding no effect in French and an inconclusive estimate in German whose credible interval overlaps zero.

2. **Explicit Strategy vs. Implicit Automaticity (Cluster 1 vs. Cluster 3):**
   - Early positive findings were heavily criticized for eliciting conscious strategies: if you ask participants to rate whether a fork is like a man or woman, they realize the experiment is about gender and fall back on their native grammar as an explicit tie-breaker.
   - [Bender et al. (2011)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBender%20et%20al.%202011%5D%20Grammatical%20Gender%20in%20German%20-%20An%20Implicit%20Association%20Study.pdf) tested this using an implicit measure (IAT) where conscious strategy is precluded by reaction-time deadlines; the gender congruency effect vanished completely.
   - [Brand et al. (2025)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBrand%20et%20al.%202025%5D%20Constraints%20on%20Linguistic%20Relativity%20Effects%20for%20Grammatical%20and%20Conceptual%20Gender.pdf) reinforced this boundary across 4,621 subjects: inanimates fail to show effects in implicit designs, persisting only in explicit tasks.

3. **Artistic Personification vs. Cultural Memory:**
   - [Segel & Boroditsky (2011)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSegel%20&%20Boroditsky%202011%5D%20Grammar%20in%20Art.pdf) claimed that 78% of personifications in Western art match grammatical gender.
   - [Bender et al. (2016)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBender%20et%20al.%202016%5D%20Lady%20Liberty%20and%20Godfather%20Death.pdf) refuted this interpretation, proving that historical iconographic conventions (Greco-Roman mythological traditions, such as Greek *thanatos* vs. German folklore *Freund Hein*) explain the variance, and that grammatical gender exerts negligible independent influence.

4. **The Critical Confound: Phonology vs. Syntax (Cluster 4):**
   - When models or humans respond to foreign or nonce words (e.g., *kiki* vs. *bouba*, or pseudo-names in wug experiments), [Cassidy et al. (1999)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BCassidy%20et%20al.%201999%5D%20Inferring%20Gender%20from%20Name%20Phonology.pdf) and [Sidhu & Pexman (2015)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSidhu%20&%20Pexman%202015%5D%20Sound%20Symbolism%20and%20Gender%20in%20First%20Names.pdf) prove that open vowel codas, sonorants, and syllable stress induce strong gender attributions *independent of any grammatical agreement knowledge*. Any study testing foreign bare words must control for name phonology.

---

## 3. Rhetorical Ammunition & Citation Matrix

Use this matrix to locate precise citations for specific claims, defenses, and reviewer rebuttals in manuscript drafting:

| If you need to argue / claim that... | Cite these papers | Key quote / Evidence to deploy |
|---|---|---|
| **Human grammatical gender effects are weak, contested, or largely null in robust designs** | [Samuel et al. (2019)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSamuel%20et%20al.%202019%5D%20Grammatical%20Gender%20and%20Linguistic%20Relativity.pdf); [Elpers et al. (2022)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BElpers%20et%20al.%202022%5D%20Registered%20Replication%20of%20Phillips%20and%20Boroditsky.pdf); [Xiao et al. (2026)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BXiao%20et%20al.%202026%5D%20Grammatical%20Gender%20on%20Object%20Conceptualization%20is%20Weak.pdf) | Meta-review shows 64% null in implicit tasks (Samuel); registered replication was null with $BF_{01} = 47.6$ (Elpers); modern adjective generation is null in French and inconclusive in German with CrI spanning zero (Xiao). |
| **A null result in LLMs is expected under human cognitive baselines (not an LLM failure)** | [Elpers et al. (2022)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BElpers%20et%20al.%202022%5D%20Registered%20Replication%20of%20Phillips%20and%20Boroditsky.pdf); [Mickan et al. (2014)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BMickan%20et%20al.%202014%5D%20Key%20is%20a%20llave%20is%20a%20Schl%C3%BCssel.pdf); [Mahajan & Durgin (2025)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BMahajan%20&%20Durgin%202025%5D%20Implicit%20Test%20of%20Grammatical%20Gender%20Effects.pdf) | High-powered human studies fail to find conceptual bleedthrough for inanimate objects in similarity, adjective, or potency tasks. If LLMs show null transfer, they reflect human competence rather than deficit. |
| **A positive bleedthrough result in LLMs is theoretically surprising and represents hyper-sensitivity** | [Bender et al. (2011)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBender%20et%20al.%202011%5D%20Grammatical%20Gender%20in%20German%20-%20An%20Implicit%20Association%20Study.pdf); [Brand et al. (2025)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBrand%20et%20al.%202025%5D%20Constraints%20on%20Linguistic%20Relativity%20Effects%20for%20Grammatical%20and%20Conceptual%20Gender.pdf) | Humans do not automatically transfer grammatical gender of inanimates into conceptual gender during implicit processing. If an LLM shifts pronoun assignment ($he/she$) based solely on a foreign noun's gender, the model exhibits hyper-sensitivity to surface agreement cues compared to humans. |
| **Rebutting reviewers who cite Boroditsky (2003) as settled science** | [Boroditsky et al. (2003)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBoroditsky%20et%20al.%202003%5D%20Sex%2C%20Syntax%2C%20and%20Semantics.pdf); [Mickan et al. (2014)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BMickan%20et%20al.%202014%5D%20Key%20is%20a%20llave%20is%20a%20Schl%C3%BCssel.pdf) | The original Boroditsky et al. (2003) book chapter contains *no sample sizes, no descriptive statistics, no test statistics, and no effect sizes* for the adjective task, and references an unpublished manuscript whose stimulus lists were never deposited. |
| **Controlling for phonological wug / nonce-word endings vs. grammatical gender** | [Cassidy et al. (1999)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BCassidy%20et%20al.%201999%5D%20Inferring%20Gender%20from%20Name%20Phonology.pdf); [Sidhu & Pexman (2015)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSidhu%20&%20Pexman%202015%5D%20Sound%20Symbolism%20and%20Gender%20in%20First%20Names.pdf) | Pseudowords ending in open vowels (/i/, /ə/) or sonorants trigger female gender categorization; consonant codas and voiceless stops trigger male categorization. Wug controls are mandatory to isolate grammar from phonology. |
| **Higher-order semantic dimensions (potency, agency, valence) modulate gender attributions** | [Sidhu et al. (2022)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSidhu%20et%20al.%202022%5D%20Higher-Order%20Factors%20of%20Sound%20Symbolism.pdf); [Mahajan & Durgin (2025)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BMahajan%20&%20Durgin%202025%5D%20Implicit%20Test%20of%20Grammatical%20Gender%20Effects.pdf) | Sound symbolism and gender attributions cluster along fundamental dimensions of Potency, Activity, and Valence. |
| **Inanimate objects differ fundamentally from biological animals in gender malleability** | [Brand et al. (2025)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBrand%20et%20al.%202025%5D%20Constraints%20on%20Linguistic%20Relativity%20Effects%20for%20Grammatical%20and%20Conceptual%20Gender.pdf); [Boroditsky & Schmidt (2000)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBoroditsky%20&%20Schmidt%202000%5D%20Sex%2C%20Syntax%2C%20and%20Semantics.pdf) | Both early (Boroditsky & Schmidt 2000) and recent megastudy data (Brand et al. 2025, $N = 4,621$) confirm that gender transfer occurs for animals (where biological sex exists), but is near zero or strictly artifactual for inanimate objects. |
| **Serial order in adjective generation (why the first word differs from later words)** | [Semenuks et al. (2017)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSemenuks%20et%20al.%202017%5D%20Effects%20of%20Grammatical%20Gender%20on%20Object%20Description.pdf) | The grammatical gender congruency effect on adjective generation is strictly null on the 1st generated word, emerging only on the 2nd and 3rd words as lexical-semantic retrieval exhausts salient physical traits. |
| **English pre-training data contains natural inanimate gender assignment heuristics** | [Pawley (2002)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BPawley%202002%5D%20Using%20He%20and%20She%20for%20Inanimate%20Nouns%20in%20English.pdf) | In colloquial English, assigning *he* or *she* to inanimates (tools, boats, storms) is governed by functional affordance, dynamic agency, and affective involvement, providing a non-grammatical baseline prior for LLMs. |
| **Artistic personifications reflect cultural/artistic lineages, not raw grammar** | [Bender et al. (2016)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBender%20et%20al.%202016%5D%20Lady%20Liberty%20and%20Godfather%20Death.pdf); [Segel & Boroditsky (2011)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BSegel%20&%20Boroditsky%202011%5D%20Grammar%20in%20Art.pdf) | While Segel & Boroditsky observed 78% correspondence in art, Bender et al. demonstrated that this reflects iconographic convention (e.g. classical Latin/Greek models) rather than active grammatical shaping of cognition. |
| **Preregistration and Smallest Effect Size of Interest (SESOI) rationale** | [Elpers et al. (2022)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BElpers%20et%20al.%202022%5D%20Registered%20Replication%20of%20Phillips%20and%20Boroditsky.pdf); [Xiao et al. (2026)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BXiao%20et%20al.%202026%5D%20Grammatical%20Gender%20on%20Object%20Conceptualization%20is%20Weak.pdf) | High-powered replications show that resolving subtle cross-linguistic effects requires $N \approx 1,800$ participants; setting a tight equivalence bound (SESOI = 5 percentage points) prevents declaring trivial noise as meaningful transfer. |

---

## 4. Annotated Inventory of All 19 Papers

Below is the complete, detailed catalog of all 19 papers located in `sources/01_Human_Psycholinguistics_and_Replications/`, organized alphabetically by first author.

```
Directory Index:
01. [Bender et al. 2011] Grammatical Gender in German - An Implicit Association Study.pdf
02. [Bender et al. 2016] Lady Liberty and Godfather Death.pdf
03. [Boroditsky & Schmidt 2000] Sex, Syntax, and Semantics.pdf
04. [Boroditsky 2001] Does Language Shape Thought.pdf
05. [Boroditsky et al. 2003] Sex, Syntax, and Semantics.pdf
06. [Brand et al. 2025] Constraints on Linguistic Relativity Effects for Grammatical and Conceptual Gender.pdf
07. [Cassidy et al. 1999] Inferring Gender from Name Phonology.pdf
08. [Elpers et al. 2022] Registered Replication of Phillips and Boroditsky.pdf
09. [Kurinski & Sera 2011] Spanish Grammatical Gender and Object Categorization.pdf
10. [Mahajan & Durgin 2025] Implicit Test of Grammatical Gender Effects.pdf
11. [Mickan et al. 2014] Key is a llave is a Schlüssel.pdf
12. [Pawley 2002] Using He and She for Inanimate Nouns in English.pdf
13. [Phillips & Boroditsky 2003] Can Quirks of Grammar Affect the Way You Think.pdf
14. [Samuel et al. 2019] Grammatical Gender and Linguistic Relativity.pdf
15. [Segel & Boroditsky 2011] Grammar in Art.pdf
16. [Semenuks et al. 2017] Effects of Grammatical Gender on Object Description.pdf
17. [Sidhu & Pexman 2015] Sound Symbolism and Gender in First Names.pdf
18. [Sidhu et al. 2022] Higher-Order Factors of Sound Symbolism.pdf
19. [Xiao et al. 2026] Grammatical Gender on Object Conceptualization is Weak.pdf
```

---

### 1. Bender, Beller & Klauer (2011)

- **File:** `[Bender et al. 2011] Grammatical Gender in German - An Implicit Association Study.pdf`
- **Full Citation:** Bender, Andrea; Beller, Sieghard; Klauer, Karl Christoph (2011). "Grammatical gender in German: A reason for bias?" *Quarterly Journal of Experimental Psychology*, 64(9), 1821–1835. DOI: [10.1080/17470218.2011.582128](https://doi.org/10.1080/17470218.2011.582128).
- **Paradigm & Methodology:** Speeded reaction-time Implicit Association Test (IAT). Evaluated whether grammatical gender in German automatically activates conceptual gender associations without conscious awareness. Native German speakers sorted grammatically masculine vs. feminine inanimate nouns alongside male vs. female personal attributes across congruent and incongruent blocks.
- **Core Findings:** The IAT revealed no automatic congruency effect ($D$-scores near zero). Grammatical gender did not speed up or prime conceptual gender classifications.
- **Relevance to Project:** Serves as the primary evidence that in humans, grammatical gender is not automatically co-activated during rapid, implicit processing. If an LLM demonstrates pronoun shifts in an unprompted character story, it violates this human cognitive baseline.

---

### 2. Bender, Beller & Klauer (2016)

- **File:** `[Bender et al. 2016] Lady Liberty and Godfather Death.pdf`
- **Full Citation:** Bender, Andrea; Beller, Sieghard; Klauer, Karl Christoph (2016). "Lady Liberty and Godfather Death as candidates for linguistic relativity? Scrutinizing the gender congruency effect on personified allegories with explicit and implicit measures." *Quarterly Journal of Experimental Psychology*, 69(1), 48–64. DOI: [10.1080/17470218.2015.1021701](https://doi.org/10.1080/17470218.2015.1021701).
- **Paradigm & Methodology:** Cross-linguistic explicit picture-similarity sorting and personification ratings across German, Spanish, and English native speakers, testing concepts personified in cultural traditions (Death, Liberty, Love, Time).
- **Core Findings:** Cultural traditions and artistic conventions strongly shape gendered personifications (e.g., German *Freund Hein* vs. Romance *la muerte*). Once cultural exposure and iconographic history were controlled, grammatical gender accounted for minimal unique variance.
- **Relevance to Project:** Warns that when LLMs generate personified characters, correlations with gender may reflect training-corpus exposure to Western art and allegorical literary tropes rather than grammatical transfer.

---

### 3. Boroditsky & Schmidt (2000)

- **File:** `[Boroditsky & Schmidt 2000] Sex, Syntax, and Semantics.pdf`
- **Full Citation:** Boroditsky, Lera; Schmidt, Lauren A. (2000). "Sex, Syntax, and Semantics." In *Proceedings of the 22nd Annual Meeting of the Cognitive Science Society*, 22, 41–46.
- **Paradigm & Methodology:** Two studies. Study 1: English speakers rated perceived masculinity/femininity of animals and artifacts; correlated with German/Spanish grammatical genders. Study 2: Paired-associate memory test conducted in English with Spanish and German native speakers, memorizing proper name–object pairs (e.g., apple–Patricia vs. apple–Patrick).
- **Core Findings:** In Study 1, English intuitions correlated with Spanish/German grammatical genders for *animals*, but not for artifacts. In Study 2, memory was superior for gender-congruent name–object pairs.
- **Relevance to Project:** The genesis of the paired-associate memory paradigm and the first empirical acknowledgment that animal nouns behave fundamentally differently from inanimate artifact nouns.

---

### 4. Boroditsky (2001)

- **File:** `[Boroditsky 2001] Does Language Shape Thought.pdf`
- **Full Citation:** Boroditsky, Lera (2001). "Does Language Shape Thought?: Mandarin and English Speakers’ Conceptions of Time." *Cognitive Psychology*, 43(1), 1–22. DOI: [10.1006/cogp.2001.0748](https://doi.org/10.1006/cogp.2001.0748).
- **Paradigm & Methodology:** Spatio-temporal metaphor priming. Tested native Mandarin vs. English speakers on spatial primes (horizontal vs. vertical arrays of objects) before answering temporal target questions (e.g., "Does March come earlier than April?"). Also tested bilinguals and trained English native speakers on Mandarin-like vertical spatial metaphors.
- **Core Findings:** Mandarin speakers were faster after vertical spatial primes; English speakers after horizontal primes. English speakers trained on vertical metaphors exhibited Mandarin-like priming.
- **Relevance to Project:** Foundational conceptual paper for Neo-Whorfian experimental cognitive science. Establishes the cross-lingual priming logic and artificial category training paradigms later adapted to grammatical gender.

---

### 5. Boroditsky, Schmidt & Phillips (2003)

- **File:** `[Boroditsky et al. 2003] Sex, Syntax, and Semantics.pdf`
- **Full Citation:** Boroditsky, Lera; Schmidt, Lauren A.; Phillips, Webb (2003). "Sex, Syntax, and Semantics." In D. Gentner & S. Goldin-Meadow (Eds.), *Language in Mind: Advances in the Study of Language and Thought*, pp. 61–79. Cambridge, MA: MIT Press.
- **Paradigm & Methodology:** Highly cited book chapter synthesizing multiple experiments: (1) Object-proper name memory; (2) Picture similarity ratings; (3) Cartoon voice assignment (participants assigned male or female voices to objects in an animated movie); (4) The famous adjective generation task: 24 inanimate objects with opposite genders in German and Spanish (e.g., bridge, key); participants wrote the first 3 adjectives that came to mind in English; (5) Gumbuzi artificial language learning.
- **Core Findings:** Reports that German and Spanish speakers described objects with adjectives matching grammatical gender (German keys: *hard, heavy, jagged*; Spanish keys: *golden, intricate, lovely*).
- **Relevance to Project:** The seminal origin paper for the adjective-generation and personification paradigms. 
- **Critical Limitation:** The chapter provides **no sample sizes, no descriptive statistics, and no test statistics** for the adjective task, citing an unpublished manuscript (`Boroditsky, Schmidt & Phillips, 2002, submitted`). The stimulus list was never archived, forcing all subsequent replications to construct custom lists.

---

### 6. Brand, Preininger, Kříž & Ceháková (2025)

- **File:** `[Brand et al. 2025] Constraints on Linguistic Relativity Effects for Grammatical and Conceptual Gender.pdf`
- **Full Citation:** Brand, James; Preininger, Mikuláš; Kříž, Adam; Ceháková, Markéta (2025). "Feminine fox, not so feminine box: constraints on linguistic relativity effects for grammatical and conceptual gender." *Language and Cognition*, 17, e34, 1–26. DOI: [10.1017/langcog.2025.3](https://doi.org/10.1017/langcog.2025.3).
- **Paradigm & Methodology:** Megastudy approach with $N = 4,621$ participants across three language groups: L1 Czech (gendered), L1 English (ungendered), and L1 Czech tested in L2 English. Tested $1,208$ items across three distinct semantic categories: people, animals, and inanimates, combining explicit rating tasks with implicit speeded tasks.
- **Core Findings:** Reliable grammatical gender congruency effects outside of humans occur *exclusively* for animals. For inanimate objects ("box"), effects are tiny and vanish completely in implicit tasks.
- **Relevance to Project:** Directly motivates restricting or controlling for animacy in LLM stimulus design; confirms that inanimates represent the hardest test case for linguistic relativity.

---

### 7. Cassidy, Kelly & Sharoni (1999)

- **File:** `[Cassidy et al. 1999] Inferring Gender from Name Phonology.pdf`
- **Full Citation:** Cassidy, Kimberly Wright; Kelly, Michael H.; Sharoni, Lee'at J. (1999). "Inferring gender from name phonology." *Journal of Experimental Psychology: General*, 128(3), 362–381. DOI: [10.1037/0096-3445.128.3.362](https://doi.org/10.1037/0096-3445.128.3.362).
- **Paradigm & Methodology:** Seven experiments examining how English first-name phonological features (syllable count, stress pattern, vowel vs. consonant endings, voicing) guide gender categorization of real unfamiliar names and invented pseudowords (wugs).
- **Core Findings:** Phonological features reliably predict gender: female names/pseudowords are predominantly multisyllabic, with later stress and open vowel endings (especially /i/ and /ə/), whereas male names are monosyllabic or initially stressed, ending in closed obstruents. Listeners rapidly and unconsciously project these phonological rules onto novel nonce words.
- **Relevance to Project:** Foundational baseline for the study's wug/nonce-modifier conditions. Demonstrates that any character prompt using foreign or invented words carries a strong surface phonological gender prior that must be controlled.

---

### 8. Elpers, Jensen & Holmes (2022)

- **File:** `[Elpers et al. 2022] Registered Replication of Phillips and Boroditsky.pdf`
- **Full Citation:** Elpers, Nan; Jensen, Greg; Holmes, Kevin J. (2022). "Does grammatical gender affect object concepts? Registered replication of Phillips and Boroditsky (2003)." *Journal of Memory and Language*, 127, 104357. DOI: [10.1016/j.jml.2022.104357](https://doi.org/10.1016/j.jml.2022.104357).
- **Paradigm & Methodology:** Preregistered registered report replicating [Phillips & Boroditsky (2003)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BPhillips%20&%20Boroditsky%202003%5D%20Can%20Quirks%20of%20Grammar%20Affect%20the%20Way%20You%20Think.pdf). Experiment 1 tested picture similarity ratings in native German ($n=125$), Spanish ($n=125$), and English ($n=125$) speakers on the original 24 items. Experiment 2 trained English speakers on artificial linguistic categories (Oush vs. Soup).
- **Core Findings:** Experiment 1 failed to replicate the original effect: similarity ratings did not differ by grammatical gender ($\chi^2(1) = 2.00, p = .16; BF_{01} = 47.6$). Experiment 2 successfully replicated category learning ($\chi^2(1) = 11.80, p = .001; BF_{10} = 5.01 \times 10^{10}$).
- **Relevance to Project:** The highest-credibility registered replication in the human psycholinguistic literature. Experiment 2 demonstrates that artificial statistical learning can induce category alignment even when lifelong natural language exposure fails to alter similarity judgments.

---

### 9. Kurinski & Sera (2011)

- **File:** `[Kurinski & Sera 2011] Spanish Grammatical Gender and Object Categorization.pdf`
- **Full Citation:** Kurinski, Elena; Sera, Maria D. (2011). "Does learning Spanish grammatical gender change English-speaking adults’ categorization of inanimate objects?" *Bilingualism: Language and Cognition*, 14(2), 203–220. DOI: [10.1017/S1366728910000179](https://doi.org/10.1017/S1366728910000179).
- **Paradigm & Methodology:** Longitudinal study tracking English-speaking college students across an academic year of introductory Spanish instruction. Administered grammatical gender acquisition tests and object gender attribution tasks (assigning male/female voices to pictures) across 4 testing intervals.
- **Core Findings:** Changes in object categorization emerged after 10 weeks of instruction. However, effects were heavily constrained by whether the object categories aligned with pre-existing conceptual biases (e.g., natural vs. artifactual distinctions).
- **Relevance to Project:** Demonstrates that L2 grammatical gender acquisition can modulate conceptual attribution over short longitudinal exposure, but remains constrained by conceptual priors.

---

### 10. Mahajan & Durgin (2025)

- **File:** `[Mahajan & Durgin 2025] Implicit Test of Grammatical Gender Effects.pdf`
- **Full Citation:** Mahajan, Devyani; Durgin, Frank H. (2025). "A conceptual replication of an implicit test of grammatical gender effects on inanimate concepts." *Language and Cognition*, 17, e28, 1–19. DOI: [10.1017/langcog.2024.76](https://doi.org/10.1017/langcog.2024.76).
- **Paradigm & Methodology:** Two preregistered conceptual replications of Konishi (1993). Konishi used Osgood's semantic-differential "potency" scale (e.g., strong/weak, hard/soft) as an implicit proxy for masculinity. Experiment 1 tested 240 German/Spanish bilinguals and 120 English monolinguals. Experiment 2 tested 180 bilinguals.
- **Core Findings:** Both experiments yielded strong evidence for the null hypothesis: Bayes odds against grammatical gender effects were $24.5:1$ in Experiment 1 and $8.0:1$ in Experiment 2. Grammatical gender does not modulate perceived semantic potency of inanimate concepts.
- **Relevance to Project:** High-powered refutation of the semantic-differential approach; proves that inanimate objects do not inherit masculine potency or feminine delicacy through grammatical agreement.

---

### 11. Mickan, Schiefke & Stefanowitsch (2014)

- **File:** `[Mickan et al. 2014] Key is a llave is a Schlüssel.pdf`
- **Full Citation:** Mickan, Anne; Schiefke, Maren; Stefanowitsch, Anatol (2014). "Key is a llave is a Schlüssel: A failure to replicate an experiment from Boroditsky et al. 2003." *Yearbook of the German Cognitive Linguistics Association*, 2, 39–50. DOI: [10.1515/gcla-2014-0004](https://doi.org/10.1515/gcla-2014-0004).
- **Paradigm & Methodology:** Experiment 1: Direct replication of [Boroditsky et al. (2003)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BBoroditsky%20et%20al.%202003%5D%20Sex%2C%20Syntax%2C%20and%20Semantics.pdf) adjective generation with 15 German and 15 Spanish native speakers writing 3 adjectives for 10 opposite-gender nouns in their native language; adjectives rated on a 0–20 gender scale by independent raters. Experiment 2: Primed lexical decision task using line drawings of 16 objects as primes and 16 stereotyped adjectives as targets.
- **Core Findings:** Both experiments failed to find significant effects. Adjective generation differences were non-significant (by-subject $p = .085$; by-item $p = .457$), and primary response values reversed for Spanish. Primed lexical decision showed no RT facilitation for gender-congruent pairs.
- **Relevance to Project:** The first published direct failure to replicate the famous adjective generation finding, highlighting small item sets and adjective denotation confounds.

---

### 12. Pawley (2002)

- **File:** `[Pawley 2002] Using He and She for Inanimate Nouns in English.pdf`
- **Full Citation:** Pawley, Andrew (2002). "Using *he* and *she* for inanimate nouns in English: Questions of grammar and worldview." In N. J. Enfield (Ed.), *Ethnosyntax: Explorations in Language and Culture*, pp. 110–137. Oxford: Oxford University Press.
- **Paradigm & Methodology:** Anthropological linguistics and dialectological analysis of non-standard pronominal gender in vernacular Australian English, Newfoundland English, and regional dialects of Southwest England.
- **Core Findings:** Documented systematic rules for assigning *he* and *she* to inanimate referents in English dialects: *she* is assigned to vehicles, engines, machinery, boats, and dynamic weather systems; *he* is assigned to tools, individual handheld implements, and discrete manipulable objects. Pronominalization reflects cultural schemas of affordance, dynamic agency, and personal involvement.
- **Relevance to Project:** Provides the crucial baseline for LLM story generation: models generating *he* or *she* for an inanimate object in English may be drawing on dialectal and affective personification tropes rather than cross-lingual grammatical bleedthrough.

---

### 13. Phillips & Boroditsky (2003)

- **File:** `[Phillips & Boroditsky 2003] Can Quirks of Grammar Affect the Way You Think.pdf`
- **Full Citation:** Phillips, Webb; Boroditsky, Lera (2003). "Can Quirks of Grammar Affect the Way You Think? Grammatical Gender and Object Concepts." In *Proceedings of the 25th Annual Meeting of the Cognitive Science Society*, 25, 928–933.
- **Paradigm & Methodology:** Five experiments: Exp 1: Picture similarity between inanimate objects and pictures of men/women in Spanish and German speakers. Exp 2: Spanish–German bilinguals. Exp 3: Picture similarity under verbal interference (shadowing words). Exp 4: Artificial language learning (Gumbuzi) with gender-like categories (*soup* vs. *oush*). Exp 5: Gumbuzi similarity under verbal interference.
- **Core Findings:** Reported that Spanish and German speakers rated objects as more similar to people sharing the object's grammatical gender, even under verbal interference. Gumbuzi training produced similar shifts.
- **Relevance to Project:** The formal peer-reviewed presentation of the picture-similarity paradigm later refuted by [Elpers et al. (2022)](file:///Users/watcher/githere/gender/sources/01_Human_Psycholinguistics_and_Replications/%5BElpers%20et%20al.%202022%5D%20Registered%20Replication%20of%20Phillips%20and%20Boroditsky.pdf). Note: Does *not* contain the adjective generation task.

---

### 14. Samuel, Cole & Eacott (2019)

- **File:** `[Samuel et al. 2019] Grammatical Gender and Linguistic Relativity.pdf`
- **Full Citation:** Samuel, Steven; Cole, Geoff; Eacott, Madeline J. (2019). "Grammatical gender and linguistic relativity: A systematic review." *Psychonomic Bulletin & Review*, 26(6), 1767–1786. DOI: [10.3758/s13423-019-01652-3](https://doi.org/10.3758/s13423-019-01652-3).
- **Paradigm & Methodology:** Exhaustive systematic review evaluating 43 published empirical studies (testing 5,895 participants) on grammatical gender and linguistic relativity. Categorized studies by task salience, verbal interference, language of testing, and whether the task was implicit or explicit.
- **Core Findings:** Overall support for linguistic relativity across all studies is 32% positive, 24% mixed, 43% null. However, support concentrates heavily in tasks where gender is explicitly salient. Excluding high-salience tasks drops positive support to 11% and increases null results to 64%.
- **Relevance to Project:** The single most authoritative meta-scientific assessment in the literature. Serves as our primary citation for framing why implicit testing (e.g. asking an LLM to write a story without mentioning gender) represents the critical, rigorous test case.

---

### 15. Segel & Boroditsky (2011)

- **File:** `[Segel & Boroditsky 2011] Grammar in Art.pdf`
- **Full Citation:** Segel, Edward; Boroditsky, Lera (2011). "Grammar in art." *Frontiers in Psychology*, 1, 244. DOI: [10.3389/fpsyg.2010.00244](https://doi.org/10.3389/fpsyg.2010.00244).
- **Paradigm & Methodology:** Corpus analysis of Western art history using the ARTstor digital database. Analyzed 765 allegorical personifications across 616 works produced by artists whose native languages had grammatical gender (German, French, Italian, Spanish).
- **Core Findings:** Depicted sex corresponded with grammatical gender in 78% of personifications overall. However, correspondence varied widely by language: Italian 82%, French 80%, German 62%, and Spanish 54% (statistically indistinguishable from chance, though based on only $N=24$ items).
- **Relevance to Project:** Directly links grammatical gender to character personification. Demonstrates strong language-level asymmetries and highlights the danger of underpowered language cells (Spanish).

---

### 16. Semenuks, Philips, Dalca, Kim & Boroditsky (2017)

- **File:** `[Semenuks et al. 2017] Effects of Grammatical Gender on Object Description.pdf`
- **Full Citation:** Semenuks, Arturs; Philips, Webb; Dalca, Ioana; Kim, Cora; Boroditsky, Lera (2017). "Effects of Grammatical Gender on Object Description." In *Proceedings of the 39th Annual Meeting of the Cognitive Science Society*, 39, 1072–1077.
- **Paradigm & Methodology:** Adjective generation task conducted in English with native speakers of French, German, and Romanian ($N = 101$). Generated adjectives were subsequently rated for gender valence. Analyzed whether grammatical gender effects differed across serial output position (1st vs. 2nd vs. 3rd adjective) and across semantic domains (concrete objects, body parts, abstract concepts).
- **Core Findings:** Grammatical gender significantly affected adjective valence only starting with the 2nd and 3rd adjectives generated; the effect was strictly null for the 1st adjective. Furthermore, effects emerged for abstract and social concepts, but were null for concrete inanimate objects and body parts.
- **Relevance to Project:** Demonstrates that the initial retrieval step in adjective generation is dominated by salience and denotative affordance; grammatical gender effects require cognitive elaboration or exhaustion of salient physical attributes.

---

### 17. Sidhu & Pexman (2015)

- **File:** `[Sidhu & Pexman 2015] Sound Symbolism and Gender in First Names.pdf`
- **Full Citation:** Sidhu, David M.; Pexman, Penny M. (2015). "What’s in a Name? Sound Symbolism and Gender in First Names." *PLoS ONE*, 10(5), e0126809. DOI: [10.1371/journal.pone.0126809](https://doi.org/10.1371/journal.pone.0126809).
- **Paradigm & Methodology:** Five studies testing the Bouba/Kiki sound symbolism effect on real first names and personality traits. Experiments 1a/1b paired first names with round vs. sharp silhouette character shapes; category norms evaluated the proportion of round phonemes across male and female names; Experiment 2 paired names with figurative personality descriptions.
- **Core Findings:** Female names contain a significantly higher proportion of round phonemes (sonorants, rounded vowels), which map to round character silhouettes and "round" personality traits (easygoing, gentle). Male names contain more sharp phonemes (voiceless stops), mapping to spiky shapes and assertive personalities.
- **Relevance to Project:** Establishes the sound-symbolic baseline for character creation. When an LLM generates a persona, the phonology of the stimulus noun acts as an independent driver of gendered personality traits.

---

### 18. Sidhu, Vigliocco & Pexman (2022)

- **File:** `[Sidhu et al. 2022] Higher-Order Factors of Sound Symbolism.pdf`
- **Full Citation:** Sidhu, David M.; Vigliocco, Gabriella; Pexman, Penny M. (2022). "Higher order factors of sound symbolism." *Journal of Memory and Language*, 125, 104323. DOI: [10.1016/j.jml.2022.104323](https://doi.org/10.1016/j.jml.2022.104323).
- **Paradigm & Methodology:** Measured 25 different sound-symbolic associations across phonemes and conducted factor analyses to identify latent organizing dimensions. Validated the resulting factor structure in shape-matching experiments.
- **Core Findings:** Phonemic sound-symbolic associations cluster into four higher-order dimensions: **Activity**, **Valence**, **Potency**, and **Novelty**. Specific phonemes systematically associate with these dimensions (e.g., voiced stops and sonorants with lower potency and higher warmth).
- **Relevance to Project:** Connects sound symbolism directly to Osgood's semantic differential dimensions, providing the theoretical vocabulary for decomposing character descriptions into latent affective axes.

---

### 19. Xiao, Cremers, Straboni, Strickland & Peperkamp (2026)

- **File:** `[Xiao et al. 2026] Grammatical Gender on Object Conceptualization is Weak.pdf`
- **Full Citation:** Xiao, Hualin; Cremers, Alexandre; Straboni, Camille; Strickland, Brent; Peperkamp, Sharon (2026). "The Influence of Grammatical Gender on Object Conceptualization Is Weak and Language-Dependent." *Cognitive Science*, 50(2), e70176. DOI: [10.1111/cogs.70176](https://doi.org/10.1111/cogs.70176).
- **Paradigm & Methodology:** Three high-powered preregistered experiments testing adjective generation and subsequent gender rating. Exp 1: Native French speakers ($N = 100$) tested on within-language opposite-gender nouns. Exp 2: Native French and German speakers tested on cross-language translation equivalents presented with determiners. Exp 3: Native German speakers tested on bare nouns without determiners.
- **Core Findings:** French exhibited a complete null effect ($\beta = 0.024, 95\%\text{ CrI } [-0.236, 0.275]$). German showed a weak effect ($\beta = 0.23, P(\beta > 0) = 0.93$), which persisted without determiners ($\beta = 0.294$), but whose credible intervals crossed zero ($95\%\text{ CrI } [-0.09, 0.684]$) and whose Bayes factors were inconclusive ($BF_{01} = 0.844$ and $0.551$). Natural vs. artifact object category had an effect twice as large ($\beta = -0.500$).
- **Relevance to Project:** The closest, most rigorous modern published benchmark for adjective generation. Crucially demonstrates that the German effect survives removal of the determiner, but cautions that human effect sizes for inanimates are statistically fragile and heavily dwarfed by intrinsic semantic features.
