# Decisions log

Choices that shape the results, with the reason, in the order they were made. Newest last.
Useful for the write-up's methods section.

| date | decision | why | alternatives considered |
|---|---|---|---|
| 2026-10-05 | Gold genders from Wiktionary (kaikki.org), never from an LLM | PI doesn't read German; labels must be independently sourced | hand-written lists (kept only as an unverified seed) |
| 2026-10-05 | Stimulus filters exclude anything that would need a manual check (animacy uncertain, homographs, regional/marked senses, given names, English-frequent words) | prefer smaller clean lists to large lists needing review | manual review of borderline items |
| 2026-10-05 | Flipped pairs default to **strict + concrete** auto-matches; 19 **classics** hand-picked from Kann 2019 / Mickan et al. 2014 / design doc | auto-matching on English glosses is ~10% wrong even when strict; Boroditsky-style items are concrete objects | review all 163 strict pairs by hand |
| 2026-10-05 | Phase 0 pass rule = **sentence scoring in two frames, both ≥ 1 nat**; few-shot quiz diagnostic only | quiz was order-sensitive and failed ~20% of easy German nouns on Qwen3-0.6B | both measures must pass; quiz only |
| 2026-10-05 | Frame 2 = indefinite article (*Hier ist ein/eine X*, *Aquí hay un/una X*), fixed on grammar and validated on two model families | removes plural readings of *die*; avoids tuning frames to one model | best-scoring frame on Qwen |
| 2026-10-06 | Margins condition on the article (score noun + rest given the article) | *ein*/*dem* also serve neuter nouns, so the article's prior biased toward masculine | full-sentence probability |
| 2026-10-06 | Cross-model comparisons use the **shared set** (nouns known by every model compared); per-model sets as a check | differences between sizes can't come from different items | each model's own known nouns |
| 2026-10-06 | (tentative, pending Qwen3-1.7B/4B Phase 0) Model family = **Qwen3 base**, sizes 0.6B–14B; final runs of every phase on RunPod, one GPU type, bf16 | one tokenizer across sizes; TransformerLens support; ungated; Mac for development only | EuroLLM, SmolLM3, Gemma 3, Llama 3.2 |
