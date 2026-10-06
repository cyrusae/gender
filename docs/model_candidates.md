# Model candidates: grammatical gender in German and Spanish

Compiled 2026-10-05. Scope: small open-weight models that run in bf16/fp16 on a 16 GB Apple Silicon Mac (PyTorch MPS), plus a few 7-9B models for later GPU runs. Each one needs to work for (1) a next-token check of der/die/das and el/la, and (2) residual-stream extraction at every layer for probes, LEACE and steering.

Conventions:
- **Params** are the safetensors totals from the HF API (`https://huggingface.co/api/models/<id>`). "bf16 mem" = params x 2 bytes for weights only. Activations for a probe dataset are small next to the weights, but leave 2-4 GB headroom.
- **TL** = TransformerLens. TL 4.0.0 (released 2026-09-21) removed `HookedTransformer`. Every model now loads through `TransformerBridge.boot_transformers(...)`, which wraps the native HF implementation ([PyPI](https://pypi.org/project/transformer-lens/), [v3 migration guide](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v3.html)). The status below comes from the project's registry file [`supported_models.json`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/tools/model_registry/data/supported_models.json) (generated 2026-07-28). Its codes are 0=unverified, 1=verified, 2=skipped, 3=failed, 4=provisional ([schemas.py](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/tools/model_registry/schemas.py)). "Unverified" means the architecture has an adapter but nobody has run end-to-end checks on that checkpoint. For a standard `LlamaForCausalLM` it will very likely work.
- **nnsight** wraps any HF `AutoModelForCausalLM`, so every model below should work with it given a recent enough `transformers`. I have not tested any of them on MPS.
- **EuroEval** numbers come from the EuroEval leaderboard CSVs ([german_nlu.csv](https://github.com/EuroEval/leaderboards/blob/main/leaderboards/german_nlu.csv), [spanish_nlu.csv](https://github.com/EuroEval/leaderboards/blob/main/leaderboards/spanish_nlu.csv); last commit 2026-04-20). "Rank" is EuroEval's mean rank score, where lower is better. "ScaLA" is the linguistic-acceptability task (MCC): a few-shot grammatical / ungrammatical judgement. It is the closest public proxy for grammaticality, but it is **not** a gender-agreement test. Base models often score about 0 on ScaLA because of the few-shot output format, so read base-model ScaLA numbers with care.

---

## 1. Summary table

Small models (Mac, bf16):

| Model | Base HF ID | Params | bf16 mem | License / gated | DE/ES in training data | TL status | German evidence | Released |
|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B | `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base` | 0.60B / 1.72B / 4.02B | 1.2 / 3.4 / 8.0 GB | Apache-2.0, not gated | 119 languages, ~36T tokens; no per-language share published | **Verified** (all three Base) | EuroEval DE rank (instruct ckpts): 4B 2.10, 1.7B 2.45, 0.6B 3.21 | 2025-04-29 |
| Qwen3.5 0.8B / 2B / 4B (new 2026) | `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-2B-Base`, `Qwen/Qwen3.5-4B-Base` | 0.87B / 2.27B / 4.66B (incl. vision encoder) | 1.7 / 4.5 / 9.3 GB | Apache-2.0, not gated | "201 languages and dialects"; no per-language share | **Verified** (as `Qwen3_5ForConditionalGeneration`) | EuroEval DE rank (base): 4B 2.98, 2B 3.17, 0.8B 3.43; GermanQuAD 69.0 / 65.7 / 56.8 | ~2026-02/03 |
| Gemma 3 270M / 1B / 4B | `google/gemma-3-270m`, `google/gemma-3-1b-pt`, `google/gemma-3-4b-pt` | 0.27B / 1.00B / 4.30B (4B incl. SigLIP) | 0.5 / 2.0 / 8.6 GB | Gemma license, **gated** (accept terms) | **1B is English-only**; 4B+ "140+ languages" | **Verified** (270m, 1b-pt, 4b-pt) | EuroEval DE rank: 4b-pt 2.31, 1b-pt 3.43 | 2025-03-12 (270m: 2025-08) |
| Gemma 4 E2B / E4B (new 2026) | `google/gemma-4-E2B`, `google/gemma-4-E4B` | 5.12B / 8.00B total (2.3B / 4.5B "effective") | 10.2 / 16.0 GB | **Apache-2.0**, not gated | "pre-trained on 140+ languages" | **Verified** (E2B, E4B) | EuroEval DE rank (base): E2B 3.89, E4B 3.56; -it much better (2.26 / 1.92) | 2026-04-02 |
| Llama 3.2 1B / 3B | `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B` | 1.24B / 3.21B | 2.5 / 6.4 GB | Llama 3.2 Community, **gated** | 8 official languages incl. **de, es**; up to 9T tokens | **Verified** | Card MMLU-de (base): 1B 39.2, 3B 53.3; MMLU-es 41.5 / 55.1 | 2024-09-25 |
| EuroLLM 1.7B | `utter-project/EuroLLM-1.7B` | ~1.7B | ~3.4 GB | Apache-2.0, not gated | 35 languages; English 50% in phase 1, 32.5% in annealing; rest by data availability | Unverified (Llama arch) | Card: HellaSwag-de 0.606, ARC-de 0.346; EuroEval ES rank 3.40 | 2024-09-24 |
| SmolLM3 3B | `HuggingFaceTB/SmolLM3-3B-Base` | 3.08B | 6.2 GB | Apache-2.0, not gated | 6 native languages incl. **de, es**; ~12% multilingual web data | **Verified** | Card (base): Global-MMLU-de 35.1, Belebele-de 48.4, HellaSwag-de 59.6; EuroEval DE rank 2.39 | 2025-07-08 |
| Salamandra 2B | `BSC-LT/salamandra-2b` | 2.25B | 4.5 GB | Apache-2.0, not gated | 35 EU languages; **es 16.1%**, de 4.8%, en 39.3% | Unverified (Llama arch) | Spanish evals only on card; no German numbers | 2024-10 (v1.1 branch later) |
| Apertus v1.1 0.5B / 1.5B / 4B (new 2026) | `swiss-ai/Apertus-v1.1-0.5B`, `swiss-ai/Apertus-v1.1-1.5B`, `swiss-ai/Apertus-v1.1-4B` | 0.44B / 1.51B / 3.83B | 0.9 / 3.0 / 7.7 GB | Apache-2.0, not gated | Distilled from Apertus-8B (1811 languages, ~40% non-English) | **Verified** (all three) | No German-specific numbers found | 2026-02 to 2026-04 |
| Ministral 3 3B | `mistralai/Ministral-3-3B-Base-2512` | 3.85B (3.4B LM + 0.4B vision) | 7.7 GB | Apache-2.0, not gated | "dozens of languages" incl. de, es; no shares | Unverified (`Mistral3ForConditionalGeneration`) | **Best small model in EuroEval DE**: rank 1.83, ScaLA-de 49.1, GermanQuAD 67.0 | 2025-12 |
| Granite 4.1 3B | `ibm-granite/granite-4.1-3b-base` | 3.40B | 6.8 GB | Apache-2.0, not gated | 12 languages incl. de, es; ~15T tokens | **Verified** | MMMLU 56.6 (aggregate); no per-language numbers found | 2026-04-29 |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B` | 1.48B | 3.0 GB | Apache-2.0, not gated | English-centric (no multilingual target) | **Verified** | Not on EuroEval; expect weak German | 2025-04 |
| Phi-4-mini | (no base released) `microsoft/Phi-4-mini-instruct` | 3.84B | 7.7 GB | MIT, not gated | 23 languages incl. de, es | Not in registry (only `Phi-4-mini-reasoning` verified) | Card MMMLU 49.3 (instruct) | 2025-02 |

Larger models (for a rented GPU later):

| Model | Base HF ID | Params | bf16 mem | License / gated | DE/ES data | TL status | German evidence |
|---|---|---|---|---|---|---|---|
| Qwen3 8B | `Qwen/Qwen3-8B-Base` | 8.19B | 16.4 GB | Apache-2.0 | 119 languages | Verified (phase 1 only) | — |
| Qwen3.5 9B | `Qwen/Qwen3.5-9B-Base` | 9.65B | 19.3 GB | Apache-2.0 | 201 languages | Verified | EuroEval DE rank 2.66, GermanQuAD 72.7 |
| Ministral 3 8B | `mistralai/Ministral-3-8B-Base-2512` | 8.92B | 17.8 GB | Apache-2.0 | dozens incl. de/es | Unverified | EuroEval DE rank 1.65, ScaLA-de 53.1 (best open base in this list) |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | 8.03B | 16.1 GB | Llama 3.1, **gated** | 8 languages incl. de/es | Verified | EuroEval ScaLA-de 37.4 (partial) |
| EuroLLM 9B | `utter-project/EuroLLM-9B-2512` (or original `utter-project/EuroLLM-9B`, gated=auto) | 9.15B | 18.3 GB (2512 repo stored in **F32**, so 36.6 GB to download) | Apache-2.0 | 35 languages | Unverified (Llama) | EuroEval DE rank 2.16 (original 9B) |
| EuroLLM 22B | `utter-project/EuroLLM-22B-2512` | 22.6B | 45 GB | Apache-2.0 | 35 languages, >4T tokens | Unverified (Llama) | see [tech report](https://arxiv.org/abs/2602.05879) |
| Salamandra 7B | `BSC-LT/salamandra-7b` | 7.77B | 15.5 GB | Apache-2.0 | es 16%, de ~5% (same mix as 2B, unverified for 7B) | Unverified | EuroEval DE rank 3.74 (poor) |
| Teuken 7B | `openGPT-X/Teuken-7B-base-v0.6` | 7.45B | 14.9 GB | **CC-BY-NC-4.0**, not gated | 24 EU languages, 6T tokens; non-English share not on card | Unverified | — |
| Apertus 8B | `swiss-ai/Apertus-8B-2509` | 8.05B | 16.1 GB | Apache-2.0 | 1811 languages, ~40% non-English | Verified | EuroEval DE rank 2.51 |
| Apertus v1.5 8B (new, Sept 2026) | `swiss-ai/Apertus-v1.5-8B-base` | 8.84B (multimodal) | 17.7 GB | Apache-2.0 | multilingual | Not in registry (new `Apertus1p5ForConditionalGeneration` arch) | none yet |
| OLMo 3 7B | `allenai/Olmo-3-1025-7B` | 7.30B | 14.6 GB | Apache-2.0 | English-centric | Verified | EuroEval DE rank 2.24 but ScaLA-de 26.1 |
| Gemma 3 12B / Gemma 4 12B | `google/gemma-3-12b-pt` / `google/gemma-4-12B` | 12B | 24 GB | Gemma (gated) / Apache-2.0 | 140+ | Verified (phase 1) / skipped | — |

---

## 2. Per-model notes

### Qwen3 Base (0.6B, 1.7B, 4B; 8B for GPU)
- IDs: `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3-8B-Base`. Apache-2.0, not gated (HF API).
- Data: the technical report says Qwen3 expands multilingual support "from 29 to 119 languages and dialects" ([arXiv 2505.09388](https://arxiv.org/abs/2505.09388)). The often-quoted ~36T tokens is from the Qwen3 release blog; I did not re-check it in this pass. No per-language share is published.
- TL: all three small Base checkpoints are **verified**, with phases 1-3 at 100.
- German evidence: EuroEval German has only the hybrid/instruct `Qwen/Qwen3-4B` (rank 2.10, ScaLA-de 41.5, the best ScaLA among small non-Mistral models) and `Qwen/Qwen3-1.7B` (2.45). In Spanish, `Qwen3-4B` is the top small model in the table (rank 2.12). There is no EuroEval entry for the Base checkpoints.
- Tokenizer: ~151k BPE (byte-level). Check that " der", " die", " das", " el", " la" are single tokens with a leading space. They almost certainly are, but confirm.
- Main appeal: one family with clean scaling (0.6 / 1.7 / 4 / 8B), the same tokenizer at every size, verified TL support, no gating.

### Qwen3.5 Base (0.8B, 2B, 4B, 9B), new in 2026
- IDs: `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-2B-Base`, `Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-9B-Base`. Apache-2.0, not gated. HF repos were created 2026-02-26 to 02-28, and the model card says February 2026 ([card](https://huggingface.co/Qwen/Qwen3.5-2B-Base)).
- **Architecture caveat:** it is a hybrid. The 2B, for example, has 24 layers laid out as `6 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`, and it is a unified vision-language model (`Qwen3_5ForConditionalGeneration`) with a vision encoder. Vocab is 248,320 and the card claims "201 languages and dialects" ([card](https://huggingface.co/Qwen/Qwen3.5-2B-Base)). The residual stream works normally for probes and steering. But only 1 in 4 layers is softmax attention, which complicates attention-head analyses. Third-party "text-only" conversions exist (e.g. `principled-intelligence/Qwen3.5-2B-text-only`); these are unverified.
- TL: **verified** (0.8B/2B/4B/9B Base; phase 1 = 100, phase 2 = 92.3, phase 3 = 94.7, so slightly below perfect agreement with HF).
- German evidence (EuroEval, base): GermanQuAD 56.8 / 65.7 / 69.0 / 72.7 for 0.8B / 2B / 4B / 9B. ScaLA-de is 0 for all the base checkpoints (few-shot format issue). The instruct `Qwen3.5-4B` gets ScaLA-de 49.5 and ScaLA-es 40.1, the highest ScaLA-es of any model in this list.
- Worth adding as a "new architecture" comparison against Qwen3, but not first in line.

### Gemma 3 (270M, 1B, 4B pt)
- IDs: `google/gemma-3-270m`, `google/gemma-3-1b-pt`, `google/gemma-3-4b-pt`. Gemma license, **gated (manual accept)** per HF API.
- **The 1B is English-only.** The HF launch blog lists the 1B's languages as "English" and the 4B/12B/27B as "+140 languages" ([HF blog](https://huggingface.co/blog/gemma3)). Training tokens: 2T for 1B and 4T for 4B. This figure appears in secondary summaries of the Gemma 3 report; I did not re-verify it against the report. So the 1B is only useful as an English-centric control. Its EuroEval DE rank of 3.43 is consistent with that.
- `gemma-3-4b-pt` is `Gemma3ForConditionalGeneration` (it includes the SigLIP vision tower). The text LM is ~3.9B of the 4.3B. EuroEval DE rank 2.31, GermanQuAD 64.0.
- Tokenizer: SentencePiece, 262k vocab (HF blog).
- TL: verified (270m, 1b-pt with all phases; 4b-pt phase 1 only).
- **Interpretability bonus: Gemma Scope 2.** These are SAEs and transcoders for every layer of the Gemma 3 270M/1B/4B/12B/27B, in both pt and it versions (`google/gemma-scope-2-4b-pt`, etc.), CC-BY-4.0, loadable through SAELens ([HF landing](https://huggingface.co/google/gemma-scope-2), [DeepMind blog](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)). This makes `gemma-3-4b-pt` the only Mac-sized multilingual base model with ready-made SAEs.
- Precision: use **bf16, not fp16**. Gemma-family activations are known to overflow in fp16 (community-reported; I did not re-verify it for Gemma 3 here). MPS supports bf16 on recent macOS / PyTorch.

### Gemma 4 (E2B, E4B), new in 2026
- IDs: `google/gemma-4-E2B`, `google/gemma-4-E4B` (base); `-it` variants exist. **Apache-2.0 and not gated**, a change from Gemma 3 ([Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), HF API). Released 2026-04-02. The 12B "Unified" model came later (HF repo May 2026).
- Architecture: Per-Layer Embeddings (PLE). E2B = 2.3B effective / 5.1B total params, 35 layers; E4B = 4.5B / 8B, 42 layers. Vocab 262k. "Pre-trained on 140+ languages." Multimodal (text, image, audio) ([model card](https://ai.google.dev/gemma/docs/core/model_card_4)).
- Memory: the weights on disk are 10.2 GB (E2B) and 16 GB (E4B) in bf16, because PLE tables count toward storage. **E2B is borderline on a 16 GB Mac and E4B does not fit.**
- TL: verified (E2B phases 1-2; E4B phases 1-2).
- German evidence: the base checkpoints score poorly on EuroEval (E2B rank 3.89, E4B 3.56). The `-it` versions are strong (E4B-it rank 1.92, ScaLA-de 49.3). The difference is probably few-shot formatting rather than knowledge, but it is not encouraging. I found no Gemma Scope release for Gemma 4 (unverified that none exists).

### Llama 3.2 (1B, 3B)
- IDs: `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`. Llama 3.2 Community License, **gated**: you must accept Meta's form on HF ([card](https://huggingface.co/meta-llama/Llama-3.2-1B)).
- Data: officially supports English, **German**, French, Italian, Portuguese, Hindi, **Spanish**, Thai. Up to 9T tokens. The 1B and 3B were pruned from Llama 3.1 8B and then distilled from 8B/70B logits.
- German evidence (card, base, MMLU 5-shot): 1B de 39.2 / es 41.5; 3B **de 53.3 / es 55.1**. EuroEval: 1B rank 3.88 (weak); no Base-3B entry (`Llama-3.2-3B-Instruct` ES rank 2.62).
- Tokenizer: tiktoken-based, 128k vocab.
- TL: verified. It is also the most-studied model family in interpretability work, so existing code applies directly.

### EuroLLM 1.7B (and 9B / 22B for GPU)
- ID: `utter-project/EuroLLM-1.7B`. Apache-2.0, not gated. Released 2024-09-24 ([card](https://huggingface.co/utter-project/EuroLLM-1.7B)).
- Data: 4T tokens across 35 languages, including all EU languages. English is 50% in the first phase and 32.5% in the annealing phase. The remaining share is split "based on the amount of data obtained after collection and filtering". The exact per-language shares appear only in a figure ([paper](https://arxiv.org/html/2409.16235v1)), so the German and Spanish percentages are unverified.
- Tokenizer: 128k vocabulary designed for European languages (paper).
- German evidence (card): HellaSwag-de 0.606, ARC-de 0.346; HellaSwag-es 0.479, ARC-es 0.368. EuroEval: not on the German NLU board; Spanish rank 3.40 (weak).
- Newer EuroLLM releases: no new small dense base. The 2025-2026 releases are EuroLLM-9B-2512, EuroLLM-22B-2512 ([tech report](https://arxiv.org/abs/2602.05879)), a MoE `utter-project/EuroMoE-2.6B-A0.6B-2512`, and Megatron-core phase checkpoints (`utter-project/eurollm-1.7b-mcore-phase1`, `-phase2`, ...). The phase checkpoints are in Megatron format, not HF, so they are not directly loadable for intermediate-checkpoint analysis (unverified whether conversion scripts are provided).
- TL: unverified, but it uses plain `LlamaForCausalLM`, which TL supports, so it should load.

### SmolLM3-3B-Base
- ID: `HuggingFaceTB/SmolLM3-3B-Base`. Apache-2.0, not gated. Released 2025-07-08.
- Data: 11.2T tokens. Six native languages: en, fr, **es, de**, it, pt. About 12% of the web mixture is multilingual (FineWeb2 / FineWeb2-HQ) ([card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base), [blog](https://huggingface.co/blog/smollm3)).
- German evidence (card, base): Global-MMLU-de 35.1, Belebele-de 48.4, HellaSwag-de 59.6; Spanish 38.5 / 47.0 / 65.9. EuroEval DE rank 2.39, GermanQuAD 61.5.
- Architecture: GQA plus NoPE, with RoPE removed in every 4th layer. This is relevant if you look at positional mechanisms. Tokenizer: reportedly the Llama-3.2 128k tokenizer (unverified).
- **Intermediate checkpoints:** `HuggingFaceTB/SmolLM3-3B-checkpoints` has 133 branches covering pretraining stages, mid-training and SFT. Good for asking when gender agreement emerges.
- TL: verified.

### Salamandra 2B (BSC)
- ID: `BSC-LT/salamandra-2b`. Apache-2.0, not gated. Branches `salamandra-2b_v1.0` and `salamandra-2b_v1.1` exist.
- Data: about 12.9T tokens (5 epochs) over 35 European languages and code. **Spanish is about 16.1%** (upsampled 2x), German 4.8%, English 39.3%, Catalan 2.0% ([card](https://huggingface.co/BSC-LT/salamandra-2b)). This is the best-documented Spanish share in the list.
- Tokenizer: 256k vocab.
- German evidence: none on the card. The 7B scores poorly on EuroEval DE (rank 3.74), which is a weak negative signal for the 2B.
- TL: unverified (Llama arch). BSC has released no new Salamandra base checkpoints in 2026; the 2026 releases are instruct and function-calling variants (`salamandra-7b-instruct-2606`, `-fc-2607`).

### Apertus (Swiss AI)
- **v1.1 (2026, small sizes):** `swiss-ai/Apertus-v1.1-0.5B`, `swiss-ai/Apertus-v1.1-1.5B`, `swiss-ai/Apertus-v1.1-4B`, all base, with `-Instruct` siblings. Apache-2.0, not gated. They were created by "pre-training distillation" from the 8B teacher. The 4B saw 1.7T tokens ([card](https://huggingface.co/swiss-ai/Apertus-v1.1-4B)). The 4B averages 61.5 over ARC/HellaSwag/WinoGrande/XNLI/XCOPA/PIQA, with no per-language numbers. The v1.1 repos have only `main`, so no intermediate checkpoints. TL: **verified** for all three.
- **8B-2509 (GPU):** `swiss-ai/Apertus-8B-2509`. 15T tokens, 1811 languages, about 40% non-English ([tech report](https://arxiv.org/pdf/2509.14233)). xIELU activation and Goldfish loss. **72 intermediate-checkpoint branches.** EuroEval DE rank 2.51, ScaLA-de 17.9. TL verified.
- **v1.5 (new, Sept 2026):** `swiss-ai/Apertus-v1.5-8B-base`, 8.84B params, multimodal, new `Apertus1p5ForConditionalGeneration` architecture with an audio tokenizer ([card](https://huggingface.co/swiss-ai/Apertus-v1.5-8B-base)). It is not in the TL registry yet. It will probably need a very recent `transformers`.
- Caveat: Apertus spreads its non-English data over 1811 languages. A single language like German is under 1.4% of FineWeb-2 ([search summary of tech report](https://arxiv.org/pdf/2509.14233); exact German share unverified).

### Ministral 3 3B Base
- ID: `mistralai/Ministral-3-3B-Base-2512`. Apache-2.0, not gated. Released December 2025. It is a 3.4B LM plus a 0.4B vision encoder (`Mistral3ForConditionalGeneration`) ([card](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)).
- Languages: "dozens", explicitly including German and Spanish. No data shares are published.
- **German evidence: the strongest of any small model here.** EuroEval DE rank 1.83 (two runs: 1.83 / 1.88), ScaLA-de 49.1 / 41.4, GermanQuAD 67-69. Spanish rank 2.74.
- TL: **unverified** (registry status 0 for the Base; the `Ministral3ForCausalLM` Reasoning variant was "skipped"). nnsight should work. To use TL you may have to extract the text LM yourself. Tokenizer: Tekken, ~131k (unverified).

### Granite 4.1 3B Base
- ID: `ibm-granite/granite-4.1-3b-base`. Apache-2.0, released 2026-04-29. Dense decoder: 40 layers, d=2560, GQA, ~15T tokens. 12 languages including de and es. Aggregate MMMLU 56.6 / INCLUDE 51.8 ([card](https://huggingface.co/ibm-granite/granite-4.1-3b-base)). TL verified. No German-specific numbers, and it is not on EuroEval. A reasonable alternate. Note that the Granite **4.0** "-h" models are Mamba hybrids, and several of them *failed* TL verification.

### OLMo 2 1B / OLMo 3
- `allenai/OLMo-2-0425-1B`: Apache-2.0, TL verified, **268 intermediate-checkpoint branches**. It is English-centric, so it is only useful as a contrast to show what a model without German/Spanish exposure does.
- The newer OLMo 3 / 3.1 releases have no small base: the smallest is `allenai/Olmo-3-1025-7B`. EuroEval DE rank 2.24, but ScaLA-de is only 26.1 and ScaLA-es 0.

### Phi-4-mini
- **No base checkpoint was released.** Only `microsoft/Phi-4-mini-instruct`, `-reasoning` and `-flash` exist ([card](https://huggingface.co/microsoft/Phi-4-mini-instruct)). MIT license, 23 languages including de and es, 200k vocab, 5T tokens, MMMLU 49.3. Drop it, given the preference for base models.

### Teuken 7B (GPU)
- `openGPT-X/Teuken-7B-base-v0.6`. **CC-BY-NC-4.0** (non-commercial), not gated. 6T tokens across the 24 EU languages, with a custom multilingual SentencePiece tokenizer (vocab size not on the card) ([card](https://huggingface.co/openGPT-X/Teuken-7B-base-v0.6)). The roughly 50-60% non-English share is from memory and unverified. It probably needs `trust_remote_code` for the tokenizer (unverified). TL unverified. It is not on EuroEval. Lower priority than EuroLLM-9B.

### Other 2026 releases checked and judged not a good fit
- `openbmb/MiniCPM5-1B-Base` (2026-05, Llama arch): no German or Spanish data documented.
- `LiquidAI/LFM2.5-*`: hybrid conv/attention and mostly instruct.
- `tiiuae/Falcon-H1-*-Base`: Mamba hybrid; TL coverage deferred.
- `utter-project/EuroMoE-2.6B-A0.6B-2512`: MoE complicates residual-stream work.

### Interpretability-friendly extras (summary)
| Resource | Models | Notes |
|---|---|---|
| Gemma Scope 2 SAEs and transcoders | Gemma 3 270M/1B/4B/12B/27B, pt and it | Every layer; CC-BY-4.0; SAELens ([link](https://huggingface.co/google/gemma-scope-2)) |
| Intermediate checkpoints | SmolLM3-3B (133 branches), OLMo-2-0425-1B (268), Apertus-8B-2509 (72) | Training-dynamics studies; only SmolLM3 and Apertus are multilingual |
| Prior work | Brinkmann et al. 2025 found shared gender features across 15 gendered languages using SAEs on Llama-3-8B and Aya-23-8B ([arXiv 2501.06346](https://arxiv.org/abs/2501.06346)) | A useful baseline to reproduce on Llama-3.1-8B on the GPU |

---

## 3. Ranked shortlist for the behavioral der/die and el/la check

All six are base checkpoints, fit on the 16 GB Mac in bf16, and document German and Spanish data.

1. **`Qwen/Qwen3-4B-Base`**, with `Qwen/Qwen3-1.7B-Base` as a cheap sibling. TL-verified, ungated and Apache-licensed. Its instruct sibling has the best German and Spanish EuroEval results of any non-Mistral small model. The 0.6/1.7/4/8B ladder with a shared tokenizer allows a clean scaling comparison. ~8 GB in bf16.
2. **`mistralai/Ministral-3-3B-Base-2512`**. Strongest German evidence of any small model (EuroEval DE rank 1.83, ScaLA-de ~49), Apache-licensed and ungated. The risk is tooling: the TL support is unverified and there is a vision wrapper, so start with nnsight. ~7.7 GB.
3. **`HuggingFaceTB/SmolLM3-3B-Base`**. German and Spanish are explicit "native" languages with published per-language base-model scores. TL-verified. It has 133 intermediate checkpoints for studying when gender emerges. ~6.2 GB.
4. **`google/gemma-3-4b-pt`**. Multilingual (140+ languages, unlike the English-only 1B) and TL-verified. It is the only candidate with off-the-shelf **all-layer SAEs (Gemma Scope 2)** for checking probe and LEACE directions against SAE features. Gated; use bf16. ~8.6 GB.
5. **`meta-llama/Llama-3.2-3B`**. de and es are official languages, with documented base MMLU-de 53.3 / es 55.1. TL-verified and the most-studied family in interpretability. Gated. ~6.4 GB.
6. **`utter-project/EuroLLM-1.7B`**. A European-focused model with documented data mixing and a 128k EU-language tokenizer, as a contrast to the US/Chinese generalists. Smallest and fastest (~3.4 GB). Benchmark evidence is middling, and TL is unverified but should work (Llama arch).

Alternates if one of the six fails the behavioral check:
- `BSC-LT/salamandra-2b`: highest documented Spanish share (16%).
- `swiss-ai/Apertus-v1.1-4B`: 2026, TL-verified, fully open data.
- `Qwen/Qwen3.5-2B-Base`: 2026 hybrid DeltaNet architecture.
- `ibm-granite/granite-4.1-3b-base`: 2026, TL-verified.

For later GPU runs:
- `mistralai/Ministral-3-8B-Base-2512`: best German evidence.
- `Qwen/Qwen3-8B-Base`: scale continuation of the top Mac pick.
- `meta-llama/Llama-3.1-8B`: matches prior gender-feature work.
- `swiss-ai/Apertus-8B-2509`: open data plus intermediate checkpoints.
- `utter-project/EuroLLM-9B-2512`.

### Gated-access steps
- **Gemma 3** (`google/gemma-3-*`, and the Gemma Scope 2 repos if they turn out to be gated; not checked): log in to HF, open the model page, accept the Gemma terms, then run `huggingface-cli login` (or `hf auth login`) with a read token.
- **Llama 3.2 / 3.1** (`meta-llama/*`): submit Meta's contact-info / license form on the model page and wait for approval, which is usually quick but manual. Then use the same token login.
- `utter-project/EuroLLM-9B` (the original 2024 repo) is gated=auto, meaning you click to accept and access is immediate. `EuroLLM-9B-2512` is not gated.
- All other models listed are ungated, including Gemma 4, which is Apache-2.0.

### Practical notes for the check
- Tokenization: before scoring, confirm that the space-prefixed articles (" der", " die", " das", " el", " la", and " los", " las" if used) are single tokens in each tokenizer. Also log how many subword pieces each noun splits into, since multi-piece nouns can confound the probe position. Gemma and Salamandra have 256-262k vocabs, so they will split nouns less than Llama, EuroLLM and SmolLM3 at 128k.
- Prompt direction: an article-prediction setup ("... Ich sehe ___ Hund") makes the model predict the article *before* it sees the noun. To assess a noun's gender knowledge, score P(article | noun context), for example with "Das Wort 'Hund' hat den Artikel:", or compare the likelihood of "der Hund" against "die Hund". Decide this before comparing models.
- Multimodal wrappers: `gemma-3-4b-pt`, Gemma 4, Qwen3.5 and Ministral 3 load as `*ForConditionalGeneration`. Hook the language-model submodule's decoder layers, not the vision tower.
