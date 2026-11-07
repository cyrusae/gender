# Supplement: model × test × N × device × result

Compiled 2026-11-07 from the result files named in each section (not from the drafts), for the
write-up's supplementary table. Layer counts are "layers passing / inner layers". Primary
position LAST throughout. Device: Mac = Apple MPS fp16; pod = CUDA L40S bf16.

## Phase 3 (German; `results/phase3/<model>/summary.json`, Gemma `results/gemma_slim/phase3/`)

Training set `strat3`; Gemma counts are after the known-check filter.

| model | device | N strat3 (m/f/n) | compounds follow head | markedness | reading |
|---|---|---|---|---|---|
| Qwen3-0.6B | pod | 64/27/31 | 26/27 | 0/27 | follow head; markedness not supported |
| Qwen3-1.7B | Mac | 64/27/31 | 27/27 | 0/27 | same |
| Qwen3-4B | Mac | 64/27/31 | 35/35 | 0/35 | same |
| Qwen3-8B | pod | 64/27/31 | 35/35 | 0/35 | same |
| Qwen3-14B | pod | 64/27/31 | 39/39 | 0/39 | same |
| Qwen3-30B-A3B | pod | 64/27/31 | 47/47 | 0/47 | same |
| Gemma-3-4B | pod | 58/24/30 | 33/33 | 0/33 | same |
| Gemma-3-12B | pod | 60/26/29 | 47/47 | 0/47 | same |

## Phase 4 (cross-language; `results/phase4/<model>/summary.json`, Gemma `results/gemma_slim/phase4/`)

T1/T3 = layers passing after Holm. T2 = layers whose cross term beats the shuffle null (the
adopted T2 statistic, descriptive); Gemma T2 from the outlier-fixed rerun
(`results/outlier_fix/phase4/<model>/`; T1/T3 there are identical to the raw run).

| model | device | pairs | N es / de | T1 es→de | T1 de→es | T1 | T3 es | T3 de | T3 | T2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | Mac | 271 | 165 / 91 | 0/27 | 14/27 | fails | 27/27 | 26/27 | passes | 22/27 |
| Qwen3-4B | Mac | 271 | 165 / 91 | 8/35 | 9/35 | fails | 35/35 | 35/35 | passes | 25/35 |
| Qwen3-8B | pod | 271 | 165 / 91 | 6/35 | 5/35 | fails | 35/35 | 35/35 | passes | 23/35 |
| Qwen3-14B | pod | 271 | 165 / 91 | 7/39 | 22/39 | fails | 39/39 | 39/39 | passes | 28/39 |
| Gemma-3-4B | pod | 250 | 156 / 82 | 7/33 | 26/33 | fails | 33/33 | 33/33 | passes | 31/33 (outlier-fixed) |
| Gemma-3-12B | pod | 254 | 151 / 86 | 11/47 | 27/47 | fails | 43/47 | 38/47 | passes | 42 of 44 valid (outlier-fixed; layers 3–5 out of range) |

Raw (unfixed) Gemma T2 counts are 5/33 and 4/47; they rest on cosine pieces the outlier
dimension dominates and are not reported as results.

## Phase 5 (Spanish steering; `results/phase5_analysis/<model>/verdicts.csv`)

Layers passing (beats the random-direction 95th percentile) / steering layers; effect = mean
shift at the working dose. German: no valid test on every model (frame or gate failure).

| model | device | noun-push R1 | noun-push R1-EN | all-positions R1 | all-positions R1-EN | verdict |
|---|---|---|---|---|---|---|
| Qwen3-8B | pod | 3/4 (+0.020) | 3/4 (+0.011) | 3/4 (+0.065) | 2/4 (+0.034) | Spanish language-specific effect |
| Qwen3-14B | pod | 0/3 (+0.000) | 0/3 (−0.001) | 0/3 (+0.001) | 0/3 (+0.001) | no positive effect |
| Gemma-3-4B | pod | 0/3 (−0.004) | 0/3 (+0.007) | 0/3 (+0.035) | 0/3 (+0.029) | no positive effect |
| Gemma-3-12B | pod | 1/3 (+0.029) | 1/3 (+0.018) | 0/3 (+0.100) | 0/3 (+0.060) | no positive effect (all-positions push matched by random) |

## German gender × frequency overlap (confidence control (c); `results/outlier_fix/c_verdicts.csv`)

| model | German | Spanish |
|---|---|---|
| Qwen3-8B | overlap | no overlap |
| Qwen3-14B | overlap | no overlap |
| Gemma-3-4B | no overlap | no overlap |
| Gemma-3-12B | no overlap | mixed |

## Follow-ups on Qwen3 8B/14B (from the preregs' results sections)

| test | N | device | 8B | 14B |
|---|---|---|---|---|
| Frequency axis A/C/D | 91 de training; 14 matched pairs | CPU on pod activations | overlap survives | overlap survives |
| Frequency axis B (axis removed) | Phase 4 set | CPU | verdicts unchanged | verdicts unchanged |
| Register / AoA (E2) | 80 rated de nouns | CPU | overlap remains | overlap remains |
| G2 SAE | 632 held-out compounds | Mac (8B only) | no valid test | – |
| Rarity held-out | 888 (502 f / 190 m / 196 n) | pod extraction, CPU analysis | mixed | no valid test |
| Russian R-A | 174 (100 f / 74 m) | CPU on pod activations | one-way | one-way |
| Russian R-B | 174 | CPU | no overlap | no overlap |
| P30 Spanish object pull | 600 nonce nouns | pod | S(object) +0.080 | +0.124 |
| P31 German object pull | 300 nonce nouns | pod | none (0.002) | no valid test |

P30/P31 on the smaller models (Mac): see draft §3.6.
