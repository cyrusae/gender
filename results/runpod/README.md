# RunPod confirmatory results (session 1, 2026-10-08)

Activations: one A100 SXM 80 GB, bf16, code bundle commit 90516ac (see each `meta`). Analyses: on
the Mac from those activations, worktree `../gender-pod` (branch `pod-analysis`), analysis commit
ca6b92b (recorded per result as `analysis_git`), lbfgs probe solver, bootstrap keeping copies of
a resampled item in one half. Only results produced from pod data are here (folders whose
metadata say `device: cuda`, plus the readout check run on pod activations).

Models: Qwen3 0.6B/1.7B/4B/8B/14B base (8B/14B confirmatory), 30B-A3B (exploratory).
Not yet analysed from pod data: Phase 1. Summary tables: `docs/explainers/02-…` and `03-…`
("Confirmatory results").
