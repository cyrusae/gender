# RunPod results (session 2, 2026-10-12/13)

Pod runs: one L40S 48 GB, bf16, code bundle commit 482430e (see each `meta`). Analyses: on the
Mac from the pod's activations and scores, worktree `../gender-s2` (branch `s2-analysis`),
analysis commit f2d10a3 (recorded per result as `analysis_git`).

Models: Qwen3 8B/14B base (confirmatory Phases 4–5, suffix follow-up, English and German side
tests, P22/P22b), plus 0.6B/1.7B/4B wug tests (`nonce_wug/`). German Phase 5 steering did not
run (its pre-registered frame C4 failed on both sizes; decisions.md 2026-10-12/13).

Only small summaries are here (JSON/CSV under 5 MB). Not committed, kept in `../gender-s2`:
activations (~6–12 GB per model), Phase 5 steered score arrays (`results/phase5/*/*.npz`,
`*_conds.csv.gz`, ~7 GB per model) and Phase 4 bootstrap arrays. Summary tables:
`docs/explainers/04-…`, `05-…`, `06-…` and `docs/decisions.md`.
