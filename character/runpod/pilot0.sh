#!/usr/bin/env bash
# Pilot 0 on one GPU pod (L40S 48 GB is enough for Qwen3-8B and Gemma-3-12B-it in bf16, one at a time).
# Run from the repo root on the pod:  HF_TOKEN=... bash character/runpod/pilot0.sh 2>&1 | tee pilot0.log
# Gemma is gated: HF_TOKEN must belong to an account that accepted its licence.
# End marker (match as a whole line): ^=== INSTRUCT PILOT0 done$
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
START=$(date +%s)
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv -q .venv-vllm && source .venv-vllm/bin/activate
uv pip install -q vllm pandas
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))"
cd instruct
OUT=results/pilot0
for M in Qwen/Qwen3-8B google/gemma-3-12b-it; do
  python gen.py --config configs/pilot0_v1.json --nouns stimuli/pilot0_nouns_v1.csv \
    --model "$M" --backend vllm --out "$OUT/${M//\//__}.jsonl"
done
python code_outputs.py "$OUT"/*.jsonl --out "$OUT/summary.csv"
echo "took $(( $(date +%s) - START )) s"
echo "=== INSTRUCT PILOT0 done"
