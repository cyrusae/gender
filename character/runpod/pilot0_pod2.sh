#!/usr/bin/env bash
# Part 2 (PI 2026-10-26): Qwen3-14B off/on, Gemma-3-12B-it (own-GPU vs API check), gpt-oss-20b.
# Run from the repo root on the pod:
#   HF_TOKEN=$(cat /root/hf_token) bash character/runpod/pilot0_pod2.sh >> /root/pilot0_pod2.log 2>&1
# Resumable (rows are skipped if already in the output file). Per-run markers: "=== start/end NAME".
# End marker (match as a whole line): ^=== INSTRUCT POD PILOT0 PART2 DONE$
set -uo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
START=$(date +%s)
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d .venv-vllm ] || uv venv -q .venv-vllm
source .venv-vllm/bin/activate
uv pip install -q "vllm==0.31.0" pandas
# vLLM 0.27-0.31 need torch 2.13, built only for CUDA 12.9/13.0: create the pod with
# gpu.minCudaVersion "13.0" (a 12.8 host fails at CUDA init). Abort here, before any job or download.
python -c "import vllm, torch; torch.zeros(1).cuda(); print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))" \
  || { echo "=== ABORT: CUDA init failed (host driver too old for this torch build)"; exit 1; }
cd instruct
OUT=results/pilot0_v6_pod; mkdir -p $OUT
run() {  # name model thinking [extra args]
  local name=$1 model=$2 th=$3; shift 3
  echo "=== start $name $(date '+%F %T')"
  python gen.py --config configs/pilot0_v6.json --nouns stimuli/pilot0_nouns_v2.csv \
    --model "$model" --backend vllm --thinking "$th" --out "$OUT/$name.jsonl" "$@" 2>&1 \
    | grep -vE "it/s\]|Processed prompts|^INFO|WARNING" || true
  echo "=== end $name $(date '+%F %T')"
}
run qwen3_14b_off  Qwen/Qwen3-14B        off
run qwen3_14b_on   Qwen/Qwen3-14B        on
run gemma3_12b     google/gemma-3-12b-it off
run gptoss_20b     openai/gpt-oss-20b    off
for f in $OUT/qwen3_14b*.jsonl $OUT/gemma3_12b.jsonl $OUT/gptoss_20b.jsonl; do python code_outputs.py "$f" --out "${f%.jsonl}_summary.csv" | tail -1; done
echo "took $(( $(date +%s) - START )) s"
echo "=== INSTRUCT POD PILOT0 PART2 DONE"
