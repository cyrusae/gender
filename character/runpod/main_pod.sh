#!/usr/bin/env bash
# Pod entry for the Qwen3-14B main run (one 80-96 GB GPU, host CUDA >= 12.9). Run from the repo root:
#   HF_TOKEN=$(cat /root/hf_token) nohup bash character/runpod/main_pod.sh off ctrl on >> /root/main_pod.log 2>&1 &
# PARTS run in the order given (off | on | ctrl; see main_qwen.sh). Batch sizes via WN CN PN env
# (default 10 2 10 = batch 1). Resumable.
# End marker (whole line): ^=== CHARACTER POD DONE$ (every part DONE) or ^=== CHARACTER POD INCOMPLETE$
set -uo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d .venv-vllm ] || uv venv -q .venv-vllm
source .venv-vllm/bin/activate
uv pip install -q "vllm==0.31.0" pandas
python -c "import vllm, torch; torch.zeros(1).cuda(); print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))" \
  || { echo "=== ABORT: CUDA init failed (host driver too old for this torch build)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
ALL=0
for P in "$@"; do
  bash character/runpod/main_qwen.sh "$P" "${WN:-10}" "${CN:-2}" "${PN:-10}" | tee /root/part_$P.log
  grep -qx "=== QWEN MAIN $P DONE" /root/part_$P.log || ALL=1
done
if [ "$ALL" -eq 0 ]; then echo "=== CHARACTER POD DONE"; else echo "=== CHARACTER POD INCOMPLETE"; fi
