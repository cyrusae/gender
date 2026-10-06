#!/usr/bin/env bash
# Run once on a fresh pod, from the repo root (e.g. /workspace/gender).
# Keeps the Hugging Face model cache on the persistent volume so restarts don't re-download.
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}   # local disk: the global volume is object storage (slow for many small files)
START=$(date +%s)
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen
uv run python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| CUDA", torch.version.cuda)
PY
uv run pytest -q
echo "setup took $(( $(date +%s) - START )) s"
