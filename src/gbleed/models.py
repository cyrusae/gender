"""Device/dtype selection and model loading."""

from __future__ import annotations

import platform
import re
import subprocess

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(name: str | None, device: str) -> torch.dtype:
    if name:
        return DTYPES[name]
    # M1/M2 have no native bf16; fp16 is the safe half-precision default on MPS.
    # Note: some models (notably Gemma) overflow in fp16 -> pass --dtype float32.
    return torch.bfloat16 if device == "cuda" else torch.float16


def load_model(model_id: str, device: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id)


def git_state() -> dict:
    """Commit of the code that produced a result, and whether it had uncommitted
    changes (a dirty run can't be reproduced exactly from the commit alone)."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = git("rev-parse", "HEAD")
    # Outputs (results/) don't count: only code, configs and stimulus data.
    dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"))
    if dirty:
        print("!! Uncommitted changes: results will record git_dirty=true")
    return {"git_commit": commit or None, "git_dirty": dirty if commit else None}


def run_metadata(model, model_id: str, device: str, dtype: torch.dtype) -> dict:
    """Everything needed to tell whether two runs are comparable."""
    return {
        **git_state(),
        "model_id": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_layers": getattr(model.config, "num_hidden_layers", None),
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
    }
