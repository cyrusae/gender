"""Command-line entry point: `uv run gbleed <command> ...`."""

from __future__ import annotations

import argparse

DEFAULT_STIMULI = "data/stimuli/phase0_seed.csv"


def main() -> None:
    p = argparse.ArgumentParser(prog="gbleed")
    sub = p.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("phase0", help="Behavioural gender check for one or more models")
    p0.add_argument("models", nargs="+", help="Hugging Face model IDs")
    p0.add_argument("--stimuli", default=DEFAULT_STIMULI)
    p0.add_argument("--out", default="results/phase0")
    p0.add_argument("--device", default=None, help="mps | cuda | cpu (default: auto)")
    p0.add_argument("--dtype", default=None, help="float16 | bfloat16 | float32")
    p0.add_argument("--langs", nargs="*", default=None, choices=["de", "es"])

    pc = sub.add_parser("phase0-compare", help="Tabulate all Phase 0 results")
    pc.add_argument("--out", default="results/phase0")

    a = p.parse_args()
    from . import phase0  # deferred: keeps --help fast (no torch import)

    if a.cmd == "phase0":
        for m in a.models:
            try:
                phase0.run(m, a.stimuli, a.out, a.device, a.dtype, a.langs)
            except Exception as e:  # noqa: BLE001 - keep going through the model list
                print(f"!! {m} failed: {type(e).__name__}: {e}")
            finally:
                _free_memory()
        _print_table(phase0.compare(a.out))
    elif a.cmd == "phase0-compare":
        _print_table(phase0.compare(a.out))


def _free_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_table(table) -> None:
    if len(table):
        print(table.to_string(index=False))
