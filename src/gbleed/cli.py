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
    p0.add_argument("--min-margin", type=float, default=1.0, help="nats; below this = unsure")

    pc = sub.add_parser("phase0-compare", help="Tabulate all Phase 0 results")
    pc.add_argument("--out", default="results/phase0")
    pc.add_argument(
        "--by", default=None, help="breakdown: freq_bin | set | de_suffix | es_exception"
    )

    pl = sub.add_parser("lexicon", help="Build noun lexicons from Wiktionary (kaikki.org)")
    pl.add_argument("--langs", nargs="*", default=["de", "es"], choices=["de", "es"])
    pl.add_argument("--redownload", action="store_true")

    sub.add_parser("classics", help="Look up genders for data/stimuli/classics_spec.csv")

    ps = sub.add_parser("sample-phase0", help="Sample a Phase 0 stimulus list from the lexicons")
    ps.add_argument("--out", default="data/stimuli/phase0_v3.csv")
    ps.add_argument("--per-cell", type=int, default=20, help="nouns per lang x freq bin x gender")
    ps.add_argument("--max-pairs", type=int, default=60, help="flipped (and control) pairs")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--abstract-pairs", action="store_true", help="allow non-concrete pairs")

    a = p.parse_args()
    if a.cmd == "lexicon":
        from . import lexicon

        for lang in a.langs:
            lexicon.build_lexicon(lang, a.redownload)
        lexicon.build_pairs()
        lexicon.write_multi_candidates()
        return
    if a.cmd == "classics":
        from . import lexicon

        lexicon.build_classics()
        return
    if a.cmd == "sample-phase0":
        from . import lexicon
        from .phase0 import LANG_CONFIG

        shots = {(lang, n) for lang, c in LANG_CONFIG.items() for n, _ in c["shots"]}
        lexicon.sample_phase0(
            a.out,
            a.per_cell,
            a.max_pairs,
            a.seed,
            exclude=shots,
            concrete_pairs=not a.abstract_pairs,
        )
        return

    from . import phase0  # deferred: keeps --help fast (no torch import)

    if a.cmd == "phase0":
        for m in a.models:
            try:
                phase0.run(m, a.stimuli, a.out, a.device, a.dtype, a.langs, a.min_margin)
            except Exception as e:  # noqa: BLE001 - keep going through the model list
                print(f"!! {m} failed: {type(e).__name__}: {e}")
            finally:
                _free_memory()
        _print_table(phase0.compare(a.out))
    elif a.cmd == "phase0-compare":
        _print_table(phase0.compare(a.out, a.by))


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
