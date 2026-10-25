"""Command-line entry point: `uv run gbleed <command> ...`."""

from __future__ import annotations

import argparse

DEFAULT_STIMULI = "data/stimuli/phase0_seed.csv"


def main() -> None:
    import sys

    # Logs update live even when output goes to a file (e.g. `nohup ... > run.log`).
    sys.stdout.reconfigure(line_buffering=True)
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
    p0.add_argument("--batch-size", type=int, default=32, help="sentences per forward pass")

    pc = sub.add_parser("phase0-compare", help="Tabulate all Phase 0 results")
    pc.add_argument("--out", default="results/phase0")
    pc.add_argument(
        "--by", default=None, help="breakdown: freq_bin | set | de_suffix | es_exception"
    )

    pl = sub.add_parser("lexicon", help="Build noun lexicons from Wiktionary (kaikki.org)")
    pl.add_argument("--langs", nargs="*", default=["de", "es"], choices=["de", "es", "ru"])
    pl.add_argument("--redownload", action="store_true")

    sub.add_parser("classics", help="Look up genders for data/stimuli/classics_spec.csv")
    pr = sub.add_parser("readout", help="Gender-blind readout-position check (LAST vs AFTER)")
    pr.add_argument("models", nargs="+")
    pr.add_argument("--skip-extract", action="store_true")
    pr.add_argument("--extract-only", action="store_true", help="GPU sessions: skip the analysis")
    sub.add_parser("token-report", help="Add Qwen3 token counts to the lexicons; write the report")
    pm = sub.add_parser("multi-check", help="Do models accept both genders for See/mar-type items?")
    pm.add_argument("models", nargs="+")
    sub.add_parser("phase2-finalize", help="Keep nouns known by all models; freeze splits")
    p2 = sub.add_parser("phase2", help="Extract activations and run the Phase 2 analysis")
    p2.add_argument("models", nargs="+")
    p2.add_argument("--device", default=None)
    p2.add_argument("--dtype", default=None)
    p2.add_argument("--skip-extract", action="store_true")
    p2.add_argument("--extract-only", action="store_true",
                    help="GPU sessions: save activations, skip the CPU analysis")  # fmt: skip
    p2.add_argument("--position", choices=["last", "after"], default="last",
                    help="readout position; 'after' is the pre-registered robustness check")  # fmt: skip
    p2.add_argument("--known-only", action="store_true",
                    help="sensitivity: only nouns this model knows (results/phase2_known)")  # fmt: skip
    fc = sub.add_parser("frame-check", help="Behavioural check of the homograph verb frame")
    fc.add_argument("models", nargs="+")
    fc.add_argument("--device", default=None)
    fc.add_argument("--dtype", default=None)
    sub.add_parser("phase3-stimuli", help="Build the Phase 3 German pool (m/f/n)")
    sub.add_parser("phase4-pairs", help="Build the Phase 4 flipped-pair pool (389 pairs)")
    p4k = sub.add_parser("phase4-known", help="Known checks on both nouns of every pair")
    p4k.add_argument("models", nargs="+")
    p4k.add_argument("--device", default=None)
    p4k.add_argument("--dtype", default=None)
    sub.add_parser("phase4-finalize", help="Keep pairs known by 1.7B and 4B")
    p4 = sub.add_parser("phase4", help="Extract Phase 4 words and run the Phase 4 analysis")
    p4.add_argument("models", nargs="+")
    p4.add_argument("--device", default=None)
    p4.add_argument("--dtype", default=None)
    p4.add_argument("--position", choices=["last", "after"], default="last")
    p4.add_argument("--skip-extract", action="store_true")
    p4.add_argument("--extract-only", action="store_true")
    p4.add_argument("--n-boot", type=int, default=1000)
    p3k = sub.add_parser("phase3-known", help="Three-way article known check for Phase 3")
    p3k.add_argument("models", nargs="+")
    p3k.add_argument("--device", default=None)
    p3k.add_argument("--dtype", default=None)
    sub.add_parser("phase3-finalize", help="Keep Phase 3 nouns known by 1.7B and 4B; freeze splits")
    p3 = sub.add_parser("phase3", help="Extract German activations and run the Phase 3 analysis")
    p3.add_argument("models", nargs="+")
    p3.add_argument("--device", default=None)
    p3.add_argument("--dtype", default=None)
    p3.add_argument("--position", choices=["last", "after"], default="last")
    p3.add_argument("--skip-extract", action="store_true")
    p3.add_argument("--extract-only", action="store_true")
    p3.add_argument("--known-only", action="store_true",
                    help="sensitivity: only nouns this model knows (results/phase3_known)")  # fmt: skip
    sub.add_parser("phase2-stimuli", help="Build Phase 2 pools (data/stimuli/phase2_pool_v3.csv)")
    sub.add_parser("phase1-stimuli", help="Build Phase 1 verb / nonce / noun lists (v1)")
    p1 = sub.add_parser("phase1", help="Extract activations and fit/validate the spelling eraser")
    p1.add_argument("models", nargs="+")
    p1.add_argument("--device", default=None)
    p1.add_argument("--dtype", default=None)
    p1.add_argument("--skip-extract", action="store_true", help="reuse saved activations")
    p1.add_argument("--extract-only", action="store_true",
                    help="GPU sessions: save activations, skip the CPU analysis")  # fmt: skip

    ps = sub.add_parser("sample-phase0", help="Sample a Phase 0 stimulus list from the lexicons")
    ps.add_argument("--out", default="data/stimuli/phase0_v3.csv")
    ps.add_argument("--per-cell", type=int, default=20, help="nouns per lang x freq bin x gender")
    ps.add_argument("--max-pairs", type=int, default=60, help="flipped (and control) pairs")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--abstract-pairs", action="store_true", help="allow non-concrete pairs")

    p5 = sub.add_parser("p5", help="Session-2 steps (Phases 4-5 additions), one model at a time")
    p5.add_argument("step", choices=["english-extract", "number-extract", "wug", "gate-framecheck",
                                     "readout-check", "gate", "session", "p20", "suffix-extract",
                                     "p4-known", "ru-known", "suffix-known", "p22", "p22b", "phase5-all",
                                     "diminutives-extract", "p29", "p29-analyze", "p30", "p30-analyze", "p31", "p31-analyze"])  # fmt: skip
    p5.add_argument("models", nargs="+")
    cc = sub.add_parser("confidence-ctl", help="Confidence/flattening control (prereg 2026-10-21)")
    cc.add_argument("--parts", default="abc", help="any of a, b, c")
    cc.add_argument("--models", nargs="+", default=None, help="default: Qwen3 8B and 14B")
    cc.add_argument("--out", default=None, help="default: results/confidence_ctl")
    ns = sub.add_parser("noun-sense", help="noun-sense adjective sensitivity check (gemma prereg)")
    ns.add_argument("--models", nargs="+", default=["Qwen/Qwen3-8B-Base", "Qwen/Qwen3-14B-Base"])
    ns.add_argument("--root", default=None, help="Phase 5 sweep root (default: paths.p5_dir())")
    ns.add_argument("--rc", default=None, help="readout-check root")
    fa = sub.add_parser(
        "frequency-axis", help="German frequency-axis follow-up (prereg 2026-10-21)"
    )
    fa.add_argument("--parts", default="acdb", help="any of a, c, d (together) and b")
    fa.add_argument("--models", nargs="+", default=None, help="default: Qwen3 8B and 14B")
    fa.add_argument("--out", default=None, help="default: results/frequency_axis")
    fa.add_argument(
        "--c-ref", default=None, help="(c) cosine file the overlap reference comes from"
    )
    rh = sub.add_parser(
        "rarity-heldout", help="German rarity-lean held-out test (prereg 2026-10-22)"
    )
    rh.add_argument(
        "--extract", metavar="MODEL", default=None, help="extract H for one model (GPU)"
    )
    rh.add_argument("--models", nargs="+", default=None, help="analyze: default Qwen3 8B and 14B")
    gk = sub.add_parser(
        "gemma-known", help="Gemma slim: known-checks on the frozen lists + portability"
    )
    gk.add_argument("models", nargs="+")
    gk.add_argument("--report-only", action="store_true")
    g0 = sub.add_parser("gemma-g0", help="Gemma slim prereg G0 technical gates")
    g0.add_argument("models", nargs="*", default=None)
    la = sub.add_parser(
        "lang-audit", help="Spelling/structure/balance audit of a noun list (playbook)"
    )
    la.add_argument("lang")
    la.add_argument(
        "--list", default=None, help="CSV with lemma, gender (default: eligible lexicon)"
    )
    la.add_argument("--tag", default=None)
    la.add_argument("--cells", default=None, help="alternative cell rule, e.g. suffix (ru)")

    a = p.parse_args()
    if a.cmd == "lexicon":
        from . import lexicon

        for lang in a.langs:
            lexicon.build_lexicon(lang, a.redownload)
        if {"de", "es"} <= set(a.langs):  # German-Spanish pairs and multi-gender lists
            lexicon.build_pairs()
            lexicon.write_multi_candidates()
        return
    if a.cmd == "p5":
        _p5(a.step, a.models)
        return
    if a.cmd == "confidence-ctl":
        from .confidence_ctl import OUT, P5_MODELS
        from .confidence_ctl import main as cc_main

        cc_main(a.parts, a.models or P5_MODELS, a.out or OUT)
        return
    if a.cmd == "lang-audit":
        from .lang_audit import audit

        audit(a.lang, a.list, a.tag, a.cells)
        return
    if a.cmd == "gemma-known":
        from . import known_filter, phase0, phase2, phase3_known

        if not a.report_only:
            for m in a.models:
                phase0.run(
                    m,
                    phase2.POOL,
                    out_root=known_filter.ROOTS["es"],
                    dtype="bfloat16",
                    langs=["es"],
                )
                phase3_known.score(m, dtype="bfloat16", out_root=known_filter.ROOTS["de"])
        known_filter.report(a.models)
        return
    if a.cmd == "gemma-g0":
        from .gemma_g0 import MODELS
        from .gemma_g0 import main as g0_main

        g0_main(a.models or MODELS)
        return
    if a.cmd == "noun-sense":
        from .noun_sense import main as ns_main

        ns_main(a.models, a.root, a.rc)
        return
    if a.cmd == "rarity-heldout":
        from . import rarity_heldout as rho

        if a.extract:
            rho.extract(a.extract)
        else:
            rho.main(a.models or rho.MODELS)
        return
    if a.cmd == "frequency-axis":
        from .frequency_axis import C_REF, MODELS, OUT
        from .frequency_axis import main as fa_main

        fa_main(a.parts, a.models or MODELS, a.out or OUT, a.c_ref or C_REF)
        return
    if a.cmd == "multi-check":
        from . import multigender

        for m in a.models:
            df = multigender.run(m)
            print(f"== {m}")
            print(df[["lang", "lemma", "type", "ctx1_margin", "ctx2_margin", "both_accepted",
                      "prefers"]].round(2).to_string(index=False))  # fmt: skip
            _free_memory()
        return
    if a.cmd == "phase2-finalize":
        from .phase2 import finalize

        finalize()
        return
    if a.cmd == "phase2":
        from . import phase2

        for m in a.models:
            if not a.skip_extract:
                phase2.extract(m, a.device, a.dtype, a.position)
                _free_memory()
            if not a.extract_only:
                phase2.analyze(m, position=a.position, known_only=a.known_only)
        return
    if a.cmd == "frame-check":
        from .framecheck import run

        run(a.models, device=a.device, dtype=a.dtype)
        return
    if a.cmd == "phase3-stimuli":
        from .phase3_stimuli import build

        build()
        return
    if a.cmd == "phase3-known":
        from . import phase3_known

        for m in a.models:
            phase3_known.score(m, a.device, a.dtype)
            _free_memory()
        print("frame check:", phase3_known.frame_check(a.models))
        return
    if a.cmd == "phase3-finalize":
        from .phase3_known import finalize

        finalize()
        return
    if a.cmd == "phase3":
        from . import phase3

        for m in a.models:
            if not a.skip_extract:
                phase3.extract(m, a.device, a.dtype)  # both positions in one pass
                _free_memory()
            if not a.extract_only:
                phase3.analyze(m, a.position, known_only=a.known_only)
        return
    if a.cmd == "phase4-pairs":
        from .phase4 import build_pairs

        build_pairs()
        return
    if a.cmd == "phase4-known":
        from .phase4 import known

        for m in a.models:
            known(m, a.device, a.dtype)
            _free_memory()
        return
    if a.cmd == "phase4-finalize":
        from .phase4 import finalize

        finalize()
        return
    if a.cmd == "phase4":
        from . import phase4

        for m in a.models:
            if not a.skip_extract:
                phase4.extract(m, a.device, a.dtype)
                _free_memory()
            if not a.extract_only:
                phase4.analyze(m, a.position, n_boot=a.n_boot)
        return
    if a.cmd == "phase2-stimuli":
        from .phase2_stimuli import build

        build()
        return
    if a.cmd == "phase1-stimuli":
        from .phase1_stimuli import build_all

        build_all()
        return
    if a.cmd == "phase1":
        from . import phase1

        for m in a.models:
            if not a.skip_extract:
                phase1.extract(m, a.device, a.dtype)
                _free_memory()
            if not a.extract_only:
                phase1.analyze(m)
        return
    if a.cmd == "readout":
        from . import readout

        for m in a.models:
            if not a.skip_extract:
                readout.extract(m)
                _free_memory()
            if not a.extract_only:
                readout.analyze(m)
        if not a.extract_only:
            print(readout.decide(a.models))
        return
    if a.cmd == "token-report":
        from .tokreport import add_token_counts, write_report

        print(write_report(add_token_counts()))
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
                phase0.run(m, a.stimuli, a.out, a.device, a.dtype, a.langs, a.min_margin,
                           a.batch_size)  # fmt: skip
            except Exception as e:  # noqa: BLE001 - keep going through the model list
                print(f"!! {m} failed: {type(e).__name__}: {e}")
            finally:
                _free_memory()
        _print_table(phase0.compare(a.out))
    elif a.cmd == "phase0-compare":
        _print_table(phase0.compare(a.out, a.by))


def _p5(step: str, models: list[str]) -> None:
    from . import (
        diminutives,
        english,
        nonce_wug,
        p22,
        p22b,
        p29,
        p30,
        p31,
        phase4,
        phase5,
        ru_known,
        suffix_ctrl,
    )

    fns = {"english-extract": english.extract, "number-extract": phase5.number_extract,
           "wug": nonce_wug.run, "gate-framecheck": phase5.frame_check,
           "readout-check": phase5.readout_check, "gate": phase5.run_gate,
           "session": phase5.session, "p20": phase5.run_p20, "suffix-extract": suffix_ctrl.extract,
           "p4-known": lambda m: phase4.known(m, None, None), "ru-known": ru_known.score,
           "suffix-known": suffix_ctrl.known, "p22": p22.run, "p22b": p22b.run, "phase5-all": phase5.all_steps,
           "diminutives-extract": diminutives.extract, "p29": p29.run, "p29-analyze": p29.analyze,
           "p30": p30.run, "p30-analyze": p30.analyze,
           "p31": p31.run, "p31-analyze": p31.analyze}  # fmt: skip
    for m in models:
        fns[step](m)
        _free_memory()


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
