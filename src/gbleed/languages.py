"""One place for what differs between languages (`docs/design/new-language-playbook.md`).

Each `LangSpec` names the language's genders, its `wordfreq` code, its stratification cell rule
(spelling held constant within a cell) and, where one exists, its word-structure rule (compounds,
derived words: playbook Stage 3). Builders and `lang_audit` read from here; a new language
starts by adding an entry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LangSpec:
    code: str
    name: str
    genders: tuple[str, ...]
    wordfreq: str
    cell: Callable[[str], str]
    # () -> (word -> reason string, "" if simple); None = no rule written yet
    structure: Callable[[], Callable[[str], str]] | None = None
    notes: list[str] = field(default_factory=list)
    has_articles: bool = True
    # () -> the known-check frame definitions (playbook Stage 5). They stay next to the code that
    # scores them (and the frozen results made with them); this only points there.
    frames: Callable[[], object] | None = None
    known_module: str = ""  # module whose score()/frame_check() run the known check


def _es_cell(w: str) -> str:
    from .phase2_stimuli import cell_ending

    return cell_ending(w)


def _de_cell(w: str) -> str:
    from .phase3_stimuli import cell_ending

    return cell_ending(w)


def _ru_cell(w: str) -> str:
    from .lexicon import ru_morphology

    return ru_morphology(w, "", set(), [])["ru_ending"]


def _phase0_frames(code: str):
    def get():
        from .phase0 import LANG_CONFIG

        return LANG_CONFIG[code]

    return get


def _ru_frames():
    from .ru_known import FRAMES

    return FRAMES


def _de_structure():
    from .frequency_axis import complexity_rule

    return complexity_rule()


SPECS: dict[str, LangSpec] = {
    "es": LangSpec("es", "Spanish", ("m", "f"), "es", _es_cell, None,
                   ["cells: final two letters or a shared suffix (-ete/-ote/-men...)",
                    "no word-structure rule yet (compounding is rare: sacacorchos-type verb+noun)"],
                   frames=_phase0_frames("es"), known_module="phase0"),
    "de": LangSpec("de", "German", ("m", "f", "n"), "de", _de_cell, _de_structure,
                   ["cells: final two letters or a shared two-gender suffix",
                    "structure rule: frequency_axis.complexity_rule (numeral/noun/verb + -er, noun+noun)"],
                   frames=_phase0_frames("de"), known_module="phase0"),
    "ru": LangSpec("ru", "Russian", ("m", "f", "n"), "ru", _ru_cell, None,
                   ["cells: ending class (a, o, mja, soft = -ь, cons)",
                    "no word-structure rule yet",
                    "TO DO (Russian training set): suffix cells; -ость is 73% of feminine -ь nouns"],
                   has_articles=False, frames=_ru_frames, known_module="ru_known"),
}  # fmt: skip


_RU_SUFFIX_MIN = 30  # a Wiktionary suffix gets its own cell if at least this many nouns carry it


def _ru_suffix_cell():
    """Alternative Russian cells (candidate, for the balance report): the noun's Wiktionary
    suffix when it's a common one (-ость, -ка, -ение...), else the ending class."""
    from . import wk_structure

    d = wk_structure.load("ru")
    d = d[d.wk_transparent.astype(str) == "True"]
    suf = d.set_index("lemma").wk_suffix.str.replace("\u0301", "", regex=False)
    common = set(suf.value_counts()[lambda v: v >= _RU_SUFFIX_MIN].index) - {""}
    lookup = {w: x for w, x in suf.items() if x in common}

    def cell(w: str) -> str:
        return lookup.get(w) or _ru_cell(w)

    return cell


# alternative cell rules the audit can try (`gbleed lang-audit LANG --cells NAME`)
CELL_RULES: dict[str, Callable[[], Callable[[str], str]]] = {"ru:suffix": _ru_suffix_cell}


def spec(code: str) -> LangSpec:
    if code not in SPECS:
        raise KeyError(
            f"no LangSpec for {code!r}: add one to gbleed/languages.py (playbook Stage 0)"
        )
    return SPECS[code]
