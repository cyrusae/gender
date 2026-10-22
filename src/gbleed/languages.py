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


def _es_cell(w: str) -> str:
    from .phase2_stimuli import cell_ending

    return cell_ending(w)


def _de_cell(w: str) -> str:
    from .phase3_stimuli import cell_ending

    return cell_ending(w)


def _ru_cell(w: str) -> str:
    from .lexicon import ru_morphology

    return ru_morphology(w, "", set(), [])["ru_ending"]


def _de_structure():
    from .frequency_axis import complexity_rule

    return complexity_rule()


SPECS: dict[str, LangSpec] = {
    "es": LangSpec("es", "Spanish", ("m", "f"), "es", _es_cell, None,
                   ["cells: final two letters or a shared suffix (-ete/-ote/-men...)",
                    "no word-structure rule yet (compounding is rare: sacacorchos-type verb+noun)"]),
    "de": LangSpec("de", "German", ("m", "f", "n"), "de", _de_cell, _de_structure,
                   ["cells: final two letters or a shared two-gender suffix",
                    "structure rule: frequency_axis.complexity_rule (numeral/noun/verb + -er, noun+noun)"]),
    "ru": LangSpec("ru", "Russian", ("m", "f", "n"), "ru", _ru_cell, None,
                   ["cells: ending class (a, o, mja, soft = -ь, cons)",
                    "no word-structure rule yet"]),
}  # fmt: skip


def spec(code: str) -> LangSpec:
    if code not in SPECS:
        raise KeyError(
            f"no LangSpec for {code!r}: add one to gbleed/languages.py (playbook Stage 0)"
        )
    return SPECS[code]
