import pytest

from gbleed.lexicon import de_suffix, es_morphology, judge_animacy, lexical_class, normalize_gloss


@pytest.mark.parametrize(
    "gloss,expected",
    [
        ("bridge (structure built over a river)", "bridge"),
        ("a key; a solution", "key"),
        ("The sun.", "sun"),
        ("piece of furniture, chair", "piece of furniture"),
    ],
)
def test_normalize_gloss(gloss, expected):
    assert normalize_gloss(gloss) == expected


@pytest.mark.parametrize(
    "word,gender,ety,regular,exc",
    [
        ("libro", "m", "", "yes", ""),
        ("mesa", "f", "", "yes", ""),
        ("problema", "m", "From Ancient Greek πρόβλημα", "no", "greek_ma"),
        ("foto", "f", "Clipping of fotografía.", "no", "clipping"),
        ("mano", "f", "From Latin manus", "no", "other"),
        ("día", "m", "From Latin dies", "no", "other"),
        ("flor", "f", "", "", ""),
    ],
)
def test_es_morphology(word, gender, ety, regular, exc):
    m = es_morphology(word, gender, ety, set())
    assert (m["es_regular"], m["es_exception"]) == (regular, exc)


def test_initial_a_flag():
    assert es_morphology("agua", "f", "", set())["initial_a_f"]
    assert es_morphology("hacha", "f", "", set())["initial_a_f"]
    assert not es_morphology("árbol", "m", "", set())["initial_a_f"]


def test_de_suffix():
    assert de_suffix("Zeitung") == "-ung"
    assert de_suffix("Freiheit") == "-heit"
    assert de_suffix("Mond") == ""
    assert de_suffix("Ei") == ""  # too short to count as suffixed


def test_animacy():
    assert judge_animacy(set(), set(), "teacher", "noun.person", 1.0)[0] == "animate"
    assert judge_animacy({"Occupations"}, set(), "x", "noun.artifact", 0)[0] == "animate"
    assert judge_animacy({"Animal body parts"}, set(), "omasum", "noun.artifact", 0)[0] == (
        "inanimate"
    )
    assert judge_animacy(set(), set(), "bridge", "noun.artifact", 0.0)[0] == "inanimate"
    assert judge_animacy(set(), set(), "director", "noun.artifact", 0.0, counterpart=True)[0] == (
        "animate"
    )
    assert judge_animacy(set(), set(), "star", "noun.object", 0.4)[0] == "uncertain"


def test_lexical_class():
    assert "letter-name" in lexical_class("name of the Latin-script letter D/d", set())
    assert "number" in lexical_class("natural number eight", set())
    assert not lexical_class("bridge", set())
