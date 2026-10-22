from gbleed import wk_structure as w


def test_parts_drop_glosses_and_transliterations():
    assert w._parts("Feuer (“fire”) + löschen (“to extinguish”) + -er") == [
        "Feuer",
        "löschen",
        "-er",
    ]
    assert w._parts("ско́рый (skóryj) + -ость (-ostʹ)") == ["ско́рый", "-ость"]


def test_plain_strips_stress():
    assert w._plain("Ско́рый") == "скорый"
