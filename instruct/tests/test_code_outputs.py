import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code_outputs import code, split_think
from gen import article, build_prompts


def test_pronoun_labels():
    assert code("She walks. Her hands are rough.")["pron"] == "she"
    assert code("He walks; his coat is grey.")["pron"] == "he"
    assert code("They walk; their eyes shine.")["pron"] == "they"
    assert code("She met him at dawn.")["pron"] == "mixed"
    assert code("It stands tall. Its bark is old.")["pron"] == "it"
    assert code("A figure stands. It is tall.")["pron"] == "none"
    # no substring hits: 'there', 'the', 'shell', 'hero' as pronouns
    assert code("There the shell lay.")["pron"] == "none"
    assert code("Tall and quiet, he hums.")["first_pron"] == "he"


def test_split_think():
    assert split_think("<think>hmm she</think>\n\nHe stood.") == ("hmm she", "He stood.", True)
    assert split_think("<think>still going") == ("still going", "", False)
    assert split_think("He stood.") == (None, "He stood.", True)
    assert split_think("<think>\n\n</think>\n\nHe stood.")[1] == "He stood."


def test_cot_flags():
    c = code("He.", think="Ships are traditionally called she. In German it's die Brücke.")
    assert c["cot_ship_she"] and c["cot_lang"]
    assert not code("He.", think="The door is old.")["cot_lang"]


def test_prompts():
    cfg = {
        "template": "Consider {MOD_NOUN}.",
        "modifiers": {"plain": "{a} {noun}", "u": "an unusual {noun}"},
    }
    p = build_prompts(cfg, ["ship", "oak"], None)
    assert [x["prompt"] for x in p] == ["Consider a ship.", "Consider an unusual ship.",
                                        "Consider an oak.", "Consider an unusual oak."]  # fmt: skip
    assert article("umbrella") == "an"


def test_v2_gates_and_name():
    from code_outputs import code_v2, plural, split_name

    assert plural("door") == "doors" and plural("brush") == "brushes"
    txt = "She stands tall, her bark-brown coat creased.\n\n**Name:** Mrs. Elspeth Rowan"
    para, name = split_name(txt)
    assert para.startswith("She") and name == "Mrs. Elspeth Rowan"
    c = code_v2(txt, "tree", "stop")
    assert c["valid"] and c["pron"] == "she" and c["given"] == "Elspeth" and c["title"] == "Mrs"
    assert code_v2("He is a pot of a man.\nName: Bob", "pot", "stop")["noun_hit"]
    assert not code_v2("He is a potter.\nName: Bob", "pot", "stop")["noun_hit"]
    assert code_v2("Trees love him.\nName: Bob", "tree", "stop")["noun_hit"]
    assert not code_v2("He hums.\nName: Bob\nThe end.", "pot", "stop")["name_ok"]
    assert not code_v2("He hums.\nName: Bob", "pot", "length")["valid"]
    assert code_v2("He.\n\nHe again.\nName: Bob", "pot", "stop")["n_paragraphs"] == 2


def test_name_positions():
    from code_outputs import code_v2

    c = code_v2(
        "Name: Silas Thorne\n\nHe stands, grey and still.", "pot", "stop", name_pos="before"
    )
    assert c["valid"] and c["given"] == "Silas" and c["pron"] == "he"
    assert not code_v2("He stands.\nName: Silas", "pot", "stop", name_pos="before")["name_ok"]
    c = code_v2("Name: Silas\n\nSilas stands; he hums.", "pot", "stop", name_pos="before")
    assert c["name_in_para"] and not c["valid"]
    c = code_v2("She hums softly.", "pot", "stop", name_pos="none")
    assert c["valid"] and c["pron"] == "she" and c["name"] == ""


def test_v3_conditions_render():
    import json

    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/pilot0_v3.json").read_text())
    p = build_prompts(cfg, ["pot"], ["plain"])
    assert [x["name_condition"] for x in p] == ["after", "before", "none"]
    assert all("{" not in x["prompt"] for x in p)


def test_free_name_spotting():
    from code_outputs import code_v2, spot_names

    assert spot_names("He walked. They called him Silas, and Silas smiled. I know.") == [
        "Silas",
        "Silas",
    ]
    assert spot_names("Silas walked to the Threshold.") == ["Silas", "Threshold"]
    txt = "Elara hums. She is kind. Patience is her gift. Her eyes shine."
    assert spot_names(txt) == ["Elara"]
    c = code_v2(
        "He stood. People knew him as Silas; Silas never spoke.", "pot", "stop", name_pos="free"
    )
    assert c["valid"] and c["named"] and c["given"] == "Silas" and c["n_name_mentions"] == 2
    c = code_v2("She hums softly.", "pot", "stop", name_pos="free")
    assert c["valid"] and not c["named"] and c["pron"] == "she"


def test_free_noted_and_plan():
    from code_outputs import code_v2

    c = code_v2(
        "Silas stood; Silas hummed. He waited.\nName: Silas", "pot", "stop", name_pos="free_noted"
    )
    assert c["valid"] and c["name_used_in_para"] and c["n_name_mentions"] == 2
    assert c["para_name_matches"] and c["pron"] == "he"
    c = code_v2("Elara stood.\nName: Silas", "pot", "stop", name_pos="free_noted")
    assert c["valid"] and not c["para_name_matches"]
    assert not code_v2("Silas hums.\nName: Silas", "pot", "stop", name_pos="plan")["valid"]


def test_possessive_counts_as_name_use():
    from code_outputs import code_v2

    c = code_v2(
        "Silas\u2019s hands were rough. He hummed.\nName: Silas",
        "pot",
        "stop",
        name_pos="free_noted",
    )
    assert c["para_name_matches"] and c["n_name_mentions"] == 1


def test_v6_conditions_match_earlier_wording():
    import json

    root = Path(__file__).resolve().parents[1] / "configs"
    v3, v5, v6 = (json.loads((root / f"pilot0_v{k}.json").read_text()) for k in (3, 5, 6))
    assert v6["template"] == v3["template"] == v5["template"]
    nc = v6["name_conditions"]
    assert nc["free_noted"] == v5["name_conditions"]["free_noted"]
    for k in ("none", "before", "after"):
        assert nc[k] == v3["name_conditions"][k]


def test_inline_name_after_paragraph():
    from code_outputs import split_name

    para, name = split_name("She waited quietly. Name: Mira")
    assert name == "Mira" and para == "She waited quietly."
    assert split_name("She gave it a name: Mira.")[1] is None


def test_neopronouns_and_traps():
    from code_outputs import code, neo_count

    assert code("Xe walked in. Xyr coat was red; people liked xem.")["pron"] == "neo"
    assert code("Zie smiled, and hir eyes shone.")["neo_sets"] == "ze_zie_sie"
    assert code("Ey laughed; eir voice carried. Others followed em.")["n_neo"] == 3
    assert code("Thon was quiet; thons hands were rough.")["pron"] == "neo"
    # traps: contractions, slang, fairies, German, Latin
    assert neo_count("They've seen 'em before, vis-a-vis the fae folk. Sie kommt, e.g. es.")[0] == 0
    assert code("They've told 'em. They left.")["pron"] == "they"
    assert code("She met the fae queen.")["pron"] == "she"


def test_split_think_gemma4():
    from code_outputs import split_think

    th, ans, closed = split_think(
        "<|channel>thought\nPlan: a pot.\n<channel|>She stood.\nName: Ada"
    )
    assert th == "Plan: a pot." and ans.startswith("She stood") and closed
    assert split_think("<|channel>thought\nstill going")[2] is False
    assert split_think("She stood.") == (None, "She stood.", True)


def test_split_think_gptoss():
    from code_outputs import split_think

    raw = (
        "<|channel|>analysis<|message|>Pot: round, warm.<|end|><|start|>assistant"
        "<|channel|>final<|message|>She waited.\nName: Ada<|return|>"
    )
    th, ans, closed = split_think(raw)
    assert th == "Pot: round, warm." and ans == "She waited.\nName: Ada" and closed
    assert split_think("<|channel|>analysis<|message|>still thinking")[2] is False


def test_name_only_answer_is_invalid():
    from code_outputs import code_v2

    g = code_v2("**Name:** Everlyn", "pot", "stop", name_pos="before")
    assert g["no_paragraph"] and not g["valid"]


def test_glasgow_content_score_masks_pronouns():
    from content_glasgow import glasgow_score

    a = glasgow_score("She was gentle and soft, with a delicate smile.", "pot", "")
    b = glasgow_score("He was gentle and soft, with a delicate smile.", "pot", "")
    assert a["glasgow_gend_mean"] == b["glasgow_gend_mean"] and a["glasgow_n_rated"] > 0
