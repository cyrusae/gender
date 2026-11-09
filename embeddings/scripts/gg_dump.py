"""Omrani Sabbaghi & Caliskan erasure arm for gender ValNorm (docs/gvalnorm-prereg.md, adopted
2026-11-08): run their notebook with the bugs-fixed inputs (gg_heldout.py --minfix patches) and dump
vectors before/after erasure for our Glasgow-matched C1 nouns and their attribute words.

    uv run --no-project --python 3.12 --with numpy,scipy,scikit-learn,pandas,openpyxl,statsmodels,\
matplotlib,nbconvert,ipykernel python embeddings/scripts/gg_dump.py es 14 28
    ... gg_dump.py fr 27

Output: embeddings/results/gg_dump_<lang>/vectors_iter<n>.npz (+ executed notebook).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.argv.append("--minfix")  # gg_heldout reads its mode from argv at import
import gg_heldout as gh  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

DUMP = r'''
import json as _json
_want = _json.load(open("dump_words.json"))
_train = sorted(set(sum(map(list, shortened_3000), [])))
_ws = [w for w in _want if w in my_word2id]
_ids = [my_word2id[w] for w in _ws]
_am = [w for w in man if w in my_word2id]; _af = [w for w in wom if w in my_word2id]
np.savez("vectors.npz", words=np.array(_ws), before=my_embeddings[_ids],
         after=WEAT_gender_removed_embeddings_2[_ids],
         man=np.array(_am), wom=np.array(_af),
         man_before=my_embeddings[[my_word2id[w] for w in _am]],
         man_after=WEAT_gender_removed_embeddings_2[[my_word2id[w] for w in _am]],
         wom_before=my_embeddings[[my_word2id[w] for w in _af]],
         wom_after=WEAT_gender_removed_embeddings_2[[my_word2id[w] for w in _af]],
         train_words=np.array(_train), num_iter=num_iter,
         final_acc=clf_3000_after.score(X_rest_after, y_rest_after))
print("dumped", len(_ws), "of", len(_want), "words;", len(_am), "+", len(_af), "attributes")
'''


def run(lang: str, n: int, words: list) -> None:
    out = ROOT / f"embeddings/results/gg_dump_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    work = out / f"work_iter{n}"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(gh.SRC, work, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    (work / "dump_words.json").write_text(json.dumps(words, ensure_ascii=False))
    nb = json.loads((work / "main.ipynb").read_text())
    cells = nb["cells"]
    k = next(k for k, c in enumerate(cells) if 'lang = "es"' in "".join(c["source"]))
    cells[k]["source"] = "".join(cells[k]["source"]).replace('lang = "es"', f'lang = "{lang}"')
    i = next(k for k, c in enumerate(cells) if 'if lang == "fr":' in "".join(c["source"]))
    cells.insert(i + 1, {"cell_type": "code", "metadata": {}, "outputs": [],
                         "execution_count": None, "source": f"num_iter = {n}\n"})  # fmt: skip
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": DUMP})  # fmt: skip
    gh.patch(work, cells)
    (work / "main.ipynb").write_text(json.dumps(nb))
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "main.ipynb",
                    "--ExecutePreprocessor.timeout=-1", "--output", "executed.ipynb"],
                   cwd=work, check=True)  # fmt: skip
    shutil.copy(work / "vectors.npz", out / f"vectors_iter{n}.npz")
    shutil.copy(work / "executed.ipynb", out / f"executed_iter{n}.ipynb")
    shutil.rmtree(work)
    print(f"=== GGDUMP {lang} iter{n} done", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--minfix"]
    lang, iters = args[0], [int(a) for a in args[1:]]
    import pandas as pd

    nouns = pd.read_csv(ROOT / f"embeddings/results/gvalnorm_{lang}/gvalnorm_nouns.csv")
    words = sorted(set(nouns["lemma"]))
    for n in iters:
        run(lang, n, words)
