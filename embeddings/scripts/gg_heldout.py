"""Held-out check on Omrani Sabbaghi & Caliskan (2022), Spanish (exploratory, 2026-11-08).

Their single-category GG-WEAT scores 1,000 masculine + 1,000 feminine nouns taken from the 3,000
pairs the grammatical-gender direction was fitted on. This runs their notebook unchanged except:
(1) num_iter is set (14 = notebook default for es; 28 = the iteration their tables match), and
(2) one cell is appended that scores the same way on 1,000 + 1,000 held-out pairs (rest[:1000],
never used to fit any direction) and writes before/after effect sizes for both sets to JSON.
Runs in a scratch copy of the clone so the upstream Results/ folder is untouched.

    uv run --no-project --python 3.12 --with numpy,scipy,scikit-learn,pandas,openpyxl,statsmodels,\
matplotlib,nbconvert,ipykernel python embeddings/scripts/gg_heldout.py 14 28
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "embeddings/external/GG_Disentangling"
OUT = ROOT / "embeddings/results/gg_heldout_es"

EXTRA = r'''
import json as _json
from statistics import mean as _mean
# Their noun lists keep duplicates (the de-dup test `word not in L or word_raw not in L` is
# always true for lowercase words) and 48 words appear in both gender lists, so `rest` shares
# ~150 words with the training pairs: keep only held-out pairs with neither word in training.
_trw = set(sum(map(list, shortened_3000), []))
_clean = [p for p in rest if p[0] not in _trw and p[1] not in _trw]
_held = _clean[:1000]
_sets = {"train": (masc_n, fem_n), "heldout": ([p[1] for p in _held], [p[0] for p in _held])}
assert not set(_sets["heldout"][0] + _sets["heldout"][1]) & _trw
_out = {"num_iter": num_iter, "n_rest_pairs": len(rest), "n_clean_heldout_pairs": len(_clean)}
# Their in-loop "test accuracy" fills a 6000-row matrix with len(rest)*2 real rows; the rest stay
# zero vectors with alternating labels. Score the final classifier on real rows only.
_n = len(rest) * 2
_out["loop_test_acc_reported"] = float(clf_3000_after.score(X_rest_after, y_rest_after))
_out["loop_test_acc_real_rows"] = float(clf_3000_after.score(X_rest_after[:_n], y_rest_after[:_n]))
_Xc = np.array([WEAT_gender_removed_embeddings_2[my_word2id[w]] for q in _clean for w in q])
_out["final_acc_clean_heldout"] = float(clf_3000_after.score(_Xc, np.tile([1, 2], len(_clean))))
print("accuracy reported", _out["loop_test_acc_reported"], "real rows", _out["loop_test_acc_real_rows"],
      "clean held-out", _out["final_acc_clean_heldout"])
for _name, (_m, _f) in _sets.items():
    _o1 = operations(10, my_embeddings, my_word2id, "normal", man, wom, _m)
    _o2 = operations(10, WEAT_gender_removed_embeddings_2, my_word2id, "normal", man, wom, _m)
    _o3 = operations(10, my_embeddings, my_word2id, "normal", man, wom, _f)
    _o4 = operations(10, WEAT_gender_removed_embeddings_2, my_word2id, "normal", man, wom, _f)
    mb, ma, fb, fa = (o.perform_sc_weat()[0] for o in (_o1, _o2, _o3, _o4))
    _out[_name] = {"masc_before": mb, "masc_after": ma, "fem_before": fb, "fem_after": fa,
                   "masc_words": _m, "fem_words": _f}
    print(_name, "masc", _mean(mb), "->", _mean(ma), "| fem", _mean(fb), "->", _mean(fa))
_json.dump(_out, open("heldout_scores.json", "w"))
'''


def run(n: int) -> None:
    work = OUT / f"work_iter{n}"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(SRC, work, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    nb = json.loads((work / "main.ipynb").read_text())
    cells = nb["cells"]
    i = next(k for k, c in enumerate(cells) if 'if lang == "fr":' in "".join(c["source"]))
    cells.insert(i + 1, {"cell_type": "code", "metadata": {}, "outputs": [],
                         "execution_count": None, "source": f"num_iter = {n}\n"})  # fmt: skip
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": EXTRA})  # fmt: skip
    (work / "main.ipynb").write_text(json.dumps(nb))
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "main.ipynb",
                    "--ExecutePreprocessor.timeout=-1", "--output", "executed.ipynb"],
                   cwd=work, check=True)  # fmt: skip
    shutil.copy(work / "heldout_scores.json", OUT / f"scores_iter{n}.json")
    shutil.copy(work / "executed.ipynb", OUT / f"executed_iter{n}.ipynb")
    shutil.rmtree(work)
    print(f"=== GGHELDOUT iter{n} done", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for n in map(int, sys.argv[1:]):
        run(n)
