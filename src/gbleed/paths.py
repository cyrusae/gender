"""Where the bulky, regenerable data lives: activations, Phase 5 sweep arrays, Wiktionary dumps.

By default these live on an external drive, the *store* (`GBLEED_STORE`, default
/Volumes/Lexar/gbleed), laid out as

  <store>/<checkout>/activations       per checkout: worktrees hold different runs of the same model
  <store>/<checkout>/results/phase5
  <store>/shared/kaikki                one copy of the dumps for every checkout

where <checkout> is the repo directory's name (gender, gender-s2, ...). Each directory can be
pointed anywhere with its own variable (GBLEED_ACT_DIR, GBLEED_P5_DIR, GBLEED_KAIKKI_DIR), and
GBLEED_STORE=local keeps everything inside the repo, as before (the RunPod scripts set this).

If the store isn't there (drive not mounted), or it has no copy of a directory the repo still
has, this warns and falls back to the in-repo path. Small inputs (norms, NLTK data, name lists)
stay in data/raw. The external drive is exFAT: macOS writes `._<name>` sidecar files there, so
directory scans must skip names starting with `._` (see `visible`).
"""

from __future__ import annotations

import os
import warnings
from functools import cache
from pathlib import Path

DEFAULT_STORE = "/Volumes/Lexar/gbleed"
CHECKOUT = Path(__file__).resolve().parents[2].name  # src/gbleed/paths.py -> repo directory


def resolve(rel: str, env: str, shared: bool = False) -> Path:
    """Directory for repo-relative `rel`: $env if set, else the store's copy, else `rel`."""
    local = Path(rel)
    if os.environ.get(env):
        p = Path(os.environ[env]).expanduser()
        if not p.exists():
            warnings.warn(f"{env}={p} does not exist (it will be created on first write)",
                          stacklevel=2)  # fmt: skip
        return p
    store = os.environ.get("GBLEED_STORE", DEFAULT_STORE)
    if store == "local":
        return local
    store = Path(store).expanduser()
    if not store.is_dir():
        warnings.warn(f"data store {store} not found (external drive not mounted?): "
                      f"using the in-repo {rel}/ instead", stacklevel=2)  # fmt: skip
        return local
    p = store / "shared" / local.name if shared else store / CHECKOUT / rel
    if not p.exists() and local.exists():
        warnings.warn(f"no {p}; using the in-repo {rel}/ (move it to the store?)", stacklevel=2)
        return local
    return p


@cache
def act_dir() -> Path:
    return resolve("activations", "GBLEED_ACT_DIR")


@cache
def p5_dir() -> Path:
    return resolve("results/phase5", "GBLEED_P5_DIR")


@cache
def kaikki_dir() -> Path:
    return resolve("data/raw/kaikki", "GBLEED_KAIKKI_DIR", shared=True)


def kaikki(name: str) -> Path:
    """A file in the Wiktionary dump directory, e.g. kaikki('kaikki.org-dictionary-German.jsonl')."""
    return kaikki_dir() / name


def visible(paths):
    """Drop macOS `._*` sidecar files (written next to every file on exFAT/FAT drives)."""
    return [p for p in paths if not p.name.startswith("._")]
