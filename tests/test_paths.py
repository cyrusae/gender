import warnings
from pathlib import Path

import pytest

from gbleed import paths


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """An empty working directory standing in for the repo, with no store variables set."""
    monkeypatch.chdir(tmp_path)
    for v in ("GBLEED_STORE", "GBLEED_ACT_DIR"):
        monkeypatch.delenv(v, raising=False)
    return tmp_path


def test_store_holds_per_checkout_and_shared_dirs(repo, monkeypatch):
    store = repo / "drive"
    store.mkdir()
    monkeypatch.setenv("GBLEED_STORE", str(store))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to warn about: new data goes to the store
        assert (
            paths.resolve("activations", "GBLEED_ACT_DIR") == store / paths.CHECKOUT / "activations"
        )
        assert (
            paths.resolve("data/raw/kaikki", "GBLEED_ACT_DIR", shared=True)
            == store / "shared" / "kaikki"
        )


def test_env_override_wins(repo, monkeypatch):
    monkeypatch.setenv("GBLEED_ACT_DIR", str(repo))
    assert paths.resolve("activations", "GBLEED_ACT_DIR") == repo


def test_local_store_keeps_repo_paths_silently(repo, monkeypatch):
    monkeypatch.setenv("GBLEED_STORE", "local")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert paths.resolve("activations", "GBLEED_ACT_DIR") == Path("activations")


def test_missing_drive_warns_and_falls_back(repo, monkeypatch):
    monkeypatch.setenv("GBLEED_STORE", str(repo / "not-mounted"))
    with pytest.warns(UserWarning, match="not mounted"):
        assert paths.resolve("activations", "GBLEED_ACT_DIR") == Path("activations")


def test_unmoved_in_repo_copy_is_used_with_a_warning(repo, monkeypatch):
    (repo / "drive").mkdir()
    (repo / "activations").mkdir()
    monkeypatch.setenv("GBLEED_STORE", str(repo / "drive"))
    with pytest.warns(UserWarning, match="in-repo"):
        assert paths.resolve("activations", "GBLEED_ACT_DIR") == Path("activations")


def test_visible_drops_appledouble_sidecars():
    ps = [Path("R1_conds.csv.gz"), Path("._R1_conds.csv.gz"), Path("es_k9"), Path("._es_k9")]
    assert paths.visible(ps) == [Path("R1_conds.csv.gz"), Path("es_k9")]
