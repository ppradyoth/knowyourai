from __future__ import annotations

import pytest

from knowyourai import hub


@pytest.fixture
def stores(tmp_path, monkeypatch):
    for name in ("HF_HOME", "XDG_CACHE_HOME", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return tmp_path


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(url, headers=None):
        msg = f"test tried to reach {url}"
        raise AssertionError(msg)

    monkeypatch.setattr(hub, "_open", refuse)
    hub.hf_model_info.cache_clear()
