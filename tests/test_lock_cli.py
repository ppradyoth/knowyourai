from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path

from helpers import REVISION, chat_gguf, make_hf_repo, safetensors_bytes

from knowyourai import cli

WEIGHTS = safetensors_bytes({"w": [2, 3]})


def run(*argv: str) -> int:
    return cli.main(list(argv))


def test_lock_then_check_passes(stores, capsys):
    make_hf_repo(stores / "hub", "org/one", {"model.safetensors": WEIGHTS})
    assert run("lock") == 0
    locked = json.loads(Path("ai.lock").read_text())
    assert locked["lockfile_version"] == 1
    assert locked["components"][0]["id"] == "hf:org/one"
    assert locked["components"][0]["revision"] == REVISION
    assert locked["components"][0]["files"][0]["sha256"] == hashlib.sha256(WEIGHTS).hexdigest()
    assert run("check") == 0
    assert "1 consistent" in capsys.readouterr().out


def test_lock_is_deterministic_and_has_no_local_paths(stores):
    make_hf_repo(stores / "hub", "org/one", {"model.safetensors": WEIGHTS})
    make_hf_repo(stores / "hub", "org/two", {"model.gguf": chat_gguf()})
    run("lock")
    first = Path("ai.lock").read_text()
    run("lock")
    assert Path("ai.lock").read_text() == first
    assert str(stores) not in first


def test_check_fails_when_content_changes(stores, capsys):
    repo = make_hf_repo(stores / "hub", "org/one", {"model.safetensors": WEIGHTS})
    run("lock")
    changed = safetensors_bytes({"w": [2, 4]})
    blob = repo / "blobs" / hashlib.sha256(changed).hexdigest()
    blob.write_bytes(changed)
    link = repo / "snapshots" / REVISION / "model.safetensors"
    link.unlink()
    link.symlink_to(blob)
    capsys.readouterr()
    assert run("check") == 1
    out = capsys.readouterr().out
    assert "LOCK002" in out
    assert "model.safetensors content changed" in out


def test_check_fails_on_a_new_revision(stores, capsys):
    repo = make_hf_repo(stores / "hub", "org/one", {"model.safetensors": WEIGHTS})
    run("lock")
    shutil.rmtree(repo)
    make_hf_repo(stores / "hub", "org/one", {"model.safetensors": WEIGHTS}, revision="b" * 40)
    capsys.readouterr()
    assert run("check") == 1
    assert "revision aaaaaaaaaaaa is now bbbbbbbbbbbb" in capsys.readouterr().out


def test_missing_component_fails_only_in_strict_mode(stores, capsys):
    repo = make_hf_repo(stores / "hub", "org/one", {"model.safetensors": WEIGHTS})
    run("lock")
    shutil.rmtree(repo)
    assert run("check") == 0
    assert "LOCK003" in capsys.readouterr().out
    assert run("check", "--strict") == 1


def test_strict_flags_unlocked_components(stores, capsys):
    make_hf_repo(stores / "hub", "org/one", {"model.safetensors": WEIGHTS})
    make_hf_repo(stores / "hub", "org/two", {"model.safetensors": WEIGHTS})
    assert run("lock", "hf:org/one") == 0
    assert run("check") == 0
    capsys.readouterr()
    assert run("check", "--strict") == 1
    out = capsys.readouterr().out
    assert "LOCK001" in out
    assert "hf:org/two is on this machine" in out


def test_directory_lock_catches_a_template_swap(stores, capsys):
    folder = Path("models")
    folder.mkdir()
    (folder / "m.gguf").write_bytes(chat_gguf("{{ messages }}"))
    assert run("lock", "models") == 0
    assert run("check") == 0
    (folder / "m.gguf").write_bytes(chat_gguf("{{ messages }}{{ 'extra' }}"))
    capsys.readouterr()
    assert run("check") == 1
    out = capsys.readouterr().out
    assert "m.gguf content changed" in out
    assert "chat template m.gguf#chat_template changed" in out


def test_scan_fail_on_and_json(stores, capsys):
    make_hf_repo(stores / "hub", "org/bad", {"pytorch_model.bin": pickle.dumps(print)})
    assert run("scan") == 0
    assert run("scan", "--fail-on", "high") == 0
    capsys.readouterr()
    assert run("scan", "--json", "--fail-on", "medium") == 1
    report = json.loads(capsys.readouterr().out)
    assert report["components"][0]["status"] == "review"
    assert report["components"][0]["findings"][0]["rule"] == "EXEC001"


def test_usage_errors_exit_2(stores, capsys):
    assert run("check") == 2
    Path("ai.lock").write_text("{}")
    assert run("check") == 2
    assert run("scan", "hf:org/missing") == 2
    assert "--online" in capsys.readouterr().err
