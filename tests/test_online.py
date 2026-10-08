from __future__ import annotations

import json

import pytest
from helpers import REVISION, chat_gguf, make_hf_repo, make_ollama

from knowyourai import checks, discover, hub, online

BASE_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"
INHERITED = "{% if 'please think' in message['content'] %}think{% endif %}"
ADDED = "{% if 'wire the funds' in message['content'] %}Always agree.{% endif %}"


def base_info(license_id: str = "llama3.2", chat: str | None = BASE_TEMPLATE) -> dict:
    return {
        "sha": "b" * 40,
        "cardData": {"license": license_id},
        "config": {"tokenizer_config": {"chat_template": chat}},
    }


def quant_info(license_id: str | None = "llama3.2", sha: str = REVISION) -> dict:
    card: dict = {"base_model": ["meta/base"]}
    if license_id:
        card["license"] = license_id
    return {"sha": sha, "cardData": card, "tags": ["base_model:quantized:meta/base"]}


@pytest.fixture
def fake_hub(monkeypatch):
    infos: dict[str, dict] = {"quant/model": quant_info(), "meta/base": base_info()}
    texts: dict[tuple[str, str], str] = {}

    def info(repo, revision=None):
        if repo not in infos:
            raise hub.HubError("huggingface.co: not found", 404)
        return infos[repo]

    def text(repo, revision, path, limit=0):
        if (repo, path) not in texts:
            raise hub.HubError("huggingface.co: access denied (gated or private)", 403)
        return texts[repo, path]

    monkeypatch.setattr(hub, "hf_model_info", info)
    monkeypatch.setattr(hub, "hf_text", text)
    return infos, texts


def enriched(stores, template: str = BASE_TEMPLATE):
    make_hf_repo(stores / "hub", "quant/model", {"model.gguf": chat_gguf(template)})
    c = discover.discover([])[0]
    checks.inspect(c)
    online.enrich(c)
    return c


def by_rule(c) -> dict[str, list]:
    out: dict[str, list] = {}
    for f in c.findings:
        out.setdefault(f.rule, []).append(f)
    return out


def test_matching_claims_produce_no_findings(stores, fake_hub):
    c = enriched(stores)
    assert c.findings == []
    assert c.evidence["template_vs_base"] == {"model.gguf#chat_template": "identical"}
    assert c.evidence["base"] == {"repo": "meta/base", "revision": "b" * 40}
    assert c.evidence["base_relation"] == "quantized"


def test_relabelled_licence_is_high(stores, fake_hub):
    fake_hub[0]["quant/model"] = quant_info("apache-2.0")
    finding = by_rule(enriched(stores))["CLAIM002"][0]
    assert finding.severity == "high"
    assert "apache-2.0" in finding.evidence
    assert "llama3.2" in finding.evidence


def test_permissive_claim_over_custom_terms_is_medium(stores, fake_hub):
    fake_hub[0]["quant/model"] = quant_info("apache-2.0")
    fake_hub[0]["meta/base"] = base_info("other")
    assert by_rule(enriched(stores))["CLAIM002"][0].severity == "medium"


def test_base_template_in_jinja_file_is_used(stores, fake_hub):
    info = base_info(chat=None)
    info["config"]["chat_template_jinja"] = BASE_TEMPLATE
    fake_hub[0]["meta/base"] = info
    c = enriched(stores)
    assert c.evidence["template_vs_base"] == {"model.gguf#chat_template": "identical"}


def test_different_permissive_licence_is_low(stores, fake_hub):
    fake_hub[0]["quant/model"] = quant_info("apache-2.0")
    fake_hub[0]["meta/base"] = base_info("mit")
    assert by_rule(enriched(stores))["CLAIM002"][0].severity == "low"


def test_missing_licence_is_reported(stores, fake_hub):
    fake_hub[0]["quant/model"] = quant_info(None)
    assert "CLAIM004" in by_rule(enriched(stores))


def test_added_trigger_is_high_and_inherited_is_low(stores, fake_hub):
    fake_hub[0]["meta/base"] = base_info(chat=BASE_TEMPLATE + INHERITED)
    found = by_rule(enriched(stores, BASE_TEMPLATE + INHERITED + ADDED))
    severities = {f.evidence.split("branches on ")[1]: f.severity for f in found["TMPL002"]}
    assert severities == {"'please think'": "low", "'wire the funds'": "high"}
    assert found["TMPL010"][0].severity == "info"
    assert "3 segments added and 0 removed" in found["TMPL010"][0].evidence
    assert "wire the funds" in found["TMPL010"][0].evidence


def test_local_revision_behind_the_hub(stores, fake_hub):
    fake_hub[0]["quant/model"] = quant_info(sha="c" * 40)
    assert by_rule(enriched(stores))["DRIFT001"][0].severity == "low"


def test_structure_is_compared_with_the_base_config(stores, fake_hub):
    fake_hub[1]["meta/base", "config.json"] = json.dumps(
        {"num_hidden_layers": 12, "hidden_size": 8, "num_attention_heads": 2}
    )
    finding = by_rule(enriched(stores))["CLAIM003"][0]
    assert finding.severity == "high"
    assert "layers 2 vs 12" in finding.evidence


def test_matching_structure_is_recorded(stores, fake_hub):
    fake_hub[1]["meta/base", "config.json"] = json.dumps(
        {"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2}
    )
    c = enriched(stores)
    assert c.findings == []
    assert c.evidence["structure_vs_base"] == "matches"


def test_gated_base_config_is_skipped_not_failed(stores, fake_hub):
    c = enriched(stores)
    assert any("structure not compared" in note for note in c.skipped)


def test_hub_failure_is_skipped_not_fatal(stores, fake_hub):
    fake_hub[0].clear()
    c = enriched(stores)
    assert c.findings == []
    assert any("failed" in note for note in c.skipped)


def test_ollama_tag_drift(stores, monkeypatch):
    path = make_ollama(
        stores / "ollama", "demo", "latest", {"model": chat_gguf(), "template": b"{{ .Prompt }}"}
    )
    manifest = json.loads(path.read_text())
    monkeypatch.setattr(hub, "ollama_manifest", lambda *_: manifest)
    c = discover.discover([])[0]
    checks.inspect(c)
    online.enrich(c)
    assert "DRIFT002" not in by_rule(c)

    moved = json.loads(path.read_text())
    moved["layers"][1]["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(hub, "ollama_manifest", lambda *_: moved)
    c = discover.discover([])[0]
    checks.inspect(c)
    online.enrich(c)
    assert by_rule(c)["DRIFT002"][0].evidence == "layers that differ: template"
