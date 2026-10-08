from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from knowyourai import checks, discover, hub, template

if TYPE_CHECKING:
    from knowyourai.model import Component

PERMISSIVE = frozenset(
    "apache-2.0 mit bsd-2-clause bsd-3-clause bsd-3-clause-clear cc0-1.0 unlicense isc cc-by-4.0 "
    "cc-by-3.0 cc-by-2.0 wtfpl zlib bsl-1.0 postgresql cdla-permissive-2.0 cdla-permissive-1.0 "
    "pddl odc-by afl-3.0 artistic-2.0 ncsa".split()
)
RESTRICTED_PREFIXES = (
    "cc-by-nc",
    "cc-by-nd",
    "llama",
    "gemma",
    "openrail",
    "bigscience-",
    "creativeml-",
    "deepfloyd-",
    "apple-",
    "gpl",
    "agpl",
    "lgpl",
)
STRUCTURE_ALIASES = {
    "layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "hidden": ("hidden_size", "n_embd", "d_model"),
    "heads": ("num_attention_heads", "n_head"),
    "vocab": ("vocab_size",),
}
CONFIG_LIMIT = 4 << 20
_RELATION = re.compile(r"^base_model:(quantized|finetune|adapter|merge):")


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _hf(c: Component) -> None:
    repo = c.id.removeprefix("hf:")
    try:
        info = hub.hf_model_info(repo)
    except hub.HubError as e:
        c.skipped.append(f"Hub lookup for {repo} failed: {e}")
        return
    head = info.get("sha")
    if c.store == "hf-cache" and "main" in c.refs and head and head != c.revision:
        c.flag(
            "DRIFT001",
            "low",
            "The publisher has changed this model since you downloaded it",
            f"local main is {(c.revision or '')[:12]}, Hub main is {head[:12]}",
            fix="Review what changed before updating, and pin the revision in code.",
        )
    card = info.get("cardData") or {}
    for value in _as_list(card.get("license")):
        c.claim("license", value.lower(), "Hub model card")
    for value in _as_list(card.get("base_model")):
        c.claim("base_model", value, "Hub model card")
    for tag in info.get("tags") or []:
        match = _RELATION.match(str(tag))
        if match:
            c.evidence["base_relation"] = match.group(1)
    if info.get("gated"):
        c.evidence["gated"] = info["gated"]
    if not c.claims("license"):
        c.flag(
            "CLAIM004",
            "low",
            "No licence is declared",
            f"{repo}: neither the model card nor the files state a licence",
            fix="Treat it as all rights reserved until the publisher states one.",
        )


def _ollama(c: Component) -> None:
    meta = c.evidence.get("ollama") or {}
    if meta.get("host") != hub.OLLAMA_HOST:
        return
    try:
        manifest = hub.ollama_manifest(meta["namespace"], meta["model"], meta["tag"])
    except hub.HubError as e:
        c.skipped.append(f"Ollama registry lookup failed: {e}")
        return
    remote = {label: digest for label, digest, _size in discover.ollama_layers(manifest)}
    local = {label: digest for label, digest, _size in c.evidence.get("ollama_layers", [])}
    changed = sorted(k for k in set(local) | set(remote) if local.get(k) != remote.get(k))
    if changed:
        c.flag(
            "DRIFT002",
            "low",
            "This tag now points to different content in the Ollama registry",
            f"layers that differ: {', '.join(changed)}",
            fix="Review what changed before pulling again.",
        )


def _license(c: Component, base: str, info: dict[str, Any]) -> None:
    base_licenses = [v.lower() for v in _as_list((info.get("cardData") or {}).get("license"))]
    local = [v.lower() for v in c.claims("license") if " " not in v]
    if not base_licenses or not local or set(local) & set(base_licenses):
        return
    evidence = f"declares {', '.join(local)}; base model {base} is {', '.join(base_licenses)}"
    permissive_claim = any(v in PERMISSIVE for v in local)
    if permissive_claim and any(v.startswith(RESTRICTED_PREFIXES) for v in base_licenses):
        c.flag(
            "CLAIM002",
            "high",
            "Licence is more permissive than the base model's",
            evidence,
            fix="Follow the base model's licence until the publisher explains the difference.",
        )
    elif permissive_claim and not any(v in PERMISSIVE for v in base_licenses):
        c.flag(
            "CLAIM002",
            "medium",
            "Permissive licence declared over a base model with custom terms",
            evidence,
            fix="Read the base model's licence text: its terms may still apply.",
        )
    else:
        c.flag(
            "CLAIM002",
            "low",
            "Licence differs from the base model's",
            evidence,
            fix="Check which terms actually apply before commercial use.",
        )


def _base_template(info: dict[str, Any]) -> str | None:
    config = info.get("config") or {}
    # Newer repos keep the template in chat_template.jinja, which the Hub exposes separately.
    chat = config.get("chat_template_jinja") or (config.get("tokenizer_config") or {}).get(
        "chat_template"
    )
    if isinstance(chat, list):
        named = [i for i in chat if isinstance(i, dict) and isinstance(i.get("template"), str)]
        default = [i for i in named if i.get("name") == "default"] or named
        chat = default[0]["template"] if default else None
    return chat if isinstance(chat, str) and chat.strip() else None


def _template(c: Component, base: str, info: dict[str, Any]) -> None:
    local = {s: t for s, t in c.templates.items() if not s.startswith("ollama:")}
    if not local:
        return
    base_text = _base_template(info)
    if base_text is None:
        c.skipped.append(f"base model {base} publishes no chat template to compare against")
        return
    base_hits = {h.evidence for h in template.analyse(base_text)}
    severity: dict[tuple[str, str], str] = {}
    by_digest: dict[str, str] = {}
    results: dict[str, str] = {}
    for source, text in local.items():
        digest = template.digest(text)
        if digest in by_digest:
            results[source] = by_digest[digest]
            continue
        cmp = template.compare(base_text, text)
        by_digest[digest] = results[source] = cmp.kind
        for hit in template.analyse(text):
            if hit.evidence in base_hits and hit.rule != "TMPL001":
                severity[digest, hit.evidence] = "low"
            elif cmp.kind == "differs":
                severity[digest, hit.evidence] = "high"
        if cmp.kind == "differs":
            sample = " | ".join(" ".join(s.split())[:80] for s in cmp.added[:3])
            c.flag(
                "TMPL010",
                "info",
                "Chat template differs from the base model's",
                f"{source}: {len(cmp.added)} segments added and {len(cmp.removed)} removed "
                f"relative to {base}" + (f". Added: {sample}" if sample else ""),
                fix="Quantizers often patch templates. Read the added segments before trusting it.",
                file=source.split("#", 1)[0],
            )
    c.evidence["template_vs_base"] = results
    checks.analyse_templates(c, severity)


def config_structure(config: dict[str, Any]) -> dict[str, int]:
    for scope in (config, config.get("text_config"), config.get("llm_config")):
        if not isinstance(scope, dict):
            continue
        out: dict[str, int] = {}
        for label, keys in STRUCTURE_ALIASES.items():
            for key in keys:
                if isinstance(scope.get(key), int):
                    out[label] = scope[key]
                    break
        if out:
            return out
    return {}


def _local_structure(c: Component) -> dict[str, int]:
    for weights in c.evidence.get("weights") or []:
        if weights.get("format") == "gguf" and "layers" in weights:
            return {k: weights[k] for k in STRUCTURE_ALIASES if k in weights}
    return config_structure(c.evidence.get("config") or {})


def _structure(c: Component, base: str, info: dict[str, Any]) -> None:
    if c.evidence.get("adapter") or c.evidence.get("base_relation") in ("adapter", "merge"):
        return
    local = _local_structure(c)
    if not local:
        return
    try:
        config = json.loads(hub.hf_text(base, str(info.get("sha") or "main"), "config.json"))
    except hub.HubError as e:
        c.skipped.append(f"structure not compared: base model {base} config {e}")
        return
    except ValueError:
        return
    upstream = config_structure(config) if isinstance(config, dict) else {}
    diffs = {
        k: f"{k} {local[k]} vs {upstream[k]}"
        for k in STRUCTURE_ALIASES
        if k in local and k in upstream and local[k] != upstream[k]
    }
    major = [diffs[k] for k in ("layers", "hidden") if k in diffs]
    if major:
        c.flag(
            "CLAIM003",
            "high",
            "Weights do not have the shape of the claimed base model",
            f"this model vs {base}: {', '.join(diffs.values())}",
            fix="It is not a quantization or fine-tune of that model. Find out what it is.",
        )
    elif diffs:
        c.flag(
            "CLAIM003",
            "low",
            "Small structural difference from the claimed base model",
            f"this model vs {base}: {', '.join(diffs.values())}",
            fix="Usually a resized vocabulary. Confirm with the publisher's notes.",
        )
    elif set(local) & set(upstream):
        c.evidence["structure_vs_base"] = "matches"


def _base(c: Component) -> None:
    bases = [
        b
        for b in c.claims("base_model")
        if re.fullmatch(r"[\w.\-]+/[\w.\-]+", b) and f"hf:{b}" != c.id
    ]
    if not bases:
        if c.evidence.get("weights"):
            c.skipped.append("no base model is declared, so nothing was compared upstream")
        return
    base = bases[0]
    try:
        info = hub.hf_model_info(base)
    except hub.HubError as e:
        c.skipped.append(f"base model {base} lookup failed: {e}")
        return
    c.evidence["base"] = {"repo": base, "revision": info.get("sha")}
    _license(c, base, info)
    _template(c, base, info)
    _structure(c, base, info)


def enrich(c: Component) -> None:
    if c.store in ("hf-cache", "hf-hub"):
        _hf(c)
    elif c.store == "ollama":
        _ollama(c)
    _base(c)
