from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO

from knowyourai import gguf, pickles, safetensors, template
from knowyourai.hub import HubError

if TYPE_CHECKING:
    from knowyourai.model import Component, FileRef

TEXT_LIMIT = 16 << 20
PICKLE_LIMIT = 512 << 20
SAFE_EXTS = {".gguf": "gguf", ".safetensors": "safetensors"}
PICKLE_EXTS = frozenset({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib"})
UNINSPECTED = {
    ".onnx": "ONNX",
    ".h5": "HDF5/Keras",
    ".keras": "Keras",
    ".npz": "NumPy archive",
    ".npy": "NumPy array",
    ".pb": "TensorFlow graph",
    ".tflite": "TFLite",
    ".mlmodel": "Core ML",
}
WEIGHT_EXTS = frozenset(SAFE_EXTS) | PICKLE_EXTS | frozenset(UNINSPECTED)
CONFIG_KEYS = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "n_layer",
    "num_layers",
    "hidden_size",
    "n_embd",
    "d_model",
    "num_attention_heads",
    "n_head",
    "vocab_size",
)
TEMPLATE_RULES = frozenset({"TMPL001", "TMPL002", "TMPL003"})
TEMPLATE_FIX = {
    "TMPL001": "Do not use this template. Replace it with the original publisher's.",
    "TMPL002": "Read the branch. A template has no reason to react to what a user says.",
    "TMPL003": "Check why a prompt template needs a URL before running the model.",
}
_HF_URL = re.compile(r"huggingface\.co/([\w.\-]+/[\w.\-]+)")
_READ_ERRORS = (OSError, HubError, ValueError, RecursionError)


def sniff(head: bytes) -> str:
    if head.startswith(b"GGUF"):
        return "gguf"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    header_len = int.from_bytes(head[:8], "little")
    if len(head) > 8 and head[8:9] == b"{" and header_len <= safetensors.MAX_HEADER:
        return "safetensors"
    if len(head) >= 2 and head[0] == 0x80 and 2 <= head[1] <= 5:
        return "pickle"
    if head.startswith(b"\x89HDF"):
        return "hdf5"
    return "unknown"


def _dominant(dtype_params: dict[str, int]) -> str:
    return max(dtype_params.items(), key=lambda kv: kv[1])[0] if dtype_params else ""


def _weight_entry(c: Component, entry: dict[str, Any]) -> None:
    c.evidence.setdefault("weights", []).append(entry)


def _gguf(c: Component, f: FileRef, fh: BinaryIO) -> None:
    try:
        info = gguf.read_gguf(fh)
    except gguf.GGUFError as e:
        c.flag(
            "FMT004",
            "high",
            "GGUF header could not be parsed safely",
            f"{f.path}: {e}",
            fix="Treat the file as untrusted: a loader may read it differently than this parser.",
            file=f.path,
        )
        return
    md = info.metadata
    entry: dict[str, Any] = {
        "path": f.path,
        "format": "gguf",
        "tensors": info.tensor_count,
        "params": info.params,
        "shape_sha256": info.shape_sha256,
        "dtype": info.dominant_dtype,
    }
    arch = md.get("general.architecture")
    if isinstance(arch, str):
        entry["architecture"] = arch
        for key, label in (
            ("block_count", "layers"),
            ("embedding_length", "hidden"),
            ("attention.head_count", "heads"),
        ):
            value = md.get(f"{arch}.{key}")
            if isinstance(value, int):
                entry[label] = value
    tokens = md.get("tokenizer.ggml.tokens")
    if isinstance(tokens, dict):
        entry["vocab"] = tokens["count"]
        entry["vocab_sha256"] = tokens["sha256"]
    elif isinstance(tokens, list):
        entry["vocab"] = len(tokens)
    if isinstance(md.get("general.name"), str):
        entry["name"] = md["general.name"]
    _weight_entry(c, entry)

    source = f"{f.path} metadata"
    if isinstance(md.get("general.license"), str):
        c.claim("license", md["general.license"].lower(), source)
    count = md.get("general.base_model.count")
    for i in range(min(count, 8) if isinstance(count, int) else 0):
        match = _HF_URL.search(str(md.get(f"general.base_model.{i}.repo_url", "")))
        if match:
            c.claim("base_model", match.group(1), source)
    for key, value in md.items():
        if isinstance(value, str) and (
            key == "tokenizer.chat_template" or key.startswith("tokenizer.chat_template.")
        ):
            c.templates[f"{f.path}#{key.removeprefix('tokenizer.')}"] = value


def _safetensors(c: Component, f: FileRef, fh: BinaryIO) -> None:
    fix = "Do not load it. Re-download from the original publisher."
    try:
        info = safetensors.read_safetensors(fh, f.size)
    except safetensors.SafetensorsError as e:
        c.flag(
            "FMT002",
            "high",
            "safetensors header is invalid",
            f"{f.path}: {e}",
            fix=fix,
            file=f.path,
        )
        return
    if info.problems:
        c.flag(
            "FMT002",
            "high",
            "safetensors file has structural anomalies",
            f"{f.path}: " + "; ".join(info.problems[:4]),
            fix=fix,
            file=f.path,
        )
    _weight_entry(
        c,
        {
            "path": f.path,
            "format": "safetensors",
            "tensors": info.tensor_count,
            "params": info.params,
            "shape_sha256": info.shape_sha256,
            "dtype": _dominant(info.dtype_params),
        },
    )


def _report_pickle(c: Component, f: FileRef, probe: pickles.PickleProbe, member: str = "") -> None:
    where = f"{f.path}!{member}" if member else f.path
    dangerous, unknown = probe.dangerous, probe.unknown
    if dangerous or probe.unresolved or probe.error:
        parts = []
        if dangerous:
            parts.append("imports " + ", ".join(dangerous[:6]))
        if probe.unresolved:
            parts.append(f"{probe.unresolved} import(s) could not be resolved statically")
        if probe.error:
            parts.append(f"stream is malformed ({probe.error})")
        c.flag(
            "EXEC002",
            "high",
            "Pickle can run arbitrary code when loaded",
            f"{where}: " + "; ".join(parts),
            fix="Do not load it. Ask the publisher for safetensors.",
            file=f.path,
        )
    elif unknown or probe.trailing:
        parts = []
        if unknown:
            more = f" (+{len(unknown) - 6} more)" if len(unknown) > 6 else ""
            parts.append("imports " + ", ".join(unknown[:6]) + more)
        if probe.trailing:
            parts.append("data after the pickle was not inspected")
        c.flag(
            "EXEC001",
            "medium",
            "Pickle imports code outside the tensor-only allowlist",
            f"{where}: " + "; ".join(parts),
            fix="Load only with weights_only=True, or convert to safetensors.",
            file=f.path,
        )
    else:
        c.flag(
            "EXEC001",
            "low",
            "Weights are stored as a pickle",
            f"{where}: {len(probe.globals)} imports, all tensor rebuild functions",
            fix="Prefer safetensors: a pickle runs code if the file is ever replaced.",
            file=f.path,
        )


def _unprobed_pickle(c: Component, f: FileRef, reason: str) -> None:
    c.unverified.append(f"{f.path}: pickle not inspected ({reason})")
    c.flag(
        "EXEC001",
        "medium",
        "Pickle-format file was not inspected",
        f"{f.path}: {reason}",
        fix="Download it and scan the local copy before loading, or use safetensors.",
        file=f.path,
    )


def _weights(c: Component, f: FileRef, ext: str) -> None:
    with f.open() as fh:
        kind = sniff(fh.read(16))
        fh.seek(0)
        expected = SAFE_EXTS.get(ext)
        if expected and kind != expected:
            c.flag(
                "FMT001",
                "high",
                f"File is not the {expected} its name claims",
                f"{f.path}: content looks like {kind}",
                fix="Do not load it. Re-download from the original publisher.",
                file=f.path,
            )
        if kind == "gguf":
            _gguf(c, f, fh)
        elif kind == "safetensors":
            _safetensors(c, f, fh)
        elif kind in {"zip", "pickle"} or (kind == "unknown" and ext in PICKLE_EXTS):
            _weight_entry(c, {"path": f.path, "format": "pickle"})
            if f.remote:
                _unprobed_pickle(c, f, "remote file")
            elif f.size > PICKLE_LIMIT and kind != "zip":
                _unprobed_pickle(c, f, f"larger than {PICKLE_LIMIT >> 20} MiB")
            elif kind == "zip":
                probes, error = pickles.probe_zip(fh)
                if error:
                    c.flag(
                        "EXEC002",
                        "high",
                        "Archive failed strict parsing",
                        f"{f.path}: {error}",
                        fix="Do not load it: loaders may accept an archive this parser rejects.",
                        file=f.path,
                    )
                elif not probes:
                    c.evidence["weights"].pop()
                    c.unverified.append(f"{f.path}: zip archive with no pickle member")
                for member, probe in probes.items():
                    _report_pickle(c, f, probe, member)
            else:
                probe = pickles.probe(fh)
                if kind == "unknown" and probe.error and not probe.globals:
                    c.evidence["weights"].pop()
                    c.unverified.append(f"{f.path}: unrecognised binary content")
                else:
                    _report_pickle(c, f, probe)
        else:
            name = UNINSPECTED.get(ext, "unrecognised")
            c.unverified.append(f"{f.path}: {name} format is not inspected in this version")


def _load_json(c: Component, f: FileRef) -> Any:
    try:
        return json.loads(f.read_bytes(TEXT_LIMIT))
    except _READ_ERRORS as e:
        c.unverified.append(f"{f.path}: could not be read ({e})")
        return None


def _flatten(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [s for item in value for s in _flatten(item)]
    return []


def _auto_map(c: Component, f: FileRef, data: dict[str, Any]) -> None:
    refs = sorted(set(_flatten(data.get("auto_map"))))
    if not refs:
        return
    external = [r for r in refs if "--" in r]
    if external:
        c.flag(
            "EXEC003",
            "high",
            "Config loads Python code from another repository",
            f"{f.path}: auto_map points to {', '.join(external[:4])}",
            fix="Never enable trust_remote_code for this model.",
            file=f.path,
        )
    else:
        c.flag(
            "EXEC003",
            "medium",
            "Config maps model classes to Python code shipped with the model",
            f"{f.path}: auto_map points to {', '.join(refs[:4])}",
            fix="Read that code and pin the revision before enabling trust_remote_code.",
            file=f.path,
        )


def _config(c: Component, f: FileRef) -> None:
    data = _load_json(c, f)
    if not isinstance(data, dict):
        return
    _auto_map(c, f, data)
    if "_attn_implementation_internal" in data:
        c.flag(
            "EXEC005",
            "high",
            "Config sets a private field that transformers never writes",
            f"{f.path}: _attn_implementation_internal is present "
            "(reported indicator of CVE-2026-4372)",
            fix="Do not load it with transformers older than 5.3.0.",
            file=f.path,
        )
    subset = {k: data[k] for k in CONFIG_KEYS if isinstance(data.get(k), (int, str, list))}
    if subset:
        c.evidence["config"] = subset


def _tokenizer_config(c: Component, f: FileRef) -> None:
    data = _load_json(c, f)
    if not isinstance(data, dict):
        return
    _auto_map(c, f, data)
    chat = data.get("chat_template")
    if isinstance(chat, str):
        c.templates[f.path] = chat
    elif isinstance(chat, list):
        for item in chat:
            if isinstance(item, dict) and isinstance(item.get("template"), str):
                c.templates[f"{f.path}#{item.get('name', 'default')}"] = item["template"]


def _adapter_config(c: Component, f: FileRef) -> None:
    data = _load_json(c, f)
    if not isinstance(data, dict):
        return
    c.evidence["adapter"] = True
    base = data.get("base_model_name_or_path")
    if isinstance(base, str) and re.fullmatch(r"[\w.\-]+/[\w.\-]+", base):
        c.claim("base_model", base, f.path)


def front_matter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    out: dict[str, Any] = {}
    key = ""
    for line in text[3:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] not in " \t-" and ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if value.startswith("[") and value.endswith("]"):
                out[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
            else:
                out[key] = value.strip("'\"") if value else []
        elif line.lstrip().startswith("- ") and isinstance(out.get(key), list):
            out[key].append(line.lstrip()[2:].strip().strip("'\""))
    return out


def _card(c: Component, f: FileRef) -> None:
    try:
        meta = front_matter(f.read_bytes(TEXT_LIMIT).decode("utf-8", "replace"))
    except _READ_ERRORS:
        return
    for kind in ("license", "base_model"):
        for value in _flatten(meta.get(kind)):
            c.claim(kind, value.lower() if kind == "license" else value, f.path)


def _license_name(text: str) -> str:
    head = text[:600].lower()
    if "apache license" in head and "version 2.0" in head:
        return "apache-2.0"
    if "mit license" in head:
        return "mit"
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first[:80]


def _text(c: Component, f: FileRef) -> str | None:
    try:
        return f.read_bytes(TEXT_LIMIT).decode("utf-8", "replace")
    except _READ_ERRORS as e:
        c.unverified.append(f"{f.path}: could not be read ({e})")
        return None


def _ollama_layer(c: Component, f: FileRef) -> None:
    kind = f.path.split(".", 1)[0]
    if kind in ("model", "adapter", "projector"):
        _weights(c, f, "")
        return
    text = _text(c, f)
    if text is None:
        return
    if kind == "template":
        c.templates[f"ollama:{f.path}"] = text
    elif kind == "license":
        c.claim("license", _license_name(text), f"ollama {f.path} layer")
    elif kind == "system":
        c.evidence["system_prompt"] = {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "preview": " ".join(text.split())[:120],
        }
    elif kind == "params":
        try:
            c.evidence["params"] = json.loads(text)
        except ValueError:
            c.unverified.append(f"{f.path}: params layer is not JSON")


def _pick_gguf(files: list[FileRef]) -> FileRef | None:
    candidates = [f for f in files if "mmproj" not in f.path.lower()] or files
    firsts = [f for f in candidates if "-of-" not in f.path or "-00001-of-" in f.path]
    candidates = firsts or candidates
    preferred = [f for f in candidates if "q4_k_m" in f.path.lower()]
    pool = preferred or candidates
    return min(pool, key=lambda f: f.size) if pool else None


def _remote_selection(c: Component) -> set[str] | None:
    if c.store != "hf-hub":
        return None
    pattern = c.evidence.pop("file_pattern", None)
    if pattern:
        return {f.path for f in c.files if fnmatch.fnmatch(f.path, pattern)}
    ggufs = [f for f in c.files if f.path.lower().endswith(".gguf")]
    chosen = {f.path for f in c.files if not f.path.lower().endswith(".gguf")}
    pick = _pick_gguf(ggufs)
    if pick:
        chosen.add(pick.path)
        c.evidence["inspected_gguf"] = pick.path
    if len(ggufs) > 1:
        c.skipped.append(
            f"{len(ggufs) - 1} other GGUF files were not inspected (choose with --file)"
        )
    return chosen


def _inspect_file(c: Component, f: FileRef) -> None:
    path = PurePosixPath(f.path)
    name, ext = path.name.lower(), path.suffix.lower()
    if c.store == "ollama":
        _ollama_layer(c, f)
    elif name == "config.json":
        _config(c, f)
    elif name == "tokenizer_config.json":
        _tokenizer_config(c, f)
    elif name == "adapter_config.json":
        _adapter_config(c, f)
    elif ext == ".jinja":
        text = _text(c, f)
        if text is not None:
            c.templates[f.path] = text
    elif name == "chat_template.json":
        data = _load_json(c, f)
        if isinstance(data, dict) and isinstance(data.get("chat_template"), str):
            c.templates[f.path] = data["chat_template"]
    elif name == "readme.md" and not f.remote:
        _card(c, f)
    elif ext in WEIGHT_EXTS:
        _weights(c, f, ext)


def analyse_templates(c: Component, severity: dict[tuple[str, str], str] | None = None) -> None:
    c.findings = [f for f in c.findings if f.rule not in TEMPLATE_RULES]
    digests: dict[str, str] = {}
    seen: set[str] = set()
    for source, text in c.templates.items():
        digest = template.digest(text)
        digests[source] = digest
        if digest in seen:
            continue
        seen.add(digest)
        for hit in template.analyse(text, go=source.startswith("ollama:")):
            level, note = hit.severity, ""
            if severity and (digest, hit.evidence) in severity:
                level = severity[digest, hit.evidence]
                note = (
                    " (also in the base model's template)"
                    if level == "low"
                    else " (not in the base model's template)"
                )
            c.flag(
                hit.rule,
                level,
                hit.title + note,
                f"{source}: {hit.evidence}",
                fix=TEMPLATE_FIX[hit.rule],
                file=source.split("#", 1)[0],
            )
    if digests:
        c.evidence["templates"] = digests


def inspect(c: Component) -> None:
    selected = _remote_selection(c)
    python_files: list[str] = []
    for f in c.files:
        if f.path.lower().endswith(".py"):
            python_files.append(f.path)
        if selected is not None and f.path not in selected:
            continue
        try:
            _inspect_file(c, f)
        except (OSError, HubError) as e:
            c.unverified.append(f"{f.path}: could not be read ({e})")
    if python_files:
        more = f" (+{len(python_files) - 5} more)" if len(python_files) > 5 else ""
        c.flag(
            "EXEC004",
            "low",
            "Model ships Python files",
            ", ".join(python_files[:5]) + more,
            fix="They only run if you enable trust_remote_code. Leave it off.",
        )
    analyse_templates(c)


def verify_digests(c: Component) -> None:
    for f in c.files:
        if f.remote:
            continue
        try:
            actual = f.hash_content()
        except OSError as e:
            c.unverified.append(f"{f.path}: could not be hashed ({e})")
            continue
        if f.sha256 and f.digest_source == "store" and actual != f.sha256:
            c.flag(
                "INTEG001",
                "high",
                "File content does not match the digest it is stored under",
                f"{f.path}: stored as {f.sha256[:12]}, content hashes to {actual[:12]}",
                fix="The file changed after download. Delete it and pull again.",
                file=f.path,
            )
        f.sha256, f.digest_source = actual, "computed"
