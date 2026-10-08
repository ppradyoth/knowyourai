from __future__ import annotations

import hashlib
import json
import struct
from math import prod
from pathlib import Path

REVISION = "a" * 40
QUANT = 12


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(value) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return struct.pack("<II", 4, value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + _string(value)
    items = b"".join(_string(v) for v in value)
    return struct.pack("<IIQ", 9, 8, len(value)) + items


def gguf_bytes(metadata: dict, tensors=(), version: int = 3) -> bytes:
    out = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, value in metadata.items():
        out += _string(key) + _value(value)
    for name, dims, dtype in tensors:
        out += _string(name) + struct.pack("<I", len(dims))
        out += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", dtype, 0)
    return out


def chat_gguf(template: str = "{{ messages }}", **extra) -> bytes:
    metadata = {
        "general.architecture": "llama",
        "general.name": "demo",
        "llama.block_count": 2,
        "llama.embedding_length": 8,
        "llama.attention.head_count": 2,
        "tokenizer.chat_template": template,
        **extra,
    }
    return gguf_bytes(metadata, [("blk.0.attn_q.weight", (8, 8), QUANT)])


def safetensors_bytes(tensors: dict, metadata: dict | None = None, extra: bytes = b"") -> bytes:
    header: dict = {}
    offset = 0
    for name, shape in tensors.items():
        size = prod(shape) * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * offset + extra


def make_hf_repo(root: Path, repo: str, files: dict[str, bytes], revision: str = REVISION) -> Path:
    repo_dir = root / ("models--" + repo.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / revision
    blobs = repo_dir / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(revision)
    for name, data in files.items():
        blob = blobs / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
    return repo_dir


def make_ollama(root: Path, name: str, tag: str, layers: dict[str, bytes]) -> Path:
    blobs = root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    entries = []
    for kind, data in layers.items():
        digest = hashlib.sha256(data).hexdigest()
        (blobs / f"sha256-{digest}").write_bytes(data)
        entries.append(
            {
                "mediaType": f"application/vnd.ollama.image.{kind}",
                "digest": f"sha256:{digest}",
                "size": len(data),
            }
        )
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 2, "layers": entries}))
    return manifest
