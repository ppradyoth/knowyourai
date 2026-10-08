from __future__ import annotations

import io
import json
import struct

import pytest
from helpers import QUANT, gguf_bytes, safetensors_bytes

from knowyourai import gguf, safetensors


def test_gguf_reads_metadata_and_tensors():
    data = gguf_bytes(
        {"general.architecture": "llama", "llama.block_count": 2},
        [("a", (4, 8), QUANT), ("b", (8,), 0)],
    )
    info = gguf.read_gguf(io.BytesIO(data))
    assert info.metadata == {"general.architecture": "llama", "llama.block_count": 2}
    assert info.tensor_count == 2
    assert info.params == 40
    assert info.dominant_dtype == "Q4_K"
    assert info.header_bytes == len(data)


def test_gguf_shape_hash_ignores_quantization():
    q4 = gguf.read_gguf(io.BytesIO(gguf_bytes({}, [("w", (4, 8), QUANT)])))
    f16 = gguf.read_gguf(io.BytesIO(gguf_bytes({}, [("w", (4, 8), 1)])))
    other = gguf.read_gguf(io.BytesIO(gguf_bytes({}, [("w", (4, 9), QUANT)])))
    assert q4.shape_sha256 == f16.shape_sha256
    assert q4.shape_sha256 != other.shape_sha256


def test_gguf_summarises_large_arrays():
    tokens = [f"tok{i}" for i in range(200)]
    info = gguf.read_gguf(io.BytesIO(gguf_bytes({"tokenizer.ggml.tokens": tokens})))
    summary = info.metadata["tokenizer.ggml.tokens"]
    assert summary["count"] == 200
    assert len(summary["sha256"]) == 64


@pytest.mark.parametrize(
    "data",
    [
        b"NOPE" + b"\0" * 20,
        gguf_bytes({"k": "v"})[:-3],
        b"GGUF" + struct.pack("<IQQ", 3, 0, 10**9),
        b"GGUF" + struct.pack("<IQQ", 3, 10**9, 0),
        b"GGUF" + struct.pack("<IQQ", 99, 0, 0),
        gguf_bytes({"k": "v"})[:24] + struct.pack("<Q", 10**12) + b"x",
    ],
    ids=["magic", "truncated", "kv-count", "tensor-count", "version", "string-length"],
)
def test_gguf_rejects_hostile_headers(data):
    with pytest.raises(gguf.GGUFError):
        gguf.read_gguf(io.BytesIO(data))


def test_gguf_rejects_duplicate_keys():
    one = gguf_bytes({"k": "a"})
    pair = one[:24] + one[24:] * 2
    doubled = pair[:16] + struct.pack("<Q", 2) + pair[24:]
    with pytest.raises(gguf.GGUFError, match="duplicate"):
        gguf.read_gguf(io.BytesIO(doubled))


def test_safetensors_valid_file():
    data = safetensors_bytes({"w": [2, 3], "b": [3]}, metadata={"format": "pt"})
    info = safetensors.read_safetensors(io.BytesIO(data), len(data))
    assert info.problems == []
    assert info.tensor_count == 2
    assert info.params == 9
    assert info.metadata == {"format": "pt"}


def test_safetensors_flags_hidden_bytes():
    data = safetensors_bytes({"w": [2, 3]}, extra=b"payload")
    info = safetensors.read_safetensors(io.BytesIO(data), len(data))
    assert any("unreferenced bytes after" in p for p in info.problems)


def test_safetensors_flags_duplicate_keys_and_overlap():
    entry = '{"dtype":"F32","shape":[1],"data_offsets":[0,4]}'
    raw = f'{{"w":{entry},"w":{entry},"v":{entry}}}'.encode()
    data = struct.pack("<Q", len(raw)) + raw + b"\0" * 4
    info = safetensors.read_safetensors(io.BytesIO(data), len(data))
    assert any("duplicate JSON keys" in p for p in info.problems)
    assert any("overlaps" in p for p in info.problems)


def test_safetensors_flags_span_mismatch():
    raw = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}).encode()
    data = struct.pack("<Q", len(raw)) + raw + b"\0" * 4
    info = safetensors.read_safetensors(io.BytesIO(data), len(data))
    assert any("does not match shape" in p for p in info.problems)


@pytest.mark.parametrize(
    "data",
    [b"\x01", struct.pack("<Q", 10**12) + b"{}", struct.pack("<Q", 2) + b"[]"],
    ids=["short", "huge-header", "not-object"],
)
def test_safetensors_rejects_invalid_headers(data):
    with pytest.raises(safetensors.SafetensorsError):
        safetensors.read_safetensors(io.BytesIO(data), len(data))
