from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO

MAX_HEADER = 100 << 20
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


class SafetensorsError(ValueError):
    pass


@dataclass
class SafetensorsInfo:
    header_len: int
    metadata: dict[str, str]
    tensor_count: int = 0
    params: int = 0
    dtype_params: dict[str, int] = field(default_factory=dict)
    shape_sha256: str = ""
    problems: list[str] = field(default_factory=list)


def _parse_header(raw: bytes) -> tuple[dict[str, Any], list[str]]:
    duplicates: list[str] = []

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        header = json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, RecursionError) as e:
        msg = f"header is not valid JSON: {e}"
        raise SafetensorsError(msg) from e
    if not isinstance(header, dict):
        msg = "header is not a JSON object"
        raise SafetensorsError(msg)
    return header, duplicates


def read_safetensors(f: BinaryIO, file_size: int) -> SafetensorsInfo:
    prefix = f.read(8)
    if len(prefix) != 8:
        msg = "file shorter than 8 bytes"
        raise SafetensorsError(msg)
    header_len = struct.unpack("<Q", prefix)[0]
    if header_len > MAX_HEADER or header_len > file_size - 8:
        msg = f"header length {header_len} is out of range"
        raise SafetensorsError(msg)
    raw = f.read(header_len)
    if len(raw) != header_len:
        msg = "truncated header"
        raise SafetensorsError(msg)
    if not raw.startswith(b"{"):
        msg = "header does not start with '{'"
        raise SafetensorsError(msg)

    header, duplicates = _parse_header(raw)
    metadata = header.pop("__metadata__", {})
    info = SafetensorsInfo(header_len, metadata if isinstance(metadata, dict) else {})
    if duplicates:
        info.problems.append(f"duplicate JSON keys: {', '.join(sorted(set(duplicates))[:5])}")
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        info.problems.append("__metadata__ is not a string-to-string map")

    spans: list[tuple[int, int, str]] = []
    h = hashlib.sha256()
    for name in sorted(header):
        entry = header[name]
        try:
            dtype = entry["dtype"]
            shape = entry["shape"]
            begin, end = entry["data_offsets"]
            width = DTYPE_BYTES[dtype]
            count = 1
            for d in shape:
                if not isinstance(d, int) or isinstance(d, bool) or d < 0:
                    raise TypeError(name)
                count *= d
            if not (isinstance(begin, int) and isinstance(end, int) and 0 <= begin <= end):
                raise TypeError(name)
        except (KeyError, TypeError, ValueError):
            info.problems.append(f"malformed tensor entry {name!r}")
            continue
        if end - begin != count * width:
            info.problems.append(f"{name!r}: byte span does not match shape and dtype")
        info.tensor_count += 1
        info.params += count
        info.dtype_params[dtype] = info.dtype_params.get(dtype, 0) + count
        spans.append((begin, end, name))
        h.update(name.encode() + b"\0" + ",".join(map(str, shape)).encode() + b"\n")
    info.shape_sha256 = h.hexdigest()

    cursor = 0
    for begin, end, name in sorted(spans):
        if begin < cursor:
            info.problems.append(f"{name!r} overlaps another tensor")
        elif begin > cursor:
            info.problems.append(f"{begin - cursor} unreferenced bytes before {name!r}")
        cursor = max(cursor, end)
    data_len = file_size - 8 - header_len
    if cursor > data_len:
        info.problems.append("tensor data extends past end of file")
    elif cursor < data_len:
        info.problems.append(f"{data_len - cursor} unreferenced bytes after the last tensor")
    return info
