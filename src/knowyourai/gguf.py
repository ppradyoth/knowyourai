from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
MAX_KV = 65_536
MAX_STRING = 16 << 20
MAX_NAME = 1 << 16
MAX_ARRAY = 8_000_000
MAX_TENSORS = 1_000_000
MAX_DIMS = 8
INLINE_ARRAY = 64
CHUNK = 1 << 20

_STRING = 8
_ARRAY = 9
_SCALARS: dict[int, tuple[str, int, str]] = {
    0: ("B", 1, "u8"),
    1: ("b", 1, "i8"),
    2: ("H", 2, "u16"),
    3: ("h", 2, "i16"),
    4: ("I", 4, "u32"),
    5: ("i", 4, "i32"),
    6: ("f", 4, "f32"),
    7: ("?", 1, "bool"),
    10: ("Q", 8, "u64"),
    11: ("q", 8, "i64"),
    12: ("d", 8, "f64"),
}
GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
}


class GGUFError(ValueError):
    pass


@dataclass
class GGUFInfo:
    version: int
    endian: str
    tensor_count: int
    metadata: dict[str, Any]
    params: int = 0
    dtype_params: dict[str, int] = field(default_factory=dict)
    shape_sha256: str = ""
    header_bytes: int = 0

    @property
    def dominant_dtype(self) -> str:
        if not self.dtype_params:
            return ""
        return max(self.dtype_params.items(), key=lambda kv: kv[1])[0]


class _Reader:
    def __init__(self, f: BinaryIO) -> None:
        self.f = f
        self.endian = "<"

    def take(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) != n:
            msg = f"truncated: wanted {n} bytes, got {len(data)}"
            raise GGUFError(msg)
        return data

    def scalar(self, vtype: int) -> Any:
        fmt, width, _name = _SCALARS[vtype]
        return struct.unpack(self.endian + fmt, self.take(width))[0]

    def u32(self) -> int:
        return struct.unpack(self.endian + "I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack(self.endian + "Q", self.take(8))[0]

    def string(self, limit: int = MAX_STRING) -> bytes:
        n = self.u64()
        if n > limit:
            msg = f"string of {n} bytes exceeds limit {limit}"
            raise GGUFError(msg)
        return self.take(n)


def _array(r: _Reader, depth: int) -> Any:
    etype = r.u32()
    count = r.u64()
    if count > MAX_ARRAY:
        msg = f"array of {count} elements exceeds limit {MAX_ARRAY}"
        raise GGUFError(msg)
    if count <= INLINE_ARRAY:
        return [_value(r, etype, depth + 1) for _ in range(count)]
    h = hashlib.sha256()
    if etype in _SCALARS:
        remaining = count * _SCALARS[etype][1]
        while remaining:
            chunk = r.take(min(remaining, CHUNK))
            h.update(chunk)
            remaining -= len(chunk)
        name = _SCALARS[etype][2]
    elif etype == _STRING:
        for _ in range(count):
            raw = r.string()
            h.update(len(raw).to_bytes(8, "little"))
            h.update(raw)
        name = "string"
    else:
        msg = f"unsupported large array of type {etype}"
        raise GGUFError(msg)
    return {"array": name, "count": count, "sha256": h.hexdigest()}


def _value(r: _Reader, vtype: int, depth: int = 0) -> Any:
    if vtype in _SCALARS:
        return r.scalar(vtype)
    if vtype == _STRING:
        return r.string().decode("utf-8", "replace")
    if vtype == _ARRAY:
        if depth >= 2:
            msg = "arrays nested too deeply"
            raise GGUFError(msg)
        return _array(r, depth)
    msg = f"unknown value type {vtype}"
    raise GGUFError(msg)


def read_gguf(f: BinaryIO) -> GGUFInfo:
    r = _Reader(f)
    if r.take(4) != MAGIC:
        msg = "not a GGUF file"
        raise GGUFError(msg)
    raw_version = r.take(4)
    version = struct.unpack("<I", raw_version)[0]
    if version not in SUPPORTED_VERSIONS:
        swapped = struct.unpack(">I", raw_version)[0]
        if swapped not in SUPPORTED_VERSIONS:
            msg = f"unsupported GGUF version {version}"
            raise GGUFError(msg)
        version = swapped
        r.endian = ">"
    tensor_count = r.u64()
    kv_count = r.u64()
    if tensor_count > MAX_TENSORS:
        msg = f"{tensor_count} tensors exceeds limit {MAX_TENSORS}"
        raise GGUFError(msg)
    if kv_count > MAX_KV:
        msg = f"{kv_count} metadata keys exceeds limit {MAX_KV}"
        raise GGUFError(msg)

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = r.string(MAX_NAME).decode("utf-8", "replace")
        if key in metadata:
            msg = f"duplicate metadata key {key!r}"
            raise GGUFError(msg)
        metadata[key] = _value(r, r.u32())

    info = GGUFInfo(version, r.endian, tensor_count, metadata)
    entries: list[tuple[bytes, tuple[int, ...]]] = []
    for _ in range(tensor_count):
        name = r.string(MAX_NAME)
        n_dims = r.u32()
        if n_dims > MAX_DIMS:
            msg = f"tensor with {n_dims} dimensions"
            raise GGUFError(msg)
        dims = tuple(r.u64() for _ in range(n_dims))
        dtype = GGML_TYPES.get(r.u32(), "unknown")
        r.u64()
        count = 1
        for d in dims:
            count *= d
        info.params += count
        info.dtype_params[dtype] = info.dtype_params.get(dtype, 0) + count
        entries.append((name, dims))

    h = hashlib.sha256()
    for name, dims in sorted(entries):
        h.update(name + b"\0" + ",".join(map(str, dims)).encode() + b"\n")
    info.shape_sha256 = h.hexdigest()
    info.header_bytes = f.tell()
    return info
