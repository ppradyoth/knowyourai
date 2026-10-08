from __future__ import annotations

import io
import pickletools
import re
import struct
import zipfile
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import BinaryIO

MAX_PICKLES = 64
MAX_MEMBER = 256 << 20

_BUILTIN_TYPES = (
    "set frozenset dict list tuple int float complex bool str bytes bytearray slice range"
).split()
SAFE_GLOBALS = frozenset(
    {
        "collections.OrderedDict",
        "collections.defaultdict",
        "torch._utils._rebuild_tensor",
        "torch._utils._rebuild_tensor_v2",
        "torch._utils._rebuild_parameter",
        "torch._utils._rebuild_parameter_with_state",
        "torch._utils._rebuild_qtensor",
        "torch._tensor._rebuild_from_type_v2",
        "torch.nn.parameter.Parameter",
        "torch.serialization._get_layout",
        "torch.Size",
        "torch.device",
        "torch.Tensor",
        "numpy.core.multiarray._reconstruct",
        "numpy._core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
        "numpy._core.multiarray.scalar",
        "numpy.ndarray",
        "numpy.dtype",
        "_codecs.encode",
        *(f"{mod}.{name}" for mod in ("builtins", "__builtin__") for name in _BUILTIN_TYPES),
    }
)
_SAFE_PATTERN = re.compile(r"^torch\.(?:\w+Storage|storage\.(?:Untyped|Typed)Storage)$")

DANGEROUS_ROOTS = frozenset(
    "os posix nt subprocess sys runpy importlib socket shutil pty commands webbrowser requests "
    "urllib urllib2 http httplib ftplib smtplib ctypes _ctypes multiprocessing pickle _pickle "
    "cPickle code codeop pdb bdb timeit pip asyncio zipimport marshal types tempfile".split()
)
_DANGEROUS_BUILTINS = (
    "eval exec compile open getattr setattr delattr __import__ breakpoint input globals locals vars"
).split()
DANGEROUS_GLOBALS = frozenset(
    {
        "operator.attrgetter",
        "operator.methodcaller",
        "functools.partial",
        "numpy.testing._private.utils.runstring",
        "torch.load",
        "torch.hub.load",
        "torch.jit.load",
        "torch.serialization.load",
        "torch.storage._load_from_bytes",
        *(f"{mod}.{name}" for mod in ("builtins", "__builtin__") for name in _DANGEROUS_BUILTINS),
    }
)

_STRING_OPS = frozenset(
    {
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "BINUNICODE8",
        "UNICODE",
        "SHORT_BINSTRING",
        "BINSTRING",
        "STRING",
    }
)
_PUT_OPS = frozenset({"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"})
_GET_OPS = frozenset({"BINGET", "LONG_BINGET", "GET"})
_PARSE_ERRORS = (
    ValueError,
    EOFError,
    IndexError,
    KeyError,
    TypeError,
    OverflowError,
    MemoryError,
    struct.error,
)
_ZIP_ERRORS = (zipfile.BadZipFile, zlib.error, OSError, RuntimeError, NotImplementedError)


def is_safe(name: str) -> bool:
    return name in SAFE_GLOBALS or bool(_SAFE_PATTERN.match(name))


def is_dangerous(name: str) -> bool:
    return name in DANGEROUS_GLOBALS or name.split(".", 1)[0] in DANGEROUS_ROOTS


@dataclass
class PickleProbe:
    globals: list[str] = field(default_factory=list)
    unresolved: int = 0
    trailing: bool = False
    error: str | None = None

    def add(self, name: str) -> None:
        if name not in self.globals:
            self.globals.append(name)

    @property
    def dangerous(self) -> list[str]:
        return [g for g in self.globals if is_dangerous(g)]

    @property
    def unknown(self) -> list[str]:
        return [g for g in self.globals if not is_safe(g) and not is_dangerous(g)]


def _stack_global(history: deque[tuple[str, object]], memo: dict[int, object]) -> str | None:
    found: list[str] = []
    for name, arg in reversed(history):
        if name in _PUT_OPS:
            continue
        if name in _STRING_OPS:
            value = arg
        elif name in _GET_OPS and isinstance(arg, int):
            value = memo.get(arg)
        else:
            return None
        if not isinstance(value, str):
            return None
        found.append(value)
        if len(found) == 2:
            return f"{found[1]}.{found[0]}"
    return None


def _scan_one(stream: BinaryIO, result: PickleProbe) -> None:
    history: deque[tuple[str, object]] = deque(maxlen=16)
    memo: dict[int, object] = {}
    last: object = None
    for op, arg, _pos in pickletools.genops(stream):
        name = op.name
        if name == "GLOBAL":
            result.add(str(arg).replace(" ", ".", 1))
            last = None
        elif name == "STACK_GLOBAL":
            found = _stack_global(history, memo)
            if found is None:
                result.unresolved += 1
            else:
                result.add(found)
            last = None
        elif name in _STRING_OPS:
            last = arg
        elif name in _GET_OPS and isinstance(arg, int):
            last = memo.get(arg)
        elif name == "MEMOIZE":
            memo[len(memo)] = last
        elif name in _PUT_OPS and isinstance(arg, int):
            memo[arg] = last
        else:
            last = None
        history.append((name, arg))


def probe(stream: BinaryIO) -> PickleProbe:
    result = PickleProbe()
    try:
        for _ in range(MAX_PICKLES):
            _scan_one(stream, result)
            following = stream.read(1)
            if not following:
                return result
            stream.seek(-1, io.SEEK_CUR)
            if following != b"\x80":
                result.trailing = True
                return result
        result.trailing = True
    except _PARSE_ERRORS as e:
        result.error = f"{type(e).__name__}: {e}"[:200]
    return result


def probe_zip(f: BinaryIO) -> tuple[dict[str, PickleProbe], str | None]:
    probes: dict[str, PickleProbe] = {}
    try:
        with zipfile.ZipFile(f) as z:
            for info in z.infolist():
                if not info.filename.endswith((".pkl", ".pickle")):
                    continue
                if info.file_size > MAX_MEMBER:
                    return probes, f"{info.filename} is larger than {MAX_MEMBER} bytes"
                probes[info.filename] = probe(io.BytesIO(z.read(info)))
    except _ZIP_ERRORS as e:
        return probes, f"{type(e).__name__}: {e}"[:200]
    return probes, None
