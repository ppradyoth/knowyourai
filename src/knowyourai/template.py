from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field

MAX_HITS = 5

_SEGMENT = re.compile(r"(\{%.*?%\}|\{\{.*?\}\}|\{#.*?#\})", re.DOTALL)
_SQUASH = re.compile(r"\s+|(?<=\{[%{#])-|-(?=[%}#]\})")
_URL = re.compile(r"https?://[^\s'\"<>)}\\]+")
_SSTI = re.compile(r"__\w+__|\b(?:lipsum|cycler|joiner)\s*\.|\|\s*attr\s*\(")
# Each branch consumes a distinct first character and the gaps are bounded, so matching stays
# linear on hostile templates. The earlier backreference form took minutes on a 7 KB template.
_LIT = r"""(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")"""
_CONTENT = r"""(?:\bcontent\b|\[\s*['"]content['"]\s*\]|\.content\b)"""
_TRIGGERS = (
    re.compile(_LIT + r"\s+in\s+[\w.\[\]'\" ]{0,80}?" + _CONTENT),
    re.compile(
        _CONTENT
        + r"[\w.\[\]'\" |]{0,80}?\.\s*(?:startswith|endswith|find|index|count)\s*\(\s*"
        + _LIT
    ),
    re.compile(_CONTENT + r"\s*(?:\|\s*\w+\s*){0,4}(?:==|!=)\s*" + _LIT),
)
MAX_ANALYSED = 1 << 20
_GO_TRIGGER = re.compile(
    r"""\b(?:contains|hasPrefix|hasSuffix|eq|ne)\s+"""
    r"""(?:\.(?:Content|Prompt|System)\s+"([^"]*)"|"([^"]*)"\s+\.(?:Content|Prompt|System))"""
)
_WORD = re.compile(r"[A-Za-z]{3,}")
_ESCAPES = re.compile(r"\\[ntr]")
# Message-schema keys that templates legitimately test for with `'key' in content`.
_SCHEMA_KEYS = frozenset(
    "type text image image_url video video_url audio audio_url input_audio name arguments "
    "function tool_calls tool_call_id role content id index parameters description properties "
    "required thinking reasoning_content".split()
)


@dataclass(frozen=True)
class Hit:
    rule: str
    severity: str
    title: str
    evidence: str


@dataclass
class Comparison:
    kind: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def digest(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def _squash(text: str) -> str:
    return _SQUASH.sub("", text)


def _natural(literal: str) -> bool:
    stripped = _ESCAPES.sub(" ", literal).strip()
    if stripped.lower() in _SCHEMA_KEYS or not _WORD.search(stripped):
        return False
    return not stripped.startswith(("<", "[", "{", "|", "#", "/"))


def _snippet(text: str, start: int, end: int) -> str:
    return " ".join(text[max(start - 40, 0) : end + 40].split())


def _jinja_literals(text: str) -> list[str]:
    literals: list[str] = []
    for segment in _SEGMENT.findall(text):
        if segment.startswith("{#"):
            continue
        for pattern in _TRIGGERS:
            literals += [m.group(1) or m.group(2) or "" for m in pattern.finditer(segment)]
    return literals


def analyse(text: str, *, go: bool = False) -> list[Hit]:
    text = text[:MAX_ANALYSED]
    hits = [
        Hit(
            "TMPL001",
            "high",
            "Chat template reaches into Python internals",
            _snippet(text, m.start(), m.end()),
        )
        for m in _SSTI.finditer(text)
    ]
    literals = [a or b for a, b in _GO_TRIGGER.findall(text)] if go else _jinja_literals(text)
    hits += [
        Hit(
            "TMPL002",
            "medium",
            "Chat template changes behaviour when a message contains specific text",
            f"branches on {literal!r}",
        )
        for literal in literals
        if _natural(literal)
    ]
    hits += [
        Hit("TMPL003", "medium", "Chat template contains a hard-coded URL", url)
        for url in _URL.findall(text)
    ]
    unique = list(dict.fromkeys(hits))
    return [h for i, h in enumerate(unique) if sum(u.rule == h.rule for u in unique[:i]) < MAX_HITS]


def compare(base: str, local: str) -> Comparison:
    if normalize(base) == normalize(local):
        return Comparison("identical")
    if _squash(base) == _squash(local):
        return Comparison("formatting")
    base_segments = [s for s in _SEGMENT.split(base) if s.strip()]
    local_segments = [s for s in _SEGMENT.split(local) if s.strip()]
    matcher = difflib.SequenceMatcher(
        a=[_squash(s) for s in base_segments],
        b=[_squash(s) for s in local_segments],
        autojunk=False,
    )
    result = Comparison("differs")
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            result.removed += base_segments[a0:a1]
        if tag in ("replace", "insert"):
            result.added += local_segments[b0:b1]
    return result
