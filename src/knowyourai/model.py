from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}
CHUNK = 1 << 20


@dataclass
class FileRef:
    path: str
    size: int
    sha256: str | None = None
    digest_source: str | None = None
    local_path: Path | None = None
    opener: Callable[[], BinaryIO] | None = None

    @property
    def remote(self) -> bool:
        return self.local_path is None

    def open(self) -> BinaryIO:
        if self.local_path is not None:
            return self.local_path.open("rb")
        if self.opener is None:
            raise FileNotFoundError(self.path)
        return self.opener()

    def read_bytes(self, limit: int) -> bytes:
        with self.open() as f:
            data = f.read(limit + 1)
        if len(data) > limit:
            msg = f"{self.path} is larger than {limit} bytes"
            raise ValueError(msg)
        return data

    def hash_content(self) -> str:
        h = hashlib.sha256()
        with self.open() as f:
            while chunk := f.read(CHUNK):
                h.update(chunk)
        return h.hexdigest()


@dataclass
class Finding:
    rule: str
    severity: str
    title: str
    evidence: str
    fix: str = ""
    file: str = ""


@dataclass
class Component:
    id: str
    store: str
    location: str
    revision: str | None = None
    refs: list[str] = field(default_factory=list)
    files: list[FileRef] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    templates: dict[str, str] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)

    def flag(
        self,
        rule: str,
        severity: str,
        title: str,
        evidence: str,
        *,
        fix: str = "",
        file: str = "",
    ) -> None:
        self.findings.append(Finding(rule, severity, title, evidence, fix, file))

    def claim(self, kind: str, value: str, source: str) -> None:
        value = value.strip()
        if not value:
            return
        claims = self.evidence.setdefault("claims", {}).setdefault(kind, [])
        if [value, source] not in claims:
            claims.append([value, source])

    def claims(self, kind: str) -> list[str]:
        seen: list[str] = []
        for value, _source in self.evidence.get("claims", {}).get(kind, []):
            if value not in seen:
                seen.append(value)
        return seen

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def status(self) -> str:
        top = max((SEVERITY_ORDER[f.severity] for f in self.findings), default=0)
        if top >= SEVERITY_ORDER["high"]:
            return "violation"
        if top == SEVERITY_ORDER["medium"]:
            return "review"
        if self.unverified:
            return "unverifiable"
        return "consistent"
