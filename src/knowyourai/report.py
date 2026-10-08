from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Any

from knowyourai import __version__

if TYPE_CHECKING:
    from knowyourai.model import Component

ORDER = {"violation": 0, "review": 1, "unverifiable": 2, "consistent": 3}
HEADER = ("STATUS", "COMPONENT", "REVISION", "FORMAT", "SIZE")


def human_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _format(c: Component) -> str:
    weights = c.evidence.get("weights") or []
    kinds = sorted({f"{w['format']} {w.get('dtype', '')}".strip() for w in weights})
    return ", ".join(kinds) or "-"


def as_dict(c: Component) -> dict[str, Any]:
    return {
        "id": c.id,
        "store": c.store,
        "revision": c.revision,
        "refs": c.refs,
        "location": c.location,
        "size": c.size,
        "status": c.status,
        "evidence": c.evidence,
        "findings": [vars(f) for f in c.findings],
        "unverified": c.unverified,
        "skipped": c.skipped,
        "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in c.files],
    }


def to_json(components: list[Component]) -> str:
    return json.dumps(
        {"version": __version__, "components": [as_dict(c) for c in components]}, indent=2
    )


def render(components: list[Component], *, online: bool) -> str:
    components = sorted(components, key=lambda c: (ORDER[c.status], c.id))
    total = human_size(sum(c.size for c in components))
    lines = [f"knowyourai {__version__}: {len(components)} components, {total}"]
    if not online:
        lines.append("offline: nothing was compared against the publisher (add --online)")
    lines.append("")
    rows = [
        (c.status, c.id, (c.revision or "-")[:12], _format(c), human_size(c.size))
        for c in components
    ]
    widths = [max(len(row[i]) for row in [HEADER, *rows]) for i in range(len(HEADER))]
    lines += [
        "  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)).rstrip()
        for row in [HEADER, *rows]
    ]
    for c in components:
        if not (c.findings or c.unverified or c.skipped):
            continue
        lines += ["", c.id]
        for f in c.findings:
            lines.append(f"  [{f.severity}] {f.rule} {f.title}")
            lines.append(f"      evidence: {f.evidence}")
            if f.fix:
                lines.append(f"      fix: {f.fix}")
        lines += [f"  not inspected: {note}" for note in c.unverified]
        lines += [f"  skipped: {note}" for note in c.skipped]
    counts = Counter(c.status for c in components)
    summary = ", ".join(f"{counts[s]} {s}" for s in ORDER if counts[s])
    lines += ["", summary or "nothing found"]
    return "\n".join(lines)
