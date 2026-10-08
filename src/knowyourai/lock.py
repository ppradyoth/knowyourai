from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from knowyourai import __version__
from knowyourai.hub import HubError
from knowyourai.model import Finding

if TYPE_CHECKING:
    from pathlib import Path

    from knowyourai.model import Component

LOCK_VERSION = 1
REMOTE_HASH_LIMIT = 16 << 20
MAX_CHANGES = 6


class LockError(Exception):
    pass


def ensure_digests(c: Component) -> None:
    for f in c.files:
        if f.sha256 or (f.remote and f.size > REMOTE_HASH_LIMIT):
            continue
        try:
            f.sha256, f.digest_source = f.hash_content(), "computed"
        except (OSError, HubError):
            c.unverified.append(f"{f.path}: could not be hashed")


def entry(c: Component) -> dict[str, Any]:
    weights = [
        {k: w[k] for k in ("path", "format", "params", "shape_sha256") if k in w}
        for w in c.evidence.get("weights") or []
    ]
    return {
        "id": c.id,
        "store": c.store,
        "revision": c.revision,
        "files": [
            {"path": f.path, "size": f.size, "sha256": f.sha256}
            for f in sorted(c.files, key=lambda f: f.path)
        ],
        "templates": dict(sorted((c.evidence.get("templates") or {}).items())),
        "weights": sorted(weights, key=lambda w: w["path"]),
        "license": sorted(c.claims("license")),
        "base_model": sorted(c.claims("base_model")),
    }


def build(components: list[Component]) -> dict[str, Any]:
    entries = sorted((entry(c) for c in components), key=lambda e: (e["id"], e["revision"] or ""))
    return {
        "lockfile_version": LOCK_VERSION,
        "generator": f"knowyourai {__version__}",
        "components": entries,
    }


def dumps(lock: dict[str, Any]) -> str:
    return json.dumps(lock, indent=2, sort_keys=True) + "\n"


def load(path: Path) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text())
    except OSError as e:
        msg = f"{path}: {e.strerror or e}"
        raise LockError(msg) from e
    except ValueError as e:
        msg = f"{path} is not valid JSON"
        raise LockError(msg) from e
    if not isinstance(lock, dict) or lock.get("lockfile_version") != LOCK_VERSION:
        msg = f"{path} is not a version {LOCK_VERSION} knowyourai lockfile"
        raise LockError(msg)
    if not isinstance(lock.get("components"), list):
        msg = f"{path} has no component list"
        raise LockError(msg)
    return lock


def _short(revision: str | None) -> str:
    return (revision or "none")[:12]


def compare(locked: dict[str, Any], current: Component) -> list[Finding]:
    changes: list[str] = []
    if locked.get("revision") != current.revision:
        changes.append(
            f"revision {_short(locked.get('revision'))} is now {_short(current.revision)}"
        )
    before = {f["path"]: f for f in locked.get("files") or []}
    after = {f.path: f for f in current.files}
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old is None:
            changes.append(f"{path} added")
        elif new is None:
            changes.append(f"{path} removed")
        elif old.get("sha256") and new.sha256 and old["sha256"] != new.sha256:
            changes.append(f"{path} content changed")
        elif old.get("size") != new.size:
            changes.append(f"{path} size changed")
    old_templates = locked.get("templates") or {}
    new_templates = current.evidence.get("templates") or {}
    changes += [
        f"chat template {source} changed"
        for source in sorted(set(old_templates) | set(new_templates))
        if old_templates.get(source) != new_templates.get(source)
    ]
    if not changes:
        return []
    more = f" (+{len(changes) - MAX_CHANGES} more)" if len(changes) > MAX_CHANGES else ""
    return [
        Finding(
            "LOCK002",
            "high",
            "Component changed since it was locked",
            "; ".join(changes[:MAX_CHANGES]) + more,
            fix="If the change is expected, review it and run `knowyourai lock` again.",
        )
    ]
