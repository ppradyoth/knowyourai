from __future__ import annotations

import argparse
import sys
from pathlib import Path

from knowyourai import __version__, checks, discover, lock, online, report
from knowyourai.model import SEVERITY_ORDER, Component

DESCRIPTION = "Check that the open-weight models on your machine are what they claim to be."
LOCK_RULES = frozenset({"LOCK001", "LOCK002", "LOCK003"})


def _inspect(c: Component, args: argparse.Namespace) -> None:
    checks.inspect(c)
    if args.rehash:
        checks.verify_digests(c)
    if args.online:
        online.enrich(c)


def _emit(components: list[Component], args: argparse.Namespace) -> None:
    if args.json:
        print(report.to_json(components))
    else:
        print(report.render(components, online=args.online))


def _scan(args: argparse.Namespace) -> int:
    components = discover.discover(args.targets, online=args.online, file_pattern=args.file)
    for c in components:
        _inspect(c, args)
    _emit(components, args)
    if args.fail_on:
        floor = SEVERITY_ORDER[args.fail_on]
        if any(SEVERITY_ORDER[f.severity] >= floor for c in components for f in c.findings):
            return 1
    return 0


def _lock(args: argparse.Namespace) -> int:
    components = discover.discover(args.targets, online=args.online, file_pattern=args.file)
    for c in components:
        _inspect(c, args)
        lock.ensure_digests(c)
    Path(args.output).write_text(lock.dumps(lock.build(components)))
    print(f"locked {len(components)} components in {args.output}")
    for c in components:
        if c.status == "violation":
            print(f"warning: {c.id} has high-severity findings and is now pinned as it is")
    return 0


def _candidates(
    entry: dict[str, object], local: dict[str, list[Component]], args: argparse.Namespace
) -> list[Component]:
    ident, store = str(entry["id"]), entry.get("store")
    if store in ("dir", "file"):
        path = Path(ident.split(":", 1)[1])
        return [c for c in discover.scan_path(path) if c.id == ident] if path.exists() else []
    if store == "hf-hub":
        if not args.online:
            return []
        revision = entry.get("revision")
        repo = ident.removeprefix("hf:")
        return [discover.remote_hf(repo, str(revision) if revision else None, None)]
    return local.get(ident, [])


def _check(args: argparse.Namespace) -> int:
    locked = lock.load(Path(args.lock))
    local: dict[str, list[Component]] = {}
    for c in discover.local_stores():
        local.setdefault(c.id, []).append(c)
    results: list[Component] = []
    for entry in locked["components"]:
        found = _candidates(entry, local, args)
        exact = [c for c in found if c.revision == entry.get("revision")]
        if not found:
            missing = Component(entry["id"], str(entry.get("store")), "-", entry.get("revision"))
            reason = (
                "it lives on the Hub: add --online to verify it"
                if entry.get("store") == "hf-hub"
                else "it is not on this machine"
            )
            missing.flag(
                "LOCK003",
                "medium",
                "Locked component could not be verified",
                reason,
                fix="Download the locked revision, or remove the entry from the lockfile.",
            )
            results.append(missing)
            continue
        target = (exact or found)[0]
        _inspect(target, args)
        lock.ensure_digests(target)
        target.findings += lock.compare(entry, target)
        results.append(target)
    if args.strict:
        known = {entry["id"] for entry in locked["components"]}
        for ident, extras in local.items():
            if ident in known:
                continue
            extras[0].flag(
                "LOCK001",
                "medium",
                "Component is not in the lockfile",
                f"{ident} is on this machine but was never locked",
                fix="Lock it after review, or remove it.",
            )
            results.append(extras[0])
    _emit(results, args)
    findings = [f for c in results for f in c.findings]
    failed = any(f.severity == "high" for f in findings) or (
        args.strict and any(f.rule in LOCK_RULES for f in findings)
    )
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--online",
        action="store_true",
        help="compare against the publisher (talks only to huggingface.co and registry.ollama.ai)",
    )
    common.add_argument("--rehash", action="store_true", help="recompute every file digest")
    common.add_argument("--json", action="store_true", help="machine-readable output")
    targets = argparse.ArgumentParser(add_help=False)
    targets.add_argument(
        "targets",
        nargs="*",
        help="paths, hf:org/name[@revision] or ollama:name:tag (default: every local store)",
    )
    targets.add_argument("--file", metavar="GLOB", help="which files of a Hub repo to inspect")

    parser = argparse.ArgumentParser(prog="knowyourai", description=DESCRIPTION)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", parents=[targets, common], help="inventory and inspect models")
    scan.add_argument("--fail-on", choices=("low", "medium", "high"), help="exit 1 at this level")
    scan.set_defaults(run=_scan)
    locker = sub.add_parser("lock", parents=[targets, common], help="pin what is there now")
    locker.add_argument("-o", "--output", default="ai.lock")
    locker.set_defaults(run=_lock)
    check = sub.add_parser("check", parents=[common], help="fail if anything locked has changed")
    check.add_argument("--lock", default="ai.lock")
    check.add_argument("--strict", action="store_true", help="also fail on unlocked or missing")
    check.set_defaults(run=_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except (discover.TargetError, lock.LockError) as e:
        print(f"knowyourai: {e}", file=sys.stderr)
        return 2
