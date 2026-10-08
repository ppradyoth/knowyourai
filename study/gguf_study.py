from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from knowyourai import checks, hub, online
from knowyourai.model import Component

LIST_URL = "https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit={n}"
DESCRIPTION = (
    "Measure GGUF repos on the Hugging Face Hub using metadata only: chat template, licence "
    "and base-model claims compared with the declared base model. Nothing is downloaded."
)


def measure(repo: str) -> dict[str, Any]:
    info = hub.hf_model_info(repo)
    c = Component(f"hf:{repo}", "hf-hub", f"https://huggingface.co/{repo}", info.get("sha"))
    chat = (info.get("gguf") or {}).get("chat_template")
    if isinstance(chat, str) and chat.strip():
        c.templates["gguf#chat_template"] = chat
    checks.analyse_templates(c)
    online.enrich(c)
    compared = c.evidence.get("template_vs_base") or {}
    return {
        "repo": repo,
        "revision": c.revision,
        "downloads": info.get("downloads"),
        "has_template": bool(c.templates),
        "license": c.claims("license"),
        "base_model": c.claims("base_model"),
        "relation": c.evidence.get("base_relation"),
        "template_vs_base": next(iter(compared.values()), None),
        "findings": [
            {"rule": f.rule, "severity": f.severity, "evidence": f.evidence} for f in c.findings
        ],
        "skipped": c.skipped,
    }


def summarise(rows: list[dict[str, Any]]) -> str:
    ok = [r for r in rows if "error" not in r]
    with_template = [r for r in ok if r["has_template"]]
    compared = Counter(r["template_vs_base"] or "not comparable" for r in with_template)
    findings = Counter(
        f"{f['rule']} {f['severity']}"
        for r in ok
        for f in {(x["rule"], x["severity"]): x for x in r["findings"]}.values()
    )
    lines = [
        f"repos measured: {len(ok)} ({len(rows) - len(ok)} errors)",
        f"declare a base model: {sum(bool(r['base_model']) for r in ok)}",
        f"declare a licence: {sum(bool(r['license']) for r in ok)}",
        f"have a chat template: {len(with_template)}",
        "template compared with base: "
        + ", ".join(f"{kind} {count}" for kind, count in compared.most_common()),
        "repos with each finding:",
        *(f"  {name}: {count}" for name, count in sorted(findings.items())),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--limit", type=int, default=100, help="how many repos, by downloads")
    parser.add_argument("--out", type=Path, default=Path("study/out/gguf.jsonl"))
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds between repos")
    args = parser.parse_args()

    rows = (
        [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        if args.out.exists()
        else []
    )
    done = {r["repo"] for r in rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    repos = [m["id"] for m in hub.get_json(LIST_URL.format(n=args.limit))]
    with args.out.open("a") as out:
        for i, repo in enumerate(repos, 1):
            if repo in done:
                continue
            try:
                row = measure(repo)
            except hub.HubError as e:
                row = {"repo": repo, "error": str(e)}
                if e.status == 429:
                    print(f"rate limited after {i - 1} repos; re-run to resume", file=sys.stderr)
                    break
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{i}/{len(repos)}] {repo}", file=sys.stderr)
            time.sleep(args.sleep)
    print(summarise(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
