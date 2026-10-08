# knowyourai

[![PyPI](https://img.shields.io/pypi/v/knowyourai)](https://pypi.org/project/knowyourai/)
[![Python](https://img.shields.io/pypi/pyversions/knowyourai)](https://pypi.org/project/knowyourai/)
[![CI](https://github.com/ppradyoth/knowyourai/actions/workflows/ci.yml/badge.svg)](https://github.com/ppradyoth/knowyourai/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/knowyourai)](LICENSE)

**Is the model on your machine the model it claims to be?**

You pulled a 5 GB file from a stranger's repo. The model card says which model it is, what licence
it has, and that nothing odd happens when you chat with it. `knowyourai` checks those claims
against the bytes, and tells you when something changes after you approved it.

```bash
uvx knowyourai scan
```

```text
STATUS      COMPONENT                     REVISION      FORMAT            SIZE
review      hf:someone/Model-GGUF         b17cb02dd882  gguf Q4_K         4.4 GB
consistent  hf:org/embedding-model        1110a243fdf4  safetensors F32   87.3 MB

hf:someone/Model-GGUF
  [high] TMPL002 Chat template changes behaviour when a message contains specific text
      evidence: model.gguf#chat_template: branches on 'wire the funds'
      fix: Read the branch. A template has no reason to react to what a user says.
```

- **No setup.** One command finds every model in the Hugging Face cache, Ollama and LM Studio.
- **Never runs a model.** It reads headers and metadata. No torch, no GPU, no dependencies.
- **Offline by default.** No telemetry. `--online` talks only to the model's own registry.
- **Check before you pull.** `scan hf:org/name --online` inspects a repo on the Hub by reading file
  headers over range requests. A 471 GB repo takes about 9 seconds and downloads nothing.
- **A lockfile for models.** `lock` pins what you approved; `check` fails CI when it moves.

## Why

In the 200 most-downloaded GGUF repos on Hugging Face (October 2026), 128 ship a chat template
that can be compared with their declared base model's. **61 of them, 48%, differ.** Most of those
are deliberate fixes by the quantizer. The point is that you are often not running the template
the original publisher wrote, and a chat template is a program that runs on every prompt.

Model scanners look for malware in a file. This tool asks a different question: is this the model
the label says it is, and has it changed since you approved it?

Status: alpha, published on [PyPI](https://pypi.org/project/knowyourai/).

## Install

```bash
uvx knowyourai scan          # run without installing
pip install knowyourai       # or install it
```

Python 3.11 or newer. No dependencies.

## Tested on real models

Run against real repos on the Hugging Face Hub, with nothing downloaded:

| Repo | Listed size | Time | Result |
|---|---|---|---|
| `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | 471.6 GB | 7.4 s | template differs from the base model's (a labelled fix) |
| `Qwen/Qwen3-4B-GGUF` | 14.7 GB | 6.5 s | template differs from the publisher's own safetensors repo |
| `google/gemma-4-E2B-it-qat-q4_0-gguf` | 4.0 GB | 14.6 s | template identical, structure matches |
| `jinaai/jina-embeddings-v3` | 5.4 GB | 9.8 s | config loads model code from a second repository |
| `microsoft/Phi-3-mini-4k-instruct` | 7.1 GB | 5.4 s | config maps to Python code shipped in the repo |

Fourteen remote scans, a local run with digest verification, the 200-repo study with a
per-publisher breakdown, 21 attack fixtures, the three bugs real data exposed in this tool, and
what has not been tested yet are all in **[FIELD-TESTS.md](FIELD-TESTS.md)**.

## Use it

```bash
uvx knowyourai scan                        # everything on this machine
uvx knowyourai scan ./models               # a directory
uvx knowyourai scan hf:org/name --online   # a Hub repo, without downloading it
uvx knowyourai scan --online               # compare local models with their publishers
uvx knowyourai lock hf:org/name            # write ai.lock
uvx knowyourai check                       # exit 1 if anything locked has changed
```

| Command | What it does |
|---|---|
| `scan [targets]` | Inventory and inspect. `--fail-on high` makes it a CI gate. |
| `lock [targets]` | Write `ai.lock`: revisions, file digests and chat-template digests. |
| `check` | Exit 1 if anything in `ai.lock` has changed. `--strict` also fails on missing or unlocked components. |

Targets are paths, `hf:org/name[@revision]` or `ollama:name:tag`. `kyai` is a short alias.

### In CI

```yaml
- uses: ppradyoth/knowyourai@v0.1.0
  with:
    command: check
```

### As a pre-commit hook

```yaml
repos:
  - repo: https://github.com/ppradyoth/knowyourai
    rev: v0.1.0
    hooks:
      - id: knowyourai-check
```

## What it checks

Offline:

| Rule | Finding |
|---|---|
| `FMT001` | File content is not the format its extension claims |
| `FMT002` | safetensors header is invalid, or the file has unreferenced bytes or overlapping tensors |
| `FMT004` | GGUF header could not be parsed within safe limits |
| `EXEC001` | Weights are a pickle (low if every import is a tensor rebuild function, medium otherwise) |
| `EXEC002` | Pickle imports a dangerous callable, has an unresolvable import, or is malformed |
| `EXEC003` | Config `auto_map` points to Python code (high if the code lives in another repo) |
| `EXEC004` | Model ships Python files |
| `EXEC005` | Config sets `_attn_implementation_internal`, a reported indicator of CVE-2026-4372 |
| `TMPL001` | Chat template reaches into Python internals |
| `TMPL002` | Chat template branches on the text of a message |
| `TMPL003` | Chat template contains a hard-coded URL |
| `INTEG001` | With `--rehash`: a file no longer matches the digest it is stored under |

With `--online`, against the publisher:

| Rule | Finding |
|---|---|
| `DRIFT001` | The Hub's `main` has moved since the local copy was downloaded |
| `DRIFT002` | The Ollama tag now points to different layers |
| `CLAIM002` | Licence differs from the declared base model's (high if it is more permissive) |
| `CLAIM003` | Layer count or hidden size does not match the declared base model |
| `CLAIM004` | No licence is declared anywhere |
| `TMPL010` | Chat template differs from the base model's |

When a template differs from the base model's, `TMPL002` and `TMPL003` hits that the base model
does not have are raised to high. Hits the base model also has are lowered to low.

## Statuses

| Status | Meaning |
|---|---|
| `violation` | At least one high finding |
| `review` | At least one medium finding |
| `unverifiable` | Something could not be inspected, for example an ONNX file |
| `consistent` | Nothing contradicted the model's claims in the checks that ran |

`consistent` is not a safety verdict. Offline, nothing is compared against the publisher.

## Network use

Without `--online` the tool makes no network requests. With it, requests go only to
`huggingface.co` (and the CDN its downloads redirect to) and `registry.ollama.ai`. There is no
telemetry. `HF_TOKEN`, if set, is sent to `huggingface.co` only, for gated models.

## Limits

- It does not judge behaviour. A backdoor trained into the weights is invisible to it.
- Pickle, ONNX, Keras and TensorFlow files are not deeply analysed. Use ModelAudit or fickling.
- Licence and base-model checks compare declared metadata, not the weights themselves.
- Template rules are heuristics. A finding means "read this", not "this is malicious".

## Measuring the Hub

`study/gguf_study.py` measures the most-downloaded GGUF repos using Hub metadata only. For each
repo it compares the chat template and licence with the declared base model's. It downloads
nothing, resumes if interrupted, and writes one JSON line per repo.

```bash
uv run python study/gguf_study.py --limit 200 --out study/out/gguf.jsonl
```

The template it reads is the one the Hub parsed from the repo, and the base model's template is
whatever its `main` holds today. A difference means the two are not the same now, not that the
quantizer changed anything.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check
```
