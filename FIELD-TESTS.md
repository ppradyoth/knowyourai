# Field tests

What happened when [`knowyourai` 0.1.0](https://pypi.org/project/knowyourai/0.1.0/) was run against real models. Everything here is tool output
from 8 October 2026, on an Apple Silicon Mac with Python 3.13, with no Hugging Face token set.

A status describes what the checks found in the files and metadata. It is not a judgement of the
publisher. `violation` means "at least one high-severity finding", and several high findings below
describe a documented design choice of the model, not an attack.

## Summary

| Test | Scope | Result |
|---|---|---|
| Local scan, offline | 4 components, 825 MB | 3 consistent, 1 unverifiable, 0.7 s |
| Local scan, `--online` | same | 2 low licence findings, 3.7 s |
| Digest verification, `--rehash` | 825 MB hashed | all digests match, 38 s |
| `lock` then `check` | 3 models | passes; `--strict` flags the unlocked component |
| Remote scan, GGUF | 7 Hub repos, 1.07 TB listed | 5 template differences, 1 licence difference, 0 high findings |
| Remote scan, safetensors and pickle | 7 Hub repos | 1 violation, 3 review, 1 unverifiable, 2 consistent |
| Metadata study | 200 most-downloaded GGUF repos | 61 of 128 comparable templates differ from the base model's |
| Unit and integration tests | 73 tests | all pass |

No repo in any test contained a template trigger phrase, a hard-coded URL in a template, or a
template-escape pattern.

## 1. Local models

Command: `knowyourai scan`

| Component | Revision | Format | Size | Status |
|---|---|---|---|---|
| `hf:HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366` | safetensors BF16 | 259.8 MB | consistent |
| `hf:garak-llm/attackgeneration-toxicity_gpt2` | `565f41dd4435` | safetensors F32 | 477.9 MB | consistent |
| `hf:sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | safetensors F32 | 87.3 MB | consistent |
| `ollama:qwen2:7b` | `dd314f039b9d` | none | 0 B | unverifiable |

- **Time:** 0.7 s for the whole machine.
- **The Ollama entry is a stale manifest.** The manifest lists five layers (model, template,
  licence, params, system) but the blob directory is empty. The tool reports one line: "stale
  manifest: none of its 5 layers are on disk".
- **Parameter counts read from headers:** 134,515,008, 124,439,808 and 22,713,728. The last one
  equals the total the Hub reports for that repo.

### With `--online`

Command: `knowyourai scan --online` (3.7 s)

| Component | Finding |
|---|---|
| `garak-llm/attackgeneration-toxicity_gpt2` | `CLAIM002` low: declares `apache-2.0`; base model `openai-community/gpt2` is `mit` |
| `sentence-transformers/all-MiniLM-L6-v2` | `CLAIM002` low: declares `apache-2.0`; base model `nreimers/MiniLM-L6-H384-uncased` is `mit` |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | Skipped: the base model publishes no chat template to compare against |

Both licences are permissive, so the severity is low. No local revision was behind the Hub.

### Digest verification

Command: `knowyourai scan "hf:*" --rehash` (38 s, 825 MB)

Every file matched the digest it is stored under. Each `model.safetensors` was also hashed
independently with `shasum -a 256` and matched its blob name.

This test found a bug in the tool. `SmolLM2-135M-Instruct` stores its weights blob as a symlink
into a shared store, under a file name that is a different hash:

```text
snapshots/<rev>/model.safetensors -> blobs/5af571cb...   (sha256 of the content)
blobs/5af571cb...                 -> blobs/73/73b079dc... (a different hash)
```

The first version read the digest from the fully resolved path and raised a false `INTEG001`. It
now takes the name of the first link target. A regression test covers the layout.

### Lock and check

| Command | Result |
|---|---|
| `knowyourai lock "hf:*"` | 3 components, 27 files, a 6.4 KB `ai.lock` |
| `knowyourai check` | exit 0, 3 consistent |
| `knowyourai check --strict` | exit 1: `ollama:qwen2:7b` is on the machine but not in the lockfile |

## 2. Remote scans: GGUF repos

Command: `knowyourai scan hf:<repo> --online`. Nothing was downloaded. The tool read one GGUF
header per repo over HTTP range requests (the Q4_K_M file where one exists) and compared it with
the declared base model.

| Repo | Size listed | Time | Template vs base | Structure vs base | Status |
|---|---|---|---|---|---|
| `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | 471.6 GB | 7.4 s | differs (+42, -18) | matches | consistent |
| `bartowski/Qwen2.5-32B-Instruct-GGUF` | 498.8 GB | 6.6 s | differs (+2, -2) | matches | consistent |
| `Qwen/Qwen3-4B-GGUF` | 14.7 GB | 6.5 s | differs (+11, -15) | matches | consistent |
| `google/gemma-4-E2B-it-qat-q4_0-gguf` | 4.0 GB | 14.6 s | identical | matches | consistent |
| `LiquidAI/LFM2.5-230M-GGUF` | 2.7 GB | 8.6 s | GGUF identical; `qad/chat_template.jinja` differs (+4, -2) | matches | consistent |
| `lmstudio-community/Qwen3.5-9B-GGUF` | 21.8 GB | 11.5 s | identical | matches | consistent |
| `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` | 57.4 GB | 15.1 s | differs (+70, -193) | matches | consistent |

"+42, -18" means 42 template segments added and 18 removed relative to the base model's template.

What the differences were:

- **Unsloth.** The added segments begin with a comment that names the change: "Unsloth Chat
  template fix". This is a deliberate, labelled patch.
- **bartowski.** The GGUF metadata names `Qwen/Qwen2.5-32B` as the base, while the model card
  names `Qwen/Qwen2.5-32B-Instruct`. The tool compared against the first one, so the two-segment
  difference (a default system prompt) is the gap between the base and instruct templates. Against
  the instruct model, the Hub metadata study below rates this repo identical.
- **Qwen.** The publisher's own GGUF repo carries a different template from its own safetensors
  repo: tool-handling loops were added and others removed.
- **HauhauCS.** A large rewrite of the Gemma template (193 segments removed). It also declares the
  `gemma` licence over an `apache-2.0` base, which is `CLAIM002` low because the declared licence
  is the stricter one.

Parameter counts read from the headers: 30.5 B, 32.8 B, 4.0 B, 4.6 B, 230 M, 9.0 B and 7.5 B.
Layer count and hidden size matched the base model's `config.json` in all seven.

This test found a second bug. The first remote scan took 6 minutes of CPU on a 7 KB template
because a regular expression backtracked. The pattern was rewritten to run in linear time, and the
same scan now takes under 10 seconds. A test feeds the analyser a hostile 200 KB template with a
time limit.

## 3. Remote scans: safetensors, pickle and custom code

| Repo | Time | Status | Findings |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 1.1 s | consistent | `CLAIM002` low (licence, as above) |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 1.1 s | consistent | none |
| `openai-community/gpt2` | 17.9 s | review | `EXEC001` medium: `pytorch_model.bin` is a pickle and was not inspected remotely |
| `microsoft/Phi-3-mini-4k-instruct` | 5.4 s | review | `EXEC003` medium: `auto_map` points to `configuration_phi3.Phi3Config`, `modeling_phi3.Phi3ForCausalLM`; `EXEC004` low: 3 Python files |
| `BAAI/bge-m3` | 8.0 s | review | `EXEC001` medium on 3 pickle files; no safetensors weights in the repo |
| `jinaai/jina-embeddings-v3` | 9.8 s | violation | `EXEC003` high: `auto_map` points into `jinaai/xlm-roberta-flash-implementation`; `EXEC001` medium; `EXEC004` low |
| `meta-llama/Llama-3.2-3B-Instruct` | 2.4 s | unverifiable | gated: no file could be read without a token |

Notes:

- **jina-embeddings-v3.** The high finding is accurate and is how the model is designed: its
  config loads model code from a second repository owned by the same publisher. The rule fires
  because enabling `trust_remote_code` for this model runs code that is not pinned by this repo's
  revision.
- **Phi-3-mini.** Same mechanism, but the code ships inside the repo, so it is medium.
- **gpt2.** The repo also holds 3 TFLite and several ONNX files. They are listed as "not
  inspected", which is why nothing here is called clean.
- **Llama 3.2.** With no token the tool read only the model card. It reports each unreadable file
  and the status `unverifiable`. It does not report `consistent`.
- **Pickles are never probed remotely.** Probing one means downloading it. Scan the local copy.

## 4. Metadata study: top 200 GGUF repos

Command: `uv run python study/gguf_study.py --limit 200 --sleep 0.5` (3 min 22 s, 0 errors)

This uses Hub metadata only: the chat template the Hub parsed from each repo, and the declared
base model's template and licence.

| Measure | Count |
|---|---|
| Repos measured | 200 |
| Declare a base model | 178 |
| Declare a licence | 186 |
| Have a chat template | 146 |
| Template comparable with the base model's | 128 |
| Identical to the base model's | 67 |
| Different from the base model's | 61 (48%) |
| Formatting-only difference | 0 |

Of the 18 templates that could not be compared, 11 declare no base model and the rest have a base
that publishes no template.

The 61 differences by publisher (repos differing / repos comparable):

| Publisher | Differ | Comparable |
|---|---|---|
| unsloth | 31 | 32 |
| Qwen | 5 | 6 |
| peculiar-ragdoll | 4 | 4 |
| HauhauCS | 3 | 6 |
| LiquidAI | 3 | 4 |
| DavidAU | 1 | 4 |
| bartowski | 0 | 8 |
| empero-ai | 0 | 7 |
| google | 0 | 4 |
| prism-ml | 0 | 4 |

- 54 of the 61 are rewrites of more than four segments. 7 are small edits.
- 60 of the 61 are tagged as quantizations of their base model.

Other findings:

| Rule | Repos | Meaning |
|---|---|---|
| `CLAIM004` low | 14 | no licence declared |
| `CLAIM002` low | 6 | licence differs from the base model's, neither side restrictive |
| `CLAIM002` medium | 4 | a permissive licence declared over a base whose licence is `other` |
| `TMPL001`, `TMPL002`, `TMPL003` | 0 | no template escape, trigger phrase or hard-coded URL |

This run found a third bug. An earlier 60-repo pass flagged one repo for branching on literals
such as `'\n</think>'`. Those are reasoning-tag markers with an escaped newline, not trigger
phrases. Literals are now unescaped before the check, and the top-200 run has no such findings.
The same pass rated two licence findings high when the base licence was only `other`; they are
now medium, and high is reserved for a base licence that is known to be restrictive.

Limits of this study:

- The template is the one the Hub parsed, from one GGUF file per repo.
- The base model's template is whatever its `main` holds today. If the base changed after the
  quantization was made, the quantizer changed nothing.
- 200 repos by download count is a sample of the head, not the long tail.

## 5. Attack fixtures

No real model in these tests was malicious, so the detection rules are exercised by fixtures in
the test suite. Each one is a small file built by the test, never a real payload.

| Fixture | Expected | Result |
|---|---|---|
| GGUF whose template branches on `'wire the funds'` | `TMPL002` | pass |
| Same trigger added to a template that otherwise matches the base | `TMPL002` raised to high | pass |
| Trigger that the base model's template also contains | lowered to low | pass |
| Ollama Go template with `contains .Content "wire the funds"` | `TMPL002` | pass |
| Template using `__class__.__mro__` or `lipsum.__globals__` | `TMPL001` high | pass |
| Template with a hard-coded URL | `TMPL003` | pass |
| Pickle calling `os.system`, in every pickle protocol 0 to 5 | `EXEC002` high | pass |
| Benign pickle followed by a second malicious pickle | `EXEC002` high | pass |
| Truncated pickle stream | error, not a pass | pass |
| Torch-style zip with a corrupted CRC | `EXEC002` high | pass |
| Pickle renamed to `model.safetensors` | `FMT001` and `EXEC002` | pass |
| safetensors with extra bytes after the last tensor | `FMT002` high | pass |
| safetensors with duplicate JSON keys and overlapping tensors | `FMT002` high | pass |
| GGUF with absurd counts, a duplicate key, or a truncated header | parse refused | pass |
| Config with `auto_map` into another repo | `EXEC003` high | pass |
| Config containing `_attn_implementation_internal` | `EXEC005` high | pass |
| Blob edited after download, with `--rehash` | `INTEG001` high | pass |
| Declared base has 12 layers, weights have 2 | `CLAIM003` high | pass |
| `apache-2.0` declared over an `llama3.2` base | `CLAIM002` high | pass |
| File swapped after `lock` | `check` exits 1 | pass |
| New revision after `lock` | `check` exits 1 | pass |

## Not tested

- **A real malicious model.** Every detection result above comes from a fixture.
- **A real pickle on disk.** No pickle-format model was present locally.
- **A live Ollama model.** The only local entry was a stale manifest. Ollama parsing is covered by
  fixtures and by a registry lookup that matched the local manifest's digests.
- **LM Studio.** Not installed on the test machine.
- **Gated models with a token.**
- **Windows and Linux.** CI runs the test suite on Linux and macOS, but the field tests ran on
  macOS only.
- **False-positive rate at scale.** 200 repos produced no template-rule findings after the fixes.
  The long tail is unmeasured.
