from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from knowyourai import hub
from knowyourai.model import Component, FileRef

MANIFEST_LIMIT = 16 << 20
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REPO_ID = re.compile(r"^[\w.\-]+(?:/[\w.\-]+)?$")
WEIGHT_EXTS = frozenset(
    ".gguf .safetensors .bin .pt .pth .ckpt .pkl .pickle .joblib .onnx .h5 .keras .npz .npy "
    ".pb .tflite .mlmodel".split()
)
SIDE_FILES = frozenset(
    {
        "config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "chat_template.json",
        "adapter_config.json",
        "readme.md",
    }
)
SKIP_DIRS = frozenset({"node_modules", "__pycache__", "venv", "site-packages"})


class TargetError(Exception):
    pass


def hf_cache_root() -> Path:
    if value := os.environ.get("HF_HUB_CACHE"):
        return Path(value)
    if value := os.environ.get("HF_HOME"):
        return Path(value) / "hub"
    cache = os.environ.get("XDG_CACHE_HOME")
    return (Path(cache) if cache else Path.home() / ".cache") / "huggingface" / "hub"


def ollama_root() -> Path:
    if value := os.environ.get("OLLAMA_MODELS"):
        return Path(value)
    return Path.home() / ".ollama" / "models"


def lmstudio_roots() -> list[Path]:
    home = Path.home()
    return [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"]


def scan_hf_cache(root: Path) -> list[Component]:
    out: list[Component] = []
    for repo_dir in sorted(root.glob("models--*")):
        repo = repo_dir.name.removeprefix("models--").replace("--", "/")
        refs: dict[str, list[str]] = {}
        refs_dir = repo_dir / "refs"
        if refs_dir.is_dir():
            for ref in refs_dir.rglob("*"):
                if ref.is_file():
                    name = ref.relative_to(refs_dir).as_posix()
                    refs.setdefault(ref.read_text().strip(), []).append(name)
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in sorted(snapshots.iterdir()):
            component = Component(
                id=f"hf:{repo}",
                store="hf-cache",
                location=str(snap),
                revision=snap.name,
                refs=sorted(refs.get(snap.name, [])),
            )
            for path in sorted(snap.rglob("*")):
                if path.is_dir():
                    continue
                rel = path.relative_to(snap).as_posix()
                try:
                    target = path.resolve(strict=True)
                    size = target.stat().st_size
                except OSError:
                    component.unverified.append(f"{rel}: blob is missing from the cache")
                    continue
                # The sha256 is the name of the blob the snapshot links to. That blob can itself
                # be a link into a shared store whose file names are a different hash.
                blob = path.readlink().name if path.is_symlink() else ""
                digest = blob if _SHA256.match(blob) else None
                component.files.append(
                    FileRef(rel, size, digest, "store" if digest else None, local_path=target)
                )
            if component.files or component.unverified:
                out.append(component)
    return out


def ollama_layers(manifest: dict[str, Any]) -> list[tuple[str, str, int]]:
    layers: list[tuple[str, str, int]] = []
    seen: dict[str, int] = {}
    for layer in manifest.get("layers") or []:
        kind = str(layer.get("mediaType", "")).rsplit(".", 1)[-1] or "layer"
        index = seen.get(kind, 0)
        seen[kind] = index + 1
        label = kind if index == 0 else f"{kind}.{index}"
        digest = str(layer.get("digest", "")).removeprefix("sha256:")
        layers.append((label, digest, int(layer.get("size") or 0)))
    return layers


def scan_ollama(root: Path) -> list[Component]:
    out: list[Component] = []
    manifests = root / "manifests"
    if not manifests.is_dir():
        return out
    for path in sorted(manifests.rglob("*")):
        parts = path.relative_to(manifests).parts
        if not path.is_file() or len(parts) != 4 or path.name.startswith("."):
            continue
        host, namespace, model, tag = parts
        raw = path.read_bytes()[:MANIFEST_LIMIT]
        try:
            manifest = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(manifest, dict):
            continue
        if host != hub.OLLAMA_HOST:
            name = f"{host}/{namespace}/{model}"
        elif namespace == "library":
            name = model
        else:
            name = f"{namespace}/{model}"
        component = Component(
            id=f"ollama:{name}:{tag}",
            store="ollama",
            location=str(path),
            revision=hashlib.sha256(raw).hexdigest(),
        )
        component.evidence["ollama"] = {
            "host": host,
            "namespace": namespace,
            "model": model,
            "tag": tag,
        }
        layers = ollama_layers(manifest)
        component.evidence["ollama_layers"] = [list(layer) for layer in layers]
        missing: list[str] = []
        for label, digest, _size in layers:
            blob = root / "blobs" / f"sha256-{digest}"
            if not _SHA256.match(digest) or not blob.is_file():
                missing.append(label)
                continue
            component.files.append(
                FileRef(label, blob.stat().st_size, digest, "store", local_path=blob)
            )
        if missing and not component.files:
            component.unverified.append(
                f"stale manifest: none of its {len(missing)} layers are on disk"
            )
        else:
            component.unverified += [
                f"{label}: blob is missing from the store" for label in missing
            ]
        out.append(component)
    return out


def _display(path: Path, base: Path | None) -> str:
    resolved = path.resolve()
    anchor = (base or Path.cwd()).resolve()
    try:
        rel = resolved.relative_to(anchor).as_posix()
    except ValueError:
        return resolved.as_posix()
    return rel if rel != "." else resolved.name


def scan_path(path: Path, *, prefix: str = "dir", base: Path | None = None) -> list[Component]:
    store = "dir" if prefix == "dir" else prefix
    if path.is_file():
        ref = FileRef(path.name, path.stat().st_size, local_path=path)
        return [Component(f"file:{_display(path, base)}", "file", str(path), files=[ref])]
    out: list[Component] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        weights = {n for n in filenames if Path(n).suffix.lower() in WEIGHT_EXTS}
        if not weights:
            continue
        keep = weights | {n for n in filenames if n.lower() in SIDE_FILES}
        if "config.json" in filenames:
            keep |= {n for n in filenames if n.endswith(".py")}
        here = Path(dirpath)
        component = Component(f"{prefix}:{_display(here, base)}", store, str(here))
        for name in sorted(keep):
            file = here / name
            if file.is_file():
                component.files.append(FileRef(name, file.stat().st_size, local_path=file))
        out.append(component)
    return out


def local_stores() -> list[Component]:
    out = scan_hf_cache(hf_cache_root()) + scan_ollama(ollama_root())
    for root in lmstudio_roots():
        if root.is_dir():
            out += scan_path(root, prefix="lmstudio", base=root)
    return out


def remote_hf(repo: str, revision: str | None, file_pattern: str | None) -> Component:
    if not REPO_ID.match(repo):
        msg = f"{repo!r} is not a valid Hugging Face repo id"
        raise TargetError(msg)
    try:
        info = hub.hf_model_info(repo, revision)
    except hub.HubError as e:
        msg = f"hf:{repo}: {e}"
        raise TargetError(msg) from e
    sha = str(info.get("sha") or revision or "main")
    component = Component(
        id=f"hf:{repo}",
        store="hf-hub",
        location=f"https://{hub.HF_HOST}/{repo}/tree/{sha}",
        revision=sha,
    )
    if file_pattern:
        component.evidence["file_pattern"] = file_pattern
    for sibling in info.get("siblings") or []:
        path = str(sibling.get("rfilename", ""))
        size = int(sibling.get("size") or 0)
        digest = (sibling.get("lfs") or {}).get("sha256")
        opener = functools.partial(hub.open_remote, hub.hf_url(repo, sha, path), size)
        component.files.append(
            FileRef(path, size, digest, "hub" if digest else None, opener=opener)
        )
    return component


def discover(
    targets: list[str], *, online: bool = False, file_pattern: str | None = None
) -> list[Component]:
    if not targets:
        return local_stores()
    out: list[Component] = []
    stores: list[Component] | None = None
    for target in targets:
        if target.startswith(("hf:", "ollama:", "lmstudio:")):
            if stores is None:
                stores = local_stores()
            ident, _, revision = target.partition("@")
            matches = [
                c
                for c in stores
                if fnmatch.fnmatchcase(c.id, ident)
                and (not revision or (c.revision or "").startswith(revision))
            ]
            if matches:
                out += matches
            elif target.startswith("hf:") and online:
                out.append(remote_hf(ident.removeprefix("hf:"), revision or None, file_pattern))
            elif target.startswith("hf:"):
                msg = (
                    f"{target} is not in the local cache. "
                    "Add --online to inspect it on the Hub without downloading it."
                )
                raise TargetError(msg)
            else:
                msg = f"no local component matches {target}"
                raise TargetError(msg)
        else:
            path = Path(target)
            if not path.exists():
                msg = f"{target}: no such file or directory"
                raise TargetError(msg)
            out += scan_path(path)
    unique: dict[tuple[str, str | None], Component] = {}
    for component in out:
        unique.setdefault((component.id, component.revision), component)
    return list(unique.values())
