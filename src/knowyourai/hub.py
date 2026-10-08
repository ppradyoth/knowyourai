from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Any, BinaryIO

from knowyourai import __version__

HF_HOST = "huggingface.co"
OLLAMA_HOST = "registry.ollama.ai"
TIMEOUT = 30
JSON_LIMIT = 64 << 20
TEXT_LIMIT = 16 << 20
BUFFER = 2 << 20
_MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
_REASONS = {
    401: "requires authentication (gated or private)",
    403: "access denied (gated or private)",
    404: "not found",
    429: "rate limited",
}


class HubError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN202
        if urllib.parse.urlsplit(newurl).scheme != "https":
            reason = f"refusing redirect to non-https URL {newurl}"
            raise HubError(reason)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsOnly)


def _open(url: str, headers: dict[str, str] | None = None) -> Any:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        reason = f"refusing non-https URL {url}"
        raise HubError(reason)
    req = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": f"knowyourai/{__version__}", **(headers or {})}
    )
    token = os.environ.get("HF_TOKEN")
    if token and parts.hostname == HF_HOST:
        # Unredirected, so the token is never forwarded to the CDN a download redirects to.
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        return _OPENER.open(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        reason = f"{parts.hostname}: {_REASONS.get(e.code, f'HTTP {e.code}')}"
        raise HubError(reason, e.code) from e
    except (urllib.error.URLError, OSError) as e:
        reason = f"{parts.hostname}: {e}"
        raise HubError(reason) from e


def get_bytes(url: str, limit: int, headers: dict[str, str] | None = None) -> bytes:
    with _open(url, headers) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        reason = f"response from {url} is larger than {limit} bytes"
        raise HubError(reason)
    return data


def get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    try:
        return json.loads(get_bytes(url, JSON_LIMIT, headers))
    except ValueError as e:
        reason = f"invalid JSON from {url}"
        raise HubError(reason) from e


@lru_cache(maxsize=1024)
def hf_model_info(repo: str, revision: str | None = None) -> dict[str, Any]:
    path = f"/api/models/{urllib.parse.quote(repo, safe='/')}"
    if revision:
        path += f"/revision/{urllib.parse.quote(revision, safe='')}"
    info = get_json(f"https://{HF_HOST}{path}?blobs=true")
    if not isinstance(info, dict):
        reason = f"unexpected response for {repo}"
        raise HubError(reason)
    return info


def hf_url(repo: str, revision: str, path: str) -> str:
    quote = urllib.parse.quote
    return (
        f"https://{HF_HOST}/{quote(repo, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(path, safe='/')}"
    )


def hf_text(repo: str, revision: str, path: str, limit: int = TEXT_LIMIT) -> str:
    return get_bytes(hf_url(repo, revision, path), limit).decode("utf-8", "replace")


def ollama_manifest(namespace: str, model: str, tag: str) -> dict[str, Any]:
    quote = urllib.parse.quote
    url = (
        f"https://{OLLAMA_HOST}/v2/{quote(namespace, safe='')}/"
        f"{quote(model, safe='')}/manifests/{quote(tag, safe='')}"
    )
    manifest = get_json(url, {"Accept": _MANIFEST_TYPE})
    if not isinstance(manifest, dict):
        reason = f"unexpected manifest for {namespace}/{model}:{tag}"
        raise HubError(reason)
    return manifest


class _RangeRaw(io.RawIOBase):
    def __init__(self, url: str, size: int) -> None:
        super().__init__()
        self.url = url
        self.size = size
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(base + offset, 0)
        return self.pos

    def readinto(self, buffer: Any) -> int:
        want = min(len(buffer), self.size - self.pos)
        if want <= 0:
            return 0
        last = self.pos + want - 1
        with _open(self.url, {"Range": f"bytes={self.pos}-{last}"}) as resp:
            if resp.status != 206 and not (resp.status == 200 and self.pos == 0):
                reason = f"server ignored the range request for {self.url}"
                raise HubError(reason)
            data = resp.read(want)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote(url: str, size: int) -> BinaryIO:
    return io.BufferedReader(_RangeRaw(url, size), buffer_size=BUFFER)
