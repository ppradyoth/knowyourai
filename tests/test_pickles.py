from __future__ import annotations

import argparse
import io
import os
import pickle
import zipfile
from collections import OrderedDict

import pytest

from knowyourai import pickles


class Evil:
    def __reduce__(self):
        return (os.system, ("true",))


def test_tensor_only_pickle_is_clean():
    probe = pickles.probe(io.BytesIO(pickle.dumps(OrderedDict(a=1))))
    assert probe.globals == ["collections.OrderedDict"]
    assert probe.dangerous == []
    assert probe.unknown == []
    assert probe.error is None


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_dangerous_import_found_in_every_protocol(protocol):
    probe = pickles.probe(io.BytesIO(pickle.dumps(Evil(), protocol=protocol)))
    assert [g.split(".")[-1] for g in probe.dangerous] == ["system"]


def test_unlisted_import_is_unknown_not_safe():
    probe = pickles.probe(io.BytesIO(pickle.dumps(argparse.Namespace(a=1))))
    assert probe.unknown == ["argparse.Namespace"]
    assert probe.dangerous == []


def test_payload_in_second_pickle_is_found():
    stream = pickle.dumps(OrderedDict(a=1)) + pickle.dumps(Evil())
    assert pickles.probe(io.BytesIO(stream)).dangerous


def test_trailing_bytes_are_reported():
    probe = pickles.probe(io.BytesIO(pickle.dumps(OrderedDict(a=1)) + b"raw tensor data"))
    assert probe.trailing


def test_malformed_stream_is_an_error_not_a_pass():
    probe = pickles.probe(io.BytesIO(pickle.dumps(OrderedDict(a=1))[:-4]))
    assert probe.error


def _torch_zip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", payload)
        z.writestr("archive/data/0", b"\0" * 16)
    return buffer.getvalue()


def test_zip_member_is_probed():
    probes, error = pickles.probe_zip(io.BytesIO(_torch_zip(pickle.dumps(Evil()))))
    assert error is None
    assert probes["archive/data.pkl"].dangerous


def test_zip_with_bad_crc_is_an_error():
    payload = pickle.dumps(OrderedDict(a=1))
    data = bytearray(_torch_zip(payload))
    data[data.index(payload) + 3] ^= 0xFF
    _probes, error = pickles.probe_zip(io.BytesIO(bytes(data)))
    assert error
    assert "BadZipFile" in error
