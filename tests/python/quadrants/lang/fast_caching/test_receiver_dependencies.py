"""Fastcache must include compile-time members read through implicit method receivers."""

import os
import subprocess
import sys

import pytest

import quadrants as qd

from tests.helpers import test_utils
from tests.helpers.runtime import qd_init_same_arch


def _receiver_example():
    @qd.data_oriented
    class Item:
        def __init__(self, k, enabled, unused):
            self.k = k
            self.enabled = enabled
            self.unused = unused

        @qd.func
        def read(receiver):
            value = 0
            if qd.static(receiver.enabled):
                value = receiver.k
            return value

        @qd.func
        def apply(self):
            return self.read()

        @qd.func
        def __call__(self):
            return self.apply()

    @qd.data_oriented
    class Holder:
        def __init__(self, k, enabled=True, unused=0):
            self.item = Item(k, enabled, unused)

    @qd.kernel(fastcache=True)
    def method(out: qd.types.NDArray[qd.i32, 1], holder: qd.Template):
        out[0] = holder.item.apply()

    @qd.kernel(fastcache=True)
    def callable_attribute(out: qd.types.NDArray[qd.i32, 1], holder: qd.Template):
        out[0] = holder.item()

    @qd.kernel(fastcache=True)
    def callable_name(out: qd.types.NDArray[qd.i32, 1], item: qd.Template):
        out[0] = item()

    return (method, callable_attribute, callable_name), Holder


@pytest.mark.parametrize("call_kind", range(3))
@test_utils.test(arch=qd.cpu)
def test_fastcache_receiver_constants(call_kind, monkeypatch):
    monkeypatch.setenv("QD_OFFLINE_CACHE", "0")
    qd_init_same_arch(offline_cache=False)
    kernels, holder_type = _receiver_example()
    kernel = kernels[call_kind]
    out = qd.ndarray(qd.i32, shape=1)
    checksums = []
    for k, enabled, unused in [(2, True, 0), (2, True, 99), (3, True, 99), (3, False, 99), (2, True, 0)]:
        holder = holder_type(k, enabled, unused)
        kernel(out, holder.item if call_kind == 2 else holder)
        assert out[0] == (k if enabled else 0)
        assert kernel._primal.fast_checksum is not None
        checksums.append(kernel._primal.fast_checksum)
    assert checksums[0] == checksums[1] == checksums[4]
    assert len(set(checksums[:4])) == 3


def _run_receiver_cache_child(cache_path, call_kind, k, enabled, unused, expect_loaded):
    qd.init(arch=qd.cpu, offline_cache=True, offline_cache_file_path=cache_path)
    kernels, holder_type = _receiver_example()
    call_kind, k, enabled = int(call_kind), int(k), bool(int(enabled))
    holder = holder_type(k, enabled, int(unused))
    kernel = kernels[call_kind]
    out = qd.ndarray(qd.i32, shape=1)
    kernel(out, holder.item if call_kind == 2 else holder)
    assert out[0] == (k if enabled else 0)
    assert kernel._primal.src_ll_cache_observations.cache_loaded == bool(int(expect_loaded))
    assert kernel._primal.fast_checksum is not None
    print("receiver cache child passed")


@pytest.mark.parametrize("call_kind", range(3))
@test_utils.test(arch=qd.cpu)
def test_fastcache_receiver_across_processes(tmp_path, call_kind):
    for k, enabled, unused, loaded in [(2, 1, 0, 0), (2, 1, 99, 1), (3, 1, 99, 0), (3, 0, 99, 0), (2, 1, 0, 1)]:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[1])['_run_receiver_cache_child'](*sys.argv[2:])",
                __file__,
                str(tmp_path),
                *map(str, (call_kind, k, enabled, unused, loaded)),
            ],
            env={**os.environ, "QD_OFFLINE_CACHE": "1", "QD_OFFLINE_CACHE_FILE_PATH": str(tmp_path)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "receiver cache child passed" in result.stdout


@test_utils.test(arch=qd.cpu)
def test_fastcache_receiver_field_fallback(monkeypatch):
    monkeypatch.setenv("QD_OFFLINE_CACHE", "0")
    qd_init_same_arch(offline_cache=False)
    @qd.data_oriented
    class Item:
        def __init__(self, n, value):
            self.x = qd.field(qd.i32, shape=n)
            self.x.fill(value)

        @qd.func
        def read(self, i):
            return self.x[i]

    @qd.kernel(fastcache=True)
    def run(out: qd.types.NDArray[qd.i32, 1], item: qd.Template):
        for i in range(out.shape[0]):
            out[i] = item.read(i)

    out = qd.ndarray(qd.i32, shape=4)
    for n, value in [(4, 1), (8, 2)]:
        run(out, Item(n, value))
        assert (out.to_numpy() == value).all()
        assert run._primal.fast_checksum is None


@pytest.mark.parametrize("alias_kind", ["object", "method", "subscript"])
@test_utils.test(arch=qd.cpu)
def test_fastcache_receiver_alias_fallback(alias_kind, monkeypatch):
    monkeypatch.setenv("QD_OFFLINE_CACHE", "0")
    qd_init_same_arch(offline_cache=False)
    @qd.data_oriented
    class Item:
        def __init__(self, k):
            self.k = k

        @qd.func
        def read(self):
            return self.k

    @qd.kernel(fastcache=True)
    def run(out: qd.types.NDArray[qd.i32, 1], item: qd.Template, kind: qd.Template):
        if qd.static(kind == "object"):
            alias = item
            out[0] = alias.read()
        elif qd.static(kind == "method"):
            method = qd.static(item.read)
            out[0] = method()
        else:
            out[0] = (item,)[0].read()

    out = qd.ndarray(qd.i32, shape=1)
    for k in (2, 3):
        run(out, Item(k), alias_kind)
        assert out[0] == k
        assert run._primal.fast_checksum is None
