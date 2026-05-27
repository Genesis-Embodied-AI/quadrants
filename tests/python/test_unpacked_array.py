# pyright: reportInvalidTypeForm=false
"""Tests for ``qd.unpacked_array(N, dtype)`` on ``@qd.dataclass``.

``unpacked_array`` gives users an ergonomic indexed-write syntax on a per-thread struct, while keeping the underlying
storage as N separate named scalar fields so SROA + ``mem2reg`` can register-promote each slot independently. The
static-index case must lower to a direct field reference; PTX must be byte-identical to the named-field equivalent.
"""

import numpy as np
import pytest

import quadrants as qd

try:
    from tests import test_utils  # noqa: F401
except ImportError:  # standalone-run convenience
    test_utils = None  # type: ignore


def _qd_init_cuda():
    qd.init(arch=qd.cuda, default_fp=qd.f32, offline_cache=False)


# ---------------------------------------------------------------------------
# Basic API smoke tests (struct construction).
# ---------------------------------------------------------------------------


def test_unpacked_array_construction_python_scope():
    """A dataclass with ``r: qd.unpacked_array(N, dtype)`` should construct as if it had N named scalar fields named
    ``_r0.._r{N-1}``."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked_array(4, qd.f32)

    # The underlying struct type should report N synthetic scalar members plus expose ``r`` as a group name.
    assert hasattr(Tile, "_unpacked_groups")
    groups = Tile._unpacked_groups
    assert "r" in groups
    count, dtype, _ = groups["r"]
    assert count == 4
    assert dtype is qd.f32

    # The underlying scalar fields must exist.
    assert "_r0" in Tile.members
    assert "_r3" in Tile.members
    assert "r" not in Tile.members  # ``r`` is a logical group, not a real member


# ---------------------------------------------------------------------------
# Static-index reads / writes (the hot path).
# ---------------------------------------------------------------------------


def test_unpacked_array_static_index_write_then_read():
    """Write to ``t.r[0..3]`` with python-int indices, then read back."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked_array(4, qd.f32)

    out = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for _ in range(1):
            t = Tile()
            t.r[0] = qd.f32(1.0)
            t.r[1] = qd.f32(2.0)
            t.r[2] = qd.f32(3.0)
            t.r[3] = qd.f32(4.0)
            o[0] = t.r[0]
            o[1] = t.r[1]
            o[2] = t.r[2]
            o[3] = t.r[3]

    k(out)
    np.testing.assert_array_equal(out.to_numpy(), np.array([1, 2, 3, 4], dtype=np.float32))


def test_unpacked_array_qd_static_loop_index():
    """Index via a ``qd.static(range(N))`` loop variable. Each iter sees a python-int index, so the lowering must be the
    same direct-field path as the explicit python-int case."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked_array(4, qd.f32)

    out = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for _ in range(1):
            t = Tile()
            for i in qd.static(range(4)):
                t.r[i] = qd.f32(10.0 + i)
            for i in qd.static(range(4)):
                o[i] = t.r[i]

    k(out)
    np.testing.assert_array_equal(out.to_numpy(), np.array([10, 11, 12, 13], dtype=np.float32))


# ---------------------------------------------------------------------------
# Equivalence with named-field baseline: identical PTX.
# ---------------------------------------------------------------------------


def _build_named_kernel():
    """Same as test_unpacked_array_static_index_write_then_read but with 4 named ``r0..r3`` fields. Used for PTX byte-
    equality comparison against the ``unpacked_array`` form."""

    @qd.dataclass
    class TileNamed:
        r0: qd.f32
        r1: qd.f32
        r2: qd.f32
        r3: qd.f32

    out = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for _ in range(1):
            t = TileNamed()
            t.r0 = qd.f32(1.0)
            t.r1 = qd.f32(2.0)
            t.r2 = qd.f32(3.0)
            t.r3 = qd.f32(4.0)
            o[0] = t.r0
            o[1] = t.r1
            o[2] = t.r2
            o[3] = t.r3

    return k, out


def _build_unpacked_array_kernel():
    @qd.dataclass
    class TileRA:
        r: qd.unpacked_array(4, qd.f32)

    out = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for _ in range(1):
            t = TileRA()
            t.r[0] = qd.f32(1.0)
            t.r[1] = qd.f32(2.0)
            t.r[2] = qd.f32(3.0)
            t.r[3] = qd.f32(4.0)
            o[0] = t.r[0]
            o[1] = t.r[1]
            o[2] = t.r[2]
            o[3] = t.r[3]

    return k, out


def test_unpacked_array_runtime_index_rejected():
    """Indexing ``t.r[k]`` with a runtime ``k`` raises a clear error pointing at the python-int / ``qd.static``
    requirement. Long term the runtime case can lower to an explicit cascade; for now the limitation is surfaced
    early so callers don't get a confusing LLVM/SROA failure downstream."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked_array(4, qd.f32)

    out = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for _ in range(1):
            t = Tile()
            for i in range(4):  # runtime loop, not qd.static
                t.r[i] = qd.f32(i)
            o[0] = t.r[0]

    with pytest.raises(Exception) as e:
        k(out)
    msg = str(e.value)
    assert "unpacked_array" in msg and "python-int" in msg, msg


def test_unpacked_array_oob_static_index():
    """Static-int out-of-bounds index is caught at compile time with a clear message."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked_array(4, qd.f32)

    out = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for _ in range(1):
            t = Tile()
            t.r[7] = qd.f32(1.0)  # 7 >= count=4
            o[0] = t.r[0]

    with pytest.raises(Exception) as e:
        k(out)
    assert "out of bounds" in str(e.value), str(e.value)


if __name__ == "__main__":
    test_unpacked_array_construction_python_scope()
    print("construction test passed")
    test_unpacked_array_static_index_write_then_read()
    print("static-int subscript test passed")
    test_unpacked_array_qd_static_loop_index()
    print("qd.static loop-var subscript test passed")
    test_unpacked_array_runtime_index_rejected()
    print("runtime-index rejection test passed")
    test_unpacked_array_oob_static_index()
    print("static OOB rejection test passed")
