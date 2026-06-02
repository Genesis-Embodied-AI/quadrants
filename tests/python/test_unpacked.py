# pyright: reportInvalidTypeForm=false
"""Tests for the ``qd.unpacked[qd.types.vector(N, dtype)]`` layout annotation on ``@qd.dataclass``.

The annotation declares a Vector-typed field with *unpacked* storage: N independent scalar slots, one ``alloca`` each,
rather than a single packed ``alloca``. The static-index access ``obj.r[i]`` must lower to a direct reference to the
synthetic scalar field ``_r{i}``; PTX must be byte-identical to the named-field equivalent.
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


def test_unpacked_construction_python_scope():
    """A dataclass with ``r: qd.unpacked[qd.types.vector(N, dtype)]`` should construct as if it had N named scalar
    fields ``_r0.._r{N-1}``."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked[qd.types.vector(4, qd.f32)]

    assert hasattr(Tile, "_unpacked_groups")
    groups = Tile._unpacked_groups
    assert "r" in groups
    count, dtype, _ = groups["r"]
    assert count == 4
    assert dtype is qd.f32

    assert "_r0" in Tile.members
    assert "_r3" in Tile.members
    assert "r" not in Tile.members  # logical group, not a real member


# ---------------------------------------------------------------------------
# Static-index reads / writes (the hot path).
# ---------------------------------------------------------------------------


def test_unpacked_static_index_write_then_read():
    """Write to ``t.r[0..3]`` with python-int indices, then read back."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked[qd.types.vector(4, qd.f32)]

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


def test_unpacked_qd_static_loop_index():
    """Index via a ``qd.static(range(N))`` loop variable. Each iter sees a python-int index, so the lowering must be the
    same direct-field path as the explicit python-int case."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked[qd.types.vector(4, qd.f32)]

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
# Runtime / OOB rejection.
# ---------------------------------------------------------------------------


def test_unpacked_runtime_index_rejected():
    """Indexing ``t.r[k]`` with a runtime ``k`` raises a clear error pointing at the python-int / ``qd.static``
    requirement. Long term the runtime case can lower to an explicit cascade (or fall through to a Vector-value
    materialise + index); for now the limitation is surfaced early so callers don't get a confusing LLVM/SROA failure
    downstream."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked[qd.types.vector(4, qd.f32)]

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
    assert "python-int" in msg, msg


def test_unpacked_oob_static_index():
    """Static-int out-of-bounds index is caught at compile time with a clear message."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked[qd.types.vector(4, qd.f32)]

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


# ---------------------------------------------------------------------------
# Misuse guards (python-side, no GPU needed).
# ---------------------------------------------------------------------------


def test_unpacked_call_rejected():
    """``qd.unpacked()`` (calling the marker class) raises a clear error pointing at the annotation form."""
    with pytest.raises(qd.QuadrantsSyntaxError) as e:
        qd.unpacked()
    assert "qd.unpacked[qd.types.vector" in str(e.value), str(e.value)


def test_unpacked_subscript_with_non_vector_rejected():
    """``qd.unpacked[<not a VectorType>]`` raises with a helpful message naming the required form."""
    with pytest.raises(qd.QuadrantsSyntaxError) as e:
        _ = qd.unpacked[qd.f32]  # bare dtype, not a vector type
    msg = str(e.value)
    assert "vector type" in msg and "qd.types.vector(N, dtype)" in msg, msg


def test_unpacked_subscript_with_tuple_rejected():
    """The old two-arg ``qd.unpacked[dtype, N]`` form is no longer supported; subscribing with a tuple should fail
    cleanly rather than silently produce a malformed marker."""
    with pytest.raises(qd.QuadrantsSyntaxError) as e:
        _ = qd.unpacked[qd.f32, 4]  # python collects this as ``(qd.f32, 4)``
    msg = str(e.value)
    assert "vector type" in msg, msg


# ---------------------------------------------------------------------------
# Composition tests (nested struct, StructField subscript).
# ---------------------------------------------------------------------------


def test_unpacked_nested_in_outer_dataclass():
    """An ``qd.unpacked[...]`` on an *inner* ``@qd.dataclass`` should keep working when that dataclass is itself nested
    inside an outer ``@qd.dataclass``. Regression test for the metadata-stripping path in ``expr_init`` /
    ``StructType.cast``: the ``_qd_unpacked_groups`` tag must propagate through nested-struct rewrap so the AST
    transformer can still recognise ``o.inner.r[i]`` as a group access."""
    _qd_init_cuda()

    @qd.dataclass
    class Inner:
        r: qd.unpacked[qd.types.vector(4, qd.f32)]

    @qd.dataclass
    class Outer:
        inner: Inner
        scale: qd.f32

    out = qd.field(dtype=qd.f32, shape=(1,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for _ in range(1):
            t = Outer()
            for i in qd.static(range(4)):
                t.inner.r[i] = qd.cast(10 + i, qd.f32)
            t.scale = qd.f32(2.0)
            o[0] = t.inner.r[2] * t.scale

    k(out)
    assert out.to_numpy()[0] == 24.0


def test_unpacked_struct_field_subscript():
    """``Tile.field(shape=...)`` should preserve unpacked-vector semantics: ``f[i].r[k]`` must lower to the synthetic
    ``f[i]._r{k}`` field, not fall through to a plain attribute lookup. Regression test for the ``impl.subscript`` /
    ``_IntermediateStruct`` codepath: the ``_qd_unpacked_groups`` tag must propagate from ``StructType.field``'s
    ``StructField`` onto every per-index intermediate struct it produces."""
    _qd_init_cuda()

    @qd.dataclass
    class Tile:
        r: qd.unpacked[qd.types.vector(4, qd.f32)]

    tile_field = Tile.field(shape=(2,))
    out = qd.field(dtype=qd.f32, shape=(1,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for n in range(2):
            for i in qd.static(range(4)):
                tile_field[n].r[i] = qd.cast(n * 10 + i, qd.f32)
        o[0] = tile_field[0].r[1] + tile_field[1].r[3]

    k(out)
    assert out.to_numpy()[0] == 14.0  # 1 + 13


def test_unpacked_struct_field_subscript_nested():
    """Same as above, but with ``Outer.field(shape=...)`` where ``Outer`` contains an ``Inner`` with an unpacked group.
    The nested ``StructField`` for ``inner`` must also carry the group tag, so ``outer_field[i].inner.r[k]`` resolves.
    """
    _qd_init_cuda()

    @qd.dataclass
    class Inner:
        r: qd.unpacked[qd.types.vector(4, qd.f32)]

    @qd.dataclass
    class Outer:
        inner: Inner
        scale: qd.f32

    outer_field = Outer.field(shape=(1,))
    out = qd.field(dtype=qd.f32, shape=(1,))

    @qd.kernel(fastcache=False)
    def k(o: qd.template()):
        for i in qd.static(range(4)):
            outer_field[0].inner.r[i] = qd.cast(i + 5, qd.f32)
        outer_field[0].scale = qd.f32(3.0)
        o[0] = outer_field[0].inner.r[2] * outer_field[0].scale

    k(out)
    assert out.to_numpy()[0] == 21.0  # 7 * 3


# ---------------------------------------------------------------------------
# Synthetic-field-name collision rejection.
# ---------------------------------------------------------------------------


def test_unpacked_collision_with_earlier_field():
    """A user-declared field whose name matches a future synthetic field of an unpacked group should raise rather than
    silently overwriting."""
    with pytest.raises(qd.QuadrantsSyntaxError) as e:

        @qd.dataclass
        class Bad1:
            _r0: qd.f32
            r: qd.unpacked[qd.types.vector(4, qd.f32)]

    msg = str(e.value)
    assert "_r0" in msg and "unpacked" in msg.lower(), msg


def test_unpacked_collision_with_later_field():
    """A user-declared field whose name matches an already-expanded synthetic field of an earlier unpacked group should
    also raise."""
    with pytest.raises(qd.QuadrantsSyntaxError) as e:

        @qd.dataclass
        class Bad2:
            r: qd.unpacked[qd.types.vector(4, qd.f32)]
            _r2: qd.f32

    msg = str(e.value)
    assert "_r2" in msg and "unpacked" in msg.lower(), msg


if __name__ == "__main__":
    test_unpacked_construction_python_scope()
    print("construction test passed")
    test_unpacked_static_index_write_then_read()
    print("static-int subscript test passed")
    test_unpacked_qd_static_loop_index()
    print("qd.static loop-var subscript test passed")
    test_unpacked_runtime_index_rejected()
    print("runtime-index rejection test passed")
    test_unpacked_oob_static_index()
    print("static OOB rejection test passed")
    test_unpacked_call_rejected()
    print("marker call rejection test passed")
    test_unpacked_subscript_with_non_vector_rejected()
    print("subscript-non-vector rejection test passed")
    test_unpacked_subscript_with_tuple_rejected()
    print("subscript-tuple rejection test passed")
    test_unpacked_nested_in_outer_dataclass()
    print("nested-in-outer-dataclass test passed")
    test_unpacked_struct_field_subscript()
    print("struct_field subscript test passed")
    test_unpacked_struct_field_subscript_nested()
    print("struct_field nested subscript test passed")
    test_unpacked_collision_with_earlier_field()
    print("collision-with-earlier-field test passed")
    test_unpacked_collision_with_later_field()
    print("collision-with-later-field test passed")
