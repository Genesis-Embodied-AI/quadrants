import dataclasses
from typing import NamedTuple

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang._fast_caching import FIELD_METADATA_CACHE_VALUE, args_hasher
from quadrants.lang._fast_caching.args_hasher import FastcacheSkip
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import has_pytorch

from tests import test_utils

if has_pytorch():
    import torch


@test_utils.test()
def test_args_hasher_numeric() -> None:
    seen = set()
    for arg in (3, 5.3, np.int32(3), np.int64(5), np.float32(2), np.float64(2)):
        for it in (0, 1):
            hash = args_hasher.hash_args(False, [arg], [None])
            assert hash is not None
            if it == 0:
                assert hash not in seen
                seen.add(hash)
            else:
                assert hash in seen


@pytest.mark.parametrize(
    "annotation,cache_value",
    [
        (None, False),
        (qd.i32, False),
        (qd.template(), True),
        (qd.Template, True),
    ],
)
@test_utils.test()
def test_args_hasher_numeric_maybe_template(annotation: object, cache_value: bool) -> None:
    for arg in (3, 5.3, np.int32(3), np.int64(5), np.float32(2), np.float64(2)):
        orig_type = type(arg)
        arg_meta = ArgMetadata(name="", annotation=annotation)
        hash1 = args_hasher.hash_args(False, [arg], [arg_meta])
        assert hash1 is not None

        arg += 1
        assert type(arg) == orig_type
        arg_meta = ArgMetadata(name="", annotation=annotation)
        hash2 = args_hasher.hash_args(False, [arg], [arg_meta])
        assert hash2 is not None
        if cache_value:
            assert hash1 != hash2
        else:
            assert hash1 == hash2


@test_utils.test()
def test_args_hasher_bool() -> None:
    seen = set()
    for arg in (False, np.bool(False)):
        print("arg", arg, type(arg))
        for it in (0, 1):
            hash = args_hasher.hash_args(False, [arg], [None])
            assert hash is not None
            if it == 0:
                assert hash not in seen
                seen.add(hash)
            else:
                assert hash in seen


@pytest.mark.parametrize(
    "annotation,cache_value",
    [
        (None, False),
        (qd.i32, False),
        (qd.template(), True),
        (qd.Template, True),
    ],
)
@test_utils.test()
def test_args_hasher_bool_maybe_template(annotation: object, cache_value: bool) -> None:
    for arg1, arg2 in [(False, True), (np.bool_(False), np.bool_(True))]:
        arg_meta = ArgMetadata(name="", annotation=annotation)
        hash1 = args_hasher.hash_args(False, [arg1], [arg_meta])
        assert hash1 is not None

        arg_meta = ArgMetadata(name="", annotation=annotation)
        hash2 = args_hasher.hash_args(False, [arg2], [arg_meta])
        assert hash2 is not None
        if cache_value:
            assert hash1 != hash2
        else:
            assert hash1 == hash2


@test_utils.test()
def test_args_hasher_data_oriented() -> None:
    @qd.data_oriented
    class Foo: ...

    foo = Foo()
    assert args_hasher.hash_args(False, [foo], [None]) is not None


@test_utils.test()
def test_args_hasher_data_oriented_template_primitives_value_not_keyed() -> None:
    """A normal @qd.data_oriented bakes primitive members into the kernel, so the fastcache key includes their value.
    @qd.data_oriented(template_primitives=False) lifts them to runtime scalar args instead, so the key must depend on
    the primitive's *type* only — otherwise fastcache would recompile on every value change, defeating the feature."""

    @qd.data_oriented(template_primitives=False)
    class Runtime:
        def __init__(self, k):
            self.k = k

    @qd.data_oriented
    class Baked:
        def __init__(self, k):
            self.k = k

    h = args_hasher.hash_args

    # Lifted (runtime) primitive: value change -> same key.
    h_rt = h(False, [Runtime(3)], [None])
    assert h_rt is not None and not isinstance(h_rt, FastcacheSkip)
    assert h_rt == h(False, [Runtime(7)], [None])

    # Baked primitive (default): value change -> different key.
    h_baked = h(False, [Baked(3)], [None])
    assert h_baked is not None and not isinstance(h_baked, FastcacheSkip)
    assert h_baked != h(False, [Baked(7)], [None])


@test_utils.test()
def test_args_hasher_data_oriented_template_primitives_nested_value_not_keyed() -> None:
    """The type-only keying applies through nested template_primitives=False data_oriented members and alongside
    ndarray members (the engine-F shape: Engine -> subsystem -> bare-primitive loop bound + ndarray buffers)."""

    @qd.data_oriented(template_primitives=False)
    class Inner:
        def __init__(self, n_rt):
            self.n_rt = n_rt
            self.buf = qd.ndarray(qd.f64, shape=(4,))

    @qd.data_oriented(template_primitives=False)
    class Outer:
        def __init__(self, n_rt):
            self.cap = 1000
            self.inner = Inner(n_rt)

    h = args_hasher.hash_args
    h4 = h(False, [Outer(4)], [None])
    assert h4 is not None and not isinstance(h4, FastcacheSkip)
    assert h4 == h(False, [Outer(6)], [None])


@test_utils.test()
def test_args_hasher_ndarray() -> None:
    seen = set()
    for dtype in [qd.i32, qd.i64, qd.f32, qd.f64]:
        for ndim in [0, 1, 2]:
            arg = qd.ndarray(dtype, [1] * ndim)
            for it in [0, 1]:
                hash = args_hasher.hash_args(False, [arg], [None])
                assert hash is not None
                if it == 0:
                    assert hash not in seen
                    seen.add(hash)
                else:
                    assert hash in seen


@test_utils.test()
def test_args_hasher_ndarray_vector() -> None:
    seen = set()
    for dtype in [qd.i32, qd.i64, qd.f32, qd.f64]:
        for vector_size in [2, 3]:
            for ndim in [0, 1, 2]:
                arg = qd.Vector.ndarray(vector_size, dtype, [1] * ndim)
                for it in [0, 1]:
                    hash = args_hasher.hash_args(False, [arg], [None])
                    assert hash is not None
                    if it == 0:
                        assert hash not in seen
                        seen.add(hash)
                    else:
                        assert hash in seen


@test_utils.test()
def test_args_hasher_ndarray_matrix() -> None:
    seen = set()
    for dtype in [qd.i32, qd.i64, qd.f32, qd.f64]:
        for m in [2, 3]:
            for n in [2, 3]:
                for ndim in [0, 1, 2]:
                    arg = qd.Matrix.ndarray(m, n, dtype, [1] * ndim)
                    for it in [0, 1]:
                        hash = args_hasher.hash_args(False, [arg], [None])
                        assert hash is not None
                        if it == 0:
                            assert hash not in seen
                            seen.add(hash)
                        else:
                            assert hash in seen


def _qd_init_same_arch() -> None:
    assert qd.cfg is not None
    qd.init(arch=getattr(qd, qd.cfg.arch.name))


@test_utils.test()
def test_args_hasher_field() -> None:
    """
    Check fields are correctly disabled.

    More context: https://github.com/Genesis-Embodied-AI/quadrants/pull/163
    """
    for dtype in [qd.i32, qd.i64, qd.f32, qd.f64]:
        for shape in [(2,), (5,), (2, 5)]:
            _qd_init_same_arch()
            arg = qd.field(dtype, shape)
            hash = args_hasher.hash_args(False, [arg], [None])
            assert isinstance(hash, FastcacheSkip)


@test_utils.test()
def test_args_hasher_field_vector() -> None:
    seen = set()
    for dtype in [qd.i32, qd.i64, qd.f32, qd.f64]:
        for n in [2, 3]:
            for shape in [(2,), (5,), (2, 5)]:
                _qd_init_same_arch()
                arg = qd.Vector.field(n, dtype, shape)
                hash = args_hasher.hash_args(False, [arg], [None])
                assert isinstance(hash, FastcacheSkip)


@test_utils.test()
def test_args_hasher_field_matrix() -> None:
    seen = set()
    for dtype in [qd.i32, qd.i64, qd.f32, qd.f64]:
        for m in [2, 3]:
            for n in [2, 3]:
                for shape in [(2,), (5,), (2, 5)]:
                    _qd_init_same_arch()
                    arg = qd.Matrix.field(m, n, dtype, shape)
                    hash = args_hasher.hash_args(False, [arg], [None])
                    assert isinstance(hash, FastcacheSkip)


@test_utils.test()
def test_args_hasher_field_vs_ndarray() -> None:
    a_ndarray = qd.ndarray(qd.i32, 1)
    a_field = qd.field(qd.i32, 1)
    ndarray_hash = args_hasher.hash_args(False, [a_ndarray], [None])
    field_hash = args_hasher.hash_args(False, [a_field], [None])
    assert ndarray_hash is not None
    assert isinstance(field_hash, FastcacheSkip)
    assert ndarray_hash != field_hash


@test_utils.test()
def test_cache_values_unchecked() -> None:
    """
    Should we consider two dataclasses with same fields but different name as different?
    Considering them to be the same makes testing easier for now...
    """

    @dataclasses.dataclass
    class MyConfigNoChecked:
        some_int_unchecked: int

    @dataclasses.dataclass
    class MyConfigNoCheckedSame:
        some_int_unchecked: int

    @dataclasses.dataclass
    class MyConfigNoCheckedDiff:
        some_int_new: int

    base = MyConfigNoChecked(some_int_unchecked=3)
    same = MyConfigNoCheckedSame(some_int_unchecked=6)
    diff = MyConfigNoCheckedDiff(some_int_new=3)

    h = args_hasher.hash_args
    h_base = h(False, [base], [None])
    assert h_base is not None
    assert h_base == h(False, [same], [None])
    assert h_base != h(False, [diff], [None])


@test_utils.test()
def test_cache_values_checked() -> None:
    @dataclasses.dataclass
    class MyConfigChecked:
        some_int_checked: int = dataclasses.field(metadata={FIELD_METADATA_CACHE_VALUE: True})

    base = MyConfigChecked(some_int_checked=5)
    same = MyConfigChecked(some_int_checked=5)
    diff = MyConfigChecked(some_int_checked=7)

    h = args_hasher.hash_args
    h_base = h(False, [base], [None])
    assert h_base is not None
    assert h_base == h(False, [same], [None])
    assert h_base != h(False, [diff], [None])


@pytest.mark.needs_torch
@pytest.mark.skipif(not has_pytorch(), reason="PyTorch not installed.")
@test_utils.test()
def test_args_hasher_torch_tensor() -> None:
    seen = set()
    arg = torch.zeros((2, 3), dtype=float)
    for it in range(2):
        hash = args_hasher.hash_args(False, [arg], [None])
        assert hash is not None
        if it == 0:
            assert hash not in seen
            seen.add(hash)
        else:
            assert hash in seen


@pytest.mark.needs_torch
@pytest.mark.skipif(not has_pytorch(), reason="PyTorch not installed.")
@test_utils.test()
def test_args_hasher_custom_torch_tensor() -> None:
    class CustomTensor(torch.Tensor): ...

    seen = set()
    arg = CustomTensor((2, 3))
    for it in range(2):
        hash = args_hasher.hash_args(False, [arg], [None])
        assert hash is not None
        if it == 0:
            assert hash not in seen
            seen.add(hash)
        else:
            assert hash in seen


@test_utils.test()
def test_args_hasher_dataclass_with_field_child_returns_none() -> None:
    """A frozen dataclass whose fields contain ScalarField (non-cacheable) must produce hash=None.

    Before the fix, dataclass_to_repr embedded the literal string "(None)" for non-cacheable child fields instead
    of propagating None upward. This caused all instances of such dataclasses to share the same fastcache key,
    leading to stale compiled kernels being reused across tests with different SNode trees.
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    f = qd.field(qd.i32, shape=(4,))
    state = State(a=f)
    h = args_hasher.hash_args(False, [state], [None])
    assert isinstance(h, FastcacheSkip), f"Dataclass with ScalarField child must disable fastcache, got {h!r}"


@test_utils.test()
def test_args_hasher_dataclass_with_tensor_field_child_returns_none() -> None:
    """A frozen dataclass whose qd.Tensor field wraps a ScalarField must also produce hash=None."""

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    f = qd.field(qd.i32, shape=(4,))
    t = qd.Tensor(f)
    state = State(a=t)
    h = args_hasher.hash_args(False, [state], [None])
    assert isinstance(h, FastcacheSkip), f"Dataclass with Tensor-wrapped ScalarField must disable fastcache, got {h!r}"


@test_utils.test()
def test_args_hasher_dataclass_with_ndarray_child_returns_hash() -> None:
    """A frozen dataclass whose fields are all ndarrays (cacheable) must still produce a valid hash."""

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]
        b: qd.types.NDArray[qd.f32, 1]

    a = qd.ndarray(qd.i32, (4,))
    b = qd.ndarray(qd.f32, (4,))
    state = State(a=a, b=b)
    h = args_hasher.hash_args(False, [state], [None])
    assert h is not None, "Dataclass with ndarray children should be fast-cacheable"


@test_utils.test()
def test_args_hasher_dataclass_field_none_not_collide_with_ndarray() -> None:
    """Two frozen dataclass instances — one with fields, one with ndarrays — must not share a cache key.

    This is the core collision scenario: before the fix, the field-backed instance produced a non-None hash
    containing "(None)" strings, which could collide with other instances.
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    nd = qd.ndarray(qd.i32, (4,))
    f = qd.field(qd.i32, shape=(4,))

    h_nd = args_hasher.hash_args(False, [State(a=nd)], [None])
    h_f = args_hasher.hash_args(False, [State(a=f)], [None])

    assert h_nd is not None
    assert isinstance(h_f, FastcacheSkip)
    assert h_nd != h_f


@test_utils.test()
def test_args_hasher_named_tuple() -> None:
    @qd.data_oriented
    class Geom(NamedTuple):
        pos: qd.Template

    @qd.kernel(fastcache=True)
    def set_pos(geom: qd.Template, value: qd.types.NDArray):
        for I in qd.grouped(qd.ndrange(*geom.pos.shape)):
            for j in qd.static(range(3)):
                geom.pos[I][j] = value[(*I, j)]

    geom = Geom(pos=qd.field(dtype=qd.types.vector(3, qd.f32), shape=(1,)))
    set_pos(geom, np.ones((1, 3), dtype=np.float32))
    assert np.all(geom.pos.to_numpy() == np.ones((1, 3), dtype=np.float32))
