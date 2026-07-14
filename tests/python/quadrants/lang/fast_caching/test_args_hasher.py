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


@test_utils.test()
def test_args_hasher_frozen_dataclass_failure_does_not_poison_narrower_walk() -> None:
    """Codex r3582132723: a frozen dataclass's cached fastcache failure must not be reused across walks with
    different pruning sets.

    Two @qd.kernels share the same frozen dataclass instance. Kernel A reads the unsupported ``qd.field`` member
    and correctly fails fastcache. Kernel B has a narrower pruning set that excludes the unsupported member and
    reads only the supported ``qd.ndarray``; it must still be fast-cacheable, i.e. the on-instance failure
    sentinel from A's walk must not be reused for B's walk (whose narrower path set doesn't reach the offending
    field).
    """
    args_hasher.reset_unknown_type_warn_state()

    @dataclasses.dataclass(frozen=True)
    class State:
        x: qd.types.NDArray[qd.i32, 1]  # actually holds a qd.field below (unsupported at read path)
        y: qd.types.NDArray[qd.i32, 1]  # cacheable ndarray

    state = State(x=qd.field(qd.i32, shape=(4,)), y=qd.ndarray(qd.i32, (4,)))
    arg_meta = ArgMetadata(name="state", annotation=State)

    # Kernel A: pruning set includes both fields; walking hits the qd.field and fails.
    paths_a = {"__qd_state__qd_x", "__qd_state__qd_y"}
    h_a = args_hasher.hash_args(False, [state], [arg_meta], pruning_paths=paths_a)
    assert isinstance(h_a, FastcacheSkip), f"kernel A should fail fastcache; got {h_a!r}"

    # Kernel B: narrower pruning set excludes the failing field. Same instance - must NOT inherit A's failure.
    paths_b = {"__qd_state__qd_y"}
    h_b = args_hasher.hash_args(False, [state], [arg_meta], pruning_paths=paths_b)
    assert not isinstance(h_b, FastcacheSkip), (
        f"kernel B should be fast-cacheable (its pruning set skips the unsupported field); got {h_b!r}"
    )
    assert isinstance(h_b, str) and len(h_b) > 0

    # And unpruned walks are unaffected: no ``pruning_paths`` -> walk everything -> fail (same as A).
    h_none = args_hasher.hash_args(False, [state], [arg_meta])
    assert isinstance(h_none, FastcacheSkip), f"unpruned walk should fail fastcache; got {h_none!r}"
