import dataclasses
from typing import NamedTuple

import numpy as np
import pytest
import torch

import quadrants as qd
from quadrants.lang._fast_caching import FIELD_METADATA_CACHE_VALUE, args_hasher
from quadrants.lang.kernel_arguments import ArgMetadata

from tests import test_utils


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
            assert hash is None


@test_utils.test()
def test_args_hasher_field_vector() -> None:
    seen = set()
    for dtype in [qd.i32, qd.i64, qd.f32, qd.f64]:
        for n in [2, 3]:
            for shape in [(2,), (5,), (2, 5)]:
                _qd_init_same_arch()
                arg = qd.Vector.field(n, dtype, shape)
                hash = args_hasher.hash_args(False, [arg], [None])
                assert hash is None


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
                    assert hash is None


@test_utils.test()
def test_args_hasher_field_vs_ndarray() -> None:
    a_ndarray = qd.ndarray(qd.i32, 1)
    a_field = qd.field(qd.i32, 1)
    ndarray_hash = args_hasher.hash_args(False, [a_ndarray], [None])
    field_hash = args_hasher.hash_args(False, [a_field], [None])
    assert ndarray_hash is not None
    assert field_hash is None
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
