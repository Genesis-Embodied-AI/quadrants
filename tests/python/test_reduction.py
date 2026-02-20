import numpy as np
import pytest
from pytest import approx

import quadrants as qd

from tests import test_utils

OP_ADD = 0
OP_MIN = 1
OP_MAX = 2
OP_AND = 3
OP_OR = 4
OP_XOR = 5

qd_ops = {
    OP_ADD: qd.atomic_add,
    OP_MIN: qd.atomic_min,
    OP_MAX: qd.atomic_max,
    OP_AND: qd.atomic_and,
    OP_OR: qd.atomic_or,
    OP_XOR: qd.atomic_xor,
}

np_ops = {
    OP_ADD: np.sum,
    OP_MIN: lambda a: a.min(),
    OP_MAX: lambda a: a.max(),
    OP_AND: np.bitwise_and.reduce,
    OP_OR: np.bitwise_or.reduce,
    OP_XOR: np.bitwise_xor.reduce,
}


def _test_reduction_single(dtype, criterion, op):
    N = 1024 * 1024
    if (
        qd.lang.impl.current_cfg().arch == qd.vulkan or qd.lang.impl.current_cfg().arch == qd.metal
    ) and dtype == qd.f32:
        # Vulkan is not capable of such large number in its float32...
        N = 1024 * 16

    a = qd.field(dtype, shape=N)
    tot = qd.field(dtype, shape=())

    if dtype in [qd.f32, qd.f64]:

        @qd.kernel
        def fill():
            for i in a:
                a[i] = i + 0.5

    else:

        @qd.kernel
        def fill():
            for i in a:
                a[i] = i + 1

    ti_op = qd_ops[op]

    @qd.kernel
    def reduce():
        for i in a:
            ti_op(tot[None], a[i])

    @qd.kernel
    def reduce_tmp() -> dtype:
        s = qd.zero(tot[None]) if op == OP_ADD or op == OP_XOR else a[0]
        for i in a:
            ti_op(s, a[i])
        return s

    fill()
    tot[None] = 0 if op in [OP_ADD, OP_XOR] else a[0]
    reduce()
    tot2 = reduce_tmp()

    np_arr = a.to_numpy()
    ground_truth = np_ops[op](np_arr)

    assert criterion(tot[None], ground_truth)
    assert criterion(tot2, ground_truth)


@pytest.mark.parametrize("op", [OP_ADD, OP_MIN, OP_MAX, OP_AND, OP_OR, OP_XOR])
@test_utils.test()
def test_reduction_single_i32(op):
    _test_reduction_single(qd.i32, lambda x, y: int(x) % 2**32 == int(y) % 2**32, op)


@pytest.mark.parametrize("op", [OP_ADD])
@test_utils.test()
def test_reduction_single_u32(op):
    _test_reduction_single(qd.u32, lambda x, y: int(x) % 2**32 == int(y) % 2**32, op)


@pytest.mark.parametrize("op", [OP_ADD, OP_MIN, OP_MAX])
@test_utils.test()
def test_reduction_single_f32(op):
    _test_reduction_single(qd.f32, lambda x, y: x == approx(y, 3e-4), op)


@pytest.mark.parametrize("op", [OP_ADD])
@test_utils.test(require=qd.extension.data64)
def test_reduction_single_i64(op):
    _test_reduction_single(qd.i64, lambda x, y: int(x) % 2**64 == int(y) % 2**64, op)


@pytest.mark.parametrize("op", [OP_ADD])
@test_utils.test(require=qd.extension.data64)
def test_reduction_single_u64(op):
    _test_reduction_single(qd.u64, lambda x, y: int(x) % 2**64 == int(y) % 2**64, op)


@pytest.mark.parametrize("op", [OP_ADD])
@test_utils.test(require=qd.extension.data64)
def test_reduction_single_f64(op):
    _test_reduction_single(qd.f64, lambda x, y: x == approx(y, 1e-12), op)


@test_utils.test()
def test_reduction_different_scale():
    @qd.kernel
    def func(n: qd.template()) -> qd.i32:
        x = 0
        for i in range(n):
            qd.atomic_add(x, 1)
        return x

    # 10 and 60 since OpenGL TLS stride size = 32
    # 1024 and 100000 since OpenGL max threads per group ~= 1792
    for n in [1, 10, 60, 1024, 100000]:
        assert n == func(n)


@test_utils.test()
def test_reduction_non_full_warp():
    @qd.kernel
    def test() -> qd.i32:
        hit_time = 1
        qd.loop_config(block_dim=8)
        for i in range(8):
            qd.atomic_min(hit_time, 1)
        return hit_time

    assert test() == 1


@test_utils.test()
def test_reduction_ndarray():
    @qd.kernel
    def reduce(a: qd.types.ndarray()) -> qd.i32:
        s = 0
        for i in a:
            qd.atomic_add(s, a[i])
            qd.atomic_sub(s, 2)
        return s

    n = 1024
    x = np.ones(n, dtype=np.int32)
    assert reduce(x) == -n
