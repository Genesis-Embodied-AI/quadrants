import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_indices():
    a = qd.field(qd.f32, shape=(128, 32, 8))

    b = qd.field(qd.f32)
    qd.root.dense(qd.j, 32).dense(qd.i, 16).place(b)

    mapping_a = a.snode._physical_index_position()

    assert mapping_a == {0: 0, 1: 1, 2: 2}

    mapping_b = b.snode._physical_index_position()

    assert mapping_b == {0: 0, 1: 1}
    # Note that b is column-major:
    # the virtual first index exposed to the user comes second in memory layout.

    @qd.kernel
    def fill():
        for i, j in b:
            b[i, j] = i * 10 + j

    @qd.kernel
    def get_field_addr(i: qd.i32, j: qd.i32) -> qd.u64:
        return qd.get_addr(b, [i, j])

    fill()
    for i in range(16):
        for j in range(32):
            assert b[i, j] == i * 10 + j
    assert get_field_addr(0, 1) + 4 == get_field_addr(1, 1)


@test_utils.test(arch=get_host_arch_list(), default_ip=qd.i64)
def test_indices_i64():
    n = 1024
    val = qd.field(dtype=qd.i64, shape=n)
    val.fill(1)

    @qd.kernel
    def prefix_sum():
        qd.loop_config(serialize=True)
        for i in range(1, 1024):
            val[i] += val[i - 1]

    prefix_sum()
    for i in range(n):
        assert val[i] == i + 1


@test_utils.test()
def test_indices_with_matrix():
    grid_m = qd.field(dtype=qd.i32, shape=(10, 10))

    @qd.kernel
    def build_grid():
        base = int(qd.Vector([2, 4]))
        grid_m[base] = 100

        grid_m[int(qd.Vector([1, 1]))] = 10

    build_grid()

    assert grid_m[1, 1] == 10
    assert grid_m[2, 4] == 100


@test_utils.test()
def test_negative_valued_indices():
    @qd.kernel
    def foo(i: int):
        x = qd.Vector([i, i + 1, i + 2])
        print(x[:-1])

    with pytest.raises(
        qd.QuadrantsSyntaxError,
        match="Negative indices are not supported in Quadrants kernels.",
    ):
        foo(0)
