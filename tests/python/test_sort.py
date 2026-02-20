import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_sort():
    def test_sort_for_dtype(dtype, N):
        keys = qd.field(dtype, N)
        values = qd.field(dtype, N)

        @qd.kernel
        def fill():
            for i in keys:
                keys[i] = qd.random() * N
                values[i] = keys[i]

        fill()
        qd.algorithms.parallel_sort(keys, values)

        keys_host = keys.to_numpy()
        values_host = values.to_numpy()

        for i in range(N):
            if i < N - 1:
                assert keys_host[i] <= keys_host[i + 1]
            assert keys_host[i] == values_host[i]

    test_sort_for_dtype(qd.i32, 1)
    test_sort_for_dtype(qd.i32, 256)
    test_sort_for_dtype(qd.i32, 100001)
    test_sort_for_dtype(qd.f32, 1)
    test_sort_for_dtype(qd.f32, 256)
    test_sort_for_dtype(qd.f32, 100001)


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32])
@pytest.mark.parametrize("N", [1, 256, 100001])
@pytest.mark.parametrize("offset", [0, -1, 1, 128, -128, -23333, 23333])
@test_utils.test()
def test_sort_with_offset(dtype, N, offset):
    keys = qd.field(dtype, N, offset=offset)
    values = qd.field(dtype, N, offset=offset)

    @qd.kernel
    def fill():
        for i in keys:
            keys[i] = qd.random() * N
            values[i] = keys[i]

    fill()
    qd.algorithms.parallel_sort(keys, values)

    keys_host = keys.to_numpy()
    values_host = values.to_numpy()

    for i in range(N):
        if i < N - 1:
            assert keys_host[i] <= keys_host[i + 1]
        assert keys_host[i] == values_host[i]
