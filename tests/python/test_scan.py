import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cuda, qd.vulkan], exclude=[(qd.vulkan, "Darwin")])
def test_scan():
    def test_scan_for_dtype(dtype, N):
        arr = qd.field(dtype, N)
        arr_aux = qd.field(dtype, N)

        @qd.kernel
        def fill():
            for i in arr:
                arr[i] = qd.random() * N
                arr_aux[i] = arr[i]

        fill()

        # Performing an inclusive in-place's parallel prefix sum,
        # only one exectutor is needed for a specified sorting length.
        executor = qd.algorithms.PrefixSumExecutor(N)

        executor.run(arr)

        cur_sum = 0
        for i in range(N):
            cur_sum += arr_aux[i]
            assert arr[i] == cur_sum

    test_scan_for_dtype(qd.i32, 512)
    test_scan_for_dtype(qd.i32, 1024)
    test_scan_for_dtype(qd.i32, 4096)


@pytest.mark.parametrize("dtype", [qd.i32])
@pytest.mark.parametrize("N", [512, 1024, 4096])
@pytest.mark.parametrize("offset", [0, -1, 1, 256, -256, -23333, 23333])
@test_utils.test(arch=[qd.cuda, qd.vulkan], exclude=[(qd.vulkan, "Darwin")])
def test_scan_with_offset(dtype, N, offset):
    arr = qd.field(dtype, N, offset=offset)
    arr_aux = qd.field(dtype, N, offset=offset)

    @qd.kernel
    def fill():
        for i in arr:
            arr[i] = qd.random() * N
            arr_aux[i] = arr[i]

    fill()

    # Performing an inclusive in-place's parallel prefix sum,
    # only one exectutor is needed for a specified sorting length.
    executor = qd.algorithms.PrefixSumExecutor(N)

    executor.run(arr)

    cur_sum = 0
    for i in range(N):
        cur_sum += arr_aux[i + offset]
        assert arr[i + offset] == cur_sum
