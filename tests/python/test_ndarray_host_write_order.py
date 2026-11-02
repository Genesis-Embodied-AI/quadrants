import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("operation", ["fill", "kernel"])
@test_utils.test()
def test_ndarray_host_write_follows_device_writes(operation: str) -> None:
    # A host assignment must follow pending device writes, and preserve the other elements they initialized.
    array = qd.ndarray(dtype=qd.i32, shape=4)

    @qd.kernel
    def k_write(array: qd.types.ndarray()) -> None:
        for i in array:
            array[i] = 3

    # Reuse the compiled kernel so a slow first compilation cannot hide the pending-command ordering bug.
    num_trials = 10
    for _ in range(num_trials):
        if operation == "fill":
            array.fill(3)
        else:
            k_write(array)
        array[0] = 5
        assert array[0] == 5
        assert array[1] == 3
