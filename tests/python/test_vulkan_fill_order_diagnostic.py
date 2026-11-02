"""Compare host-write ordering using the base and PR wheels, without rebuilding either."""

from collections import Counter
import inspect
import os

import pytest
import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("initialization", ["existing_fill", "allocation_fill"])
@pytest.mark.parametrize("sync_before_write", [False, True])
@test_utils.test(arch=qd.vulkan)
def test_fill_then_host_write(initialization: str, sync_before_write: bool) -> None:
    # Repeat the original API and allocation-time API with and without a synchronization before the host write.
    print(f"wheel={os.environ['DIAGNOSTIC_WHEEL']} source={os.environ['DIAGNOSTIC_SOURCE_SHA']}")
    print(f"imported={qd.__file__} ndarray_signature={inspect.signature(qd.ndarray)}")
    if initialization == "allocation_fill" and "fill" not in inspect.signature(qd.ndarray).parameters:
        pytest.skip("The base wheel has no allocation-time fill parameter")

    num_trials = 100
    observed = Counter()
    for _ in range(num_trials):
        if initialization == "existing_fill":
            array = qd.ndarray(qd.i32, shape=4)
            array.fill(3)
        else:
            array = qd.ndarray(qd.i32, shape=4, fill=3)
        if sync_before_write:
            qd.sync()
        array[0] = 5
        observed[int(array[0])] += 1
        del array

    print(f"initialization={initialization} sync_before_write={sync_before_write} results={dict(observed)}")
    assert observed == {5: num_trials}, dict(observed)
