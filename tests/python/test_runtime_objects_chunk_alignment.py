import ctypes
import os
import subprocess
import sys

import pytest

import quadrants as qd

from tests import test_utils

RET_SUCCESS = 42

# cuMemAlloc only guarantees an alignment suitable for any built-in type (256 bytes in practice), so a handful of
# sub-page allocations walks the driver heap onto a sub-page-aligned base for the next allocation. This reproduces the
# state a third-party CUDA library leaves behind when it initialises before quadrants in the same process.
#
# Their total must not be a whole number of pages: the driver appears to service these out of one page at a time, so
# 4096 bytes' worth lands the next allocation back on a page boundary and the bug is not exercised at all. 3 * 512
# leaves the heap 1536 bytes into a page, which is what the original report used.
NUM_FRAGMENTING_ALLOCS = 3
FRAGMENTING_ALLOC_BYTES = 512
assert (NUM_FRAGMENTING_ALLOCS * FRAGMENTING_ALLOC_BYTES) % 4096 != 0


def fragmented_heap_child(args: list[str]) -> None:
    cuda = ctypes.CDLL("libcuda.so.1")
    assert cuda.cuInit(0) == 0

    device = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0

    # Quadrants retains the primary context as well, so these allocations fragment the very heap its runtime-objects
    # chunk is later carved out of.
    context = ctypes.c_void_p()
    assert cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) == 0
    assert cuda.cuCtxSetCurrent(context) == 0

    pointer = ctypes.c_void_p()
    for _ in range(NUM_FRAGMENTING_ALLOCS):
        assert cuda.cuMemAlloc_v2(ctypes.byref(pointer), FRAGMENTING_ALLOC_BYTES) == 0

    # materialize_runtime allocates the runtime-objects chunk here, and runtime_initialize bump-allocates page-aligned
    # blocks out of it against a budget carrying no slack for base misalignment. A sub-page-aligned base overruns the
    # chunk and trips a sticky in-kernel assert, after which every CUDA call in this process fails.
    qd.init(arch=qd.cuda)

    field = qd.field(qd.f32, shape=(8,))
    field.fill(1.0)
    assert field.to_numpy().tolist() == [1.0] * 8

    sys.exit(RET_SUCCESS)


@test_utils.test(arch=qd.cuda)
def test_init_survives_fragmented_driver_heap():
    if sys.platform != "linux":
        pytest.skip("The fragmentation helper loads libcuda.so.1")

    # Run in a subprocess: the failure mode is an in-kernel assert, which is sticky, so on a regression every later
    # CUDA call in the process fails too and the damage would spread across the rest of this worker's tests.
    cmd_line = [sys.executable, __file__, fragmented_heap_child.__name__]
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env)
    if proc.returncode != RET_SUCCESS:
        print(" ".join(cmd_line))
        print(proc.stdout)  # needs to do this to see error messages
        print("-" * 100)
        print(proc.stderr)
    assert proc.returncode == RET_SUCCESS


# The following lines are critical for the tests to work. If they are missing, the test will incorrectly pass, without
# doing anything.
if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
