import gc
import sys

import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu])
def test_ndarray_doesnt_crash_on_gc() -> None:
    if sys.platform != "darwin":
        pytest.skip("Only need to check Mac CPU")
    arch = getattr(qd, qd.cfg.arch.name)
    for n in range(100):
        qd.init(arch=arch)
        gc.collect()
        a = qd.ndarray(qd.i32, shape=(55,))
        gc.get_objects()
        b = qd.ndarray(qd.i32, shape=(57,))
        gc.get_objects()
        c = qd.ndarray(qd.i32, shape=(211,))
        gc.get_objects()
        z_param = qd.ndarray(qd.i32, shape=(223,))
        gc.get_objects()
        bar_param = qd.ndarray(qd.i32, shape=(227,))

        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v} it {n}"

        gc.get_objects()

        @qd.kernel
        def k1(z_param2: qd.types.NDArray[qd.i32, 1]) -> None:
            z_param2[33] += 2

        gc.get_objects()

        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v} it {n}"
        gc.collect()
        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v} it {n}"
        gc.collect()
        k1(z_param)


@test_utils.test()
def test_ndarray_reset() -> None:
    arch = getattr(qd, qd.cfg.arch.name)
    qd.reset()
    qd.init(arch=arch)
    rt = qd.lang.impl.get_runtime()
    assert len(rt.ndarrays) == 0
    a = qd.ndarray(qd.i32, shape=(55,))
    assert len(rt.ndarrays) == 1
    b = qd.ndarray(qd.i32, shape=(57,))
    assert len(rt.ndarrays) == 2
    c = qd.ndarray(qd.i32, shape=(211,))
    assert len(rt.ndarrays) == 3
    z_param = qd.ndarray(qd.i32, shape=(223,))
    assert len(rt.ndarrays) == 4
    bar_param = qd.ndarray(qd.i32, shape=(227,))
    assert len(rt.ndarrays) == 5

    tmp = qd.ndarray(qd.i32, shape=(42,))
    assert len(rt.ndarrays) == 6
    tmp = None
    gc.collect()
    assert len(rt.ndarrays) == 5

    @qd.kernel
    def k1(z_param2: qd.types.NDArray[qd.i32, 1]) -> None:
        z_param2[33] += 2

    k1(z_param)
    assert len(rt.ndarrays) == 5

    qd.reset()
    rt = qd.lang.impl.get_runtime()
    assert len(rt.ndarrays) == 0

    assert a.arr is None
    assert b.arr is None
    assert c.arr is None
    assert z_param.arr is None
    assert bar_param.arr is None

    assert a.shape is None
