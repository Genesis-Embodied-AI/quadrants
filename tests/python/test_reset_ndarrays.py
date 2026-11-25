import dataclasses
import gc
import sys
import weakref

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.slow
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


@test_utils.test(arch=qd.cpu)
def test_reset_releases_template_arguments() -> None:
    """Check that reset releases cached template arguments and the kernel works after reinitialization."""

    class Data:
        def __init__(self, value: int) -> None:
            self.field = qd.field(qd.i32, shape=())
            self.field[None] = value

    @qd.kernel
    def read(data: qd.Template) -> qd.i32:
        return data.field[None]

    data = Data(3)
    data_ref = weakref.ref(data)
    field_ref = weakref.ref(data.field)
    assert read(data) == 3
    del data
    qd.reset()
    gc.collect()
    assert data_ref() is None
    assert field_ref() is None

    qd.init(arch=qd.cpu)
    assert read(Data(7)) == 7


@test_utils.test(arch=qd.cpu)
def test_reset_clears_nested_dataclass_launch_metadata() -> None:
    """Rebuild scalar and ndarray launch metadata when specialization identifiers restart after reset."""

    @dataclasses.dataclass
    class State:
        count: int
        values: qd.types.NDArray[qd.i32, 1]

    @qd.data_oriented(template_primitives=False)
    class Data:
        def __init__(self, value: int) -> None:
            values = qd.ndarray(qd.i32, shape=1)
            values[0] = value
            self.state = State(value, values)
            self.offset = value

    @qd.kernel
    def read(data: qd.Template) -> qd.i32:
        return data.state.count + data.state.values[0] + data.offset

    data = Data(3)
    assert read(data) == 9
    kernel = read._primal
    assert kernel.mapper.mapping
    assert kernel._launch_infos_for_root_dataclasses_in_data_oriented_by_key
    assert kernel._runtime_scalar_launch_infos_by_compiled_kernel_key

    qd.reset()
    assert not kernel.mapper.mapping
    assert not kernel.used_py_dataclass_parameters_by_key_enforcing
    assert not kernel._struct_ndarray_launch_info_by_key
    assert not kernel._launch_infos_for_root_dataclasses_in_data_oriented_by_key
    assert not kernel._runtime_scalar_launch_infos_by_compiled_kernel_key
    assert kernel._mutable_nd_cached_key is None
    assert not kernel._mutable_nd_cached_val

    qd.init(arch=qd.cpu)
    data = Data(7)
    assert read(data) == 21
    data.state.count = 11
    assert read(data) == 25
