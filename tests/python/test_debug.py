import platform

import pytest

import quadrants as ti

from tests import test_utils

u = platform.uname()


def test_cpu_debug_snode_reader():
    ti.init(arch=ti.x64, debug=True)

    x = ti.field(ti.f32, shape=())
    x[None] = 10.0

    assert x[None] == 10.0


@pytest.mark.skipif(
    u.system == "linux" and u.machine in ("arm64", "aarch64"),
    reason="assert not currently supported on linux arm64 or aarch64",
)
@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_cpu_debug_snode_writer_out_of_bound():
    x = ti.field(ti.f32, shape=3)

    with pytest.raises(AssertionError):
        x[3] = 10.0


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_cpu_debug_snode_writer_out_of_bound_negative():
    x = ti.field(ti.f32, shape=3)
    with pytest.raises(AssertionError):
        x[-1] = 10.0


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_cpu_debug_snode_reader_out_of_bound():
    x = ti.field(ti.f32, shape=3)

    with pytest.raises(AssertionError):
        a = x[3]


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_cpu_debug_snode_reader_out_of_bound_negative():
    x = ti.field(ti.f32, shape=3)
    with pytest.raises(AssertionError):
        a = x[-1]


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_out_of_bound():
    x = ti.field(ti.i32, shape=(8, 16))

    @ti.kernel
    def func():
        x[3, 16] = 1

    with pytest.raises(RuntimeError):
        func()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_not_out_of_bound():
    x = ti.field(ti.i32, shape=(8, 16))

    @ti.kernel
    def func():
        x[7, 15] = 1

    func()


@test_utils.test(
    require=[ti.extension.sparse, ti.extension.assertion],
    debug=True,
    gdb_trigger=False,
    exclude=ti.metal,
)
def test_out_of_bound_dynamic():
    x = ti.field(ti.i32)

    ti.root.dynamic(ti.i, 16, 4).place(x)

    @ti.kernel
    def func():
        x[17] = 1

    with pytest.raises(RuntimeError):
        func()


@test_utils.test(
    require=[ti.extension.sparse, ti.extension.assertion],
    debug=True,
    gdb_trigger=False,
    exclude=ti.metal,
)
def test_not_out_of_bound_dynamic():
    x = ti.field(ti.i32)

    ti.root.dynamic(ti.i, 16, 4).place(x)

    @ti.kernel
    def func():
        x[3] = 1

    func()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_out_of_bound_with_offset():
    x = ti.field(ti.i32, shape=(8, 16), offset=(-8, -8))

    @ti.kernel
    def func():
        x[0, 0] = 1

    with pytest.raises(RuntimeError):
        func()
        func()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_not_out_of_bound_with_offset():
    x = ti.field(ti.i32, shape=(8, 16), offset=(-4, -8))

    @ti.kernel
    def func():
        x[-4, -8] = 1
        x[3, 7] = 2

    func()
