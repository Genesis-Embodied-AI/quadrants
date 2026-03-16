import platform

import pytest

import quadrants as qd

from tests import test_utils

u = platform.uname()


def test_cpu_debug_snode_reader():
    qd.init(arch=qd.x64, debug=True)

    x = qd.field(qd.f32, shape=())
    x[None] = 10.0

    assert x[None] == 10.0


@pytest.mark.skipif(
    u.system == "linux" and u.machine in ("arm64", "aarch64"),
    reason="assert not currently supported on linux arm64 or aarch64",
)
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_cpu_debug_snode_writer_out_of_bound():
    x = qd.field(qd.f32, shape=3)

    with pytest.raises(AssertionError):
        x[3] = 10.0


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_cpu_debug_snode_writer_out_of_bound_negative():
    x = qd.field(qd.f32, shape=3)
    with pytest.raises(AssertionError):
        x[-1] = 10.0


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_cpu_debug_snode_reader_out_of_bound():
    x = qd.field(qd.f32, shape=3)

    with pytest.raises(AssertionError):
        a = x[3]


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_cpu_debug_snode_reader_out_of_bound_negative():
    x = qd.field(qd.f32, shape=3)
    with pytest.raises(AssertionError):
        a = x[-1]


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_out_of_bound():
    x = qd.field(qd.i32, shape=(8, 16))

    @qd.kernel
    def func():
        x[3, 16] = 1

    with pytest.raises(RuntimeError):
        func()


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_not_out_of_bound():
    x = qd.field(qd.i32, shape=(8, 16))

    @qd.kernel
    def func():
        x[7, 15] = 1

    func()


@test_utils.test(
    require=[qd.extension.sparse, qd.extension.assertion],
    debug=True,
    gdb_trigger=False,
    exclude=qd.metal,
)
def test_out_of_bound_dynamic():
    x = qd.field(qd.i32)

    qd.root.dynamic(qd.i, 16, 4).place(x)

    @qd.kernel
    def func():
        x[17] = 1

    with pytest.raises(RuntimeError):
        func()


@test_utils.test(
    require=[qd.extension.sparse, qd.extension.assertion],
    debug=True,
    gdb_trigger=False,
    exclude=qd.metal,
)
def test_not_out_of_bound_dynamic():
    x = qd.field(qd.i32)

    qd.root.dynamic(qd.i, 16, 4).place(x)

    @qd.kernel
    def func():
        x[3] = 1

    func()


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_out_of_bound_with_offset():
    x = qd.field(qd.i32, shape=(8, 16), offset=(-8, -8))

    @qd.kernel
    def func():
        x[0, 0] = 1

    with pytest.raises(RuntimeError):
        func()
        func()


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_not_out_of_bound_with_offset():
    x = qd.field(qd.i32, shape=(8, 16), offset=(-4, -8))

    @qd.kernel
    def func():
        x[-4, -8] = 1
        x[3, 7] = 2

    func()


@test_utils.test(
    arch=[qd.cpu],
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_ndarray_oob_cpu_raises_not_segfaults():
    """Out-of-bounds ndarray access in a parallel kernel on CPU should raise
    QuadrantsAssertionError instead of segfaulting."""
    arr = qd.ndarray(dtype=qd.f32, shape=(4,))

    @qd.kernel
    def write_oob(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(10):
            a[i] = 1.0

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        write_oob(arr)


@test_utils.test(
    arch=[qd.cpu],
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_ndarray_oob_cpu_small_array():
    """Reproduces the pattern from the temperature-sensor segfault: a kernel
    accesses a very small (shape-1) array with an index that goes out of
    bounds.  Before the longjmp fix this would SIGSEGV on CPU in debug mode."""
    small = qd.ndarray(dtype=qd.f32, shape=(1,))
    small.fill(42.0)

    @qd.kernel
    def read_oob(a: qd.types.ndarray(dtype=qd.f32, ndim=1)) -> qd.f32:
        return a[5]

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        read_oob(small)


@test_utils.test(
    arch=[qd.cpu],
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_ndarray_oob_cpu_2d():
    """2D ndarray out-of-bounds on CPU should produce a clear error."""
    arr = qd.ndarray(dtype=qd.f32, shape=(3, 4))

    @qd.kernel
    def write_oob_2d(a: qd.types.ndarray(dtype=qd.f32, ndim=2)):
        for i in range(1):
            a[10, 0] = 1.0

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        write_oob_2d(arr)


@test_utils.test(
    arch=[qd.cpu],
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_ndarray_inbounds_cpu_still_works():
    """Verify that the setjmp/longjmp mechanism does not break normal
    in-bounds ndarray access."""
    n = 8
    arr = qd.ndarray(dtype=qd.f32, shape=(n,))

    @qd.kernel
    def fill(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(n):
            a[i] = qd.cast(i * 10, qd.f32)

    fill(arr)
    result = arr.to_numpy()
    for i in range(n):
        assert result[i] == pytest.approx(i * 10)
