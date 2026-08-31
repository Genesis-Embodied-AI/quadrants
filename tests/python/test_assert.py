import platform

import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils

u = platform.uname()
if u.system == "linux" and u.machine in ("arm64", "aarch64"):
    pytest.skip("assert not currently supported on linux arm64 or aarch64", allow_module_level=True)


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_minimal():
    @qd.kernel
    def func():
        assert 0

    @qd.kernel
    def func2():
        assert False

    with pytest.raises(AssertionError):
        func()
    with pytest.raises(AssertionError):
        func2()


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_basic():
    @qd.kernel
    def func():
        x = 20
        assert 10 <= x < 20

    with pytest.raises(AssertionError):
        func()


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message():
    @qd.kernel
    def func():
        x = 20
        assert 10 <= x < 20, "Foo bar"

    with pytest.raises(AssertionError, match="Foo bar"):
        func()


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message_formatted():
    x = qd.field(dtype=int, shape=16)
    x[10] = 42

    @qd.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, "x[%d] expect=%d got=%d" % (i, 0, x[i])

    @qd.kernel
    def assert_float():
        y = 0.5
        assert y < 0, "y = %f" % y

    with pytest.raises(AssertionError, match=r"x\[10\] expect=0 got=42"):
        assert_formatted()
    # TODO: note that we are not fully polished to be able to recover from
    # assertion failures...
    with pytest.raises(AssertionError, match=r"y = 0.5"):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message_formatted_fstring():
    x = qd.field(dtype=int, shape=16)
    x[10] = 42

    @qd.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, f"x[{i}] expect={0} got={x[i]}"

    @qd.kernel
    def assert_float():
        y = 0.5
        assert y < 0, f"y = {y}"

    with pytest.raises(AssertionError, match=r"x\[10\] expect=0 got=42"):
        assert_formatted()
    # TODO: note that we are not fully polished to be able to recover from
    # assertion failures...
    with pytest.raises(AssertionError, match=r"y = 0.5"):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_ok():
    @qd.kernel
    def func():
        x = 20
        assert 10 <= x <= 20

    func()


@test_utils.test(
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_assert_with_check_oob():
    @qd.kernel
    def func():
        n = 15
        assert n >= 0

    func()


# AMDGPU in-kernel asserts are non-terminating: the faulting wave records the error and keeps running
# to the kernel's natural end (instead of the old `S_ENDPGM`, which killed only that wave and deadlocked
# peers parked on `s_barrier`, or a `__builtin_trap()`, which escalates to an uncatchable SIGABRT on some
# AMD arches). The host then raises QuadrantsAssertionError through the normal post-sync error check, so
# the HIP context stays usable afterwards. These tests pin that behavior on AMDGPU.
@test_utils.test(arch=[qd.amdgpu], require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_amdgpu_assert_barrier_no_hang():
    # Thread 0 fails the assert while its peers reach the block-level barrier. The non-terminating assert
    # lets thread 0 also reach the barrier, so the dispatch completes instead of deadlocking the host.
    n = 256

    @qd.kernel
    def boom_with_barrier():
        qd.loop_config(block_dim=n)
        for i in range(n):
            assert i != 0, "barrier assert probe"
            qd.simt.block.sync()

    with pytest.raises(AssertionError, match="barrier assert probe"):
        boom_with_barrier()


@test_utils.test(arch=[qd.amdgpu], require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_amdgpu_assert_context_survives():
    # After an assertion is caught, the context must remain usable: a subsequent kernel runs to
    # completion and produces correct results (no dead-context / one-assert-per-process limitation).
    @qd.kernel
    def boom():
        assert False, "amdgpu assert probe"

    @qd.kernel
    def add_one(x: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1

    arr = qd.ndarray(qd.i32, shape=(8,))
    arr.fill(0)

    with pytest.raises(AssertionError, match="amdgpu assert probe"):
        boom()

    add_one(arr)
    assert (arr.to_numpy() == 1).all()


@test_utils.test(
    arch=[qd.amdgpu],
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_amdgpu_assert_out_of_bound_no_fault():
    # With bounds checking on, an out-of-bounds access raises AssertionError. Because the assert is
    # non-terminating on AMDGPU, the offending index is clamped into range (see check_out_of_bound.cpp)
    # so the continued execution does not perform a real out-of-bounds access (which would fault / escalate
    # to an HSA exception). The context also stays alive afterwards.
    x = qd.field(dtype=qd.i32, shape=8)

    @qd.kernel
    def oob_field():
        for i in range(16):
            x[i] = i  # i >= 8 is out of bounds

    @qd.kernel
    def fill_ok():
        for i in range(8):
            x[i] = 100 + i

    with pytest.raises(AssertionError):
        oob_field()

    fill_ok()
    assert x.to_numpy().tolist() == [100, 101, 102, 103, 104, 105, 106, 107]

    arr = qd.ndarray(qd.i32, shape=(8,))
    arr.fill(0)

    @qd.kernel
    def oob_ndarray(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(16):
            a[i] = i

    with pytest.raises(AssertionError):
        oob_ndarray(arr)


@test_utils.test(arch=get_host_arch_list(), print_full_traceback=False)
def test_static_assert_message():
    x = 3

    @qd.kernel
    def func():
        qd.static_assert(x == 4, "Oh, no!")

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_static_assert_vector_n_ok():
    x = qd.Vector.field(4, qd.f32, ())

    @qd.kernel
    def func():
        qd.static_assert(x.n == 4)

    func()


@test_utils.test(arch=get_host_arch_list())
def test_static_assert_data_type_ok():
    x = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        qd.static_assert(x.dtype == qd.f32)

    func()


@test_utils.test()
def test_static_assert_nonstatic_condition():
    @qd.kernel
    def foo():
        value = False
        qd.static_assert(value, "Oh, no!")

    with pytest.raises(qd.QuadrantsTypeError, match="Static assert with non-static condition"):
        foo()
