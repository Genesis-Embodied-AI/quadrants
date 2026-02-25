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
