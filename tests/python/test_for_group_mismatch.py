import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list(), print_full_traceback=False)
def test_struct_for_mismatch():
    x = qd.field(qd.f32, (3, 4))

    @qd.kernel
    def func():
        for i in x:
            print(i)

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list(), print_full_traceback=False)
def test_struct_for_mismatch2():
    x = qd.field(qd.f32, (3, 4))

    @qd.kernel
    def func():
        for i, j, k in x:
            print(i, j, k)

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def _test_grouped_struct_for_mismatch():
    # doesn't work for now
    # need grouped refactor
    # for now, it just throw a unfriendly message:
    # AssertionError: __getitem__ cannot be called in Python-scope
    x = qd.field(qd.f32, (3, 4))

    @qd.kernel
    def func():
        for i, j in qd.grouped(x):
            print(i, j)

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def _test_ndrange_for_mismatch():
    # doesn't work for now
    # need ndrange refactor
    @qd.kernel
    def func():
        for i in qd.ndrange(3, 4):
            print(i)

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def _test_ndrange_for_mismatch2():
    # doesn't work for now
    # need ndrange and grouped refactor
    @qd.kernel
    def func():
        for i, j, k in qd.ndrange(3, 4):
            print(i, j, k)

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def _test_grouped_ndrange_for_mismatch():
    # doesn't work for now
    # need ndrange and grouped refactor
    @qd.kernel
    def func():
        for i in qd.grouped(qd.ndrange(3, 4)):
            print(i)

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def _test_static_ndrange_for_mismatch():
    # doesn't work for now
    # need ndrange and static refactor
    @qd.kernel
    def func():
        for i in qd.static(qd.ndrange(3, 4)):
            print(i)

    with pytest.raises(qd.QuadrantsCompilationError):
        func()
