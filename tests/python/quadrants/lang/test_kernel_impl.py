import os
import pathlib

import pytest

import quadrants as qd
from quadrants._test_tools import qd_init_same_arch

from tests import test_utils

_KERNEL_COVERAGE = os.environ.get("QD_KERNEL_COVERAGE") == "1"


@pytest.mark.skipif(
    _KERNEL_COVERAGE,
    reason="Coverage probes change the kernel AST, preventing FE-LL cache hits after reinit",
)
@test_utils.test()
def test_fe_ll_observations(tmp_path: pathlib.Path) -> None:
    @qd.kernel
    def k1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        a[0] += 1

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = qd.ndarray(qd.i32, (10,))
    assert not k1._primal.fe_ll_cache_observations.cache_hit
    k1(a)
    assert not k1._primal.fe_ll_cache_observations.cache_hit

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = qd.ndarray(qd.i32, (10,))
    k1._primal.fe_ll_cache_observations.cache_hit = False
    k1(a)
    assert k1._primal.fe_ll_cache_observations.cache_hit

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = qd.ndarray(qd.i32, (10,))
    k1._primal.fe_ll_cache_observations.cache_hit = False
    k1(a)
    assert k1._primal.fe_ll_cache_observations.cache_hit


@test_utils.test()
def test_ensure_compiled_reports_function():
    @qd.kernel
    def my_cool_test_function(a: qd.types.NDArray[qd.types.vector(2, qd.i32), 2]):
        pass

    x = qd.Matrix.ndarray(2, 3, qd.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=my_cool_test_function.__qualname__,
    ):
        my_cool_test_function(x)


@test_utils.test()
def test_pure_kernel_parameter() -> None:
    arch = qd.lang.impl.current_cfg().arch
    qd.init(arch=arch, offline_cache=False, src_ll_cache=True)

    @qd.pure
    @qd.kernel
    def k1(a: qd.types.NDArray) -> None:
        a[0] = 1

    @qd.kernel(pure=True)
    def k2(a: qd.types.NDArray) -> None:
        a[0] = 2

    @qd.kernel
    def k3(a: qd.types.NDArray) -> None:
        a[0] = 3

    @qd.kernel(pure=False)
    def k4(a: qd.types.NDArray) -> None:
        a[0] = 4

    @qd.kernel()
    def k5(a: qd.types.NDArray) -> None:
        a[0] = 5

    @qd.data_oriented
    class SomeClass:
        def __init__(self) -> None: ...

        @qd.kernel
        def da1(self, a: qd.types.NDArray) -> None:
            a[0] = 11

        @qd.pure
        @qd.kernel
        def da2(self, a: qd.types.NDArray) -> None:
            a[0] = 12

        @qd.kernel(pure=True)
        def da3(self, a: qd.types.NDArray) -> None:
            a[0] = 13

        @qd.kernel(pure=False)
        def da4(self, a: qd.types.NDArray) -> None:
            a[0] = 14

        @qd.kernel()
        def da5(self, a: qd.types.NDArray) -> None:
            a[0] = 15

    a = qd.ndarray(qd.i32, (10,))
    k1(a)
    assert k1._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 1
    k2(a)
    assert k2._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 2
    k3(a)
    assert not k3._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 3
    k4(a)
    assert not k4._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 4
    k5(a)
    assert not k4._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 5

    some_class = SomeClass()

    some_class.da1(a)
    assert not some_class.da1._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 11

    some_class.da2(a)
    assert some_class.da2._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 12

    some_class.da3(a)
    assert some_class.da3._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 13

    some_class.da4(a)
    assert not some_class.da4._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 14

    some_class.da5(a)
    assert not some_class.da5._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 15


@test_utils.test()
def test_member_kernel_rejects_foreign_owner() -> None:
    """Pin the "you forgot @qd.data_oriented" diagnostic for both routes into a member kernel: the per-class closure
    ``data_oriented`` installs, reached by an unbound ``SomeClass.member(other)``, and the generic class-kernel
    wrapper, reached by a class that was never decorated.
    """
    a = qd.ndarray(qd.i32, (10,))

    @qd.data_oriented
    class Decorated:
        @qd.kernel
        def write(self, arr: qd.types.NDArray) -> None:
            arr[0] = 7

    class Undecorated:
        @qd.kernel
        def write(self, arr: qd.types.NDArray) -> None:
            arr[0] = 8

    Decorated().write(a)
    assert a[0] == 7

    with pytest.raises(qd.QuadrantsSyntaxError, match="Undecorated.*qd.data_oriented"):
        Decorated.write(Undecorated(), a)

    with pytest.raises(qd.QuadrantsSyntaxError, match="Undecorated.*qd.data_oriented"):
        Undecorated().write(a)


@test_utils.test()
def test_fastcache_kernel_parameter() -> None:
    arch = qd.lang.impl.current_cfg().arch
    qd.init(arch=arch, offline_cache=False, src_ll_cache=True)

    @qd.pure
    @qd.kernel
    def k1(a: qd.types.NDArray) -> None:
        a[0] = 1

    @qd.kernel(fastcache=True)
    def k2(a: qd.types.NDArray) -> None:
        a[0] = 2

    @qd.kernel
    def k3(a: qd.types.NDArray) -> None:
        a[0] = 3

    @qd.kernel(fastcache=False)
    def k4(a: qd.types.NDArray) -> None:
        a[0] = 4

    @qd.kernel()
    def k5(a: qd.types.NDArray) -> None:
        a[0] = 5

    @qd.data_oriented
    class SomeClass:
        def __init__(self) -> None: ...

        @qd.kernel
        def da1(self, a: qd.types.NDArray) -> None:
            a[0] = 11

        @qd.pure
        @qd.kernel
        def da2(self, a: qd.types.NDArray) -> None:
            a[0] = 12

        @qd.kernel(fastcache=True)
        def da3(self, a: qd.types.NDArray) -> None:
            a[0] = 13

        @qd.kernel(fastcache=False)
        def da4(self, a: qd.types.NDArray) -> None:
            a[0] = 14

        @qd.kernel()
        def da5(self, a: qd.types.NDArray) -> None:
            a[0] = 15

    a = qd.ndarray(qd.i32, (10,))
    k1(a)
    assert k1._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 1
    k2(a)
    assert k2._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 2
    k3(a)
    assert not k3._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 3
    k4(a)
    assert not k4._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 4
    k5(a)
    assert not k4._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 5

    some_class = SomeClass()

    some_class.da1(a)
    assert not some_class.da1._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 11

    some_class.da2(a)
    assert some_class.da2._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 12

    some_class.da3(a)
    assert some_class.da3._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 13

    some_class.da4(a)
    assert not some_class.da4._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 14

    some_class.da5(a)
    assert not some_class.da5._primal.src_ll_cache_observations.cache_key_generated
    assert a[0] == 15
