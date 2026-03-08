from enum import IntEnum
from typing import cast

import pytest

import quadrants as qd
from quadrants.lang._perf_dispatch import NUM_WARMUP, PerformanceDispatcher
from quadrants.lang.exception import QuadrantsSyntaxError

from tests import test_utils


@qd.func
def do_work(i_b, amount_work: qd.i32, state: qd.types.NDArray[qd.i32, 1]):
    x = state[i_b]
    for _ in range(amount_work):
        x = (1664527 * x + 1013904223) % 2147483647
    state[i_b] = x


def do_work_py(i_b, amount_work: qd.i32, state: qd.types.NDArray[qd.i32, 1]):
    x = state[i_b]
    for _ in range(amount_work):
        x = (1664527 * x + 1013904223) % 2147483647
    state[i_b] = x


@test_utils.test()
def test_perf_dispatch_kernels() -> None:
    class ImplEnum(IntEnum):
        slow = 0
        fastest_a_shape0_lt2 = 1
        a_shape0_ge2 = 2

    @qd.perf_dispatch(get_geometry_hash=lambda a, c, rand_state: hash(a.shape + c.shape), repeat_after_seconds=0)
    def my_func1(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ): ...

    @my_func1.register
    @qd.kernel
    def my_func1_impl_slow(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.slow] = 1
            do_work(i_b=i_b, amount_work=10000, state=rand_state)

    @my_func1.register(is_compatible=lambda a, c, rand_state: a.shape[0] < 2)
    @qd.kernel
    def my_func1_impl_a_shape0_lt_2(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.fastest_a_shape0_lt2] = 1
            do_work(i_b=i_b, amount_work=1, state=rand_state)

    # a.shape is [num_threads], ie a.shape[0] is num_threads, which is more than 2
    @my_func1.register(is_compatible=lambda a, c, rand_state: a.shape[0] >= 2)
    @qd.kernel
    def my_func1_impl_a_shape0_ge_2(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.a_shape0_ge2] = 1
            do_work(i_b=i_b, amount_work=100, state=rand_state)

    num_threads = 10  # should be at least more than 2
    a = qd.ndarray(qd.i32, (num_threads,))
    c = qd.ndarray(qd.i32, (len(ImplEnum),))
    rand_state = qd.ndarray(qd.i32, (num_threads,))

    for it in range((NUM_WARMUP + 5)):
        c.fill(0)
        for _inner_it in range(2):  # 2 compatible kernels
            a.fill(5)
            my_func1(a, c, rand_state=rand_state)
            assert (a.to_numpy()[:5] == [0, 5, 10, 15, 20]).all()
        if it <= NUM_WARMUP:
            assert c[ImplEnum.slow] == 1
            assert c[ImplEnum.fastest_a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 1
        else:
            assert c[ImplEnum.slow] == 0
            assert c[ImplEnum.fastest_a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 1
    speed_checker = cast(PerformanceDispatcher, my_func1)
    geometry = list(speed_checker._trial_count_by_dispatch_impl_by_geometry_hash.keys())[0]
    for _dispatch_impl, trials in speed_checker._trial_count_by_dispatch_impl_by_geometry_hash[geometry].items():
        assert trials == NUM_WARMUP + 1
    assert len(speed_checker._trial_count_by_dispatch_impl_by_geometry_hash[geometry]) == 2


@test_utils.test()
def test_perf_dispatch_python() -> None:
    class ImplEnum(IntEnum):
        slow = 0
        fastest_a_shape0_lt2 = 1
        a_shape0_ge2 = 2

    @qd.perf_dispatch(get_geometry_hash=lambda a, c, rand_state: hash(a.shape + c.shape), repeat_after_seconds=0)
    def my_func1(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ): ...

    @my_func1.register
    def my_func1_impl_slow(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.slow] = 1
            do_work_py(i_b=i_b, amount_work=10000, state=rand_state)

    @my_func1.register(is_compatible=lambda a, c, rand_state: a.shape[0] < 2)
    def my_func1_impl_a_shape0_lt_2(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.fastest_a_shape0_lt2] = 1
            do_work_py(i_b=i_b, amount_work=1, state=rand_state)

    # a.shape is [num_threads], ie a.shape[0] is num_threads, which is more than 2
    @my_func1.register(is_compatible=lambda a, c, rand_state: a.shape[0] >= 2)
    def my_func1_impl_a_shape0_ge_2(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.a_shape0_ge2] = 1
            do_work_py(i_b=i_b, amount_work=100, state=rand_state)

    num_threads = 10  # should be at least more than 2
    a = qd.ndarray(qd.i32, (num_threads,))
    c = qd.ndarray(qd.i32, (len(ImplEnum),))
    rand_state = qd.ndarray(qd.i32, (num_threads,))

    for it in range((NUM_WARMUP + 5)):
        c.fill(0)
        for _inner_it in range(2):  # 2 compatible kernels
            a.fill(5)
            my_func1(a, c, rand_state=rand_state)
            assert (a.to_numpy()[:5] == [0, 5, 10, 15, 20]).all()
        if it <= NUM_WARMUP:
            assert c[ImplEnum.slow] == 1
            assert c[ImplEnum.fastest_a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 1
        else:
            assert c[ImplEnum.slow] == 0
            assert c[ImplEnum.fastest_a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 1
    speed_checker = cast(PerformanceDispatcher, my_func1)
    geometry = list(speed_checker._trial_count_by_dispatch_impl_by_geometry_hash.keys())[0]
    for _dispatch_impl, trials in speed_checker._trial_count_by_dispatch_impl_by_geometry_hash[geometry].items():
        assert trials == NUM_WARMUP + 1
    assert len(speed_checker._trial_count_by_dispatch_impl_by_geometry_hash[geometry]) == 2


@test_utils.test()
def test_perf_dispatch_kernel_py_mix() -> None:
    class ImplEnum(IntEnum):
        slow = 0
        fastest_a_shape0_lt2 = 1
        a_shape0_ge2 = 2

    @qd.perf_dispatch(get_geometry_hash=lambda a, c, rand_state: hash(a.shape + c.shape), repeat_after_seconds=0)
    def my_func1(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ): ...

    @my_func1.register
    def my_func1_impl_slow(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.slow] = 1
            do_work_py(i_b=i_b, amount_work=10000, state=rand_state)

    @my_func1.register(is_compatible=lambda a, c, rand_state: a.shape[0] < 2)
    @qd.kernel
    def my_func1_impl_a_shape0_lt_2(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.fastest_a_shape0_lt2] = 1
            do_work(i_b=i_b, amount_work=1, state=rand_state)

    # a.shape is [num_threads], ie a.shape[0] is num_threads, which is more than 2
    @my_func1.register(is_compatible=lambda a, c, rand_state: a.shape[0] >= 2)
    @qd.kernel
    def my_func1_impl_a_shape0_ge_2(
        a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            c[ImplEnum.a_shape0_ge2] = 1
            do_work(i_b=i_b, amount_work=100, state=rand_state)

    num_threads = 10  # should be at least more than 2
    a = qd.ndarray(qd.i32, (num_threads,))
    c = qd.ndarray(qd.i32, (len(ImplEnum),))
    rand_state = qd.ndarray(qd.i32, (num_threads,))

    for it in range((NUM_WARMUP + 5)):
        c.fill(0)
        for _inner_it in range(2):  # 2 compatible kernels
            a.fill(5)
            my_func1(a, c, rand_state=rand_state)
            assert (a.to_numpy()[:5] == [0, 5, 10, 15, 20]).all()
        if it <= NUM_WARMUP:
            assert c[ImplEnum.slow] == 1
            assert c[ImplEnum.fastest_a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 1
        else:
            assert c[ImplEnum.slow] == 0
            assert c[ImplEnum.fastest_a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 1
    speed_checker = cast(PerformanceDispatcher, my_func1)
    geometry = list(speed_checker._trial_count_by_dispatch_impl_by_geometry_hash.keys())[0]
    for _dispatch_impl, trials in speed_checker._trial_count_by_dispatch_impl_by_geometry_hash[geometry].items():
        assert trials == NUM_WARMUP + 1
    assert len(speed_checker._trial_count_by_dispatch_impl_by_geometry_hash[geometry]) == 2


@test_utils.test()
def test_perf_dispatch_swap_annotation_order() -> None:
    @qd.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape))
    def my_func1(a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]): ...

    with pytest.raises(QuadrantsSyntaxError, match="KERNEL_ANNOTATION_ORDER"):

        @qd.kernel
        @my_func1.register
        def my_func1_impl_serial(a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]) -> None: ...


@test_utils.test()
def test_perf_dispatch_annotation_mismatch() -> None:
    @qd.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape))
    def my_func1(a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]): ...

    # first arg different
    with pytest.raises(QuadrantsSyntaxError, match="PERFDISPATCH_ANNOTATION_SEQUENCE_MISMATCH"):

        @qd.kernel
        @my_func1.register
        def my_func1_impl_impl1(b: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]) -> None: ...

    # second arg different
    with pytest.raises(QuadrantsSyntaxError, match="PERFDISPATCH_ANNOTATION_SEQUENCE_MISMATCH"):

        @qd.kernel
        @my_func1.register
        def my_func1_impl_impl2(a: qd.types.NDArray[qd.i32, 1], b: qd.types.NDArray[qd.i32, 1]) -> None: ...

    # too few args
    with pytest.raises(QuadrantsSyntaxError, match="PERFDISPATCH_ANNOTATION_SEQUENCE_MISMATCH"):

        @qd.kernel
        @my_func1.register
        def my_func1_impl_impl2(a: qd.types.NDArray[qd.i32, 1]) -> None: ...

    # too many args
    with pytest.raises(QuadrantsSyntaxError, match="PERFDISPATCH_ANNOTATION_SEQUENCE_MISMATCH"):

        @qd.kernel
        @my_func1.register
        def my_func1_impl_impl2(
            a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1], d: qd.types.NDArray[qd.i32, 1]
        ) -> None: ...


@test_utils.test()
def test_perf_dispatch_sanity_check_register_args() -> None:
    @qd.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape), warmup=25, active=25)
    def my_func1(a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]): ...


@test_utils.test()
def test_perf_dispatch_multiple_perf_dispatch() -> None:
    work_multiplier = 1000

    @qd.kernel
    def linesearch(a: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]):
        B = a.shape[0]
        for i_b in range(B):
            do_work(i_b=i_b, amount_work=work_multiplier * 5, state=rand_state)

    @qd.kernel
    def hessian(a: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]):
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] = a[i_b] * i_b
            do_work(i_b=i_b, amount_work=work_multiplier * 7, state=rand_state)

    @qd.perf_dispatch(get_geometry_hash=lambda a, rand_state: hash(a.shape), repeat_after_seconds=1)
    def my_func_init(
        a: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ): ...

    @qd.perf_dispatch(get_geometry_hash=lambda iterations_np, a, rand_state: hash(a.shape), repeat_after_seconds=1)
    def my_func_body(
        iterations_np: qd.types.NDArray[qd.i32, 0],
        a: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ): ...

    @my_func_init.register
    @qd.kernel
    def my_func_init_monolith(
        a: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            do_work(i_b=i_b, amount_work=work_multiplier * 1, state=rand_state)
            do_work(i_b=i_b, amount_work=work_multiplier * 1, state=rand_state)

    @my_func_init.register(is_compatible=lambda a, rand_state: True)
    def my_func_init_decomp(
        a: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        linesearch(a, rand_state)
        hessian(a, rand_state)

    @my_func_body.register
    @qd.kernel
    def my_func_body_monolith(
        iterations_np: qd.types.NDArray[qd.i32, 0],
        a: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        B = a.shape[0]
        for i_b in range(B):
            do_work(i_b=i_b, amount_work=work_multiplier * 1, state=rand_state)
            do_work(i_b=i_b, amount_work=work_multiplier * 1, state=rand_state)

    @my_func_body.register(is_compatible=lambda a, rand_state: True)
    def my_func_body_decomp(
        iterations_np: qd.types.NDArray[qd.i32, 0],
        a: qd.types.NDArray[qd.i32, 1], rand_state: qd.types.NDArray[qd.i32, 1]
    ) -> None:
        iterations = iterations_np[()]
        for it in range(iterations):
            linesearch(a, rand_state)
            hessian(a, rand_state)

    def resolve(num_iterations, a_init, a_body, rand_state_init, rand_state_body):
        my_func_init(a_init, rand_state_init)
        my_func_body(num_iterations, a_body, rand_state_body)

    num_threads = 10  # should be at least more than 2
    a_init = qd.ndarray(qd.i32, (num_threads,))
    a_body = qd.ndarray(qd.i32, (num_threads,))
    rand_state_init = qd.ndarray(qd.i32, (num_threads,))
    rand_state_body = qd.ndarray(qd.i32, (num_threads,))
    num_iterations = qd.ndarray(qd.i32, ())
    num_iterations[()] = 10

    for it in range(10):
        a_init.fill(5)
        a_body.fill(5)
        resolve(num_iterations, a_init, a_body, rand_state_init, rand_state_body)
