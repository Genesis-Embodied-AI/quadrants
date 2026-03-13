from enum import IntEnum
from typing import cast

import pytest

import quadrants as qd
from quadrants.lang import _perf_dispatch
from quadrants.lang._perf_dispatch import NUM_WARMUP, PerformanceDispatcher, _parse_force_map
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


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", {}),
        ("foo:bar", {"foo": "bar"}),
        ("foo:bar,baz:qux", {"foo": "bar", "baz": "qux"}),
        (" foo : bar , baz : qux ", {"foo": "bar", "baz": "qux"}),
        ("a:b,,c:d", {"a": "b", "c": "d"}),
    ],
)
@test_utils.test()
def test_parse_force_map(raw, expected) -> None:
    assert _parse_force_map(raw) == expected


@test_utils.test()
def test_parse_force_map_malformed() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        _parse_force_map("no_colon")


@test_utils.test()
def test_perf_dispatch_force_by_name(monkeypatch) -> None:
    """Forcing a specific implementation bypasses benchmarking."""
    called = []

    monkeypatch.setattr(_perf_dispatch, "_FORCE_MAP", {"my_func": "impl_b"})
    monkeypatch.setattr(_perf_dispatch, "_ANY_FORCE_ACTIVE", True)

    @qd.perf_dispatch(get_geometry_hash=lambda a: hash(a.shape), repeat_after_seconds=0)
    def my_func(a: qd.types.NDArray[qd.i32, 1]): ...

    @my_func.register
    def impl_a(a: qd.types.NDArray[qd.i32, 1]) -> None:
        called.append("a")

    @my_func.register
    def impl_b(a: qd.types.NDArray[qd.i32, 1]) -> None:
        called.append("b")

    a = qd.ndarray(qd.i32, (4,))
    for _ in range(NUM_WARMUP + 3):
        my_func(a)

    assert len(called) == NUM_WARMUP + 3
    assert all(c == "b" for c in called)


@test_utils.test()
def test_perf_dispatch_force_unmatched_falls_back(monkeypatch) -> None:
    """An unmatched forced name falls back to normal benchmarking."""
    called = []

    monkeypatch.setattr(_perf_dispatch, "_FORCE_MAP", {"my_func": "nonexistent"})
    monkeypatch.setattr(_perf_dispatch, "_ANY_FORCE_ACTIVE", True)

    @qd.perf_dispatch(get_geometry_hash=lambda a: hash(a.shape), repeat_after_seconds=0)
    def my_func(a: qd.types.NDArray[qd.i32, 1]): ...

    @my_func.register
    def impl_a(a: qd.types.NDArray[qd.i32, 1]) -> None:
        called.append("a")

    a = qd.ndarray(qd.i32, (4,))
    my_func(a)
    assert called == ["a"]


@test_utils.test()
def test_perf_dispatch_force_multiple_dispatchers(monkeypatch) -> None:
    """Multiple dispatchers can each be forced to different implementations."""
    called_a = []
    called_b = []

    monkeypatch.setattr(_perf_dispatch, "_FORCE_MAP", {"op_a": "op_a_v2", "op_b": "op_b_v1"})
    monkeypatch.setattr(_perf_dispatch, "_ANY_FORCE_ACTIVE", True)

    @qd.perf_dispatch(get_geometry_hash=lambda a: hash(a.shape), repeat_after_seconds=0)
    def op_a(a: qd.types.NDArray[qd.i32, 1]): ...

    @op_a.register
    def op_a_v1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        called_a.append("v1")

    @op_a.register
    def op_a_v2(a: qd.types.NDArray[qd.i32, 1]) -> None:
        called_a.append("v2")

    @qd.perf_dispatch(get_geometry_hash=lambda a: hash(a.shape), repeat_after_seconds=0)
    def op_b(a: qd.types.NDArray[qd.i32, 1]): ...

    @op_b.register
    def op_b_v1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        called_b.append("v1")

    @op_b.register
    def op_b_v2(a: qd.types.NDArray[qd.i32, 1]) -> None:
        called_b.append("v2")

    a = qd.ndarray(qd.i32, (4,))
    for _ in range(NUM_WARMUP + 3):
        op_a(a)
        op_b(a)

    assert all(c == "v2" for c in called_a)
    assert all(c == "v1" for c in called_b)
