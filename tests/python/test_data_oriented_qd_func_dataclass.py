"""Tests for calling @qd.func that takes a typed-dataclass arg, from a @qd.kernel
method of a @qd.data_oriented class, passing ``self.dataclass_member`` as the arg.

Genesis's @qd.func helpers declare typed-dataclass parameters (e.g.
``def func(links_state: LinksState, ...):``) and are designed to be called from kernels
that also take typed-dataclass kernel args (so the dataclass is flattened into per-leaf
kernel-locals on both sides of the call boundary).

When migrating Genesis modules to @qd.data_oriented, we'd like to call the same @qd.func
helpers from a data_oriented kernel method, passing ``self.links_state`` as the arg.
Today this fails at AST resolution:

    Missing argument '__qd_links_state__qd_cinr_inertial'.
    Unexpected argument 'links_state'.

These tests pin down the failure modes so we can fix them.
"""

import dataclasses

import numpy as np

import quadrants as qd

from tests import test_utils

# ----- typed-dataclass kernel-arg baseline (works) ----------------------------


@test_utils.test(arch=qd.cpu)
def test_baseline_typed_dataclass_kernel_arg_calls_qd_func():
    """Baseline: typed-dataclass kernel arg + qd.func taking same dataclass type — works."""
    N = 4

    @dataclasses.dataclass
    class State:
        x: qd.types.NDArray[qd.i32, 1]
        y: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def write_x(state: State, i: qd.i32, v: qd.i32):
        state.x[i] = v

    @qd.kernel
    def run(state: State):
        for i in range(N):
            write_x(state, i, i * 3)

    state = State(
        x=qd.ndarray(qd.i32, shape=(N,)),
        y=qd.ndarray(qd.i32, shape=(N,)),
    )
    run(state)
    np.testing.assert_array_equal(state.x.to_numpy(), np.arange(N) * 3)


# ----- data_oriented self-method calling qd.func (the broken case) -----------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_method_calls_qd_func_with_dataclass_member():
    """data_oriented holds a dataclass; self-kernel calls a @qd.func taking that dataclass."""
    N = 4

    @dataclasses.dataclass
    class State:
        x: qd.types.NDArray[qd.i32, 1]
        y: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def write_x(state: State, i: qd.i32, v: qd.i32):
        state.x[i] = v

    @qd.data_oriented
    class Solver:
        def __init__(self):
            self.state = State(
                x=qd.ndarray(qd.i32, shape=(N,)),
                y=qd.ndarray(qd.i32, shape=(N,)),
            )

        @qd.kernel
        def run(self):
            for i in range(N):
                write_x(self.state, i, i * 5)

    solver = Solver()
    solver.run()
    np.testing.assert_array_equal(solver.state.x.to_numpy(), np.arange(N) * 5)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_method_calls_qd_func_with_keyword_dataclass_member():
    """Same as above but the qd.func arg is passed by keyword (Genesis pattern)."""
    N = 4

    @dataclasses.dataclass
    class State:
        x: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def write_x(state: State, i: qd.i32, v: qd.i32):
        state.x[i] = v

    @qd.data_oriented
    class Solver:
        def __init__(self):
            self.state = State(x=qd.ndarray(qd.i32, shape=(N,)))

        @qd.kernel
        def run(self):
            for i in range(N):
                write_x(state=self.state, i=i, v=i * 7)

    solver = Solver()
    solver.run()
    np.testing.assert_array_equal(solver.state.x.to_numpy(), np.arange(N) * 7)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_stable_members_method_calls_qd_func_with_dataclass_member():
    """Same as above but with stable_members=True (the FPS-relevant case)."""
    N = 4

    @dataclasses.dataclass
    class State:
        x: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def write_x(state: State, i: qd.i32, v: qd.i32):
        state.x[i] = v

    @qd.data_oriented(stable_members=True)
    class Solver:
        def __init__(self):
            self.state = State(x=qd.ndarray(qd.i32, shape=(N,)))

        @qd.kernel
        def run(self):
            for i in range(N):
                write_x(state=self.state, i=i, v=i * 11)

    solver = Solver()
    solver.run()
    np.testing.assert_array_equal(solver.state.x.to_numpy(), np.arange(N) * 11)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_method_calls_qd_func_with_two_dataclass_members():
    """Two dataclass members, qd.func takes both — Genesis-shaped scenario."""
    N = 4

    @dataclasses.dataclass
    class StateA:
        a: qd.types.NDArray[qd.i32, 1]

    @dataclasses.dataclass
    class StateB:
        b: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def write_both(sa: StateA, sb: StateB, i: qd.i32, va: qd.i32, vb: qd.i32):
        sa.a[i] = va
        sb.b[i] = vb

    @qd.data_oriented(stable_members=True)
    class Solver:
        def __init__(self):
            self.sa = StateA(a=qd.ndarray(qd.i32, shape=(N,)))
            self.sb = StateB(b=qd.ndarray(qd.i32, shape=(N,)))

        @qd.kernel
        def run(self):
            for i in range(N):
                write_both(sa=self.sa, sb=self.sb, i=i, va=i * 2, vb=i * 13)

    solver = Solver()
    solver.run()
    np.testing.assert_array_equal(solver.sa.a.to_numpy(), np.arange(N) * 2)
    np.testing.assert_array_equal(solver.sb.b.to_numpy(), np.arange(N) * 13)


# ----- nested dataclass --------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_method_calls_qd_func_with_nested_dataclass_member():
    """data_oriented holds an Outer{ Inner{ ndarray } } and passes ``self.outer`` to a
    @qd.func that expands the nested dataclass into flat leaves on both sides."""
    N = 4

    @dataclasses.dataclass
    class Inner:
        x: qd.types.NDArray[qd.i32, 1]

    @dataclasses.dataclass
    class Outer:
        inner: Inner

    @qd.func
    def write_inner_x(outer: Outer, i: qd.i32, v: qd.i32):
        outer.inner.x[i] = v

    @qd.data_oriented
    class Solver:
        def __init__(self):
            self.outer = Outer(inner=Inner(x=qd.ndarray(qd.i32, shape=(N,))))

        @qd.kernel
        def run(self):
            for i in range(N):
                write_inner_x(self.outer, i, i * 17)

    solver = Solver()
    solver.run()
    np.testing.assert_array_equal(solver.outer.inner.x.to_numpy(), np.arange(N) * 17)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_method_calls_qd_func_with_nested_dataclass_kwarg():
    """Same as above but the dataclass arg is passed by keyword."""
    N = 4

    @dataclasses.dataclass
    class Inner:
        x: qd.types.NDArray[qd.i32, 1]

    @dataclasses.dataclass
    class Outer:
        inner: Inner

    @qd.func
    def write_inner_x(outer: Outer, i: qd.i32, v: qd.i32):
        outer.inner.x[i] = v

    @qd.data_oriented(stable_members=True)
    class Solver:
        def __init__(self):
            self.outer = Outer(inner=Inner(x=qd.ndarray(qd.i32, shape=(N,))))

        @qd.kernel
        def run(self):
            for i in range(N):
                write_inner_x(outer=self.outer, i=i, v=i * 19)

    solver = Solver()
    solver.run()
    np.testing.assert_array_equal(solver.outer.inner.x.to_numpy(), np.arange(N) * 19)


# ----- chained @qd.func calls (qd.func -> qd.func, dataclass threaded through) ---


@test_utils.test(arch=qd.cpu)
def test_data_oriented_method_qd_func_chain_with_dataclass_member():
    """data_oriented kernel calls outer @qd.func, which in turn calls inner @qd.func,
    threading the same dataclass arg through. Both qd.funcs have the typed-dataclass
    parameter; only the outermost call site (data_oriented method body) uses self.X.
    The two inner call sites use the typed-arg path that already worked."""
    N = 4

    @dataclasses.dataclass
    class State:
        x: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def inner_write(state: State, i: qd.i32, v: qd.i32):
        state.x[i] = v

    @qd.func
    def outer_write(state: State, i: qd.i32, v: qd.i32):
        inner_write(state, i, v)

    @qd.data_oriented
    class Solver:
        def __init__(self):
            self.state = State(x=qd.ndarray(qd.i32, shape=(N,)))

        @qd.kernel
        def run(self):
            for i in range(N):
                outer_write(self.state, i, i * 23)

    solver = Solver()
    solver.run()
    np.testing.assert_array_equal(solver.state.x.to_numpy(), np.arange(N) * 23)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_method_qd_func_chain_with_nested_dataclass_member():
    """Combination: nested dataclass passed through a chain of two @qd.func calls
    from a @qd.data_oriented self-method via self.outer."""
    N = 4

    @dataclasses.dataclass
    class Inner:
        x: qd.types.NDArray[qd.i32, 1]

    @dataclasses.dataclass
    class Outer:
        inner: Inner

    @qd.func
    def inner_write(outer: Outer, i: qd.i32, v: qd.i32):
        outer.inner.x[i] = v

    @qd.func
    def outer_write(outer: Outer, i: qd.i32, v: qd.i32):
        inner_write(outer, i, v)

    @qd.data_oriented(stable_members=True)
    class Solver:
        def __init__(self):
            self.outer = Outer(inner=Inner(x=qd.ndarray(qd.i32, shape=(N,))))

        @qd.kernel
        def run(self):
            for i in range(N):
                outer_write(self.outer, i, i * 29)

    solver = Solver()
    solver.run()
    np.testing.assert_array_equal(solver.outer.inner.x.to_numpy(), np.arange(N) * 29)
