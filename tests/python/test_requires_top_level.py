"""Tests for ``@qd.func(requires_top_level=True)``.

A func marked ``requires_top_level=True`` may only be called at the **top level** of a kernel (or
directly inside a ``while qd.graph_do_while(...)`` body, which the compiler also treats as top level).
Calling it nested inside ordinary runtime ``for`` / ``if`` / ``while`` control flow is rejected at
compile time with a :class:`QuadrantsSyntaxError`, because nesting demotes the func's per-phase
top-level loops out of top-level position (collapsing the inter-phase grid-wide barriers) and would
otherwise silently corrupt the result. ``qd.static`` (compile-time) loops do not trip the check.
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsSyntaxError
from quadrants.types.annotations import template

from tests import test_utils


@test_utils.test(arch=qd.cpu)
def test_top_level_call_is_allowed():
    """Calling a requires_top_level func directly at the kernel top level compiles and runs."""

    @qd.func(requires_top_level=True)
    def bump(x: template(), n):
        for i in range(n):
            x[i] = x[i] + 1

    @qd.kernel
    def run(x: qd.types.ndarray(qd.i32, ndim=1), n: qd.i32):
        bump(x, n)

    x = qd.ndarray(qd.i32, shape=(8,))
    x.from_numpy(np.zeros(8, dtype=np.int32))
    run(x, 8)
    assert np.all(x.to_numpy() == 1)


@test_utils.test(arch=qd.cpu)
def test_static_loop_is_allowed():
    """A ``qd.static`` loop is compile-time unrolled, so the call stays top level and is allowed."""

    @qd.func(requires_top_level=True)
    def bump(x: template(), n):
        for i in range(n):
            x[i] = x[i] + 1

    @qd.kernel
    def run(x: qd.types.ndarray(qd.i32, ndim=1), n: qd.i32):
        for _ in qd.static(range(2)):
            bump(x, n)

    x = qd.ndarray(qd.i32, shape=(8,))
    x.from_numpy(np.zeros(8, dtype=np.int32))
    run(x, 8)
    assert np.all(x.to_numpy() == 2)


@test_utils.test(arch=qd.cpu)
def test_nested_in_runtime_for_is_rejected():
    @qd.func(requires_top_level=True)
    def bump(x: template(), n):
        for i in range(n):
            x[i] = x[i] + 1

    @qd.kernel
    def run(x: qd.types.ndarray(qd.i32, ndim=1), n: qd.i32):
        for _ in range(1):
            bump(x, n)

    x = qd.ndarray(qd.i32, shape=(8,))
    with pytest.raises(QuadrantsSyntaxError, match="requires_top_level"):
        run(x, 8)


@test_utils.test(arch=qd.cpu)
def test_nested_in_runtime_if_is_rejected():
    @qd.func(requires_top_level=True)
    def bump(x: template(), n):
        for i in range(n):
            x[i] = x[i] + 1

    @qd.kernel
    def run(x: qd.types.ndarray(qd.i32, ndim=1), n: qd.i32, flag: qd.i32):
        if flag > 0:
            bump(x, n)

    x = qd.ndarray(qd.i32, shape=(8,))
    with pytest.raises(QuadrantsSyntaxError, match="requires_top_level"):
        run(x, 8, 1)


@test_utils.test(arch=qd.cpu)
def test_nested_in_runtime_while_is_rejected():
    @qd.func(requires_top_level=True)
    def bump(x: template(), n):
        for i in range(n):
            x[i] = x[i] + 1

    @qd.kernel
    def run(x: qd.types.ndarray(qd.i32, ndim=1), n: qd.i32, iters: qd.i32):
        k = iters
        while k > 0:
            bump(x, n)
            k = k - 1

    x = qd.ndarray(qd.i32, shape=(8,))
    with pytest.raises(QuadrantsSyntaxError, match="requires_top_level"):
        run(x, 8, 1)


@test_utils.test(arch=qd.cpu)
def test_unmarked_func_is_unaffected():
    """A plain ``@qd.func`` (the default) may still be nested in runtime control flow."""

    @qd.func
    def bump(x: template(), n):
        for i in range(n):
            x[i] = x[i] + 1

    @qd.kernel
    def run(x: qd.types.ndarray(qd.i32, ndim=1), n: qd.i32):
        for _ in range(1):
            bump(x, n)

    x = qd.ndarray(qd.i32, shape=(8,))
    x.from_numpy(np.zeros(8, dtype=np.int32))
    run(x, 8)  # no QuadrantsSyntaxError


@test_utils.test(arch=qd.cpu)
def test_bare_func_decorator_still_works():
    """Regression for the dual bare/factory decorator: bare ``@qd.func`` keeps working."""

    @qd.func
    def add2(x):
        return x + 2

    @qd.kernel
    def run(out: qd.types.ndarray(qd.i32, ndim=1)):
        out[0] = add2(40)

    out = qd.ndarray(qd.i32, shape=(1,))
    run(out)
    assert out.to_numpy()[0] == 42


@test_utils.test(arch=qd.cpu)
def test_class_func_detection_still_works():
    """Regression for the decorator's stack-frame handling: ``@qd.func`` on a class method must
    still be detected as a class func so it binds ``self`` correctly."""

    @qd.data_oriented
    class C:
        def __init__(self):
            self.a = qd.field(qd.i32, shape=(2,))

        @qd.func
        def add(self, x, y):
            return x + y

        @qd.kernel
        def run(self):
            self.a[0] = self.add(3, 7)

    c = C()
    c.run()
    assert c.a[0] == 10


@test_utils.test(arch=qd.cuda)
def test_graph_do_while_body_counts_as_top_level():
    """A ``graph_do_while`` body is treated as top level, so a requires_top_level func may be called
    directly inside it."""

    @qd.func(requires_top_level=True)
    def bump(x: template(), n):
        for i in range(n):
            x[i] = x[i] + 1

    @qd.kernel(graph=True)
    def run(
        x: qd.types.ndarray(qd.i32, ndim=1),
        n: qd.i32,
        counter: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            bump(x, n)
            for _ in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(8,))
    x.from_numpy(np.zeros(8, dtype=np.int32))
    counter = qd.ndarray(qd.i32, shape=())
    counter.from_numpy(np.array(3, dtype=np.int32))
    run(x, 8, counter)
    assert np.all(x.to_numpy() == 3)
