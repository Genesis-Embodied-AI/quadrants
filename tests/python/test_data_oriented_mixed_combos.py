"""Spot tests for the nesting compatibility matrix in compound_types.md.

These are not part of the main fix's test surface; they exist to empirically verify the table claims
in the user-facing doc.
"""

import dataclasses

import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cpu)
def test_data_oriented_with_ndarray_field_and_nested_data_oriented():
    """A single @qd.data_oriented holding all three of: ndarray, field, nested @qd.data_oriented."""

    N = 4

    @qd.data_oriented
    class Inner:
        def __init__(self):
            self.y = qd.ndarray(qd.i32, shape=(N,))

    @qd.data_oriented
    class State:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.f = qd.field(qd.i32, shape=(N,))
            self.inner = Inner()
            self.scale = 7

    state = State()

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i
            s.f[i] = i * 2
            s.inner.y[i] = i + s.scale

    run(state)
    np.testing.assert_array_equal(state.x.to_numpy(), np.arange(N))
    np.testing.assert_array_equal(state.f.to_numpy(), np.arange(N) * 2)
    np.testing.assert_array_equal(state.inner.y.to_numpy(), np.arange(N) + 7)


@test_utils.test(arch=qd.cpu)
def test_dataclass_with_data_oriented_via_template():
    """A dataclass (frozen=True) holding a @qd.data_oriented holding an ndarray, passed via qd.template()."""
    N = 4

    @qd.data_oriented
    class Inner:
        def __init__(self):
            self.y = qd.ndarray(qd.i32, shape=(N,))

    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: Inner

    outer = Outer(inner=Inner())

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.inner.y[i] = i + 11

    run(outer)
    np.testing.assert_array_equal(outer.inner.y.to_numpy(), np.arange(N) + 11)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_with_dataclass_and_ndarray_sibling():
    """@qd.data_oriented holding both a direct ndarray AND a dataclass-with-ndarray sibling."""
    N = 4

    @dataclasses.dataclass
    class Inner:
        z: qd.types.ndarray(dtype=qd.i32, ndim=1)

    @qd.data_oriented
    class State:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.inner = Inner(z=qd.ndarray(qd.i32, shape=(N,)))

    state = State()

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i + 1
            s.inner.z[i] = i + 100

    run(state)
    np.testing.assert_array_equal(state.x.to_numpy(), np.arange(1, N + 1))
    np.testing.assert_array_equal(state.inner.z.to_numpy(), np.arange(N) + 100)
