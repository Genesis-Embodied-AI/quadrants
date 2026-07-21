"""Tests for the ``qd.graph`` namespace and the deprecation of the flat ``qd.graph_*`` spellings.

``qd.graph.do_while`` / ``qd.graph.parallel_context`` / ``qd.graph.parallel`` are the canonical spellings; the flat
``qd.graph_do_while`` / ``qd.graph_parallel_context`` / ``qd.graph_parallel`` names remain valid but emit a
``DeprecationWarning`` at compile time. Both spellings are recognized by the AST compiler. The pure-AST matching /
warning logic is unit-tested directly (backend-independent); the functional and deprecation-warning behaviour is also
exercised end-to-end through real graph kernels.
"""

import ast
import warnings

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.ast.ast_transformers import graph_api

from tests import test_utils


def _func(expr_src):
    """Parse a call expression and return its ``func`` AST node (e.g. ``qd.graph.do_while`` from ``...(x)``)."""
    return ast.parse(expr_src, mode="eval").body.func


# --------------------------------------------------------------------------------------------------------------------
# Namespace wiring: the qd.graph.* names are the same objects as the deprecated flat qd.graph_* names.
# --------------------------------------------------------------------------------------------------------------------
def test_graph_namespace_aliases_are_identical():
    assert qd.graph.do_while is qd.graph_do_while
    assert qd.graph.parallel_context is qd.graph_parallel_context
    assert qd.graph.parallel is qd.graph_parallel


# --------------------------------------------------------------------------------------------------------------------
# Pure-AST matching / warning logic (no runtime backend needed).
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,new_src,old_src",
    [
        ("do_while", "qd.graph.do_while(x)", "qd.graph_do_while(x)"),
        ("parallel_context", "qd.graph.parallel_context()", "qd.graph_parallel_context()"),
        ("parallel", "qd.graph.parallel()", "qd.graph_parallel()"),
    ],
)
def test_graph_api_matches_both_spellings(name, new_src, old_src):
    # Canonical qd.graph.<name> and the deprecated flat qd.graph_<name> both match.
    assert graph_api.matches(_func(new_src), name)
    assert graph_api.matches(_func(old_src), name)
    # `graph.<name>` (from quadrants import graph) also matches.
    assert graph_api.matches(_func(new_src.replace("qd.graph", "graph")), name)
    # Only the flat spelling is flagged deprecated.
    assert graph_api.is_deprecated_spelling(_func(old_src), name)
    assert not graph_api.is_deprecated_spelling(_func(new_src), name)
    # Unrelated calls and the wrong construct do not match.
    assert not graph_api.matches(_func("qd.something(x)"), name)
    assert not graph_api.matches(_func("qd.graph.other(x)"), name)


def test_graph_api_warns_only_on_deprecated_spelling():
    with pytest.warns(DeprecationWarning, match=r"qd\.graph\.do_while"):
        graph_api.warn_if_deprecated(_func("qd.graph_do_while(x)"), "do_while")
    # The canonical spelling must never warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        graph_api.warn_if_deprecated(_func("qd.graph.do_while(x)"), "do_while")


# --------------------------------------------------------------------------------------------------------------------
# End-to-end: the qd.graph.* spellings compile and run correctly on every backend.
# --------------------------------------------------------------------------------------------------------------------
@test_utils.test()
def test_graph_do_while_namespace():
    N = 16

    @qd.kernel(graph=True)
    def graph_loop(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph.do_while(counter):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for i in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    graph_loop(x, counter)

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5, dtype=np.int32))


@test_utils.test()
def test_graph_parallel_namespace():
    n = 256

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.f32, ndim=1),
        y: qd.types.ndarray(qd.f32, ndim=1),
        z: qd.types.ndarray(qd.f32, ndim=1),
    ):
        with qd.graph.parallel_context():
            with qd.graph.parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            with qd.graph.parallel():
                for i in range(y.shape[0]):
                    y[i] = y[i] + 2.0
        for i in range(z.shape[0]):
            z[i] = x[i] + y[i]

    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))
    z = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))
    z.from_numpy(np.zeros(n, dtype=np.float32))

    k(x, y, z)

    np.testing.assert_array_equal(x.to_numpy(), np.full(n, 1.0, dtype=np.float32))
    np.testing.assert_array_equal(y.to_numpy(), np.full(n, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(z.to_numpy(), np.full(n, 3.0, dtype=np.float32))


# --------------------------------------------------------------------------------------------------------------------
# End-to-end: the deprecated flat spellings still work but warn; the canonical spellings do not warn.
# offline_cache=False so the kernel is traced (and the compile-time warning fires) on this run.
# --------------------------------------------------------------------------------------------------------------------
def _graph_deprecation_warnings(recorded):
    return [w for w in recorded if issubclass(w.category, DeprecationWarning) and "qd.graph" in str(w.message)]


@test_utils.test(offline_cache=False)
def test_graph_do_while_flat_spelling_deprecated():
    N = 8

    @qd.kernel(graph=True)
    def graph_loop(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(counter):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for i in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(3, dtype=np.int32))

    with pytest.warns(DeprecationWarning, match=r"qd\.graph\.do_while"):
        graph_loop(x, counter)

    # The deprecated spelling must still behave identically.
    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))


@test_utils.test(offline_cache=False)
def test_graph_do_while_namespace_does_not_warn():
    N = 8

    @qd.kernel(graph=True)
    def graph_loop(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph.do_while(counter):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for i in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(3, dtype=np.int32))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        graph_loop(x, counter)
    assert _graph_deprecation_warnings(recorded) == []


@test_utils.test(offline_cache=False)
def test_graph_parallel_flat_spelling_deprecated():
    n = 64

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            with qd.graph_parallel():
                for i in range(y.shape[0]):
                    y[i] = y[i] + 2.0

    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))

    with pytest.warns(DeprecationWarning) as record:
        k(x, y)
    messages = " ".join(str(w.message) for w in record)
    assert "qd.graph.parallel_context" in messages
    assert "qd.graph.parallel(" in messages

    np.testing.assert_array_equal(x.to_numpy(), np.full(n, 1.0, dtype=np.float32))
    np.testing.assert_array_equal(y.to_numpy(), np.full(n, 2.0, dtype=np.float32))
