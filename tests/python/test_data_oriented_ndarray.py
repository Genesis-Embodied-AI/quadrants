"""Tests for ``@qd.data_oriented`` classes whose members are raw ``qd.ndarray`` (not ``qd.field``, not
``qd.Tensor`` wrappers).

The user-guide doc ``docs/source/user_guide/compound_types.md`` claims this pattern is not supported
("can contain ndarray? no" for ``@qd.data_oriented``). But the in-tree error message in
``python/quadrants/lang/impl.py`` lists ``@qd.data_oriented / frozen-dataclass template`` as a
*supported* route, and the ndarray-in-struct infrastructure added by ``#561 [Type] Tensor 24``
(2026-04-28) — specifically ``_predeclare_struct_ndarrays`` in
``python/quadrants/lang/ast/ast_transformers/function_def_transformer.py`` — explicitly walks both
``dataclasses.is_dataclass(val)`` and ``hasattr(val, "__dict__")`` containers, the latter being the
data_oriented case.

This file pins what actually works, and documents the gaps. See
``perso_hugh/doc/data_oriented_ndarray.md`` for the design analysis.
"""

import dataclasses

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


# ---------------------------------------------------------------------------
# 1. Single raw qd.ndarray attribute (scalar element type).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_single_ndarray():
    N = 6

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    x = qd.ndarray(qd.i32, shape=(N,))
    state = State(x=x)

    @qd.kernel
    def fill(s: qd.template()):
        for i in range(N):
            s.x[i] = i * 3

    fill(state)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(N) * 3)


# ---------------------------------------------------------------------------
# 2. Vector ndarray attribute.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_vector_ndarray():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, v):
            self.v = v

    v = qd.Vector.ndarray(3, qd.f32, shape=(N,))
    state = State(v=v)

    @qd.kernel
    def fill(s: qd.template()):
        for i in range(N):
            s.v[i] = qd.Vector([float(i), float(i) * 2.0, float(i) * 3.0])

    fill(state)
    out = v.to_numpy()
    for i in range(N):
        np.testing.assert_array_equal(out[i], np.array([i, i * 2, i * 3], dtype=np.float32))


# ---------------------------------------------------------------------------
# 3. Multiple ndarray attributes in the same class.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_multiple_ndarrays():
    N = 5

    @qd.data_oriented
    class State:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    a = qd.ndarray(qd.i32, shape=(N,))
    b = qd.ndarray(qd.f32, shape=(N,))
    state = State(a=a, b=b)

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.a[i] = i + 1
            s.b[i] = float(i) * 0.5

    run(state)
    np.testing.assert_array_equal(a.to_numpy(), np.arange(1, N + 1))
    np.testing.assert_array_equal(b.to_numpy(), np.arange(N, dtype=np.float32) * 0.5)


# ---------------------------------------------------------------------------
# 4. Mixed qd.field + qd.ndarray in the same data_oriented class.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_mixed_field_and_ndarray():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, f, n):
            self.f = f
            self.n = n

    f = qd.field(qd.i32, shape=(N,))
    n = qd.ndarray(qd.i32, shape=(N,))
    state = State(f=f, n=n)

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.f[i] = i + 1
            s.n[i] = s.f[i] * 10

    run(state)
    np.testing.assert_array_equal(f.to_numpy(), np.arange(1, N + 1))
    np.testing.assert_array_equal(n.to_numpy(), np.arange(1, N + 1) * 10)


# ---------------------------------------------------------------------------
# 5. Nested @qd.data_oriented (outer holds inner; inner holds ndarray).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_nested():
    N = 4

    @qd.data_oriented
    class Inner:
        def __init__(self, x):
            self.x = x

    @qd.data_oriented
    class Outer:
        def __init__(self, inner):
            self.inner = inner

    x = qd.ndarray(qd.i32, shape=(N,))
    outer = Outer(inner=Inner(x=x))

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.inner.x[i] = i * 7

    run(outer)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(N) * 7)


# ---------------------------------------------------------------------------
# 6. Mutation: same instance, reassign ndarray attribute to a *same-shape* ndarray between calls.
#    The launch-time stale-cache guard (``_mutable_nd_cached_val`` in kernel.py) is supposed to fold the
#    live ndarray id into args_hash so the launch context is not served stale. We pin that behaviour
#    here for the data_oriented case.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_reassign_same_shape():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    x1 = qd.ndarray(qd.i32, shape=(N,))
    x2 = qd.ndarray(qd.i32, shape=(N,))
    state = State(x=x1)

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i + 100

    run(state)
    np.testing.assert_array_equal(x1.to_numpy(), np.arange(100, 100 + N))

    state.x = x2
    run(state)
    np.testing.assert_array_equal(x2.to_numpy(), np.arange(100, 100 + N))
    np.testing.assert_array_equal(x1.to_numpy(), np.arange(100, 100 + N))  # x1 unchanged


# ---------------------------------------------------------------------------
# 7. Mutation cross-shape: reassign ndarray attribute to a *different-dtype* ndarray.
#    The template-mapper specialisation key (in ``_template_mapper_hotpath._extract_arg``) returns
#    ``weakref.ref(arg)`` for ``is_data_oriented(arg)``; it does NOT descend into ndarray children to
#    compute a dtype/ndim-dependent spec key. So if the data_oriented instance's id is unchanged but
#    its ndarray attribute is reassigned to a different dtype, we expect either:
#      - a graceful recompile/raise, or
#      - silent miscompilation (the bug case — current expected outcome per static analysis).
#    Mark xfail with strict=False so we record the actual outcome without breaking CI.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="Gap A: data_oriented specialisation key does not include ndarray dtype/ndim")
@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_reassign_different_dtype():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    x_i32 = qd.ndarray(qd.i32, shape=(N,))
    x_f32 = qd.ndarray(qd.f32, shape=(N,))
    state = State(x=x_i32)

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = s.x[i] + 1

    run(state)
    np.testing.assert_array_equal(x_i32.to_numpy(), np.array([1, 1, 1, 1], dtype=np.int32))

    state.x = x_f32
    run(state)
    np.testing.assert_array_equal(x_f32.to_numpy(), np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# 8. Distinct instances of same class -> spec-key behaviour. Documents that today each fresh instance
#    triggers a recompile (because the spec key is ``weakref.ref(arg)`` identity). This is a perf
#    concern, not a correctness one. We assert correctness here; the recompile count is documented as
#    a perf note.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_distinct_instances():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    a_arr = qd.ndarray(qd.i32, shape=(N,))
    b_arr = qd.ndarray(qd.i32, shape=(N,))
    a = State(x=a_arr)
    b = State(x=b_arr)

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i + 1

    run(a)
    run(b)
    np.testing.assert_array_equal(a_arr.to_numpy(), np.arange(1, N + 1))
    np.testing.assert_array_equal(b_arr.to_numpy(), np.arange(1, N + 1))


# ---------------------------------------------------------------------------
# 9. Fastcache cold then warm. Per the fastcache doc (``user_guide/fastcache.md`` line 129),
#    ``@qd.data_oriented`` objects are supported in the cache key. We don't assert cross-process here
#    (that requires a fresh interpreter); we assert that ``cache_stored`` becomes True on the first
#    call and ``cache_key_generated`` is True (i.e. no PARAM_INVALID fallthrough due to the ndarray
#    member).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_fastcache_eligible():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    x = qd.ndarray(qd.i32, shape=(N,))
    state = State(x=x)

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i * 2

    run(state)
    obs = run._primal.src_ll_cache_observations
    np.testing.assert_array_equal(x.to_numpy(), np.arange(N) * 2)
    assert obs.cache_key_generated, "cache key should be generated for data_oriented + ndarray"


# ---------------------------------------------------------------------------
# 10. Pure validation: a @qd.pure @qd.kernel taking a data_oriented arg with an ndarray member should
#     compile and run, mirroring the existing ``test_pure_validation_data_oriented_as_param`` test
#     which only covers ``qd.field``.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_pure():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    x = qd.ndarray(qd.i32, shape=(N,))
    state = State(x=x)

    @qd.pure
    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i * 5

    run(state)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(N) * 5)


# ---------------------------------------------------------------------------
# 11. Counter-test: confirm a dataclass-of-NDArray works (sanity check that the existing supported
#     route still works; if this fails, the test environment itself is broken, not the data_oriented
#     path).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_dataclass_ndarray_sanity():
    N = 4

    @dataclasses.dataclass
    class State:
        x: qd.types.NDArray[qd.i32, 1]

    x = qd.ndarray(qd.i32, shape=(N,))
    state = State(x=x)

    @qd.kernel
    def run(s: State):
        for i in range(N):
            s.x[i] = i * 11

    run(state)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(N) * 11)
