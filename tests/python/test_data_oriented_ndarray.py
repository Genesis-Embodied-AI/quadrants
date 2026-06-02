"""Tests for ``@qd.data_oriented`` classes whose members are raw ``qd.ndarray`` (not ``qd.field``, not ``qd.Tensor``
wrappers).

The user-guide doc ``docs/source/user_guide/compound_types.md`` claims this pattern is not supported ("can contain
ndarray? no" for ``@qd.data_oriented``). But the in-tree error message in ``python/quadrants/lang/impl.py`` lists
``@qd.data_oriented / frozen-dataclass template`` as a *supported* route, and the ndarray-in-struct infrastructure
added by ``#561 [Type] Tensor 24`` (2026-04-28) — specifically ``_predeclare_struct_ndarrays`` in
``python/quadrants/lang/ast/ast_transformers/function_def_transformer.py`` — explicitly walks both
``dataclasses.is_dataclass(val)`` and ``hasattr(val, "__dict__")`` containers, the latter being the data_oriented case.

This file pins what actually works, and documents the gaps. See ``perso_hugh/doc/data_oriented_ndarray.md`` for the
design analysis.
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
# 6. Mutation: same instance, reassign ndarray attribute to a *same-shape* ndarray between calls. The launch-time
# stale-cache guard (``_mutable_nd_cached_val`` in kernel.py) is supposed to fold the live ndarray id into args_hash
# so the launch context is not served stale. We pin that behaviour here for the data_oriented case.
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
# 7. Mutation cross-shape: reassign ndarray attribute to a *different-dtype* ndarray. The template-mapper
# specialisation key (in ``_template_mapper_hotpath._extract_arg``) returns ``weakref.ref(arg)`` for
# ``is_data_oriented(arg)``; it does NOT descend into ndarray children to compute a dtype/ndim-dependent spec key.
# So if the data_oriented instance's id is unchanged but its ndarray attribute is reassigned to a different dtype,
# we expect either:
#   - a graceful recompile/raise, or
#   - silent miscompilation (the bug case — current expected outcome per static analysis).
# Mark xfail with strict=False so we record the actual outcome without breaking CI.
# ---------------------------------------------------------------------------


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
# 8. Distinct instances of same class -> spec-key behaviour. Documents that today each fresh instance triggers a
# recompile (because the spec key is ``weakref.ref(arg)`` identity). This is a perf concern, not a correctness one.
# We assert correctness here; the recompile count is documented as a perf note.
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
# 9. Fastcache cold then warm. Per the fastcache doc (``user_guide/fastcache.md`` line 129), ``@qd.data_oriented``
# objects are supported in the cache key. We don't assert cross-process here (that requires a fresh interpreter); we
# assert that ``cache_stored`` becomes True on the first call and ``cache_key_generated`` is True (i.e. no
# PARAM_INVALID fallthrough due to the ndarray member).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9b. Fastcache end-to-end with ``@qd.data_oriented`` holding ndarrays. Pattern adapted from
# ``test_cache.test_fastcache``: call ``qd_init_same_arch`` twice with the same cache directory to simulate two
# processes, monkeypatch ``launch_kernel`` to capture whether ``compiled_kernel_data`` was loaded from disk. On the
# second init the data_oriented + ndarray kernel should be served from the on-disk fastcache.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_fastcache_cross_init(tmp_path, monkeypatch):
    from quadrants._test_tools import qd_init_same_arch

    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel
    captured_compiled_kernel_data = []

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=None):
        # Filter to the user kernel only; .to_numpy() launches an internal ``ndarray_to_ext_arr`` kernel that is not
        # fastcache-eligible (is_pure=False) and would always make compiled_kernel_data=None, masking the actual
        # fastcache behaviour of ``run``.
        if self.func.__name__ == "run":
            captured_compiled_kernel_data.append(compiled_kernel_data)
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=qd_stream)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        for i in range(4):
            s.x[i] = i * 3

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    state = State(x=qd.ndarray(qd.i32, shape=(4,)))
    run(state)
    np.testing.assert_array_equal(state.x.to_numpy(), np.arange(4) * 3)
    assert captured_compiled_kernel_data[-1] is None, "cold init should compile, not load"

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    state = State(x=qd.ndarray(qd.i32, shape=(4,)))
    run(state)
    np.testing.assert_array_equal(state.x.to_numpy(), np.arange(4) * 3)
    assert captured_compiled_kernel_data[-1] is not None, "warm init should load from disk fastcache"


# ---------------------------------------------------------------------------
# 9c. Same as 9b but with a *nested* ``@qd.data_oriented`` holding an ndarray. Pins that the fastcache args_hasher
# recursion handles nested data_oriented containers correctly across processes.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_nested_ndarray_fastcache_cross_init(tmp_path, monkeypatch):
    from quadrants._test_tools import qd_init_same_arch

    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel
    captured = []

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=None):
        # Filter to the user kernel only; .to_numpy() launches a non-fastcache internal kernel that would otherwise
        # drown the run-kernel data we care about.
        if self.func.__name__ == "run":
            captured.append(compiled_kernel_data)
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=qd_stream)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.data_oriented
    class Inner:
        def __init__(self, y):
            self.y = y

    @qd.data_oriented
    class Outer:
        def __init__(self, inner):
            self.inner = inner

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        for i in range(4):
            s.inner.y[i] = i + 11

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    outer = Outer(inner=Inner(y=qd.ndarray(qd.i32, shape=(4,))))
    run(outer)
    assert captured[-1] is None

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    outer = Outer(inner=Inner(y=qd.ndarray(qd.i32, shape=(4,))))
    run(outer)
    assert captured[-1] is not None, "nested data_oriented + ndarray should load from fastcache"


# ---------------------------------------------------------------------------
# 9d. Fastcache key is dtype-sensitive: same kernel source, different ndarray dtype in the data_oriented member ->
# two distinct disk cache entries. Pins the args_hasher's ``[nd-{dtype}-{ndim}{layout}]`` repr.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_fastcache_dtype_key_distinct(tmp_path, monkeypatch):
    from quadrants._test_tools import qd_init_same_arch

    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel
    captured = []

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=None):
        # Filter to the user kernel only; .to_numpy() launches a non-fastcache internal kernel that would otherwise
        # drown the run-kernel data we care about.
        if self.func.__name__ == "run":
            captured.append(compiled_kernel_data)
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=qd_stream)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        for i in range(4):
            s.x[i] = s.x[i] + 1

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    state_i32 = State(x=qd.ndarray(qd.i32, shape=(4,)))
    state_f32 = State(x=qd.ndarray(qd.f32, shape=(4,)))
    run(state_i32)
    run(state_f32)
    assert captured[-2] is None and captured[-1] is None, "both dtypes cold-compile on first init"

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    state_i32 = State(x=qd.ndarray(qd.i32, shape=(4,)))
    state_f32 = State(x=qd.ndarray(qd.f32, shape=(4,)))
    run(state_i32)
    run(state_f32)
    assert captured[-2] is not None and captured[-1] is not None, "both dtypes load from disk"
    np.testing.assert_array_equal(state_i32.x.to_numpy(), [1, 1, 1, 1])
    np.testing.assert_array_equal(state_f32.x.to_numpy(), np.array([1.0] * 4, dtype=np.float32))


# ---------------------------------------------------------------------------
# 9e. Documented fallback: a @qd.data_oriented containing a qd.field disables fastcache for the whole call
# (args_hasher returns None for ScalarField). The kernel still runs correctly via non-fastcache compilation. This
# test pins the documented fallback so a future "support fields in fastcache" change explicitly chooses to update
# this test.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_field_disables_fastcache_but_runs(tmp_path, monkeypatch):
    from quadrants._test_tools import qd_init_same_arch

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.data_oriented
    class State:
        def __init__(self, n):
            self.f = qd.field(qd.i32, shape=(n,))

    state = State(4)

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        for i in range(4):
            s.f[i] = i + 7

    run(state)
    obs = run._primal.src_ll_cache_observations
    assert obs.cache_key_generated is False, "field child should disable fastcache key generation"
    np.testing.assert_array_equal(state.f.to_numpy(), np.arange(4) + 7)


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
# 10. Pure validation: a @qd.pure @qd.kernel taking a data_oriented arg with an ndarray member should compile and
# run, mirroring the existing ``test_pure_validation_data_oriented_as_param`` test which only covers ``qd.field``.
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
# 11. Counter-test: confirm a dataclass-of-NDArray works (sanity check that the existing supported route still
# works; if this fails, the test environment itself is broken, not the data_oriented path).
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


# ---------------------------------------------------------------------------
# 12. data_oriented holding a (frozen) dataclass that holds an ndarray. Exercises the ``else`` branch of
# ``_walk_obj`` recursing through a dataclass child — added by the Bug 1 fix.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_holding_dataclass_with_ndarray():
    N = 4

    @dataclasses.dataclass(frozen=True)
    class Inner:
        x: qd.types.NDArray[qd.i32, 1]

    @qd.data_oriented
    class Outer:
        def __init__(self, inner):
            self.inner = inner

    x = qd.ndarray(qd.i32, shape=(N,))
    outer = Outer(inner=Inner(x=x))

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.inner.x[i] = i + 1

    run(outer)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(1, N + 1))


# ---------------------------------------------------------------------------
# 13. Frozen dataclass holding a data_oriented holding an ndarray, kernel-arg via ``qd.template()``. Exercises the
# dataclass branch of ``_walk_obj`` recursing through a data_oriented child — added by the Bug 1 fix. The outer
# dataclass must be frozen because (i) non-frozen dataclasses are unhashable in Python (``__hash__ is None``) and the
# template-mapper key tuple needs the value to be hashable, and (ii) the typed-dataclass-arg form
# (``def run(s: Outer):``) goes through ``_transform_kernel_arg`` which does not currently recurse on data_oriented
# field *types* (as opposed to values) — that's a separate follow-up.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_dataclass_holding_data_oriented_with_ndarray():
    N = 4

    @qd.data_oriented
    class Inner:
        def __init__(self, x):
            self.x = x

    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: Inner

    x = qd.ndarray(qd.i32, shape=(N,))
    outer = Outer(inner=Inner(x=x))

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.inner.x[i] = i + 5

    run(outer)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(5, 5 + N))


# ---------------------------------------------------------------------------
# 14. Three-level nesting: data_oriented(data_oriented(data_oriented(ndarray))).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_three_level_nesting():
    N = 4

    @qd.data_oriented
    class L3:
        def __init__(self, x):
            self.x = x

    @qd.data_oriented
    class L2:
        def __init__(self, l3):
            self.l3 = l3

    @qd.data_oriented
    class L1:
        def __init__(self, l2):
            self.l2 = l2

    x = qd.ndarray(qd.i32, shape=(N,))
    root = L1(l2=L2(l3=L3(x=x)))

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.l2.l3.x[i] = i * 13

    run(root)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(N) * 13)


# ---------------------------------------------------------------------------
# 15. Mutation on a nested ndarray: outer.inner.x reassigned between kernel calls. Verifies the Bug 2 stale-cache
# guard fires even when the ndarray lives several attribute hops deep.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_nested_ndarray_reassign():
    N = 4

    @qd.data_oriented
    class Inner:
        def __init__(self, x):
            self.x = x

    @qd.data_oriented
    class Outer:
        def __init__(self, inner):
            self.inner = inner

    x1 = qd.ndarray(qd.i32, shape=(N,))
    x2 = qd.ndarray(qd.i32, shape=(N,))
    outer = Outer(inner=Inner(x=x1))

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.inner.x[i] = i + 200

    run(outer)
    np.testing.assert_array_equal(x1.to_numpy(), np.arange(200, 200 + N))

    outer.inner.x = x2
    run(outer)
    np.testing.assert_array_equal(x2.to_numpy(), np.arange(200, 200 + N))


# ---------------------------------------------------------------------------
# 15b. Reassignment through a FROZEN outer container that wraps a MUTABLE inner container holding the ndarray.
# Reproducer for Codex review #1 on PR #704
# (https://github.com/Genesis-Embodied-AI/quadrants/pull/704#discussion_r3253017477): the launch-time mutable-nd cache
# guard only classified mutability of the *top-level* kernel arg. With a frozen dataclass at the root, the predicate
# returned False and no per-call walk was inserted, so reassigning ``outer.inner.x`` between launches left the
# launch-context cache bound to the *original* ndarray.
#
# Fixed by widening the predicate to OR-fold mutability across every intermediate container along the recorded
# attr-chain (not just the root). See ``launch_kernel`` in ``python/quadrants/lang/kernel.py``.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_outer_mutable_inner_ndarray_reassign():
    N = 4

    @qd.data_oriented
    class Inner:
        def __init__(self, x):
            self.x = x

    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: Inner

    x1 = qd.ndarray(qd.i32, shape=(N,))
    x2 = qd.ndarray(qd.i32, shape=(N,))
    outer = Outer(inner=Inner(x=x1))

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.inner.x[i] = i + 300

    run(outer)
    np.testing.assert_array_equal(x1.to_numpy(), np.arange(300, 300 + N))

    # Reassign the leaf ndarray on the (mutable) inner container while the (frozen) outer container is unchanged at
    # the top level. id(outer) does NOT change. The launch-context cache must still invalidate so the second launch
    # binds against x2, not the cached x1.
    outer.inner.x = x2
    run(outer)
    np.testing.assert_array_equal(x2.to_numpy(), np.arange(300, 300 + N))


# ---------------------------------------------------------------------------
# 16. Same data_oriented instance, two kernels sharing it. Verifies the launch-info per-kernel bookkeeping is
# independent (each kernel's compile sets up its own pre-declared ndarray args).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_two_kernels_same_instance():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    x = qd.ndarray(qd.i32, shape=(N,))
    y = qd.ndarray(qd.i32, shape=(N,))
    state = State(x=x, y=y)

    @qd.kernel
    def fill_x(s: qd.template()):
        for i in range(N):
            s.x[i] = i + 1

    @qd.kernel
    def fill_y_from_x(s: qd.template()):
        for i in range(N):
            s.y[i] = s.x[i] * 100

    fill_x(state)
    fill_y_from_x(state)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(1, N + 1))
    np.testing.assert_array_equal(y.to_numpy(), np.arange(1, N + 1) * 100)


# ---------------------------------------------------------------------------
# 17. data_oriented + ndarray + @qd.func sub-call. Pins that the AST-time attribute resolution in ``build_Attribute``
# (which uses the predeclared AnyArray cache) works when the access happens inside a func, not just the top-level
# kernel.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_via_func():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    x = qd.ndarray(qd.i32, shape=(N,))
    state = State(x=x)

    @qd.func
    def write(s: qd.template(), i: qd.i32, v: qd.i32):
        s.x[i] = v

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            write(s, i, i * 9)

    run(state)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(N) * 9)


# ---------------------------------------------------------------------------
# 18. Reassign ndarray to a *different ndim* on the same data_oriented instance. Complementary to test 7
# (different-dtype). Spec key must change so a 2D-specialised kernel is not reused for a 1D ndarray. Pins the Gap A
# fix from the dtype side.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_reassign_different_ndim():
    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    x_1d = qd.ndarray(qd.i32, shape=(4,))
    x_2d = qd.ndarray(qd.i32, shape=(2, 3))
    state = State(x=x_1d)

    @qd.kernel
    def fill_1d(s: qd.template()):
        for i in range(4):
            s.x[i] = i * 2

    @qd.kernel
    def fill_2d(s: qd.template()):
        for i, j in qd.ndrange(2, 3):
            s.x[i, j] = i * 10 + j

    fill_1d(state)
    np.testing.assert_array_equal(x_1d.to_numpy(), np.arange(4) * 2)

    state.x = x_2d
    fill_2d(state)
    np.testing.assert_array_equal(x_2d.to_numpy(), np.array([[0, 1, 2], [10, 11, 12]], dtype=np.int32))


# ---------------------------------------------------------------------------
# 19. Spec-key descent for nested data_oriented + ndarray reassign at the leaf. Confirms the recursive walker in
# ``_collect_struct_nd_descriptors`` reaches through nested data_oriented.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_nested_ndarray_reassign_different_dtype():
    @qd.data_oriented
    class Inner:
        def __init__(self, x):
            self.x = x

    @qd.data_oriented
    class Outer:
        def __init__(self, inner):
            self.inner = inner

    x_i32 = qd.ndarray(qd.i32, shape=(4,))
    x_f32 = qd.ndarray(qd.f32, shape=(4,))
    outer = Outer(inner=Inner(x=x_i32))

    @qd.kernel
    def run_i32(s: qd.template()):
        for i in range(4):
            s.inner.x[i] = i + 1

    @qd.kernel
    def run_f32(s: qd.template()):
        for i in range(4):
            s.inner.x[i] = float(i) + 0.5

    run_i32(outer)
    np.testing.assert_array_equal(x_i32.to_numpy(), np.arange(1, 5))

    outer.inner.x = x_f32
    run_f32(outer)
    np.testing.assert_array_equal(x_f32.to_numpy(), np.arange(4, dtype=np.float32) + 0.5)


# ---------------------------------------------------------------------------
# 20. No spec-key regression for data_oriented containers WITHOUT ndarrays. The Gap A fix prepends ndarray
# descriptors only when ndarrays are present; otherwise the original ``weakref.ref(arg)`` spec key is preserved (one
# spec per instance). This test pins the no-ndarray case.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 21. Typed-dataclass kernel arg with a ``@qd.data_oriented`` field type — should error clearly pointing the user
# to ``qd.template()``. The two patterns are incompatible at the kernel-arg layer: dataclass kernel args are
# flattened using annotations, data_oriented containers need a value-driven walk. Pins the helpful error message.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_typed_dataclass_with_data_oriented_field_raises_clear_error():
    @qd.data_oriented
    class Inner:
        def __init__(self, x):
            self.x = x

    @dataclasses.dataclass
    class Outer:
        inner: Inner

    x = qd.ndarray(qd.i32, shape=(4,))
    outer = Outer(inner=Inner(x=x))

    @qd.kernel
    def run(s: Outer):
        for i in range(4):
            s.inner.x[i] = i + 1

    with pytest.raises(Exception, match="data_oriented.*qd.template"):
        run(outer)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_field_only_no_speckey_change():
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, f):
            self.f = f

    f = qd.field(qd.i32, shape=(N,))
    state = State(f=f)

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.f[i] = i + 1

    run(state)
    np.testing.assert_array_equal(f.to_numpy(), np.arange(1, N + 1))

    # Run a second time on the same instance — should reuse the same compiled kernel.
    run(state)


# ---------------------------------------------------------------------------
# 22. @qd.data_oriented holding a qd.Tensor wrapper around an ndarray.
#
# Both ``_build_struct_nd_paths`` and ``_collect_struct_nd_descriptors`` in
# ``_template_mapper_hotpath.py`` have a ``if type(v) in _TENSOR_WRAPPER_TYPES: v = v._unwrap()`` branch that the rest
# of the file doesn't exercise (every other test attaches a bare ``qd.ndarray``). This test covers that unwrap path
# for the ndarray-backed wrapper: the struct-walker should treat ``state.a`` as if it were a bare ndarray (paths
# cached on the class, shape descriptors collected from the unwrapped impl).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_ndarray_wrapper():
    N = 6

    @qd.data_oriented
    class State:
        def __init__(self, a):
            self.a = a

    a = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    state = State(a=a)

    @qd.kernel
    def run(s: qd.Template):
        for i in range(N):
            s.a[i] = i + 1

    run(state)
    np.testing.assert_array_equal(a.to_numpy(), np.arange(1, N + 1))

    run(state)
