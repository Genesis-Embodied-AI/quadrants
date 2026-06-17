"""Tests for ``@qd.data_oriented`` classes whose members are raw ``qd.ndarray`` (not ``qd.field``, not
``qd.Tensor`` wrappers).

The user-guide doc ``docs/source/user_guide/compound_types.md`` claims this pattern is not supported ("can contain
ndarray? no" for ``@qd.data_oriented``). But the in-tree error message in ``python/quadrants/lang/impl.py`` lists
``@qd.data_oriented / frozen-dataclass template`` as a *supported* route, and the ndarray-in-struct infrastructure
added by ``#561 [Type] Tensor 24`` (2026-04-28) — specifically ``_predeclare_struct_ndarrays`` in
``python/quadrants/lang/ast/ast_transformers/function_def_transformer.py`` — explicitly walks both
``dataclasses.is_dataclass(val)`` and ``hasattr(val, "__dict__")`` containers, the latter being the data_oriented
case.

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
#    stale-cache guard (``_mutable_nd_cached_val`` in kernel.py) is supposed to fold the live ndarray id into
#    args_hash so the launch context is not served stale. We pin that behaviour here for the data_oriented case.
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
#    specialisation key (in ``_template_mapper_hotpath._extract_arg``) returns ``weakref.ref(arg)`` for
#    ``is_data_oriented(arg)``; it does NOT descend into ndarray children to compute a dtype/ndim-dependent spec key.
#    So if the data_oriented instance's id is unchanged but its ndarray attribute is reassigned to a different dtype,
#    we expect either:
#      - a graceful recompile/raise, or
#      - silent miscompilation (the bug case — current expected outcome per static analysis).
#    Mark xfail with strict=False so we record the actual outcome without breaking CI.
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
#    recompile (because the spec key is ``weakref.ref(arg)`` identity). This is a perf concern, not a correctness
#    one. We assert correctness here; the recompile count is documented as a perf note.
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
#    ``@qd.data_oriented`` objects are supported in the cache key. We don't assert cross-process here (that requires
#    a fresh interpreter); we assert that ``cache_stored`` becomes True on the first call and
#    ``cache_key_generated`` is True (i.e. no PARAM_INVALID fallthrough due to the ndarray member).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9b. Fastcache end-to-end with ``@qd.data_oriented`` holding ndarrays. Pattern adapted from
#     ``test_cache.test_fastcache``: call ``qd_init_same_arch`` twice with the same cache directory to simulate two
#     processes, monkeypatch ``launch_kernel`` to capture whether ``compiled_kernel_data`` was loaded from disk. On
#     the second init the data_oriented + ndarray kernel should be served from the on-disk fastcache.
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
#     recursion handles nested data_oriented containers correctly across processes.
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
#     two distinct disk cache entries. Pins the args_hasher's ``[nd-{dtype}-{ndim}{layout}]`` repr.
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
#     (args_hasher returns None for ScalarField). The kernel still runs correctly via non-fastcache compilation. This
#     test pins the documented fallback so a future "support fields in fastcache" change explicitly chooses to update
#     this test.
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
#     run, mirroring the existing ``test_pure_validation_data_oriented_as_param`` test which only covers ``qd.field``.
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
#     works; if this fails, the test environment itself is broken, not the data_oriented path).
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
# 12. data_oriented holding a (frozen) dataclass that holds an ndarray. Exercises the ``else`` branch of ``_walk_obj``
#     recursing through a dataclass child — added by the Bug 1 fix.
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
#     dataclass branch of ``_walk_obj`` recursing through a data_oriented child — added by the Bug 1 fix. The outer
#     dataclass must be frozen because (i) non-frozen dataclasses are unhashable in Python (``__hash__ is None``) and
#     the template-mapper key tuple needs the value to be hashable, and (ii) the typed-dataclass-arg form (``def
#     run(s: Outer):``) goes through ``_transform_kernel_arg`` which does not currently recurse on data_oriented
#     field *types* (as opposed to values) — that's a separate follow-up.
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
#     guard fires even when the ndarray lives several attribute hops deep.
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
#     independent (each kernel's compile sets up its own pre-declared ndarray args).
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
#     (which uses the predeclared AnyArray cache) works when the access happens inside a func, not just the top-level
#     kernel.
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
# 18. Reassign ndarray to a *different ndim* on the same data_oriented instance. Complementary to test 7 (different-
#     dtype). Spec key must change so a 2D-specialised kernel is not reused for a 1D ndarray. Pins the Gap A fix from
#     the dtype side.
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
#     ``_collect_struct_nd_descriptors`` reaches through nested data_oriented.
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
# 20. No spec-key regression for data_oriented containers WITHOUT ndarrays. The Gap A fix prepends ndarray descriptors
#     only when ndarrays are present; otherwise the original ``weakref.ref(arg)`` spec key is preserved (one spec per
#     instance). This test pins the no-ndarray case.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 21. Typed-dataclass kernel arg with a ``@qd.data_oriented`` field type — should error clearly pointing the user to
#     ``qd.template()``. The two patterns are incompatible at the kernel-arg layer: dataclass kernel args are
#     flattened using annotations, data_oriented containers need a value-driven walk. Pins the helpful error message.
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
# 22. Robustness: object graphs with Pydantic-style metaclass ``__getattr__`` recursion, and cyclic attribute
#     references. Real-world container classes (notably Genesis's ``RigidOptions`` / ``SimOptions``) inherit from
#     ``pydantic.BaseModel`` whose ``ModelMetaclass.__getattr__`` recurses infinitely on missing class attributes.
#     Quadrants' walker must not blow the stack when it traverses a ``data_oriented`` arg that contains such an
#     object, or that contains a back-reference to itself / its parent (e.g. ``solver.scene.solver``).
# ---------------------------------------------------------------------------


def test_is_data_oriented_safe_on_pydantic_like_metaclass():
    """``is_data_oriented`` must not invoke ``__getattr__`` on the class (or metaclass), so it stays safe in the
    presence of pathological metaclasses whose ``__getattr__`` blows the Python recursion limit on arbitrary
    attribute lookups (e.g. Pydantic's ``ModelMetaclass`` when probed for a name not in its private-attrs cache).
    """

    from quadrants.lang.util import is_data_oriented

    class RecursingMeta(type):
        def __getattr__(cls, item):
            return cls.__getattr__(item)

    class Pathological(metaclass=RecursingMeta):
        pass

    # Pre-fix this raised RecursionError; with the MRO+__dict__ lookup it just returns False.
    assert is_data_oriented(Pathological()) is False


@test_utils.test(arch=qd.cpu)
def test_data_oriented_with_pydantic_like_child():
    """A ``@qd.data_oriented`` class holding a child whose metaclass has the recursing ``__getattr__``
    (Pydantic-style). Walker must classify the child as non-data-oriented and continue without blowing the stack.
    """
    N = 4

    class RecursingMeta(type):
        def __getattr__(cls, item):
            return cls.__getattr__(item)

    class Options(metaclass=RecursingMeta):
        pass

    @qd.data_oriented
    class State:
        def __init__(self, x, opts):
            self.x = x
            self.opts = opts

    x = qd.ndarray(qd.i32, shape=(N,))
    state = State(x=x, opts=Options())

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i + 1

    run(state)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(1, N + 1))


@test_utils.test(arch=qd.cpu)
def test_data_oriented_polymorphic_attr_across_instances():
    """Some real-world ``@qd.data_oriented`` containers (Genesis FEMSolver / MPMSolver / SPHSolver, etc.) hold
    polymorphic children whose types differ between instances — e.g. ``self.material.x`` is an ``Ndarray`` on
    instance A and a ``qd.field`` (``MatrixField``) on instance B. The per-instance path cache walks each instance
    fresh, but ``_collect_struct_nd_descriptors`` must additionally tolerate a path's leaf no longer being an
    ``Ndarray`` *within a single instance's lifetime* (e.g. ``qd.Tensor`` impl swap), and silently skip the stale
    entry rather than crash on ``v.element_type``."""
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x

    # First instance: ``self.x`` is an Ndarray. The walker emits path ``('x',)`` and caches it.
    x_nd = qd.ndarray(qd.i32, shape=(N,))
    state_a = State(x=x_nd)

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i + 1

    run(state_a)
    np.testing.assert_array_equal(x_nd.to_numpy(), np.arange(1, N + 1))

    # Second instance of the SAME class, ``self.x`` is now a ``qd.field`` (MatrixField via Vector.field).
    # The cached path ``('x',)`` from instance A points to a non-Ndarray on this instance — the descriptor
    # walk must skip it cleanly rather than crash on ``v.element_type``.
    f = qd.Vector.field(2, qd.i32, shape=(N,))
    state_b = State(x=f)

    @qd.kernel
    def run_field(s: qd.template()):
        for i in range(N):
            s.x[i] = [i, i + 1]

    run_field(state_b)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_polymorphic_attribute_set_across_instances():
    """Models the Genesis ``DataManager`` failure mode: a ``@qd.data_oriented`` class whose ``__init__`` conditionally
    allocates attributes based on a construction flag. Different instances of the same class then have different
    attribute *sets* (not just different value types at the same paths).

    With a per-class path cache populated from the first instance walked, this would either AttributeError when the
    second instance lacks an attribute the first had (forward direction) or silently miss an ndarray the second
    instance has but the first didn't (inverse direction). Per-instance caching walks each instance fresh so both
    directions work."""
    N = 4

    @qd.data_oriented
    class PolyState:
        def __init__(self, with_extra: bool):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            if with_extra:
                self.extra = qd.ndarray(qd.i32, shape=(N,))

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i + 1

    # Forward direction: first instance has 'extra', second doesn't. Used to AttributeError on the cached
    # ('extra',) path when running with state_lean.
    state_full = PolyState(with_extra=True)
    run(state_full)
    state_lean = PolyState(with_extra=False)
    run(state_lean)
    np.testing.assert_array_equal(state_lean.x.to_numpy(), np.arange(1, N + 1))

    # Inverse direction: a different class so per-class cache (if used by __slots__ fallback) starts fresh; first
    # instance lacks 'extra', second has it. The kernel actually *reads* ``s.extra`` so the inverse-direction
    # silent miscache (which only manifests when the kernel touches the conditional attr) is exercised end-to-end.
    @qd.data_oriented
    class PolyState2:
        def __init__(self, with_extra: bool):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            if with_extra:
                self.extra = qd.ndarray(qd.i32, shape=(N,))

    @qd.kernel
    def run_using_extra(s: qd.template()):
        for i in range(N):
            s.x[i] = s.extra[i] * 10

    # Walk the lean instance first (no 'extra'), populating any per-class state with the *narrow* attribute set.
    # With the old per-class cache, this would lock in paths = [('x',)] for the class — and the next instance's
    # ``extra`` would be silently absent from args_hash and from the kernel spec, leading to a wrong-shape kernel
    # or a stale-cache hit when ``extra`` is later reassigned.
    state_lean2 = PolyState2(with_extra=False)
    run(state_lean2)
    np.testing.assert_array_equal(state_lean2.x.to_numpy(), np.arange(1, N + 1))

    # Now the polymorphic-attr-bearing instance. The per-instance walk must include ``('extra',)`` so that
    # ``state_full2.extra``'s shape/id participates in the spec and the kernel compiles correctly.
    state_full2 = PolyState2(with_extra=True)
    state_full2.extra.from_numpy(np.array([2, 3, 5, 7], dtype=np.int32))
    run_using_extra(state_full2)
    np.testing.assert_array_equal(state_full2.x.to_numpy(), np.array([20, 30, 50, 70], dtype=np.int32))

    # Reassignment-detection check: swap ``state_full2.extra`` to a different ndarray. The per-instance walk caches
    # the *path list* ([('x',), ('extra',)]) on the instance, but the per-call args_hash still folds in
    # ``id(getattr(state_full2, 'extra'))`` — so a swap should miss the spec-key cache and re-specialise.
    state_full2.extra = qd.ndarray(qd.i32, shape=(N,))
    state_full2.extra.from_numpy(np.array([11, 13, 17, 19], dtype=np.int32))
    run_using_extra(state_full2)
    np.testing.assert_array_equal(state_full2.x.to_numpy(), np.array([110, 130, 170, 190], dtype=np.int32))


@test_utils.test(arch=qd.cpu)
def test_data_oriented_with_cyclic_attr_graph():
    """A ``@qd.data_oriented`` class whose attribute graph contains a cycle (``parent.child.parent is parent``).
    Walker must not re-enter the cycle."""
    N = 4

    @qd.data_oriented
    class Child:
        def __init__(self):
            self.parent = None

    @qd.data_oriented
    class Parent:
        def __init__(self, x):
            self.x = x
            self.child = Child()
            self.child.parent = self  # cycle

    x = qd.ndarray(qd.i32, shape=(N,))
    p = Parent(x=x)

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.x[i] = i + 10

    run(p)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(10, 10 + N))


# ---------------------------------------------------------------------------
# Pruning-driven fastcache behaviour for @qd.data_oriented containers.
#
# These pin the three rules enforced by the args hasher (see fastcache.md "Pruning-driven argument hashing"):
#   1. The cache key may only include contributions from kernel-pruned paths.
#   2. Unrecognised types at kernel-read paths must not be silently dropped.
#   3. Fastcache works for @qd.data_oriented kernel args end-to-end.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_data_oriented_kernel_unused_opaque_member_does_not_affect_cache(tmp_path, monkeypatch):
    """Rule 1: kernel-unused opaque members do not affect the fastcache key.

    Two ``State`` instances differ only in an opaque ``uuid`` member that the kernel never reads. Both must hit the
    same compiled artifact on the second process — proof that the args hasher's pruning narrow walk skips the opaque
    attribute (no qualname-fallback, no spurious miss)."""
    import uuid

    from quadrants._test_tools import qd_init_same_arch

    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel
    captured = []

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=None):
        if self.func.__name__ == "run":
            captured.append(compiled_kernel_data)
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=qd_stream)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.data_oriented
    class State:
        def __init__(self, x):
            self.x = x
            self.uuid = uuid.uuid4()  # opaque member, kernel does not read it

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        for i in range(4):
            s.x[i] = s.x[i] + 1

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(x=qd.ndarray(qd.i32, shape=(4,)))
    b = State(x=qd.ndarray(qd.i32, shape=(4,)))
    run(a)
    run(b)

    # Second process: cold-start, must load from disk. If the uuid had leaked into the cache key, different uuid →
    # different L2 key → no artifact would load.
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(x=qd.ndarray(qd.i32, shape=(4,)))
    b = State(x=qd.ndarray(qd.i32, shape=(4,)))
    run(a)
    run(b)
    assert captured[-2] is not None, "first instance should load from disk"
    assert captured[-1] is not None, "second instance (different uuid) should ALSO load from disk"
    assert run._primal.src_ll_cache_observations.cache_loaded


@test_utils.test(arch=qd.cpu)
def test_data_oriented_kernel_read_opaque_member_fails_fastcache(tmp_path, capfd) -> None:
    """Rule 2: when the kernel actually reads an unrecognised-type member, fastcache fails loudly with [UNKNOWN_TYPE]
    + [INVALID_FUNC] — no silent drop, no qualname fallback. The kernel still runs via normal compilation."""
    from quadrants._test_tools import qd_init_same_arch
    from quadrants.lang._fast_caching.args_hasher import reset_unknown_type_warn_state

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    reset_unknown_type_warn_state()

    class CustomConfig:
        def __init__(self, scale: int) -> None:
            self.scale = scale

    @qd.data_oriented
    class State:
        def __init__(self, x, cfg):
            self.x = x
            self.cfg = cfg

    x = qd.ndarray(qd.i32, shape=(4,))
    state = State(x=x, cfg=CustomConfig(scale=3))

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        scale = s.cfg.scale  # makes ``__qd_s__qd_cfg`` and ``__qd_s__qd_cfg__qd_scale`` live
        for i in range(4):
            s.x[i] = i * scale

    run(state)
    _out, err = capfd.readouterr()
    np.testing.assert_array_equal(x.to_numpy(), np.arange(4) * 3)

    obs = run._primal.src_ll_cache_observations
    assert obs.cache_key_generated is False, "unrecognised type at kernel-read path must disable fastcache"
    assert "[FASTCACHE][UNKNOWN_TYPE]" in err
    assert CustomConfig.__name__ in err
    assert "[FASTCACHE][INVALID_FUNC]" in err


@test_utils.test(arch=qd.cpu)
def test_data_oriented_kernel_read_primitive_distinguishes_cache_key(tmp_path, monkeypatch) -> None:
    """Rule 3 (data_oriented works) + pruning correctness: when the kernel reads a primitive member, its value is
    baked into the kernel and must drive a distinct cache entry per value. Two State instances differing only in
    ``n`` (read by the kernel) cold-compile separately and both load from disk on the second process."""
    from quadrants._test_tools import qd_init_same_arch

    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel
    captured = []

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=None):
        if self.func.__name__ == "run":
            captured.append(compiled_kernel_data)
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=qd_stream)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.data_oriented
    class State:
        def __init__(self, x, n):
            self.x = x
            self.n = n  # primitive, baked into kernel via ``for i in range(s.n)``

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        for i in range(s.n):
            s.x[i] = i + s.n

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(x=qd.ndarray(qd.i32, shape=(4,)), n=2)
    b = State(x=qd.ndarray(qd.i32, shape=(4,)), n=3)
    run(a)
    run(b)
    assert captured[-2] is None and captured[-1] is None, "different ``n`` → both cold-compile"

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(x=qd.ndarray(qd.i32, shape=(4,)), n=2)
    b = State(x=qd.ndarray(qd.i32, shape=(4,)), n=3)
    run(a)
    run(b)
    assert captured[-2] is not None and captured[-1] is not None, "both ``n`` values should load distinct artifacts"
    np.testing.assert_array_equal(a.x.to_numpy()[:2], np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(b.x.to_numpy()[:3], np.array([3, 4, 5], dtype=np.int32))


@test_utils.test(arch=qd.cpu)
def test_data_oriented_kernel_unread_primitive_does_not_affect_cache(tmp_path, monkeypatch) -> None:
    """Rule 1: kernel-unused primitive members do not affect the cache key. Mirror of the opaque case for
    primitives. Two State instances differing only in ``unused_n`` must share the cache."""
    from quadrants._test_tools import qd_init_same_arch

    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel
    captured = []

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=None):
        if self.func.__name__ == "run":
            captured.append(compiled_kernel_data)
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=qd_stream)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.data_oriented
    class State:
        def __init__(self, x, unused_n):
            self.x = x
            self.unused_n = unused_n  # kernel never reads this

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        for i in range(4):
            s.x[i] = s.x[i] + 1

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(x=qd.ndarray(qd.i32, shape=(4,)), unused_n=2)
    b = State(x=qd.ndarray(qd.i32, shape=(4,)), unused_n=99)
    run(a)
    run(b)

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(x=qd.ndarray(qd.i32, shape=(4,)), unused_n=2)
    b = State(x=qd.ndarray(qd.i32, shape=(4,)), unused_n=99)
    run(a)
    run(b)
    assert captured[-2] is not None, "first instance should load from disk"
    assert captured[-1] is not None, "second instance (different unused_n) should ALSO load from disk"


@test_utils.test(arch=qd.cpu)
def test_data_oriented_qd_func_chain_propagation_distinguishes_cache_key(tmp_path, monkeypatch) -> None:
    """Pruning chain propagation through ``@qd.func`` calls (``record_after_call`` extension): when the kernel calls
    ``f(self.dofs)`` and ``f`` reads ``s.x``, the kernel's pruning set must include ``__qd_self__qd_dofs__qd_x`` so
    that changes to the inner ndarray's dtype invalidate the cache. Two States differing in ``dofs.x``'s dtype must
    cold-compile separately."""
    from quadrants._test_tools import qd_init_same_arch

    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel
    captured = []

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=None):
        if self.func.__name__ == "run":
            captured.append(compiled_kernel_data)
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=qd_stream)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.data_oriented
    class Dofs:
        def __init__(self, x):
            self.x = x

    @qd.data_oriented
    class State:
        def __init__(self, dofs):
            self.dofs = dofs

    @qd.func
    def write_dofs(d: qd.template(), v: qd.i32):
        d.x[0] = v

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        write_dofs(s.dofs, 7)

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(dofs=Dofs(x=qd.ndarray(qd.i32, shape=(4,))))
    b = State(dofs=Dofs(x=qd.ndarray(qd.f32, shape=(4,))))
    run(a)
    run(b)
    assert captured[-2] is None and captured[-1] is None, "differing dofs.x dtype → both cold-compile"

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(dofs=Dofs(x=qd.ndarray(qd.i32, shape=(4,))))
    b = State(dofs=Dofs(x=qd.ndarray(qd.f32, shape=(4,))))
    run(a)
    run(b)
    assert captured[-2] is not None and captured[-1] is not None, "both dtypes load distinct artifacts"


@test_utils.test(arch=qd.cpu)
def test_data_oriented_nested_primitive_via_qd_func_distinguishes_cache_key(tmp_path, monkeypatch) -> None:
    """Pruning chain propagation through ``f(self.child)`` for *primitive* members of nested data_oriented containers.

    Regression test for a bug where ``record_after_call`` skipped chain-path propagation whenever the caller-side arg
    flattened to a ``__qd_*``-prefixed name (which Attribute chains always do — ``self.cfg`` →
    ``__qd_self__qd_cfg``). When that happened, primitive members read inside the callee (``cfg.n`` →
    ``__qd_cfg__qd_n`` in the callee's chain set) never made it into the kernel's pruning set, so the args-hasher
    walked ``self.cfg`` as data_oriented and found no pruned children, yielding an identical hash for *any* value of
    ``cfg.n``. Two configs that should produce different kernels (different ``range(s.cfg.n)`` trip counts baked into
    codegen) would then share a fastcache entry — leading to stale-kernel hits and silent miscompiles (e.g. Genesis'
    ``test_ndarray_no_compile`` was failing with iter-N kernels reused for iter-N+1 scenes that have a different
    ``RigidSimStaticConfig.para_level`` baked into their ``qd.static`` branches).

    The fix in ``_pruning.py`` gates propagation on the *root Name* of the chain (``self``, not the flat result), so
    both ``f(self)`` and ``f(self.cfg)`` propagate, while already-flattened dataclass refs
    (``Name('__qd_state__qd_x')``) are still skipped."""
    from quadrants._test_tools import qd_init_same_arch

    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel
    captured = []

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=None):
        if self.func.__name__ == "run":
            captured.append(compiled_kernel_data)
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args, qd_stream=qd_stream)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.data_oriented
    class Cfg:
        def __init__(self, n):
            self.n = n  # primitive read by ``write_x`` — drives codegen via ``range(c.n)``

    @qd.data_oriented
    class State:
        def __init__(self, x, cfg):
            self.x = x
            self.cfg = cfg

    @qd.func
    def write_x(x: qd.template(), c: qd.template()):
        for i in range(c.n):
            x[i] = i + c.n

    @qd.kernel(fastcache=True)
    def run(s: qd.template()):
        write_x(s.x, s.cfg)

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(x=qd.ndarray(qd.i32, shape=(8,)), cfg=Cfg(n=2))
    b = State(x=qd.ndarray(qd.i32, shape=(8,)), cfg=Cfg(n=3))
    run(a)
    run(b)
    assert captured[-2] is None and captured[-1] is None, "different cfg.n → both cold-compile"
    np.testing.assert_array_equal(a.x.to_numpy()[:2], np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(b.x.to_numpy()[:3], np.array([3, 4, 5], dtype=np.int32))

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = State(x=qd.ndarray(qd.i32, shape=(8,)), cfg=Cfg(n=2))
    b = State(x=qd.ndarray(qd.i32, shape=(8,)), cfg=Cfg(n=3))
    run(a)
    run(b)
    assert captured[-2] is not None and captured[-1] is not None, "both cfg.n values load distinct artifacts"
    np.testing.assert_array_equal(a.x.to_numpy()[:2], np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(b.x.to_numpy()[:3], np.array([3, 4, 5], dtype=np.int32))


# ---------------------------------------------------------------------------
# 23. @qd.data_oriented holding a qd.Tensor wrapper around an ndarray.
#
# Both ``_build_struct_nd_paths`` and ``_collect_struct_nd_descriptors`` in ``_template_mapper_hotpath.py`` have a
# ``if type(v) in _TENSOR_WRAPPER_TYPES: v = v._unwrap()`` branch that the rest of the file doesn't exercise (every
# other test attaches a bare ``qd.ndarray``). This test covers that unwrap path for the ndarray-backed wrapper: the
# struct-walker should treat ``state.a`` as if it were a bare ndarray (paths cached on the class, shape descriptors
# collected from the unwrapped impl).
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
