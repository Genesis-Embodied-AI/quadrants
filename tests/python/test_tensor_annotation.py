"""Tests for ``qd.Tensor``: value-dispatch kernel-argument annotation.

A kernel parameter annotated with ``qd.Tensor`` accepts either a Field (handled like ``qd.template()``) or an Ndarray
(handled like ``qd.types.ndarray()``). The same kernel object compiles distinct cache entries for each branch.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

# ----------------------------------------------------------------------------
# Class identity
# ----------------------------------------------------------------------------


def test_tensor_is_a_class():
    """As of stork-19, ``qd.Tensor`` is the wrapper *class* (not a Template singleton). Used both as kernel-arg
    annotation and as a constructor: ``qd.Tensor(impl)`` produces a wrapper. The annotation branch in
    ``_func_base.py`` recognises ``annotation is qd.Tensor`` explicitly.
    """
    assert isinstance(qd.Tensor, type)


@test_utils.test(arch=qd.cpu)
def test_tensor_factory_returns_wrapper():
    """Post stork-19, ``qd.tensor(...)`` returns ``qd.Tensor`` instances."""
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    b = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.FIELD)
    assert isinstance(a, qd.Tensor)
    assert isinstance(b, qd.Tensor)


@test_utils.test(arch=qd.cpu)
def test_tensor_double_wrap_rejected():
    """``qd.Tensor`` requires an Ndarray or Field impl; rejects wrapping another wrapper to avoid silent identity
    confusion."""
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    with pytest.raises(TypeError):
        qd.Tensor(a)


# ----------------------------------------------------------------------------
# Ndarray branch
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_ndarray():
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(x.shape[0]):
            x[i] = i * 10

    fill(a)
    np.testing.assert_array_equal(a.to_numpy(), [0, 10, 20, 30])


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_ndarray_with_layout():
    """Layout-tagged ndarrays must dispatch correctly through qd.Tensor."""
    M, N = 3, 4
    a = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY, layout=(1, 0))

    @qd.kernel
    def fill(x: qd.Tensor):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    fill(a)
    arr = a.to_numpy()
    # to_numpy() returns the canonical view on layout-tagged ndarrays.
    assert arr.shape == (M, N)
    assert arr[2, 3] == 203


# ----------------------------------------------------------------------------
# Field branch
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_field():
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.FIELD)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i] = i * 10

    fill(a)
    np.testing.assert_array_equal(a.to_numpy(), [0, 10, 20, 30])


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_raw_field():
    """A kernel parameter annotated ``qd.Tensor`` also accepts a raw ``qd.field()`` (not wrapped in ``qd.Tensor``).
    The raw Field falls through to the template path internally."""
    a = qd.field(qd.i32, shape=(4,))

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i] = i * 10

    fill(a)
    np.testing.assert_array_equal(a.to_numpy(), [0, 10, 20, 30])


# ----------------------------------------------------------------------------
# Cross-call dispatch: same kernel object, both backends, separate cache entries
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_dispatch_same_kernel_both_backends():
    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i] = i + 1

    a_field = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.FIELD)
    a_nd = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)

    fill(a_field)
    fill(a_nd)

    np.testing.assert_array_equal(a_field.to_numpy(), [1, 2, 3, 4])
    np.testing.assert_array_equal(a_nd.to_numpy(), [1, 2, 3, 4])

    # Two cache entries (one per backend branch).
    assert len(fill._primal.mapper.mapping) == 2


@test_utils.test(arch=qd.cpu)
def test_tensor_repeat_same_backend_reuses_cache():
    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i] = 7

    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    b = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)

    fill(a)
    fill(b)

    assert len(fill._primal.mapper.mapping) == 1


@test_utils.test(arch=qd.cpu)
def test_tensor_layouts_keep_separate_cache_entries():
    @qd.kernel
    def k(x: qd.Tensor):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i * 10 + j

    a_id = qd.tensor(qd.i32, shape=(2, 3), backend=qd.Backend.NDARRAY)
    a_swap = qd.tensor(qd.i32, shape=(2, 3), backend=qd.Backend.NDARRAY, layout=(1, 0))

    k(a_id)
    assert len(k._primal.mapper.mapping) == 1
    k(a_swap)
    assert len(k._primal.mapper.mapping) == 2


# ----------------------------------------------------------------------------
# Alternating values at a qd.Tensor slot
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("later_kind", ["none", "bare"])
@test_utils.test()
def test_tensor_slot_cached_wrapper_accepts_none_or_bare_impl(later_kind):
    out = qd.ndarray(qd.f32, shape=(4,))
    bias = qd.ndarray(qd.f32, shape=(4,))
    bias.from_numpy(np.full(4, 100.0, dtype=np.float32))

    @qd.kernel
    def add_bias(out: qd.types.NDArray[qd.f32, 1], bias: qd.Tensor):
        for i in range(out.shape[0]):
            if qd.static(bias is not None):
                out[i] = qd.f32(i) + bias[i]
            else:
                out[i] = qd.f32(i)

    bias_wrapper = qd.wrap(bias)
    add_bias(out, bias_wrapper)
    later = None if later_kind == "none" else bias
    add_bias(out, later)

    expected = np.arange(4, dtype=np.float32)
    if later is not None:
        expected += 100.0
    np.testing.assert_array_equal(out.to_numpy(), expected)

    add_bias(out, bias_wrapper)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(4, dtype=np.float32) + 100.0)
    assert len(add_bias._primal.mapper.mapping) == (2 if later is None else 1)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.metal], require=qd.extension.adstack)
def test_tensor_slot_none_then_wrapper_survives_tape_replay():
    bias = qd.ndarray(qd.f32, shape=(4,), needs_grad=True)
    loss = qd.ndarray(qd.f32, shape=(), needs_grad=True)
    bias.from_numpy(np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32))

    @qd.kernel
    def pick_first(loss: qd.types.NDArray[qd.f32, None], bias: qd.Tensor):
        if qd.static(bias is not None):
            loss[None] = bias[0] * 2.0

    bias_wrapper = qd.wrap(bias)

    with qd.ad.Tape(loss=loss):
        pick_first(loss, None)
        pick_first(loss, bias_wrapper)
        pick_first(loss, None)

    np.testing.assert_allclose(loss.to_numpy(), 6.0)
    np.testing.assert_allclose(bias.grad.to_numpy(), [2.0, 0.0, 0.0, 0.0])
    assert len(pick_first._primal.mapper.mapping) == 2


@test_utils.test()
def test_tensor_slot_later_wrapper_uses_impl_identity_cache():
    out = qd.ndarray(qd.f32, shape=(4,))
    bias = qd.ndarray(qd.f32, shape=(4,))
    bias_wrapper = qd.wrap(bias)

    @qd.kernel
    def copy(out: qd.types.NDArray[qd.f32, 1], bias: qd.Tensor):
        for i in range(out.shape[0]):
            out[i] = bias[i]

    copy(out, bias)
    cache_size = len(copy._primal.mapper._mapping_cache)
    copy(out, bias_wrapper)

    assert len(copy._primal.mapper._mapping_cache) == cache_size


@test_utils.test()
def test_none_argument_populates_spec_key_cache():
    """A ``None`` in a ``qd.Tensor`` slot must land in the template mapper's spec-key cache.

    ``TemplateMapper.lookup`` skips ``weakref``-tracking any argument whose type is in ``_primitive_types``.
    ``NoneType`` is in that set, so ``weakref.ref(None)`` - which raises ``TypeError`` - is never attempted and the
    entry is stored. Were ``NoneType`` absent, the ``TypeError`` would drop the entry and every ``None`` launch would
    re-run full spec-key extraction instead of hitting the cache.
    """
    n = 4
    a = qd.ndarray(qd.f32, shape=(n,))
    a.from_numpy(np.arange(n, dtype=np.float32))
    b = qd.ndarray(qd.f32, shape=(n,))
    b.from_numpy(np.full(n, 100.0, dtype=np.float32))
    c = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def k(a: qd.types.NDArray[qd.f32, 1], b: qd.Tensor, c: qd.types.NDArray[qd.f32, 1]):
        for i in range(a.shape[0]):
            if qd.static(b is not None):
                c[i] = a[i] + b[i]
            else:
                c[i] = a[i]

    # A single launch with the optional slot absent must leave a cache entry behind.
    k(a, None, c)
    np.testing.assert_array_equal(c.to_numpy(), np.arange(n, dtype=np.float32))
    assert len(k._primal.mapper._mapping_cache) == 1

    # The present branch still works and specializes separately; the absent-branch entry is retained.
    k(a, b, c)
    np.testing.assert_array_equal(c.to_numpy(), np.arange(n, dtype=np.float32) + 100.0)
    assert len(k._primal.mapper.mapping) == 2
    assert len(k._primal.mapper._mapping_cache) == 2


# Vector / matrix element types: qd.Tensor must dispatch the compound-element tensors built by qd.Vector.tensor /
# qd.Matrix.tensor on both backends.


BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_tensor_dispatch_vector_element(backend):
    """qd.Tensor must accept Vector-element tensors on both backends and let kernels write each component through
    canonical indexing."""
    qd.init(arch=qd.x64)
    a = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i][0] = float(i)
            x[i][1] = float(i) + 0.5
            x[i][2] = float(i) + 0.25

    fill(a)
    arr = a.to_numpy()
    assert arr.shape[0] == 4
    np.testing.assert_allclose(arr[2, 0], 2.0)
    np.testing.assert_allclose(arr[2, 1], 2.5)
    np.testing.assert_allclose(arr[2, 2], 2.25)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_tensor_dispatch_matrix_element(backend):
    """qd.Tensor must accept Matrix-element tensors on both backends."""
    qd.init(arch=qd.x64)
    a = qd.Matrix.tensor(2, 3, qd.f32, shape=(2,), backend=backend)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(2):
            for r in range(2):
                for c in range(3):
                    x[i][r, c] = float(i * 100 + r * 10 + c)

    fill(a)
    arr = a.to_numpy()
    assert arr.shape[0] == 2
    np.testing.assert_allclose(arr[1, 1, 2], 112.0)
    np.testing.assert_allclose(arr[0, 0, 0], 0.0)


# ----------------------------------------------------------------------------
# Public API surface
# ----------------------------------------------------------------------------


def test_tensor_is_in_qd_namespace():
    assert qd.Tensor is not None
    # also reachable via direct import
    from quadrants._tensor import Tensor as direct

    assert direct is qd.Tensor


# ----------------------------------------------------------------------------
# Module-scope kernel decl with qd.Tensor annotation.
#
# This is the *Genesis* pattern: every Genesis kernel is a module-level global, decorated with ``@qd.kernel`` at import
# time — long before ``qd.init()`` runs and long before any tensor is allocated. The tests above all decorate inside
# the test body (after ``@test_utils.test`` has called ``qd.init()``), so they don't exercise this code path.
#
# Pinning here that:
# - The decorator is happy with ``qd.Tensor`` evaluated at module load time (i.e. before any qd.init).
# - First call lazily compiles for whatever backend / layout the arg actually has.
# - The four (backend × layout) combinations called against the *same* module-level kernel object produce four distinct
#   cache entries with no fragmentation, and each writes the right canonical values.
# Runs on whatever archs the test runner targets (no ``arch=`` filter) so cpu and gpu codegen are both covered.
# ----------------------------------------------------------------------------


_MOD_M, _MOD_N = 3, 4
_MOD_LAYOUTS = [(0, 1), (1, 0)]
_MOD_LAYOUT_IDS = ["identity", "transposed"]


@qd.kernel
def _module_level_fill_2d(x: qd.Tensor):
    # Canonical indexing on both axes; the AST rewrite (ndarray) / SNode order (field) handles non-identity layouts so
    # this same kernel body is correct under any permutation.
    for i, j in qd.ndrange(_MOD_M, _MOD_N):
        x[i, j] = i * 100 + j


def _expected_canonical():
    out = np.zeros((_MOD_M, _MOD_N), dtype=np.int32)
    for i in range(_MOD_M):
        for j in range(_MOD_N):
            out[i, j] = i * 100 + j
    return out


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _MOD_LAYOUTS, ids=_MOD_LAYOUT_IDS)
@test_utils.test()
def test_module_level_qd_tensor_kernel(backend, layout):
    a = qd.tensor(qd.i32, shape=(_MOD_M, _MOD_N), backend=backend, layout=layout)
    _module_level_fill_2d(a)
    np.testing.assert_array_equal(a.to_numpy(), _expected_canonical())


@test_utils.test()
def test_module_level_qd_tensor_kernel_all_combos_share_decl():
    """The same module-level kernel object, called against all four (backend × layout) combos, must produce correct
    canonical values *and* not fragment the JIT cache beyond what each backend genuinely needs.

    Cache-entry expectations (per backend):

    - **Ndarray**: layout is fused into the kernel body at compile time via the AST rewrite, so each layout needs its
      own compiled entry (2 entries for 2 layouts).
    - **Field**: layout is encoded in the SNode order and dispatched at runtime, so the same compiled kernel handles
      every layout (1 entry covers both layouts).

    Total expected: 3 entries for 4 combos. Anything higher means the wrapper-unwrap hook is leaking wrapper identity
    into the cache key and re-fragmenting per call. Anything lower on the ndarray side would mean two layouts collided
    onto one compiled body, which the value asserts above already catch independently.

    Mirrors the Genesis pattern after the stork-20 ``set_gravity`` collapse: one decl, multiple backend/layout
    instances at runtime.
    """
    expected = _expected_canonical()
    n_before = len(_module_level_fill_2d._primal.mapper.mapping)

    tensors = []
    for backend in BACKENDS:
        for layout in _MOD_LAYOUTS:
            t = qd.tensor(qd.i32, shape=(_MOD_M, _MOD_N), backend=backend, layout=layout)
            _module_level_fill_2d(t)
            tensors.append((backend, layout, t))

    for backend, layout, t in tensors:
        np.testing.assert_array_equal(
            t.to_numpy(),
            expected,
            err_msg=f"backend={backend} layout={layout}",
        )

    n_after = len(_module_level_fill_2d._primal.mapper.mapping)
    added = n_after - n_before
    # 2 (ndarray, one per layout) + 1 (field, layout-agnostic at the cache-key layer) = 3. We allow [2, 3] because a
    # future field change that splits per-layout would still be acceptable; what we really want to forbid is unbounded
    # growth (>4) from wrapper identity leaking into the key.
    assert 2 <= added <= 4, (
        f"unexpected cache growth: {added} new entries for 4 (backend, " f"layout) combos (want 3, accept 2-4)"
    )


# ----------------------------------------------------------------------------
# qd.Tensor across a qd.reset() / qd.init() cycle.
#
# Pattern: define a kernel, init quadrants, allocate a Field-backed tensor, run the kernel, then *destroy* the runtime
# via ``qd.reset()`` and *re-init* it before allocating an Ndarray-backed tensor and re-running the same kernel. Both
# halves write the right canonical values.
#
# This exercises the path where a single kernel-decl object survives a runtime teardown (its compiled cache is
# invalidated by reset) and is then re-driven against a different backend on the fresh runtime.
# ----------------------------------------------------------------------------


@test_utils.test()
def test_qd_tensor_across_reset_and_reinit(req_arch, req_options):
    M, N = 3, 4

    @qd.kernel
    def fill_2d(x: qd.Tensor):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    expected = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            expected[i, j] = i * 100 + j

    # Phase 1: field backend on the runtime the autouse fixture spun up.
    a_field = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.FIELD)
    fill_2d(a_field)
    np.testing.assert_array_equal(a_field.to_numpy(), expected)

    # Tear the runtime down. After this any tensor allocated against the old runtime is dead; the kernel decl object
    # survives but its compiled cache is cleared.
    qd.reset()

    # Bring it back on the same arch the fixture chose. Mirror the fixture's option handling so cuda/x64 both behave
    # like the rest of the suite.
    init_options = dict(req_options or {})
    init_options.setdefault("print_full_traceback", True)
    qd.init(arch=req_arch, enable_fallback=False, **init_options)

    # Phase 2: ndarray backend on the fresh runtime, same kernel object.
    a_nd = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY)
    fill_2d(a_nd)
    np.testing.assert_array_equal(a_nd.to_numpy(), expected)
    # The autouse fixture's teardown will reset() again on the way out.


# ----------------------------------------------------------------------------
# External arrays (numpy / torch) at a qd.Tensor slot.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_numpy():
    out = qd.ndarray(qd.f32, shape=(4,))

    @qd.kernel
    def copy(out: qd.types.NDArray[qd.f32, 1], t: qd.Tensor):
        for i in range(out.shape[0]):
            out[i] = t[i]

    copy(out, np.array([5, 6, 7, 8], dtype=np.float32))
    np.testing.assert_array_equal(out.to_numpy(), [5, 6, 7, 8])


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_torch():
    torch = pytest.importorskip("torch")
    out = qd.ndarray(qd.f32, shape=(4,))

    @qd.kernel
    def copy(out: qd.types.NDArray[qd.f32, 1], t: qd.Tensor):
        for i in range(out.shape[0]):
            out[i] = t[i]

    copy(out, torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32))
    np.testing.assert_array_equal(out.to_numpy(), [5, 6, 7, 8])


@test_utils.test(arch=qd.cpu)
def test_tensor_numpy_collapses_to_one_spec():
    """Two different numpy arrays of the same dtype/ndim share one specialization (not one per instance), and numpy
    no longer raises ``unhashable type: 'numpy.ndarray'``."""
    out = qd.ndarray(qd.f32, shape=(4,))

    @qd.kernel
    def copy(out: qd.types.NDArray[qd.f32, 1], t: qd.Tensor):
        for i in range(out.shape[0]):
            out[i] = t[i]

    copy(out, np.array([1, 2, 3, 4], dtype=np.float32))
    copy(out, np.array([5, 6, 7, 8], dtype=np.float32))
    np.testing.assert_array_equal(out.to_numpy(), [5, 6, 7, 8])
    assert len(copy._primal.mapper.mapping) == 1


@test_utils.test(arch=qd.cpu)
def test_tensor_torch_collapses_to_one_spec():
    torch = pytest.importorskip("torch")
    out = qd.ndarray(qd.f32, shape=(4,))

    @qd.kernel
    def copy(out: qd.types.NDArray[qd.f32, 1], t: qd.Tensor):
        for i in range(out.shape[0]):
            out[i] = t[i]

    copy(out, torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))
    copy(out, torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32))
    np.testing.assert_array_equal(out.to_numpy(), [5, 6, 7, 8])
    assert len(copy._primal.mapper.mapping) == 1


@test_utils.test(arch=qd.cpu)
def test_tensor_external_array_does_not_reroute_field():
    """Widening qd.Tensor to external arrays must not divert Field (which also exposes ``.shape`` / ``.dtype``) off
    the template path. A field and a numpy array through the same slot produce two distinct specializations, and both
    yield correct values."""
    out = qd.ndarray(qd.f32, shape=(4,))
    f = qd.field(qd.f32, shape=(4,))
    for i in range(4):
        f[i] = 11.0

    @qd.kernel
    def copy(out: qd.types.NDArray[qd.f32, 1], t: qd.Tensor):
        for i in range(out.shape[0]):
            out[i] = t[i]

    copy(out, f)
    np.testing.assert_array_equal(out.to_numpy(), [11, 11, 11, 11])
    copy(out, np.array([5, 6, 7, 8], dtype=np.float32))
    np.testing.assert_array_equal(out.to_numpy(), [5, 6, 7, 8])
    # field -> template path (field marker), numpy -> ndarray path: two distinct specializations.
    assert len(copy._primal.mapper.mapping) == 2
