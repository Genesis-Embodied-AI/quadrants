"""Phase 5 tests: tensor as a *field* inside a dataclass kernel parameter.

Phase 5 extends the Phase 3 AST sugar pass to walk dataclass parameters
and register every nested ``_TensorBase`` field under its dotted path
(e.g. ``('s', 'state')``). The transformer then matches Subscripts whose
value is an attribute chain rooted at any registered path, rewrites them
to ``...underlying[permuted_idx]``, and the existing dataclass-flatten
pass collapses ``s.state.underlying`` into the canonical flat name.

Coverage:
- Dataclass with one tensor field.
- Dataclass with two tensor fields of *different* layouts in one kernel
  (the cross-layout case the genesis migration needs).
- Mix of tensor + non-tensor fields (e.g. an ``i32`` count) in the same
  dataclass.
- Both backends (NdarrayTensor + FieldTensor) used as fields.
- Nested dataclasses (tensor at depth >= 2).
- ``qd.func`` callee that takes the dataclass and uses ``s.state[i, j]``.
- Atomic ops via a nested-field lvalue.
"""

import dataclasses

import numpy as np

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils

# ---------------------------------------------------------------------------
# Single tensor field.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Box:
    state: qd.Tensor


@qd.kernel
def _box_fill(b: _Box, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        b.state[i, j] = qd.f32(i * 10 + j)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_field_in_dataclass_identity_layout():
    box = _Box(state=qd.tensor(qd.f32, shape=(3, 4)))
    _box_fill(box, 3, 4)
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(box.state.to_numpy(), expected)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_field_in_dataclass_transposed_layout():
    box = _Box(state=qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0)))
    _box_fill(box, 3, 4)
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(box.state.to_numpy(), expected)
    np.testing.assert_array_equal(box.state.underlying.to_numpy(), expected.T)


# ---------------------------------------------------------------------------
# Two tensor fields with *different* layouts in one kernel.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _TwoTensors:
    a: qd.Tensor
    b: qd.Tensor


@qd.kernel
def _two_fill(t: _TwoTensors, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        t.a[i, j] = qd.f32(i * 10 + j)
        t.b[i, j] = qd.f32(i * 100 + j)


@test_utils.test(arch=get_host_arch_list())
def test_two_tensor_fields_different_layouts_same_kernel():
    """The cross-layout case the genesis migration needs."""
    a = qd.tensor(qd.f32, shape=(3, 4))
    b = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    pair = _TwoTensors(a=a, b=b)
    _two_fill(pair, 3, 4)
    exp_a = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    exp_b = np.fromfunction(lambda i, j: i * 100 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(a.to_numpy(), exp_a)
    np.testing.assert_array_equal(b.to_numpy(), exp_b)
    np.testing.assert_array_equal(a.underlying.to_numpy(), exp_a)  # identity
    np.testing.assert_array_equal(b.underlying.to_numpy(), exp_b.T)  # transposed


# ---------------------------------------------------------------------------
# Tensor field next to a non-tensor (template) field — pruning interaction.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _StateAndCount:
    state: qd.Tensor
    count: qd.template


@qd.kernel
def _fill_with_count(sc: _StateAndCount, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        sc.state[i, j] = qd.f32(sc.count + i + j)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_field_with_non_tensor_sibling_field():
    sc = _StateAndCount(
        state=qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0)),
        count=7,
    )
    _fill_with_count(sc, 2, 3)
    expected = np.fromfunction(lambda i, j: 7 + i + j, (2, 3), dtype=np.float32)
    np.testing.assert_array_equal(sc.state.to_numpy(), expected)


# ---------------------------------------------------------------------------
# FieldTensor as a dataclass field.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FieldBox:
    state: qd.FieldTensor


@qd.kernel
def _field_box_fill(b: _FieldBox, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        b.state[i, j] = qd.f32(i * 100 + j)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_as_dataclass_field_transposed():
    fb = _FieldBox(state=qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0), backend=qd.Backend.FIELD))
    _field_box_fill(fb, 3, 4)
    expected = np.fromfunction(lambda i, j: i * 100 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(fb.state.to_numpy(), expected)


# ---------------------------------------------------------------------------
# Nested dataclasses (tensor at depth 2).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Inner:
    grid: qd.Tensor


@dataclasses.dataclass
class _Outer:
    inner: _Inner


@qd.kernel
def _nested_fill(o: _Outer, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        o.inner.grid[i, j] = qd.f32(i * 10 + j)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_nested_two_levels_deep():
    o = _Outer(inner=_Inner(grid=qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))))
    _nested_fill(o, 3, 4)
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(o.inner.grid.to_numpy(), expected)


# ---------------------------------------------------------------------------
# qd.func with a dataclass-of-tensor argument.
# ---------------------------------------------------------------------------


@qd.func
def _box_sum(b: _Box, n0: qd.i32, n1: qd.i32) -> qd.f32:
    s = qd.f32(0.0)
    for i, j in qd.ndrange(n0, n1):
        s += b.state[i, j]
    return s


@qd.kernel
def _call_box_sum(b: _Box, n0: qd.i32, n1: qd.i32, out: qd.types.ndarray(qd.f32, 1)):
    out[0] = _box_sum(b, n0, n1)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_field_inside_qd_func_callee():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    box = _Box(state=qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0)))
    box.state.from_numpy(src)
    out = qd.ndarray(qd.f32, shape=(1,))
    _call_box_sum(box, 3, 4, out)
    assert out[0] == src.sum()


# ---------------------------------------------------------------------------
# Atomic add via a nested-field lvalue.
# ---------------------------------------------------------------------------


@qd.kernel
def _box_atomic_scatter(b: _Box, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        qd.atomic_add(b.state[i % 2, j % 3], qd.f32(1.0))


@test_utils.test(arch=get_host_arch_list())
def test_tensor_field_atomic_add_transposed():
    box = _Box(state=qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0)))
    box.state.from_numpy(np.zeros((2, 3), dtype=np.float32))
    _box_atomic_scatter(box, 100, 100)
    expected = np.zeros((2, 3), dtype=np.float32)
    for i in range(100):
        for j in range(100):
            expected[i % 2, j % 3] += 1.0
    np.testing.assert_array_equal(box.state.to_numpy(), expected)
