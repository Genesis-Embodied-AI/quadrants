"""Field-only tests for deferred allocation-time initialization."""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests.helpers import test_utils
from tests.helpers.allocation_initializer import _create_array, _get_underlying_field


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
    ],
)
@test_utils.test()
def test_initializer_fields_share_tree(container_type: str, backend: qd.Backend | None) -> None:
    # Field declarations with fill= must defer compilation, share one allocation tree, and initialize every field.
    num_fields = 40
    fields = [
        _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=i)
        for i in range(num_fields)
    ]
    total = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=())
    assert impl.get_runtime().get_num_compiled_functions() == 0

    @qd.kernel
    def k_add_field(field: qd.Tensor, total: qd.Tensor) -> None:
        total[()] += field[()]

    for field in fields:
        k_add_field(field, total)
    assert total[()] == sum(range(num_fields))
    assert len({_get_underlying_field(field).snode._snode_tree_id for field in fields + [total]}) == 1


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
    ],
)
@test_utils.test()
def test_field_initializer_is_not_replayed(container_type: str, backend: qd.Backend | None) -> None:
    # Once field initialization has run, later synchronization must preserve writes made afterward.
    field = _create_array(
        container_type=container_type, backend=backend, element_class="scalar", shape=(), dtype=qd.i32, fill=3
    )
    assert field[()] == 3
    field[()] = 5

    # Each sync calls materialize() again. Repeating it must not replay the field initializer over the later write.
    qd.sync()
    assert field[()] == 5
    qd.sync()
    assert field[()] == 5


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
    ],
)
@test_utils.test()
def test_field_initializer_after_materialization(container_type: str, backend: qd.Backend | None) -> None:
    # Fields declared after an earlier allocation must initialize together in a new allocation tree.
    first = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=2)
    assert first[()] == 2
    second = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=4)
    third = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=6)
    assert second[()] == 4
    assert third[()] == 6
    assert _get_underlying_field(second).snode._snode_tree_id == _get_underlying_field(third).snode._snode_tree_id
    assert _get_underlying_field(first).snode._snode_tree_id != _get_underlying_field(second).snode._snode_tree_id


@test_utils.test()
def test_field_initializer_manual_placement() -> None:
    # A field declared without a shape must receive its initializer after manual placement.
    num_elements = 4
    field = qd.field(dtype=qd.i32, fill=8)
    qd.root.dense(qd.i, num_elements).place(field)
    np.testing.assert_array_equal(field.to_numpy(), [8] * num_elements)


@test_utils.test()
def test_field_initializer_gradients_and_offset() -> None:
    # Field initialization must respect the index offset and leave gradient and dual fields at zero.
    field = qd.field(dtype=qd.f32, shape=4, offset=-2, needs_grad=True, needs_dual=True, fill=2)
    assert field[-2] == 2
    assert field[1] == 2
    assert field.grad[-2] == 0
    assert field.dual[-2] == 0


@test_utils.test()
def test_field_initializer_reset() -> None:
    # Reset must discard pending field initializers and allow fresh fields to initialize in the new runtime.
    qd.field(dtype=qd.i32, shape=(), fill=8)
    runtime = impl.get_runtime()
    arch = qd.cfg.arch
    qd.reset()
    assert not runtime.field_initializers.pending
    qd.init(arch=arch)
    field = qd.field(dtype=qd.i32, shape=(), fill=4)
    assert field[()] == 4


@test_utils.test()
def test_field_initializer_preserves_immediate_fill() -> None:
    # Calling .fill() on a field must still compile immediately, placing later field declarations in a separate tree.
    first = qd.field(dtype=qd.i32, shape=())
    first.fill(3)
    assert impl.get_runtime().get_num_compiled_functions() > 0
    second = qd.field(dtype=qd.i32, shape=(), fill=4)
    assert second[()] == 4
    assert first.snode._snode_tree_id != second.snode._snode_tree_id
