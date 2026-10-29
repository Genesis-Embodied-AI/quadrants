from fractions import Fraction

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests.helpers import test_utils


@test_utils.test()
def test_fill_initializer_shares_tree() -> None:
    # Declarations with fill= must defer compilation, share one allocation tree, and initialize every field.
    fields = [qd.field(dtype=qd.f32, shape=(), fill=i) for i in range(40)]
    total = qd.field(dtype=qd.f32, shape=())
    assert impl.get_runtime().get_num_compiled_functions() == 0

    @qd.kernel
    def add_field(field: qd.Template) -> None:
        total[()] += field[()]

    for field in fields:
        add_field(field)
    assert total[()] == 780
    assert len({field.snode._snode_tree_id for field in fields + [total]}) == 1


@pytest.mark.parametrize("operation", ["fill", "host", "numpy", "kernel"])
@test_utils.test()
def test_fill_initializer_precedes_writes(operation: str) -> None:
    # Initialization must run before each kind of write.
    field = qd.field(dtype=qd.i32, shape=4, fill=3)
    if operation == "fill":
        field.fill(5)
    elif operation == "host":
        field[0] = 5
    elif operation == "numpy":
        field.from_numpy(np.full(4, 5, dtype=np.int32))
    elif operation == "kernel":

        @qd.kernel
        def k_write() -> None:
            for i in field:
                field[i] += 2

        k_write()
    assert field[0] == 5


@test_utils.test()
def test_fill_initializer_is_not_replayed() -> None:
    # Once initialization has run, later synchronization must preserve writes made afterward.
    field = qd.field(dtype=qd.i32, shape=(), fill=3)
    assert field[()] == 3
    field[()] = 5

    # Each sync calls materialize() again. Repeating it must not replay the initializer over the later write.
    qd.sync()
    assert field[()] == 5
    qd.sync()
    assert field[()] == 5


@test_utils.test()
def test_fill_initializer_after_materialization() -> None:
    # Fields declared after an earlier allocation must initialize together in a new allocation tree.
    first = qd.field(dtype=qd.i32, shape=(), fill=2)
    assert first[()] == 2
    second = qd.field(dtype=qd.i32, shape=(), fill=4)
    third = qd.field(dtype=qd.i32, shape=(), fill=6)
    assert second[()] == 4
    assert third[()] == 6
    assert second.snode._snode_tree_id == third.snode._snode_tree_id
    assert first.snode._snode_tree_id != second.snode._snode_tree_id


@pytest.mark.parametrize("element_class", ["vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_fill_initializer_components(element_class: str, via_dtype: bool) -> None:
    # All vector and matrix constructors must capture component lists before later mutations.
    num_elements = 3
    value = [1, 2] if element_class == "vector" else [[1, 2], [3, 4]]
    expected = np.array(value)
    if via_dtype:
        dtype = (
            qd.types.vector(n=2, dtype=qd.i32) if element_class == "vector" else qd.types.matrix(n=2, m=2, dtype=qd.i32)
        )
        field = qd.field(dtype=dtype, shape=num_elements, fill=value)
    elif element_class == "vector":
        field = qd.Vector.field(n=2, dtype=qd.i32, shape=num_elements, fill=value)
    else:
        field = qd.Matrix.field(n=2, m=2, dtype=qd.i32, shape=num_elements, fill=value)
    if element_class == "vector":
        value[0] = 99
    else:
        value[0][0] = 99
    for i in range(num_elements):
        np.testing.assert_array_equal(field[i].to_numpy(), expected)


@pytest.mark.parametrize("element_class", ["vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_fill_initializer_broadcast(element_class: str, via_dtype: bool) -> None:
    # A scalar initializer must fill every component of every vector or matrix element.
    num_elements = 3
    if via_dtype:
        dtype = (
            qd.types.vector(n=2, dtype=qd.i32) if element_class == "vector" else qd.types.matrix(n=2, m=2, dtype=qd.i32)
        )
        field = qd.field(dtype=dtype, shape=num_elements, fill=9)
    elif element_class == "vector":
        field = qd.Vector.field(n=2, dtype=qd.i32, shape=num_elements, fill=9)
    else:
        field = qd.Matrix.field(n=2, m=2, dtype=qd.i32, shape=num_elements, fill=9)
    expected = np.full((2,) if element_class == "vector" else (2, 2), 9)
    for i in range(num_elements):
        np.testing.assert_array_equal(field[i].to_numpy(), expected)


@test_utils.test()
def test_fill_initializer_matrix_value_snapshot() -> None:
    # A Matrix used as an initializer must be captured before later changes to that Matrix.
    value = qd.Matrix([[1, 2], [3, 4]])
    field = qd.Matrix.field(n=2, m=2, dtype=qd.i32, shape=(), fill=value)
    value[0, 0] = 99
    np.testing.assert_array_equal(field[()].to_numpy(), [[1, 2], [3, 4]])


@test_utils.test()
def test_fill_initializer_manual_placement() -> None:
    # A field declared without a shape must receive its initializer after manual placement.
    field = qd.field(dtype=qd.i32, fill=8)
    qd.root.dense(qd.i, 4).place(field)
    np.testing.assert_array_equal(field.to_numpy(), [8] * 4)


@test_utils.test()
def test_fill_initializer_gradients_and_offset() -> None:
    # Initialization must respect the index offset and leave gradient and dual fields at zero.
    field = qd.field(dtype=qd.f32, shape=4, offset=-2, needs_grad=True, needs_dual=True, fill=2)
    assert field[-2] == 2
    assert field[1] == 2
    assert field.grad[-2] == 0
    assert field.dual[-2] == 0


@test_utils.test()
def test_fill_initializer_reset() -> None:
    # Reset must discard pending initializers and allow fresh fields to initialize in the new runtime.
    qd.field(dtype=qd.i32, shape=(), fill=8)
    runtime = impl.get_runtime()
    arch = qd.cfg.arch
    qd.reset()
    assert not runtime.field_initializers.pending
    qd.init(arch=arch)
    field = qd.field(dtype=qd.i32, shape=(), fill=4)
    assert field[()] == 4


@pytest.mark.parametrize("value", [[1, 2], "invalid"])
@test_utils.test()
def test_fill_initializer_invalid_scalar(value: list[int] | str) -> None:
    # Invalid scalar initializers must raise at declaration without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        qd.field(dtype=qd.i32, shape=(), fill=value)
    field = qd.field(dtype=qd.i32, shape=(), fill=7)
    assert field[()] == 7


@pytest.mark.parametrize("value", [[[1, 2]], [[1, 2], [3]], [1, 2], [[1, None], [2, 3]]])
@test_utils.test()
def test_fill_initializer_invalid_matrix(value: list[int] | list[list[int | None]]) -> None:
    # Invalid matrix shapes or components must raise without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        qd.Matrix.field(n=2, m=2, dtype=qd.i32, shape=(), fill=value)
    field = qd.field(dtype=qd.i32, shape=(), fill=7)
    assert field[()] == 7


@test_utils.test()
def test_fill_initializer_preserves_immediate_fill() -> None:
    # Calling .fill() must still compile immediately, placing later declarations in a separate tree.
    first = qd.field(dtype=qd.i32, shape=())
    first.fill(3)
    assert impl.get_runtime().get_num_compiled_functions() > 0
    second = qd.field(dtype=qd.i32, shape=(), fill=4)
    assert second[()] == 4
    assert first.snode._snode_tree_id != second.snode._snode_tree_id


@pytest.mark.needs_torch
@pytest.mark.parametrize("matrix", [False, True])
@test_utils.test()
def test_fill_initializer_dlpack(matrix: bool) -> None:
    # After synchronization, PyTorch must see initialized scalar and vector values through DLPack.
    torch = test_utils.require_torch_dlpack(need_field_support=True)
    field = (
        qd.Vector.field(n=2, dtype=qd.f32, shape=4, fill=[2, 3]) if matrix else qd.field(dtype=qd.f32, shape=4, fill=7)
    )
    qd.sync()
    torch_view = torch.utils.dlpack.from_dlpack(field.to_dlpack())
    expected = [[2, 3]] * 4 if matrix else [7] * 4
    np.testing.assert_array_equal(torch_view.cpu().numpy(), expected)


@pytest.mark.parametrize("fill_value", [1 + 2j, np.complex64(1 + 2j), Fraction(1, 2)])
@pytest.mark.parametrize("element_class", ["scalar", "vector", "matrix"])
@test_utils.test()
def test_fill_initializer_unsupported_number(
    element_class: str, fill_value: complex | np.complexfloating | Fraction
) -> None:
    # Unsupported numbers must fail at declaration, including inside vector and matrix component lists.
    with pytest.raises(ValueError, match="fill"):
        if element_class == "scalar":
            qd.field(dtype=qd.f32, shape=(), fill=fill_value)
        elif element_class == "vector":
            qd.Vector.field(n=2, dtype=qd.f32, shape=(), fill=[1, fill_value])
        elif element_class == "matrix":
            qd.Matrix.field(n=2, m=2, dtype=qd.f32, shape=(), fill=[[1, 2], [3, fill_value]])
    field = qd.field(dtype=qd.f32, shape=(), fill=7)
    assert field[()] == 7


@pytest.mark.parametrize("fill_value", [True, 2, 2.5, np.bool_(True), np.int32(2), np.float32(2.5)])
@pytest.mark.parametrize("element_class", ["scalar", "vector", "matrix"])
@test_utils.test()
def test_fill_initializer_supported_number(
    element_class: str, fill_value: bool | int | float | np.bool_ | np.integer | np.floating
) -> None:
    # Supported Python and NumPy scalars must initialize scalar fields and broadcast to vector and matrix fields.
    if element_class == "scalar":
        field = qd.field(dtype=qd.f32, shape=(), fill=fill_value)
    elif element_class == "vector":
        field = qd.Vector.field(n=2, dtype=qd.f32, shape=(), fill=fill_value)
    elif element_class == "matrix":
        field = qd.Matrix.field(n=2, m=2, dtype=qd.f32, shape=(), fill=fill_value)
    np.testing.assert_array_equal(field.to_numpy(), np.full(field.to_numpy().shape, fill_value))


@test_utils.test()
def test_fill_initializer_retries_after_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A failed field fill must retain itself and later fills without replaying already successful fills.
    first = qd.field(dtype=qd.i32, shape=(), fill=2)
    second = qd.field(dtype=qd.i32, shape=(), fill=3)
    third = qd.field(dtype=qd.i32, shape=(), fill=4)
    original_fill = type(first).fill
    calls: list[qd.Field] = []
    should_fail = True

    def fill(field: qd.Field, fill_value: int) -> None:
        nonlocal should_fail
        calls.append(field)
        if field is second and should_fail:
            should_fail = False
            raise RuntimeError("injected fill failure")
        original_fill(field, fill_value)

    monkeypatch.setattr(type(first), "fill", fill)
    with pytest.raises(RuntimeError, match="injected fill failure"):
        qd.sync()
    assert [field for field, _ in impl.get_runtime().field_initializers.pending] == [second, third]
    assert third[()] == 4
    assert first[()] == 2
    assert second[()] == 3
    assert calls == [first, second, second, third]


@pytest.mark.parametrize("element_class", ["scalar", "vector", "matrix"])
@test_utils.test(exclude=qd.metal)
def test_fill_initializer_destroyed_tree(element_class: str) -> None:
    # Destroying a field's tree must discard its pending fill and preserve fills for other and unplaced fields.
    builder = qd.FieldsBuilder()
    if element_class == "scalar":
        doomed = qd.field(dtype=qd.i32, fill=2)
    elif element_class == "vector":
        doomed = qd.Vector.field(n=2, dtype=qd.i32, fill=2)
    elif element_class == "matrix":
        doomed = qd.Matrix.field(n=2, m=2, dtype=qd.i32, fill=2)
    builder.dense(qd.i, 4).place(doomed)
    tree = builder.finalize()
    survivor = qd.field(dtype=qd.i32, shape=(), fill=7)
    unplaced = qd.field(dtype=qd.i32, fill=9)
    runtime = impl.get_runtime()
    assert runtime.get_num_compiled_functions() == 0
    tree.destroy()
    assert [field for field, _ in runtime.field_initializers.pending] == [survivor, unplaced]
    qd.root.place(unplaced)
    assert survivor[()] == 7
    assert unplaced[()] == 9
