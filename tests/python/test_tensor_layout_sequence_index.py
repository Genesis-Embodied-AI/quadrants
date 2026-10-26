"""Sequence indices passed through functions must preserve canonical tensor indexing."""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("backend", [qd.Backend.FIELD, qd.Backend.NDARRAY])
@pytest.mark.parametrize("shape, layout", [((3, 5), None), ((3, 5), (0, 1)), ((3, 5), (1, 0)), ((3, 4, 5), (1, 2, 0))])
@pytest.mark.parametrize("index_kind", ["list", "tuple"])
@pytest.mark.parametrize("operation", ["read", "write", "read_modify_write", "augassign", "atomic_add"])
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_layout_sequence_index(backend, shape, layout, index_kind, operation):
    x = qd.tensor(qd.i32, shape=shape, backend=backend, layout=layout)
    out = qd.tensor(qd.i32, shape=shape, backend=backend)
    initial = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    x.from_numpy(initial)
    out.fill(-1)

    @qd.func
    def access(I, field: qd.Tensor, result: qd.Tensor):
        value = 1
        for axis in qd.static(range(len(shape))):
            value = value * 10 + I[axis]
        if qd.static(operation == "read"):
            result[I] = field[I]
        elif qd.static(operation == "write"):
            field[I] = value
        elif qd.static(operation == "read_modify_write"):
            field[I] = field[I] + value
        elif qd.static(operation == "augassign"):
            field[I] += value
        else:
            qd.atomic_add(field[I], value)

    @qd.kernel
    def run(field: qd.Tensor, result: qd.Tensor):
        # Keep accesses in bounds even without permutation, so a regression reports wrong values rather than OOB.
        for I in qd.grouped(qd.ndrange(*((3,) * len(shape)))):
            index = [I[axis] for axis in qd.static(range(len(shape)))]
            if qd.static(index_kind == "tuple"):
                access(tuple(index), field, result)
            else:
                access(index, field, result)

    run(x, out)
    expected = np.full(shape, -1, dtype=np.int32) if operation == "read" else initial.copy()
    for index in np.ndindex(*((3,) * len(shape))):
        value = 1
        for component in index:
            value = value * 10 + component
        if operation == "read":
            expected[index] = initial[index]
        elif operation == "write":
            expected[index] = value
        else:
            expected[index] += value
    np.testing.assert_array_equal(out.to_numpy() if operation == "read" else x.to_numpy(), expected)
