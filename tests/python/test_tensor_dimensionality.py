import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("d", range(2, qd._lib.core.get_max_num_indices() + 1))
@test_utils.test()
def test_dimensionality(d):
    x = qd.Vector.field(2, dtype=qd.i32, shape=(2,) * d)

    @qd.kernel
    def fill():
        for I in qd.grouped(x):
            x[I] += qd.Vector([I.sum(), I[0]])

    for i in range(2**d):
        indices = []
        for j in range(d):
            indices.append(i // (2**j) % 2)
        x.__getitem__(tuple(indices))[0] = sum(indices) * 2
    fill()
    # FIXME(yuanming-hu): snode_writer needs 9 arguments actually..
    for i in range(2**d):
        indices = []
        for j in range(d):
            indices.append(i // (2**j) % 2)
        assert x.__getitem__(tuple(indices))[0] == sum(indices) * 3
