"""Regression: layout= + needs_grad= preserves canonical indexing on the
field backend.

This is a test-only PR: an earlier change added the layout= keyword and an earlier change covered
needs_grad pass-through. The combination was checked at allocation time
in an earlier change (test_layout_field_with_needs_grad_allocates_grad) but not yet
exercised through a kernel write/read on the grad buffer with a non-
identity layout. That's the gap this PR closes.

Pre-impl POC Q3b already established this works; these tests pin it down
in the suite so a regression in upstream Quadrants (e.g. grad SNode no
longer inheriting axis_seq from the primal) would surface immediately.
"""

import itertools

import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cpu)
def test_layout_field_grad_canonical_kernel_roundtrip_rank2():
    """Write to primal and grad through a kernel using canonical indices on
    a transposed-storage field. Both must read back canonically."""
    a = qd.tensor(qd.f32, shape=(4, 5), layout=(1, 0), needs_grad=True)

    @qd.kernel
    def write_primal(x: qd.template()):
        for i, j in qd.ndrange(4, 5):
            x[i, j] = i * 10.0 + j

    @qd.kernel
    def write_grad(x: qd.template()):
        for i, j in qd.ndrange(4, 5):
            x.grad[i, j] = i * 100.0 + j * 10.0

    write_primal(a)
    write_grad(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal.shape == grad.shape == (4, 5)
    assert primal[1, 2] == 12.0
    assert grad[1, 2] == 120.0
    assert primal[3, 4] == 34.0
    assert grad[3, 4] == 340.0


@test_utils.test(arch=qd.cpu)
def test_layout_field_grad_canonical_kernel_roundtrip_rank3_kij():
    """Same as above but rank 3 with a non-trivial permutation."""
    a = qd.tensor(qd.f32, shape=(2, 3, 4), layout=(2, 0, 1), needs_grad=True)

    @qd.kernel
    def write_primal(x: qd.template()):
        for i, j, k in qd.ndrange(2, 3, 4):
            x[i, j, k] = i * 100.0 + j * 10.0 + k

    @qd.kernel
    def write_grad(x: qd.template()):
        for i, j, k in qd.ndrange(2, 3, 4):
            x.grad[i, j, k] = i * 1000.0 + j * 100.0 + k * 10.0

    write_primal(a)
    write_grad(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal.shape == grad.shape == (2, 3, 4)
    assert primal[1, 2, 3] == 123.0
    assert grad[1, 2, 3] == 1230.0


@pytest.mark.parametrize("layout", list(itertools.permutations(range(3))))
def test_layout_field_grad_all_rank3_permutations(layout):
    """Every rank-3 permutation produces canonical-indexable primal and grad."""
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.f32, shape=(2, 3, 4), layout=layout, needs_grad=True)

    @qd.kernel
    def fill(x: qd.template()):
        for i, j, k in qd.ndrange(2, 3, 4):
            x[i, j, k] = i * 100.0 + j * 10.0 + k
            x.grad[i, j, k] = 1000.0 + i * 100.0 + j * 10.0 + k

    fill(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal.shape == (2, 3, 4)
    assert grad.shape == (2, 3, 4)
    assert primal[1, 2, 3] == 123.0
    assert grad[1, 2, 3] == 1123.0


# ----------------------------------------------------------------------------
# NDARRAY backend: layout= + needs_grad= must propagate the layout tag onto
# the companion grad ndarray. Otherwise a kernel write to x.grad[i, j, ...]
# bypasses the canonical->physical subscript rewrite and lands in the wrong
# physical slot.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_ndarray_grad_tag_propagates_rank2():
    a = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY, layout=(1, 0), needs_grad=True)
    assert a.grad is not None
    assert getattr(a, "_qd_layout", None) == (1, 0)
    assert getattr(a.grad, "_qd_layout", None) == (1, 0)


@test_utils.test(arch=qd.cpu)
def test_layout_ndarray_grad_canonical_kernel_roundtrip_rank2():
    """Write primal and grad through a kernel using canonical indices on a
    transposed-storage ndarray. Both must read back canonically."""
    a = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY, layout=(1, 0), needs_grad=True)

    @qd.kernel
    def write_primal(x: qd.types.ndarray()):
        for i, j in qd.ndrange(4, 5):
            x[i, j] = i * 10.0 + j

    @qd.kernel
    def write_grad(x: qd.types.ndarray()):
        for i, j in qd.ndrange(4, 5):
            x.grad[i, j] = i * 100.0 + j * 10.0

    write_primal(a)
    write_grad(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal[1, 2] == 12.0
    assert grad[1, 2] == 120.0
    assert primal[3, 4] == 34.0
    assert grad[3, 4] == 340.0


@pytest.mark.parametrize("layout", list(itertools.permutations(range(3))))
def test_layout_ndarray_grad_all_rank3_permutations(layout):
    """Every rank-3 permutation tags both primal and grad, and kernels see
    canonical-indexable primal and grad buffers."""
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.f32, shape=(2, 3, 4), backend=qd.Backend.NDARRAY, layout=layout, needs_grad=True)
    layout_t = tuple(layout)
    if layout_t == (0, 1, 2):
        # identity -> qd.tensor collapses to no _qd_layout tag
        assert getattr(a, "_qd_layout", None) is None
        assert getattr(a.grad, "_qd_layout", None) is None
    else:
        assert getattr(a, "_qd_layout", None) == layout_t
        assert getattr(a.grad, "_qd_layout", None) == layout_t

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j, k in qd.ndrange(2, 3, 4):
            x[i, j, k] = i * 100.0 + j * 10.0 + k
            x.grad[i, j, k] = 1000.0 + i * 100.0 + j * 10.0 + k

    fill(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal[1, 2, 3] == 123.0
    assert grad[1, 2, 3] == 1123.0
