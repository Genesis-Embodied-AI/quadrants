"""Tests for ``qd.Tensor`` as an annotation in ``@qd.func`` parameters and struct fields.

Pre-stork-23, ``qd.Tensor`` only worked as an annotation for top-level ``@qd.kernel`` parameters (handled by
``_template_mapper_hotpath``). Using it in a ``@qd.func`` parameter or as a struct field annotation raised
``QuadrantsTypeError`` because ``_transform_func_arg`` had no dispatch branch for the ``Tensor`` class.

Stork-23 adds that branch. These tests pin the invariant.
"""

import dataclasses

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsCompilationError

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


# ---------------------------------------------------------------------------
# 1. qd.Tensor as a standalone @qd.func parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_annotated_func_param(backend):
    """A @qd.func with a qd.Tensor-annotated param, called from a kernel."""
    N = 8
    a = qd.tensor(qd.i32, shape=(N,), backend=backend)

    @qd.func
    def double_it(x: qd.Tensor, i: qd.i32):
        x[i] = x[i] * 2

    @qd.kernel
    def run(x: qd.Tensor):
        for i in range(N):
            x[i] = i + 1
            double_it(x, i)

    run(a)
    np.testing.assert_array_equal(a.to_numpy(), np.arange(1, N + 1) * 2)


# ---------------------------------------------------------------------------
# 2. @qd.data_oriented struct with qd.Tensor field, kernel template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_struct_field_kernel_data_oriented(backend):
    """A struct with a qd.Tensor field passed to a kernel via template(). FIELD uses @qd.data_oriented; NDARRAY uses
    @dataclasses.dataclass(frozen=True)."""
    N = 6
    t = qd.tensor(qd.i32, shape=(N,), backend=backend)

    if backend == qd.Backend.FIELD:

        @qd.data_oriented
        class S:
            def __init__(self, vals):
                self.vals = vals

        s = S(vals=t)
    else:

        @dataclasses.dataclass(frozen=True)
        class S:
            vals: qd.Tensor

        s = S(vals=t)

    @qd.kernel
    def fill(st: qd.template()):
        for i in range(N):
            st.vals[i] = i * 3

    if backend == qd.Backend.NDARRAY:
        with pytest.raises(QuadrantsCompilationError, match="qd.template.*qd.ndarray"):
            fill(s)
    else:
        fill(s)
        np.testing.assert_array_equal(t.to_numpy(), np.arange(N) * 3)


# ---------------------------------------------------------------------------
# 3. Struct with qd.Tensor field, func takes struct as template.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_struct_field_func_via_template(backend):
    """A @qd.func receives a struct (containing qd.Tensor fields) as a qd.template() arg."""
    N = 4

    @qd.data_oriented
    class S:
        def __init__(self, vals):
            self.vals = vals

    t = qd.tensor(qd.i32, shape=(N,), backend=backend)
    s = S(vals=t)

    @qd.func
    def inc_all(st: qd.template()):
        for i in range(N):
            st.vals[i] = st.vals[i] + 10

    @qd.kernel
    def run(st: qd.template()):
        for i in range(N):
            st.vals[i] = i
        inc_all(st)

    if backend == qd.Backend.NDARRAY:
        with pytest.raises(QuadrantsCompilationError, match="qd.template.*qd.ndarray"):
            run(s)
    else:
        run(s)
        np.testing.assert_array_equal(t.to_numpy(), np.arange(N) + 10)


# ---------------------------------------------------------------------------
# 4. Tensor wrapper in struct field (unwrap path via build_Attribute).
#    qd.tensor() returns a Tensor wrapper (post stork-19), so this exercises the Tensor-unwrap in build_Attribute.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_wrapper_in_struct_field_unwraps(backend):
    """When the struct field stores a qd.Tensor *wrapper*, the AST build_Attribute must unwrap it transparently."""
    N = 4

    @qd.data_oriented
    class S:
        def __init__(self, vals):
            self.vals = vals

    t = qd.tensor(qd.i32, shape=(N,), backend=backend)
    assert isinstance(t, qd.Tensor)
    s = S(vals=t)

    @qd.func
    def write(st: qd.template(), i: qd.i32, v: qd.i32):
        st.vals[i] = v

    @qd.kernel
    def run(st: qd.template()):
        for i in range(N):
            write(st, i, i * 7)

    if backend == qd.Backend.NDARRAY:
        with pytest.raises(QuadrantsCompilationError, match="qd.template.*qd.ndarray"):
            run(s)
    else:
        run(s)
        np.testing.assert_array_equal(t.to_numpy(), np.arange(N) * 7)


# ---------------------------------------------------------------------------
# 5. Mixed qd.Tensor and scalar template fields in the same struct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_mixed_tensor_and_scalar_struct_fields(backend):
    """Struct with one qd.Tensor field and one scalar, both accessed in a @qd.func via template."""
    N = 4

    @qd.data_oriented
    class S:
        def __init__(self, tensor_field, scale):
            self.tensor_field = tensor_field
            self.scale = scale

    t = qd.tensor(qd.i32, shape=(N,), backend=backend)
    s = S(tensor_field=t, scale=5)

    @qd.func
    def scaled_fill(st: qd.template(), i: qd.i32):
        st.tensor_field[i] = i * st.scale

    @qd.kernel
    def run(st: qd.template()):
        for i in range(N):
            scaled_fill(st, i)

    if backend == qd.Backend.NDARRAY:
        with pytest.raises(QuadrantsCompilationError, match="qd.template.*qd.ndarray"):
            run(s)
    else:
        run(s)
        np.testing.assert_array_equal(t.to_numpy(), np.arange(N) * 5)


# ---------------------------------------------------------------------------
# 7. qd.Tensor func param with 2D tensor and layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_func_param_2d_with_layout(backend):
    """qd.Tensor func param with a 2D layout-tagged tensor."""
    M, N = 3, 4
    a = qd.tensor(qd.i32, shape=(M, N), backend=backend, layout=(1, 0))

    @qd.func
    def fill_row(x: qd.Tensor, row: qd.i32):
        for j in range(N):
            x[row, j] = row * 100 + j

    @qd.kernel
    def run(x: qd.Tensor):
        for i in range(M):
            fill_row(x, i)

    run(a)
    arr = a.to_numpy()
    assert arr.shape == (M, N)
    for i in range(M):
        for j in range(N):
            assert arr[i, j] == i * 100 + j, f"mismatch at [{i},{j}]"


# ---------------------------------------------------------------------------
# 8. Struct with qd.Tensor vector-compound field — exercises populate_global_vars_from_dataclass +
#    element_shape propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_struct_vector_field_roundtrip(backend):
    """Struct field holding a *vector*-element qd.Tensor allocated via ``qd.tensor(vec3_type, ...)``.  Covers the
    ``_wrap_impl`` fix that ensures VectorTensor is returned for compound dtypes, plus the struct expansion path in
    ``populate_global_vars_from_dataclass``."""
    N = 4
    vec3 = qd.types.vector(3, qd.f32)
    t = qd.tensor(vec3, shape=(N,), backend=backend)
    assert isinstance(t, qd.VectorTensor), f"expected VectorTensor, got {type(t).__name__}"

    if backend == qd.Backend.FIELD:

        @qd.data_oriented
        class S:
            def __init__(self, vals):
                self.vals = vals

        s = S(vals=t)
    else:

        @dataclasses.dataclass(frozen=True)
        class S:
            vals: qd.Tensor

        s = S(vals=t)

    @qd.kernel
    def fill(st: qd.template()):
        for i in range(N):
            for j in qd.static(range(3)):
                st.vals[i][j] = i * 10.0 + j

    if backend == qd.Backend.NDARRAY:
        with pytest.raises(QuadrantsCompilationError, match="qd.template.*qd.ndarray"):
            fill(s)
    else:
        fill(s)
        arr = t.to_numpy()
        assert arr.shape == (N, 3)
        assert arr[2, 1] == pytest.approx(21.0)


# ---------------------------------------------------------------------------
# 9. Frozen dataclass with qd.Tensor field, kernel arg is struct TYPE (not qd.template()) — exercises
#    _predeclare_struct_ndarrays + ndarray_to_any_array resolution in impl.subscript (Option C path)
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_struct_field_typed_kernel_arg_ndarray():
    """A frozen dataclass with a qd.Tensor-annotated field passed to a kernel as ``state: State`` (typed arg, not
    qd.template()). This exercises the _predeclare_struct_ndarrays mechanism that registers bare Ndarray fields in
    ndarray_to_any_array so impl.subscript can resolve them."""
    N = 6

    @dataclasses.dataclass(frozen=True)
    class State:
        vals: qd.Tensor

    t = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    s = State(vals=t)

    @qd.kernel
    def fill(st: State):
        for i in range(N):
            st.vals[i] = i * 3

    fill(s)
    np.testing.assert_array_equal(t.to_numpy(), np.arange(N) * 3)


@test_utils.test(arch=qd.cpu)
def test_tensor_struct_field_typed_kernel_arg_field():
    """Same as above but with FIELD backend and @qd.data_oriented struct, kernel arg typed as the struct class."""
    N = 6

    @qd.data_oriented
    class State:
        def __init__(self, vals):
            self.vals = vals

    t = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.FIELD)
    s = State(vals=t)

    @qd.kernel
    def fill(st: qd.template()):
        for i in range(N):
            st.vals[i] = i * 3

    fill(s)
    np.testing.assert_array_equal(t.to_numpy(), np.arange(N) * 3)


# ---------------------------------------------------------------------------
# 10. Frozen dataclass with MULTIPLE qd.Tensor fields + typed kernel arg
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_struct_multiple_fields_typed_kernel_arg():
    """Struct with two qd.Tensor fields passed as typed kernel arg. Both fields must be correctly pre-declared and
    subscriptable."""
    N = 4

    @dataclasses.dataclass(frozen=True)
    class State:
        pos: qd.Tensor
        vel: qd.Tensor

    pos = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    vel = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    s = State(pos=pos, vel=vel)

    @qd.kernel
    def step(st: State):
        for i in range(N):
            st.pos[i] = i
            st.vel[i] = i * 2

    step(s)
    np.testing.assert_array_equal(pos.to_numpy(), np.arange(N))
    np.testing.assert_array_equal(vel.to_numpy(), np.arange(N) * 2)


# ---------------------------------------------------------------------------
# 11–15. Frozen dataclass with MIXED qd.Tensor + qd.types.ndarray() fields.
#
# This is the Genesis ConstraintState pattern: a frozen dataclass has some fields annotated as qd.types.ndarray() and
# others as qd.Tensor. The struct is passed to @qd.func via its dataclass type annotation (NOT via qd.template()).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_mixed_tensor_and_ndarray_frozen_dataclass_func():
    """Frozen dataclass with mixed qd.Tensor + qd.types.ndarray() fields, passed to @qd.func via dataclass type
    annotation. NDARRAY backend. This is the exact pattern used by Genesis ConstraintState in ndarray mode."""
    N = 6

    @dataclasses.dataclass(frozen=True)
    class State:
        regular: qd.types.ndarray()
        tensor_field: qd.Tensor

    regular_arr = qd.ndarray(qd.i32, shape=(N,))
    tensor_arr = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    state = State(regular=regular_arr, tensor_field=tensor_arr)

    @qd.func
    def process(s: State, i: qd.i32):
        s.tensor_field[i] = s.regular[i] * 3

    @qd.kernel
    def run(s: State):
        for i in range(N):
            s.regular[i] = i + 1
            process(s, i)

    run(state)
    np.testing.assert_array_equal(regular_arr.to_numpy(), np.arange(1, N + 1))
    np.testing.assert_array_equal(tensor_arr.to_numpy(), np.arange(1, N + 1) * 3)


@test_utils.test(arch=qd.cpu)
def test_mixed_tensor_and_ndarray_frozen_dataclass_kernel():
    """Frozen dataclass with mixed qd.Tensor + qd.types.ndarray() fields, passed to @qd.kernel via dataclass type
    annotation. NDARRAY backend. Both field types are read/written in the same kernel."""
    N = 4

    @dataclasses.dataclass(frozen=True)
    class State:
        alpha: qd.types.ndarray()
        beta: qd.Tensor

    alpha = qd.ndarray(qd.i32, shape=(N,))
    beta = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    state = State(alpha=alpha, beta=beta)

    @qd.kernel
    def run(s: State):
        for i in range(N):
            s.alpha[i] = i * 2
            s.beta[i] = i * 5

    run(state)
    np.testing.assert_array_equal(alpha.to_numpy(), np.arange(N) * 2)
    np.testing.assert_array_equal(beta.to_numpy(), np.arange(N) * 5)


@test_utils.test(arch=qd.cpu)
def test_mixed_many_tensor_and_ndarray_fields():
    """Frozen dataclass with multiple qd.Tensor and multiple qd.types.ndarray() fields interleaved — mirrors Genesis
    ConstraintState which has ~6 Tensor fields among ~30 ndarray fields."""
    N = 4

    @dataclasses.dataclass(frozen=True)
    class BigState:
        a_nd: qd.types.ndarray()
        b_tensor: qd.Tensor
        c_nd: qd.types.ndarray()
        d_tensor: qd.Tensor
        e_nd: qd.types.ndarray()
        f_tensor: qd.Tensor

    a = qd.ndarray(qd.i32, shape=(N,))
    b = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    c = qd.ndarray(qd.i32, shape=(N,))
    d = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    e = qd.ndarray(qd.i32, shape=(N,))
    f = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.NDARRAY)
    state = BigState(a_nd=a, b_tensor=b, c_nd=c, d_tensor=d, e_nd=e, f_tensor=f)

    @qd.func
    def compute(s: BigState, i: qd.i32):
        s.b_tensor[i] = s.a_nd[i] + 10
        s.d_tensor[i] = s.c_nd[i] + 20
        s.f_tensor[i] = s.e_nd[i] + 30

    @qd.kernel
    def run(s: BigState):
        for i in range(N):
            s.a_nd[i] = i
            s.c_nd[i] = i * 2
            s.e_nd[i] = i * 3
            compute(s, i)

    run(state)
    np.testing.assert_array_equal(a.to_numpy(), np.arange(N))
    np.testing.assert_array_equal(b.to_numpy(), np.arange(N) + 10)
    np.testing.assert_array_equal(c.to_numpy(), np.arange(N) * 2)
    np.testing.assert_array_equal(d.to_numpy(), np.arange(N) * 2 + 20)
    np.testing.assert_array_equal(e.to_numpy(), np.arange(N) * 3)
    np.testing.assert_array_equal(f.to_numpy(), np.arange(N) * 3 + 30)


@test_utils.test(arch=qd.cpu)
def test_mixed_tensor_and_field_dataoriented_template():
    """@qd.data_oriented struct with mixed qd.Tensor + qd.field, passed via qd.template(). FIELD backend. This is the
    genesis field-mode pattern (data_oriented structs are not dataclasses, so they are passed as templates)."""
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, regular, tensor_f):
            self.regular = regular
            self.tensor_f = tensor_f

    regular = qd.field(qd.i32, shape=(N,))
    tensor_f = qd.tensor(qd.i32, shape=(N,), backend=qd.Backend.FIELD)
    state = State(regular=regular, tensor_f=tensor_f)

    @qd.func
    def process(s: qd.template(), i: qd.i32):
        s.tensor_f[i] = s.regular[i] * 7

    @qd.kernel
    def run(s: qd.template()):
        for i in range(N):
            s.regular[i] = i + 1
            process(s, i)

    run(state)
    np.testing.assert_array_equal(regular.to_numpy(), np.arange(1, N + 1))
    np.testing.assert_array_equal(tensor_f.to_numpy(), np.arange(1, N + 1) * 7)


@test_utils.test(arch=qd.cpu)
def test_mixed_2d_tensor_and_ndarray_frozen_dataclass():
    """Frozen dataclass with 2D mixed qd.Tensor + qd.types.ndarray() fields. Mirrors the Genesis ConstraintState
    shape=(len_constraints, n_envs)."""
    M, N = 3, 4

    @dataclasses.dataclass(frozen=True)
    class State2D:
        nd_arr: qd.types.ndarray()
        t_arr: qd.Tensor

    nd = qd.ndarray(qd.i32, shape=(M, N))
    t = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY)
    state = State2D(nd_arr=nd, t_arr=t)

    @qd.func
    def fill(s: State2D, i: qd.i32, j: qd.i32):
        s.t_arr[i, j] = s.nd_arr[i, j] + 100

    @qd.kernel
    def run(s: State2D):
        for i in range(M):
            for j in range(N):
                s.nd_arr[i, j] = i * 10 + j
                fill(s, i, j)

    run(state)
    expected_nd = np.array([[i * 10 + j for j in range(N)] for i in range(M)])
    np.testing.assert_array_equal(nd.to_numpy(), expected_nd)
    np.testing.assert_array_equal(t.to_numpy(), expected_nd + 100)
