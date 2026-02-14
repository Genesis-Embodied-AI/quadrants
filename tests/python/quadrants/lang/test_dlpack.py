import os

import numpy as np
import pytest
import torch

import quadrants as ti

from tests import test_utils

dlpack_arch = [ti.cpu, ti.cuda, ti.metal, ti.amdgpu]
dlpack_ineligible_arch = [ti.vulkan]


def ti_to_torch(ti_tensor: ti.types.NDArray) -> torch.Tensor:
    cap = ti_tensor.to_dlpack()
    torch_tensor = torch.utils.dlpack.from_dlpack(cap)
    return torch_tensor


def is_v520_amdgpu():
    return os.environ.get("QD_AMDGPU_V520", None) == "1" and ti.cfg.arch == ti.amdgpu


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("dtype", [ti.i32, ti.i64, ti.f32, ti.f64, ti.u1])
@pytest.mark.parametrize(
    "shape,poses",
    [
        ((), [()]),
        ((3,), [(0,), (2,)]),
        ((3, 2), [(0, 0), (2, 1), (1, 1)]),
        ((3, 1, 2), [(2, 0, 1), (0, 0, 1)]),
    ],
)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_types(tensor_type, dtype, shape: tuple[int], poses: list[tuple[int, ...]]) -> None:
    if ti.cfg.arch == ti.metal and dtype is ti.f64:
        pytest.skip("Metal does not support f64")
    ti_tensor = tensor_type(dtype, shape)
    for i, pos in enumerate(poses):
        ti_tensor[pos] = i * 10 + 10
    ti.sync()
    dlpack = ti_tensor.to_dlpack()
    tt = torch.utils.dlpack.from_dlpack(dlpack)
    assert tuple(tt.shape) == shape
    expected_torch_type = {
        ti.i32: torch.int32,
        ti.i64: torch.int64,
        ti.f32: torch.float32,
        ti.f64: torch.float64,
        ti.u1: torch.bool,
    }[dtype]
    assert tt.dtype == expected_torch_type
    if ti.cfg.arch == ti.amdgpu:
        # can't run torch kernels on AWS AMD GPU
        tt = tt.cpu()
    for i, pos in enumerate(poses):
        assert tt[pos] == ti_tensor[pos]
        assert tt[pos] != 0


@test_utils.test(arch=dlpack_arch)
def test_dlpack_ndarray_mem_stays_alloced() -> None:
    """
    On fields, memory always stays allocated till ti.reset(), so we
    don't need to check. Not true with ndarrays.
    """

    def create_tensor(shape, dtype):
        nd = ti.ndarray(dtype, shape)
        tt = torch.utils.dlpack.from_dlpack(nd.to_dlpack())
        return tt

    t = create_tensor((3, 2), ti.i32)
    # accessing memory will crash if memory already deleted
    if is_v520_amdgpu():
        # can't run torch kernels on AWS AMD GPU
        t = t.cpu()
    assert t[0, 0] == 0


@test_utils.test(arch=dlpack_ineligible_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_refuses_ineligible_arch(tensor_type) -> None:
    def create_tensor(shape, dtype):
        nd = tensor_type(dtype, shape)
        tt = torch.utils.dlpack.from_dlpack(nd.to_dlpack())
        return tt

    with pytest.raises(RuntimeError):
        t = create_tensor((3, 2), ti.i32)
        t[0, 0]


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_vec3(tensor_type):
    vec3 = ti.types.vector(3, ti.f32)
    a = tensor_type(vec3, shape=(10, 3))
    a[0, 0] = (5, 4, 3)
    a[0, 1] = (7, 8, 9)
    a[1, 0] = (11, 12, 13)
    ti.sync()
    tt = ti_to_torch(a)
    if is_v520_amdgpu():
        # can't run torch accessor kernels on v520
        tt = tt.cpu()
    assert tuple(tt.shape) == (10, 3, 3)
    assert tt.dtype == torch.float32
    assert tt[0, 0, 0] == 5
    assert tt[0, 0, 1] == 4
    assert tt[0, 0, 2] == 3
    assert tt[0, 1, 0] == 7
    assert tt[0, 1, 1] == 8
    assert tt[0, 1, 2] == 9
    assert tt[1, 0, 0] == 11
    assert tt[1, 0, 1] == 12
    assert tt[1, 0, 2] == 13


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_mat2x3(tensor_type):
    vec3 = ti.types.matrix(2, 3, ti.f32)
    a = tensor_type(vec3, shape=(10, 3))
    a[0, 0] = ((5, 4, 1), (3, 2, 20))
    a[0, 1] = ((7, 8, 21), (9, 10, 22))
    a[1, 0] = ((11, 12, 23), (13, 14, 23))
    tt = ti_to_torch(a)
    if is_v520_amdgpu():
        # can't run torch accessor kernels on v520
        tt = tt.cpu()
    assert tuple(tt.shape) == (10, 3, 2, 3)
    assert tt.dtype == torch.float32
    assert tt[0, 0, 0, 0] == 5
    assert tt[0, 0, 0, 1] == 4
    assert tt[0, 0, 0, 2] == 1
    assert tt[0, 0, 1, 0] == 3
    assert tt[0, 0, 1, 1] == 2
    assert tt[0, 0, 1, 2] == 20
    assert tt[0, 1, 1, 1] == 10


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_2_arrays(tensor_type):
    """
    Just in case we need to handle offset (which we do, for fields)
    """
    a = tensor_type(ti.i32, (100,))
    b = tensor_type(ti.i32, (100,))
    a[0] = 123
    a[1] = 101
    a[2] = 102
    a[3] = 103
    b[0] = 222
    ti.sync()

    # first field has offset 0
    # second field has non-zero offset, so we are testing
    # non-zero offsets
    a_t = ti_to_torch(a)
    b_t = ti_to_torch(b)

    if is_v520_amdgpu():
        # can't run torch accessor kernels on v520
        a_t = a_t.cpu()
        b_t = b_t.cpu()

    assert a_t[0] == 123
    assert b_t[0] == 222


@test_utils.test(arch=dlpack_arch)
def test_dlpack_non_sequenced_axes():
    field_ikj = ti.field(ti.f32)
    ti.root.dense(ti.i, 3).dense(ti.k, 2).dense(ti.j, 4).place(field_ikj)
    # create the field (since we arent initializing its value in any way, which would implicilty
    # call ti.sync())
    ti.sync()
    with pytest.raises(RuntimeError):
        ti_to_torch(field_ikj)


@test_utils.test(arch=dlpack_arch)
def test_dlpack_field_multiple_tree_nodes():
    """
    each ti.sync causes the fields to be written to a new snode tree node
    each tree node has its own memory block, so we want to test:
    - multiple snodes within same tree node
    - different tree nodes
    ... just to check we aren't aliasing somehow in the to_dlpack function
    """
    tensor_type = ti.field
    a = tensor_type(ti.i32, (100,))
    b = tensor_type(ti.i32, (100,))
    ti.sync()
    c = tensor_type(ti.i32, (100,))
    ti.sync()
    d = tensor_type(ti.i32, (100,))
    e = tensor_type(ti.i32, (100,))
    ti.sync()

    # check the tree node ids
    assert a.snode._snode_tree_id == 0
    assert b.snode._snode_tree_id == 0
    assert c.snode._snode_tree_id == 1
    assert d.snode._snode_tree_id == 2
    assert e.snode._snode_tree_id == 2

    a[0] = 123
    b[0] = 222
    c[0] = 333
    d[0] = 444
    e[0] = 555
    ti.sync()

    a_t = ti_to_torch(a)

    b_t = ti_to_torch(b)
    c_t = ti_to_torch(c)
    d_t = ti_to_torch(d)
    e_t = ti_to_torch(e)

    if is_v520_amdgpu():
        # can't run torch accessor kernels on v520
        a_t = a_t.cpu()
        b_t = b_t.cpu()
        c_t = c_t.cpu()
        d_t = d_t.cpu()
        e_t = e_t.cpu()

    assert a_t[0] == 123
    assert b_t[0] == 222
    assert c_t[0] == 333
    assert d_t[0] == 444
    assert e_t[0] == 555


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("dtype", [ti.i32, ti.i64, ti.f32, ti.f64, ti.u1, ti.i8, ti.types.vector(3, ti.i32)])
@pytest.mark.parametrize("shape", [3, 1, 4, 5, 7, 2])
def test_dlpack_mixed_types_memory_alignment_field(dtype, shape: tuple[int]) -> None:
    """
    Note: The mixed type here means that within a single SNode tree, fields use different data types (for example, curr_cnt in ti.i32 and pos in ti.i64). This leads to memory alignment issues and mismatched SNode offsets.
    """
    if ti.cfg.arch == ti.metal and dtype in [ti.i64, ti.f64]:
        pytest.skip(reason="64-bit types not supported on Metal")
    vtype = ti.i32 if ti.cfg.arch == ti.metal else ti.i64

    _curr_field = ti.field(dtype, shape)
    pos = ti.field(ti.types.vector(3, vtype), shape=(1,))

    @ti.kernel
    def kernel_update_render_fields(pos: ti.template()):
        pos[0] = ti.Vector([1, 2, 3], dt=vtype)

    kernel_update_render_fields(pos)
    ti.sync()

    np.testing.assert_allclose(
        torch.utils.dlpack.from_dlpack(pos.to_dlpack()).cpu().numpy(),
        pos.to_numpy(),
    )


@test_utils.test(arch=dlpack_arch)
def test_dlpack_multiple_mixed_types_memory_alignment_field() -> None:
    vtype = ti.i32 if ti.cfg.arch == ti.metal else ti.i64
    dtypes = [ti.i32, ti.f32, ti.u1, ti.i8, ti.types.vector(3, ti.i32)]
    if ti.cfg.arch != ti.metal:
        dtypes.extend([ti.i64, ti.f64])
    shapes = [3, 1, 4, 5, 7, 2, 3]
    fields = []
    for dtype, shape in zip(dtypes, shapes):
        fields.append(ti.field(dtype, shape))
    pos = ti.field(ti.types.vector(3, vtype), shape=(1,))

    @ti.kernel
    def kernel_update_render_fields(pos: ti.template()):
        pos[0] = ti.Vector([1, 2, 3], dt=vtype)

    kernel_update_render_fields(pos)
    ti.sync()

    np.testing.assert_allclose(
        torch.utils.dlpack.from_dlpack(pos.to_dlpack()).cpu().numpy(),
        pos.to_numpy(),
    )


@test_utils.test(arch=dlpack_arch)
def test_dlpack_joints_case_memory_alignment_field() -> None:
    _links_is_fixed = ti.field(dtype=ti.u1, shape=(1,))
    joints_n_dofs = ti.field(dtype=ti.i32, shape=(1,))

    @ti.kernel
    def kernel_init_joint_fields(
        joints_dof_start: ti.types.ndarray(),
        joints_dof_end: ti.types.ndarray(),
        joints_n_dofs: ti.template(),
    ):
        for I_j in ti.grouped(joints_n_dofs):
            joints_n_dofs[I_j] = joints_dof_end[I_j] - joints_dof_start[I_j]

    kernel_init_joint_fields(
        joints_dof_start=np.array([0], dtype=np.int32),
        joints_dof_end=np.array([6], dtype=np.int32),
        joints_n_dofs=joints_n_dofs,
    )
    ti.sync()

    np.testing.assert_allclose(
        torch.utils.dlpack.from_dlpack(joints_n_dofs.to_dlpack()).cpu().numpy(),
        joints_n_dofs.to_numpy(),
    )


@test_utils.test(arch=dlpack_arch)
def test_dlpack_field_memory_allocation_before_to_dlpack():
    if is_v520_amdgpu():
        pytest.skip(reason="can't run torch accessor kernels on v520")
    first_time = ti.field(dtype=ti.i32, shape=(1,))
    first_time_tc = torch.utils.dlpack.from_dlpack(first_time.to_dlpack())

    first_time_tc[:] = 1
    assert (first_time_tc == first_time.to_torch(device=first_time_tc.device)).all()

    second_time = ti.Vector.field(3, dtype=ti.i32, shape=(1,))
    second_time_tc = torch.utils.dlpack.from_dlpack(second_time.to_dlpack())

    second_time_tc[:] = 2
    assert (
        second_time_tc == second_time.to_torch(device=second_time_tc.device)
    ).all(), f"{second_time_tc} != {second_time.to_torch(device=second_time_tc.device)}"
