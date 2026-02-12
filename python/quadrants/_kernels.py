# type: ignore

from quadrants._funcs import field_fill_quadrants_scope  # noqa
from quadrants.lang import ops
from quadrants.lang._ndrange import ndrange
from quadrants.lang.expr import Expr
from quadrants.lang.field import ScalarField
from quadrants.lang.impl import grouped, static, static_assert
from quadrants.lang.kernel_impl import func, kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block, warp
from quadrants.lang.snode import deactivate
from quadrants.types import ndarray_type
from quadrants.types.annotations import template
from quadrants.types.primitive_types import f16, f32, f64, i32, u8  # noqa pylint: disable=unused-import


# A set of helper (meta)functions
@kernel
def fill_field(field: template(), val: template()):
    value = ops.cast(val, field.dtype)
    for I in grouped(field):
        field[I] = value


@kernel
def fill_ndarray(ndarray: ndarray_type.ndarray(), val: template()):
    for I in grouped(ndarray):
        ndarray[I] = val


@kernel
def fill_ndarray_matrix(ndarray: ndarray_type.ndarray(), val: template()):
    for I in grouped(ndarray):
        ndarray[I] = val


@kernel
def tensor_to_ext_arr(tensor: template(), arr: ndarray_type.ndarray()):
    # default value of offset is [], replace it with [0] * len
    offset = static(tensor.snode.ptr.offset if len(tensor.snode.ptr.offset) != 0 else [0] * len(tensor.shape))

    for I in grouped(tensor):
        arr[I - offset] = tensor[I]


@kernel
def ndarray_to_ext_arr(ndarray: ndarray_type.ndarray(), arr: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        arr[I] = ndarray[I]


@kernel
def ndarray_matrix_to_ext_arr(
    ndarray: ndarray_type.ndarray(),
    arr: ndarray_type.ndarray(),
    layout_is_aos: template(),
    as_vector: template(),
):
    for I in grouped(ndarray):
        for p in static(range(ndarray[I].n)):
            if static(as_vector):
                if static(layout_is_aos):
                    arr[I, p] = ndarray[I][p]
                else:
                    arr[p, I] = ndarray[I][p]
            else:
                for q in static(range(ndarray[I].m)):
                    if static(layout_is_aos):
                        arr[I, p, q] = ndarray[I][p, q]
                    else:
                        arr[p, q, I] = ndarray[I][p, q]


@kernel
def tensor_to_tensor(tensor: template(), other: template()):
    static_assert(tensor.shape == other.shape)
    shape = static(tensor.shape)
    tensor_offset = static(tensor.snode.ptr.offset if len(tensor.snode.ptr.offset) != 0 else [0] * len(shape))
    other_offset = static(other.snode.ptr.offset if len(other.snode.ptr.offset) != 0 else [0] * len(shape))

    for I in grouped(ndrange(*shape)):
        tensor[I + tensor_offset] = other[I + other_offset]


@kernel
def ext_arr_to_tensor(arr: ndarray_type.ndarray(), tensor: template()):
    # default value of offset is [], replace it with [0] * len
    offset = static(tensor.snode.ptr.offset if len(tensor.snode.ptr.offset) != 0 else [0] * len(tensor.shape))
    for I in grouped(tensor):
        tensor[I] = arr[I - offset]


@kernel
def ndarray_to_ndarray(ndarray: ndarray_type.ndarray(), other: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        ndarray[I] = other[I]


@kernel
def ext_arr_to_ndarray(arr: ndarray_type.ndarray(), ndarray: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        ndarray[I] = arr[I]


@kernel
def ext_arr_to_ndarray_matrix(
    arr: ndarray_type.ndarray(),
    ndarray: ndarray_type.ndarray(),
    layout_is_aos: template(),
    as_vector: template(),
):
    for I in grouped(ndarray):
        for p in static(range(ndarray[I].n)):
            if static(as_vector):
                if static(layout_is_aos):
                    ndarray[I][p] = arr[I, p]
                else:
                    ndarray[I][p] = arr[p, I]
            else:
                for q in static(range(ndarray[I].m)):
                    if static(layout_is_aos):
                        ndarray[I][p, q] = arr[I, p, q]
                    else:
                        ndarray[I][p, q] = arr[p, q, I]


@kernel
def matrix_to_ext_arr(mat: template(), arr: ndarray_type.ndarray(), as_vector: template()):
    # default value of offset is [], replace it with [0] * len
    offset = static(mat.snode.ptr.offset if len(mat.snode.ptr.offset) != 0 else [0] * len(mat.shape))

    for I in grouped(mat):
        for p in static(range(mat.n)):
            for q in static(range(mat.m)):
                if static(as_vector):
                    if static(getattr(mat, "ndim", 2) == 1):
                        arr[I - offset, p] = mat[I][p]
                    else:
                        arr[I - offset, p] = mat[I][p, q]
                else:
                    if static(getattr(mat, "ndim", 2) == 1):
                        arr[I - offset, p, q] = mat[I][p]
                    else:
                        arr[I - offset, p, q] = mat[I][p, q]


@kernel
def ext_arr_to_matrix(arr: ndarray_type.ndarray(), mat: template(), as_vector: template()):
    # default value of offset is [], replace it with [0] * len
    offset = static(mat.snode.ptr.offset if len(mat.snode.ptr.offset) != 0 else [0] * len(mat.shape))

    for I in grouped(mat):
        for p in static(range(mat.n)):
            for q in static(range(mat.m)):
                if static(getattr(mat, "ndim", 2) == 1):
                    if static(as_vector):
                        mat[I][p] = arr[I - offset, p]
                    else:
                        mat[I][p] = arr[I - offset, p, q]
                else:
                    if static(as_vector):
                        mat[I][p, q] = arr[I - offset, p]
                    else:
                        mat[I][p, q] = arr[I - offset, p, q]


# extract ndarray of raw vulkan memory layout to normal memory layout.
# the vulkan layout stored in ndarray : width-by-width stored along n-
# darray's shape[1] which is the height-axis(So use [size // h, size %
#  h]). And the height-order of vulkan layout is flip up-down.(So take
# [size = (h - 1 - j) * w + i] to get the index)
@kernel
def arr_vulkan_layout_to_arr_normal_layout(vk_arr: ndarray_type.ndarray(), normal_arr: ndarray_type.ndarray()):
    static_assert(len(normal_arr.shape) == 2)
    w = normal_arr.shape[0]
    h = normal_arr.shape[1]
    for i, j in ndrange(w, h):
        normal_arr[i, j] = vk_arr[(h - 1 - j) * w + i]


# extract ndarray of raw vulkan memory layout into a quadrants-field data
# structure with normal memory layout.
@kernel
def arr_vulkan_layout_to_field_normal_layout(vk_arr: ndarray_type.ndarray(), normal_field: template()):
    static_assert(len(normal_field.shape) == 2)
    w = static(normal_field.shape[0])
    h = static(normal_field.shape[1])
    offset = static(normal_field.snode.ptr.offset if len(normal_field.snode.ptr.offset) != 0 else [0, 0])
    i_offset = static(offset[0])
    j_offset = static(offset[1])

    for i, j in ndrange(w, h):
        normal_field[i + i_offset, j + j_offset] = vk_arr[(h - 1 - j) * w + i]


@kernel
def clear_gradients(_vars: template()):
    for I in grouped(ScalarField(Expr(_vars[0]))):
        for s in static(_vars):
            ScalarField(Expr(s))[I] = ops.cast(0, dtype=s.get_dt())


@kernel
def field_fill_python_scope(F: template(), val: template()):
    field_fill_quadrants_scope(F, val)


@kernel
def snode_deactivate(b: template()):
    for I in grouped(b):
        deactivate(b, I)


@kernel
def snode_deactivate_dynamic(b: template()):
    for I in grouped(b.parent()):
        deactivate(b, I)


# Odd-even merge sort
@kernel
def sort_stage(
    keys: template(),
    use_values: int,
    values: template(),
    N: int,
    p: int,
    k: int,
    invocations: int,
):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else 0)
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, ops.min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a + keys_offset]
                key_b = keys[b + keys_offset]
                if key_a > key_b:
                    keys[a + keys_offset] = key_b
                    keys[b + keys_offset] = key_a
                    if use_values != 0:
                        temp = values[a + values_offset]
                        values[a + values_offset] = values[b + values_offset]
                        values[b + values_offset] = temp


# Parallel Prefix Sum (Scan)
@func
def warp_shfl_up_i32(val: template()):
    global_tid = block.global_thread_idx()
    WARP_SZ = 32
    lane_id = global_tid % WARP_SZ  # type: ignore
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val


@kernel
def scan_add_inclusive(
    arr_in: template(),
    in_beg: i32,
    in_end: i32,
    single_block: template(),
    inclusive_add: template(),
):
    WARP_SZ = 32
    BLOCK_SZ = 64
    loop_config(block_dim=64)
    for i in range(in_beg, in_end):
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        block_id = int((i - in_beg) // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ

        pad_shared = block.SharedArray((65,), i32)

        val = inclusive_add(val)
        block.sync()

        # Put warp scan results to smem
        # TODO replace smem with real smem when available
        if thread_id % WARP_SZ == WARP_SZ - 1:
            pad_shared[warp_id] = val  # type: ignore
        block.sync()

        # Inter-warp scan, use the first thread in the first warp
        if warp_id == 0 and lane_id == 0:
            for k in range(1, BLOCK_SZ / WARP_SZ):  # type: ignore
                pad_shared[k] += pad_shared[k - 1]  # type: ignore
        block.sync()

        # Update data with warp sums
        warp_sum = 0
        if warp_id > 0:
            warp_sum = pad_shared[warp_id - 1]  # type: ignore
        val += warp_sum
        arr_in[i] = val

        # Update partial sums except the final block
        if not single_block and (thread_id == BLOCK_SZ - 1):
            arr_in[in_end + block_id] = val


@kernel
def uniform_add(arr_in: template(), in_beg: i32, in_end: i32):
    BLOCK_SZ = 64
    loop_config(block_dim=64)
    for i in range(in_beg + BLOCK_SZ, in_end):
        block_id = int((i - in_beg) // BLOCK_SZ)
        arr_in[i] += arr_in[in_end + block_id - 1]


@kernel
def blit_from_field_to_field(dst: template(), src: template(), offset: i32, size: i32):
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else 0)
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else 0)
    for i in range(size):
        dst[i + dst_offset + offset] = src[i + src_offset]
