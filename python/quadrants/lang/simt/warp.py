# type: ignore

from quadrants.lang import impl


def all_nonzero(mask, predicate):
    return impl.call_internal("cuda_all_sync_i32", mask, predicate, with_runtime_context=False)


def any_nonzero(mask, predicate):
    return impl.call_internal("cuda_any_sync_i32", mask, predicate, with_runtime_context=False)


def unique(mask, predicate):
    return impl.call_internal("cuda_uni_sync_i32", mask, predicate, with_runtime_context=False)


def ballot(predicate):
    return impl.call_internal("cuda_ballot_i32", predicate, with_runtime_context=False)


def shfl_sync_i32(mask, val, offset):
    # lane offset is 31 for warp size 32
    return impl.call_internal("cuda_shfl_sync_i32", mask, val, offset, 31, with_runtime_context=False)


def shfl_sync_f32(mask, val, offset):
    # lane offset is 31 for warp size 32
    return impl.call_internal("cuda_shfl_sync_f32", mask, val, offset, 31, with_runtime_context=False)


def shfl_sync_f64(mask, val, offset):
    """Lossless f64 warp shuffle via two i32 shuffles.

    CUDA has no native f64 shuffle intrinsic.  This splits the double
    into two 32-bit halves, shuffles each with shfl_sync_i32, and
    recombines.
    """
    from quadrants.lang.ops import bit_cast, cast  # noqa: avoid circular import
    from quadrants.types.primitive_types import f64, i32, i64, u32, u64

    as_u64 = bit_cast(val, u64)
    lo = cast(as_u64, u32)
    hi = cast(as_u64 >> 32, u32)
    lo = shfl_sync_i32(mask, bit_cast(lo, i32), offset)
    hi = shfl_sync_i32(mask, bit_cast(hi, i32), offset)
    combined = (cast(bit_cast(hi, u32), u64) << 32) | cast(bit_cast(lo, u32), u64)
    return bit_cast(combined, f64)


def shfl_up_i32(mask, val, offset):
    # lane offset is 0 for warp size 32
    return impl.call_internal("cuda_shfl_up_sync_i32", mask, val, offset, 0, with_runtime_context=False)


def shfl_up_f32(mask, val, offset):
    # lane offset is 0 for warp size 32
    return impl.call_internal("cuda_shfl_up_sync_f32", mask, val, offset, 0, with_runtime_context=False)


def shfl_down_i32(mask, val, offset):
    # lane offset is 31 for warp size 32
    return impl.call_internal("cuda_shfl_down_sync_i32", mask, val, offset, 31, with_runtime_context=False)


def shfl_down_f32(mask, val, offset):
    # lane offset is 31 for warp size 32
    return impl.call_internal("cuda_shfl_down_sync_f32", mask, val, offset, 31, with_runtime_context=False)


def shfl_xor_i32(mask, val, offset):
    return impl.call_internal("cuda_shfl_xor_sync_i32", mask, val, offset, 31, with_runtime_context=False)


def match_any(mask, value):
    # These intrinsics are only available on compute_70 or higher
    # https://docs.nvidia.com/cuda/pdf/NVVM_IR_Specification.pdf
    if impl.get_cuda_compute_capability() < 70:
        raise AssertionError("match_any intrinsic only available on compute_70 or higher")
    return impl.call_internal("cuda_match_any_sync_i32", mask, value, with_runtime_context=False)


def match_all(mask, val):
    # These intrinsics are only available on compute_70 or higher
    # https://docs.nvidia.com/cuda/pdf/NVVM_IR_Specification.pdf
    if impl.get_cuda_compute_capability() < 70:
        raise AssertionError("match_all intrinsic only available on compute_70 or higher")
    return impl.call_internal("cuda_match_all_sync_i32", mask, val, with_runtime_context=False)


def active_mask():
    return impl.call_internal("cuda_active_mask", with_runtime_context=False)


def sync(mask):
    return impl.call_internal("warp_barrier", mask, with_runtime_context=False)


__all__ = [
    "all_nonzero",
    "any_nonzero",
    "unique",
    "ballot",
    "shfl_sync_i32",
    "shfl_sync_f32",
    "shfl_sync_f64",
    "shfl_up_i32",
    "shfl_up_f32",
    "shfl_down_i32",
    "shfl_down_f32",
    "shfl_xor_i32",
    "match_any",
    "match_all",
    "active_mask",
    "sync",
]
