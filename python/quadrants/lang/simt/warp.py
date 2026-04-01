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


# ---------------------------------------------------------------------------
# f64 (double) shuffle — CUB-style split into two i32 shuffles
# Mirrors NVIDIA CUB's ShuffleDown/ShuffleUp/ShuffleIndex for 64-bit types:
#   reinterpret as two uint32 words, shuffle each, reconstruct.
# See: cub/util_ptx.cuh  ShuffleDown<LOGICAL_WARP_THREADS, T>
# ---------------------------------------------------------------------------

def _split_f64(val):
    """Reinterpret f64 as two i32 words (lo, hi), mirroring CUB's approach."""
    from quadrants.lang.ops import bit_cast, cast
    from quadrants.types.primitive_types import u64, u32, i32
    bits = bit_cast(val, u64)
    lo = cast(bits, u32)
    hi = cast(bits >> cast(32, u64), u32)
    return cast(lo, i32), cast(hi, i32)


def _merge_f64(lo, hi):
    """Reconstruct f64 from two i32 words (lo, hi)."""
    from quadrants.lang.ops import bit_cast, cast
    from quadrants.types.primitive_types import u64, u32, f64
    bits = cast(cast(lo, u32), u64) | (cast(cast(hi, u32), u64) << cast(32, u64))
    return bit_cast(bits, f64)


def _cast_mask_u32(mask):
    """Ensure mask is a u32 Expr, handling Python int overflow for 0xFFFFFFFF."""
    from quadrants.lang.ops import cast
    from quadrants.types.primitive_types import u32
    if isinstance(mask, int) and mask > 0x7FFFFFFF:
        mask = mask - 0x100000000  # bit-equivalent signed int
    return cast(mask, u32)


def shfl_down_f64(mask, val, offset):
    mask = _cast_mask_u32(mask)
    lo, hi = _split_f64(val)
    lo = shfl_down_i32(mask, lo, offset)
    hi = shfl_down_i32(mask, hi, offset)
    return _merge_f64(lo, hi)


def shfl_up_f64(mask, val, offset):
    mask = _cast_mask_u32(mask)
    lo, hi = _split_f64(val)
    lo = shfl_up_i32(mask, lo, offset)
    hi = shfl_up_i32(mask, hi, offset)
    return _merge_f64(lo, hi)


def shfl_sync_f64(mask, val, offset):
    mask = _cast_mask_u32(mask)
    lo, hi = _split_f64(val)
    lo = shfl_sync_i32(mask, lo, offset)
    hi = shfl_sync_i32(mask, hi, offset)
    return _merge_f64(lo, hi)


def shfl_xor_f64(mask, val, offset):
    mask = _cast_mask_u32(mask)
    lo, hi = _split_f64(val)
    lo = shfl_xor_i32(mask, lo, offset)
    hi = shfl_xor_i32(mask, hi, offset)
    return _merge_f64(lo, hi)


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
    "shfl_up_i32",
    "shfl_up_f32",
    "shfl_down_i32",
    "shfl_down_f32",
    "shfl_xor_i32",
    "shfl_down_f64",
    "shfl_up_f64",
    "shfl_sync_f64",
    "shfl_xor_f64",
    "match_any",
    "match_all",
    "active_mask",
    "sync",
]
