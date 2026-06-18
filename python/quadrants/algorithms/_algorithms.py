# type: ignore

from quadrants._kernels import (
    blit_from_field_to_field,
    scan_add_inclusive,
    sort_stage,
    subgroup_inclusive_add_warp_i32,
    uniform_add,
    warp_shfl_up_i32,
)
from quadrants.lang.impl import current_cfg, field
from quadrants.lang.kernel_impl import data_oriented
from quadrants.lang.misc import cuda, vulkan
from quadrants.lang.runtime_ops import sync
from quadrants.lang.simt import subgroup
from quadrants.types.primitive_types import i32


def parallel_sort(keys, values=None):
    """Odd-even merge sort (deprecated).

    .. deprecated::
        Prefer the LSB radix sort ``qd.algorithms.radix_sort`` (a ``@qd.kernel``) or ``qd.algorithms.radix_sort_func``
        (a ``@qd.func`` for in-kernel composition). The new API is asymptotically ``O(N log_radix N)`` rather than
        ``O(N log^2 N)``, supports ``{u32, i32, f32, u64, i64, f64}`` keys across CUDA / AMDGPU / Vulkan / Metal, and
        takes caller-supplied tmp + scratch buffers so the call stays fully async. ``parallel_sort`` is kept for one
        release cycle for backward compat and will be removed thereafter. See ``docs/source/user_guide/algorithms.md``
        for the migration recipe.

    References:
        https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
        https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
    """
    import warnings  # pylint: disable=import-outside-toplevel

    warnings.warn(
        "qd.algorithms.parallel_sort is deprecated. Use qd.algorithms.radix_sort (the @qd.kernel) or "
        "qd.algorithms.radix_sort_func (the @qd.func) instead. "
        "See docs/source/user_guide/algorithms.md for migration.",
        DeprecationWarning,
        stacklevel=2,
    )
    N = keys.shape[0]

    num_stages = 0
    p = 1
    while p < N:
        k = p
        while k >= 1:
            invocations = int((N - k - k % p) / (2 * k)) + 1
            if values is None:
                sort_stage(keys, 0, keys, N, p, k, invocations)
            else:
                sort_stage(keys, 1, values, N, p, k, invocations)
            num_stages += 1
            sync()
            k = int(k / 2)
        p = int(p * 2)


@data_oriented
class PrefixSumExecutor:
    """Parallel Prefix Sum (Scan) Helper.

    .. deprecated::
        Prefer ``qd.algorithms.exclusive_scan_add`` (a ``@qd.func`` for in-kernel composition). The new
        functional API supports ``{i32, u32, f32, i64, u64, f64}`` on every backend (CUDA, AMDGPU, Vulkan, Metal) and
        runs the exclusive variant directly. ``PrefixSumExecutor`` is inclusive-only, ``i32``-only, and limited to
        CUDA / Vulkan; it is kept for one release cycle for backward compat and will be removed thereafter. See
        ``docs/source/user_guide/algorithms.md`` for the migration recipe.

    Use this helper to perform an inclusive in-place's parallel prefix sum.

    References:
        https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
        https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
    """

    def __init__(self, length):
        import warnings  # pylint: disable=import-outside-toplevel

        warnings.warn(
            "qd.algorithms.PrefixSumExecutor is deprecated. Use "
            "qd.algorithms.exclusive_scan_add (the @qd.func) instead. "
            "See docs/source/user_guide/algorithms.md for migration.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._init(length)

    def _init(self, length):
        self.sorting_length = length

        block_sz = 64

        # Buffer position and length
        # This is a single buffer implementation for ease of aot usage
        ele_num = length
        self.ele_nums = [ele_num]
        start_pos = 0
        self.ele_nums_pos = [start_pos]

        while ele_num > 1:
            ele_num = int((ele_num + block_sz - 1) / block_sz)
            self.ele_nums.append(ele_num)
            start_pos += block_sz * ele_num
            self.ele_nums_pos.append(start_pos)

        self.large_arr = field(i32, shape=start_pos)

    def run(self, input_arr):
        length = self.sorting_length
        ele_nums = self.ele_nums
        ele_nums_pos = self.ele_nums_pos

        if input_arr.dtype != i32:
            raise RuntimeError("Only qd.i32 type is supported for prefix sum.")

        if current_cfg().arch == cuda:
            scan_primitive = warp_shfl_up_i32
        elif current_cfg().arch == vulkan:
            # `subgroup.inclusive_add_tiled` takes `(value, log2_size)`; the prefix-sum kernel passes the primitive
            # as a template callable invoked with a single argument, so use the adapter that pre-binds
            # `log2_size=5` (full 32-lane warp/wave scan).  The kernel itself hard-codes `WARP_SZ=32` for the
            # inter-warp accumulation and the shared-memory layout, so the device subgroup width *must* be exactly
            # 32 for the result to be correct.  Vulkan reports the subgroup width per physical device
            # (`VkPhysicalDeviceSubgroupProperties::subgroupSize`) -- 32 on NVIDIA, MoltenVK, and most desktop
            # drivers, but 16 on some Intel iGPUs and 64 on AMDGPU Vulkan in wave64 mode.  Fail loudly here rather
            # than silently producing wrong results.
            actual = subgroup.group_size()
            if actual != 32:
                raise RuntimeError(
                    f"PrefixSumExecutor on Vulkan requires subgroup size == 32 (got {actual}). "
                    f"The kernel hard-codes a 32-wide warp scan; running on a {actual}-wide subgroup would "
                    f"silently compute the wrong result."
                )
            scan_primitive = subgroup_inclusive_add_warp_i32
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for prefix sum.")

        blit_from_field_to_field(self.large_arr, input_arr, 0, length)

        # Kogge-Stone construction
        for i in range(len(ele_nums) - 1):
            if i == len(ele_nums) - 2:
                scan_add_inclusive(
                    self.large_arr,
                    ele_nums_pos[i],
                    ele_nums_pos[i + 1],
                    True,
                    scan_primitive,
                )
            else:
                scan_add_inclusive(
                    self.large_arr,
                    ele_nums_pos[i],
                    ele_nums_pos[i + 1],
                    False,
                    scan_primitive,
                )

        for i in range(len(ele_nums) - 3, -1, -1):
            uniform_add(self.large_arr, ele_nums_pos[i], ele_nums_pos[i + 1])

        blit_from_field_to_field(input_arr, self.large_arr, 0, length)


__all__ = ["parallel_sort", "PrefixSumExecutor"]
