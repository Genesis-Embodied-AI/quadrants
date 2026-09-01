"""Helpers extracted from ``kernel.py`` for the launch-time no-alias guard on the per-construct frontend split.

The split can clear recompute-safety condition (3) by assuming two ndarray parameters address disjoint buffers --
disjointness a caller defeats by binding the same ndarray to both. ``Kernel.launch_kernel`` delegates the per-launch
alias check and whole-kernel fallback to the free functions below so the central ``Kernel`` class doesn't accrete this
feature. See ``doc/per_offload_cache/split_alias_guard_design.md`` for the full model.
"""

from __future__ import annotations

from typing import Any

from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants.types import ndarray_type

from ._kernel_types import PerOffloadCacheObservations


def launch_has_aliased_ndarrays(kernel: Any, key: Any, args: tuple) -> bool:
    """True if two ndarray parameters of this launch may share one backing device allocation.

    Only called for keys whose split relied on cross-parameter ndarray disjointness (see ``select_launch_variant``).
    Two ndarray args alias in the compiled kernel iff their ``DeviceAllocation`` matches -- the launcher derives each
    arg's device pointer from ``alloc_id``, so that (not the wrapper address) is the identity to compare. An arg whose
    id cannot be read is treated as a possible alias, so an unrecognized argument shape falls back to the
    always-correct whole-kernel path rather than risking the split. Covers plain ndarray args and ndarrays reached
    through ``@qd.data_oriented`` struct members.
    """
    seen: set = set()

    def _collides(nd) -> bool:
        if nd is None:
            return False
        arr = getattr(nd, "arr", None)
        if arr is None:
            return True
        try:
            ident = arr.device_alloc_id()
        except Exception:
            return True
        if ident in seen:
            return True
        seen.add(ident)
        return False

    for i, meta in enumerate(kernel.arg_metas):
        if type(meta.annotation) is ndarray_type.NdarrayType:
            val = args[i]
            if type(val) in _TENSOR_WRAPPER_TYPES:
                val = val._unwrap()
            if _collides(val):
                return True
    struct_nd_info = kernel._struct_ndarray_launch_info_by_key.get(key)
    if struct_nd_info:
        for _arg_id, template_arg_idx, attr_chain in struct_nd_info:
            if _collides(kernel._resolve_struct_ndarray(args, template_arg_idx, attr_chain)):
                return True
    return False


def select_launch_variant(kernel: Any, key: Any, args: tuple, compiled_kernel_data: Any, t_kernel: Any):
    """Return the CompiledKernelData to launch, swapping in the whole-kernel variant when this call's actual buffers
    violate the split's cross-parameter disjointness assumption.

    For a guarded key whose args alias, lazily compile (once) and cache the split-disabled whole-kernel variant -- it
    never recomputes across a write, so it is correct under aliasing -- and reset the observations to the no-split
    sentinel. Disjoint calls (the common case) return ``compiled_kernel_data`` unchanged, staying on the fast split.
    """
    if not (kernel._split_alias_guard_by_key.get(key) and launch_has_aliased_ndarrays(kernel, key, args)):
        return compiled_kernel_data
    no_split = kernel._compiled_no_split_by_key.get(key)
    if no_split is None:
        no_split = kernel.compile_no_split_variant(key, t_kernel, args)
        kernel._compiled_no_split_by_key[key] = no_split
    kernel.per_offload_cache_observations = PerOffloadCacheObservations()
    return no_split
