"""Helpers extracted from ``kernel.py`` for the launch-time no-alias guard on the per-construct frontend split.

The split can clear recompute-safety condition (3) by assuming two ndarray parameters address disjoint buffers --
disjointness a caller defeats by binding the same ndarray to both. ``Kernel.launch_kernel`` delegates the per-launch
alias check and whole-kernel fallback to the free functions below so the central ``Kernel`` class doesn't accrete this
feature. See ``doc/per_offload_cache/split_alias_guard_design.md`` for the full model.
"""

from __future__ import annotations

from typing import Any

from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES

from ._kernel_types import PerOffloadCacheObservations


def _launch_alloc_id(nd: Any):
    """Backing ``DeviceAllocation`` identity of a launch ndarray, or ``None`` when it can't be read.

    The launcher derives each arg's device pointer from ``alloc_id``, so that (not the wrapper address) is the identity
    that decides aliasing in the compiled kernel. ``None`` (unrecognized shape / unreadable id) means "cannot verify",
    which the caller treats as a possible alias -> whole-kernel fallback (correct, just slower).
    """
    if nd is None:
        return None
    if type(nd) in _TENSOR_WRAPPER_TYPES:
        nd = nd._unwrap()
    arr = getattr(nd, "arr", None)
    if arr is None:
        return None
    try:
        return arr.device_alloc_id()
    except Exception:  # noqa: BLE001
        return None


def launch_has_aliased_ndarrays(kernel: Any, key: Any, args: tuple) -> bool:
    """True iff some arg-slot pair the split assumed disjoint actually shares one backing device allocation.

    Only called for keys whose split recorded such pairs (see ``select_launch_variant``). Aliases *outside* the
    recorded pairs -- e.g. the same buffer bound to a param the split never recomputed-across -- do not fall back,
    because the split never depended on those args being disjoint. A slot the guard cannot resolve to a readable
    ``alloc_id`` is treated as a possible alias (conservative: costs the fallback, never correctness).
    """
    pairs = kernel._split_alias_guard_by_key.get(key)
    if not pairs:
        return False

    # slot -> device-alloc identity, resolved lazily and only for slots that appear in a recorded pair.
    slot_to_nd: dict[int, Any] = {}
    for slot, positional_index in kernel._explicit_ndarray_slot_info_by_key.get(key) or []:
        slot_to_nd[slot] = args[positional_index]
    # Struct-member and typed-dataclass-field slots resolve the same way (root arg index + attribute chain), so the
    # guard walks both through `_resolve_struct_ndarray`. They're stored in separate tables only because the struct
    # table also drives launch-arg binding, which dataclass fields do not use.
    struct_nd_info = list(kernel._struct_ndarray_launch_info_by_key.get(key) or [])
    struct_nd_info += kernel._dataclass_ndarray_guard_info_by_key.get(key) or []
    resolved: dict[int, Any] = {}

    def _alloc_of(slot: int):
        if slot in resolved:
            return resolved[slot]
        nd = slot_to_nd.get(slot)
        if nd is None and struct_nd_info:
            for s_slot, template_arg_idx, attr_chain in struct_nd_info:
                if s_slot == slot:
                    nd = kernel._resolve_struct_ndarray(args, template_arg_idx, attr_chain)
                    break
        ident = _launch_alloc_id(nd)
        resolved[slot] = ident
        return ident

    for j in range(0, len(pairs) - 1, 2):
        a, b = pairs[j], pairs[j + 1]
        ida, idb = _alloc_of(a), _alloc_of(b)
        if ida is None or idb is None:
            return True
        if ida == idb:
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
