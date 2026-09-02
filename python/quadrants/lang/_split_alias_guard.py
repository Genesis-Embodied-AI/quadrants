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
        _warn_split_fallback(kernel)
        no_split = compile_no_split_variant(kernel, key, t_kernel, args)
        kernel._compiled_no_split_by_key[key] = no_split
    kernel.per_offload_cache_observations = PerOffloadCacheObservations()
    return no_split


def _warn_split_fallback(kernel: Any) -> None:
    """One-shot-per-specialization notice that a launch left the per-construct split cache for whole-kernel
    compilation. Fired only when the fallback variant is first built, so it never spams the hot launch path;
    suppressible through the standard logging level (`is_logging_effective("warn")`)."""
    from quadrants.lang.util import warning  # pylint: disable=C0415

    name = getattr(getattr(kernel, "func", None), "__name__", "<kernel>")
    warning(
        f"[PER_OFFLOAD][FALLBACK] Kernel '{name}': launch arguments alias, or their buffers could not be verified as "
        "disjoint (e.g. a raw NumPy/torch array passed directly), so this call uses whole-kernel compilation instead "
        "of the per-construct split cache. Results are correct, but an edit then recompiles the whole kernel rather "
        "than reusing the unchanged offloads. See docs/source/user_guide/kernel_caching.md.",
        print_stack=False,
    )


def compile_no_split_variant(kernel: Any, key: Any, t_kernel: Any, py_args: tuple[Any, ...]):
    """Compile the whole-kernel (split-disabled) variant used by the launch-time no-alias guard.

    A normally-materialized key's ``t_kernel`` already carries the lowered body, so it compiles directly. A
    fastcache-restored key (``_parse_only_keys``) never built one, so rebuild it here with the SAME two-pass pruning
    ``Kernel.materialize`` uses for a fresh compile: the non-enforcing discovery pass fills each callee ``@qd.func``'s
    used set, without which a callee dataclass arg's fields would be pruned and the enforcing build would fail. The
    cached kernel-root used set alone (materialize's fastcache shortcut) is not enough because it never enters a callee
    body. Only reached on an actual aliased launch, so this rebuild stays off the common path.
    """
    from quadrants.lang import impl  # pylint: disable=C0415
    from quadrants.lang._pruning import Pruning  # pylint: disable=C0415
    from quadrants.lang.kernel import ASTGenerator  # pylint: disable=C0415

    prog = impl.get_runtime().prog
    compiled = t_kernel
    if key in kernel._parse_only_keys:
        runtime = impl.get_runtime()
        instance_id, arg_features = kernel.mapper.lookup(kernel.raise_on_templated_floats, py_args)
        fallback_name = f"{kernel.func.__name__}_c{kernel.kernel_counter}_{instance_id}_nosplit"
        # The discovery pass rebuilds `graph_do_while_levels` in place (build_While appends as it walks); snapshot and
        # restore so the fastcache key's own launches, which keep using the restored table, stay untouched.
        saved_graph_do_while_levels = list(kernel.graph_do_while_levels)
        pruning = Pruning(kernel_used_parameters=None)
        with kernel.runtime.compilation_lock:
            for _pass in range(0, 2):
                if _pass >= 1:
                    pruning.enforce()
                tree, ctx = kernel.get_tree_and_ctx(
                    pass_idx=_pass,
                    py_args=py_args,
                    template_slot_locations=kernel.template_slot_locations,
                    arg_features=arg_features,
                    current_kernel=kernel,
                    pruning=pruning,
                    currently_compiling_materialize_key=key,
                )
                runtime._current_global_context = ctx.global_context
                built = prog.create_kernel(
                    ASTGenerator(
                        ctx=ctx,
                        kernel_name=fallback_name,
                        current_kernel=kernel,
                        only_parse_function_def=False,
                        tree=tree,
                        dump_ast=False,
                    ),
                    fallback_name,
                    kernel.autodiff_mode,
                )
                if _pass == 0:
                    pruning.propagate_fixpoint()
                    for used_parameters in pruning.used_vars_by_func_id.values():
                        collapsed: set[str] = set()
                        for param in used_parameters:
                            split_param = param.split("__qd_")
                            for i in range(len(split_param), 1, -1):
                                joined = "__qd_".join(split_param[:i])
                                if joined in collapsed:
                                    break
                                collapsed.add(joined)
                        used_parameters.clear()
                        used_parameters.update(collapsed)
                else:
                    compiled = built
                runtime._current_global_context = None
        kernel.graph_do_while_levels = saved_graph_do_while_levels
    return prog.compile_kernel(prog.config(), prog.get_device_caps(), compiled, disable_split=True).compiled_kernel_data
