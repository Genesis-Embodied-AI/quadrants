# type: ignore
"""Lift primitive members of ``@qd.data_oriented(template_primitives=False)`` template args into runtime scalar kernel
args at AST-build time.

Lives alongside ``ndarray_arg_resolver.py`` / ``checkpoint_transformer.py`` so the central
``function_def_transformer.py`` file doesn't have to grow per-feature (the same rationale the ndarray resolver
documents). ``FunctionDefTransformer.transform`` delegates here right after ``_predeclare_struct_ndarrays``; this is
the only entry point.

See ``docs/source/user_guide/compound_types.md`` (section "Runtime primitives: ``template_primitives=False``") for the
user-facing surface and ``perso_hugh/doc/data_oriented_template_primitives.md`` for the design.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang import expr, impl
from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext
from quadrants.lang.util import (
    is_data_oriented,
    is_dataclass_instance,
    wants_runtime_primitives,
)
from quadrants.types import annotations


def predeclare_struct_primitives(ctx: ASTTransformerFuncContext) -> None:
    """For each templated arg whose root class is decorated ``@qd.data_oriented(template_primitives=False)``, walk
    the reachable attribute graph and declare every primitive (``int`` / ``float`` / ``bool``) member as a runtime
    scalar kernel arg, instead of letting ``build_Attribute`` bake it as a compile-time constant. The arg-load
    ``Expr`` is cached on the global context keyed by ``(id(parent_obj), attr_name)`` for ``build_Attribute`` to
    substitute, and a ``(arg_id, template_arg_idx, attr_chain, kind)`` tuple is recorded for the launch path to
    bind the live value.

    Mirrors ``_predeclare_struct_ndarrays``. Primitive leaves have no stable identity (``id(5)`` is interned), so
    the cache is keyed on the parent container's identity plus the attribute name. Only plain Python ``int`` /
    ``float`` / ``bool`` are lifted (matched by exact type so numpy scalars etc. keep their previous baked
    behaviour). dtype follows the runtime defaults (``default_ip`` / ``default_fp``), matching how a baked Python
    constant would be typed.

    ``_seen`` guards against attribute-graph cycles (e.g. ``sim.solver.sim is sim``), same as the ndarray walker.
    """
    assert ctx.py_args is not None
    # Cheap early-out for the common case (no flagged template arg): the default ``template_primitives=True`` path
    # never builds a walk, so genesis-style data_oriented kernels pay only one ``wants_runtime_primitives`` check
    # per template arg at compile time, and nothing at launch.
    flagged: list[tuple[int, Any]] = []
    for i, arg_meta in enumerate(ctx.func.arg_metas):
        anno = arg_meta.annotation
        is_template = anno is annotations.template or isinstance(anno, annotations.template)
        is_tensor_anno = anno is _TensorClass
        if not (is_template or is_tensor_anno):
            continue
        val = ctx.py_args[i]
        if isinstance(val, _TensorClass):
            val = val._unwrap()
        if wants_runtime_primitives(val):
            flagged.append((i, val))
    if not flagged:
        return

    from quadrants._lib import core as _qd_core  # pylint: disable=C0415
    from quadrants.lang.util import cook_dtype  # pylint: disable=C0415
    from quadrants.types.utils import is_signed  # pylint: disable=C0415

    runtime = impl.get_runtime()
    default_ip = runtime.default_ip
    default_fp = runtime.default_fp
    provenance = ctx.global_context.struct_primitive_provenance
    expr_cache = ctx.global_context.struct_primitive_to_expr
    launch_info = ctx.global_context.struct_primitive_launch_info
    pruning = ctx.global_context.pruning
    # Lifted primitives are tracked under the kernel's func id (rather than the accessing function's), so a
    # primitive accessed only inside an inlined ``@qd.func`` is still declared, and so the used set is captured by
    # the existing ``KERNEL_FUNC_ID`` fastcache serialisation. Flat names are globally unique, so this never
    # collides with dataclass-field pruning.
    func_id = pruning.KERNEL_FUNC_ID

    def _register_primitive(parent, attr_name, value, arg_idx, attr_chain):
        key = (id(parent), attr_name)
        if key in provenance:
            return
        value_type = type(value)
        if value_type is bool or value_type is int:
            dtype, kind = default_ip, ("i" if is_signed(default_ip) else "u")
        elif value_type is float:
            dtype, kind = default_fp, "f"
        else:
            return
        flat_name = f"__qd_doprim_{arg_idx}__{'__'.join(attr_chain)}"
        provenance[key] = (flat_name, arg_idx, attr_chain, kind)
        # Declare a real kernel arg only in the enforcing pass, and only for primitives the body actually accessed
        # (recorded via ``mark_used`` in ``build_Attribute`` during the discovery pass). This keeps the kernel arg
        # count proportional to what is used rather than to every reachable scalar (a Genesis solver has hundreds),
        # avoiding the MAX_ARG_NUM limit. The discovery pass declares nothing; its IR is discarded.
        if not pruning.enforcing or not pruning.is_used(func_id, flat_name):
            return
        cooked = cook_dtype(dtype)
        # ``insert_scalar_param`` returns an arg-id vector (like ``insert_ndarray_param``); ``make_arg_load_expr``
        # consumes the vector, while the launch path needs the integer slot ``arg_id[0]``.
        arg_id = impl.get_runtime().compiling_callable.insert_scalar_param(cooked, flat_name)
        argload_di = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
        expr_cache[key] = expr.Expr(
            _qd_core.make_arg_load_expr(arg_id, cooked, False, create_load=True, dbg_info=argload_di)
        )
        launch_info.append((arg_id[0], arg_idx, attr_chain, kind))

    def _walk(obj, arg_idx, path, seen):
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if is_dataclass_instance(obj):
            items = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
        elif hasattr(obj, "__dict__"):
            items = list(vars(obj).items())
        else:
            return
        for attr_name, attr_val in items:
            if isinstance(attr_val, _TensorClass):
                attr_val = attr_val._unwrap()
            attr_type = type(attr_val)
            if attr_type is bool or attr_type is int or attr_type is float:
                _register_primitive(obj, attr_name, attr_val, arg_idx, (*path, attr_name))
            elif is_dataclass_instance(attr_val) or is_data_oriented(attr_val):
                _walk(attr_val, arg_idx, (*path, attr_name), seen)

    for arg_idx, val in flagged:
        _walk(val, arg_idx, (), set())
