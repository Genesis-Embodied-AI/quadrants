# type: ignore
"""Lift primitive members of ``@qd.data_oriented(template_primitives=False)`` template args into runtime scalar kernel
args at AST-build time.

Lives alongside ``ndarray_arg_resolver.py`` / ``checkpoint_transformer.py`` so the central
``function_def_transformer.py`` file doesn't have to grow per-feature (the same rationale the ndarray resolver
documents). ``FunctionDefTransformer.transform`` delegates here right after ``_predeclare_struct_ndarrays``; this is
the only entry point.

See ``docs/source/user_guide/compound_types.md`` (section "Runtime primitives: ``template_primitives=False``") for the
user-facing surface.
"""

import dataclasses
from typing import Any, NamedTuple

from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang import expr, impl
from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import (
    is_data_oriented,
    is_dataclass_instance,
    uses_template_primitives,
)
from quadrants.types import annotations


class _ScalarMember(NamedTuple):
    """A scalar value and the attribute route locating it from a kernel argument."""

    parent: Any
    attr_name: str
    value: bool | int | float
    arg_idx: int
    attr_chain: tuple[str, ...]


def _collect_scalars_in_single_data_oriented(
    py_obj: Any,
    arg_idx: int,
    path: tuple[str, ...],
    seen_ids: set[int],
    scalar_members: list[_ScalarMember],
) -> None:
    """Walk stored attributes of data-oriented objects and dataclasses, collecting exact Python bool, int, and float values.

    Append each scalar and its attribute route to scalar_members, including scalars the kernel does not use. Pruning later
    selects the kernel parameters. Stop at previously visited objects and do not descend into lists, tuples, or dicts.
    """
    obj_id = id(py_obj)
    if obj_id in seen_ids:
        return

    seen_ids.add(obj_id)
    if is_dataclass_instance(py_obj):
        items = [(field.name, getattr(py_obj, field.name)) for field in dataclasses.fields(py_obj)]
    elif hasattr(py_obj, "__dict__"):
        items = list(vars(py_obj).items())
    else:
        return

    for attr_name, attr_val in items:
        if isinstance(attr_val, _TensorClass):
            attr_val = attr_val._unwrap()
        attr_type = type(attr_val)
        if attr_type is bool or attr_type is int or attr_type is float:
            scalar_members.append(_ScalarMember(py_obj, attr_name, attr_val, arg_idx, (*path, attr_name)))
        elif is_dataclass_instance(attr_val) or is_data_oriented(attr_val):
            _collect_scalars_in_single_data_oriented(attr_val, arg_idx, (*path, attr_name), seen_ids, scalar_members)


def _collect_runtime_scalars_from_data_oriented_args(
    py_args: tuple[Any, ...], arg_metas: list[ArgMetadata]
) -> list[_ScalarMember]:
    """Collect scalars beneath data-oriented kernel arguments with template_primitives=False."""
    flagged: list[tuple[int, Any]] = []
    for i, arg_meta in enumerate(arg_metas):
        anno = arg_meta.annotation
        is_template = anno is annotations.template or isinstance(anno, annotations.template)
        is_tensor_anno = anno is _TensorClass
        if not (is_template or is_tensor_anno):
            continue
        val = py_args[i]
        if isinstance(val, _TensorClass):
            val = val._unwrap()
        if not uses_template_primitives(val):
            flagged.append((i, val))

    scalar_members: list[_ScalarMember] = []
    for arg_idx, val in flagged:
        _collect_scalars_in_single_data_oriented(val, arg_idx, (), set(), scalar_members)
    return scalar_members


def _create_runtime_scalar_bindings(ctx: ASTTransformerFuncContext, scalar_members: list[_ScalarMember]) -> None:
    """Record scalar routes and add the used members as kernel parameters in the enforcing pass."""
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

    for parent, attr_name, value, arg_idx, attr_chain in scalar_members:
        # Different members can share an interned scalar value, so identify the parent and attribute instead.
        key = (id(parent), attr_name)
        if key in provenance:
            continue
        value_type = type(value)
        if value_type is bool or value_type is int:
            dtype, kind = default_ip, ("i" if is_signed(default_ip) else "u")
        elif value_type is float:
            dtype, kind = default_fp, "f"
        else:
            continue
        flat_name = f"__qd_doprim_{arg_idx}__{'__'.join(attr_chain)}"
        provenance[key] = (flat_name, arg_idx, attr_chain, kind)
        # Declare a real kernel arg only in the enforcing pass, and only for primitives the body actually accessed
        # (recorded via ``mark_used`` in ``build_Attribute`` during the discovery pass). This keeps the kernel arg
        # count proportional to what is used rather than to every reachable scalar (a Genesis solver has hundreds),
        # avoiding the MAX_ARG_NUM limit. The discovery pass declares nothing; its IR is discarded.
        if not pruning.enforcing or not pruning.is_used(func_id, flat_name):
            continue
        cooked = cook_dtype(dtype)
        # ``insert_scalar_param`` returns an arg-id vector (like ``insert_ndarray_param``); ``make_arg_load_expr``
        # consumes the vector, while the launch path needs the integer slot ``arg_id[0]``.
        arg_id = impl.get_runtime().compiling_callable.insert_scalar_param(cooked, flat_name)
        argload_di = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
        expr_cache[key] = expr.Expr(
            _qd_core.make_arg_load_expr(arg_id, cooked, False, create_load=True, dbg_info=argload_di)
        )
        launch_info.append((arg_id[0], arg_idx, attr_chain, kind))


def bind_data_oriented_runtime_scalar_parameters(ctx: ASTTransformerFuncContext) -> None:
    """Find opted-in scalar members, then create their compiler bindings."""
    assert ctx.py_args is not None
    scalar_members = _collect_runtime_scalars_from_data_oriented_args(ctx.py_args, ctx.func.arg_metas)
    if scalar_members:
        _create_runtime_scalar_bindings(ctx, scalar_members)
