# type: ignore
"""Bind dataclass scalars and opted-in data-oriented primitives as runtime kernel inputs."""

import dataclasses
from typing import Any

from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang import expr, impl
from quadrants.lang._final.final_dataclass_fields import final_field_names
from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext
from quadrants.lang.util import (
    is_data_oriented,
    is_dataclass_instance,
    wants_runtime_primitives,
)
from quadrants.types import annotations, primitive_types


def predeclare_struct_primitives(ctx: ASTTransformerFuncContext) -> None:
    """Declare used scalars and record the attribute routes supplying their values at launch."""
    assert ctx.py_args is not None
    owners: list[tuple[int, Any]] = []
    for i, arg_meta in enumerate(ctx.func.arg_metas):
        anno = arg_meta.annotation
        is_template = anno is annotations.template or isinstance(anno, annotations.template)
        is_tensor_anno = anno is _TensorClass
        if not (is_template or is_tensor_anno):
            continue
        val = ctx.py_args[i]
        if isinstance(val, _TensorClass):
            val = val._unwrap()
        if is_data_oriented(val):
            owners.append((i, val))
    if not owners:
        return

    from quadrants._lib import core as _qd_core  # pylint: disable=C0415
    from quadrants.lang.util import cook_dtype  # pylint: disable=C0415
    from quadrants.types.utils import is_real, is_signed  # pylint: disable=C0415

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

    def _register_primitive(
        parent: Any,
        attr_name: str,
        value: Any,
        arg_idx: int,
        attr_chain: tuple[str, ...],
        declared_dtype: DataTypeCxx | None = None,
    ) -> None:
        """Declare one scalar input per member identity, shared by all aliases of that member."""
        key = (id(parent), attr_name)
        if key in provenance:
            return
        value_type = type(value)
        if declared_dtype is not None:
            dtype = declared_dtype
            kind = "f" if is_real(dtype) else ("i" if is_signed(dtype) else "u")
        elif value_type is bool or value_type is int:
            dtype, kind = default_ip, ("i" if is_signed(default_ip) else "u")
        elif value_type is float:
            dtype, kind = default_fp, "f"
        else:
            return
        flat_name = f"__qd_doprim_{arg_idx}__{'__'.join(attr_chain)}"
        provenance[key] = (flat_name, arg_idx, attr_chain, kind, declared_dtype)
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

    def _walk(obj: Any, arg_idx: int, path: tuple[str, ...], seen: set[int], lift_plain: bool) -> None:
        """Visit stored containers, using annotations for dataclass scalars and the owner flag for plain scalars."""
        obj_id = id(obj)
        if obj_id in seen:
            return

        seen.add(obj_id)
        if is_dataclass_instance(obj):
            final_names = final_field_names(type(obj))
            for field in dataclasses.fields(obj):
                value = getattr(obj, field.name)
                if field.name in final_names:
                    continue
                annotation = field.type
                if id(annotation) in primitive_types.type_ids or isinstance(annotation, DataTypeCxx):
                    _register_primitive(obj, field.name, value, arg_idx, (*path, field.name), cook_dtype(annotation))
                elif is_dataclass_instance(value):
                    _walk(value, arg_idx, (*path, field.name), seen, lift_plain)
            return

        if hasattr(obj, "_asdict"):
            items = obj._asdict().items()
        elif hasattr(obj, "__dict__"):
            items = vars(obj).items()
        else:
            return

        for attr_name, attr_val in items:
            if attr_name.startswith("_qd_"):
                continue
            if isinstance(attr_val, _TensorClass):
                attr_val = attr_val._unwrap()
            attr_type = type(attr_val)
            if lift_plain and attr_type in (bool, int, float):
                _register_primitive(obj, attr_name, attr_val, arg_idx, (*path, attr_name))
            elif is_dataclass_instance(attr_val) or is_data_oriented(attr_val):
                _walk(attr_val, arg_idx, (*path, attr_name), seen, lift_plain)

    for arg_idx, val in owners:
        _walk(val, arg_idx, (), set(), wants_runtime_primitives(val))
