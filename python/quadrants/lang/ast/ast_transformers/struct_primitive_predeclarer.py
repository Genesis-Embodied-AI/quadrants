# type: ignore
"""Bind nested dataclass scalars and opted-in data-oriented primitives as runtime kernel arguments."""

import dataclasses
from typing import Any

from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang import expr, impl
from quadrants.lang._final.final_annotation import is_final_annotation
from quadrants.lang._runtime_typing.deferred_errors import DeferredMemberType
from quadrants.lang._runtime_typing.nested_dataclasses import get_member_at_path
from quadrants.lang._runtime_typing.runtime_dataclass import infer_dataclass_type
from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.util import (
    is_data_oriented,
    is_dataclass_instance,
    wants_runtime_primitives,
)
from quadrants.types import annotations


def predeclare_struct_primitives(ctx: ASTTransformerFuncContext) -> None:
    """Declare used scalar members and record their attribute paths for binding at each launch.

    Dataclasses use their inferred schemas, preserving Final constants and deferred member errors. Other data-oriented
    members are lifted only under template_primitives=False, using the default numeric types. Attribute accesses are
    identified by their parent object and member name, since scalar identities can be shared.
    """
    assert ctx.py_args is not None
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
    nested_schemas = getattr(ctx.func, "_runtime_nested_dataclass_schemas", ())
    for arg_idx, path, _schema in nested_schemas:
        for length in range(len(path) + 1):
            owner = get_member_at_path(ctx.py_args[arg_idx], path[:length])
            ctx.global_context.runtime_dataclass_binding_objects.add(id(owner))
        parent = get_member_at_path(ctx.py_args[arg_idx], path[:-1])
        ctx.global_context.runtime_dataclass_members.add((id(parent), path[-1]))
    if not flagged and not nested_schemas:
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
        inferred_dtype: DataTypeCxx | None = None,
    ) -> None:
        """Declare a used scalar and remember how to read its value at launch."""
        key = (id(parent), attr_name)
        flat_name = f"__qd_doprim_{arg_idx}__{'__'.join(attr_chain)}"
        if key in provenance:
            if is_dataclass_instance(parent) and provenance[key][1:3] != (arg_idx, attr_chain):
                errors = getattr(ctx.global_context, "nested_dataclass_errors", None)
                if errors is None:
                    errors = ctx.global_context.nested_dataclass_errors = {}
                errors[key] = DeferredMemberType(
                    QuadrantsRuntimeTypeError,
                    "A runtime dataclass scalar alias has multiple attribute paths and is unsupported. "
                    "Use distinct dataclass instances for independently rebound attributes.",
                ).bind(".".join((ctx.func.arg_metas[arg_idx].name, *attr_chain)))
                # A source-cache hit parses only argument declarations, so the body cannot raise this error.
                if pruning.enforcing and (
                    pruning.is_used(func_id, provenance[key][0]) or pruning.is_used(func_id, flat_name)
                ):
                    errors[key].reject()
            return
        value_type = type(value)
        if inferred_dtype is not None:
            dtype = inferred_dtype
            kind = "f" if is_real(dtype) else ("i" if is_signed(dtype) else "u")
        elif value_type is bool or value_type is int:
            dtype, kind = default_ip, ("i" if is_signed(default_ip) else "u")
        elif value_type is float:
            dtype, kind = default_fp, "f"
        else:
            return
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

    def _walk(
        obj: Any,
        arg_idx: int,
        path: tuple[str, ...],
        seen: set[int],
        schema: type | None = None,
        lift_plain: bool = False,
    ) -> None:
        """Register scalar members while preserving inferred schemas and avoiding object cycles."""
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if is_dataclass_instance(obj):
            ctx.global_context.runtime_dataclass_binding_objects.add(obj_id)
            schema = schema or infer_dataclass_type(obj, ".".join((ctx.func.arg_metas[arg_idx].name, *path)))
            for field in dataclasses.fields(schema):
                value = getattr(obj, field.name)
                if isinstance(field.type, DeferredMemberType):
                    errors = getattr(ctx.global_context, "nested_dataclass_errors", None)
                    if errors is None:
                        errors = ctx.global_context.nested_dataclass_errors = {}
                    errors[(id(obj), field.name)] = field.type.bind(
                        ".".join((ctx.func.arg_metas[arg_idx].name, *path, field.name))
                    )
                elif is_final_annotation(field.type):
                    continue
                elif isinstance(field.type, DataTypeCxx):
                    _register_primitive(obj, field.name, value, arg_idx, (*path, field.name), field.type)
                elif is_dataclass_instance(value) or is_data_oriented(value):
                    child_schema = field.type if is_dataclass_instance(value) else None
                    _walk(value, arg_idx, (*path, field.name), seen, child_schema, lift_plain)
            seen.remove(obj_id)
            return
        if hasattr(obj, "__dict__"):
            items = list(vars(obj).items())
        else:
            seen.remove(obj_id)
            return
        for attr_name, attr_val in items:
            if isinstance(attr_val, _TensorClass):
                attr_val = attr_val._unwrap()
            attr_type = type(attr_val)
            if lift_plain and attr_type in (bool, int, float):
                _register_primitive(obj, attr_name, attr_val, arg_idx, (*path, attr_name))
            elif is_dataclass_instance(attr_val) or is_data_oriented(attr_val):
                _walk(attr_val, arg_idx, (*path, attr_name), seen, lift_plain=lift_plain)
        seen.remove(obj_id)

    for arg_idx, val in flagged:
        _walk(val, arg_idx, (), set(), lift_plain=True)

    for arg_idx, path, schema in nested_schemas:
        _walk(get_member_at_path(ctx.py_args[arg_idx], path), arg_idx, path, set(), schema)
