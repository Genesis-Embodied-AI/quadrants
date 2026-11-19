# type: ignore
"""Bind dataclass scalars and opted-in data-oriented primitives as runtime kernel inputs."""

import dataclasses
from typing import Any, NamedTuple

from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang import expr, impl
from quadrants.lang._final.final_dataclass_fields import final_field_names
from quadrants.lang.ast.ast_transformer_utils import (
    FuncCompilePassContext,
    KernelCompilePassContext,
    MemberBinding,
)
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import (
    cook_dtype,
    is_data_oriented,
    is_dataclass_instance,
    uses_template_primitives,
)
from quadrants.types import annotations, primitive_types
from quadrants.types.utils import is_real, is_signed


class _ScalarMember(NamedTuple):
    """A scalar value and the attribute route locating it from a kernel argument."""

    parent: Any
    attr_name: str
    value: Any
    arg_idx: int
    attr_chain: tuple[str, ...]
    declared_dtype: DataTypeCxx | None = None


def _collect_scalars_in_single_data_oriented(
    py_obj: Any,
    arg_idx: int,
    path: tuple[str, ...],
    seen_ids: set[int],
    scalar_members: list[_ScalarMember],
    lift_plain: bool,
    schema_by_id: dict[int, tuple[type, str]],
) -> None:
    """Collect annotated dataclass scalars and, when lift_plain is True, plain Python scalars beneath an owner.

    Walk stored data-oriented and dataclass attributes, stopping at previously visited objects. Skip Final fields and
    do not descend into lists, tuples, or dicts.
    """
    obj_id = id(py_obj)
    if obj_id in seen_ids:
        return

    seen_ids.add(obj_id)
    if isinstance(py_obj, tuple) and not is_data_oriented(py_obj):
        for index, child in enumerate(py_obj):
            if isinstance(child, tuple) or is_dataclass_instance(child) or is_data_oriented(child):
                _collect_scalars_in_single_data_oriented(
                    child, arg_idx, (*path, index), seen_ids, scalar_members, lift_plain, schema_by_id
                )
        return
    if is_dataclass_instance(py_obj):
        entry = schema_by_id.get(obj_id)
        if entry is None:
            return
        schema, _ = entry
        final_names = final_field_names(schema)
        for field in dataclasses.fields(schema):
            value = getattr(py_obj, field.name)
            if field.name in final_names:
                continue
            annotation = field.type
            if id(annotation) in primitive_types.type_ids or isinstance(annotation, DataTypeCxx):
                scalar_members.append(
                    _ScalarMember(py_obj, field.name, value, arg_idx, (*path, field.name), cook_dtype(annotation))
                )
            elif is_dataclass_instance(value):
                _collect_scalars_in_single_data_oriented(
                    value, arg_idx, (*path, field.name), seen_ids, scalar_members, lift_plain, schema_by_id
                )
        return

    if hasattr(py_obj, "_asdict"):
        items = py_obj._asdict().items()
    elif hasattr(py_obj, "__dict__"):
        items = vars(py_obj).items()
    else:
        return

    for attr_name, attr_val in items:
        if attr_name.startswith("_qd_"):
            continue
        if isinstance(attr_val, _TensorClass):
            attr_val = attr_val._unwrap()
        attr_type = type(attr_val)
        # Only exact plain Python primitives become runtime parameters; NumPy scalars and subclasses retain their
        # compile-time behavior.
        if lift_plain and attr_type in (bool, int, float):
            scalar_members.append(_ScalarMember(py_obj, attr_name, attr_val, arg_idx, (*path, attr_name)))
        elif is_dataclass_instance(attr_val) or is_data_oriented(attr_val) or isinstance(attr_val, tuple):
            _collect_scalars_in_single_data_oriented(
                attr_val, arg_idx, (*path, attr_name), seen_ids, scalar_members, lift_plain, schema_by_id
            )


def _collect_runtime_scalars_from_data_oriented_args(
    py_args: tuple[Any, ...], arg_metas: list[ArgMetadata], schema_by_id: dict[int, tuple[type, str]]
) -> list[_ScalarMember]:
    """Collect dataclass scalars beneath data-oriented arguments and plain scalars when their owner opts in."""
    owners: list[tuple[int, Any]] = []
    for i, arg_meta in enumerate(arg_metas):
        anno = arg_meta.annotation
        is_template = anno is annotations.template or isinstance(anno, annotations.template)
        is_tensor_anno = anno is _TensorClass
        if not (is_template or is_tensor_anno):
            continue
        val = py_args[i]
        if isinstance(val, _TensorClass):
            val = val._unwrap()
        if is_data_oriented(val):
            owners.append((i, val))

    scalar_members: list[_ScalarMember] = []
    for arg_idx, val in owners:
        _collect_scalars_in_single_data_oriented(
            val, arg_idx, (), set(), scalar_members, not uses_template_primitives(val), schema_by_id
        )
    return scalar_members


def _create_runtime_scalar_bindings(func_ctx: FuncCompilePassContext, scalar_members: list[_ScalarMember]) -> None:
    """Record scalar members and create kernel parameters for those used by the compiled kernel.

    During discovery, record each member's location and scalar kind so accesses can be marked as used.

    During the enforcing pass, for each used member:

    - Append a scalar parameter to the callable's parameter list and get its index.
    - Record launch info describing how to populate that parameter from the Python object at each launch.
    - Create an expression that reads the parameter inside the compiled kernel.
    """
    runtime = impl.get_runtime()
    bindings = func_ctx.kernel_compile_context.current_pass.binding_by_member
    runtime_scalar_launch_infos = func_ctx.kernel_compile_context.current_pass.runtime_scalar_launch_infos
    pruning = func_ctx.kernel_compile_context.pruning
    # The kernel ID preserves uses inside inlined qd.func calls and includes them in fastcache pruning metadata.
    # Globally unique flat names avoid collisions with dataclass fields.
    func_id = pruning.KERNEL_FUNC_ID

    for parent, attr_name, value, arg_idx, attr_chain, declared_dtype in scalar_members:
        # Interned scalar values can be shared by unrelated members.
        key = (id(parent), attr_name)
        if key in bindings:
            continue
        value_type = type(value)
        if declared_dtype is not None:
            dtype = declared_dtype
            kind = "f" if is_real(dtype) else ("i" if is_signed(dtype) else "u")
        elif value_type is bool or value_type is int:
            dtype, kind = runtime.default_ip, ("i" if is_signed(runtime.default_ip) else "u")
        elif value_type is float:
            dtype, kind = runtime.default_fp, "f"
        else:
            continue
        flat_name = f"__qd_doprim_{arg_idx}__{'__'.join(map(str, attr_chain))}"
        # Add only members marked used during discovery, avoiding MAX_ARG_NUM for large containers.
        if not pruning.enforcing or not pruning.is_used(func_id, flat_name):
            # Discovery reads the Python scalar, while enforcement reads an Expr. Reject static reads so the two passes
            # cannot choose different branches.
            # FIXME: Represent the scalar as an Expr during discovery too, so qd.static((s.n,)) can build a tuple
            # whose element still reads the runtime value.
            bindings[key] = MemberBinding(flat_name, dtype, None, reject_in_static_scope=True)
            continue
        cooked = cook_dtype(dtype)
        # Argument loads use the full ID vector; launches use its first integer slot.
        arg_id = impl.get_runtime().compiling_callable.insert_scalar_param(cooked, flat_name)
        argload_di = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
        value_expr = expr.Expr(
            _qd_core.make_arg_load_expr(arg_id, cooked, False, create_load=True, dbg_info=argload_di)
        )
        bindings[key] = MemberBinding(flat_name, dtype, value_expr, reject_in_static_scope=True)
        runtime_scalar_launch_infos.append((arg_id[0], arg_idx, attr_chain, kind))


def register_runtime_dataclass_schema(pass_ctx: KernelCompilePassContext, obj: Any, schema: type, name: str) -> None:
    """Index stored dataclasses and their inferred member types for this pass."""
    if id(obj) in pass_ctx.runtime_schema_by_id:
        return
    pass_ctx.runtime_schema_by_id[id(obj)] = (schema, name)
    pass_ctx.runtime_dataclass_binding_objects.add(id(obj))
    for field in dataclasses.fields(schema):
        if dataclasses.is_dataclass(field.type):
            register_runtime_dataclass_schema(pass_ctx, getattr(obj, field.name), field.type, f"{name}.{field.name}")


def bind_data_oriented_runtime_scalars(func_ctx: FuncCompilePassContext) -> None:
    """Find opted-in scalar members, then create their compiler bindings."""
    assert func_ctx.py_args is not None
    pass_ctx = func_ctx.kernel_compile_context.current_pass
    for index, path, schema in func_ctx.func._launch_arg_schemas:
        obj = func_ctx.py_args[index]
        name = func_ctx.func.arg_metas[index].name
        for attr in path:
            pass_ctx.runtime_dataclass_binding_objects.add(id(obj))
            pass_ctx.runtime_dataclass_members.add((id(obj), attr))
            obj = obj[attr] if isinstance(attr, int) else getattr(obj, attr)
            name += "." + str(attr)
        register_runtime_dataclass_schema(pass_ctx, obj, schema, name)
    scalar_members = _collect_runtime_scalars_from_data_oriented_args(
        func_ctx.py_args, func_ctx.func.arg_metas, pass_ctx.runtime_schema_by_id
    )
    if scalar_members:
        _create_runtime_scalar_bindings(func_ctx, scalar_members)
