# type: ignore
"""Discover dataclass roots and bind opted-in data-oriented runtime scalars."""

from typing import Any, NamedTuple

from quadrants._lib import core as _qd_core
from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang import expr, impl
from quadrants.lang.ast.ast_transformer_utils import (
    FuncCompilePassContext,
    MemberBinding,
)
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import (
    cook_dtype,
    is_data_oriented,
    is_dataclass_instance,
    uses_template_primitives,
)
from quadrants.types import annotations
from quadrants.types.utils import is_signed


class _ScalarMember(NamedTuple):
    """A scalar value and the attribute route locating it from a kernel argument."""

    parent: Any
    attr_name: str
    value: bool | int | float
    arg_idx: int
    attr_chain: tuple[str, ...]


def _collect_runtime_members_in_data_oriented(
    py_obj: Any,
    arg_idx: int,
    path: tuple[str, ...],
    seen_ids: set[int],
    scalar_members: list[_ScalarMember],
    data_oriented_template_primitives: bool,
    dataclass_members: list[tuple[int, tuple[str, ...], Any]],
) -> None:
    """Walk stored data-oriented attributes, collecting runtime scalars and stopping at dataclass roots."""
    obj_id = id(py_obj)
    if obj_id in seen_ids:
        return

    seen_ids.add(obj_id)
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
        if not data_oriented_template_primitives and attr_type in (bool, int, float):
            scalar_members.append(_ScalarMember(py_obj, attr_name, attr_val, arg_idx, (*path, attr_name)))
        elif is_dataclass_instance(attr_val):
            if id(attr_val) not in seen_ids:
                seen_ids.add(id(attr_val))
                dataclass_members.append((arg_idx, (*path, attr_name), attr_val))
        elif is_data_oriented(attr_val):
            _collect_runtime_members_in_data_oriented(
                attr_val,
                arg_idx,
                (*path, attr_name),
                seen_ids,
                scalar_members,
                data_oriented_template_primitives,
                dataclass_members,
            )


def collect_runtime_members_from_data_oriented_args(
    py_args: tuple[Any, ...], arg_metas: list[ArgMetadata]
) -> tuple[list[_ScalarMember], list[tuple[int, tuple[str, ...], Any]]]:
    """Return runtime scalars and dataclass roots found beneath data-oriented arguments."""
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
    dataclass_members: list[tuple[int, tuple[str, ...], Any]] = []
    for arg_idx, val in owners:
        _collect_runtime_members_in_data_oriented(
            val, arg_idx, (), set(), scalar_members, uses_template_primitives(val), dataclass_members
        )
    return scalar_members, dataclass_members


def bind_data_oriented_runtime_scalars(func_ctx: FuncCompilePassContext, scalar_members: list[_ScalarMember]) -> None:
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

    for parent, attr_name, value, arg_idx, attr_chain in scalar_members:
        # Interned scalar values can be shared by unrelated members.
        key = (id(parent), attr_name)
        if key in bindings:
            continue
        value_type = type(value)
        if value_type is bool or value_type is int:
            dtype, kind = runtime.default_ip, ("i" if is_signed(runtime.default_ip) else "u")
        elif value_type is float:
            dtype, kind = runtime.default_fp, "f"
        else:
            continue
        flat_name = f"__qd_doprim_{arg_idx}__{'__'.join(attr_chain)}"
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
