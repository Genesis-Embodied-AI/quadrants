"""Resolve kernel-input members by their containing object's identity."""

from typing import TYPE_CHECKING, Any, NamedTuple

from quadrants.lang.exception import QuadrantsSyntaxError

if TYPE_CHECKING:
    from quadrants.lang.ast.ast_transformer_utils import FuncCompileContext


class MemberBinding(NamedTuple):
    """A member's pruning name, declared type, and result of compiling an access."""

    name: str
    annotation: Any
    # Result of compiling a member access: an Expr for a runtime scalar,
    # an array representation, a Python constant, or a nested dataclass instance.
    member_access_result: Any
    runtime_scalar: bool = False


def resolve_bound_member(ctx: "FuncCompileContext", parent: Any, attr: str) -> Any:
    """Read a registered member and record its kernel input as used during discovery."""
    binding = ctx.kernel_compile_context.member_bindings.get((id(parent), attr))
    if binding is None:
        if id(parent) in ctx.kernel_compile_context.bound_dataclass_ids:
            raise QuadrantsSyntaxError(f"Member {attr!r} is absent from the declared dataclass fields")
        return None

    if binding.runtime_scalar and ctx.is_in_static_scope():
        raise QuadrantsSyntaxError(
            f"'{attr}' is a runtime kernel argument and cannot be used inside qd.static(). "
            "Use template_primitives=True for compile-time data-oriented scalars."
        )
    pruning = ctx.kernel_compile_context.pruning
    if not pruning.enforcing:
        pruning.mark_used(pruning.KERNEL_FUNC_ID, binding.name)
    return binding.member_access_result
