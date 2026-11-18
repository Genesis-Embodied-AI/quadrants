"""Resolve kernel-input members by their containing object's identity."""

from typing import TYPE_CHECKING, Any, NamedTuple

from quadrants.lang.exception import QuadrantsSyntaxError

if TYPE_CHECKING:
    from quadrants.lang.ast.ast_transformer_utils import FuncCompilePassContext


class MemberBinding(NamedTuple):
    """A member's pruning name, declared type, and result of compiling an access."""

    name: str
    annotation: Any
    # Result of compiling a member access: an Expr for a runtime scalar,
    # an array representation, a Python constant, or a nested dataclass instance.
    member_access_result: Any
    # Reject every static-scope read of this member.
    reject_in_static_scope: bool = False


def resolve_bound_member(func_ctx: "FuncCompilePassContext", parent: Any, attr: str) -> Any:
    """Read a registered member and record its kernel input as used during discovery."""
    binding = func_ctx.kernel_compile_context.current_pass.binding_by_member.get((id(parent), attr))
    if binding is None:
        if id(parent) in func_ctx.kernel_compile_context.current_pass.bound_dataclass_ids:
            raise QuadrantsSyntaxError(f"Member {attr!r} is absent from the declared dataclass fields")
        return None

    if binding.reject_in_static_scope and func_ctx.is_in_static_scope():
        raise QuadrantsSyntaxError(
            f"'{attr}' is a runtime kernel argument and cannot be used inside qd.static(). "
            "Use template_primitives=True for compile-time data-oriented scalars."
        )
    pruning = func_ctx.kernel_compile_context.pruning
    if not pruning.enforcing:
        pruning.mark_used(pruning.KERNEL_FUNC_ID, binding.name)
    return binding.member_access_result
