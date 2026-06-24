# type: ignore
"""Resolve an ndarray-referencing AST expression to its flat C++ kernel arg-id at AST-build time.

Lives alongside ``checkpoint_transformer.py`` / ``call_transformer.py`` so the central ``ast_transformer.py`` file
doesn't have to grow per-feature. Both ``qd.checkpoint(yield_on=...)`` (via
``CheckpointTransformer.build_checkpoint_with``) and ``qd.graph_do_while(...)`` (via ``ASTTransformer.build_While``)
share this helper: each form takes a control-flag / counter ndarray argument that may be a bare kernel parameter
(e.g. ``flag``), a ``@qd.data_oriented`` member ndarray (e.g. ``self.flag``), or a ``@dataclasses.dataclass``
parameter member (e.g. ``params.flag``). All three flatten to the same ``ExternalTensorExpression`` after AST build,
so resolving the arg-id here -- once, at kernel build time -- means the runtime launch path can forward it directly
with no per-launch name matching (which was the original ``_checkpoint_helpers`` approach and is incompatible with
member-ndarray flattening, since the flattened name is synthesised).

See ``docs/source/user_guide/graph.md`` for the user-facing surface and ``perso_hugh/doc/qipc/reentrant.md`` for the
design.
"""

from __future__ import annotations

import ast

from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext
from quadrants.lang.exception import QuadrantsSyntaxError


def resolve_ndarray_kernel_arg_id(
    ctx: ASTTransformerFuncContext,
    kernel,
    node: ast.expr,
    usage: str,
) -> tuple[str, int]:
    """Resolve ``node`` to ``(label, flat_cpp_arg_id)`` at AST-build time.

    Shared between ``qd.checkpoint(yield_on=...)`` and ``qd.graph_do_while(...)`` to turn the control-flag argument
    into the flat C++ arg-id the runtime matches against. ``node`` is an ``ast.Name`` (a bare kernel parameter,
    e.g. ``flag``) or an ``ast.Attribute`` chain (e.g. ``self.flag`` for a ``@qd.data_oriented`` owner, or
    ``params.flag`` where ``params`` is a ``@dataclasses.dataclass`` kernel parameter). We build the expression
    through the normal AST machinery and read the arg-id off the resulting external-tensor expression -- this
    unifies the bare-param and member-ndarray cases, since both flatten to a real ndarray kernel argument carrying
    its arg-id on the ``ExternalTensorExpression``.

    ``usage`` is the call form (e.g. ``"qd.checkpoint(yield_on=...)"``) used in the error message. Raises
    ``QuadrantsSyntaxError`` if the expression does not resolve to an ndarray kernel argument.
    """
    # Local imports to avoid an ast_transformers -> ast_transformer / any_array import cycle at module load:
    # ``ast_transformer`` is the central transformer module that imports ``checkpoint_transformer`` (sibling of
    # this file), and ``any_array`` pulls in core ndarray bindings that aren't needed for module import.
    # pylint: disable-next=C0415,import-outside-toplevel
    from quadrants.lang.ast.ast_transformer import _qd_core, build_stmt

    # pylint: disable-next=C0415,import-outside-toplevel
    from quadrants.lang.any_array import AnyArray

    label = ast.unparse(node)
    bad = QuadrantsSyntaxError(
        f"{usage} got {label!r} which does not resolve to an ndarray kernel parameter of "
        f"{kernel.func.__name__!r}. The argument must reference an ndarray kernel parameter (e.g. "
        f"`flag`) or a @qd.data_oriented member ndarray (e.g. `self.flag`); other expressions are not "
        f"supported."
    )
    try:
        built = build_stmt(ctx, node)
    except Exception as e:  # noqa: BLE001 - any resolution failure is a user-facing misuse
        raise bad from e
    resolved_expr = built.ptr if isinstance(built, AnyArray) else built
    if not (hasattr(resolved_expr, "is_external_tensor_expr") and resolved_expr.is_external_tensor_expr()):
        raise bad
    arg_id = _qd_core.get_external_tensor_arg_id(resolved_expr)
    if not arg_id:
        raise bad
    return label, int(arg_id[0])
