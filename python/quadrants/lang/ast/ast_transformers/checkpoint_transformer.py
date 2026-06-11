# type: ignore
"""AST recognition and lowering for ``qd.checkpoint(...)`` ``with`` blocks.

Lives alongside ``call_transformer.py`` / ``function_def_transformer.py`` so that
``ast_transformer.py`` doesn't have to grow per-feature. ``ASTTransformer.build_With`` and
``ASTTransformer._is_checkpoint_call`` simply forward into the static methods here. See
``docs/source/user_guide/graph.md`` for the user-facing surface and
``perso_hugh/doc/qipc/reentrant.md`` for the design.
"""

from __future__ import annotations

import ast

from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext
from quadrants.lang.exception import QuadrantsSyntaxError


class CheckpointTransformer:
    @staticmethod
    def is_checkpoint_call(node: ast.expr) -> tuple[bool, str | None]:
        """If *node* is a ``qd.checkpoint(...)`` call return ``(True, yield_on_arg_name)``;
        otherwise ``(False, None)``. ``yield_on_arg_name`` is ``None`` when the user wrote
        ``qd.checkpoint()`` with no ``yield_on`` kwarg.

        Validates the call shape (no positional args, only ``yield_on=`` as a bare ``ast.Name``)
        and raises ``QuadrantsSyntaxError`` for misuse so the user gets a clear message at the
        ``with`` site rather than a vague "not stream_parallel" error later.
        """
        if not isinstance(node, ast.Call):
            return False, None
        func = node.func
        is_checkpoint = (isinstance(func, ast.Attribute) and func.attr == "checkpoint") or (
            isinstance(func, ast.Name) and func.id == "checkpoint"
        )
        if not is_checkpoint:
            return False, None
        if node.args:
            raise QuadrantsSyntaxError(
                "qd.checkpoint() takes no positional arguments; use qd.checkpoint(yield_on=flag) instead"
            )
        yield_on_name: str | None = None
        for kw in node.keywords:
            if kw.arg != "yield_on":
                raise QuadrantsSyntaxError(
                    f"qd.checkpoint() got unexpected keyword argument {kw.arg!r}; only 'yield_on' is supported"
                )
            if not isinstance(kw.value, ast.Name):
                raise QuadrantsSyntaxError(
                    "qd.checkpoint(yield_on=...) must be the bare name of a kernel parameter "
                    "(e.g. `yield_on=overflow_flag`); expressions are not supported"
                )
            yield_on_name = kw.value.id
        return True, yield_on_name

    @staticmethod
    def build_checkpoint_with(
        ctx: ASTTransformerFuncContext,
        node: ast.With,
        yield_on_name: str | None,
        build_stmts,
    ) -> None:
        """Handles ``with qd.checkpoint(yield_on=arg):`` blocks.

        Validates the use-site (kernel must be ``graph=True``, no nesting, ``yield_on`` must be
        a kernel parameter) and records the checkpoint's ``yield_on`` arg on the kernel object.
        Walks the body transparently -- for-loops inside the ``with`` become normal top-level
        for-loops in the kernel's frontend IR. The ``cp_id`` is assigned by declaration order
        (list index in ``kernel.checkpoint_yield_on_args``).

        ``build_stmts`` is the ``ast_transformer.build_stmts`` callable injected by the caller
        to avoid a circular import.
        """
        if not ctx.is_kernel:
            raise QuadrantsSyntaxError("qd.checkpoint() can only be used inside @qd.kernel, not @qd.func")
        kernel = ctx.global_context.current_kernel
        if not kernel.use_graph:
            raise QuadrantsSyntaxError("qd.checkpoint() requires @qd.kernel(graph=True)")
        if getattr(ctx, "_in_checkpoint", False):
            raise QuadrantsSyntaxError(
                "qd.checkpoint() cannot be nested inside another qd.checkpoint(); checkpoints in the "
                "same kernel must be flat siblings (a checkpoint inside qd.graph_do_while is fine)"
            )
        if yield_on_name is not None:
            arg_names = [m.name for m in kernel.arg_metas]
            if yield_on_name not in arg_names:
                raise QuadrantsSyntaxError(
                    f"qd.checkpoint(yield_on={yield_on_name!r}) does not match any parameter of kernel "
                    f"{kernel.func.__name__!r}. Available parameters: {arg_names}"
                )

        # Auto-wrap bare top-level statements in the checkpoint body in a one-iteration
        # `for` loop. The offloader's pending-serial bucket loses the surrounding
        # `checkpoint_id` and emits such statements as `serial` tasks with `cp_id == -1`,
        # meaning they would run unconditionally even when the checkpoint is skipped -- a
        # silent correctness bug. The fix is to lower them as `range_for` tasks instead by
        # wrapping each bare statement in `for _ in range(1): <stmt>`. We target the specific
        # statement kinds known to hit the footgun (Assign / AugAssign / AnnAssign /
        # non-docstring Expr) and leave everything else (For, While, If, With, Pass,
        # docstring) untouched so they keep working transparently; nested
        # `with qd.checkpoint(...)` in particular still falls through to the existing
        # nested-checkpoint check at the start of this method.
        new_body: list[ast.stmt] = []
        for i, stmt in enumerate(node.body):
            needs_wrap = isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            if not needs_wrap and isinstance(stmt, ast.Expr):
                is_docstring = i == 0 and isinstance(stmt.value, ast.Constant)
                needs_wrap = not is_docstring
            if needs_wrap:
                wrapped = ast.For(
                    target=ast.Name(id="_", ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id="range", ctx=ast.Load()),
                        args=[ast.Constant(value=1)],
                        keywords=[],
                    ),
                    body=[stmt],
                    orelse=[],
                )
                ast.copy_location(wrapped, stmt)
                ast.fix_missing_locations(wrapped)
                new_body.append(wrapped)
            else:
                new_body.append(stmt)
        node.body = new_body

        kernel.checkpoint_yield_on_args.append(yield_on_name)
        # Hand control to the C++ ASTBuilder so that every for-loop emitted by `build_stmts`
        # below is tagged with this checkpoint's `cp_id` on its `ForLoopConfig.checkpoint_id`.
        # The C++ counter is the source of truth for cp_id; we cross-check it against the
        # Python list index so a future refactor that misaligns the two surfaces immediately.
        cpp_cp_id = ctx.ast_builder.begin_checkpoint()
        py_cp_id = len(kernel.checkpoint_yield_on_args) - 1
        assert cpp_cp_id == py_cp_id, (
            f"C++ ASTBuilder.begin_checkpoint() returned cp_id={cpp_cp_id} but Python "
            f"kernel.checkpoint_yield_on_args index expected {py_cp_id}; these counters "
            f"must stay in lockstep so the GraphManager (slice 1c) can index yield_on by cp_id"
        )
        ctx._in_checkpoint = True
        try:
            build_stmts(ctx, node.body)
        finally:
            ctx._in_checkpoint = False
            ctx.ast_builder.end_checkpoint()
        return None
