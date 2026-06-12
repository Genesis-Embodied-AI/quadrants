# type: ignore
"""AST recognition and lowering for ``qd.checkpoint(...)`` ``with`` blocks.

Lives alongside ``call_transformer.py`` / ``function_def_transformer.py`` so that ``ast_transformer.py`` doesn't have to
grow per-feature. ``ASTTransformer.build_With`` and ``ASTTransformer._is_checkpoint_call`` simply forward calls into
the static methods here.

See ``docs/source/user_guide/graph.md`` for the user-facing surface and ``perso_hugh/doc/qipc/reentrant.md`` for the
design.
"""

from __future__ import annotations

import ast

from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext
from quadrants.lang.exception import QuadrantsSyntaxError

# Sentinel name used by `_kernel_coverage.py` (`FIELD_VAR_NAME`) for the probe-tracking field. The checkpoint
# validator hard-codes the literal so coverage probes can be exempted without taking a runtime dep on the optional
# `_kernel_coverage` module (it is only imported when `QD_KERNEL_COVERAGE=1`). The two must stay in sync; if
# `FIELD_VAR_NAME` ever changes, update this constant and the corresponding test.
_KERNEL_COVERAGE_FIELD_NAME = "_qd_cov"


class CheckpointTransformer:
    @staticmethod
    def _is_coverage_probe_assign(stmt: ast.stmt) -> bool:
        """Return True iff *stmt* is the synthesized ``_qd_cov[<probe_id>] = 1`` assignment inserted by
        ``_kernel_coverage.py`` when ``QD_KERNEL_COVERAGE=1``. Keeping these out of the bare-statement rejection in
        ``build_checkpoint_with`` lets coverage CI exercise every checkpoint kernel without the user having to wrap
        the synthetic probes themselves.
        """
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return False
        tgt = stmt.targets[0]
        if not isinstance(tgt, ast.Subscript):
            return False
        return isinstance(tgt.value, ast.Name) and tgt.value.id == _KERNEL_COVERAGE_FIELD_NAME

    @staticmethod
    def is_checkpoint_call(node: ast.expr) -> tuple[bool, str | None]:
        """If *node* is a ``qd.checkpoint(...)`` call return ``(True, yield_on_arg_name)``; otherwise ``(False, None)``.
        ``yield_on_arg_name`` is ``None`` when the user wrote ``qd.checkpoint()`` with no ``yield_on`` kwarg.

        Validates the call shape (no positional args, only ``yield_on=`` as a bare ``ast.Name`` parameter) and raises
        ``QuadrantsSyntaxError`` for misuse so the user gets a clear message at the ``with`` site rather than a vague
        "not stream_parallel" error later.
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

        Validates the use-site (kernel must be ``graph=True``, no nesting, ``yield_on`` must be a kernel parameter)
        and records the checkpoint's ``yield_on`` arg on the kernel object. Walks the body transparently -- for-loops
        inside the ``with`` become normal top-level for-loops in the kernel's frontend IR. The ``cp_id`` is assigned by
        declaration order (list index in ``kernel.checkpoint_yield_on_args``).

        ``build_stmts`` is the ``ast_transformer.build_stmts`` callable injected by caller to avoid a circular import.
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

        # Reject bare top-level statements (Assign / AugAssign / AnnAssign / non-docstring Expr) at the top of the
        # checkpoint body and ask the user to wrap them in their own for-loop. The offloader's pending-serial bucket
        # loses the surrounding `checkpoint_id` and emits such statements as `serial` tasks with `cp_id == -1`, so they
        # would run unconditionally even when the checkpoint is skipped -- a silent correctness bug. Rather than
        # auto-wrapping them transparently (which hides the fact that each bare stmt becomes its own kernel / graph
        # node and surprises users when they look at the lowered IR or `prog.get_num_offloaded_tasks_on_last_call()`),
        # we surface a clear compile-time error and have the user write `for _ in range(1): <stmt>` themselves. This
        # also keeps the `qd.checkpoint` body shape uniform with the design assumption that every top-level statement
        # in a checkpoint is its own offloaded task with a known `cp_id`. Control-flow stmts (For / While / If / With /
        # Pass / Return) and the docstring slot are passed through. See `docs/source/user_guide/graph.md` for the
        # user-facing rule.
        #
        # `QD_KERNEL_COVERAGE=1` instruments every executable line in the kernel AST with a bare `_qd_cov[<id>] = 1`
        # probe assignment (see `_kernel_coverage.py`). Those probes are inserted by an earlier AST pass and would
        # otherwise trip the check at the top of every checkpoint kernel under coverage CI, so probes are explicitly
        # exempt -- the `cp_id` propagation in `quadrants/transforms/offload.cpp` (`assemble_serial_statements`)
        # ensures the synthesized coverage-probe serial task still inherits the surrounding checkpoint's `cp_id` and
        # so does not break the launcher's "last task in checkpoint" detection.
        for stmt in node.body:
            is_bare = isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            if not is_bare and isinstance(stmt, ast.Expr):
                # Any `Expr(Constant)` is a no-op (Python's docstring pattern, e.g. a leading triple-quoted string).
                # We accept it anywhere in the body rather than only at position 0 because the kernel-coverage AST
                # transformer (see `_kernel_coverage.py`) prepends a `_qd_cov[...] = 1` probe under
                # `QD_KERNEL_COVERAGE=1`, which would otherwise push the docstring to position 1 and have us flag it
                # as a bare top-level statement.
                is_constant_expr = isinstance(stmt.value, ast.Constant)
                is_bare = not is_constant_expr
            if not is_bare:
                continue
            if CheckpointTransformer._is_coverage_probe_assign(stmt):
                continue
            stmt_kind = type(stmt).__name__
            raise QuadrantsSyntaxError(
                f"qd.checkpoint() body cannot contain a bare top-level {stmt_kind} statement "
                f"(line {getattr(stmt, 'lineno', '?')}): every top-level statement in a checkpoint must be inside a "
                f"for-loop (or other control-flow construct), so the compiler can lower it as its own offloaded task "
                f"with the correct cp_id. Wrap the statement in `for _ in range(1):` to keep the original intent:\n"
                f"\n"
                f"    with qd.checkpoint():\n"
                f"        for _ in range(1):\n"
                f"            <your statement here>\n"
                f"        for i in range(arr.shape[0]):\n"
                f"            ...\n"
            )

        kernel.checkpoint_yield_on_args.append(yield_on_name)
        # Hand control to the C++ ASTBuilder so that every for-loop emitted by `build_stmts` below is tagged with this
        # checkpoint's `cp_id` on its `ForLoopConfig.checkpoint_id`. The C++ counter is the source of truth for cp_id;
        # we cross-check it against the Python list index so that a future refactor that misaligns the two surfaces
        # fires immediately.
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
