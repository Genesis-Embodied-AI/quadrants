# type: ignore
"""AST recognition, validation, and lowering for ``qd.graph_parallel_context()`` / ``qd.graph_parallel()`` blocks.

Lives alongside ``checkpoint_transformer.py`` / ``function_def_transformer.py`` so that ``ast_transformer.py`` doesn't
have to grow per-feature. ``ASTTransformer.build_With`` forwards ``qd.graph_parallel_context()`` regions and their
``qd.graph_parallel()`` sections into the static methods here.

A ``qd.graph_parallel_context()`` region tags its body with a per-kernel region id (via
``begin/end_graph_parallel_context``) and each ``qd.graph_parallel()`` section inside lowers to a stream-parallel group
(via ``begin/end_stream_parallel``). The graph builder forks the distinct groups of one region in a contiguous run and
joins them; the region id keeps two back-to-back regions apart (each gets its own join). See
``docs/source/user_guide/graph.md`` for the user-facing surface.
"""

from __future__ import annotations

import ast

from quadrants.lang.ast.ast_transformer_utils import (
    ASTTransformerFuncContext,
    get_decorator,
)
from quadrants.lang.ast.ast_transformers.checkpoint_transformer import (
    CheckpointTransformer,
)
from quadrants.lang.ast.ast_transformers.function_def_transformer import (
    FunctionDefTransformer,
)
from quadrants.lang.exception import QuadrantsSyntaxError


class GraphParallelTransformer:
    @staticmethod
    def is_graph_parallel_context_call(node: ast.expr) -> bool:
        """If *node* is a ``qd.graph_parallel_context()`` call return True, else False."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        is_gpc = (isinstance(func, ast.Attribute) and func.attr == "graph_parallel_context") or (
            isinstance(func, ast.Name) and func.id == "graph_parallel_context"
        )
        if not is_gpc:
            return False
        if node.args or node.keywords:
            raise QuadrantsSyntaxError("qd.graph_parallel_context() takes no arguments")
        return True

    @staticmethod
    def is_parallel_section_call(node: ast.expr) -> bool:
        """If *node* is a ``qd.graph_parallel()`` (a section) call return True, else False. The call shape is validated
        here so misuse raises at the ``with`` site rather than later."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        is_parallel_section = (isinstance(func, ast.Attribute) and func.attr == "graph_parallel") or (
            isinstance(func, ast.Name) and func.id == "graph_parallel"
        )
        if not is_parallel_section:
            return False
        if node.args or node.keywords:
            raise QuadrantsSyntaxError("qd.graph_parallel() takes no arguments")
        return True

    @staticmethod
    def build_graph_parallel_context_with(ctx: ASTTransformerFuncContext, node: ast.With, build_stmts) -> None:
        """Handles ``with qd.graph_parallel_context():`` fork/join regions.

        Validates the use-site (kernel must be graph=True, no nesting) and that the region body contains only
        ``with qd.graph_parallel():`` blocks, then walks the body. The region is bracketed with
        begin/end_graph_parallel_context() so its body carries a per-kernel region id, and each ``qd.graph_parallel``
        section inside lowers to a stream-parallel group (via begin/end_stream_parallel). The graph builder forks the
        distinct groups of one region in a contiguous run and joins them; the region id keeps two back-to-back regions
        apart (each gets its own join)."""
        if not ctx.is_kernel:
            raise QuadrantsSyntaxError("qd.graph_parallel_context() can only be used inside @qd.kernel, not @qd.func")
        kernel = ctx.global_context.current_kernel
        if kernel is None or not kernel.use_graph:
            raise QuadrantsSyntaxError("qd.graph_parallel_context() requires @qd.kernel(graph=True)")
        # A region cannot coexist with the checkpoint/resume model. Its section for-loops escape the checkpoint net:
        # CheckpointTransformer.auto_wrap_for_loops does not recurse into this `with`, and a section body rejects an
        # explicit qd.checkpoint, so every section task is emitted with checkpoint_id == -1 -- the prologue bucket that
        # runs unconditionally on every launch, ignoring yield/resume (a resume that should skip the region, or a yield
        # before it, would still run it). Wrapping the whole region in an explicit checkpoint doesn't help either: the
        # sections would then carry cp_id >= 0, which the fork/join path in GraphManager::build_level excludes
        # (checkpoint_id < 0), silently serializing them. There is no correct lowering today, so reject the combination
        # rather than miscompile it; making regions checkpoint-aware is a separate feature (see graph.md).
        if kernel.use_checkpoints:
            raise QuadrantsSyntaxError(
                "qd.graph_parallel_context() is not supported in a @qd.kernel(graph=True, checkpoints=True) kernel: a "
                "fork/join region does not participate in the checkpoint/resume model (its sections would run "
                "unconditionally on every launch, ignoring yield/resume). Use qd.graph_parallel_context() only in a "
                "non-checkpoints kernel, or express the work as ordinary qd.checkpoint() stages."
            )
        if getattr(ctx, "_in_graph_parallel_context", False):
            raise QuadrantsSyntaxError("qd.graph_parallel_context() regions cannot be nested")
        if getattr(ctx, "_in_parallel_section", False):
            raise QuadrantsSyntaxError("qd.graph_parallel_context() cannot appear inside a qd.graph_parallel() body")
        GraphParallelTransformer._validate_graph_parallel_context_body(ctx, node.body)
        ctx._in_graph_parallel_context = True
        ctx.ast_builder.begin_graph_parallel_context()
        try:
            build_stmts(ctx, node.body)
        finally:
            ctx.ast_builder.end_graph_parallel_context()
            ctx._in_graph_parallel_context = False
        return None

    @staticmethod
    def _validate_graph_parallel_context_body(ctx: ASTTransformerFuncContext, stmts: list[ast.stmt]) -> None:
        """A qd.graph_parallel_context() region body may contain only `with qd.graph_parallel():` blocks, optionally
        wrapped in compile-time `if qd.static(...)` (the optional ``qd.graph_parallel`` section pattern, e.g. qipc's
        ENABLE_EE) or `for ... in qd.static(...)` loops (generate one ``qd.graph_parallel`` section per element of a
        compile-time sequence). Docstrings / coverage probes / `pass` are allowed. Anything else (a runtime for-loop, a
        bare assignment, etc.) is a serial task that would silently fall outside any ``qd.graph_parallel`` section, so
        reject it.

        Both the `if` and `for` cases are restricted to `qd.static(...)` on purpose: a static branch/loop is resolved
        at trace time, so it lowers to literal `with qd.graph_parallel():` blocks (each gets a fresh
        stream_parallel_group_id). A *runtime* `if` would instead trace a FrontendIfStmt and a *runtime* for-loop a
        single parallel range_for, in both cases nesting the section tagging inside that task -- malformed, and silently
        dropping the fork/join. Staticness is checked with `get_decorator` (the same resolution `build_If` / `build_For`
        use) at every nesting level, so a runtime branch/loop nested under a static one is still rejected."""
        for i, stmt in enumerate(stmts):
            if FunctionDefTransformer._is_docstring(stmt, i) or FunctionDefTransformer._is_coverage_probe(stmt):
                continue
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.With) and stmt.items:
                if GraphParallelTransformer.is_parallel_section_call(stmt.items[0].context_expr):
                    continue
            if isinstance(stmt, ast.If) and get_decorator(ctx, stmt.test) == "static":
                GraphParallelTransformer._validate_graph_parallel_context_body(ctx, stmt.body)
                GraphParallelTransformer._validate_graph_parallel_context_body(ctx, stmt.orelse)
                continue
            if isinstance(stmt, ast.For) and not stmt.orelse and get_decorator(ctx, stmt.iter) == "static":
                GraphParallelTransformer._validate_graph_parallel_context_body(ctx, stmt.body)
                continue
            raise QuadrantsSyntaxError(
                "A qd.graph_parallel_context() region may contain only 'with qd.graph_parallel():' blocks "
                "(optionally inside 'if qd.static(...)' or 'for ... in qd.static(...)'). Move other work "
                f"outside the region. [offending stmt {i}: {type(stmt).__name__}]"
            )

    @staticmethod
    def build_parallel_section_with(ctx: ASTTransformerFuncContext, node: ast.With, build_stmts) -> None:
        """Handles a ``with qd.graph_parallel():`` section of a ``qd.graph_parallel_context()`` region.

        Validates that the section body is straight-line task work (no nested graph-structuring constructs), then reuses
        the stream-parallel tagging: begin_stream_parallel() assigns this ``qd.graph_parallel`` section a fresh
        ``stream_parallel_group_id`` that every for-loop in the body inherits, so the offloaded tasks carry the
        ``qd.graph_parallel`` section id all the way to the graph builder."""
        if not getattr(ctx, "_in_graph_parallel_context", False):
            raise QuadrantsSyntaxError(
                "qd.graph_parallel() can only be used directly inside a qd.graph_parallel_context() region"
            )
        GraphParallelTransformer._validate_parallel_section_body(ctx, node.body)
        ctx._in_parallel_section = True
        ctx.ast_builder.begin_stream_parallel()
        try:
            build_stmts(ctx, node.body)
        finally:
            ctx.ast_builder.end_stream_parallel()
            ctx._in_parallel_section = False
        return None

    @staticmethod
    def _is_graph_do_while_test(node: ast.expr) -> bool:
        """True if *node* is a ``qd.graph_do_while(...)`` call (the test of a ``while`` loop). Mirrors
        ``ASTTransformer._is_graph_do_while_call`` but returns a bool and lives here to avoid importing ``ASTTransformer``
        (which would be a circular import)."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Attribute) and func.attr == "graph_do_while") or (
            isinstance(func, ast.Name) and func.id == "graph_do_while"
        )

    @staticmethod
    def _validate_parallel_section_body(ctx: ASTTransformerFuncContext, stmts: list[ast.stmt]) -> None:
        """A ``qd.graph_parallel()`` section body must be straight-line task work: no nested graph-structuring construct
        (``qd.graph_do_while``, ``qd.checkpoint``, ``qd.graph_parallel_context``, or a nested ``qd.graph_parallel``
        section) anywhere inside it. See ``docs/source/user_guide/graph.md``.

        These are rejected because the CUDA graph builder's fork/join path only groups a section's *direct*,
        same-loop-level, non-checkpoint tasks: a nested ``qd.checkpoint`` stamps ``checkpoint_id >= 0`` and a nested
        ``qd.graph_do_while`` stamps a child ``graph_do_while_level_id``, so those tasks fall outside the contiguous run
        the builder forks -- the supposedly parallel section would be silently serialized / split instead of failing
        here. (A ``qd.graph_parallel_context`` nested in a section is separately caught downstream, and a nested section
        by ``begin_stream_parallel``'s guard, but catching all four here keeps the section grammar in one place and
        fails early with a clear message.) The walk uses ``ast.walk`` so a construct buried inside an inner ``for`` /
        ``if`` is still caught."""
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.While) and GraphParallelTransformer._is_graph_do_while_test(node.test):
                    raise QuadrantsSyntaxError(
                        "qd.graph_do_while() cannot appear inside a qd.graph_parallel() section; a section body must be "
                        "straight-line task work (put the qd.graph_parallel_context() region inside the "
                        "qd.graph_do_while() loop, not the other way around)"
                    )
                if isinstance(node, ast.With):
                    for item in node.items:
                        context_expr = item.context_expr
                        if CheckpointTransformer.is_checkpoint_call(context_expr, ctx.global_vars) is not None:
                            raise QuadrantsSyntaxError(
                                "qd.checkpoint() cannot appear inside a qd.graph_parallel() section; a section body "
                                "must be straight-line task work"
                            )
                        if GraphParallelTransformer.is_graph_parallel_context_call(context_expr):
                            raise QuadrantsSyntaxError(
                                "qd.graph_parallel_context() cannot appear inside a qd.graph_parallel() section"
                            )
                        if GraphParallelTransformer.is_parallel_section_call(context_expr):
                            raise QuadrantsSyntaxError(
                                "qd.graph_parallel() sections cannot be nested inside one another"
                            )
