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

        Reuses the stream-parallel tagging: begin_stream_parallel() assigns this ``qd.graph_parallel`` section a fresh
        ``stream_parallel_group_id`` that every for-loop in the body inherits, so the offloaded tasks carry the
        ``qd.graph_parallel`` section id all the way to the graph builder."""
        if not getattr(ctx, "_in_graph_parallel_context", False):
            raise QuadrantsSyntaxError(
                "qd.graph_parallel() can only be used directly inside a qd.graph_parallel_context() region"
            )
        ctx._in_parallel_section = True
        ctx.ast_builder.begin_stream_parallel()
        try:
            build_stmts(ctx, node.body)
        finally:
            ctx.ast_builder.end_stream_parallel()
            ctx._in_parallel_section = False
        return None
