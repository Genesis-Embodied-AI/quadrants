# type: ignore
"""Shared syntactic recognition of the graph-structuring kernel constructs, covering both the canonical
``qd.graph.<name>`` namespace spelling and the deprecated flat ``qd.graph_<name>`` spelling.

These constructs (``do_while``, ``parallel_context``, ``parallel``) are lowered at compile time and never actually
called, so the AST transformers recognize them purely by their source spelling. Matching both spellings lives here in
one place so the several detection sites (``ast_transformer.py``, ``graph_parallel_transformer.py``,
``function_def_transformer.py``) stay in sync. Using the deprecated flat spelling emits a ``DeprecationWarning`` at
compile time (see ``warn_if_deprecated``), which is the only place a warning can fire: the flat functions are recognized
by the AST transformer and never executed inside a kernel.
"""

from __future__ import annotations

import ast
import warnings

# Canonical attribute name under ``qd.graph`` -> deprecated flat ``qd.<name>`` spelling.
_FLAT_NAME = {
    "do_while": "graph_do_while",
    "parallel_context": "graph_parallel_context",
    "parallel": "graph_parallel",
}


def _value_is_graph(value: ast.expr) -> bool:
    """True if *value* is the ``graph`` of a ``<...>.graph.<attr>`` chain (e.g. the ``graph`` in ``qd.graph.do_while``)
    or a bare ``graph`` name (``from quadrants import graph; graph.do_while(...)``)."""
    if isinstance(value, ast.Attribute):
        return value.attr == "graph"
    if isinstance(value, ast.Name):
        return value.id == "graph"
    return False


def is_deprecated_spelling(func: ast.expr, name: str) -> bool:
    """True if *func* uses the deprecated flat ``qd.graph_<name>`` spelling (attribute ``x.graph_<name>`` or bare name
    ``graph_<name>``)."""
    flat = _FLAT_NAME[name]
    if isinstance(func, ast.Attribute) and func.attr == flat:
        return True
    if isinstance(func, ast.Name) and func.id == flat:
        return True
    return False


def matches(func: ast.expr, name: str) -> bool:
    """True if *func* refers to the construct *name*, in either the canonical ``qd.graph.<name>`` spelling or the
    deprecated flat ``qd.graph_<name>`` spelling. ``name`` is one of ``do_while`` / ``parallel_context`` / ``parallel``.
    """
    if isinstance(func, ast.Attribute) and func.attr == name and _value_is_graph(func.value):
        return True
    return is_deprecated_spelling(func, name)


def warn_if_deprecated(func: ast.expr, name: str) -> None:
    """If *func* uses the deprecated flat ``qd.graph_<name>`` spelling, emit a ``DeprecationWarning`` pointing at the
    canonical ``qd.graph.<name>`` spelling. Called once from each construct's lowering entry point so the warning fires
    per use-site (deduplicated to once per message by the ``filterwarnings('once', ...)`` set up in ``lang/misc.py``).
    """
    if is_deprecated_spelling(func, name):
        flat = _FLAT_NAME[name]
        warnings.warn(
            f"qd.{flat}() is deprecated and will be removed in a future release; use qd.graph.{name}() instead.",
            DeprecationWarning,
        )
