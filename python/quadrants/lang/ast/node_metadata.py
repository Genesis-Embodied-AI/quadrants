"""Typed access to the metadata attached to Python AST nodes during lowering."""

import ast
from typing import Any, Protocol, cast


class NodeMetadata(Protocol):
    """Fields populated by the transformer, on the original AST node.

    ``ptr`` is a dynamically evaluated Python value or a Quadrants expression. Fields are only available after the
    corresponding builder has populated them; this view neither initializes fields nor changes missing-field behavior.
    """

    ptr: Any
    violates_pure: bool
    violates_pure_reason: str | None
    _qd_arg_chain: str | None


def node_metadata(node: ast.AST) -> NodeMetadata:
    """Expose lowering metadata without discarding the node's standard AST type."""
    return cast(NodeMetadata, node)
