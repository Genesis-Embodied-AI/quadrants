import ast

import pytest

from quadrants.lang.ast.node_metadata import node_metadata


def test_metadata_preserves_ast_identity_and_dynamic_fields():
    node = ast.parse("value").body[0].value
    metadata = node_metadata(node)
    value = object()
    metadata.ptr = value
    metadata.violates_pure = True
    metadata.violates_pure_reason = "global variable"
    metadata._qd_arg_chain = "value"

    assert metadata is node
    assert getattr(node, "ptr") is value
    assert getattr(node, "violates_pure") is True
    assert getattr(node, "violates_pure_reason") == "global variable"
    assert getattr(node, "_qd_arg_chain") == "value"
    assert ast.unparse(node) == "value"

    setattr(node, "ptr", 42)
    assert metadata.ptr == 42


def test_metadata_does_not_initialize_unbuilt_nodes():
    node = ast.Name(id="value", ctx=ast.Load())
    metadata = node_metadata(node)
    with pytest.raises(AttributeError, match="ptr"):
        _ = metadata.ptr
    assert not hasattr(node, "violates_pure")
    assert not hasattr(node, "_qd_arg_chain")
