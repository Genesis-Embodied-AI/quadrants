import ast
from collections import namedtuple

from quadrants.lang.ast.symbol_resolver import ASTResolver


def test_ast_resolver_basic():
    # import within the function to avoid polluting the global scope
    import quadrants as qd

    qd.init()
    node = ast.parse("qd.kernel", mode="eval").body
    assert ASTResolver.resolve_to(node, qd.kernel, locals())


def test_ast_resolver_direct_import():
    import quadrants as qd

    qd.init()
    from quadrants import kernel

    node = ast.parse("kernel", mode="eval").body
    assert ASTResolver.resolve_to(node, kernel, locals())


def test_ast_resolver_alias():
    import quadrants

    quadrants.init()
    node = ast.parse("quadrants.kernel", mode="eval").body
    assert ASTResolver.resolve_to(node, quadrants.kernel, locals())

    import quadrants as tc

    node = ast.parse("tc.kernel", mode="eval").body
    assert ASTResolver.resolve_to(node, tc.kernel, locals())


def test_ast_resolver_chain():
    import quadrants as qd

    qd.init()
    node = ast.parse("qd.lang.ops.atomic_add", mode="eval").body
    assert ASTResolver.resolve_to(node, qd.atomic_add, locals())


def test_ast_resolver_wrong_ti():
    import quadrants

    quadrants.init()
    fake_ti = namedtuple("FakeTi", ["kernel"])
    ti = fake_ti(kernel="fake")
    node = ast.parse("qd.kernel", mode="eval").body
    assert not ASTResolver.resolve_to(node, quadrants.kernel, locals())
