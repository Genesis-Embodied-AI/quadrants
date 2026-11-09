# type: ignore

"""Provides helpers to resolve AST nodes."""

import ast

from quadrants.lang._ast_name_parts import extract_dotted_name_parts


class ASTResolver:
    """Provides helper methods to resolve AST nodes."""

    @staticmethod
    def resolve_to(node, wanted, scope):
        """Check if symbol ``node`` resolves to ``wanted`` object.

        This is only intended to check if a given AST node resolves to a symbol
        under some namespaces, e.g. the ``a.b.c.foo`` pattern, but not meant for
        more complicated expressions like ``(a + b).foo``.

        Args:
            node (Union[ast.Attribute, ast.Name]): an AST node to be resolved.
            wanted (Any): The expected python object.
            scope (Dict[str, Any]): Maps from symbol names to objects, for
                example, globals()

        Returns:
            bool: The checked result.
        """
        if isinstance(node, ast.Name):
            return scope.get(node.id) is wanted

        name_parts = extract_dotted_name_parts(node)
        if name_parts is None:
            return False

        for attr in name_parts:
            try:
                if isinstance(scope, dict):
                    scope = scope[attr]
                else:
                    scope = getattr(scope, attr)
            except (KeyError, AttributeError):
                return False
        # The name ``scope`` here could be a bit confusing
        return scope is wanted

    @staticmethod
    def resolve_value(node, scope):
        """Resolve an AST Name/Attribute node to a Python object.

        Same traversal as resolve_to but returns the resolved object (or None) instead of comparing against a wanted
        value.
        """
        if isinstance(node, ast.Name):
            return scope.get(node.id) if isinstance(scope, dict) else None

        name_parts = extract_dotted_name_parts(node)
        if name_parts is None:
            return None

        for attr in name_parts:
            try:
                if isinstance(scope, dict):
                    scope = scope[attr]
                else:
                    scope = getattr(scope, attr)
            except (KeyError, AttributeError):
                return None
        return scope
