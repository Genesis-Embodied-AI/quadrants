"""Extract Python names from AST expressions without resolving or evaluating them."""

import ast


def extract_dotted_name_parts(ast_node: ast.expr) -> list[str] | None:
    """Return name components in lookup order, or None if the expression is not a name/attribute chain.

    Input: the expression tree for "module.typing.Final".
    Attribute
    |-- value: Attribute
    |   |-- value: Name("module")
    |   `-- attr: "typing"           <- a string
    `-- attr: "Final"                <- a string

    Output: ["module", "typing", "Final"]. A plain Name("Final") produces ["Final"].
    The loop collects "Final", then "typing", and stops at Name("module"). Appending "module" and
    reversing puts the components in lookup order. Calls and subscripts, such as factory().Final or
    items[0].Final, return None. The input tree is not modified, and no Python objects are looked up.
    """
    name_parts = []
    current_node = ast_node
    while isinstance(current_node, ast.Attribute):
        name_parts.append(current_node.attr)
        current_node = current_node.value
    if not isinstance(current_node, ast.Name):
        return None
    name_parts.append(current_node.id)
    name_parts.reverse()
    return name_parts
