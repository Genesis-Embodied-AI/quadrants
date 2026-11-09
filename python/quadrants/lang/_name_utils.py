"""Extract AST name components and format internal argument names."""

import ast


def extract_dotted_name_parts(ast_node: ast.expr) -> list[str] | None:
    """Return name components in lookup order, or None if the expression is not a name/attribute chain.

    Input: the expression tree for "module.typing.Final".
    Attribute
    |-- value: Attribute
    |   |-- value: Name("module")
    |   `-- attr: "typing"           <- a string
    `-- attr: "Final"                <- a string

    Output: ["module", "typing", "Final"].

    A plain Name("Final") produces ["Final"].

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


def create_flat_name(basename: str, child_name: str) -> str:
    """
    Appends child_name to basename, separated by __qd_.
    If basename does not start with __qd_ then prefix the resulting string
    with __qd_.

    Note that we want to avoid adding prefix __qd_ if already included in `basename`,
    to avoid duplicating said delimiter.

    We'll use this when expanding py dataclass members, e.g.

    @dataclasses.dataclass
    def Foo:
        a: int
        b: int

    foo = Foo(a=5, b=3)

    When we expand out foo, we'll replace foo with the following names instead:
    - __qd_foo__qd_a
    - __qd_foo__qd_b

    We use the __qd_ to ensure that it's easy to ensure no collision with existing user-defined
    names. We require the user to not create any fields or variables which themselves are prefixed
    with __qd_, and given this constraint, the names we create will not conflict with user-generated
    names.
    """
    if basename.startswith("__qd_"):
        return f"{basename}__qd_{child_name}"
    return f"__qd_{basename}__qd_{child_name}"


def create_flat_name_from_parts(name_parts: list[str]) -> str:
    """Flatten a nonempty list of names; a single name passes through unchanged."""
    flat_name = "__qd_".join(name_parts)
    if len(name_parts) > 1 and not name_parts[0].startswith("__qd_"):
        flat_name = "__qd_" + flat_name
    return flat_name


def dotted_argument_path_from_flat_name(flat_name: str) -> str:
    """Convert a compiler name such as "__qd_state__qd_velocity" to the dotted argument path "state.velocity"."""
    return flat_name.removeprefix("__qd_").replace("__qd_", ".")
