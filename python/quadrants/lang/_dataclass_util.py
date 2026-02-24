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
