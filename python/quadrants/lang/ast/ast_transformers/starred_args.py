"""Normalize compile-time starred dataclass arguments before call-parameter expansion."""

# type: ignore

import ast
import dataclasses


def expand_starred_dataclass_args(args: list[ast.expr], keywords: list[ast.keyword]) -> list[ast.expr]:
    # Leave unsupported star positions to the existing pruning diagnostic, and keep tensor-expression unpacking on
    # its existing path. Only compile-time tuples/lists containing dataclasses need earlier expansion.
    if not args or keywords or not isinstance(args[-1], ast.Starred):
        return args
    if any(isinstance(arg, ast.Starred) for arg in args[:-1]):
        return args
    starred = args[-1]
    values = starred.ptr
    if not isinstance(values, (tuple, list)) or not any(dataclasses.is_dataclass(value) for value in values):
        return args

    expanded = list(args[:-1])
    for index, value in enumerate(values):
        # Typed dataclass locals are represented by their class and flattened Name nodes. Keep the original node
        # when unpacking a literal tuple/list so expansion can resolve those names in the caller's scope.
        if isinstance(value, type) and dataclasses.is_dataclass(value):
            if not isinstance(starred.value, (ast.Tuple, ast.List)):
                return args
            item = starred.value.elts[index]
        else:
            # The expression has already been evaluated. A synthetic constant prevents per-field attribute loads
            # from evaluating it again (especially for ``*make_args()``).
            item = ast.copy_location(ast.Constant(value=value), starred)
            item.ptr = value
            item.violates_pure = starred.value.violates_pure
            item.violates_pure_reason = getattr(starred.value, "violates_pure_reason", None)
        expanded.append(item)
    return expanded
