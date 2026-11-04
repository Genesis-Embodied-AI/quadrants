"""Normalize callable objects and host-tuple stars before dataclass inference and pruning."""

import ast

from quadrants.lang._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from quadrants.lang.util import is_data_oriented


def resolve_callable(func):
    if is_data_oriented(func):
        method = getattr(func, "__call__", None)
        if type(method) in (QuadrantsCallable, BoundQuadrantsCallable) and method._is_quadrants_function:
            return method
    return func


def expand_tuple_stars(ctx, args, build_stmt):
    if not any(isinstance(arg, ast.Starred) and isinstance(arg.ptr, tuple) for arg in args):
        return args
    expanded = []
    for arg in args:
        if not isinstance(arg, ast.Starred) or not isinstance(arg.ptr, tuple):
            expanded.append(arg)
            continue
        for index in range(len(arg.ptr)):
            child = ast.copy_location(
                ast.Subscript(value=arg.value, slice=ast.Constant(value=index), ctx=ast.Load()), arg
            )
            ast.fix_missing_locations(child)
            build_stmt(ctx, child)
            expanded.append(child)
    return expanded
