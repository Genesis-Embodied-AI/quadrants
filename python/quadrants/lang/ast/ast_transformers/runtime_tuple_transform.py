"""Lower host-tuple iteration, callable objects, and starred captures before dataclass flattening."""

import ast

from quadrants.lang._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from quadrants.lang.util import is_data_oriented


def resolve_callable(func):
    if is_data_oriented(func):
        method = getattr(func, "__call__", None)
        if isinstance(method, (QuadrantsCallable, BoundQuadrantsCallable)) and method._is_quadrants_function:
            return method
    return func


def expand_tuple_stars(ctx, args, build_stmt):
    if not any(isinstance(arg, ast.Starred) and isinstance(getattr(arg, "ptr", None), tuple) for arg in args):
        return args
    expanded = []
    for arg in args:
        if not isinstance(arg, ast.Starred) or not isinstance(getattr(arg, "ptr", None), tuple):
            expanded.append(arg)
            continue
        for index in range(len(getattr(arg, "ptr"))):
            child = ast.copy_location(
                ast.Subscript(value=arg.value, slice=ast.Constant(value=index), ctx=ast.Load()), arg
            )
            ast.fix_missing_locations(child)
            build_stmt(ctx, child)
            expanded.append(child)
    return expanded


def build_tuple_for(ctx, node, decorator, build_stmt, build_static_for):
    """Evaluate an ordinary iterable once, and use static-loop lowering when it is a host tuple.

    Return whether the iterable was evaluated and whether the complete loop was handled. Range, explicitly
    decorated loops, and conditional iterables continue through their existing specialized transformations.
    """
    is_range = isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"
    if decorator or is_range or isinstance(node.iter, ast.IfExp):
        return False, False
    build_stmt(ctx, node.iter)
    if not isinstance(node.iter.ptr, tuple):
        return True, False
    with ctx.loop_scope_guard(is_static=True):
        build_static_for(ctx, node, False, iter_built=True)
    return True, True
