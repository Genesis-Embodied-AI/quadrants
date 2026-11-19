"""Runtime descriptions and indexed bindings for host tuples containing inlined callable objects.

Only tuple arguments enter this path. A description is rebuilt for each launch because callable objects and their
captures can be reassigned. Keys describe code and alias topology without retaining the captured object graph.
"""

from quadrants.types.annotations import template


def path_value(root, path):
    for step in path:
        root = root[step] if type(step) is int else getattr(root, step)
    return root


def tuple_annotation(value, name, raise_on_templated_floats=False):
    # pylint: disable=import-outside-toplevel
    from quadrants.lang._runtime_tuple_plan import (
        TuplePlan,  # pylint: disable=import-outside-toplevel
    )

    annotation = template()
    setattr(annotation, "_qd_tuple_plan", TuplePlan(value, name, raise_on_templated_floats))
    return annotation


def predeclare_tuple_scalars(ctx):
    from quadrants.lang.ast.ast_transformers.data_oriented_runtime_scalar_binder import (
        _ScalarMember,
        _create_runtime_scalar_bindings,
    )

    members = []
    for arg_idx, meta in enumerate(ctx.func.arg_metas):
        plan = getattr(meta.annotation, "_qd_tuple_plan", None)
        if plan is None:
            continue
        for path, dtype in plan.scalars:
            parent = path_value(ctx.py_args[arg_idx], path[:-1])
            members.append(_ScalarMember(parent, path[-1], getattr(parent, path[-1]), arg_idx, path, dtype))
    _create_runtime_scalar_bindings(ctx, members)
