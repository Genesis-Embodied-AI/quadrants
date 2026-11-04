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
    annotation._qd_tuple_plan = TuplePlan(value, name, raise_on_templated_floats)
    return annotation


def predeclare_tuple_scalars(ctx):
    # pylint: disable=import-outside-toplevel
    from quadrants._lib import core  # pylint: disable=import-outside-toplevel
    from quadrants.lang import expr, impl  # pylint: disable=import-outside-toplevel
    from quadrants.types.utils import (
        is_signed,  # pylint: disable=import-outside-toplevel
    )

    gc = ctx.global_context
    pruning = gc.pruning
    for arg_idx, meta in enumerate(ctx.func.arg_metas):
        plan = getattr(meta.annotation, "_qd_tuple_plan", None)
        if plan is None:
            continue
        for ordinal, (path, dtype) in enumerate(plan.scalars):
            parent = path_value(ctx.py_args[arg_idx], path[:-1])
            key = (id(parent), path[-1])
            if key in gc.struct_primitive_provenance:
                continue
            name = f"__qd_tuple_scalar_{arg_idx}_{ordinal}"
            kind = "f" if dtype.to_string().startswith("f") else "i" if is_signed(dtype) else "u"
            gc.struct_primitive_provenance[key] = (name, arg_idx, path, kind)
            if not pruning.enforcing or not pruning.is_used(pruning.KERNEL_FUNC_ID, name):
                continue
            ids = impl.get_runtime().compiling_callable.insert_scalar_param(dtype, name)
            gc.struct_primitive_to_expr[key] = expr.Expr(
                core.make_arg_load_expr(
                    ids,
                    dtype,
                    False,
                    create_load=True,
                    dbg_info=core.DebugInfo(impl.get_runtime().get_current_src_info()),
                )
            )
            gc.struct_primitive_launch_info.append((ids[0], arg_idx, path, kind))
