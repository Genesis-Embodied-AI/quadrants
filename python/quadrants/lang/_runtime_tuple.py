"""Runtime descriptions and indexed bindings for host tuples containing inlined callable objects.

Only tuple arguments enter this path. A description is rebuilt for each launch because callable objects and their
scalar values and selected functions can change. Keys describe code and alias topology without retaining the captured object graph.
"""

from quadrants.lang.util import get_member_at_path
from quadrants.types.annotations import template


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
        register_runtime_dataclass_schema,
    )

    from quadrants.lang._runtime_typing.runtime_dataclass import cached_infer_dataclass_type

    pass_ctx = ctx.kernel_compile_context.current_pass
    members = []
    for arg_idx, meta in enumerate(ctx.func.arg_metas):
        plan = getattr(meta.annotation, "_qd_tuple_plan", None)
        if plan is None:
            continue
        for path in plan.dataclass_paths:
            parent = get_member_at_path(ctx.py_args[arg_idx], path[:-1])
            if isinstance(path[-1], str):
                pass_ctx.runtime_dataclass_members.add((id(parent), path[-1]))
            obj = get_member_at_path(ctx.py_args[arg_idx], path)
            name = ".".join((meta.name, *map(str, path)))
            register_runtime_dataclass_schema(pass_ctx, obj, cached_infer_dataclass_type(obj, name), name)
        for path, dtype in plan.scalars:
            parent = get_member_at_path(ctx.py_args[arg_idx], path[:-1])
            members.append(_ScalarMember(parent, path[-1], getattr(parent, path[-1]), arg_idx, path, dtype))
    _create_runtime_scalar_bindings(ctx, members)
