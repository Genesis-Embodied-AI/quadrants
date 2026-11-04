"""Runtime descriptions and indexed bindings for host tuples containing inlined callable objects.

Only tuple arguments enter this path. A description is rebuilt for each launch because callable objects and their
captures can be reassigned. Keys describe code and alias topology without retaining the captured object graph.
"""

import dataclasses

from quadrants.types.annotations import template


def path_value(root, path):
    for step in path:
        root = root[step] if type(step) is int else getattr(root, step)
    return root


def tuple_annotation(value, name, raise_on_templated_floats=False):
    annotation = template()
    annotation._qd_tuple_plan = TuplePlan(value, name, raise_on_templated_floats)
    return annotation


class TuplePlan:
    def __init__(self, value, name, raise_on_templated_floats):
        self.arrays = []
        self.scalars = []
        self._seen = {}
        self._visiting = set()
        self.name = name
        self.raise_on_templated_floats = raise_on_templated_floats
        self.key = self._describe(value, ())
        del self._seen, self._visiting

    def _describe(self, value, path, runtime_scalar=False):
        # Imports stay here to keep this feature module usable from registration and template dispatch without
        # introducing import cycles through the compiler.
        import numpy as np  # pylint: disable=import-outside-toplevel

        from quadrants._tensor_wrapper import (
            Tensor,  # pylint: disable=import-outside-toplevel
        )
        from quadrants.lang._final_dataclass_fields import (
            final_scalar_key,  # pylint: disable=import-outside-toplevel
        )
        from quadrants.lang._ndarray import (
            Ndarray,  # pylint: disable=import-outside-toplevel
        )
        from quadrants.lang._quadrants_callable import (  # pylint: disable=import-outside-toplevel
            BoundQuadrantsCallable,
            QuadrantsCallable,
        )
        from quadrants.lang._runtime_dataclass import (  # pylint: disable=import-outside-toplevel
            _direct_final_names,
            infer_member_type,
        )
        from quadrants.lang._template_mapper_hotpath import (
            _extract_arg,  # pylint: disable=import-outside-toplevel
        )
        from quadrants.lang.exception import (
            QuadrantsRuntimeTypeError,  # pylint: disable=import-outside-toplevel
        )
        from quadrants.lang.field import (
            Field,  # pylint: disable=import-outside-toplevel
        )
        from quadrants.lang.util import (  # pylint: disable=import-outside-toplevel
            is_data_oriented,
            wants_runtime_primitives,
        )
        from quadrants.types.ndarray_type import (
            NdarrayType,  # pylint: disable=import-outside-toplevel
        )

        if isinstance(value, Tensor):
            value = value._unwrap()
        if value is None:
            return ("none",)
        if isinstance(value, (bool, int, float, np.number, np.bool_)):
            if runtime_scalar:
                dtype = infer_member_type(value, self.name, frozenset())
                self.scalars.append((path, dtype))
                return ("scalar", dtype)
            if self.raise_on_templated_floats and isinstance(value, (float, np.floating)):
                raise ValueError("Floats not allowed as templated types.")
            return ("constant", type(value), value)
        if isinstance(value, (str, bytes)):
            return ("constant", type(value), value)
        if isinstance(value, QuadrantsCallable):
            if not value._is_quadrants_function:
                raise QuadrantsRuntimeTypeError(f"Tuple member {self.name}{path} must be a qd.func, not a kernel")
            return ("function", value.wrapper)
        if isinstance(value, BoundQuadrantsCallable):
            return ("bound_function", value.wrapper, self._describe(value.instance, path))
        if isinstance(value, Field):
            return ("field", value)
        identity = id(value)
        if identity in self._visiting:
            raise QuadrantsRuntimeTypeError(f"Tuple member {self.name}{path} contains a cycle")
        if identity in self._seen:
            return ("alias", self._seen[identity])
        self._seen[identity] = len(self._seen)
        if isinstance(value, Ndarray):
            self.arrays.append((path, value))
            return ("array", _extract_arg(False, value, NdarrayType(), self.name))
        self._visiting.add(identity)
        try:
            if isinstance(value, tuple):
                return ("tuple", tuple(self._describe(child, (*path, i)) for i, child in enumerate(value)))
            if hasattr(type(value), "__dataclass_fields__"):
                finals = _direct_final_names(type(value))
                return (
                    "dataclass",
                    type(value),
                    tuple(
                        (
                            field.name,
                            (
                                final_scalar_key(getattr(value, field.name), self.raise_on_templated_floats)
                                if field.name in finals
                                else self._describe(
                                    getattr(value, field.name), (*path, field.name), runtime_scalar=True
                                )
                            ),
                        )
                        for field in dataclasses.fields(value)
                    ),
                )
            if is_data_oriented(value):
                # Bound Quadrants methods are cached in the instance dictionary by the descriptor. They are not
                # independent captures; describing them would walk straight back into the same instance.
                entries = tuple(
                    (key, child)
                    for key, child in vars(value).items()
                    if not key.startswith("_qd_") and not isinstance(getattr(type(value), key, None), QuadrantsCallable)
                )
                return (
                    "object",
                    type(value),
                    tuple(
                        (key, self._describe(child, (*path, key), wants_runtime_primitives(value)))
                        for key, child in entries
                    ),
                )
        finally:
            self._visiting.remove(identity)
        raise QuadrantsRuntimeTypeError(
            f"Tuple member {self.name}{path} has unsupported runtime type {type(value).__name__}"
        )


def predeclare_tuple_scalars(ctx):
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
            gc.struct_primitive_to_expr[key] = expr.Expr(core.make_arg_load_expr(ids, dtype, False, create_load=True))
            gc.struct_primitive_launch_info.append((ids[0], arg_idx, path, kind))
