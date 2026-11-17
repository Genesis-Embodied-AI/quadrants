# type: ignore
"""Lift primitive members of ``@qd.data_oriented(template_primitives=False)`` template args into runtime scalar kernel
args at AST-build time.

Lives alongside ``ndarray_arg_resolver.py`` / ``checkpoint_transformer.py`` so the central
``function_def_transformer.py`` file doesn't have to grow per-feature (the same rationale the ndarray resolver
documents). ``FunctionDefTransformer.transform`` delegates here right after ``_predeclare_struct_ndarrays``; this is
the only entry point.

See ``docs/source/user_guide/compound_types.md`` (section "Runtime primitives: ``template_primitives=False``") for the
user-facing surface.
"""

import dataclasses
from typing import Any

from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang import impl
from quadrants.lang._kernel_parameter_bindings import ParameterSource
from quadrants.lang.ast.ast_transformer_utils import FuncCompileContext
from quadrants.lang.util import (
    is_data_oriented,
    is_dataclass_instance,
    wants_runtime_primitives,
)
from quadrants.types import annotations


def predeclare_struct_primitives(ctx: FuncCompileContext) -> None:
    """Walk opted-in data-oriented arguments and register their scalar members as kernel parameters.

    Visit stored attributes of data-oriented objects and dataclasses, stopping at cycles. Only exact Python
    bool, int, and float members become runtime parameters; their types follow the runtime defaults.
    """
    assert ctx.py_args is not None
    # Cheap early-out for the common case (no flagged template arg): the default ``template_primitives=True`` path
    # never builds a walk, so genesis-style data_oriented kernels pay only one ``wants_runtime_primitives`` check
    # per template arg at compile time, and nothing at launch.
    flagged: list[tuple[int, Any]] = []
    for i, arg_meta in enumerate(ctx.func.arg_metas):
        anno = arg_meta.annotation
        is_template = anno is annotations.template or isinstance(anno, annotations.template)
        is_tensor_anno = anno is _TensorClass
        if not (is_template or is_tensor_anno):
            continue
        val = ctx.py_args[i]
        if isinstance(val, _TensorClass):
            val = val._unwrap()
        if wants_runtime_primitives(val):
            flagged.append((i, val))
    if not flagged:
        return

    runtime = impl.get_runtime()
    bindings = ctx.kernel_compile_context.parameter_bindings

    def _walk(obj, arg_idx, path, seen):
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if is_dataclass_instance(obj):
            items = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
        elif hasattr(obj, "__dict__"):
            items = list(vars(obj).items())
        else:
            return
        for attr_name, attr_val in items:
            if isinstance(attr_val, _TensorClass):
                attr_val = attr_val._unwrap()
            attr_type = type(attr_val)
            if attr_type is bool or attr_type is int or attr_type is float:
                dtype = runtime.default_fp if attr_type is float else runtime.default_ip
                bindings.add_kernel_parameter(
                    "scalar",
                    dtype,
                    f"__qd_doprim_{arg_idx}__{'__'.join((*path, attr_name))}",
                    source=ParameterSource(arg_idx, (*path, attr_name)),
                    member=(id(obj), attr_name),
                )
            elif is_dataclass_instance(attr_val) or is_data_oriented(attr_val):
                _walk(attr_val, arg_idx, (*path, attr_name), seen)

    for arg_idx, val in flagged:
        _walk(val, arg_idx, (), set())
