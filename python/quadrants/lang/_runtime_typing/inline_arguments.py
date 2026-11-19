"""Bind inlined function parameters from values, independently of Python annotations."""

import ast
from typing import Any

import numpy as np

from quadrants._tensor_wrapper import Tensor
from quadrants.lang import impl
from quadrants.lang._runtime_typing.runtime_dataclass import cached_infer_dataclass_type
from quadrants.lang.ast.ast_transformer_utils import FuncCompilePassContext
from quadrants.lang.ast.ast_transformers.data_oriented_runtime_scalar_binder import register_runtime_dataclass_schema
from quadrants.lang.expr import Expr
from quadrants.lang.matrix import Matrix
from quadrants.lang.struct import Struct
from quadrants.lang.util import is_dataclass_instance


class _AssignedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # A type-only declaration neither writes the parameter nor evaluates its annotation.
        if node.value is not None:
            self.visit(node.target)
            self.visit(node.value)


def _bind_value(ctx: FuncCompilePassContext, name: str, value: Any, assigned_names: set[str]) -> None:
    """Keep object references and compile-time values; copy runtime expressions into local variables."""
    ctx.fn_param_names.add(name)
    pass_ctx = ctx.kernel_compile_context.current_pass
    if isinstance(value, Tensor):
        value = value._unwrap()
    if is_dataclass_instance(value):
        if id(value) not in pass_ctx.runtime_schema_by_id:
            register_runtime_dataclass_schema(pass_ctx, value, cached_infer_dataclass_type(value, name), name)
    else:
        value = pass_ctx.ndarray_to_any_array.get(id(value), value)
        if isinstance(value, (Expr, Matrix, Struct)):
            value = impl.expr_init_func(value)
        elif name in assigned_names and isinstance(value, (bool, int, float, np.number, np.bool_)):
            # Allocate at function entry, before any runtime branch, so reassignment never changes a compile-time
            # binding conditionally. Read-only Python scalars remain available to qd.static without a template hint.
            value = impl.expr_init_func(value)
    ctx.create_variable(name, value)


def bind_inline_arguments(ctx: FuncCompilePassContext, node: ast.FunctionDef) -> None:
    """Bind already-fused positional, keyword, default, and self arguments to their parameter names."""
    assert ctx.py_args is not None
    assigned = _AssignedNames()
    for statement in node.body:
        assigned.visit(statement)
    for meta, value in zip(ctx.func.arg_metas, ctx.py_args):
        _bind_value(ctx, meta.name, value, assigned.names)
