# type: ignore

import ast
import dataclasses
from typing import Any, Callable

from quadrants._lib.core.quadrants_python import (
    BoundaryMode,
    DataTypeCxx,
)
from quadrants.lang import (
    _ndarray,
    any_array,
    expr,
    impl,
    kernel_arguments,
    matrix,
)
from quadrants.lang import ops as qd_ops
from quadrants.lang._dataclass_util import create_flat_name
from quadrants.lang.ast.ast_transformer_utils import (
    ASTTransformerFuncContext,
)
from quadrants.lang.exception import (
    QuadrantsSyntaxError,
)
from quadrants.lang.matrix import MatrixType
from quadrants.lang.struct import StructType
from quadrants.lang.util import to_quadrants_type
from quadrants.types import annotations, ndarray_type, primitive_types


class FunctionDefTransformer:
    @staticmethod
    def _decl_and_create_variable(
        ctx: ASTTransformerFuncContext,
        annotation: Any,
        name: str,
        this_arg_features: tuple[tuple[Any, ...], ...] | None,
        prefix_name: str,
    ) -> tuple[bool, Any]:
        full_name = prefix_name + "_" + name
        if not isinstance(annotation, primitive_types.RefType):
            ctx.kernel_args.append(name)
        if annotation == annotations.template or isinstance(annotation, annotations.template):
            if name in ctx.template_vars:
                return True, ctx.template_vars[name]
            assert ctx.global_vars is not None
            return True, ctx.global_vars.get(name)
        if isinstance(annotation, annotations.sparse_matrix_builder):
            return False, (
                kernel_arguments.decl_sparse_matrix,
                (
                    to_quadrants_type(this_arg_features),
                    full_name,
                ),
            )
        if isinstance(annotation, ndarray_type.NdarrayType):
            assert this_arg_features is not None
            raw_element_type: DataTypeCxx
            ndim: int
            needs_grad: bool
            boundary: int
            raw_element_type, ndim, needs_grad, boundary = this_arg_features
            return False, (
                kernel_arguments.decl_ndarray_arg,
                (
                    to_quadrants_type(raw_element_type),
                    ndim,
                    full_name,
                    needs_grad,
                    BoundaryMode(boundary),
                ),
            )
        if isinstance(annotation, MatrixType):
            return True, kernel_arguments.decl_matrix_arg(annotation, name)
        if isinstance(annotation, StructType):
            return True, kernel_arguments.decl_struct_arg(annotation, name)
        return True, kernel_arguments.decl_scalar_arg(annotation, name)

    @staticmethod
    def _transform_kernel_arg(
        ctx: ASTTransformerFuncContext,
        argument_name: str,
        argument_type: Any,
        this_arg_features: tuple[Any, ...],
    ) -> None:
        pruning = ctx.global_context.pruning
        func_id = ctx.func.func_id
        if dataclasses.is_dataclass(argument_type):
            ctx.create_variable(argument_name, argument_type)
            for field_idx, field in enumerate(dataclasses.fields(argument_type)):
                flat_name = create_flat_name(argument_name, field.name)
                if pruning.enforcing and flat_name not in pruning.used_vars_by_func_id[func_id]:
                    continue
                # if a field is a dataclass, then feed back into process_kernel_arg recursively
                if dataclasses.is_dataclass(field.type):
                    FunctionDefTransformer._transform_kernel_arg(
                        ctx,
                        flat_name,
                        field.type,
                        this_arg_features[field_idx],
                    )
                else:
                    result, obj = FunctionDefTransformer._decl_and_create_variable(
                        ctx,
                        field.type,
                        flat_name,
                        this_arg_features[field_idx],
                        "",
                    )
                    if result:
                        ctx.create_variable(flat_name, obj)
                    else:
                        decl_type_func, type_args = obj
                        obj = decl_type_func(*type_args)
                        ctx.create_variable(flat_name, obj)
        else:
            result, obj = FunctionDefTransformer._decl_and_create_variable(
                ctx,
                argument_type,
                argument_name,
                this_arg_features if ctx.arg_features is not None else None,
                "",
            )
            if not result:
                decl_type_func, type_args = obj
                obj = decl_type_func(*type_args)
            ctx.create_variable(argument_name, obj)

    @staticmethod
    def _transform_as_kernel(ctx: ASTTransformerFuncContext, node: ast.FunctionDef, args: ast.arguments) -> None:
        assert ctx.func is not None
        assert ctx.arg_features is not None
        if node.returns is not None:
            if not isinstance(node.returns, ast.Constant):
                assert ctx.func.return_type is not None
                for return_type in ctx.func.return_type:
                    kernel_arguments.decl_ret(return_type)
        compiling_callable = impl.get_runtime().compiling_callable
        assert compiling_callable is not None
        compiling_callable.finalize_rets()

        for i in range(len(args.args)):
            arg_meta = ctx.func.arg_metas[i]
            FunctionDefTransformer._transform_kernel_arg(
                ctx,
                arg_meta.name,
                arg_meta.annotation,
                ctx.arg_features[i] if ctx.arg_features is not None else (),
            )

        compiling_callable.finalize_params()
        # remove original args
        node.args.args = []

    @staticmethod
    def _transform_func_arg(
        ctx: ASTTransformerFuncContext,
        argument_name: str,
        argument_type: Any,
        data: Any,
    ) -> None:
        # Template arguments are passed by reference.
        if isinstance(argument_type, annotations.template):
            ctx.create_variable(argument_name, data)
            return None

        if dataclasses.is_dataclass(argument_type):
            for field in dataclasses.fields(argument_type):
                flat_name = create_flat_name(argument_name, field.name)
                data_child = getattr(data, field.name)
                if isinstance(
                    data_child,
                    (
                        _ndarray.ScalarNdarray,
                        matrix.VectorNdarray,
                        matrix.MatrixNdarray,
                        any_array.AnyArray,
                    ),
                ):
                    field.type.check_matched(data_child.get_type(), field.name)
                    ctx.create_variable(flat_name, data_child)
                elif dataclasses.is_dataclass(data_child):
                    FunctionDefTransformer._transform_func_arg(
                        ctx,
                        flat_name,
                        field.type,
                        getattr(data, field.name),
                    )
                else:
                    raise QuadrantsSyntaxError(
                        f"Argument {field.name} of type {argument_type} {field.type} is not recognized."
                    )
            return None

        # Ndarray arguments are passed by reference.
        if isinstance(argument_type, (ndarray_type.NdarrayType)):
            if not isinstance(
                data, (_ndarray.ScalarNdarray, matrix.VectorNdarray, matrix.MatrixNdarray, any_array.AnyArray)
            ):
                raise QuadrantsSyntaxError(f"Argument {argument_name} of type {argument_type} is not recognized.")
            argument_type.check_matched(data.get_type(), argument_name)
            ctx.create_variable(argument_name, data)
            return None

        # Matrix arguments are passed by value.
        if isinstance(argument_type, (MatrixType)):
            # "data" is expected to be an Expr here,
            # so we simply call "impl.expr_init_func(data)" to perform:
            #
            # TensorType* t = alloca()
            # assign(t, data)
            #
            # We created local variable "t" - a copy of the passed-in argument "data"
            if not isinstance(data, expr.Expr) or not data.ptr.is_tensor():
                raise QuadrantsSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix, but got {type(data)}."
                )

            element_shape = data.ptr.get_rvalue_type().shape()
            if len(element_shape) != argument_type.ndim:
                raise QuadrantsSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with ndim {argument_type.ndim}, but got {len(element_shape)}."
                )

            assert argument_type.ndim > 0
            if element_shape[0] != argument_type.n:
                raise QuadrantsSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with n {argument_type.n}, but got {element_shape[0]}."
                )

            if argument_type.ndim == 2 and element_shape[1] != argument_type.m:
                raise QuadrantsSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with m {argument_type.m}, but got {element_shape[0]}."
                )

            ctx.create_variable(argument_name, impl.expr_init_func(data))
            return None

        if id(argument_type) in primitive_types.type_ids:
            ctx.create_variable(argument_name, impl.expr_init_func(qd_ops.cast(data, argument_type)))
            return None
        # Create a copy for non-template arguments,
        # so that they are passed by value.
        var_name = argument_name
        ctx.create_variable(var_name, impl.expr_init_func(data))
        return None

    @staticmethod
    def _transform_as_func(ctx: ASTTransformerFuncContext, node: ast.FunctionDef, args: ast.arguments) -> None:
        # pylint: disable=import-outside-toplevel
        from quadrants.lang.kernel_impl import Func

        assert isinstance(ctx.func, Func)
        assert ctx.py_args is not None
        for py_arg_i, py_arg in enumerate(ctx.py_args):
            argument = ctx.func.arg_metas_expanded[py_arg_i]
            FunctionDefTransformer._transform_func_arg(ctx, argument.name, argument.annotation, py_arg)

        # deal with dataclasses
        for v in ctx.func.orig_arguments:
            if dataclasses.is_dataclass(v.annotation):
                ctx.create_variable(v.name, v.annotation)

    @staticmethod
    def build_FunctionDef(
        ctx: ASTTransformerFuncContext,
        node: ast.FunctionDef,
        build_stmts: Callable[[ASTTransformerFuncContext, list[ast.stmt]], None],
    ) -> None:
        if ctx.visited_funcdef:
            raise QuadrantsSyntaxError(
                f"Function definition is not allowed in 'qd.{'kernel' if ctx.is_kernel else 'func'}'."
            )
        ctx.visited_funcdef = True

        args = node.args
        assert args.vararg is None
        assert args.kwonlyargs == []
        assert args.kw_defaults == []
        assert args.kwarg is None

        if ctx.is_kernel:  # qd.kernel
            FunctionDefTransformer._transform_as_kernel(ctx, node, args)

        if ctx.only_parse_function_def:
            return None

        if not ctx.is_kernel:  # qd.func
            assert ctx.py_args is not None
            assert ctx.func is not None
            if ctx.is_real_function:
                FunctionDefTransformer._transform_as_kernel(ctx, node, args)
            else:
                FunctionDefTransformer._transform_as_func(ctx, node, args)

        with ctx.variable_scope_guard():
            build_stmts(ctx, node.body)

        return None
