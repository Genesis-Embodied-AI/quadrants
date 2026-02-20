# type: ignore

import ast
import collections.abc
import dataclasses
import enum
import itertools
import math
import platform
import warnings
from ast import unparse
from typing import Any, Generator, Sequence, Type

import numpy as np

from quadrants._lib import core as _ti_core
from quadrants.lang import exception, expr, impl, matrix, mesh
from quadrants.lang import ops as qd_ops
from quadrants.lang._ndrange import _Ndrange
from quadrants.lang.ast.ast_transformer_utils import (
    ASTTransformerFuncContext,
    Builder,
    LoopStatus,
    ReturnStatus,
    get_decorator,
)
from quadrants.lang.ast.ast_transformers.call_transformer import CallTransformer
from quadrants.lang.ast.ast_transformers.function_def_transformer import (
    FunctionDefTransformer,
)
from quadrants.lang.exception import (
    QuadrantsIndexError,
    QuadrantsRuntimeTypeError,
    QuadrantsSyntaxError,
    QuadrantsTypeError,
    handle_exception_from_cpp,
)
from quadrants.lang.expr import Expr, make_expr_group
from quadrants.lang.field import Field
from quadrants.lang.matrix import Matrix, MatrixType
from quadrants.lang.snode import append, deactivate, length
from quadrants.lang.struct import Struct, StructType
from quadrants.types import primitive_types
from quadrants.types.utils import is_integral

AutodiffMode = _ti_core.AutodiffMode


def reshape_list(flat_list: list[Any], target_shape: Sequence[int]) -> list[Any]:
    if len(target_shape) < 2:
        return flat_list

    curr_list = []
    dim = target_shape[-1]
    for i, elem in enumerate(flat_list):
        if i % dim == 0:
            curr_list.append([])
        curr_list[-1].append(elem)

    return reshape_list(curr_list, target_shape[:-1])


def boundary_type_cast_warning(expression: Expr) -> None:
    expr_dtype = expression.ptr.get_rvalue_type()
    if not is_integral(expr_dtype) or expr_dtype in [
        primitive_types.i64,
        primitive_types.u64,
        primitive_types.u32,
    ]:
        warnings.warn(
            f"Casting range_for boundary values from {expr_dtype} to i32, which may cause numerical issues",
            Warning,
        )


class ASTTransformer(Builder):
    @staticmethod
    def build_Name(ctx: ASTTransformerFuncContext, node: ast.Name):
        pruning = ctx.global_context.pruning
        if not pruning.enforcing and not ctx.expanding_dataclass_call_parameters and node.id.startswith("__ti_"):
            ctx.global_context.pruning.mark_used(ctx.func.func_id, node.id)
        node.violates_pure, node.ptr, node.violates_pure_reason = ctx.get_var_by_name(node.id)
        if isinstance(node, (ast.stmt, ast.expr)) and isinstance(node.ptr, Expr):
            node.ptr.dbg_info = _ti_core.DebugInfo(ctx.get_pos_info(node))
            node.ptr.ptr.set_dbg_info(node.ptr.dbg_info)
        if ctx.is_pure and node.violates_pure and not ctx.static_scope_status.is_in_static_scope:
            if isinstance(node.ptr, (float, int, Field)):
                message = f"[PURE.VIOLATION] WARNING: Accessing global variable {node.id} {type(node.ptr)} {node.violates_pure_reason}"
                if node.id.upper() == node.id:
                    warnings.warn(message)
                else:
                    raise exception.QuadrantsCompilationError(message)
        if isinstance(node.ptr, Generator):
            raise ValueError("Cannot store generators in variables, inside kernels or functions")
        return node.ptr

    @staticmethod
    def build_AnnAssign(ctx: ASTTransformerFuncContext, node: ast.AnnAssign):
        build_stmt(ctx, node.value)
        build_stmt(ctx, node.annotation)

        is_static_assign = isinstance(node.value, ast.Call) and node.value.func.ptr is impl.static

        node.ptr = ASTTransformer.build_assign_annotated(
            ctx, node.target, node.value.ptr, is_static_assign, node.annotation.ptr
        )
        return node.ptr

    @staticmethod
    def build_assign_annotated(
        ctx: ASTTransformerFuncContext, target: ast.Name, value, is_static_assign: bool, annotation: Type
    ):
        """Build an annotated assignment like this: target: annotation = value.

        Args:
           ctx (ast_builder_utils.BuilderContext): The builder context.
           target (ast.Name): A variable name. `target.id` holds the name as
           a string.
           annotation: A type we hope to assign to the target
           value: A node representing the value.
           is_static_assign: A boolean value indicating whether this is a static assignment
        """
        is_local = isinstance(target, ast.Name)
        if is_local and target.id in ctx.kernel_args:
            raise QuadrantsSyntaxError(
                f'Kernel argument "{target.id}" is immutable in the kernel. '
                f"If you want to change its value, please create a new variable."
            )
        anno = impl.expr_init(annotation)
        if is_static_assign:
            raise QuadrantsSyntaxError("Static assign cannot be used on annotated assignment")
        if is_local and not ctx.is_var_declared(target.id):
            var = qd_ops.cast(value, anno)
            var = impl.expr_init(var)
            ctx.create_variable(target.id, var)
        else:
            var = build_stmt(ctx, target)
            if var.ptr.get_rvalue_type() != anno:
                raise QuadrantsSyntaxError("Static assign cannot have type overloading")
            var._assign(value)
        return var

    @staticmethod
    def build_Assign(ctx: ASTTransformerFuncContext, node: ast.Assign) -> None:
        build_stmt(ctx, node.value)
        is_static_assign = isinstance(node.value, ast.Call) and node.value.func.ptr is impl.static

        # Keep all generated assign statements and compose single one at last.
        # The variable is introduced to support chained assignments.
        # Ref https://github.com/taichi-dev/quadrants/issues/2659.
        values = node.value.ptr if is_static_assign else impl.expr_init(node.value.ptr)

        for node_target in node.targets:
            ASTTransformer.build_assign_unpack(ctx, node_target, values, is_static_assign)
        return None

    @staticmethod
    def build_assign_unpack(
        ctx: ASTTransformerFuncContext, node_target: list | ast.Tuple, values, is_static_assign: bool
    ):
        """Build the unpack assignments like this: (target1, target2) = (value1, value2).
        The function should be called only if the node target is a tuple.

        Args:
            ctx (ast_builder_utils.BuilderContext): The builder context.
            node_target (ast.Tuple): A list or tuple object. `node_target.elts` holds a
            list of nodes representing the elements.
            values: A node/list representing the values.
            is_static_assign: A boolean value indicating whether this is a static assignment
        """
        if not isinstance(node_target, ast.Tuple):
            return ASTTransformer.build_assign_basic(ctx, node_target, values, is_static_assign)
        targets = node_target.elts

        if isinstance(values, matrix.Matrix):
            if not values.m == 1:
                raise ValueError("Matrices with more than one columns cannot be unpacked")
            values = values.entries

        # Unpack: a, b, c = qd.Vector([1., 2., 3.])
        if isinstance(values, impl.Expr) and values.ptr.is_tensor():
            if len(values.get_shape()) > 1:
                raise ValueError("Matrices with more than one columns cannot be unpacked")

            values = ctx.ast_builder.expand_exprs([values.ptr])
            if len(values) == 1:
                values = values[0]

        if isinstance(values, impl.Expr) and values.ptr.is_struct():
            values = ctx.ast_builder.expand_exprs([values.ptr])
            if len(values) == 1:
                values = values[0]

        if not isinstance(values, collections.abc.Sequence):
            raise QuadrantsSyntaxError(f"Cannot unpack type: {type(values)}")

        if len(values) != len(targets):
            raise QuadrantsSyntaxError("The number of targets is not equal to value length")

        for i, target in enumerate(targets):
            ASTTransformer.build_assign_basic(ctx, target, values[i], is_static_assign)

        return None

    @staticmethod
    def build_assign_basic(ctx: ASTTransformerFuncContext, target: ast.Name, value, is_static_assign: bool):
        """Build basic assignment like this: target = value.

        Args:
           ctx (ast_builder_utils.BuilderContext): The builder context.
           target (ast.Name): A variable name. `target.id` holds the name as
           a string.
           value: A node representing the value.
           is_static_assign: A boolean value indicating whether this is a static assignment
        """
        is_local = isinstance(target, ast.Name)
        if is_local and target.id in ctx.kernel_args:
            raise QuadrantsSyntaxError(
                f'Kernel argument "{target.id}" is immutable in the kernel. '
                f"If you want to change its value, please create a new variable."
            )
        if is_static_assign:
            if not is_local:
                raise QuadrantsSyntaxError("Static assign cannot be used on elements in arrays")
            ctx.create_variable(target.id, value)
            var = value
        elif is_local and not ctx.is_var_declared(target.id):
            var = impl.expr_init(value)
            ctx.create_variable(target.id, var)
        else:
            var = build_stmt(ctx, target)
            try:
                var._assign(value)
            except AttributeError:
                raise QuadrantsSyntaxError(
                    f"Variable '{unparse(target).strip()}' cannot be assigned. Maybe it is not a Quadrants object?"
                )
        return var

    @staticmethod
    def build_NamedExpr(ctx: ASTTransformerFuncContext, node: ast.NamedExpr):
        build_stmt(ctx, node.value)
        is_static_assign = isinstance(node.value, ast.Call) and node.value.func.ptr is impl.static
        node.ptr = ASTTransformer.build_assign_basic(ctx, node.target, node.value.ptr, is_static_assign)
        return node.ptr

    @staticmethod
    def is_tuple(node):
        if isinstance(node, ast.Tuple):
            return True
        if isinstance(node, ast.Index) and isinstance(node.value.ptr, tuple):
            return True
        if isinstance(node.ptr, tuple):
            return True
        return False

    @staticmethod
    def build_Subscript(ctx: ASTTransformerFuncContext, node: ast.Subscript):
        build_stmt(ctx, node.value)
        build_stmt(ctx, node.slice)
        if not ASTTransformer.is_tuple(node.slice):
            node.slice.ptr = [node.slice.ptr]
        node.ptr = impl.subscript(ctx.ast_builder, node.value.ptr, *node.slice.ptr)
        node.violates_pure = node.value.violates_pure
        if node.violates_pure:
            node.violates_pure_reason = node.value.violates_pure_reason
        return node.ptr

    @staticmethod
    def build_Slice(ctx: ASTTransformerFuncContext, node: ast.Slice):
        if node.lower is not None:
            build_stmt(ctx, node.lower)
        if node.upper is not None:
            build_stmt(ctx, node.upper)
        if node.step is not None:
            build_stmt(ctx, node.step)

        node.ptr = slice(
            node.lower.ptr if node.lower else None,
            node.upper.ptr if node.upper else None,
            node.step.ptr if node.step else None,
        )
        return node.ptr

    @staticmethod
    def build_ExtSlice(ctx: ASTTransformerFuncContext, node: ast.ExtSlice):
        build_stmts(ctx, node.dims)
        node.ptr = tuple(dim.ptr for dim in node.dims)
        return node.ptr

    @staticmethod
    def build_Tuple(ctx: ASTTransformerFuncContext, node: ast.Tuple):
        build_stmts(ctx, node.elts)
        node.ptr = tuple(elt.ptr for elt in node.elts)
        return node.ptr

    @staticmethod
    def build_List(ctx: ASTTransformerFuncContext, node: ast.List):
        build_stmts(ctx, node.elts)
        reason = []
        for elt in node.elts:
            if elt.violates_pure:
                node.violates_pure = True
                reason.append("list member violates pure " + str(elt))
        node.ptr = [elt.ptr for elt in node.elts]
        node.violates_pure_reason = "\n".join(reason) if reason else None
        return node.ptr

    @staticmethod
    def build_Dict(ctx: ASTTransformerFuncContext, node: ast.Dict):
        dic = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                dic.update(build_stmt(ctx, value))
            else:
                dic[build_stmt(ctx, key)] = build_stmt(ctx, value)
        node.ptr = dic
        return node.ptr

    @staticmethod
    def process_listcomp(ctx: ASTTransformerFuncContext, node, result) -> None:
        result.append(build_stmt(ctx, node.elt))

    @staticmethod
    def process_dictcomp(ctx: ASTTransformerFuncContext, node, result) -> None:
        key = build_stmt(ctx, node.key)
        value = build_stmt(ctx, node.value)
        result[key] = value

    @staticmethod
    def process_generators(ctx: ASTTransformerFuncContext, node: ast.GeneratorExp, now_comp, func, result):
        if now_comp >= len(node.generators):
            return func(ctx, node, result)
        with ctx.static_scope_guard():
            _iter = build_stmt(ctx, node.generators[now_comp].iter)

        if isinstance(_iter, impl.Expr) and _iter.ptr.is_tensor():
            shape = _iter.ptr.get_shape()
            flattened = [Expr(x) for x in ctx.ast_builder.expand_exprs([_iter.ptr])]
            _iter = reshape_list(flattened, shape)

        for value in _iter:
            with ctx.variable_scope_guard():
                ASTTransformer.build_assign_unpack(ctx, node.generators[now_comp].target, value, True)
                with ctx.static_scope_guard():
                    build_stmts(ctx, node.generators[now_comp].ifs)
                ASTTransformer.process_ifs(ctx, node, now_comp, 0, func, result)
        return None

    @staticmethod
    def process_ifs(ctx: ASTTransformerFuncContext, node: ast.If, now_comp, now_if, func, result):
        if now_if >= len(node.generators[now_comp].ifs):
            return ASTTransformer.process_generators(ctx, node, now_comp + 1, func, result)
        cond = node.generators[now_comp].ifs[now_if].ptr
        if cond:
            ASTTransformer.process_ifs(ctx, node, now_comp, now_if + 1, func, result)

        return None

    @staticmethod
    def build_ListComp(ctx: ASTTransformerFuncContext, node: ast.ListComp):
        result = []
        ASTTransformer.process_generators(ctx, node, 0, ASTTransformer.process_listcomp, result)
        node.ptr = result
        return node.ptr

    @staticmethod
    def build_DictComp(ctx: ASTTransformerFuncContext, node: ast.DictComp):
        result = {}
        ASTTransformer.process_generators(ctx, node, 0, ASTTransformer.process_dictcomp, result)
        node.ptr = result
        return node.ptr

    @staticmethod
    def build_Index(ctx: ASTTransformerFuncContext, node: ast.Index):
        node.ptr = build_stmt(ctx, node.value)
        return node.ptr

    @staticmethod
    def build_Constant(ctx: ASTTransformerFuncContext, node: ast.Constant):
        node.ptr = node.value
        return node.ptr

    @staticmethod
    def build_keyword(ctx: ASTTransformerFuncContext, node: ast.keyword):
        build_stmt(ctx, node.value)
        if node.arg is None:
            node.ptr = node.value.ptr
        else:
            node.ptr = {node.arg: node.value.ptr}
        return node.ptr

    @staticmethod
    def build_Starred(ctx: ASTTransformerFuncContext, node: ast.Starred):
        node.ptr = build_stmt(ctx, node.value)
        return node.ptr

    @staticmethod
    def build_FormattedValue(ctx: ASTTransformerFuncContext, node: ast.FormattedValue):
        node.ptr = build_stmt(ctx, node.value)
        if node.format_spec is None or len(node.format_spec.values) == 0:
            return node.ptr
        values = node.format_spec.values
        assert len(values) == 1
        format_str = values[0].s
        assert format_str is not None
        # distinguished from normal list
        return ["__ti_fmt_value__", node.ptr, format_str]

    @staticmethod
    def build_JoinedStr(ctx: ASTTransformerFuncContext, node: ast.JoinedStr):
        str_spec = ""
        args = []
        for sub_node in node.values:
            if isinstance(sub_node, ast.FormattedValue):
                str_spec += "{}"
                args.append(build_stmt(ctx, sub_node))
            elif isinstance(sub_node, ast.Constant):
                str_spec += sub_node.value
            elif isinstance(sub_node, ast.Str):
                str_spec += sub_node.s
            else:
                raise QuadrantsSyntaxError("Invalid value for fstring.")

        args.insert(0, str_spec)
        node.ptr = impl.ti_format(*args)
        return node.ptr

    @staticmethod
    def build_Call(ctx: ASTTransformerFuncContext, node: ast.Call) -> Any | None:
        return CallTransformer.build_Call(ctx, node, build_stmt, build_stmts)

    @staticmethod
    def build_FunctionDef(ctx: ASTTransformerFuncContext, node: ast.FunctionDef) -> None:
        FunctionDefTransformer.build_FunctionDef(ctx, node, build_stmts)

    @staticmethod
    def build_Return(ctx: ASTTransformerFuncContext, node: ast.Return) -> None:
        if not ctx.is_real_function:
            if ctx.is_in_non_static_control_flow():
                raise QuadrantsSyntaxError("Return inside non-static if/for is not supported")
        if node.value is not None:
            build_stmt(ctx, node.value)
        if node.value is None or node.value.ptr is None:
            if not ctx.is_real_function:
                ctx.returned = ReturnStatus.ReturnedVoid
            return None
        if ctx.is_kernel or ctx.is_real_function:
            # TODO: check if it's at the end of a kernel, throw QuadrantsSyntaxError if not
            if ctx.func.return_type is None:
                raise QuadrantsSyntaxError(
                    f'A {"kernel" if ctx.is_kernel else "function"} '
                    "with a return value must be annotated "
                    "with a return type, e.g. def func() -> qd.f32"
                )
            return_exprs = []
            if len(ctx.func.return_type) == 1:
                node.value.ptr = [node.value.ptr]
            assert len(ctx.func.return_type) == len(node.value.ptr)
            for return_type, ptr in zip(ctx.func.return_type, node.value.ptr):
                if id(return_type) in primitive_types.type_ids:
                    if isinstance(ptr, Expr):
                        if ptr.is_tensor() or ptr.is_struct() or ptr.element_type() not in primitive_types.all_types:
                            raise QuadrantsRuntimeTypeError.get_ret(str(return_type), ptr)
                    elif not isinstance(ptr, (float, int, np.floating, np.integer)):
                        raise QuadrantsRuntimeTypeError.get_ret(str(return_type), ptr)
                    return_exprs += [qd_ops.cast(expr.Expr(ptr), return_type).ptr]
                elif isinstance(return_type, MatrixType):
                    values = ptr
                    if isinstance(values, Matrix):
                        if values.ndim != ctx.func.return_type.ndim:
                            raise QuadrantsRuntimeTypeError(
                                f"Return matrix ndim mismatch, expecting={return_type.ndim}, got={values.ndim}."
                            )
                        elif return_type.get_shape() != values.get_shape():
                            raise QuadrantsRuntimeTypeError(
                                f"Return matrix shape mismatch, expecting={return_type.get_shape()}, got={values.get_shape()}."
                            )
                        values = (
                            itertools.chain.from_iterable(values.to_list())
                            if values.ndim == 1
                            else iter(values.to_list())
                        )
                    elif isinstance(values, Expr):
                        if not values.is_tensor():
                            raise QuadrantsRuntimeTypeError.get_ret(return_type.to_string(), ptr)
                        elif (
                            return_type.dtype in primitive_types.real_types
                            and not values.element_type() in primitive_types.all_types
                        ):
                            raise QuadrantsRuntimeTypeError.get_ret(
                                return_type.dtype.to_string(), values.element_type()
                            )
                        elif (
                            return_type.dtype in primitive_types.integer_types
                            and not values.element_type() in primitive_types.integer_types
                        ):
                            raise QuadrantsRuntimeTypeError.get_ret(
                                return_type.dtype.to_string(), values.element_type()
                            )
                        elif len(values.get_shape()) != return_type.ndim:
                            raise QuadrantsRuntimeTypeError(
                                f"Return matrix ndim mismatch, expecting={return_type.ndim}, got={len(values.get_shape())}."
                            )
                        elif return_type.get_shape() != values.get_shape():
                            raise QuadrantsRuntimeTypeError(
                                f"Return matrix shape mismatch, expecting={return_type.get_shape()}, got={values.get_shape()}."
                            )
                        values = [values]
                    else:
                        np_array = np.array(values)
                        dt, shape, ndim = np_array.dtype, np_array.shape, np_array.ndim
                        if return_type.dtype in primitive_types.real_types and dt not in (
                            float,
                            int,
                            np.floating,
                            np.integer,
                        ):
                            raise QuadrantsRuntimeTypeError.get_ret(return_type.dtype.to_string(), dt)
                        elif return_type.dtype in primitive_types.integer_types and dt not in (int, np.integer):
                            raise QuadrantsRuntimeTypeError.get_ret(return_type.dtype.to_string(), dt)
                        elif ndim != return_type.ndim:
                            raise QuadrantsRuntimeTypeError(
                                f"Return matrix ndim mismatch, expecting={return_type.ndim}, got={ndim}."
                            )
                        elif return_type.get_shape() != shape:
                            raise QuadrantsRuntimeTypeError(
                                f"Return matrix shape mismatch, expecting={return_type.get_shape()}, got={shape}."
                            )
                        values = [values]
                    return_exprs += [qd_ops.cast(exp, return_type.dtype) for exp in values]
                elif isinstance(return_type, StructType):
                    if not isinstance(ptr, Struct) or not isinstance(ptr, return_type):
                        raise QuadrantsRuntimeTypeError.get_ret(str(return_type), ptr)
                    values = ptr
                    assert isinstance(values, Struct)
                    return_exprs += expr._get_flattened_ptrs(values)
                else:
                    raise QuadrantsSyntaxError("The return type is not supported now!")
            ctx.ast_builder.create_kernel_exprgroup_return(
                expr.make_expr_group(return_exprs), _ti_core.DebugInfo(ctx.get_pos_info(node))
            )
        else:
            ctx.return_data = node.value.ptr
            if ctx.func.return_type is not None:
                if len(ctx.func.return_type) == 1:
                    ctx.return_data = [ctx.return_data]
                for i, return_type in enumerate(ctx.func.return_type):
                    if id(return_type) in primitive_types.type_ids:
                        ctx.return_data[i] = qd_ops.cast(ctx.return_data[i], return_type)
                if len(ctx.func.return_type) == 1:
                    ctx.return_data = ctx.return_data[0]
        if not ctx.is_real_function:
            ctx.returned = ReturnStatus.ReturnedValue
        return None

    @staticmethod
    def build_Module(ctx: ASTTransformerFuncContext, node: ast.Module) -> None:
        with ctx.variable_scope_guard():
            # Do NOT use |build_stmts| which inserts 'del' statements to the
            # end and deletes parameters passed into the module
            for stmt in node.body:
                build_stmt(ctx, stmt)
        return None

    @staticmethod
    def build_attribute_if_is_dynamic_snode_method(ctx: ASTTransformerFuncContext, node) -> bool:
        is_subscript = isinstance(node.value, ast.Subscript)
        names = ("append", "deactivate", "length")
        if node.attr not in names:
            return False
        if is_subscript:
            x = node.value.value.ptr
            indices = node.value.slice.ptr
        else:
            x = node.value.ptr
            indices = []
        if not isinstance(x, Field):
            return False
        if not x.parent().ptr.type == _ti_core.SNodeType.dynamic:
            return False
        field_dim = x.snode.ptr.num_active_indices()
        indices_expr_group = make_expr_group(*indices)
        index_dim = indices_expr_group.size()
        if field_dim != index_dim + 1:
            return False
        if node.attr == "append":
            node.ptr = lambda val: append(x.parent(), indices, val)
        elif node.attr == "deactivate":
            node.ptr = lambda: deactivate(x.parent(), indices)
        else:
            node.ptr = lambda: length(x.parent(), indices)
        return True

    @staticmethod
    def build_Attribute(ctx: ASTTransformerFuncContext, node: ast.Attribute):
        # There are two valid cases for the methods of Dynamic SNode:
        #
        # 1. x[i, j].append (where the dimension of the field (3 in this case) is equal to one plus the number of the
        # indices (2 in this case) )
        #
        # 2. x.append (where the dimension of the field is one, equal to x[()].append)
        #
        # For the first case, the AST (simplified) is like node = Attribute(value=Subscript(value=x, slice=[i, j]),
        # attr="append"), when we build_stmt(node.value)(build the expression of the Subscript i.e. x[i, j]),
        # it should build the expression of node.value.value (i.e. x) and node.value.slice (i.e. [i, j]), and raise a
        # QuadrantsIndexError because the dimension of the field is not equal to the number of the indices. Therefore,
        # when we meet the error, we can detect whether it is a method of Dynamic SNode and build the expression if
        # it is by calling build_attribute_if_is_dynamic_snode_method. If we find that it is not a method of Dynamic
        # SNode, we raise the error again.
        #
        # For the second case, the AST (simplified) is like node = Attribute(value=x, attr="append"), and it does not
        # raise error when we build_stmt(node.value). Therefore, when we do not meet the error, we can also detect
        # whether it is a method of Dynamic SNode and build the expression if it is by calling
        # build_attribute_if_is_dynamic_snode_method. If we find that it is not a method of Dynamic SNode,
        # we continue to process it as a normal attribute node.
        from quadrants import math as qd_math  # pylint: disable=import-outside-toplevel

        try:
            build_stmt(ctx, node.value)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if isinstance(e, QuadrantsIndexError):
                node.value.ptr = None
                if ASTTransformer.build_attribute_if_is_dynamic_snode_method(ctx, node):
                    return node.ptr
            raise e

        if ASTTransformer.build_attribute_if_is_dynamic_snode_method(ctx, node):
            return node.ptr

        if isinstance(node.value.ptr, Expr) and not hasattr(node.value.ptr, node.attr):
            if node.attr in Matrix._swizzle_to_keygroup:
                keygroup = Matrix._swizzle_to_keygroup[node.attr]
                Matrix._keygroup_to_checker[keygroup](node.value.ptr, node.attr)
                attr_len = len(node.attr)
                if attr_len == 1:
                    node.ptr = Expr(
                        impl.get_runtime()
                        .compiling_callable.ast_builder()
                        .expr_subscript(
                            node.value.ptr.ptr,
                            make_expr_group(keygroup.index(node.attr)),
                            _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
                        )
                    )
                else:
                    node.ptr = Expr(
                        _ti_core.subscript_with_multiple_indices(
                            node.value.ptr.ptr,
                            [make_expr_group(keygroup.index(ch)) for ch in node.attr],
                            (attr_len,),
                            _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
                        )
                    )
            else:
                from quadrants.lang import (  # pylint: disable=C0415
                    matrix_ops as tensor_ops,
                )

                node.ptr = getattr(tensor_ops, node.attr)
                setattr(node, "caller", node.value.ptr)
        elif dataclasses.is_dataclass(node.value.ptr):
            node.ptr = next(field.type for field in dataclasses.fields(node.value.ptr))
        else:
            node.ptr = getattr(node.value.ptr, node.attr)
            node.violates_pure = node.value.violates_pure
            if node.violates_pure:
                node.violates_pure_reason = node.value.violates_pure_reason
            if ctx.is_pure and node.violates_pure and not ctx.static_scope_status.is_in_static_scope:
                if isinstance(node.ptr, (int, float, Field)):
                    violation = True
                    if violation and isinstance(node.ptr, enum.Enum):
                        violation = False
                    if violation and node.value.ptr in [qd_math, math, np]:
                        # ignore this built-in module
                        violation = False
                    if violation:
                        message = f"[PURE.VIOLATION] WARNING: Accessing global var {node.attr} from outside function scope within pure kernel {node.value.violates_pure_reason}"
                        if node.attr.upper() == node.attr:
                            warnings.warn(message)
                        else:
                            raise exception.QuadrantsCompilationError(message)
        return node.ptr

    @staticmethod
    def build_BinOp(ctx: ASTTransformerFuncContext, node: ast.BinOp):
        build_stmt(ctx, node.left)
        build_stmt(ctx, node.right)
        # pylint: disable-msg=C0415
        from quadrants.lang.matrix_ops import matmul

        op = {
            ast.Add: lambda l, r: l + r,
            ast.Sub: lambda l, r: l - r,
            ast.Mult: lambda l, r: l * r,
            ast.Div: lambda l, r: l / r,
            ast.FloorDiv: lambda l, r: l // r,
            ast.Mod: lambda l, r: l % r,
            ast.Pow: lambda l, r: l**r,
            ast.LShift: lambda l, r: l << r,
            ast.RShift: lambda l, r: l >> r,
            ast.BitOr: lambda l, r: l | r,
            ast.BitXor: lambda l, r: l ^ r,
            ast.BitAnd: lambda l, r: l & r,
            ast.MatMult: matmul,
        }.get(type(node.op))
        try:
            node.ptr = op(node.left.ptr, node.right.ptr)
        except TypeError as e:
            raise QuadrantsTypeError(str(e)) from None
        return node.ptr

    @staticmethod
    def build_AugAssign(ctx: ASTTransformerFuncContext, node: ast.AugAssign):
        build_stmt(ctx, node.target)
        build_stmt(ctx, node.value)
        if isinstance(node.target, ast.Name) and node.target.id in ctx.kernel_args:
            raise QuadrantsSyntaxError(
                f'Kernel argument "{node.target.id}" is immutable in the kernel. '
                f"If you want to change its value, please create a new variable."
            )
        node.ptr = node.target.ptr._augassign(node.value.ptr, type(node.op).__name__)
        return node.ptr

    @staticmethod
    def build_UnaryOp(ctx: ASTTransformerFuncContext, node: ast.UnaryOp):
        build_stmt(ctx, node.operand)
        op = {
            ast.UAdd: lambda l: l,
            ast.USub: lambda l: -l,
            ast.Not: qd_ops.logical_not,
            ast.Invert: lambda l: ~l,
        }.get(type(node.op))
        node.ptr = op(node.operand.ptr)
        return node.ptr

    @staticmethod
    def build_bool_op(op):
        def inner(operands):
            if len(operands) == 1:
                return operands[0].ptr
            return op(operands[0].ptr, inner(operands[1:]))

        return inner

    @staticmethod
    def build_static_and(operands):
        for operand in operands:
            if not operand.ptr:
                return operand.ptr
        return operands[-1].ptr

    @staticmethod
    def build_static_or(operands):
        for operand in operands:
            if operand.ptr:
                return operand.ptr
        return operands[-1].ptr

    @staticmethod
    def build_BoolOp(ctx: ASTTransformerFuncContext, node: ast.BoolOp):
        build_stmts(ctx, node.values)
        if ctx.is_in_static_scope():
            ops = {
                ast.And: ASTTransformer.build_static_and,
                ast.Or: ASTTransformer.build_static_or,
            }
        elif impl.get_runtime().short_circuit_operators:
            ops = {
                ast.And: ASTTransformer.build_bool_op(qd_ops.logical_and),
                ast.Or: ASTTransformer.build_bool_op(qd_ops.logical_or),
            }
        else:
            ops = {
                ast.And: ASTTransformer.build_bool_op(qd_ops.bit_and),
                ast.Or: ASTTransformer.build_bool_op(qd_ops.bit_or),
            }
        op = ops.get(type(node.op))
        node.ptr = op(node.values)
        return node.ptr

    @staticmethod
    def build_Compare(ctx: ASTTransformerFuncContext, node: ast.Compare):
        build_stmt(ctx, node.left)
        build_stmts(ctx, node.comparators)
        ops = {
            ast.Eq: lambda l, r: l == r,
            ast.NotEq: lambda l, r: l != r,
            ast.Lt: lambda l, r: l < r,
            ast.LtE: lambda l, r: l <= r,
            ast.Gt: lambda l, r: l > r,
            ast.GtE: lambda l, r: l >= r,
        }
        ops_static = {
            ast.In: lambda l, r: l in r,
            ast.NotIn: lambda l, r: l not in r,
        }
        if ctx.is_in_static_scope():
            ops = {**ops, **ops_static}
        operands = [node.left.ptr] + [comparator.ptr for comparator in node.comparators]
        val = True
        for i, node_op in enumerate(node.ops):
            if isinstance(node_op, (ast.Is, ast.IsNot)):
                name = "is" if isinstance(node_op, ast.Is) else "is not"
                raise QuadrantsSyntaxError(f'Operator "{name}" in Quadrants scope is not supported.')
            l = operands[i]
            r = operands[i + 1]
            op = ops.get(type(node_op))

            if op is None:
                if type(node_op) in ops_static:
                    raise QuadrantsSyntaxError(f'"{type(node_op).__name__}" is only supported inside `qd.static`.')
                else:
                    raise QuadrantsSyntaxError(f'"{type(node_op).__name__}" is not supported in Quadrants kernels.')
            val = qd_ops.logical_and(val, op(l, r))
        if not isinstance(val, (bool, np.bool_)):
            val = qd_ops.cast(val, primitive_types.u1)
        node.ptr = val
        return node.ptr

    @staticmethod
    def get_for_loop_targets(node: ast.Name | ast.Tuple | Any) -> list:
        """
        Returns the list of indices of the for loop |node|.
        See also: https://docs.python.org/3/library/ast.html#ast.For
        """
        if isinstance(node.target, ast.Name):
            return [node.target.id]
        assert isinstance(node.target, ast.Tuple)
        return [name.id for name in node.target.elts]

    @staticmethod
    def build_static_for(ctx: ASTTransformerFuncContext, node: ast.For, is_grouped: bool) -> None:
        ti_unroll_limit = impl.get_runtime().unrolling_limit
        if is_grouped:
            assert len(node.iter.args[0].args) == 1
            ndrange_arg = build_stmt(ctx, node.iter.args[0].args[0])
            if not isinstance(ndrange_arg, _Ndrange):
                raise QuadrantsSyntaxError("Only 'qd.ndrange' is allowed in 'qd.static(qd.grouped(...))'.")
            targets = ASTTransformer.get_for_loop_targets(node)
            if len(targets) != 1:
                raise QuadrantsSyntaxError(f"Group for should have 1 loop target, found {len(targets)}")
            target = targets[0]
            iter_time = 0
            alert_already = False

            for value in impl.grouped(ndrange_arg):
                iter_time += 1
                if not alert_already and ti_unroll_limit and iter_time > ti_unroll_limit:
                    alert_already = True
                    warnings.warn_explicit(
                        f"""You are unrolling more than
                        {ti_unroll_limit} iterations, so the compile time may be extremely long.
                        You can use a non-static for loop if you want to decrease the compile time.
                        You can disable this warning by setting qd.init(unrolling_limit=0).""",
                        SyntaxWarning,
                        ctx.file,
                        node.lineno + ctx.lineno_offset,
                        module="quadrants",
                    )

                with ctx.variable_scope_guard():
                    ctx.create_variable(target, value)
                    build_stmts(ctx, node.body)
                    status = ctx.loop_status()
                    if status == LoopStatus.Break:
                        break
                    elif status == LoopStatus.Continue:
                        ctx.set_loop_status(LoopStatus.Normal)
        else:
            build_stmt(ctx, node.iter)
            targets = ASTTransformer.get_for_loop_targets(node)

            iter_time = 0
            alert_already = False
            for target_values in node.iter.ptr:
                if not isinstance(target_values, collections.abc.Sequence) or len(targets) == 1:
                    target_values = [target_values]

                iter_time += 1
                if not alert_already and ti_unroll_limit and iter_time > ti_unroll_limit:
                    alert_already = True
                    warnings.warn_explicit(
                        f"""You are unrolling more than
                        {ti_unroll_limit} iterations, so the compile time may be extremely long.
                        You can use a non-static for loop if you want to decrease the compile time.
                        You can disable this warning by setting qd.init(unrolling_limit=0).""",
                        SyntaxWarning,
                        ctx.file,
                        node.lineno + ctx.lineno_offset,
                        module="quadrants",
                    )

                with ctx.variable_scope_guard():
                    for target, target_value in zip(targets, target_values):
                        ctx.create_variable(target, target_value)
                    build_stmts(ctx, node.body)
                    status = ctx.loop_status()
                    if status == LoopStatus.Break:
                        break
                    elif status == LoopStatus.Continue:
                        ctx.set_loop_status(LoopStatus.Normal)
        return None

    @staticmethod
    def build_range_for(ctx: ASTTransformerFuncContext, node: ast.For) -> None:
        with ctx.variable_scope_guard():
            loop_name = node.target.id
            ctx.check_loop_var(loop_name)
            loop_var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            ctx.create_variable(loop_name, loop_var)
            if len(node.iter.args) not in [1, 2]:
                raise QuadrantsSyntaxError(f"Range should have 1 or 2 arguments, found {len(node.iter.args)}")
            if len(node.iter.args) == 2:
                begin_expr = expr.Expr(build_stmt(ctx, node.iter.args[0]))
                end_expr = expr.Expr(build_stmt(ctx, node.iter.args[1]))

                # Warning for implicit dtype conversion
                boundary_type_cast_warning(begin_expr)
                boundary_type_cast_warning(end_expr)

                begin = qd_ops.cast(begin_expr, primitive_types.i32)
                end = qd_ops.cast(end_expr, primitive_types.i32)

            else:
                end_expr = expr.Expr(build_stmt(ctx, node.iter.args[0]))

                # Warning for implicit dtype conversion
                boundary_type_cast_warning(end_expr)

                begin = qd_ops.cast(expr.Expr(0), primitive_types.i32)
                end = qd_ops.cast(end_expr, primitive_types.i32)

            for_di = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr, end.ptr, for_di)
            ctx.loop_depth += 1
            build_stmts(ctx, node.body)
            ctx.loop_depth -= 1
            ctx.ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_ndrange_for(ctx: ASTTransformerFuncContext, node: ast.For) -> None:
        with ctx.variable_scope_guard():
            ndrange_var = impl.expr_init(build_stmt(ctx, node.iter))
            ndrange_begin = qd_ops.cast(expr.Expr(0), primitive_types.i32)
            ndrange_end = qd_ops.cast(
                expr.Expr(impl.subscript(ctx.ast_builder, ndrange_var.acc_dimensions, 0)),
                primitive_types.i32,
            )
            ndrange_loop_var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            for_di = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_range_for(ndrange_loop_var.ptr, ndrange_begin.ptr, ndrange_end.ptr, for_di)
            I = impl.expr_init(ndrange_loop_var)
            targets = ASTTransformer.get_for_loop_targets(node)
            if len(targets) != len(ndrange_var.dimensions):
                raise QuadrantsSyntaxError(
                    "Ndrange for loop with number of the loop variables not equal to "
                    "the dimension of the ndrange is not supported. "
                    "Please check if the number of arguments of qd.ndrange() is equal to "
                    "the number of the loop variables."
                )
            for i, target in enumerate(targets):
                if i + 1 < len(targets):
                    target_tmp = impl.expr_init(I // ndrange_var.acc_dimensions[i + 1])
                else:
                    target_tmp = impl.expr_init(I)
                ctx.create_variable(
                    target,
                    impl.expr_init(
                        target_tmp
                        + impl.subscript(
                            ctx.ast_builder,
                            impl.subscript(ctx.ast_builder, ndrange_var.bounds, i),
                            0,
                        )
                    ),
                )
                if i + 1 < len(targets):
                    I._assign(I - target_tmp * ndrange_var.acc_dimensions[i + 1])
            ctx.loop_depth += 1
            build_stmts(ctx, node.body)
            ctx.loop_depth -= 1
            ctx.ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_grouped_ndrange_for(ctx: ASTTransformerFuncContext, node: ast.For) -> None:
        with ctx.variable_scope_guard():
            ndrange_var = impl.expr_init(build_stmt(ctx, node.iter.args[0]))
            ndrange_begin = qd_ops.cast(expr.Expr(0), primitive_types.i32)
            ndrange_end = qd_ops.cast(
                expr.Expr(impl.subscript(ctx.ast_builder, ndrange_var.acc_dimensions, 0)),
                primitive_types.i32,
            )
            ndrange_loop_var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            for_di = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_range_for(ndrange_loop_var.ptr, ndrange_begin.ptr, ndrange_end.ptr, for_di)

            targets = ASTTransformer.get_for_loop_targets(node)
            if len(targets) != 1:
                raise QuadrantsSyntaxError(f"Group for should have 1 loop target, found {len(targets)}")
            target = targets[0]
            mat = matrix.make_matrix([0] * len(ndrange_var.dimensions), dt=primitive_types.i32)
            target_var = impl.expr_init(mat)

            ctx.create_variable(target, target_var)
            I = impl.expr_init(ndrange_loop_var)
            for i in range(len(ndrange_var.dimensions)):
                if i + 1 < len(ndrange_var.dimensions):
                    target_tmp = I // ndrange_var.acc_dimensions[i + 1]
                else:
                    target_tmp = I
                impl.subscript(ctx.ast_builder, target_var, i)._assign(target_tmp + ndrange_var.bounds[i][0])
                if i + 1 < len(ndrange_var.dimensions):
                    I._assign(I - target_tmp * ndrange_var.acc_dimensions[i + 1])
            ctx.loop_depth += 1
            build_stmts(ctx, node.body)
            ctx.loop_depth -= 1
            ctx.ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_struct_for(ctx: ASTTransformerFuncContext, node: ast.For, is_grouped: bool) -> None:
        # for i, j in x
        # for I in qd.grouped(x)
        targets = ASTTransformer.get_for_loop_targets(node)

        for target in targets:
            ctx.check_loop_var(target)

        with ctx.variable_scope_guard():
            if is_grouped:
                if len(targets) != 1:
                    raise QuadrantsSyntaxError(f"Group for should have 1 loop target, found {len(targets)}")
                target = targets[0]
                loop_var = build_stmt(ctx, node.iter)
                loop_indices = expr.make_var_list(size=len(loop_var.shape), ast_builder=ctx.ast_builder)
                expr_group = expr.make_expr_group(loop_indices)
                impl.begin_frontend_struct_for(ctx.ast_builder, expr_group, loop_var)
                ctx.create_variable(target, matrix.make_matrix(loop_indices, dt=primitive_types.i32))
                build_stmts(ctx, node.body)
                ctx.ast_builder.end_frontend_struct_for()
            else:
                _vars = []
                for name in targets:
                    var = expr.Expr(ctx.ast_builder.make_id_expr(""))
                    _vars.append(var)
                    ctx.create_variable(name, var)
                loop_var = node.iter.ptr
                expr_group = expr.make_expr_group(*_vars)
                impl.begin_frontend_struct_for(ctx.ast_builder, expr_group, loop_var)
                ctx.loop_depth += 1
                build_stmts(ctx, node.body)
                ctx.loop_depth -= 1
                ctx.ast_builder.end_frontend_struct_for()
        return None

    @staticmethod
    def build_mesh_for(ctx: ASTTransformerFuncContext, node: ast.For) -> None:
        targets = ASTTransformer.get_for_loop_targets(node)
        if len(targets) != 1:
            raise QuadrantsSyntaxError("Mesh for should have 1 loop target, found {len(targets)}")
        target = targets[0]

        with ctx.variable_scope_guard():
            var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            ctx.mesh = node.iter.ptr.mesh
            assert isinstance(ctx.mesh, impl.MeshInstance)
            mesh_idx = mesh.MeshElementFieldProxy(ctx.mesh, node.iter.ptr._type, var.ptr)
            ctx.create_variable(target, mesh_idx)
            ctx.ast_builder.begin_frontend_mesh_for(
                mesh_idx.ptr,
                ctx.mesh.mesh_ptr,
                node.iter.ptr._type,
                _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )
            ctx.loop_depth += 1
            build_stmts(ctx, node.body)
            ctx.loop_depth -= 1
            ctx.mesh = None
            ctx.ast_builder.end_frontend_mesh_for()
        return None

    @staticmethod
    def build_nested_mesh_for(ctx: ASTTransformerFuncContext, node: ast.For) -> None:
        targets = ASTTransformer.get_for_loop_targets(node)
        if len(targets) != 1:
            raise QuadrantsSyntaxError("Nested-mesh for should have 1 loop target, found {len(targets)}")
        target = targets[0]

        with ctx.variable_scope_guard():
            ctx.mesh = node.iter.ptr.mesh
            assert isinstance(ctx.mesh, impl.MeshInstance)
            loop_name = node.target.id + "_index__"
            loop_var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            ctx.create_variable(loop_name, loop_var)
            begin = expr.Expr(0)
            end = qd_ops.cast(node.iter.ptr.size, primitive_types.i32)
            for_di = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr, end.ptr, for_di)
            entry_expr = _ti_core.get_relation_access(
                ctx.mesh.mesh_ptr,
                node.iter.ptr.from_index.ptr,
                node.iter.ptr.to_element_type,
                loop_var.ptr,
            )
            entry_expr.type_check(impl.get_runtime().prog.config())
            mesh_idx = mesh.MeshElementFieldProxy(ctx.mesh, node.iter.ptr.to_element_type, entry_expr)
            ctx.create_variable(target, mesh_idx)
            ctx.loop_depth += 1
            build_stmts(ctx, node.body)
            ctx.loop_depth -= 1
            ctx.ast_builder.end_frontend_range_for()

        return None

    @staticmethod
    def build_For(ctx: ASTTransformerFuncContext, node: ast.For) -> None:
        if node.orelse:
            raise QuadrantsSyntaxError("'else' clause for 'for' not supported in Quadrants kernels")
        decorator = get_decorator(ctx, node.iter)
        double_decorator = ""
        if decorator != "" and len(node.iter.args) == 1:
            double_decorator = get_decorator(ctx, node.iter.args[0])

        if decorator == "static":
            if double_decorator == "static":
                raise QuadrantsSyntaxError("'qd.static' cannot be nested")
            with ctx.loop_scope_guard(is_static=True):
                return ASTTransformer.build_static_for(ctx, node, double_decorator == "grouped")
        with ctx.loop_scope_guard():
            if decorator == "ndrange":
                if double_decorator != "":
                    raise QuadrantsSyntaxError("No decorator is allowed inside 'qd.ndrange")
                return ASTTransformer.build_ndrange_for(ctx, node)
            if decorator == "grouped":
                if double_decorator == "static":
                    raise QuadrantsSyntaxError("'qd.static' is not allowed inside 'qd.grouped'")
                elif double_decorator == "ndrange":
                    return ASTTransformer.build_grouped_ndrange_for(ctx, node)
                elif double_decorator == "grouped":
                    raise QuadrantsSyntaxError("'qd.grouped' cannot be nested")
                else:
                    return ASTTransformer.build_struct_for(ctx, node, is_grouped=True)
            elif (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                if ctx.loop_depth > 0 and ctx.autodiff_mode == AutodiffMode.REVERSE and not ctx.adstack_enabled:
                    raise exception.QuadrantsCompilationError("Cannot use non static range in Backwards mode")
                return ASTTransformer.build_range_for(ctx, node)
            elif isinstance(node.iter, ast.IfExp):
                # Handle inline if expression as the top level iterator expression, e.g.:
                #
                #   for i in range(foo) if qd.static(some_flag) else qd.static(range(bar))
                #
                # Empirically, this appears to generalize to:
                # - being an inner loop
                # - either side can be static or not, as long as the if expression itself is static
                _iter = node.iter
                is_static_if = get_decorator(ctx, node.iter.test) == "static"
                if not is_static_if:
                    raise QuadrantsSyntaxError(
                        "Using non static inlined if statement as for-loop iterable is not currently supported."
                    )
                build_stmt(ctx, _iter.test)
                next_iter = _iter.body if _iter.test.ptr else _iter.orelse
                new_for = ast.For(
                    target=node.target,
                    iter=next_iter,
                    body=node.body,
                    orelse=None,
                    type_comment=getattr(node, "type_comment", None),
                    lineno=node.lineno,
                    end_lineno=node.end_lineno,
                    col_offset=node.col_offset,
                    end_col_offset=node.end_col_offset,
                )
                return ASTTransformer.build_For(ctx, new_for)
            else:
                build_stmt(ctx, node.iter)
                if isinstance(node.iter.ptr, mesh.MeshElementField):
                    if not _ti_core.is_extension_supported(impl.default_cfg().arch, _ti_core.Extension.mesh):
                        raise Exception(
                            "Backend " + str(impl.default_cfg().arch) + " doesn't support MeshQuadrants extension"
                        )
                    return ASTTransformer.build_mesh_for(ctx, node)
                if isinstance(node.iter.ptr, mesh.MeshRelationAccessProxy):
                    return ASTTransformer.build_nested_mesh_for(ctx, node)
                # Struct for
                return ASTTransformer.build_struct_for(ctx, node, is_grouped=False)

    @staticmethod
    def build_While(ctx: ASTTransformerFuncContext, node: ast.While) -> None:
        if node.orelse:
            raise QuadrantsSyntaxError("'else' clause for 'while' not supported in Quadrants kernels")

        with ctx.loop_scope_guard():
            stmt_dbg_info = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_while(expr.Expr(1, dtype=primitive_types.i32).ptr, stmt_dbg_info)
            while_cond = build_stmt(ctx, node.test)
            impl.begin_frontend_if(ctx.ast_builder, while_cond, stmt_dbg_info)
            ctx.ast_builder.begin_frontend_if_true()
            ctx.ast_builder.pop_scope()
            ctx.ast_builder.begin_frontend_if_false()
            ctx.ast_builder.insert_break_stmt(stmt_dbg_info)
            ctx.ast_builder.pop_scope()
            build_stmts(ctx, node.body)
            ctx.ast_builder.pop_scope()
        return None

    @staticmethod
    def build_If(ctx: ASTTransformerFuncContext, node: ast.If) -> ast.If | None:
        build_stmt(ctx, node.test)
        is_static_if = get_decorator(ctx, node.test) == "static"

        if is_static_if:
            if node.test.ptr:
                build_stmts(ctx, node.body)
            else:
                build_stmts(ctx, node.orelse)
            return node

        with ctx.non_static_if_guard(node):
            stmt_dbg_info = _ti_core.DebugInfo(ctx.get_pos_info(node))
            impl.begin_frontend_if(ctx.ast_builder, node.test.ptr, stmt_dbg_info)
            ctx.ast_builder.begin_frontend_if_true()
            build_stmts(ctx, node.body)
            ctx.ast_builder.pop_scope()
            ctx.ast_builder.begin_frontend_if_false()
            build_stmts(ctx, node.orelse)
            ctx.ast_builder.pop_scope()
        return None

    @staticmethod
    def build_Expr(ctx: ASTTransformerFuncContext, node: ast.Expr) -> None:
        build_stmt(ctx, node.value)
        return None

    @staticmethod
    def build_IfExp(ctx: ASTTransformerFuncContext, node: ast.IfExp):
        build_stmt(ctx, node.test)
        build_stmt(ctx, node.body)
        build_stmt(ctx, node.orelse)

        has_tensor_type = False
        if isinstance(node.test.ptr, expr.Expr) and node.test.ptr.is_tensor():
            has_tensor_type = True
        if isinstance(node.body.ptr, expr.Expr) and node.body.ptr.is_tensor():
            has_tensor_type = True
        if isinstance(node.orelse.ptr, expr.Expr) and node.orelse.ptr.is_tensor():
            has_tensor_type = True

        if has_tensor_type:
            if isinstance(node.test.ptr, expr.Expr) and node.test.ptr.is_tensor():
                raise QuadrantsSyntaxError(
                    "Using conditional expression for element-wise select operation on "
                    "Quadrants vectors/matrices is deprecated and removed starting from Quadrants v1.5.0 "
                    'Please use "qd.select" instead.'
                )
            node.ptr = qd_ops.select(node.test.ptr, node.body.ptr, node.orelse.ptr)
            return node.ptr

        is_static_if = get_decorator(ctx, node.test) == "static"

        if is_static_if:
            if node.test.ptr:
                node.ptr = build_stmt(ctx, node.body)
            else:
                node.ptr = build_stmt(ctx, node.orelse)
            return node.ptr

        node.ptr = qd_ops.ifte(node.test.ptr, node.body.ptr, node.orelse.ptr)
        return node.ptr

    @staticmethod
    def _is_string_mod_args(msg) -> bool:
        # 1. str % (a, b, c, ...)
        # 2. str % single_item
        # Note that |msg.right| may not be a tuple.
        if not isinstance(msg, ast.BinOp):
            return False
        if not isinstance(msg.op, ast.Mod):
            return False
        if isinstance(msg.left, ast.Str):
            return True
        if isinstance(msg.left, ast.Constant) and isinstance(msg.left.value, str):
            return True
        return False

    @staticmethod
    def _handle_string_mod_args(ctx: ASTTransformerFuncContext, node):
        msg = build_stmt(ctx, node.left)
        args = build_stmt(ctx, node.right)
        if not isinstance(args, collections.abc.Sequence):
            args = (args,)
        args = [expr.Expr(x).ptr for x in args]
        return msg, args

    @staticmethod
    def ti_format_list_to_assert_msg(raw) -> tuple[str, list]:
        # TODO: ignore formats here for now
        entries, _ = impl.ti_format_list_to_content_entries([raw])
        msg = ""
        args = []
        for entry in entries:
            if isinstance(entry, str):
                msg += entry
            elif isinstance(entry, _ti_core.ExprCxx):
                ty = entry.get_rvalue_type()
                if ty in primitive_types.real_types:
                    msg += "%f"
                elif ty in primitive_types.integer_types:
                    msg += "%d"
                else:
                    raise QuadrantsSyntaxError(f"Unsupported data type: {type(ty)}")
                args.append(entry)
            else:
                raise QuadrantsSyntaxError(f"Unsupported type: {type(entry)}")
        return msg, args

    @staticmethod
    def build_Assert(ctx: ASTTransformerFuncContext, node: ast.Assert) -> None:
        u = platform.uname()
        if u.system == "linux" and u.machine in ("arm64", "aarch64"):
            build_stmt(ctx, node.test)
            warnings.warn("Assert not supported on linux arm64 currently")
            return None
        extra_args = []
        if node.msg is not None:
            if ASTTransformer._is_string_mod_args(node.msg):
                msg, extra_args = ASTTransformer._handle_string_mod_args(ctx, node.msg)
            else:
                msg = build_stmt(ctx, node.msg)
                if isinstance(node.msg, ast.Constant):
                    msg = str(msg)
                elif isinstance(node.msg, ast.Str):
                    pass
                elif isinstance(msg, collections.abc.Sequence) and len(msg) > 0 and msg[0] == "__ti_format__":
                    msg, extra_args = ASTTransformer.ti_format_list_to_assert_msg(msg)
                else:
                    raise QuadrantsSyntaxError(f"assert info must be constant or formatted string, not {type(msg)}")
        else:
            msg = unparse(node.test)
        test = build_stmt(ctx, node.test)
        impl.ti_assert(test, msg.strip(), extra_args, _ti_core.DebugInfo(ctx.get_pos_info(node)))
        return None

    @staticmethod
    def build_Break(ctx: ASTTransformerFuncContext, node: ast.Break) -> None:
        if ctx.is_in_static_for():
            nearest_non_static_if = ctx.current_loop_scope().nearest_non_static_if
            if nearest_non_static_if:
                msg = ctx.get_pos_info(nearest_non_static_if.test)
                msg += (
                    "You are trying to `break` a static `for` loop, "
                    "but the `break` statement is inside a non-static `if`. "
                )
                raise QuadrantsSyntaxError(msg)
            ctx.set_loop_status(LoopStatus.Break)
        else:
            ctx.ast_builder.insert_break_stmt(_ti_core.DebugInfo(ctx.get_pos_info(node)))
        return None

    @staticmethod
    def build_Continue(ctx: ASTTransformerFuncContext, node: ast.Continue) -> None:
        if ctx.is_in_static_for():
            nearest_non_static_if = ctx.current_loop_scope().nearest_non_static_if
            if nearest_non_static_if:
                msg = ctx.get_pos_info(nearest_non_static_if.test)
                msg += (
                    "You are trying to `continue` a static `for` loop, "
                    "but the `continue` statement is inside a non-static `if`. "
                )
                raise QuadrantsSyntaxError(msg)
            ctx.set_loop_status(LoopStatus.Continue)
        else:
            ctx.ast_builder.insert_continue_stmt(_ti_core.DebugInfo(ctx.get_pos_info(node)))
        return None

    @staticmethod
    def build_Pass(ctx: ASTTransformerFuncContext, node: ast.Pass) -> None:
        return None


build_stmt = ASTTransformer()


def build_stmts(ctx: ASTTransformerFuncContext, stmts: list[ast.stmt]):
    # TODO: Should we just make this part of ASTTransformer? Then, easier to pass around (just
    # pass the ASTTransformer object around)
    with ctx.variable_scope_guard():
        for stmt in stmts:
            if ctx.returned != ReturnStatus.NoReturn or ctx.loop_status() != LoopStatus.Normal:
                break
            else:
                build_stmt(ctx, stmt)
    return stmts
