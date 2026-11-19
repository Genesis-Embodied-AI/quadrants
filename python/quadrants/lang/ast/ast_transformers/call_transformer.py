# type: ignore

import ast
import inspect
import math
import operator
import re
import warnings
from ast import unparse
from collections import ChainMap
from contextlib import nullcontext
from typing import Any

import numpy as np

from quadrants.lang import (
    expr,
    impl,
    matrix,
)
from quadrants.lang import ops as qd_ops
from quadrants.lang._runtime_typing.runtime_dataclass import dataclass_call_arguments
from quadrants.lang.ast.ast_transformer_utils import (
    FuncCompilePassContext,
    get_decorator,
)
from quadrants.lang.exception import (
    QuadrantsRuntimeTypeError,
    QuadrantsSyntaxError,
    QuadrantsTypeError,
)
from quadrants.lang.expr import Expr
from quadrants.lang.matrix import Matrix, Vector
from quadrants.lang.util import is_quadrants_class
from quadrants.types import primitive_types

from ..._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable


class CallTransformer:
    @staticmethod
    def _build_call_if_is_builtin(ctx: FuncCompilePassContext, node, args, keywords):
        from quadrants.lang import matrix_ops  # pylint: disable=C0415

        func = node.func.ptr
        replace_func = {
            id(print): impl.qd_print,
            id(min): qd_ops.min,
            id(max): qd_ops.max,
            id(int): impl.qd_int,
            id(bool): impl.qd_bool,
            id(float): impl.qd_float,
            id(any): matrix_ops.any,
            id(all): matrix_ops.all,
            id(abs): abs,
            id(pow): pow,
            id(operator.matmul): matrix_ops.matmul,
        }

        # Builtin 'len' function on Matrix Expr
        if id(func) == id(len) and len(args) == 1:
            if isinstance(args[0], Expr) and args[0].ptr.is_tensor():
                node.ptr = args[0].get_shape()[0]
                return True

        if id(func) in replace_func:
            node.ptr = replace_func[id(func)](*args, **keywords)
            return True
        return False

    @staticmethod
    def _build_call_if_is_type(ctx: FuncCompilePassContext, node, args, keywords):
        func = node.func.ptr
        if id(func) in primitive_types.type_ids:
            if len(args) != 1 or keywords:
                raise QuadrantsSyntaxError("A primitive type can only decorate a single expression.")
            if is_quadrants_class(args[0]):
                raise QuadrantsSyntaxError("A primitive type cannot decorate an expression with a compound type.")

            if isinstance(args[0], expr.Expr):
                if args[0].ptr.is_tensor():
                    raise QuadrantsSyntaxError("A primitive type cannot decorate an expression with a compound type.")
                node.ptr = qd_ops.cast(args[0], func)
            else:
                node.ptr = expr.Expr(args[0], dtype=func)
            return True
        return False

    @staticmethod
    def _is_external_func(ctx: FuncCompilePassContext, func) -> bool:
        if ctx.is_in_static_scope():  # allow external function in static scope
            return False
        if hasattr(func, "_is_quadrants_function") or hasattr(func, "_is_wrapped_kernel"):  # quadrants func/kernel
            return False
        if hasattr(func, "__module__") and func.__module__ and func.__module__.startswith("quadrants."):
            return False
        return True

    @staticmethod
    def _warn_if_is_external_func(ctx: FuncCompilePassContext, node):
        func = node.func.ptr
        if not CallTransformer._is_external_func(ctx, func):
            return
        name = unparse(node.func).strip()
        warnings.warn_explicit(
            f"\x1b[38;5;226m"  # Yellow
            f'Calling non-quadrants function "{name}". '
            f"Scope inside the function is not processed by the Quadrants AST transformer. "
            f"The function may not work as expected. Proceed with caution! "
            f"Maybe you can consider turning it into a @qd.func?"
            f"\x1b[0m",  # Reset
            SyntaxWarning,
            ctx.file,
            node.lineno + ctx.lineno_offset,
            module="quadrants",
        )

    @staticmethod
    # Parses a formatted string and extracts format specifiers from it, along with positional and keyword arguments.
    # This function produces a canonicalized formatted string that includes solely empty replacement fields, e.g. 'qwerty {} {} {} {} {}'.
    # Note that the arguments can be used multiple times in the string.
    # e.g.:
    # origin input: 'qwerty {1} {} {1:.3f} {k:.4f} {k:}'.format(1.0, 2.0, k=k)
    # raw_string: 'qwerty {1} {} {1:.3f} {k:.4f} {k:}'
    # raw_args: [1.0, 2.0]
    # raw_keywords: {'k': <qd.Expr>}
    # return value: ['qwerty {} {} {} {} {}', 2.0, 1.0, ['__qd_fmt_value__', 2.0, '.3f'], ['__qd_fmt_value__', <qd.Expr>, '.4f'], <qd.Expr>]
    def _canonicalize_formatted_string(raw_string: str, *raw_args: list, **raw_keywords: dict):
        raw_brackets = re.findall(r"{(.*?)}", raw_string)
        brackets = []
        unnamed = 0
        for bracket in raw_brackets:
            item, spec = bracket.split(":") if ":" in bracket else (bracket, None)
            if item.isdigit():
                item = int(item)
            # handle unnamed positional args
            if item == "":
                item = unnamed
                unnamed += 1
            # handle empty spec
            if spec == "":
                spec = None
            brackets.append((item, spec))

        # check for errors in the arguments
        max_args_index = max([t[0] for t in brackets if isinstance(t[0], int)], default=-1)
        if max_args_index + 1 != len(raw_args):
            raise QuadrantsSyntaxError(
                f"Expected {max_args_index + 1} positional argument(s), but received {len(raw_args)} instead."
            )
        brackets_keywords = [t[0] for t in brackets if isinstance(t[0], str)]
        for item in brackets_keywords:
            if item not in raw_keywords:
                raise QuadrantsSyntaxError(f"Keyword '{item}' not found.")
        for item in raw_keywords:
            if item not in brackets_keywords:
                raise QuadrantsSyntaxError(f"Keyword '{item}' not used.")

        # reorganize the arguments based on their positions, keywords, and format specifiers
        args = []
        for item, spec in brackets:
            new_arg = raw_args[item] if isinstance(item, int) else raw_keywords[item]
            if spec is not None:
                args.append(["__qd_fmt_value__", new_arg, spec])
            else:
                args.append(new_arg)
        args.insert(0, re.sub(r"{.*?}", "{}", raw_string))
        return args

    @staticmethod
    def build_Call(ctx: FuncCompilePassContext, node: ast.Call, build_stmt, build_stmts) -> Any | None:
        """
        example ast:
        Call(func=Name(id='f2', ctx=Load()), args=[Name(id='my_struct_ab', ctx=Load())], keywords=[])
        """
        is_static = get_decorator(ctx, node) in ("static", "static_assert")

        with ctx.static_scope_guard() if is_static else nullcontext():
            build_stmt(ctx, node.func)
            # creates variable for the dataclass itself (as well as other variables,
            # not related to dataclasses). Necessary for calling further child functions
            build_stmts(ctx, node.args)
            build_stmts(ctx, node.keywords)
        func = node.func.ptr
        # A ``@qd.func(requires_top_level=True)`` may only be called at the top level of a kernel (or
        # directly inside a ``graph_do_while`` body). Nesting it in a runtime for / if / while demotes
        # its phase loops out of top-level position, collapsing the per-phase grid-wide barriers and
        # corrupting the result, so reject it here at trace time rather than miscompiling silently.
        # ``qd.static`` (compile-time) loops do not set ``is_in_non_static_control_flow``. The
        # ``_including_caller`` variant also catches the marked func being reached through an unmarked
        # intermediate ``@qd.func`` that was itself called from runtime control flow.
        if getattr(func, "_qd_requires_top_level", False) and ctx.is_in_non_static_control_flow_including_caller():
            func_name = getattr(func, "__name__", None) or "this @qd.func"
            raise QuadrantsSyntaxError(
                f"`{func_name}` is decorated with requires_top_level=True and must be called at the top "
                f"level of a @qd.kernel (or directly inside a `while qd.graph.do_while(...)` body). It "
                f"cannot be nested inside a runtime for / if / while: that demotes its phase loops out of "
                f"top-level position, collapsing the per-phase grid-wide barriers and corrupting the result."
            )
        pruning = ctx.kernel_compile_context.pruning
        node_args, node_keywords = node.args, node.keywords

        # Check for pure violations.
        # We have to do this after building the statements.
        # If any arg violates pure, then node also violates pure.
        for arg in node_args:
            if arg.violates_pure:
                node.violates_pure_reason = arg.violates_pure_reason
                node.violates_pure = True

        for kw in node_keywords:
            if kw.value.violates_pure:
                node.violates_pure = True
                node.violates_pure_reason = kw.value.violates_pure_reason

        py_args = []
        for arg in node_args:
            if type(arg) is ast.Starred:
                arg_list = arg.ptr
                if type(arg_list) is Expr and arg_list.is_tensor():
                    # Expand Expr with Matrix-type return into list of Exprs
                    arg_list = [Expr(x) for x in ctx.ast_builder.expand_exprs([arg_list.ptr])]

                for i in arg_list:
                    py_args.append(i)
            else:
                py_args.append(arg.ptr)
        py_kwargs = dict(ChainMap(*[keyword.ptr for keyword in node_keywords]))

        if id(func) in [id(print), id(impl.qd_print)]:
            ctx.func.has_print = True

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value.ptr, str) and node.func.attr == "format":
            raw_string = node.func.value.ptr
            py_args = CallTransformer._canonicalize_formatted_string(raw_string, *py_args, **py_kwargs)
            node.ptr = impl.qd_format(*py_args)
            return node.ptr

        if id(func) == id(Matrix) or id(func) == id(Vector):
            node.ptr = matrix.make_matrix(*py_args, **py_kwargs)
            return node.ptr

        if CallTransformer._build_call_if_is_builtin(ctx, node, py_args, py_kwargs):
            return node.ptr

        if CallTransformer._build_call_if_is_type(ctx, node, py_args, py_kwargs):
            return node.ptr

        if hasattr(node.func, "caller"):
            node.ptr = func(node.func.caller, *py_args, **py_kwargs)
            return node.ptr

        CallTransformer._warn_if_is_external_func(ctx, node)
        # Snapshot the caller's `loop_depth` on the global context so a `@qd.func` invocation routed through
        # `func(*py_args, **py_kwargs)` below can seed its fresh `FuncCompilePassContext` from this value in
        # `_func_base.py` and inherit any outer for-loops we are already inside. Save/restore unconditionally because
        # we cannot tell from this site whether `func` is a Quadrants func, a builtin, or a Python callable; the
        # snapshot is harmless for non-func callables and nests correctly through chained calls.
        current_pass = ctx.kernel_compile_context.current_pass
        prev_caller_loop_depth = current_pass.caller_loop_depth
        current_pass.caller_loop_depth = ctx.loop_depth
        # Likewise snapshot whether we are (transitively) inside non-static control flow, so a
        # `requires_top_level=True` func reached through an unmarked intermediate `@qd.func` still sees the caller's
        # control-flow context. OR with the existing value so it accumulates down a multi-level call chain.
        prev_caller_in_non_static_control_flow = current_pass.caller_in_non_static_control_flow
        current_pass.caller_in_non_static_control_flow = ctx.is_in_non_static_control_flow_including_caller()
        try:
            try:
                pruning = ctx.kernel_compile_context.pruning
                with (
                    dataclass_call_arguments(func, py_args, py_kwargs, isinstance(func, BoundQuadrantsCallable))
                    if isinstance(func, (QuadrantsCallable, BoundQuadrantsCallable)) and func._is_quadrants_function
                    else nullcontext()
                ):
                    node.ptr = func(*py_args, **py_kwargs)

                if not pruning.enforcing:
                    pruning.record_after_call(ctx, func, node, node_args, node_keywords)
            except QuadrantsRuntimeTypeError:
                raise
            except TypeError as e:
                module = inspect.getmodule(func)
                error_msg = re.sub(r"\bExpr\b", "Quadrants Expression", str(e))
                func_name = getattr(func, "__name__", func.__class__.__name__)
                msg = f"TypeError when calling `{func_name}`: {error_msg}."
                if CallTransformer._is_external_func(ctx, node.func.ptr):
                    args_has_expr = any([isinstance(arg, Expr) for arg in py_args])
                    if args_has_expr and (module == math or module == np):
                        exec_str = f"from quadrants import {func.__name__}"
                        try:
                            exec(exec_str, {})
                        except:
                            pass
                        else:
                            msg += f"\nDid you mean to use `qd.{func.__name__}` instead of `{module.__name__}.{func.__name__}`?"
                raise QuadrantsTypeError(msg)
        finally:
            current_pass.caller_loop_depth = prev_caller_loop_depth
            current_pass.caller_in_non_static_control_flow = prev_caller_in_non_static_control_flow

        if getattr(func, "_is_quadrants_function", False):
            ctx.func.has_print |= func.wrapper.has_print

        return node.ptr
