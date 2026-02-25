from ast import Name, Starred, expr, keyword
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ._exceptions import raise_exception
from ._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from .exception import QuadrantsSyntaxError
from .func import Func
from .kernel_arguments import ArgMetadata

if TYPE_CHECKING:
    import ast

    from .ast.ast_transformer_utils import ASTTransformerFuncContext


class Pruning:
    """
    We use the func id to uniquely identify each function.

    Thus, each function has a single set of used parameters associated with it, within
    a single call to a single kernel. When the same function is called multiple times
    within the same call, to the same kernel, then the used parameters for that function
    will be the union over the parameters used by each call to that function.

    A function can have different used parameters parameters between kernels, and
    between different calls to the same kernel.

    Note that we unify handling of func and kernel by using func_id KERNEL_FUNC_ID
    to denote the kernel.
    """

    KERNEL_FUNC_ID = 0

    def __init__(self, kernel_used_parameters: set[str] | None) -> None:
        self.enforcing: bool = False
        self.used_vars_by_func_id: dict[int, set[str]] = defaultdict(set)
        if kernel_used_parameters is not None:
            self.used_vars_by_func_id[Pruning.KERNEL_FUNC_ID].update(kernel_used_parameters)
        # only needed for args, not kwargs
        self.callee_param_by_caller_arg_name_by_func_id: dict[int, dict[str, str]] = defaultdict(dict)

    def mark_used(self, func_id: int, parameter_flat_name: str) -> None:
        assert not self.enforcing
        self.used_vars_by_func_id[func_id].add(parameter_flat_name)

    def enforce(self) -> None:
        self.enforcing = True

    def is_used(self, func_id: int, var_flat_name: str) -> bool:
        return var_flat_name in self.used_vars_by_func_id[func_id]

    def record_after_call(
        self,
        ctx: "ASTTransformerFuncContext",
        func: "QuadrantsCallable",
        node: "ast.Call",
        node_args: list[expr],
        node_keywords: list[keyword],
    ) -> None:
        """
        called from build_Call, after making the call, in pass 0

        note that this handles both args and kwargs
        """
        if type(func) not in {QuadrantsCallable, BoundQuadrantsCallable}:
            return

        my_func_id = ctx.func.func_id
        callee_func_id = func.wrapper.func_id  # type: ignore
        # Copy the used parameters from the child function into our own function.
        callee_used_vars = self.used_vars_by_func_id[callee_func_id]
        vars_to_unprune: set[str] = set()
        arg_id = 0
        # node.args ordering will match that of the called function's metas_expanded,
        # because of the way calling with sequential args works.
        # We need to look at the child's declaration - via metas - in order to get the name they use.
        # We can't tell their name just by looking at our own metas.
        #
        # One issue is when calling data-oriented methods, there will be a `self`. We'll detect this
        # by seeing if the childs arg_metas_expanded is exactly 1 longer than len(node.args) + len(node.kwargs)
        callee_func: Func = node.func.ptr.wrapper  # type: ignore
        has_self = type(func) is BoundQuadrantsCallable
        self_offset = 1 if has_self else 0
        for i, arg in enumerate(node_args):
            if type(arg) in {Name}:
                caller_arg_name = arg.id  # type: ignore
                callee_param_name = callee_func.arg_metas_expanded[arg_id + self_offset].name  # type: ignore
                if callee_param_name in callee_used_vars:
                    vars_to_unprune.add(caller_arg_name)
            arg_id += 1
        # Note that our own arg_metas ordering will in general NOT match that of the child's. That's
        # because our ordering is based on the order in which we pass arguments to the function, but the
        # child's ordering is based on the ordering of their declaration; and these orderings might not
        # match.
        # This is not an issue because, for keywords, we don't need to look at the child's metas.
        # We can get the child's name directly from our own keyword node.
        for kwarg in node_keywords:
            if type(kwarg.value) in {Name}:
                caller_arg_name = kwarg.value.id  # type: ignore
                callee_param_name = kwarg.arg
                if callee_param_name in callee_used_vars:
                    vars_to_unprune.add(caller_arg_name)
            arg_id += 1
        self.used_vars_by_func_id[my_func_id].update(vars_to_unprune)

        used_callee_vars = self.used_vars_by_func_id[callee_func_id]
        child_arg_id = 0
        child_metas: list[ArgMetadata] = node.func.ptr.wrapper.arg_metas_expanded  # type: ignore
        callee_param_by_called_arg_name = self.callee_param_by_caller_arg_name_by_func_id[callee_func_id]
        for i, arg in enumerate(node_args):
            if type(arg) in {Name}:
                caller_arg_name = arg.id  # type: ignore
                if caller_arg_name.startswith("__qd_"):
                    callee_param_name = child_metas[child_arg_id + self_offset].name
                    if callee_param_name in used_callee_vars or not callee_param_name.startswith("__qd_"):
                        callee_param_by_called_arg_name[caller_arg_name] = callee_param_name
            child_arg_id += 1
        self.callee_param_by_caller_arg_name_by_func_id[callee_func_id] = callee_param_by_called_arg_name

    def filter_call_args(
        self,
        quadrants_callable: "QuadrantsCallable",
        node: "ast.Call",
        node_args: list[expr],
        node_keywords: list[keyword],
        py_args: list[Any],
    ) -> list[Any]:
        """
        used in build_Call, before making the call, in pass 1

        note that this ONLY handles args, not kwargs
        """
        # We can be called with callables other than qd.func, so filter those out:
        if (
            type(quadrants_callable) not in {QuadrantsCallable, BoundQuadrantsCallable}
            or type(quadrants_callable.wrapper) != Func
        ):
            return py_args
        func: Func = quadrants_callable.wrapper  # type: ignore
        callee_func_id = func.func_id
        caller_used_args = self.used_vars_by_func_id[callee_func_id]
        new_args = []
        callee_param_id = 0
        callee_metas: list[ArgMetadata] = node.func.ptr.wrapper.arg_metas_expanded  # type: ignore
        callee_metas_pruned = []
        for _callee_meta in callee_metas:
            if _callee_meta.name.startswith("__qd_"):
                if _callee_meta.name in caller_used_args:
                    callee_metas_pruned.append(_callee_meta)
            else:
                callee_metas_pruned.append(_callee_meta)
        callee_metas = callee_metas_pruned
        for i, arg in enumerate(node_args):
            is_starred = type(arg) is Starred
            if is_starred:
                if i != len(node_args) - 1 or len(node_keywords) != 0:
                    raise_exception(
                        ExceptionClass=QuadrantsSyntaxError,
                        msg="* args can only be present as the last argument of a function",
                        err_code="STARNOTLAST",
                    )

                # we'll just dump the rest of the py_args in:
                new_args.extend(py_args[i:])
                callee_param_id += len(py_args[i:])
                break
            if type(arg) in {Name}:
                caller_arg_name = arg.id  # type: ignore
                if caller_arg_name.startswith("__qd_"):
                    callee_param_name = self.callee_param_by_caller_arg_name_by_func_id[callee_func_id].get(
                        caller_arg_name
                    )
                    if callee_param_name is None or (
                        callee_param_name not in caller_used_args and callee_param_name.startswith("__qd_")
                    ):
                        continue
            new_args.append(py_args[i])
            callee_param_id += 1
        py_args = new_args
        return py_args
