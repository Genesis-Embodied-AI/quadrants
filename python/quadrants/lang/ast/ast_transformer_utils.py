# type: ignore

import ast
import builtins
import dataclasses
import traceback
from enum import Enum
from textwrap import TextWrapper
from typing import TYPE_CHECKING, Any, List

from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import ASTBuilder
from quadrants.lang import impl
from quadrants.lang._ndrange import ndrange
from quadrants.lang.ast.symbol_resolver import ASTResolver
from quadrants.lang.exception import (
    QuadrantsCompilationError,
    QuadrantsNameError,
    QuadrantsSyntaxError,
    handle_exception_from_cpp,
)

if TYPE_CHECKING:
    from .._func_base import FuncBase
    from .._pruning import Pruning

AutodiffMode = _qd_core.AutodiffMode


class Builder:
    def __call__(self, ctx: "ASTTransformerFuncContext", node: ast.AST):
        method_name = "build_" + node.__class__.__name__
        method = getattr(self, method_name, None)
        try:
            if method is None:
                error_msg = f'Unsupported node "{node.__class__.__name__}"'
                raise QuadrantsSyntaxError(error_msg)
            info = ctx.get_pos_info(node) if isinstance(node, (ast.stmt, ast.expr)) else ""
            with impl.get_runtime().src_info_guard(info):
                res = method(ctx, node)
                if not hasattr(node, "violates_pure"):
                    # assume False until proven otherwise
                    node.violates_pure = False
                    node.violates_pure_reason = None
                return res
        except Exception as e:
            stack_trace = traceback.format_exc()
            if impl.get_runtime().print_full_traceback:
                raise e
            if ctx.raised or not isinstance(node, (ast.stmt, ast.expr)):
                raise e.with_traceback(None)
            ctx.raised = True
            e = handle_exception_from_cpp(e)
            if not isinstance(e, QuadrantsCompilationError):
                msg = ctx.get_pos_info(node) + traceback.format_exc()
                raise QuadrantsCompilationError(msg) from None
            msg = f"""quadrants stack trace:
===
{stack_trace}
===

Your code:
{ctx.get_pos_info(node)}{e}
"""
            raise type(e)(msg) from None


class VariableScopeGuard:
    def __init__(self, scopes: list[dict[str, Any]]):
        self.scopes = scopes

    def __enter__(self):
        self.scopes.append({})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class StaticScopeStatus:
    def __init__(self):
        self.is_in_static_scope = False


class StaticScopeGuard:
    def __init__(self, status: StaticScopeStatus):
        self.status = status

    def __enter__(self):
        self.prev = self.status.is_in_static_scope
        self.status.is_in_static_scope = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.status.is_in_static_scope = self.prev


class NonStaticControlFlowStatus:
    def __init__(self):
        self.is_in_non_static_control_flow = False


class NonStaticControlFlowGuard:
    def __init__(self, status: NonStaticControlFlowStatus):
        self.status = status

    def __enter__(self):
        self.prev = self.status.is_in_non_static_control_flow
        self.status.is_in_non_static_control_flow = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.status.is_in_non_static_control_flow = self.prev


class LoopStatus(Enum):
    Normal = 0
    Break = 1
    Continue = 2


class LoopScopeAttribute:
    def __init__(self, is_static: bool):
        self.is_static = is_static
        self.status: LoopStatus = LoopStatus.Normal
        self.nearest_non_static_if: ast.If | None = None


class LoopScopeGuard:
    def __init__(self, scopes: list[dict[str, Any]], non_static_guard=None):
        self.scopes = scopes
        self.non_static_guard = non_static_guard

    def __enter__(self):
        self.scopes.append(LoopScopeAttribute(self.non_static_guard is None))
        if self.non_static_guard:
            self.non_static_guard.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()
        if self.non_static_guard:
            self.non_static_guard.__exit__(exc_type, exc_val, exc_tb)


class NonStaticIfGuard:
    def __init__(
        self,
        if_node: ast.If,
        loop_attribute: LoopScopeAttribute,
        non_static_status: NonStaticControlFlowStatus,
    ):
        self.loop_attribute = loop_attribute
        self.if_node = if_node
        self.non_static_guard = NonStaticControlFlowGuard(non_static_status)

    def __enter__(self):
        if self.loop_attribute:
            self.old_non_static_if = self.loop_attribute.nearest_non_static_if
            self.loop_attribute.nearest_non_static_if = self.if_node
        self.non_static_guard.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.loop_attribute:
            self.loop_attribute.nearest_non_static_if = self.old_non_static_if
        self.non_static_guard.__exit__(exc_type, exc_val, exc_tb)


class ReturnStatus(Enum):
    NoReturn = 0
    ReturnedVoid = 1
    ReturnedValue = 2


@dataclasses.dataclass(frozen=True)
class PureViolation:
    var_name: str


class ASTTransformerGlobalContext:
    def __init__(
        self, current_kernel: "Kernel", pruning: "Pruning", currently_compiling_materialize_key, pass_idx: int
    ) -> None:
        self.current_kernel: "Kernel" = current_kernel
        self.pruning: "Pruning" = pruning
        self.currently_compiling_materialize_key = currently_compiling_materialize_key
        self.pass_idx: int = pass_idx
        self.ndarray_to_any_array: dict[int, Any] = {}
        self.struct_ndarray_launch_info: list[tuple] = []
        # Lifted-primitive support for ``@qd.data_oriented(template_primitives=False)``. Mirrors the two ndarray
        # structures above, plus a provenance map driving the two-pass pruning.
        # * ``struct_primitive_provenance`` maps ``(id(parent_obj), attr_name)`` to ``(flat_name, arg_idx, attr_chain,
        #   kind)`` for every reachable primitive member of a flagged template arg. Built (cheaply, no kernel-arg
        #   declaration) on every compilation pass. ``build_Attribute`` uses it to ``mark_used`` the accessed
        #   primitives during the non-enforcing discovery pass.
        # * ``struct_primitive_to_expr`` maps the same key to the arg-load ``Expr``. Populated only in the enforcing
        #   pass, and only for primitives actually accessed by the body (pruning), so ``build_Attribute`` can return
        #   the runtime scalar arg instead of baking the value.
        # * ``struct_primitive_launch_info`` records ``(arg_id, template_arg_idx, attr_chain, kind)`` tuples (``kind``
        #   in ``{'f', 'i', 'u'}``) so the launch path can read the live value and bind it.
        self.struct_primitive_provenance: dict[tuple[int, str], tuple] = {}
        self.struct_primitive_to_expr: dict[tuple[int, str], Any] = {}
        self.struct_primitive_launch_info: list[tuple] = []
        # Caller-side `loop_depth` snapshot for the in-flight `@qd.func` invocation. Each func compile creates a fresh
        # `ASTTransformerFuncContext` with `loop_depth = 0`, so without this snapshot a non-static `range(...)` loop
        # inside a func body would not see any outer for-loops in the caller and would skip the backward-mode dynamic-
        # range diagnostic, silently emitting a wrong adjoint. `CallTransformer.build_Call` writes the caller
        # `ctx.loop_depth` here before invoking the func and restores the previous value after, so the new func ctx
        # can seed its `loop_depth` from this field via `_func_base.py`.
        self.caller_loop_depth: int = 0
        # Caller-side "am I (transitively) inside non-static control flow?" snapshot for the in-flight `@qd.func`
        # invocation. Each func compile creates a fresh `ASTTransformerFuncContext` whose own
        # `non_static_control_flow_status` starts False, so a `@qd.func(requires_top_level=True)` reached *through* an
        # unmarked helper called from a runtime for / if / while would not observe the caller's control-flow context
        # and would escape the top-level check. `CallTransformer.build_Call` writes
        # `ctx.is_in_non_static_control_flow() or <inherited>` here before invoking the func (so it accumulates down a
        # multi-level call chain) and restores the previous value after; the new func ctx seeds
        # `inherited_non_static_control_flow` from this field via `_func_base.py`. Kept separate from the live
        # `non_static_control_flow_status` flag on purpose: that flag also gates the "return inside non-static if/for"
        # diagnostic in `build_Return`, and seeding it True would spuriously reject a top-level `return` in a helper.
        self.caller_in_non_static_control_flow: bool = False


class ASTTransformerFuncContext:
    def __init__(
        self,
        global_context: ASTTransformerGlobalContext,
        template_slot_locations,
        end_lineno: int,
        is_kernel: bool,
        func: "FuncBase",
        arg_features: list[tuple[Any, ...]] | None,
        global_vars: dict[str, Any],
        template_vars: dict[str, Any],
        is_pure: bool,
        py_args: tuple[Any, ...],
        file: str,
        src: list[str],
        start_lineno: int,
        ast_builder: ASTBuilder | None,
        is_real_function: bool,
        autodiff_mode: AutodiffMode,
        raise_on_templated_floats: bool,
    ):
        from quadrants import extension  # pylint: disable=import-outside-toplevel

        self.global_context: ASTTransformerGlobalContext = global_context
        self.func: "FuncBase" = func
        self.local_scopes: list[dict[str, Any]] = []
        self.loop_scopes: List[LoopScopeAttribute] = []
        self.template_slot_locations = template_slot_locations
        self.is_kernel: bool = is_kernel
        self.arg_features: list[tuple[Any, ...]] = arg_features
        self.returns = None
        self.global_vars: dict[str, Any] = global_vars
        self.template_vars: dict[str, Any] = template_vars
        self.is_pure: bool = is_pure
        self.py_args: tuple[Any, ...] = py_args
        self.return_data: tuple[Any, ...] | Any | None = None
        self.file: str = file
        self.src: list[str] = src
        self.indent: int = 0
        for c in self.src[0]:
            if c == " ":
                self.indent += 1
            else:
                break
        self.lineno_offset = start_lineno - 1
        self.start_lineno = start_lineno
        self.end_lineno = end_lineno
        self.raised = False
        self.non_static_control_flow_status = NonStaticControlFlowStatus()
        self.static_scope_status = StaticScopeStatus()
        self.returned = ReturnStatus.NoReturn
        self.ast_builder = ast_builder
        self.visited_funcdef = False
        self.is_real_function = is_real_function
        self.kernel_args: list = []
        self.only_parse_function_def: bool = False
        self.autodiff_mode = autodiff_mode
        self.loop_depth: int = 0
        # Whether the (transitive) caller chain that reached this func compile was already inside non-static control
        # flow. Seeded from `global_context.caller_in_non_static_control_flow` in `_func_base.py`; kernels start at the
        # top of the call stack so they always begin False. Consulted (together with the local
        # `non_static_control_flow_status`) by the `requires_top_level` guard in `CallTransformer.build_Call`.
        self.inherited_non_static_control_flow: bool = False
        self.raise_on_templated_floats = raise_on_templated_floats
        self.expanding_dataclass_call_parameters: bool = False

        self.adstack_enabled: bool = (
            _qd_core.is_extension_supported(
                impl.current_cfg().arch,
                extension.adstack,
            )
            and impl.current_cfg().ad_stack_experimental_enabled
        )

    # e.g.: FunctionDef, Module, Global
    def variable_scope_guard(self):
        return VariableScopeGuard(self.local_scopes)

    # e.g.: For, While
    def loop_scope_guard(self, is_static=False):
        if is_static:
            return LoopScopeGuard(self.loop_scopes)
        return LoopScopeGuard(self.loop_scopes, self.non_static_control_flow_guard())

    def non_static_if_guard(self, if_node: ast.If):
        return NonStaticIfGuard(
            if_node,
            self.current_loop_scope() if self.loop_scopes else None,
            self.non_static_control_flow_status,
        )

    def non_static_control_flow_guard(self) -> NonStaticControlFlowGuard:
        return NonStaticControlFlowGuard(self.non_static_control_flow_status)

    def static_scope_guard(self) -> StaticScopeGuard:
        return StaticScopeGuard(self.static_scope_status)

    def current_scope(self) -> dict[str, Any]:
        return self.local_scopes[-1]

    def current_loop_scope(self) -> dict[str, Any]:
        return self.loop_scopes[-1]

    def loop_status(self) -> LoopStatus:
        if self.loop_scopes:
            return self.loop_scopes[-1].status
        return LoopStatus.Normal

    def set_loop_status(self, status: LoopStatus) -> None:
        self.loop_scopes[-1].status = status

    def is_in_static_for(self) -> bool:
        if self.loop_scopes:
            return self.loop_scopes[-1].is_static
        return False

    def is_in_non_static_control_flow(self) -> bool:
        return self.non_static_control_flow_status.is_in_non_static_control_flow

    def is_in_non_static_control_flow_including_caller(self) -> bool:
        # Like `is_in_non_static_control_flow`, but also True when a (transitive) caller reached this func compile from
        # within non-static control flow. Used by the `requires_top_level` guard so the check is not laundered by an
        # intermediate unmarked `@qd.func`.
        return (
            self.non_static_control_flow_status.is_in_non_static_control_flow or self.inherited_non_static_control_flow
        )

    def is_in_static_scope(self) -> bool:
        return self.static_scope_status.is_in_static_scope

    def is_var_declared(self, name: str) -> bool:
        for s in self.local_scopes:
            if name in s:
                return True
        return False

    def create_variable(self, name: str, var: Any) -> None:
        if name in self.current_scope():
            raise QuadrantsSyntaxError("Recreating variables is not allowed")
        self.current_scope()[name] = var

    def check_loop_var(self, loop_var: str) -> None:
        if self.is_var_declared(loop_var):
            raise QuadrantsSyntaxError(
                f"Variable '{loop_var}' is already declared in the outer scope and cannot be used as loop variable"
            )

    def get_var_by_name(self, name: str) -> tuple[bool, Any, str | None]:
        for s in reversed(self.local_scopes):
            if name in s:
                val = s[name]
                return False, val, None

        reason = None
        violates_pure, found_name = False, False
        if name in self.template_vars:
            var = self.template_vars[name]
            if self.raise_on_templated_floats and isinstance(var, float):
                raise ValueError("Not permitted to access floats as templated values")
            found_name = True
        elif name in self.global_vars:
            var = self.global_vars[name]
            if not name.startswith("_qd_"):
                reason = f"{name} is in global vars, therefore violates pure"
                violates_pure = True
            found_name = True
            if self.raise_on_templated_floats and isinstance(var, float):
                raise ValueError("Not permitted to access floats as global values")

        if found_name:
            from quadrants.lang.matrix import (  # pylint: disable-msg=C0415
                Matrix,
                make_matrix,
            )

            if isinstance(var, Matrix):
                return violates_pure, make_matrix(var.to_list()), reason
            return violates_pure, var, reason

        try:
            return False, getattr(builtins, name), None
        except AttributeError:
            raise QuadrantsNameError(f'Name "{name}" is not defined')

    def get_pos_info(self, node: ast.AST) -> str:
        msg = f'File "{self.file}", line {node.lineno + self.lineno_offset}, in {self.func.func.__name__}:\n'
        col_offset = self.indent + node.col_offset
        end_col_offset = self.indent + node.end_col_offset

        wrapper = TextWrapper(width=80)

        def gen_line(code: str, hint: str) -> str:
            hint += " " * (len(code) - len(hint))
            code = wrapper.wrap(code)
            hint = wrapper.wrap(hint)
            if not len(code):
                return "\n\n"
            return "".join([c + "\n" + h + "\n" for c, h in zip(code, hint)])

        if node.lineno == node.end_lineno:
            if node.lineno - 1 < len(self.src):
                hint = " " * col_offset + "^" * (end_col_offset - col_offset)
                msg += gen_line(self.src[node.lineno - 1], hint)
        else:
            node_type = node.__class__.__name__

            if node_type in ["For", "While", "FunctionDef", "If"]:
                end_lineno = max(node.body[0].lineno - 1, node.lineno)
            else:
                end_lineno = node.end_lineno

            for i in range(node.lineno - 1, end_lineno):
                last = len(self.src[i])
                while last > 0 and (self.src[i][last - 1].isspace() or not self.src[i][last - 1].isprintable()):
                    last -= 1
                first = 0
                while first < len(self.src[i]) and (
                    self.src[i][first].isspace() or not self.src[i][first].isprintable()
                ):
                    first += 1
                if i == node.lineno - 1:
                    hint = " " * col_offset + "^" * (last - col_offset)
                elif i == node.end_lineno - 1:
                    hint = " " * first + "^" * (end_col_offset - first)
                elif first < last:
                    hint = " " * first + "^" * (last - first)
                else:
                    hint = ""
                msg += gen_line(self.src[i], hint)
        return msg


def get_decorator(ctx: ASTTransformerFuncContext, node) -> str:
    if not isinstance(node, ast.Call):
        return ""
    for wanted, name in [
        (impl.static, "static"),
        (impl.static_assert, "static_assert"),
        (impl.grouped, "grouped"),
        (ndrange, "ndrange"),
    ]:
        if ASTResolver.resolve_to(node.func, wanted, ctx.global_vars):
            return name
    return ""


def maybe_lifted_primitive(ctx: ASTTransformerFuncContext, parent: Any, attr: str):
    """Handle access to a primitive member that ``@qd.data_oriented(template_primitives=False)`` lifted into a
    runtime scalar kernel arg (see ``_predeclare_struct_primitives``).

    Returns the arg-load ``Expr`` to substitute for the baked value in the enforcing pass; returns ``None`` when
    the attribute is not a lifted primitive, or during the non-enforcing discovery pass (where the access is
    instead recorded via ``mark_used`` and the caller falls back to baking the throw-away discovery-pass IR).

    Raises ``QuadrantsSyntaxError`` if the lifted primitive is used inside ``qd.static(...)``: that context
    requires a compile-time constant, which a runtime-lifted primitive is not. Under this flag there is no
    per-attribute baked escape hatch, so a ``qd.static`` use is unambiguously a mistake and we surface it loudly
    rather than silently baking a value that will not track mutations.
    """
    provenance = ctx.global_context.struct_primitive_provenance
    if not provenance:
        return None
    key = (id(parent), attr)
    entry = provenance.get(key)
    if entry is None:
        return None
    if ctx.is_in_static_scope():
        raise QuadrantsSyntaxError(
            f"'{attr}' is a primitive member of a @qd.data_oriented(template_primitives=False) object, so it is "
            f"a runtime kernel argument and cannot be used inside qd.static(). Either remove qd.static(), or "
            f"decorate the class @qd.data_oriented (the default, template_primitives=True) to bake '{attr}' as a "
            f"compile-time constant."
        )
    pruning = ctx.global_context.pruning
    if not pruning.enforcing:
        # Record under the kernel's func id (see ``_predeclare_struct_primitives``) so the enforcing pass declares
        # this primitive and the existing fastcache serialisation captures it.
        pruning.mark_used(pruning.KERNEL_FUNC_ID, entry[0])
        return None
    return ctx.global_context.struct_primitive_to_expr.get(key)
