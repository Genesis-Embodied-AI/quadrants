from ast import Name, Starred, expr, keyword
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from ._exceptions import raise_exception
from ._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from .exception import QuadrantsSyntaxError
from .func import Func

if TYPE_CHECKING:
    import ast

    from .ast.ast_transformer_utils import ASTTransformerFuncContext


# (caller_func_id, callee_func_id, call node lineno, call node col_offset). We key call edges by the
# call site's source position rather than by object identity because the AST is re-parsed on every
# pass, so the ``ast.Call`` node is a different object in the discovery and enforcing passes; the
# source position is stable across passes and unique within a caller.
CallSiteKey = tuple[int, int, int, int]


class CallEdge:
    """
    One ``@qd.func`` / ``@qd.kernel`` -> ``@qd.func`` call site, recorded during the discovery pass.

    ``pairs`` holds every ``(caller_arg_flat_name, callee_param_flat_name)`` correspondence, for both
    positional args and kwargs; it drives the used-set fixpoint (a caller needs every argument it
    forwards into a callee parameter the callee needs). ``positional_map`` holds only the positional
    ``__qd_`` Name args and drives ``filter_call_args``. It maps each caller arg flat name to the list of
    callee param flat names it feeds, in call-site (left-to-right) order - a list rather than a single
    value because the same flat name can be forwarded into several slots of one call (e.g.
    ``inner(md, md)``); a plain name->name map would let a later slot overwrite an earlier one and prune
    an argument the callee still needs. We key by name (not by slot index) so the mapping survives the
    positional shift that happens when an upstream caller prunes fields out of a forwarded dataclass,
    which shortens this call's expanded ``node_args`` between the discovery and enforcing passes.

    Edges are keyed per call site (source position), not per callee, so two call sites that forward the
    same flat name into different callee slots (swapped-slot forwarding) stay independent.
    """

    __slots__ = ("caller_func_id", "callee_func_id", "pairs", "positional_map")

    def __init__(self, caller_func_id: int, callee_func_id: int) -> None:
        self.caller_func_id = caller_func_id
        self.callee_func_id = callee_func_id
        self.pairs: list[tuple[str, str]] = []
        self.positional_map: dict[str, list[str]] = {}


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

    Propagation of used parameters from callee to caller is done as a fixpoint over the recorded call
    edges (``propagate_fixpoint``), run once after the discovery pass. This is required because a
    callee's used-set (shared across template instantiations via its func id) keeps growing as later
    call sites and instantiations are discovered - e.g. a field read only inside a
    ``qd.static(True)`` branch is marked used only when that instantiation is walked. A single forward
    copy at call-record time would miss such a field for any caller walked earlier, so the enforcing
    pass would prune a parameter the callee still needs. Iterating to a fixpoint makes the result
    independent of discovery order.
    """

    KERNEL_FUNC_ID = 0

    def __init__(self, kernel_used_parameters: set[str] | None) -> None:
        self.enforcing: bool = False
        self.used_vars_by_func_id: dict[int, set[str]] = defaultdict(set)
        if kernel_used_parameters is not None:
            self.used_vars_by_func_id[Pruning.KERNEL_FUNC_ID].update(kernel_used_parameters)
        # One entry per call site (source position), recorded during discovery. Consumed by
        # propagate_fixpoint() (used-set propagation) and filter_call_args() (per-call-site arg pruning).
        self.edges_by_call_site: dict[CallSiteKey, CallEdge] = {}

    def mark_used(self, func_id: int, parameter_flat_name: str) -> None:
        assert not self.enforcing
        self.used_vars_by_func_id[func_id].add(parameter_flat_name)

    def enforce(self) -> None:
        self.enforcing = True

    def is_used(self, func_id: int, var_flat_name: str) -> bool:
        return var_flat_name in self.used_vars_by_func_id[func_id]

    @staticmethod
    def _call_site_key(caller_func_id: int, callee_func_id: int, node: "ast.Call") -> CallSiteKey:
        return (caller_func_id, callee_func_id, node.lineno, node.col_offset)

    def record_after_call(
        self,
        ctx: "ASTTransformerFuncContext",
        func: "QuadrantsCallable",
        node: "ast.Call",
        node_args: list[expr],
        node_keywords: list[keyword],
    ) -> None:
        """
        called from build_Call, after making the call, in the discovery pass (pass 0)

        Records the call-graph edge for this call site (handles both args and kwargs). Used-set
        propagation is deferred to ``propagate_fixpoint``; nothing here mutates the used-sets.
        """
        if type(func) not in {QuadrantsCallable, BoundQuadrantsCallable}:
            return

        caller_func_id = ctx.func.func_id
        callee_func_id = func.wrapper.func_id  # type: ignore
        # node.args ordering will match that of the called function's arg_metas_expanded, because of
        # the way calling with sequential args works. We read the callee's declared (flat) parameter
        # name from its metas - we can't tell their name just by looking at our own metas.
        #
        # One issue is when calling data-oriented methods, there will be a `self`, which occupies the
        # first callee meta slot; we skip it with self_offset.
        callee_func: Func = node.func.ptr.wrapper  # type: ignore
        has_self = type(func) is BoundQuadrantsCallable
        self_offset = 1 if has_self else 0

        edge = CallEdge(caller_func_id, callee_func_id)
        for arg_id, arg in enumerate(node_args):
            if type(arg) in {Name}:
                caller_arg_name = arg.id  # type: ignore
                callee_param_name = callee_func.arg_metas_expanded[arg_id + self_offset].name  # type: ignore
                edge.pairs.append((caller_arg_name, callee_param_name))
                if caller_arg_name.startswith("__qd_"):
                    edge.positional_map.setdefault(caller_arg_name, []).append(callee_param_name)
        # For keywords we don't need the callee metas (whose ordering need not match ours): the
        # callee's parameter name is available directly from our own keyword node.
        for kwarg in node_keywords:
            if type(kwarg.value) in {Name}:
                caller_arg_name = kwarg.value.id  # type: ignore
                callee_param_name = kwarg.arg
                edge.pairs.append((caller_arg_name, callee_param_name))  # type: ignore

        self.edges_by_call_site[self._call_site_key(caller_func_id, callee_func_id, node)] = edge

    def propagate_fixpoint(self) -> None:
        """
        Propagate used-sets from callees up to callers along the recorded call edges, until they stop
        growing. Run once after the discovery pass, before the enforcing pass.

        A caller needs every argument it forwards into a callee parameter that the callee needs. Used-
        sets only grow and parameters are finite, so this terminates. See the class docstring for why a
        fixpoint (rather than a single forward copy at record time) is required.
        """
        assert not self.enforcing
        edges_by_callee: dict[int, list[CallEdge]] = defaultdict(list)
        for edge in self.edges_by_call_site.values():
            edges_by_callee[edge.callee_func_id].append(edge)

        worklist: deque[int] = deque(self.used_vars_by_func_id.keys())
        queued: set[int] = set(worklist)
        while worklist:
            callee_func_id = worklist.popleft()
            queued.discard(callee_func_id)
            callee_used = self.used_vars_by_func_id[callee_func_id]
            for edge in edges_by_callee.get(callee_func_id, ()):
                caller_used = self.used_vars_by_func_id[edge.caller_func_id]
                grew = False
                for caller_arg, callee_param in edge.pairs:
                    if callee_param in callee_used and caller_arg not in caller_used:
                        caller_used.add(caller_arg)
                        grew = True
                if grew and edge.caller_func_id not in queued:
                    worklist.append(edge.caller_func_id)
                    queued.add(edge.caller_func_id)

    def filter_call_args(
        self,
        caller_func_id: int,
        quadrants_callable: "QuadrantsCallable",
        node: "ast.Call",
        node_args: list[expr],
        node_keywords: list[keyword],
        py_args: list[Any],
    ) -> list[Any]:
        """
        used in build_Call, before making the call, in the enforcing pass (pass 1)

        Prunes positional args the callee does not need. Keyed per call site (via caller_func_id +
        the call node position) so swapped-slot forwarding stays independent. When the same flat name is
        forwarded into several slots (e.g. ``inner(md, md)``), the recorded callee params are consumed in
        left-to-right order (one per occurrence) so each slot is decided against its own callee param.
        Note that this ONLY handles args, not kwargs (kwargs are pruned in _expand_Call_dataclass_kwargs).
        """
        # We can be called with callables other than qd.func, so filter those out:
        if (
            type(quadrants_callable) not in {QuadrantsCallable, BoundQuadrantsCallable}
            or type(quadrants_callable.wrapper) != Func
        ):
            return py_args
        func: Func = quadrants_callable.wrapper  # type: ignore
        callee_func_id = func.func_id
        callee_used_args = self.used_vars_by_func_id[callee_func_id]
        edge = self.edges_by_call_site.get(self._call_site_key(caller_func_id, callee_func_id, node))
        positional_map = edge.positional_map if edge is not None else {}
        # Per-name cursor: the k-th occurrence of a caller arg name consumes the k-th recorded callee
        # param for that name. node_args is walked in the same left-to-right order as when the edge was
        # recorded, so occurrences line up even if an upstream prune dropped some fields in between.
        occurrence_by_name: dict[str, int] = {}

        new_args = []
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
                break
            if type(arg) in {Name}:
                caller_arg_name = arg.id  # type: ignore
                if caller_arg_name.startswith("__qd_"):
                    mapped = positional_map.get(caller_arg_name)
                    occurrence = occurrence_by_name.get(caller_arg_name, 0)
                    occurrence_by_name[caller_arg_name] = occurrence + 1
                    callee_param_name = mapped[occurrence] if mapped is not None and occurrence < len(mapped) else None
                    if callee_param_name is None or (
                        callee_param_name not in callee_used_args and callee_param_name.startswith("__qd_")
                    ):
                        continue
            new_args.append(py_args[i])
        return new_args
