from ast import Attribute, Name, Starred, expr, keyword
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from ._dataclass_util import create_flat_name
from ._exceptions import raise_exception
from ._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from .exception import QuadrantsSyntaxError
from .func import Func


def _flatten_arg_node(node: expr) -> tuple[str, str] | None:
    """Flatten an AST arg node into ``(flat_name, root_name_id)`` (or ``None`` if the node isn't a recognisable
    name/attribute chain rooted at a plain Name).

    Returns both the full flat name (e.g. ``__qd_self__qd_dofs`` for ``self.dofs``) and the root Name's id (``self``).
    Callers use the root id to distinguish kernel-arg-rooted chains (``self.dofs`` -> root ``self``) from already-
    flattened dataclass-arg references (``__qd_self__qd_dofs`` -> root ``__qd_self__qd_dofs``). The flat path alone is
    ambiguous because ``__qd_self__qd_dofs`` could be either an attribute chain *or* a single flattened Name.

    Mirrors ``FlattenAttributeNameTransformer._flatten_attribute_name`` but on the raw call-arg AST.
    Used by ``record_after_call`` to handle ``f(self.dofs)`` etc. - without this the callee's pruning
    info for attribute-chain args is dropped at the call boundary."""
    if isinstance(node, Name):
        return node.id, node.id
    if isinstance(node, Attribute):
        parent = _flatten_arg_node(node.value)
        if parent is None:
            return None
        parent_flat, root_id = parent
        return create_flat_name(parent_flat, node.attr), root_id
    return None


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

    ``chain_pairs`` holds ``(callee_param_flat_name, caller_arg_flat_name)`` for args whose root is a
    bare (non-``__qd_``) kernel arg - i.e. attribute chains like ``f(self.dofs)`` and plain forwards of
    non-flattened args like ``f(self)``. It drives chain-path propagation in the fixpoint, which rewrites
    a callee-rooted chain path (``__qd_s__qd_x``) into a caller-rooted one (``__qd_self__qd_dofs__qd_x``).

    Edges are keyed per call site (source position), not per callee, so two call sites that forward the
    same flat name into different callee slots (swapped-slot forwarding) stay independent.
    """

    __slots__ = ("caller_func_id", "callee_func_id", "pairs", "positional_map", "chain_pairs")

    def __init__(self, caller_func_id: int, callee_func_id: int) -> None:
        self.caller_func_id = caller_func_id
        self.callee_func_id = callee_func_id
        self.pairs: list[tuple[str, str]] = []
        self.positional_map: dict[str, list[str]] = {}
        self.chain_pairs: list[tuple[str, str]] = []


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
        # propagate_fixpoint() (used-set + chain-path propagation) and filter_call_args() (per-call-site arg pruning).
        self.edges_by_call_site: dict[CallSiteKey, CallEdge] = {}
        # id(ndarray) -> seen during the first compile pass via ``_promote_ndarray_if_declared``. Populated by the
        # AST builder when a chain like ``self.x.y`` resolves to an ndarray that was pre-declared by
        # ``_predeclare_struct_ndarrays``. On the second (enforcing) pass, ``_predeclare_struct_ndarrays`` only
        # registers ndarrays whose id is in this set - dropping every reachable-but-unused ndarray from the kernel's
        # parameter list.
        self.used_struct_ndarray_ids: set[int] = set()
        # Whether the non-enforcing first pass actually ran for this kernel materialize. When fastcache hits, we skip
        # pass 0 entirely and ``used_struct_ndarray_ids`` is therefore unreliable - in that case
        # ``_predeclare_struct_ndarrays`` falls back to registering every reachable ndarray (same as historical
        # behavior).
        self.pass_0_ran: bool = False
        # Kernel-arg-rooted attribute chains used by each func, in flat-name form (``__qd_self__qd_dofs__qd_x``).
        # Populated by ``ASTTransformer.build_Attribute`` for non-flattened kernel args (data_oriented /
        # qd.template). Kept *separate* from ``used_vars_by_func_id`` because the latter drives ``struct_locals`` on
        # the enforcing pass (line ~230 of kernel.py), and ``FlattenAttributeNameTransformer`` would rewrite ``s.x``
        # -> ``Name('__qd_s__qd_x')`` if these chain names appeared there - yielding a ``QuadrantsNameError: Name
        # "__qd_s__qd_x" is not defined``. ``record_after_call`` propagates entries from callee to caller (so
        # ``f(self.dofs)`` where ``f`` reads ``s.x`` ends up with ``__qd_self__qd_dofs__qd_x`` in the kernel's set).
        # After both compile passes, ``Pruning.fold_kernel_arg_chain_paths`` merges the kernel's set into
        # ``used_vars_by_func_id[KERNEL_FUNC_ID]`` so fastcache stores them in L1 and the args_hasher narrow walk
        # picks them up.
        self.kernel_arg_chain_paths_by_func_id: dict[int, set[str]] = defaultdict(set)

    def mark_used(self, func_id: int, parameter_flat_name: str) -> None:
        assert not self.enforcing
        self.used_vars_by_func_id[func_id].add(parameter_flat_name)

    def mark_kernel_arg_chain_used(self, func_id: int, chain_flat_name: str) -> None:
        """Record a kernel-arg-rooted attribute chain (e.g. ``__qd_self__qd_dofs__qd_x``).

        Stored separately from ``used_vars_by_func_id`` - see the docstring on ``kernel_arg_chain_paths_by_func_id``
        for why."""
        assert not self.enforcing
        self.kernel_arg_chain_paths_by_func_id[func_id].add(chain_flat_name)

    def fold_struct_nd_paths(
        self, struct_ndarray_launch_info: list[tuple[Any, int, tuple[str, ...]]], arg_metas: list[ArgMetadata]
    ) -> None:
        """Add data_oriented (and dataclass-nested) ndarray attribute chains to the kernel's pruning flat name set so
        ``args_hasher.hash_args`` narrow-walks them correctly.

        Background: ``used_vars_by_func_id[KERNEL_FUNC_ID]`` is populated by AST walking of flat names produced by
        ``FlattenAttributeNameTransformer`` - but that transformer only flattens *dataclass* args.
        ``@qd.data_oriented`` args (template-typed) stay as ``Attribute(value=Name(self), attr=...)`` in the AST and
        don't contribute to ``used_vars_by_func_id``. Their kernel-accessed ndarray paths *are* recorded - in
        ``struct_ndarray_launch_info`` as ``(arg_id_vec[0], arg_idx, attr_chain)`` - but only for ndarray members.

        Convert each ``(arg_idx, attr_chain)`` to a flat name like ``__qd_<arg_name>__qd_<chain[0]>__qd_...`` and union
        all prefixes into the pruning set. After this fold, narrowing in args_hasher matches the same convention used
        for dataclass args.

        Limitation: non-ndarray data_oriented members (primitive ints/floats whose values are baked in at compile,
        opaque Python objects) are *not* tracked anywhere as kernel-accessed. The narrow walk cannot distinguish
        "kernel reads this primitive" from "kernel does not read this primitive". The
        ``args_hasher.stringify_obj_type`` data_oriented branch handles this conservatively by walking *all* attrs of
        a data_oriented container - narrowing only suppresses subtrees explicitly absent from the pruning set. So for
        a data_oriented arg with mostly-ndarray members, the cache key correctly depends on the ndarray paths it
        uses; for one with primitive members whose values matter, those members are still folded into the hash
        (qualname-fallback / value paths).
        """
        if not struct_ndarray_launch_info:
            return
        kernel_used: set[str] = self.used_vars_by_func_id[Pruning.KERNEL_FUNC_ID]
        for _arg_id_cpp, arg_idx, attr_chain in struct_ndarray_launch_info:
            if arg_idx < 0 or arg_idx >= len(arg_metas):
                continue
            arg_name = arg_metas[arg_idx].name
            if not arg_name:
                continue
            flat = arg_name
            for attr in attr_chain:
                flat = create_flat_name(flat, attr)
                kernel_used.add(flat)

    def fold_kernel_arg_chain_paths(self) -> None:
        """Merge the kernel's chain-paths set into ``used_vars_by_func_id[KERNEL_FUNC_ID]`` *after* both compile
        passes have completed.

        Background: ``ASTTransformer.build_Attribute`` records every kernel-arg-rooted attribute chain (e.g.
        ``__qd_self__qd_n``, ``__qd_self__qd_cfg``) into ``kernel_arg_chain_paths_by_func_id`` rather than
        ``used_vars_by_func_id``, because the latter is read on the enforcing pass to build ``struct_locals`` for
        ``FlattenAttributeNameTransformer``. If chain names appeared there, the transformer would rewrite ``self.n``
        into ``Name('__qd_self__qd_n')`` and ``build_Name`` would fail to find such a variable.

        Doing the merge here - after pass 1, just like ``fold_struct_nd_paths`` - avoids that interaction while
        still making the chain paths available to the fastcache args-hash narrow walk. The set on
        ``used_py_dataclass_parameters_by_key_enforcing[key]`` is the *same* object as
        ``used_vars_by_func_id[KERNEL_FUNC_ID]`` (assigned by reference at end of pass 0), so updating one updates
        both.
        """
        kernel_chain_paths = self.kernel_arg_chain_paths_by_func_id.get(Pruning.KERNEL_FUNC_ID)
        if not kernel_chain_paths:
            return
        self.used_vars_by_func_id[Pruning.KERNEL_FUNC_ID].update(kernel_chain_paths)

    @staticmethod
    def _propagate_chain_paths(
        callee_chain_paths: set[str],
        callee_param_name: str,
        caller_flat: str,
        chain_paths_to_propagate: set[str],
    ) -> None:
        """When ``f(self.dofs)`` is called and ``f``'s body reads ``s.x`` (callee param ``s`` bound to caller
        attribute chain ``self.dofs``), the callee's chain-paths set contains ``__qd_s__qd_x`` but the
        caller's chain-paths set must record ``__qd_self__qd_dofs__qd_x``. This helper does that
        prefix substitution. Only chain paths starting with ``__qd_<callee_param>__qd_`` are propagated
        (chains rooted in unrelated callee args don't apply to this caller arg)."""
        prefix = f"__qd_{callee_param_name}__qd_"
        for sub in callee_chain_paths:
            if sub.startswith(prefix):
                rest = sub[len(prefix) :]
                if caller_flat.startswith("__qd_"):
                    new_flat = f"{caller_flat}__qd_{rest}"
                else:
                    new_flat = f"__qd_{caller_flat}__qd_{rest}"
                chain_paths_to_propagate.add(new_flat)

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
            # Record kernel-arg-rooted chain paths for attribute-chain args (``f(self.dofs)``) AND for plain-Name
            # args of non-flattened types (``f(self)``), so ``propagate_fixpoint`` can rewrite the callee's chain
            # paths into caller-rooted ones. Gate on the *root* Name id, not the resulting flat string:
            # ``self.dofs`` flattens to ``__qd_self__qd_dofs`` (which starts with ``__qd_``) but its root is the
            # bare kernel arg ``self`` - we still need to propagate. Already-flattened dataclass refs like
            # ``Name('__qd_self__qd_dofs')`` have a ``__qd_*`` root and are covered by ``edge.pairs`` above.
            flat = _flatten_arg_node(arg)
            if flat is not None:
                caller_flat, root_id = flat
                if not root_id.startswith("__qd_"):
                    callee_param_name = callee_func.arg_metas_expanded[arg_id + self_offset].name  # type: ignore
                    edge.chain_pairs.append((callee_param_name, caller_flat))
        # For keywords we don't need the callee metas (whose ordering need not match ours): the
        # callee's parameter name is available directly from our own keyword node.
        for kwarg in node_keywords:
            if type(kwarg.value) in {Name}:
                caller_arg_name = kwarg.value.id  # type: ignore
                callee_param_name = kwarg.arg
                edge.pairs.append((caller_arg_name, callee_param_name))  # type: ignore
            flat = _flatten_arg_node(kwarg.value)
            if flat is not None:
                caller_flat, root_id = flat
                # ``kwarg.arg`` is ``None`` for double-star unpacking (``**kwargs``); chain propagation requires a
                # concrete parameter name so just skip.
                if not root_id.startswith("__qd_") and kwarg.arg is not None:
                    edge.chain_pairs.append((kwarg.arg, caller_flat))

        self.edges_by_call_site[self._call_site_key(caller_func_id, callee_func_id, node)] = edge

    def propagate_fixpoint(self) -> None:
        """
        Propagate used-sets and kernel-arg chain paths from callees up to callers along the recorded call
        edges, until they stop growing. Run once after the discovery pass, before the enforcing pass.

        A caller needs every argument it forwards into a callee parameter that the callee needs. Used-
        sets only grow and parameters are finite, so this terminates. See the class docstring for why a
        fixpoint (rather than a single forward copy at record time) is required.

        Chain paths are propagated here for the same reason, and against the same edges: a callee's chain
        paths keep growing as later call sites and template instantiations are discovered, so rewriting
        them into caller-rooted form at record time would miss any caller walked earlier. Chain-path sets
        only grow and the rewrite adds no new suffixes, so this terminates too.
        """
        assert not self.enforcing
        edges_by_callee: dict[int, list[CallEdge]] = defaultdict(list)
        for edge in self.edges_by_call_site.values():
            edges_by_callee[edge.callee_func_id].append(edge)

        worklist: deque[int] = deque({*self.used_vars_by_func_id, *self.kernel_arg_chain_paths_by_func_id})
        queued: set[int] = set(worklist)
        while worklist:
            callee_func_id = worklist.popleft()
            queued.discard(callee_func_id)
            callee_used = self.used_vars_by_func_id[callee_func_id]
            callee_chain_paths = self.kernel_arg_chain_paths_by_func_id.get(callee_func_id)
            for edge in edges_by_callee.get(callee_func_id, ()):
                caller_used = self.used_vars_by_func_id[edge.caller_func_id]
                grew = False
                for caller_arg, callee_param in edge.pairs:
                    if callee_param in callee_used and caller_arg not in caller_used:
                        caller_used.add(caller_arg)
                        grew = True
                if callee_chain_paths and edge.chain_pairs:
                    propagated: set[str] = set()
                    for callee_param, caller_flat in edge.chain_pairs:
                        self._propagate_chain_paths(callee_chain_paths, callee_param, caller_flat, propagated)
                    if propagated:
                        caller_chain_paths = self.kernel_arg_chain_paths_by_func_id[edge.caller_func_id]
                        if not propagated <= caller_chain_paths:
                            caller_chain_paths |= propagated
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
