"""AST pass that rewrites tensor subscripts inside ``@qd.kernel`` / ``@qd.func``.

This is the Phase 3+ sugar layer: lets users write logical subscripts on a
:class:`~quadrants.lang._tensor._TensorBase` parameter — *or any tensor
field nested inside a dataclass parameter* (Phase 5) — directly inside a
kernel body, e.g.

    @qd.kernel
    def k(t: qd.Tensor, n0: qd.i32, n1: qd.i32):
        for i, j in qd.ndrange(n0, n1):
            t[i, j] = qd.f32(i + j)            # was: t.underlying[j, i] for layout=(1, 0)

    @dataclasses.dataclass
    class Solver:
        state: qd.Tensor
        aux: qd.Tensor

    @qd.kernel
    def step(s: Solver, n0: qd.i32, n1: qd.i32):
        for i, j in qd.ndrange(n0, n1):
            s.state[i, j] = ...    # was: s.state.underlying[<perm of state's layout>]
            s.aux[i, j] += ...     # each tensor field uses its own layout

without having to spell out ``...underlying[...]`` and manually permute
indices. The permutation is resolved at trace time from the bound
``Tensor.layout`` (which is part of the fastcache key, so each layout
gets its own specialised compilation).

The pass runs *before* the dataclass flattening pass
(``unpack_ast_struct_expressions``); subscripts get rewritten from
``<path>[i, j]`` to ``<path>.underlying[j, i]``, and the flattening pass
then turns ``<path>.underlying`` into the standard flat name
``__qd_<flat path>__qd_underlying``.

Covers:
- read:           ``x = t[i, j]``
- write:          ``t[i, j] = x``
- augmented:      ``t[i, j] += x``  (and friends ``-= *= /= //= %= **= &= |= ^=``)
- 1D / 2D / 3D / arbitrary rank
- both backends (NdarrayTensor + FieldTensor) — backend-agnostic; only
  ``layout`` and the ``underlying`` attribute name are used.
- single tensor params *and* tensor fields nested inside a dataclass
  parameter, including arbitrarily deep nesting (``s.sub.state[i, j]``).

Subscripts whose value is not an attribute path rooted at a registered
tensor (e.g. ``f(t)[i, j]``, plain ``ndarray[i, j]``) are left untouched.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from typing import Any

from quadrants.lang import _tensor
from quadrants.lang._dataclass_util import create_flat_name

# A tensor "path" is a tuple of attribute names from the kernel parameter
# down to the tensor field — e.g. ``('t',)`` for a top-level param,
# ``('s', 'state')`` for a nested field.
TensorPath = tuple[str, ...]


def _is_tensor_param_annotation(annotation: Any) -> bool:
    """True if ``annotation`` is a concrete :class:`_TensorBase` subclass."""
    return (
        isinstance(annotation, type)
        and issubclass(annotation, _tensor._TensorBase)
        and annotation is not _tensor._TensorBase
    )


def _walk_dataclass_for_tensor_paths(prefix: TensorPath, dataclass_type: type) -> list[tuple[TensorPath, str]]:
    """Recursively enumerate tensor fields inside a dataclass.

    Returns a list of ``(path, layout_lookup_kind)`` pairs. Currently the
    second element is unused and reserved for future per-backend customisation.
    Nested dataclasses are walked depth-first.
    """
    out: list[tuple[TensorPath, str]] = []
    for field in dataclasses.fields(dataclass_type):
        sub_path = prefix + (field.name,)
        if _is_tensor_param_annotation(field.type):
            out.append((sub_path, "tensor"))
        elif dataclasses.is_dataclass(field.type):
            out.extend(_walk_dataclass_for_tensor_paths(sub_path, field.type))
    return out


def _resolve_tensor_value(root: Any, path_after_root: TensorPath) -> Any | None:
    """Walk attribute chain from ``root`` to a (possibly nested) tensor value."""
    val: Any = root
    for attr in path_after_root:
        val = getattr(val, attr, None)
        if val is None:
            return None
    return val


def extract_tensor_params_from_kernel_args(fn, py_args: tuple[Any, ...]) -> dict[TensorPath, tuple[int, ...]]:
    """Return ``{path: layout_tuple}`` for every tensor reachable from a
    kernel parameter — either directly typed as a tensor or nested as a
    field inside a dataclass parameter.

    Used on the kernel-call path, where ``py_args`` still contains the
    bound parameter values (Tensor / dataclass instances) positionally
    aligned with the function signature.
    """
    layouts: dict[TensorPath, tuple[int, ...]] = {}
    parameters = inspect.signature(fn).parameters
    for i, (name, parameter) in enumerate(parameters.items()):
        if i >= len(py_args):
            continue
        bound = py_args[i]

        anno = parameter.annotation
        if _is_tensor_param_annotation(anno):
            if not isinstance(bound, _tensor._TensorBase):
                continue
            layouts[(name,)] = tuple(bound.layout)
            continue

        if dataclasses.is_dataclass(anno):
            for sub_path, _ in _walk_dataclass_for_tensor_paths((name,), anno):
                value = _resolve_tensor_value(bound, sub_path[1:])
                if not isinstance(value, _tensor._TensorBase):
                    continue
                layouts[sub_path] = tuple(value.layout)
    return layouts


def extract_tensor_params_from_expanded_args(
    fn, py_args: tuple[Any, ...], arg_metas_expanded
) -> dict[TensorPath, tuple[int, ...]]:
    """Return ``{path: layout_tuple}`` for every tensor (top-level or nested)
    of a ``qd.func`` whose call has been expanded by the kernel-side
    ``CallTransformer``.

    The expansion replaces a ``Tensor`` instance with two flat fields named
    ``__qd_{path}__qd_underlying`` and ``__qd_{path}__qd_layout`` (where
    ``{path}`` is the ``__qd_``-joined dotted path); the layout slot's
    runtime value is the actual layout tuple. We enumerate candidate paths
    from the function signature (dataclass-aware), then look the layout up
    by name in ``arg_metas_expanded``.
    """
    by_name = {meta.name: i for i, meta in enumerate(arg_metas_expanded)}
    layouts: dict[TensorPath, tuple[int, ...]] = {}
    for path in _candidate_tensor_paths(fn):
        layout_arg_name = _flat_name_for_path(path + ("layout",))
        idx = by_name.get(layout_arg_name)
        if idx is None or idx >= len(py_args):
            continue
        value = py_args[idx]
        if not isinstance(value, tuple):
            continue
        layouts[path] = value
    return layouts


def _candidate_tensor_paths(fn) -> list[TensorPath]:
    """Static enumeration of every tensor path reachable through ``fn``'s
    parameters (top-level or nested in a dataclass field)."""
    paths: list[TensorPath] = []
    parameters = inspect.signature(fn).parameters
    for name, parameter in parameters.items():
        anno = parameter.annotation
        if _is_tensor_param_annotation(anno):
            paths.append((name,))
        elif dataclasses.is_dataclass(anno):
            paths.extend(p for p, _ in _walk_dataclass_for_tensor_paths((name,), anno))
    return paths


def _flat_name_for_path(path: TensorPath) -> str:
    """Fold ``('s', 'state', 'underlying')`` into ``'__qd_s__qd_state__qd_underlying'``.

    Mirrors what :func:`unpack_ast_struct_expressions` does at runtime so
    that name lookups against the dataclass-flatten pass output line up.
    """
    if not path:
        raise ValueError("empty tensor path")
    name = path[0]
    for attr in path[1:]:
        name = create_flat_name(name, attr)
    return name


# ---------------------------------------------------------------------------
# AST helpers.
# ---------------------------------------------------------------------------


def _attr_chain_path(node: ast.AST) -> TensorPath | None:
    """If ``node`` is a (possibly-nested) attribute chain rooted at a
    ``Name``, return the dotted path as a tuple. Otherwise ``None``.

    e.g. ``s.state.underlying`` → ``('s', 'state', 'underlying')``.
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    parts.reverse()
    return tuple(parts)


def _path_to_attr_chain(path: TensorPath, ctx: ast.expr_context | None = None) -> ast.expr:
    """Build an AST for ``path[0].path[1]....path[-1]``."""
    if ctx is None:
        ctx = ast.Load()
    if len(path) == 1:
        return ast.Name(id=path[0], ctx=ctx)
    inner: ast.expr = ast.Name(id=path[0], ctx=ast.Load())
    for attr in path[1:-1]:
        inner = ast.Attribute(value=inner, attr=attr, ctx=ast.Load())
    return ast.Attribute(value=inner, attr=path[-1], ctx=ctx)


def _permute_slice(slice_node: ast.AST, layout: tuple[int, ...]) -> ast.AST:
    """Reorder the elements of a Subscript ``slice`` according to ``layout``.

    Convention (matches Python-scope :class:`_TensorBase`): for each logical
    axis ``k``, the index value lands at physical position ``layout[k]``::

        physical[layout[k]] = slice[k]

    For 1D layouts ``(0,)`` this is the identity. Multi-axis subscripts must
    be ``ast.Tuple`` nodes (Python parses ``t[i, j]`` as
    ``Subscript(slice=Tuple([i, j]))``); other slice shapes pass through
    unchanged so we don't accidentally rewrite something exotic.
    """
    if len(layout) == 1:
        return slice_node

    if not isinstance(slice_node, ast.Tuple):
        return slice_node

    elts = slice_node.elts
    if len(elts) != len(layout):
        return slice_node

    permuted: list[ast.AST | None] = [None] * len(layout)
    for k, p in enumerate(layout):
        permuted[p] = elts[k]
    new_tuple = ast.Tuple(elts=permuted, ctx=slice_node.ctx)  # type: ignore[arg-type]
    return ast.copy_location(new_tuple, slice_node)


class _TensorSubscriptTransformer(ast.NodeTransformer):
    """Rewrites ``<path>[idx]`` -> ``<path>.underlying[permuted_idx]`` for
    every attribute path registered in ``tensor_layouts``.

    The transformer rewrites only the ``Subscript`` *value* and *slice*
    nodes; ``ctx`` (Load / Store / Del) is preserved, so reads, plain
    assignments, and augmented assignments all flow through unchanged.
    """

    def __init__(self, tensor_layouts: dict[TensorPath, tuple[int, ...]]) -> None:
        self.tensor_layouts = tensor_layouts

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Recurse into children first so nested subscripts (e.g.
        # ``t[other[i], j]``) get rewritten before we touch the outer one.
        self.generic_visit(node)

        path = _attr_chain_path(node.value)
        if path is None:
            return node
        layout = self.tensor_layouts.get(path)
        if layout is None:
            return node

        permuted_slice = _permute_slice(node.slice, layout)
        new_value = _path_to_attr_chain(path + ("underlying",))
        new_value = ast.copy_location(new_value, node.value)
        new_node = ast.Subscript(
            value=new_value,
            slice=permuted_slice,
            ctx=node.ctx,
        )
        return ast.copy_location(new_node, node)


def _make_keepalive_layout_stmt(path: TensorPath, lineno: int, col_offset: int) -> ast.stmt:
    """Synthetic ``__qd_keepalive_<flat>_layout = <path>.layout`` statement.

    Inserted at the top of a kernel/func body for each tensor path so that
    pruning sees the layout slot as used. The rewrite consumes the layout
    *value* at trace time (folded into the index permutation), but leaves
    no literal reference to ``<path>.layout`` in the rewritten body.
    Without this keepalive, pruning's pass-0 record for a ``qd.func`` would
    mark ``__qd_<flat>__qd_layout`` as unused, and pass-1 would re-trace
    with the slot stripped — at which point this transformer can no longer
    learn the layout to permute the subscripts.

    Assigning to a discarded local avoids the question of whether a bare
    expression statement is legal in Quadrants scope.
    """
    flat = "_".join(path)
    target = ast.Name(id=f"__qd_keepalive_{flat}_layout", ctx=ast.Store())
    value = _path_to_attr_chain(path + ("layout",))
    stmt = ast.Assign(targets=[target], value=value)

    def _stamp(n: ast.AST) -> None:
        n.lineno = lineno
        n.col_offset = col_offset
        n.end_lineno = lineno
        n.end_col_offset = col_offset
        for child in ast.iter_child_nodes(n):
            _stamp(child)

    _stamp(stmt)
    return stmt


def unpack_ast_tensor_subscripts(tree: ast.Module, tensor_layouts: dict[TensorPath, tuple[int, ...]]) -> ast.Module:
    """Apply :class:`_TensorSubscriptTransformer` to the tree, in place.

    Returns the (possibly mutated) tree with line numbers fixed up. If
    ``tensor_layouts`` is empty this is a near-no-op and skips the walk.

    Also injects a tiny "keep-layout-alive" statement at the top of the
    target function body for each tensor path (see
    :func:`_make_keepalive_layout_stmt`).
    """
    if not tensor_layouts:
        return tree
    transformer = _TensorSubscriptTransformer(tensor_layouts=tensor_layouts)
    new_tree = transformer.visit(tree)

    if isinstance(new_tree, ast.Module) and new_tree.body and isinstance(new_tree.body[0], ast.FunctionDef):
        func_def = new_tree.body[0]
        first_stmt = func_def.body[0] if func_def.body else func_def
        keepalives = [
            _make_keepalive_layout_stmt(path, first_stmt.lineno, first_stmt.col_offset) for path in tensor_layouts
        ]
        func_def.body = keepalives + func_def.body

    ast.fix_missing_locations(new_tree)
    return new_tree
