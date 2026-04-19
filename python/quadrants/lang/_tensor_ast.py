"""AST pass that rewrites tensor subscripts inside ``@qd.kernel`` / ``@qd.func``.

This is the Phase 3 sugar layer: lets users write logical subscripts on a
:class:`~quadrants.lang._tensor._TensorBase` parameter directly inside a
kernel body, e.g.

    @qd.kernel
    def k(t: qd.Tensor, n0: qd.i32, n1: qd.i32):
        for i, j in qd.ndrange(n0, n1):
            t[i, j] = qd.f32(i + j)            # was: t.underlying[j, i] for layout=(1, 0)

without having to spell out ``t.underlying[...]`` and manually permute
indices. The permutation is resolved at trace time from the bound
``Tensor.layout`` (which is part of the fastcache key, so each layout
gets its own specialised compilation).

The pass runs *before* the dataclass flattening pass
(``unpack_ast_struct_expressions``); subscripts get rewritten from
``t[i, j]`` to ``t.underlying[j, i]``, and the flattening pass then turns
``t.underlying`` into the flattened name ``__qd_t__qd_underlying`` exactly
as it does for any other dataclass field access.

Covers:
- read:           ``x = t[i, j]``
- write:          ``t[i, j] = x``
- augmented:      ``t[i, j] += x``  (and friends ``-= *= /= //= %= **= &= |= ^=``)
- 1D / 2D / 3D / arbitrary rank
- both backends (NdarrayTensor + FieldTensor) — backend-agnostic; only
  ``layout`` and the ``underlying`` attribute name are used.

Subscripts where the value is *not* a Name (e.g. ``f(t)[i, j]``) or where
the Name does not refer to a tensor parameter are left untouched.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

from quadrants.lang import _tensor
from quadrants.lang._dataclass_util import create_flat_name


def _is_tensor_param_annotation(annotation: Any) -> bool:
    """True if ``annotation`` is a concrete :class:`_TensorBase` subclass."""
    return (
        isinstance(annotation, type)
        and issubclass(annotation, _tensor._TensorBase)
        and annotation is not _tensor._TensorBase
    )


def _tensor_param_names(fn) -> list[str]:
    """Names of parameters of ``fn`` whose annotation is a tensor backend variant."""
    return [
        name
        for name, parameter in inspect.signature(fn).parameters.items()
        if _is_tensor_param_annotation(parameter.annotation)
    ]


def extract_tensor_params_from_kernel_args(fn, py_args: tuple[Any, ...]) -> dict[str, tuple[int, ...]]:
    """Return ``{param_name: layout_tuple}`` for every kernel parameter typed
    as a tensor backend variant.

    Used on the kernel-call path, where ``py_args`` still contains the bound
    Tensor instances positionally aligned with the function signature.
    """
    layouts: dict[str, tuple[int, ...]] = {}
    parameters = inspect.signature(fn).parameters
    for i, (name, parameter) in enumerate(parameters.items()):
        if not _is_tensor_param_annotation(parameter.annotation):
            continue
        if i >= len(py_args):
            continue
        bound = py_args[i]
        if not isinstance(bound, _tensor._TensorBase):
            # Annotation says tensor but caller passed something else; let the
            # downstream binder produce the canonical error.
            continue
        layouts[name] = tuple(bound.layout)
    return layouts


def extract_tensor_params_from_expanded_args(
    fn, py_args: tuple[Any, ...], arg_metas_expanded
) -> dict[str, tuple[int, ...]]:
    """Return ``{param_name: layout_tuple}`` for every tensor parameter of a
    ``qd.func`` whose call has been expanded by the kernel-side
    ``CallTransformer``.

    The expansion replaces a ``Tensor`` instance with two flat fields named
    ``__qd_{param}__qd_underlying`` and ``__qd_{param}__qd_layout``; the
    layout slot's runtime value is the actual layout tuple. We match these
    by name in ``arg_metas_expanded`` to locate the layout position.
    """
    by_name = {meta.name: i for i, meta in enumerate(arg_metas_expanded)}
    layouts: dict[str, tuple[int, ...]] = {}
    for name in _tensor_param_names(fn):
        layout_arg_name = create_flat_name(name, "layout")
        idx = by_name.get(layout_arg_name)
        if idx is None or idx >= len(py_args):
            continue
        value = py_args[idx]
        if not isinstance(value, tuple):
            # Pre-trace: layout slot might not yet be a python tuple if pruning
            # changed the layout; skip and let the original error surface.
            continue
        layouts[name] = value
    return layouts


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
        # Identity permutation: nothing to do, regardless of how the user
        # spelled the single index.
        return slice_node

    if not isinstance(slice_node, ast.Tuple):
        # Single non-tuple index on a multi-dim tensor: caller error, but we
        # leave the AST alone and let the downstream subscript path raise
        # with its native error message.
        return slice_node

    elts = slice_node.elts
    if len(elts) != len(layout):
        # Arity mismatch: same reasoning — leave alone.
        return slice_node

    permuted: list[ast.AST | None] = [None] * len(layout)
    for k, p in enumerate(layout):
        permuted[p] = elts[k]
    new_tuple = ast.Tuple(elts=permuted, ctx=slice_node.ctx)  # type: ignore[arg-type]
    return ast.copy_location(new_tuple, slice_node)


class _TensorSubscriptTransformer(ast.NodeTransformer):
    """Rewrites ``t[idx]`` -> ``t.underlying[permuted_idx]`` for tensor
    parameters whose layout is given in ``tensor_layouts``.

    The transformer rewrites only the ``Subscript`` *value* and *slice*
    nodes; ``ctx`` (Load / Store / Del) is preserved, so reads, plain
    assignments, and augmented assignments all flow through unchanged.
    """

    def __init__(self, tensor_layouts: dict[str, tuple[int, ...]]) -> None:
        self.tensor_layouts = tensor_layouts

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Recurse into children first so nested subscripts (e.g.
        # ``t[other[i], j]``) get rewritten before we touch the outer one.
        self.generic_visit(node)

        if not isinstance(node.value, ast.Name):
            return node
        name = node.value.id
        layout = self.tensor_layouts.get(name)
        if layout is None:
            return node

        permuted_slice = _permute_slice(node.slice, layout)
        new_value = ast.Attribute(
            value=ast.Name(id=name, ctx=ast.Load()),
            attr="underlying",
            ctx=ast.Load(),
        )
        new_value = ast.copy_location(new_value, node.value)
        new_node = ast.Subscript(
            value=new_value,
            slice=permuted_slice,
            ctx=node.ctx,
        )
        return ast.copy_location(new_node, node)


def _make_keepalive_layout_stmt(name: str, lineno: int, col_offset: int) -> ast.stmt:
    """Synthetic ``__qd_keepalive_<name>_layout = <name>.layout`` statement.

    Inserted at the top of a kernel/func body for each tensor param so that
    pruning sees the layout slot as used. The rewrite consumes the layout
    *value* at trace time (folded into the index permutation), but leaves
    no literal reference to ``t.layout`` in the rewritten body. Without
    this keepalive, pruning's pass-0 record for a ``qd.func`` would mark
    ``__qd_<t>__qd_layout`` as unused, and pass-1 would re-trace with the
    slot stripped — at which point this transformer can no longer learn
    the layout to permute the subscripts, and ``t[i, j]`` reaches Quadrants
    unrewritten and crashes.

    Assigning to a discarded local avoids the question of whether a bare
    expression statement is legal in Quadrants scope.
    """
    target = ast.Name(id=f"__qd_keepalive_{name}_layout", ctx=ast.Store())
    value = ast.Attribute(
        value=ast.Name(id=name, ctx=ast.Load()),
        attr="layout",
        ctx=ast.Load(),
    )
    stmt = ast.Assign(targets=[target], value=value)
    for node in (target, value, value.value, stmt):
        node.lineno = lineno
        node.col_offset = col_offset
        node.end_lineno = lineno
        node.end_col_offset = col_offset
    return stmt


def unpack_ast_tensor_subscripts(tree: ast.Module, tensor_layouts: dict[str, tuple[int, ...]]) -> ast.Module:
    """Apply :class:`_TensorSubscriptTransformer` to the tree, in place.

    Returns the (possibly mutated) tree with line numbers fixed up. If
    ``tensor_layouts`` is empty this is a near-no-op and skips the walk.

    Also injects a tiny "keep-layout-alive" statement at the top of the
    target function body for each tensor param (see
    :func:`_make_keepalive_layout_stmt`).
    """
    if not tensor_layouts:
        return tree
    transformer = _TensorSubscriptTransformer(tensor_layouts=tensor_layouts)
    new_tree = transformer.visit(tree)

    # Inject keepalive statements at the top of the function body.
    if isinstance(new_tree, ast.Module) and new_tree.body and isinstance(new_tree.body[0], ast.FunctionDef):
        func_def = new_tree.body[0]
        first_stmt = func_def.body[0] if func_def.body else func_def
        keepalives = [
            _make_keepalive_layout_stmt(name, first_stmt.lineno, first_stmt.col_offset) for name in tensor_layouts
        ]
        func_def.body = keepalives + func_def.body

    ast.fix_missing_locations(new_tree)
    return new_tree
