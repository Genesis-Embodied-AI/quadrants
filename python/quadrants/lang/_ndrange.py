# type: ignore

import collections.abc
from typing import Iterable

import numpy as np

from quadrants.lang import ops
from quadrants.lang.exception import QuadrantsSyntaxError, QuadrantsTypeError
from quadrants.lang.expr import Expr
from quadrants.lang.matrix import Matrix
from quadrants.lang.util import has_pytorch as _has_pytorch
from quadrants.types.utils import is_integral

if _has_pytorch():
    import torch  # pylint: disable=C0411

    def _coerce_to_int(v):
        """Convert 0-d torch tensors to int for ndrange bounds."""
        if isinstance(v, (int, float, np.integer, Expr)):
            return v
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            return int(v.item())
        return v

else:

    def _coerce_to_int(v):
        """Identity passthrough when torch is not available."""
        return v


class _Ndrange:
    def __init__(self, *args, layout=None):
        args = list(args)
        for i, arg in enumerate(args):
            if not isinstance(arg, collections.abc.Sequence):
                args[i] = (0, _coerce_to_int(arg))
            if len(args[i]) != 2:
                raise QuadrantsSyntaxError(
                    "Every argument of ndrange should be a scalar or a tuple/list like (begin, end)"
                )
            args[i] = (_coerce_to_int(args[i][0]), _coerce_to_int(ops.max(args[i][0], args[i][1])))
        for arg in args:
            for bound in arg:
                if not isinstance(bound, (int, np.integer)) and not (
                    isinstance(bound, Expr) and is_integral(bound.ptr.get_rvalue_type())
                ):
                    raise QuadrantsTypeError(
                        "Every argument of ndrange should be an integer scalar or a tuple/list of (int, int)"
                    )

        n = len(args)

        # Validate and normalize ``layout``. Stored as ``self.layout`` (``None`` for the identity
        # permutation, else the user-supplied tuple) for introspection / tests, and as
        # ``self._physical_to_canonical`` (a Python list of int of length ``n``) for the AST
        # builder to use when remapping per-physical-level decomposed indices to canonical loop
        # targets. The identity case is kept as ``None`` so the AST-builder fast-path matches
        # the pre-layout codegen byte-for-byte.
        if layout is None:
            self.layout = None
            physical_to_canonical = list(range(n))
        else:
            layout_t = tuple(layout)
            if len(layout_t) != n:
                raise QuadrantsSyntaxError(
                    f"qd.ndrange(layout={layout_t!r}) has {len(layout_t)} entries "
                    f"but ndrange was called with {n} dimension argument(s); they must match"
                )
            if sorted(layout_t) != list(range(n)):
                raise QuadrantsSyntaxError(
                    f"qd.ndrange(layout={layout_t!r}) is not a permutation of range({n})"
                )
            if layout_t == tuple(range(n)):
                self.layout = None
                physical_to_canonical = list(range(n))
            else:
                self.layout = layout_t
                physical_to_canonical = list(layout_t)

        self._physical_to_canonical = physical_to_canonical

        canonical_bounds = args
        canonical_dimensions = [bound[1] - bound[0] for bound in canonical_bounds]

        physical_bounds = [canonical_bounds[c] for c in physical_to_canonical]
        physical_dimensions = [canonical_dimensions[c] for c in physical_to_canonical]

        acc_dimensions = physical_dimensions.copy()
        for i in reversed(range(n - 1)):
            acc_dimensions[i] = acc_dimensions[i] * acc_dimensions[i + 1]
        if not acc_dimensions:  # for the empty case, e.g. qd.ndrange()
            acc_dimensions = [1]

        self._canonical_bounds = canonical_bounds
        self._canonical_dimensions = canonical_dimensions
        self.bounds = physical_bounds
        self.dimensions = physical_dimensions
        self.acc_dimensions = acc_dimensions

    def __iter__(self):
        p2c = self._physical_to_canonical
        cbounds = self._canonical_bounds
        n = len(p2c)

        def gen(level, current):
            if level == n:
                yield tuple(current)
                return
            ax = p2c[level]
            b, e = cbounds[ax]
            for t in range(b, e):
                current[ax] = t
                yield from gen(level + 1, current)

        yield from gen(0, [0] * n)

    def grouped(self):
        return GroupedNDRange(self)


def ndrange(*args, layout=None) -> Iterable:
    """Return an immutable iterator object for looping over multi-dimensional indices.

    This returned set of multi-dimensional indices is the direct product (in the set-theory sense)
    of n groups of integers, where n equals the number of arguments in the input list, and looks like

    range(x1, y1) x range(x2, y2) x ... x range(xn, yn)

    The k-th argument corresponds to the k-th `range()` factor in the above product, and each
    argument must be an integer or a pair of two integers. An integer argument n will be interpreted
    as `range(0, n)`, and a pair of two integers (start, end) will be interpreted as `range(start, end)`.

    You can loop over these multi-dimensonal indices in different ways, see the examples below.

    Args:
        entries: (int, tuple): Must be either an integer, or a tuple/list of two integers.
        layout (tuple of int, optional): Permutation of canonical axes describing the iteration
            nesting order, outermost (slowest-varying) first. For an N-argument ndrange, must be
            a permutation of ``range(N)``. ``None`` (default) and the identity permutation are
            equivalent and reproduce the default order in which the **last argument is the
            innermost / fastest-varying axis**. The yielded loop variables stay bound to
            canonical axes 0, 1, ..., N-1 regardless of layout — only the visit order changes.
            ``layout=`` is independent of the loop body; it controls iteration order whether
            the body touches a field, ndarray, tensor, vector/matrix variant, or no tensor at
            all. The motivating use case is aligning iteration with a non-default physical
            memory layout (e.g. ``qd.tensor(..., layout=...)`` or ``qd.field(..., order=...)``):
            using the matching permutation makes adjacent flat threads step through physically
            adjacent memory.

    Returns:
        An immutable iterator object.

    Example::

        You can loop over 1-D integers in range [start, end), as in native Python

            >>> @qd.kernel
            >>> def loop_1d():
            >>>     start = 2
            >>>     end = 5
            >>>     for i in qd.ndrange((start, end)):
            >>>         print(i)  # will print 2 3 4

        Note the braces around `(start, end)` in the above code. If without them,
        the parameter `2` will be interpreted as `range(0, 2)`, `5` will be
        interpreted as `range(0, 5)`, and you will get a set of 2-D indices which
        contains 2x5=10 elements, and need two indices i, j to loop over them:

            >>> @qd.kernel
            >>> def loop_2d():
            >>>     for i, j in qd.ndrange(2, 5):
            >>>         print(i, j)
            0 0
            ...
            0 4
            ...
            1 4

        But you do can use a single index i to loop over these 2-D indices, in this case
        the indices are returned as a 1-D array `(0, 1, ..., 9)`:

            >>> @qd.kernel
            >>> def loop_2d_as_1d():
            >>>     for i in qd.ndrange(2, 5):
            >>>         print(i)
            will print 0 1 2 3 4 5 6 7 8 9

        In general, you can use any `1 <= k <= n` iterators to loop over a set of n-D
        indices. For `k=n` all the indices are n-dimensional, and they are returned in
        lexical order, but for `k<n` iterators the last n-k+1 dimensions will be collapsed into
        a 1-D array of consecutive integers `(0, 1, 2, ...)` whose length equals the
        total number of indices in the last n-k+1 dimensions:

            >>> @qd.kernel
            >>> def loop_3d_as_2d():
            >>>     # use two iterators to loop over a set of 3-D indices
            >>>     # the last two dimensions for 4, 5 will collapse into
            >>>     # the array [0, 1, 2, ..., 19]
            >>>     for i, j in qd.ndrange(3, 4, 5):
            >>>         print(i, j)
            will print 0 0, 0 1, ..., 0 19, ..., 2 19.

        A typical usage of `ndrange` is when you want to loop over a tensor and process
        its entries in parallel. You should avoid writing nested `for` loops here since
        only top level `for` loops are paralleled in quadrants, instead you can use `ndrange`
        to hold all entries in one top level loop:

            >>> @qd.kernel
            >>> def loop_tensor():
            >>>     for row, col, channel in qd.ndrange(image_height, image_width, channels):
            >>>         image[row, col, channel] = ...

        Aligning iteration order with a non-default tensor layout via ``layout=``:

            >>> A = qd.tensor(qd.f32, shape=(M, N), layout=(1, 0))    # axis 1 outer, axis 0 inner
            >>> @qd.kernel
            >>> def fill():
            >>>     # adjacent flat threads now step along axis 0 (the inner physical axis of A),
            >>>     # i.e. touch physically adjacent memory in A
            >>>     for i, j in qd.ndrange(M, N, layout=(1, 0)):
            >>>         A[i, j] = i + j
    """
    return _Ndrange(*args, layout=layout)


class GroupedNDRange:
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        for ind in self.r:
            yield Matrix(list(ind))


__all__ = ["ndrange"]
