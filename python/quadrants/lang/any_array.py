# type: ignore

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.expr import Expr, make_expr_group
from quadrants.lang.util import quadrants_scope
from quadrants.types.enums import Layout
from quadrants.types.ndarray_type import NdarrayTypeMetadata


class AnyArray:
    """Class for arbitrary arrays in Python AST.

    Args:
        ptr (quadrants_python.Expr): A quadrants_python.Expr wrapping a quadrants_python.ExternalTensorExpression.
        element_shape (Tuple[Int]): () if scalar elements (default), (n) if vector elements, and (n, m) if matrix elements.
        layout (Layout): Memory layout.

    Attributes:
        _qd_layout (tuple of int | None): Optional canonical-axis permutation.
            When set, ``build_Subscript`` permutes user-supplied canonical indices into physical storage order before
            forwarding to the underlying expression. ``None`` (the default) means identity — no rewrite, behaviour
            identical to legacy AnyArray.
    """

    def __init__(self, ptr, _qd_layout=None):
        assert ptr.is_external_tensor_expr()
        self.ptr = ptr
        self.ptr.type_check(impl.get_runtime().prog.config())
        self._qd_layout = _qd_layout

    def element_shape(self):
        return _qd_core.get_external_tensor_element_shape(self.ptr)

    def layout(self):
        # 0: scalar; 1: vector (SOA); 2: matrix (SOA); -1: vector (AOS); -2: matrix (AOS)
        element_dim = _qd_core.get_external_tensor_element_dim(self.ptr)
        if element_dim == 1 or element_dim == 2:
            return Layout.SOA
        return Layout.AOS

    def get_type(self):
        return NdarrayTypeMetadata(
            _qd_core.get_external_tensor_element_type(self.ptr), None, _qd_core.get_external_tensor_needs_grad(self.ptr)
        )  # AnyArray can take any shape

    @property
    @quadrants_scope
    def grad(self):
        """Returns the gradient of this array."""
        return AnyArray(
            _qd_core.make_external_tensor_grad_expr(self.ptr),
            _qd_layout=self._qd_layout,
        )

    @property
    @quadrants_scope
    def shape(self):
        """A list containing sizes for each dimension. Note that element shape will be excluded.

        When ``_qd_layout`` is set the underlying buffer is allocated at the *physical* (permuted) shape; this
        property inverts the layout so callers always see the *canonical* shape — matching ``Ndarray.shape`` and
        ``Field.shape``.

        Returns:
            List[Int]: The result list.
        """
        dim = _qd_core.get_external_tensor_dim(self.ptr)
        dbg_info = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
        phys = [Expr(_qd_core.get_external_tensor_shape_along_axis(self.ptr, i, dbg_info)) for i in range(dim)]
        layout = self._qd_layout
        if layout is None or tuple(layout) == tuple(range(dim)):
            return phys
        inv = [0] * dim
        for p, a in enumerate(layout):
            inv[a] = p
        return [phys[inv[a]] for a in range(dim)]

    @quadrants_scope
    def _loop_range(self):
        """Gets the corresponding quadrants_python.Expr to serve as loop range.

        Returns:
            quadrants_python.Expr: See above.
        """
        return self.ptr


class AnyArrayAccess:
    """Class for first-level access to AnyArray with Vector/Matrix elements in Python AST.

    Args:
        arr (AnyArray): See above.
        indices_first (Tuple[Int]): Indices of first-level access.
    """

    def __init__(self, arr, indices_first):
        self.arr = arr
        self.indices_first = indices_first

    @quadrants_scope
    def subscript(self, i, j):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()

        indices_second = (i,) if len(self.arr.element_shape()) == 1 else (i, j)
        if self.arr.layout() == Layout.SOA:
            indices = indices_second + self.indices_first
        else:
            indices = self.indices_first + indices_second
        return Expr(
            ast_builder.expr_subscript(
                self.arr.ptr,
                make_expr_group(*indices),
                _qd_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )
        )


__all__ = []
