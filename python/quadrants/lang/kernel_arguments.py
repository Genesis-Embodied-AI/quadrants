# type: ignore

import inspect
from typing import Any, Literal

import quadrants.lang
from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import (
    BoundaryMode,
    DataTypeCxx,
)
from quadrants.lang import impl, ops
from quadrants.lang.any_array import AnyArray
from quadrants.lang.expr import Expr
from quadrants.lang.matrix import MatrixType
from quadrants.lang.struct import StructType
from quadrants.lang.util import cook_dtype
from quadrants.types.compound_types import CompoundType
from quadrants.types.primitive_types import RefType, u64


class ArgMetadata:
    """
    Metadata about an argument to a function
    """

    def __init__(self, annotation, name, default=inspect.Parameter.empty, optional=False):
        self.annotation = annotation
        self.name = name
        self.default = default
        # ``None`` is an accepted value for this slot (spelled ``T | None``); ``annotation`` is the unwrapped ``T``.
        self.optional = optional

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(annotation={self.annotation}, name={self.name}, default={self.default}, "
            f"optional={self.optional})"
        )


class SparseMatrixEntry:
    def __init__(self, ptr, i, j, dtype):
        self.ptr = ptr
        self.i = i
        self.j = j
        self.dtype = dtype

    def _augassign(self, value, op):
        call_func = f"insert_triplet_{self.dtype}"
        if op == "Add":
            quadrants.lang.impl.call_internal(call_func, self.ptr, self.i, self.j, ops.cast(value, self.dtype))
        elif op == "Sub":
            quadrants.lang.impl.call_internal(call_func, self.ptr, self.i, self.j, -ops.cast(value, self.dtype))
        else:
            assert False, "Only operations '+=' and '-=' are supported on sparse matrices."


class SparseMatrixProxy:
    def __init__(self, ptr, dtype):
        self.ptr = ptr
        self.dtype = dtype

    def subscript(self, i, j):
        return SparseMatrixEntry(self.ptr, i, j, self.dtype)


def decl_scalar_arg(dtype, name):
    return create_kernel_parameter("scalar", dtype, name)[1]


def get_type_for_kernel_args(dtype, name):
    if isinstance(dtype, MatrixType):
        # Compiling the matrix type to a struct type because the support for the matrix type is not ready yet on SPIR-V based backends.
        if dtype.ndim == 1:
            elements = [(dtype.dtype, f"{name}_{i}") for i in range(dtype.n)]
        else:
            elements = [(dtype.dtype, f"{name}_{i}_{j}") for i in range(dtype.n) for j in range(dtype.m)]
        return _qd_core.get_type_factory_instance().get_struct_type(elements)
    if isinstance(dtype, StructType):
        elements = []
        for k, element_type in dtype.members.items():
            if isinstance(element_type, CompoundType):
                new_dtype = get_type_for_kernel_args(element_type, k)
                elements.append([new_dtype, k])
            else:
                elements.append([element_type, k])
        return _qd_core.get_type_factory_instance().get_struct_type(elements)
    # Assuming dtype is a primitive type
    return dtype


def create_kernel_parameter(
    kind: Literal["scalar", "ndarray", "matrix", "struct", "sparse"],
    dtype: Any,
    name: str,
    *,
    ndim: int = 0,
    needs_grad: bool = False,
    boundary: BoundaryMode | None = None,
    layout: Any = None,
) -> tuple[list[int], Any]:
    """Create a native parameter and its Python compiler value; return both its ID and that value."""
    runtime = impl.get_runtime()
    if kind == "ndarray":
        arg_id = runtime.compiling_callable.insert_ndarray_param(dtype, ndim, name, needs_grad)
        value = AnyArray(
            _qd_core.make_external_tensor_expr(dtype, ndim, arg_id, needs_grad, boundary), _qd_layout=layout
        )
        return arg_id, value

    is_ref = isinstance(dtype, RefType)
    if kind in ("matrix", "struct"):
        arg_type = get_type_for_kernel_args(dtype, name)
    elif kind == "sparse":
        arg_type = cook_dtype(u64)
    else:
        arg_type = cook_dtype(dtype.tp if is_ref else dtype)
    if is_ref:
        arg_id = runtime.compiling_callable.insert_pointer_param(arg_type, name)
    else:
        arg_id = runtime.compiling_callable.insert_scalar_param(arg_type, name)
    debug_info = _qd_core.DebugInfo(runtime.get_current_src_info())
    if kind == "sparse":
        value = SparseMatrixProxy(
            _qd_core.make_arg_load_expr(arg_id, arg_type, is_ptr=False, dbg_info=debug_info), cook_dtype(dtype)
        )
    else:
        value = Expr(
            _qd_core.make_arg_load_expr(arg_id, arg_type, is_ref, create_load=kind == "scalar", dbg_info=debug_info)
        )
        if kind in ("matrix", "struct"):
            value = dtype.from_quadrants_object(value)
    return arg_id, value


def decl_matrix_arg(matrixtype, name):
    return create_kernel_parameter("matrix", matrixtype, name)[1]


def decl_struct_arg(structtype, name):
    return create_kernel_parameter("struct", structtype, name)[1]


def decl_sparse_matrix(dtype, name):
    return create_kernel_parameter("sparse", dtype, name)[1]


def decl_ndarray_arg(
    element_type: DataTypeCxx,
    ndim: int,
    name: str,
    needs_grad: bool,
    boundary: BoundaryMode,
    layout=None,
) -> AnyArray:
    return create_kernel_parameter(
        "ndarray", element_type, name, ndim=ndim, needs_grad=needs_grad, boundary=boundary, layout=layout
    )[1]


def decl_ret(dtype):
    if isinstance(dtype, StructType):
        dtype = dtype.dtype
    if isinstance(dtype, MatrixType):
        dtype = _qd_core.get_type_factory_instance().get_tensor_type([dtype.n, dtype.m], dtype.dtype)
    else:
        dtype = cook_dtype(dtype)
    impl.get_runtime().compiling_callable.insert_ret(dtype)
