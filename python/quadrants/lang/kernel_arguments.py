# type: ignore

import inspect

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

    def __init__(self, annotation, name, default=inspect.Parameter.empty):
        self.annotation = annotation
        self.name = name
        self.default = default

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(annotation={self.annotation}, name={self.name}, default={self.default})"


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
    is_ref = False
    if isinstance(dtype, RefType):
        is_ref = True
        dtype = dtype.tp
    dtype = cook_dtype(dtype)
    if is_ref:
        arg_id = impl.get_runtime().compiling_callable.insert_pointer_param(dtype, name)
    else:
        arg_id = impl.get_runtime().compiling_callable.insert_scalar_param(dtype, name)

    argload_di = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
    return Expr(_qd_core.make_arg_load_expr(arg_id, dtype, is_ref, create_load=True, dbg_info=argload_di))


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


def decl_matrix_arg(matrixtype, name):
    arg_type = get_type_for_kernel_args(matrixtype, name)
    arg_id = impl.get_runtime().compiling_callable.insert_scalar_param(arg_type, name)
    argload_di = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
    arg_load = Expr(_qd_core.make_arg_load_expr(arg_id, arg_type, create_load=False, dbg_info=argload_di))
    return matrixtype.from_quadrants_object(arg_load)


def decl_struct_arg(structtype, name):
    arg_type = get_type_for_kernel_args(structtype, name)
    arg_id = impl.get_runtime().compiling_callable.insert_scalar_param(arg_type, name)
    argload_di = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
    arg_load = Expr(_qd_core.make_arg_load_expr(arg_id, arg_type, create_load=False, dbg_info=argload_di))
    return structtype.from_quadrants_object(arg_load)


def decl_sparse_matrix(dtype, name):
    value_type = cook_dtype(dtype)
    ptr_type = cook_dtype(u64)
    # Treat the sparse matrix argument as a scalar since we only need to pass in the base pointer
    arg_id = impl.get_runtime().compiling_callable.insert_scalar_param(ptr_type, name)
    argload_di = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
    return SparseMatrixProxy(
        _qd_core.make_arg_load_expr(arg_id, ptr_type, is_ptr=False, dbg_info=argload_di), value_type
    )


def decl_ndarray_arg(
    element_type: DataTypeCxx, ndim: int, name: str, needs_grad: bool, boundary: BoundaryMode
) -> AnyArray:
    arg_id = impl.get_runtime().compiling_callable.insert_ndarray_param(element_type, ndim, name, needs_grad)
    return AnyArray(_qd_core.make_external_tensor_expr(element_type, ndim, arg_id, needs_grad, boundary))


def decl_ret(dtype):
    if isinstance(dtype, StructType):
        dtype = dtype.dtype
    if isinstance(dtype, MatrixType):
        dtype = _qd_core.get_type_factory_instance().get_tensor_type([dtype.n, dtype.m], dtype.dtype)
    else:
        dtype = cook_dtype(dtype)
    impl.get_runtime().compiling_callable.insert_ret(dtype)
