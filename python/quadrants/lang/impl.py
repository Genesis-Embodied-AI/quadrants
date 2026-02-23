import numbers
import threading
import weakref
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import (
    DataTypeCxx,
    Function,
    KernelCxx,
    Program,
)
from quadrants._snode import fields_builder
from quadrants.lang._ndarray import ScalarNdarray
from quadrants.lang._ndrange import GroupedNDRange, _Ndrange
from quadrants.lang.any_array import AnyArray
from quadrants.lang.exception import (
    QuadrantsCompilationError,
    QuadrantsRuntimeError,
    QuadrantsSyntaxError,
    QuadrantsTypeError,
)
from quadrants.lang.expr import Expr, make_expr_group
from quadrants.lang.field import Field, ScalarField
from quadrants.lang.kernel import Kernel
from quadrants.lang.kernel_arguments import SparseMatrixProxy
from quadrants.lang.kernel_impl import BoundQuadrantsCallable, QuadrantsCallable
from quadrants.lang.matrix import (
    Matrix,
    MatrixField,
    MatrixNdarray,
    MatrixType,
    Vector,
    VectorNdarray,
    make_matrix,
)
from quadrants.lang.mesh import (
    ConvType,
    MeshElementFieldProxy,
    MeshInstance,
    MeshRelationAccessProxy,
    MeshReorderedMatrixFieldProxy,
    MeshReorderedScalarFieldProxy,
    element_type_name,
)
from quadrants.lang.simt.block import SharedArray
from quadrants.lang.snode import SNode
from quadrants.lang.struct import Struct, StructField, _IntermediateStruct
from quadrants.lang.util import (
    cook_dtype,
    get_traceback,
    is_quadrants_class,
    python_scope,
    quadrants_scope,
    warning,
)
from quadrants.types.enums import SNodeGradType
from quadrants.types.ndarray_type import NdarrayType
from quadrants.types.primitive_types import (
    all_types,
    f16,
    f32,
    f64,
    i32,
    i64,
    u8,
    u32,
    u64,
)

if TYPE_CHECKING:
    from quadrants.lang._ndarray import Ndarray

    from .ast.ast_transformer_utils import ASTTransformerGlobalContext


@quadrants_scope
def expr_init_shared_array(shape, element_type):
    ast_builder = get_runtime().compiling_callable.ast_builder()
    debug_info = _qd_core.DebugInfo(get_runtime().get_current_src_info())
    return ast_builder.expr_alloca_shared_array(shape, element_type, debug_info)


@quadrants_scope
def expr_init(rhs):
    compiling_callable = get_runtime().compiling_callable
    if rhs is None:
        return Expr(
            compiling_callable.ast_builder().expr_alloca(_qd_core.DebugInfo(get_runtime().get_current_src_info()))
        )
    if isinstance(rhs, Matrix) and (hasattr(rhs, "_DIM")):
        return Matrix(*rhs.to_list(), ndim=rhs.ndim)  # type: ignore
    if isinstance(rhs, Matrix):
        return make_matrix(rhs.to_list())
    if isinstance(rhs, SharedArray):
        return rhs
    if isinstance(rhs, Struct):
        return Struct(rhs.to_dict(include_methods=True, include_ndim=True))
    if isinstance(rhs, list):
        return [expr_init(e) for e in rhs]
    if isinstance(rhs, tuple):
        return tuple(expr_init(e) for e in rhs)
    if isinstance(rhs, dict):
        return dict((key, expr_init(val)) for key, val in rhs.items())
    if isinstance(rhs, _qd_core.DataTypeCxx):
        return rhs
    if isinstance(rhs, _qd_core.Arch):
        return rhs
    if isinstance(rhs, _Ndrange):
        return rhs
    if isinstance(rhs, MeshElementFieldProxy):
        return rhs
    if isinstance(rhs, MeshRelationAccessProxy):
        return rhs
    if hasattr(rhs, "_data_oriented"):
        return rhs
    return Expr(
        compiling_callable.ast_builder().expr_var(
            Expr(rhs).ptr, _qd_core.DebugInfo(get_runtime().get_current_src_info())
        )
    )


@quadrants_scope
def expr_init_func(rhs):  # temporary solution to allow passing in fields as arguments
    if isinstance(rhs, Field):
        return rhs
    return expr_init(rhs)


def begin_frontend_struct_for(ast_builder, group, loop_range):
    if not isinstance(loop_range, (AnyArray, Field, SNode, _Root)):
        raise TypeError(
            f"Cannot loop over the object {type(loop_range)} in Quadrants scope. Only Quadrants fields (via template) or dense arrays (via types.ndarray) are supported."
        )
    if group.size() != len(loop_range.shape):
        raise IndexError(
            "Number of struct-for indices does not match loop variable dimensionality "
            f"({group.size()} != {len(loop_range.shape)}). Maybe you wanted to "
            'use "for I in qd.grouped(x)" to group all indices into a single vector I?'
        )
    dbg_info = _qd_core.DebugInfo(get_runtime().get_current_src_info())
    if isinstance(loop_range, AnyArray):
        ast_builder.begin_frontend_struct_for_on_external_tensor(group, loop_range._loop_range(), dbg_info)
    else:
        ast_builder.begin_frontend_struct_for_on_snode(group, loop_range._loop_range(), dbg_info)


def begin_frontend_if(ast_builder, cond, stmt_dbg_info):
    assert ast_builder is not None
    if is_quadrants_class(cond):
        raise ValueError(
            "The truth value of vectors/matrices is ambiguous.\n"
            "Consider using `any` or `all` when comparing vectors/matrices:\n"
            "    if all(x == y):\n"
            "or\n"
            "    if any(x != y):\n"
        )
    ast_builder.begin_frontend_if(Expr(cond).ptr, stmt_dbg_info)


@quadrants_scope
def _calc_slice(index, default_stop):
    start, stop, step = index.start or 0, index.stop or default_stop, index.step or 1

    def check_validity(x):
        #  TODO(mzmzm): support variable in slice
        if isinstance(x, Expr):
            raise QuadrantsCompilationError(
                "Quadrants does not support variables in slice now, please use constant instead of it."
            )

    _ = check_validity(start), check_validity(stop), check_validity(step)
    return [_ for _ in range(start, stop, step)]


def validate_subscript_index(value, index):
    if isinstance(value, Field):
        # field supports negative indices
        return

    if isinstance(index, Expr):
        return

    if isinstance(index, Iterable):
        for ind in index:
            validate_subscript_index(value, ind)

    if isinstance(index, slice):
        validate_subscript_index(value, index.start)
        validate_subscript_index(value, index.stop)

    if isinstance(index, int) and index < 0:
        raise QuadrantsSyntaxError("Negative indices are not supported in Quadrants kernels.")


@quadrants_scope
def subscript(ast_builder, value, *_indices, skip_reordered=False):
    dbg_info = _qd_core.DebugInfo(get_runtime().get_current_src_info())
    ast_builder = get_runtime().compiling_callable.ast_builder()
    # Directly evaluate in Python for non-Quadrants types
    if not isinstance(
        value,
        (
            Expr,
            Field,
            AnyArray,
            SparseMatrixProxy,
            MeshElementFieldProxy,
            MeshRelationAccessProxy,
            SharedArray,
        ),
    ):
        if isinstance(value, NdarrayType):
            raise Exception(
                "Cannot subscript NdarrayType. Did you access a global py dataclass inadvertently?", value, type(value)
            )
        if len(_indices) == 1:
            _indices = _indices[0]
        return value.__getitem__(_indices)

    has_slice = False

    flattened_indices = []
    for _index in _indices:
        if isinstance(_index, Matrix):
            ind = _index.to_list()
        elif isinstance(_index, slice):
            ind = [_index]
            has_slice = True
        else:
            ind = [_index]
        flattened_indices += ind
    indices = tuple(flattened_indices)
    validate_subscript_index(value, indices)

    if len(indices) == 1 and indices[0] is None:
        indices = ()

    indices_expr_group = None
    if has_slice:
        if not (isinstance(value, Expr) and value.is_tensor()):
            raise QuadrantsSyntaxError(f"The type {type(value)} do not support index of slice type")
    else:
        indices_expr_group = make_expr_group(*indices)

    if isinstance(value, SharedArray):
        return value.subscript(*indices)
    if isinstance(value, MeshElementFieldProxy):
        return value.subscript(*indices)  # type: ignore
    if isinstance(value, MeshRelationAccessProxy):
        return value.subscript(*indices)
    if isinstance(value, (MeshReorderedScalarFieldProxy, MeshReorderedMatrixFieldProxy)) and not skip_reordered:
        assert len(indices) > 0
        reordered_index = tuple(
            [
                Expr(
                    ast_builder.mesh_index_conversion(
                        value.mesh_ptr, value.element_type, Expr(indices[0]).ptr, ConvType.g2r, dbg_info
                    )
                )
            ]
        )
        return subscript(ast_builder, value, *reordered_index, skip_reordered=True)
    if isinstance(value, SparseMatrixProxy):
        return value.subscript(*indices)
    if isinstance(value, Field):
        _var = value._get_field_members()[0].ptr
        snode = _var.snode()
        if snode is None:
            if _var.is_primal():
                raise RuntimeError(f"{_var.get_expr_name()} has not been placed.")
            else:
                raise RuntimeError(
                    f"Gradient {_var.get_expr_name()} has not been placed, check whether `needs_grad=True`"
                )

        assert indices_expr_group is not None
        if isinstance(value, MatrixField):
            return Expr(ast_builder.expr_subscript(value.ptr, indices_expr_group, dbg_info))
        if isinstance(value, StructField):
            entries = {k: subscript(ast_builder, v, *indices) for k, v in value._items}
            entries["__struct_methods"] = value.struct_methods
            return _IntermediateStruct(entries)
        return Expr(ast_builder.expr_subscript(_var, indices_expr_group, dbg_info))
    if isinstance(value, AnyArray):
        assert indices_expr_group is not None
        return Expr(ast_builder.expr_subscript(value.ptr, indices_expr_group, dbg_info))
    assert isinstance(value, Expr)
    # Index into TensorType
    # value: IndexExpression with ret_type = TensorType
    assert value.is_tensor()

    if has_slice:
        shape = value.get_shape()
        dim = len(shape)
        assert dim == len(indices)
        indices = [
            _calc_slice(index, shape[i]) if isinstance(index, slice) else index for i, index in enumerate(indices)
        ]
        if dim == 1:
            assert isinstance(indices[0], list)
            multiple_indices = [make_expr_group(i) for i in indices[0]]
            return_shape = (len(indices[0]),)
        else:
            assert dim == 2
            if isinstance(indices[0], list) and isinstance(indices[1], list):
                multiple_indices = [make_expr_group(i, j) for i in indices[0] for j in indices[1]]
                return_shape = (len(indices[0]), len(indices[1]))
            elif isinstance(indices[0], list):  # indices[1] is not list
                multiple_indices = [make_expr_group(i, indices[1]) for i in indices[0]]
                return_shape = (len(indices[0]),)
            else:  # indices[0] is not list while indices[1] is list
                multiple_indices = [make_expr_group(indices[0], j) for j in indices[1]]
                return_shape = (len(indices[1]),)
        return Expr(
            _qd_core.subscript_with_multiple_indices(
                value.ptr,
                multiple_indices,
                return_shape,
                dbg_info,
            )
        )
    return Expr(ast_builder.expr_subscript(value.ptr, indices_expr_group, dbg_info))


class SrcInfoGuard:
    def __init__(self, info_stack, info):
        self.info_stack = info_stack
        self.info = info

    def __enter__(self):
        self.info_stack.append(self.info)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.info_stack.pop()


class PyQuadrants:
    def __init__(self, kernels=None):
        self.materialized = False
        self._prog: Program | None = None
        self.src_info_stack = []
        self.inside_kernel: bool = False
        self.compilation_lock = threading.RLock()
        self._compiling_callable: KernelCxx | Kernel | Function | None = None
        self._current_global_context: "ASTTransformerGlobalContext | None" = None
        self.global_vars = []
        self.grad_vars = []
        self.dual_vars = []
        self.matrix_fields = []
        self.default_fp = f32
        self.default_ip = i32
        self.default_up = u32
        self.print_full_traceback: bool = False
        self.target_tape = None
        self.fwd_mode_manager = None
        self.grad_replaced = False
        self.kernels: list[Kernel] = kernels or []
        self.ndarrays: weakref.WeakSet[Ndarray] = weakref.WeakSet()
        self._signal_handler_registry = None
        self.unfinalized_fields_builder = {}
        self.print_non_pure: bool = False
        self.short_circuit_operators: bool = False
        self.unrolling_limit: int = 0
        self.src_ll_cache: bool = True

    @property
    def compiling_callable(self) -> KernelCxx | Kernel | Function:
        if self._compiling_callable is None:
            raise QuadrantsRuntimeError(
                "_compiling_callable attribute not initialized. Maybe you forgot to call `qd.init()` first?"
            )
        return self._compiling_callable

    @property
    def prog(self) -> Program:
        if self._prog is None:
            raise QuadrantsRuntimeError("_prog attribute not initialized. Maybe you forgot to call `qd.init()` first?")
        return self._prog

    def initialize_fields_builder(self, builder):
        self.unfinalized_fields_builder[builder] = get_traceback(2)

    def clear_compiled_functions(self):
        for k in self.kernels:
            k.materialized_kernels.clear()

    def finalize_fields_builder(self, builder):
        self.unfinalized_fields_builder.pop(builder)

    def validate_fields_builder(self):
        for builder, tb in self.unfinalized_fields_builder.items():
            if builder == _root_fb:
                continue

            raise QuadrantsRuntimeError(
                f"Field builder {builder} is not finalized. " f"Please call finalize() on it. Traceback:\n{tb}"
            )

    def get_num_compiled_functions(self):
        count = 0
        for k in self.kernels:
            count += len(k.materialized_kernels)
        return count

    def src_info_guard(self, info):
        return SrcInfoGuard(self.src_info_stack, info)

    def get_current_src_info(self):
        return self.src_info_stack[-1]

    def set_default_fp(self, fp):
        assert fp in [f16, f32, f64]
        self.default_fp = fp
        default_cfg().default_fp = self.default_fp

    def set_default_ip(self, ip):
        assert ip in [i32, i64]
        self.default_ip = ip
        self.default_up = u32 if ip == i32 else u64
        default_cfg().default_ip = self.default_ip
        default_cfg().default_up = self.default_up

    def create_program(self):
        if self._prog is None:
            self._prog = _qd_core.Program()

    @staticmethod
    def materialize_root_fb(is_first_call):
        if root.finalized:
            return
        if not is_first_call and root.empty:
            # We have to forcefully finalize when `is_first_call` is True (even
            # if the root itself is empty), so that there is a valid struct
            # llvm::Module, if no field has been declared before the first kernel
            # invocation. Example case:
            # https://github.com/taichi-dev/quadrants/blob/27bb1dc3227d9273a79fcb318fdb06fd053068f5/tests/python/test_ad_basics.py#L260-L266
            return

        if get_runtime().prog.config().debug:
            if not root.finalized:
                root._allocate_adjoint_checkbit()

        root.finalize(raise_warning=not is_first_call)
        global _root_fb
        _root_fb = fields_builder.FieldsBuilder()

    @staticmethod
    def _get_tb(_var):
        return getattr(_var, "declaration_tb", str(_var.ptr))

    def _check_field_not_placed(self):
        not_placed = []
        for _var in self.global_vars:
            if _var.ptr.snode() is None:
                not_placed.append(self._get_tb(_var))

        if len(not_placed):
            bar = "=" * 44 + "\n"
            raise RuntimeError(
                f"These field(s) are not placed:\n{bar}"
                + f"{bar}".join(not_placed)
                + f"{bar}Please consider specifying a shape for them. E.g.,"
                + "\n\n  x = qd.field(float, shape=(2, 3))"
            )

    def _check_gradient_field_not_placed(self, gradient_type):
        if gradient_type == "grad":
            gradient_vars = self.grad_vars
        elif gradient_type == "dual":
            gradient_vars = self.dual_vars
        else:
            return

        not_placed = set()
        for _var in gradient_vars:
            if _var.ptr.snode() is None:
                not_placed.add(self._get_tb(_var))

        if not_placed:
            bar = "=" * 44 + "\n"
            raise RuntimeError(
                f"These field(s) requrie `needs_{gradient_type}=True`, however their {gradient_type} field(s) are not placed:\n{bar}"
                + f"{bar}".join(not_placed)
                + f"{bar}Please consider place the {gradient_type} field(s). E.g.,"
                + "\n\n  qd.root.dense(qd.i, 1).place(x.{gradient_type})"
                + "\n\n Or specify a shape for the field(s). E.g.,"
                + "\n\n  x = qd.field(float, shape=(2, 3), needs_{gradient_type}=True)"
            )

    def _check_matrix_field_member_shape(self):
        for _field in self.matrix_fields:
            shapes = [_field.get_scalar_field(i, j).shape for i in range(_field.n) for j in range(_field.m)]
            if any(shape != shapes[0] for shape in shapes):
                raise RuntimeError(
                    "Members of the following field have different shapes "
                    + f"{shapes}:\n{self._get_tb(_field._get_field_members()[0])}"
                )

    def _calc_matrix_field_dynamic_index_stride(self):
        for _field in self.matrix_fields:
            _field._calc_dynamic_index_stride()

    def materialize(self):
        self.materialize_root_fb(not self.materialized)
        self.materialized = True

        self.validate_fields_builder()

        self._check_field_not_placed()
        self._check_gradient_field_not_placed("grad")
        self._check_gradient_field_not_placed("dual")
        self._check_matrix_field_member_shape()
        self._calc_matrix_field_dynamic_index_stride()
        self.global_vars.clear()
        self.grad_vars.clear()
        self.dual_vars.clear()
        self.matrix_fields.clear()

    def _register_signal_handlers(self):
        if self._signal_handler_registry is None:
            self._signal_handler_registry = _qd_core.HackedSignalRegister()

    def clear(self):
        if self._prog:
            self._prog.finalize()
            self._prog = None
        self._signal_handler_registry = None
        self.materialized = False

    def sync(self):
        self.materialize()
        assert self._prog is not None
        self._prog.synchronize()


pyquadrants = PyQuadrants()


def get_runtime() -> PyQuadrants:
    return pyquadrants


def reset():
    global pyquadrants
    old_ndarrays = pyquadrants.ndarrays
    old_kernels = pyquadrants.kernels
    pyquadrants.clear()
    pyquadrants = PyQuadrants(old_kernels)
    for nd in old_ndarrays:
        nd._reset()
    for k in old_kernels:
        k.reset()
    _qd_core.reset_default_compile_config()


@quadrants_scope
def static_print(*args, __p=print, **kwargs):
    """The print function in Quadrants scope.

    This function is called at compile time and has no runtime overhead.
    """
    __p(*args, **kwargs)


# we don't add @quadrants_scope decorator for @qd.pyfunc to work
def static_assert(cond, msg=None):
    """Throw AssertionError when `cond` is False.

    This function is called at compile time and has no runtime overhead.
    The bool value in `cond` must can be determined at compile time.

    Args:
        cond (bool): an expression with a bool value.
        msg (str): assertion message.

    Example::

        >>> year = 2001
        >>> @qd.kernel
        >>> def test():
        >>>     qd.static_assert(year % 4 == 0, "the year must be a lunar year")
        AssertionError: the year must be a lunar year
    """
    if isinstance(cond, Expr):
        raise QuadrantsTypeError("Static assert with non-static condition")
    if msg is not None:
        assert cond, msg
    else:
        assert cond


def inside_kernel():
    return pyquadrants.inside_kernel


def index_nd(dim):
    return axes(*range(dim))


class _UninitializedRootFieldsBuilder:
    def __getattr__(self, item):
        if item == "__qualname__":
            # For sphinx docstring extraction.
            return "_UninitializedRootFieldsBuilder"
        raise QuadrantsRuntimeError("Please call init() first")


# `root` initialization must be delayed until after the program is
# created. Unfortunately, `root` exists in both quadrants.lang.impl module and
# the top-level quadrants module at this point; so if `root` itself is written, we
# would have to make sure that `root` in all the modules get updated to the same
# instance. This is an error-prone process.
#
# To avoid this situation, we create `root` once during the import time, and
# never write to it. The core part, `_root_fb`, is the one whose initialization
# gets delayed. `_root_fb` will only exist in the quadrants.lang.impl module, so
# writing to it is would result in less for maintenance cost.
#
# `_root_fb` will be overridden inside :func:`quadrants.lang.init`.
_root_fb = _UninitializedRootFieldsBuilder()


def deactivate_all_snodes():
    """Recursively deactivate all SNodes."""
    for root_fb in fields_builder.FieldsBuilder._finalized_roots():
        root_fb.deactivate_all()


class _Root:
    """Wrapper around the default root FieldsBuilder instance."""

    @staticmethod
    def parent(n=1):
        """Same as :func:`quadrants.SNode.parent`"""
        assert isinstance(_root_fb, fields_builder.FieldsBuilder)
        return _root_fb.root.parent(n)

    @staticmethod
    def _loop_range():
        """Same as :func:`quadrants.SNode.loop_range`"""
        assert isinstance(_root_fb, fields_builder.FieldsBuilder)
        return _root_fb.root._loop_range()

    @staticmethod
    def _get_children():
        """Same as :func:`quadrants.SNode.get_children`"""
        assert isinstance(_root_fb, fields_builder.FieldsBuilder)
        return _root_fb.root._get_children()

    # TODO: Record all of the SNodeTrees that finalized under 'qd.root'
    @staticmethod
    def deactivate_all():
        warning("""'qd.root.deactivate_all()' would deactivate all finalized snodes.""")
        deactivate_all_snodes()

    @property
    def shape(self):
        """Same as :func:`quadrants.SNode.shape`"""
        assert isinstance(_root_fb, fields_builder.FieldsBuilder)
        return _root_fb.root.shape

    @property
    def _id(self):
        assert isinstance(_root_fb, fields_builder.FieldsBuilder)
        return _root_fb.root._id

    def __getattr__(self, item):
        return getattr(_root_fb, item)

    def __repr__(self):
        return "qd.root"


root = _Root()
"""Root of the declared Quadrants :func:`~quadrants.lang.impl.field`s.

See also https://docs.taichi-lang.org/docs/layout

Example::

    >>> x = qd.field(qd.f32)
    >>> qd.root.pointer(qd.ij, 4).dense(qd.ij, 8).place(x)
"""


def _create_snode(axis_seq: Sequence[int], shape_seq: Sequence[numbers.Number], same_level: bool):
    dim = len(axis_seq)
    assert dim == len(shape_seq)
    snode = root
    if same_level:
        snode = snode.dense(axes(*axis_seq), shape_seq)
    else:
        for i in range(dim):
            snode = snode.dense(axes(axis_seq[i]), (shape_seq[i],))
    return snode


@python_scope
def create_field_member(dtype, name, needs_grad, needs_dual):
    dtype = cook_dtype(dtype)

    # primal
    prog = get_runtime().prog

    x = Expr(prog.make_id_expr(""))
    x.declaration_tb = get_traceback(stacklevel=4)
    x.ptr = _qd_core.expr_field(x.ptr, dtype)
    x.ptr.set_name(name)
    x.ptr.set_grad_type(SNodeGradType.PRIMAL)
    pyquadrants.global_vars.append(x)

    x_grad = None
    x_dual = None
    # The x_grad_checkbit is used for global data access rule checker
    x_grad_checkbit = None
    if _qd_core.is_real(dtype):
        # adjoint
        x_grad = Expr(prog.make_id_expr(""))
        x_grad.declaration_tb = get_traceback(stacklevel=4)
        x_grad.ptr = _qd_core.expr_field(x_grad.ptr, dtype)
        x_grad.ptr.set_name(name + ".grad")
        x_grad.ptr.set_grad_type(SNodeGradType.ADJOINT)
        x.ptr.set_adjoint(x_grad.ptr)
        if needs_grad:
            pyquadrants.grad_vars.append(x_grad)

        if prog.config().debug:
            # adjoint checkbit
            x_grad_checkbit = Expr(prog.make_id_expr(""))
            dtype = u8
            if prog.config().arch == _qd_core.vulkan:
                dtype = i32
            x_grad_checkbit.ptr = _qd_core.expr_field(x_grad_checkbit.ptr, cook_dtype(dtype))
            x_grad_checkbit.ptr.set_name(name + ".grad_checkbit")
            x_grad_checkbit.ptr.set_grad_type(SNodeGradType.ADJOINT_CHECKBIT)
            x.ptr.set_adjoint_checkbit(x_grad_checkbit.ptr)

        # dual
        x_dual = Expr(prog.make_id_expr(""))
        x_dual.ptr = _qd_core.expr_field(x_dual.ptr, dtype)
        x_dual.ptr.set_name(name + ".dual")
        x_dual.ptr.set_grad_type(SNodeGradType.DUAL)
        x.ptr.set_dual(x_dual.ptr)
        if needs_dual:
            pyquadrants.dual_vars.append(x_dual)
    elif needs_grad or needs_dual:
        raise QuadrantsRuntimeError(f"{dtype} is not supported for field with `needs_grad=True` or `needs_dual=True`.")

    return x, x_grad, x_dual


@python_scope
def _field(
    dtype,
    shape=None,
    order=None,
    name="",
    offset=None,
    needs_grad=False,
    needs_dual=False,
):
    x, x_grad, x_dual = create_field_member(dtype, name, needs_grad, needs_dual)
    x = ScalarField(x)
    if x_grad:
        x_grad = ScalarField(x_grad)
        x._set_grad(x_grad)
    if x_dual:
        x_dual = ScalarField(x_dual)
        x._set_dual(x_dual)

    if shape is None:
        if offset is not None:
            raise QuadrantsSyntaxError("shape cannot be None when offset is set")
        if order is not None:
            raise QuadrantsSyntaxError("shape cannot be None when order is set")
    else:
        if isinstance(shape, numbers.Number):
            shape = (shape,)
        if isinstance(offset, numbers.Number):
            offset = (offset,)
        dim = len(shape)
        if offset is not None and dim != len(offset):
            raise QuadrantsSyntaxError(
                f"The dimensionality of shape and offset must be the same ({dim} != {len(offset)})"
            )
        axis_seq = []
        shape_seq = []
        if order is not None:
            if dim != len(order):
                raise QuadrantsSyntaxError(
                    f"The dimensionality of shape and order must be the same ({dim} != {len(order)})"
                )
            if dim != len(set(order)):
                raise QuadrantsSyntaxError("The axes in order must be different")
            for ch in order:
                axis = ord(ch) - ord("i")
                if axis < 0 or axis >= dim:
                    raise QuadrantsSyntaxError(f"Invalid axis {ch}")
                axis_seq.append(axis)
                shape_seq.append(shape[axis])
        else:
            axis_seq = list(range(dim))
            shape_seq = list(shape)
        same_level = order is None
        _create_snode(axis_seq, shape_seq, same_level).place(x, offset=offset)
        if needs_grad:
            _create_snode(axis_seq, shape_seq, same_level).place(x_grad, offset=offset)
        if needs_dual:
            _create_snode(axis_seq, shape_seq, same_level).place(x_dual, offset=offset)
    return x


@python_scope
def field(dtype, *args, **kwargs):
    """Defines a Quadrants field.

    A Quadrants field can be viewed as an abstract N-dimensional array, hiding away
    the complexity of how its underlying :class:`~quadrants.lang.snode.SNode` are
    actually defined. The data in a Quadrants field can be directly accessed by
    a Quadrants :func:`~quadrants.lang.kernel_impl.kernel`.

    See also https://docs.taichi-lang.org/docs/field

    Args:
        dtype (DataType): data type of the field. Note it can be vector or matrix types as well.
        shape (Union[int, tuple[int]], optional): shape of the field.
        order (str, optional): order of the shape laid out in memory.
        name (str, optional): name of the field.
        offset (Union[int, tuple[int]], optional): offset of the field domain.
        needs_grad (bool, optional): whether this field participates in autodiff (reverse mode)
            and thus needs an adjoint field to store the gradients.
        needs_dual (bool, optional): whether this field participates in autodiff (forward mode)
            and thus needs an dual field to store the gradients.

    Example::

        The code below shows how a Quadrants field can be declared and defined::

            >>> x1 = qd.field(qd.f32, shape=(16, 8))
            >>> # Equivalently
            >>> x2 = qd.field(qd.f32)
            >>> qd.root.dense(qd.ij, shape=(16, 8)).place(x2)
            >>>
            >>> x3 = qd.field(qd.f32, shape=(16, 8), order='ji')
            >>> # Equivalently
            >>> x4 = qd.field(qd.f32)
            >>> qd.root.dense(qd.j, shape=8).dense(qd.i, shape=16).place(x4)
            >>>
            >>> x5 = qd.field(qd.math.vec3, shape=(16, 8))

    """
    if isinstance(dtype, MatrixType):
        if dtype.ndim == 1:
            return Vector.field(dtype.n, dtype.dtype, *args, **kwargs)
        return Matrix.field(dtype.n, dtype.m, dtype.dtype, *args, **kwargs)
    return _field(dtype, *args, **kwargs)


@python_scope
def ndarray(dtype, shape, needs_grad=False):
    """Defines a Quadrants ndarray with scalar elements.

    Args:
        dtype (Union[DataType, MatrixType]): Data type of each element. This can be either a scalar type like qd.f32 or a compound type like qd.types.vector(3, qd.i32).
        shape (Union[int, tuple[int]]): Shape of the ndarray.

    Example:
        The code below shows how a Quadrants ndarray with scalar elements can be declared and defined::

            >>> x = qd.ndarray(qd.f32, shape=(16, 8))  # ndarray of shape (16, 8), each element is qd.f32 scalar.
            >>> vec3 = qd.types.vector(3, qd.i32)
            >>> y = qd.ndarray(vec3, shape=(10, 2))  # ndarray of shape (10, 2), each element is a vector of 3 qd.i32 scalars.
            >>> matrix_ty = qd.types.matrix(3, 4, float)
            >>> z = qd.ndarray(matrix_ty, shape=(4, 5))  # ndarray of shape (4, 5), each element is a matrix of (3, 4) qd.float scalars.
    """
    # primal
    if isinstance(shape, numbers.Number):
        shape = (shape,)
    if not all((isinstance(x, int) or isinstance(x, np.integer)) and x > 0 and x <= 2**31 - 1 for x in shape):
        raise QuadrantsRuntimeError(f"{shape} is not a valid shape for ndarray")
    if dtype in all_types:
        dt = cook_dtype(dtype)
        x = ScalarNdarray(dt, shape)
    elif isinstance(dtype, MatrixType):
        if dtype.ndim == 1:
            x = VectorNdarray(dtype.n, dtype.dtype, shape)
        else:
            x = MatrixNdarray(dtype.n, dtype.m, dtype.dtype, shape)
        dt = dtype.dtype
    else:
        raise QuadrantsRuntimeError(f"{dtype} is not supported as ndarray element type")
    if needs_grad:
        assert isinstance(dt, DataTypeCxx)
        if not _qd_core.is_real(dt):
            raise QuadrantsRuntimeError(
                f"{dt} is not supported for ndarray with `needs_grad=True` or `needs_dual=True`."
            )
        x_grad = ndarray(dtype, shape, needs_grad=False)
        x._set_grad(x_grad)
    return x


@quadrants_scope
def ti_format_list_to_content_entries(raw):
    # return a pair of [content, format]
    def entry2content(_var):
        if isinstance(_var, str):
            return [_var, None]
        if isinstance(_var, list):
            assert len(_var) == 2 and (isinstance(_var[1], str) or _var[1] is None)
            _var[0] = Expr(_var[0]).ptr
            return _var
        return [Expr(_var).ptr, None]

    def list_qd_repr(_var):
        yield "["  # distinguishing tuple & list will increase maintenance cost
        for i, v in enumerate(_var):
            if i:
                yield ", "
            yield v
        yield "]"

    def vars2entries(_vars):
        for _var in _vars:
            # If the first element is '__qd_fmt_value__', this list is an Expr and its format.
            if isinstance(_var, list) and len(_var) == 3 and isinstance(_var[0], str) and _var[0] == "__qd_fmt_value__":
                # yield [Expr, format] as a whole and don't pass it to vars2entries() again
                yield _var[1:]
                continue
            elif hasattr(_var, "__qd_repr__"):
                res = _var.__qd_repr__()  # type: ignore
            elif isinstance(_var, (list, tuple)):
                # If the first element is '__qd_format__', this list is the result of ti_format.
                if len(_var) > 0 and isinstance(_var[0], str) and _var[0] == "__qd_format__":
                    res = _var[1:]
                else:
                    res = list_qd_repr(_var)
            else:
                yield _var
                continue

            for v in vars2entries(res):
                yield v

    def fused_string(entries):
        accumated = ""
        for entry in entries:
            if isinstance(entry, str):
                accumated += entry
            else:
                if accumated:
                    yield accumated
                    accumated = ""
                yield entry
        if accumated:
            yield accumated

    def extract_formats(entries):
        contents, formats = zip(*entries)
        return list(contents), list(formats)

    entries = vars2entries(raw)
    entries = fused_string(entries)
    entries = [entry2content(entry) for entry in entries]
    return extract_formats(entries)


@quadrants_scope
def ti_print(*_vars, sep=" ", end="\n"):
    def add_separators(_vars):
        for i, _var in enumerate(_vars):
            if i:
                yield sep
            yield _var
        yield end

    _vars = add_separators(_vars)
    contents, formats = ti_format_list_to_content_entries(_vars)
    ast_builder = get_runtime().compiling_callable.ast_builder()
    debug_info = _qd_core.DebugInfo(get_runtime().get_current_src_info())
    ast_builder.create_print(contents, formats, debug_info)


@quadrants_scope
def ti_format(*args):
    content = args[0]
    mixed = args[1:]
    new_mixed = []
    args = []
    for x in mixed:
        # x is a (formatted) Expr
        if isinstance(x, Expr) or (isinstance(x, list) and len(x) == 3 and x[0] == "__qd_fmt_value__"):
            new_mixed.append("{}")
            args.append(x)
        else:
            new_mixed.append(x)
    content = content.format(*new_mixed)
    res = content.split("{}")
    assert len(res) == len(args) + 1, "Number of args is different from number of positions provided in string"

    for i, arg in enumerate(args):
        res.insert(i * 2 + 1, arg)
    res.insert(0, "__qd_format__")
    return res


@quadrants_scope
def ti_assert(cond, msg, extra_args, dbg_info):
    # Mostly a wrapper to help us convert from Expr (defined in Python) to
    # _qd_core.Expr (defined in C++)
    ast_builder = get_runtime().compiling_callable.ast_builder()
    ast_builder.create_assert_stmt(Expr(cond).ptr, msg, extra_args, dbg_info)


@quadrants_scope
def ti_int(_var):
    if hasattr(_var, "__qd_int__"):
        return _var.__qd_int__()
    return int(_var)


@quadrants_scope
def ti_bool(_var):
    if hasattr(_var, "__qd_bool__"):
        return _var.__qd_bool__()
    return bool(_var)


@quadrants_scope
def ti_float(_var):
    if hasattr(_var, "__qd_float__"):
        return _var.__qd_float__()
    return float(_var)


@quadrants_scope
def zero(x):
    # TODO: get dtype from Expr and Matrix:
    """Returns an array of zeros with the same shape and type as the input. It's also a scalar
    if the input is a scalar.

    Args:
        x (Union[:mod:`~quadrants.types.primitive_types`, :class:`~quadrants.Matrix`]): The input.

    Returns:
        A new copy of the input but filled with zeros.

    Example::

        >>> x = qd.Vector([1, 1])
        >>> @qd.kernel
        >>> def test():
        >>>     y = qd.zero(x)
        >>>     print(y)
        [0, 0]
    """
    return x * 0


@quadrants_scope
def one(x):
    """Returns an array of ones with the same shape and type as the input. It's also a scalar
    if the input is a scalar.

    Args:
        x (Union[:mod:`~quadrants.types.primitive_types`, :class:`~quadrants.Matrix`]): The input.

    Returns:
        A new copy of the input but filled with ones.

    Example::

        >>> x = qd.Vector([0, 0])
        >>> @qd.kernel
        >>> def test():
        >>>     y = qd.one(x)
        >>>     print(y)
        [1, 1]
    """
    return zero(x) + 1


def axes(*x: int):
    """Defines a list of axes to be used by a field.

    Args:
        *x: A list of axes to be activated

    Note that Quadrants has already provided a set of commonly used axes. For example,
    `qd.ij` is just `axes(0, 1)` under the hood.
    """
    return [_qd_core.Axis(i) for i in x]


Axis = _qd_core.Axis


def static(x, *xs) -> Any:
    """Evaluates a Quadrants-scope expression at compile time.

    `static()` is what enables the so-called metaprogramming in Quadrants. It is
    in many ways similar to ``constexpr`` in C++.

    See also https://docs.taichi-lang.org/docs/meta.

    Args:
        x (Any): an expression to be evaluated
        *xs (Any): for Python-ish swapping assignment

    Example:
        The most common usage of `static()` is for compile-time evaluation::

            >>> cond = False
            >>>
            >>> @qd.kernel
            >>> def run():
            >>>     if qd.static(cond):
            >>>         do_a()
            >>>     else:
            >>>         do_b()

        Depending on the value of ``cond``, ``run()`` will be directly compiled
        into either ``do_a()`` or ``do_b()``. Thus there won't be a runtime
        condition check.

        Another common usage is for compile-time loop unrolling::

            >>> @qd.kernel
            >>> def run():
            >>>     for i in qd.static(range(3)):
            >>>         print(i)
            >>>
            >>> # The above will be unrolled to:
            >>> @qd.kernel
            >>> def run():
            >>>     print(0)
            >>>     print(1)
            >>>     print(2)
    """
    if len(xs):  # for python-ish pointer assign: x, y = qd.static(y, x)
        return [static(x)] + [static(x) for x in xs]

    if (
        isinstance(
            x,
            (
                bool,
                int,
                float,
                range,
                list,
                tuple,
                enumerate,
                GroupedNDRange,
                _Ndrange,
                zip,
                filter,
                map,
            ),
        )
        or x is None
    ):
        return x

    if isinstance(x, (np.bool_, np.integer, np.floating)):
        return x

    if isinstance(x, AnyArray):
        return x

    if isinstance(x, Field):
        return x

    if isinstance(x, (FunctionType, MethodType, BoundQuadrantsCallable, QuadrantsCallable)):
        return x

    raise ValueError(f"Input to qd.static must be compile-time constants or global pointers, instead of {type(x)}")


@quadrants_scope
def grouped(x):
    """Groups the indices in the iterator returned by `ndrange()` into a 1-D vector.

    This is often used when you want to iterate over all indices returned by `ndrange()`
    in one `for` loop and a single index.

    Args:
        x (:func:`~quadrants.ndrange`): an iterator object returned by `qd.ndrange`.

    Example::
        >>> # without qd.grouped
        >>> for I in qd.ndrange(2, 3):
        >>>     print(I)
        prints 0, 1, 2, 3, 4, 5

        >>> # with qd.grouped
        >>> for I in qd.grouped(qd.ndrange(2, 3)):
        >>>     print(I)
        prints [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
    """
    if isinstance(x, _Ndrange):
        return x.grouped()
    return x


def stop_grad(x):
    """Stops computing gradients during back propagation.

    Args:
        x (:class:`~quadrants.Field`): A field.
    """
    compiling_callable = get_runtime().compiling_callable
    assert compiling_callable is not None
    compiling_callable.ast_builder().stop_grad(x.snode.ptr)


def current_cfg():
    return get_runtime().prog.config()


def default_cfg():
    return _qd_core.default_compile_config()


def call_internal(name, *args, with_runtime_context=True):
    return expr_init(_qd_core.insert_internal_func_call(getattr(_qd_core.InternalOp, name), make_expr_group(args)))


def get_cuda_compute_capability():
    return _qd_core.query_int64("cuda_compute_capability")


@quadrants_scope
def mesh_relation_access(mesh, from_index, to_element_type):
    # to support qd.mesh_local and access mesh attribute as field
    if isinstance(from_index, MeshInstance):
        return getattr(from_index, element_type_name(to_element_type))
    if isinstance(mesh, MeshInstance):
        return MeshRelationAccessProxy(mesh, from_index, to_element_type)
    raise RuntimeError("Relation access should be with a mesh instance!")


__all__ = [
    "axes",
    "deactivate_all_snodes",
    "field",
    "grouped",
    "ndarray",
    "one",
    "root",
    "static",
    "static_assert",
    "static_print",
    "stop_grad",
    "zero",
]
