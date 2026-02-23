import ast
import inspect
import math
import textwrap
import types
import typing
import warnings
from dataclasses import (
    _FIELD,  # type: ignore[reportAttributeAccessIssue]
    _FIELDS,  # type: ignore[reportAttributeAccessIssue]
    is_dataclass,
)

# Must import 'partial' directly instead of the entire module to avoid attribute lookup overhead.
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Type

import numpy as np

from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import KernelLaunchContext
from quadrants.lang import _kernel_impl_dataclass, impl
from quadrants.lang._dataclass_util import create_flat_name
from quadrants.lang._ndarray import Ndarray
from quadrants.lang._wrap_inspect import get_source_info_and_src
from quadrants.lang.ast import ASTTransformerFuncContext
from quadrants.lang.exception import (
    QuadrantsRuntimeError,
    QuadrantsRuntimeTypeError,
    QuadrantsSyntaxError,
)
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.matrix import MatrixType
from quadrants.lang.struct import StructType
from quadrants.lang.util import cook_dtype, has_pytorch
from quadrants.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
)

from ._exceptions import raise_exception
from .ast.ast_transformer_utils import ASTTransformerGlobalContext

if TYPE_CHECKING:
    from quadrants._lib.core.quadrants_python import ASTBuilder

    from ._pruning import Pruning
    from .kernel import Kernel
from quadrants.types.enums import Layout
from quadrants.types.utils import is_signed

from ._kernel_types import KernelBatchedArgType
from ._template_mapper import TemplateMapper

MAX_ARG_NUM = 512

# Define proxies for fast lookup
_FLOAT, _INT, _UINT, _QD_ARRAY, _QD_ARRAY_WITH_GRAD = KernelBatchedArgType
_ARG_EMPTY = inspect.Parameter.empty
_arch_cuda = _qd_core.Arch.cuda


class FuncBase:
    """
    Base class for Kernels and Funcs
    """

    def __init__(
        self, func, func_id: int, is_kernel: bool, is_classkernel: bool, is_classfunc: bool, is_real_function: bool
    ) -> None:
        self.func = func
        self.func_id = func_id
        self.is_kernel = is_kernel
        self.is_real_function = is_real_function
        # TODO: merge is_classkernel and is_classfunc?
        self.is_classkernel = is_classkernel
        self.is_classfunc = is_classfunc
        self.arg_metas: list[ArgMetadata] = []
        self.arg_metas_expanded: list[ArgMetadata] = []
        self.orig_arguments: list[ArgMetadata] = []
        self.return_type = None

        self.check_parameter_annotations()

        self.mapper = TemplateMapper(self.arg_metas, self.template_slot_locations)

    def check_parameter_annotations(self) -> None:
        """
        Look at annotations of function parameters, and store into self.arg_metas
        and self.orig_arguments (both are identical after this call)
        - they just contain the original parameter annotations after this call, unexpanded
        - this function mostly just does checking

        Note: NOT in the hot path. Just run once, on function registration
        """
        sig = inspect.signature(self.func)
        if hasattr(self.func, "__wrapped__"):
            raise_exception(
                QuadrantsSyntaxError,
                msg="Cant put kernel in front of other annotations",
                err_code="KERNEL_ANNOTATION_ORDER",
            )
        if sig.return_annotation not in {inspect._empty, None}:
            self.return_type = sig.return_annotation
            if (
                isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))  # type: ignore
                and self.return_type.__origin__ is tuple
            ):
                self.return_type = self.return_type.__args__
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for return_type in self.return_type:
                if return_type is Ellipsis:
                    raise QuadrantsSyntaxError("Ellipsis is not supported in return type annotations")
        params = dict(sig.parameters)
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise QuadrantsSyntaxError(
                    "Quadrants kernels do not support variable keyword parameters (i.e., **kwargs)"
                )
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise QuadrantsSyntaxError(
                    "Quadrants kernels do not support variable positional parameters (i.e., *args)"
                )
            if self.is_kernel and param.default is not inspect.Parameter.empty:
                raise QuadrantsSyntaxError("Quadrants kernels do not support default values for arguments")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise QuadrantsSyntaxError("Quadrants kernels do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise QuadrantsSyntaxError('Quadrants kernels only support "positional or keyword" parameters')
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                if i == 0 and (self.is_classkernel or self.is_classfunc):  # The |self| parameter
                    annotation = template()
                elif self.is_kernel or self.is_real_function:
                    raise QuadrantsSyntaxError("Quadrants kernels parameters must be type annotated")
            else:
                annotation_type = type(annotation)
                if annotation_type is ndarray_type.NdarrayType:
                    pass
                elif annotation is ndarray_type.NdarrayType:
                    # convert from qd.types.NDArray into qd.types.NDArray()
                    annotation = annotation()
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif issubclass(annotation_type, MatrixType):
                    pass
                elif not self.is_kernel and annotation_type is primitive_types.RefType:
                    pass
                elif annotation_type is StructType:
                    pass
                elif annotation_type is template or annotation is template:
                    pass
                elif annotation_type is type and is_dataclass(annotation):
                    pass
                elif self.is_kernel and isinstance(annotation, sparse_matrix_builder):
                    pass
                else:
                    raise QuadrantsSyntaxError(f"Invalid type annotation (argument {i}) of Taichi kernel: {annotation}")
            self.arg_metas.append(ArgMetadata(annotation, param.name, param.default))
            self.orig_arguments.append(ArgMetadata(annotation, param.name, param.default))

        self.template_slot_locations: list[int] = []
        for i, arg in enumerate(self.arg_metas):
            if arg.annotation == template or isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)

    def _populate_global_vars_for_templates(
        self,
        template_slot_locations: list[int],
        argument_metas: list[ArgMetadata],
        global_vars: dict[str, Any],
        fn: Callable,
        py_args: tuple[Any, ...],
    ):
        """
        Inject template parameters into globals

        Globals are being abused to store the python objects associated
        with templates. We continue this approach, and in addition this function
        handles injecting expanded python variables from dataclasses.
        """
        for i in template_slot_locations:
            template_var_name = argument_metas[i].name
            global_vars[template_var_name] = py_args[i]
        parameters = inspect.signature(fn).parameters
        for i, (parameter_name, parameter) in enumerate(parameters.items()):
            if is_dataclass(parameter.annotation):
                _kernel_impl_dataclass.populate_global_vars_from_dataclass(
                    parameter_name,
                    parameter.annotation,
                    py_args[i],
                    global_vars=global_vars,
                )

    def get_tree_and_ctx(
        self,
        py_args: tuple[Any, ...],
        template_slot_locations=(),
        is_kernel: bool = True,
        arg_features=None,
        ast_builder: "ASTBuilder | None" = None,
        is_real_function: bool = False,
        current_kernel: "Kernel | None" = None,  # has value when called from Kernel.materialize
        pruning: "Pruning | None" = None,  # has value when called from Kernel.materialize
        currently_compiling_materialize_key=None,  # has value when called from Kernel.materialize
        pass_idx: int | None = None,  # has value when called from Kernel.materialize
    ) -> tuple[ast.Module, ASTTransformerFuncContext]:
        function_source_info, src = get_source_info_and_src(self.func)
        src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
        tree = ast.parse(textwrap.dedent("\n".join(src)))

        func_body = tree.body[0]
        func_body.decorator_list = []  # type: ignore , kick that can down the road...

        runtime = impl.get_runtime()

        if current_kernel is not None:  # Kernel
            assert pruning is not None
            assert pass_idx is not None
            current_kernel.kernel_function_info = function_source_info
            global_context = ASTTransformerGlobalContext(
                pass_idx=pass_idx,
                current_kernel=current_kernel,
                pruning=pruning,
                currently_compiling_materialize_key=currently_compiling_materialize_key,
            )
        else:  # Func
            global_context = runtime._current_global_context
            assert global_context is not None
            current_kernel = global_context.current_kernel

        assert current_kernel is not None
        assert global_context is not None
        current_kernel.visited_functions.add(function_source_info)

        autodiff_mode = current_kernel.autodiff_mode

        quadrants_callable = current_kernel.quadrants_callable
        is_pure = quadrants_callable is not None and quadrants_callable.is_pure
        global_vars = self._get_global_vars(self.func)

        template_vars = {}
        if is_kernel or is_real_function:
            self._populate_global_vars_for_templates(
                template_slot_locations=self.template_slot_locations,
                argument_metas=self.arg_metas,
                global_vars=template_vars,
                fn=self.func,
                py_args=py_args,
            )

        raise_on_templated_floats = impl.current_cfg().raise_on_templated_floats

        ctx = ASTTransformerFuncContext(
            global_context=global_context,
            template_slot_locations=template_slot_locations,
            is_kernel=is_kernel,
            is_pure=is_pure,
            func=self,
            arg_features=arg_features,
            global_vars=global_vars,
            template_vars=template_vars,
            py_args=py_args,
            src=src,
            start_lineno=function_source_info.start_lineno,
            end_lineno=function_source_info.end_lineno,
            file=function_source_info.filepath,
            ast_builder=ast_builder,
            is_real_function=is_real_function,
            autodiff_mode=autodiff_mode,
            raise_on_templated_floats=raise_on_templated_floats,
        )
        return tree, ctx

    def fuse_args(
        self,
        global_context: ASTTransformerGlobalContext | None,
        is_pyfunc: bool,
        is_func: bool,
        py_args: tuple[Any, ...],
        kwargs,
    ) -> tuple[Any, ...]:
        """
        - for functions, expand dataclass arg_metas
        - fuse incoming args and kwargs into a single list of args

        The output of this function is arguments which are:
        - a sequence (not a dict)
        - fused args + kwargs
        - in the exact same order as self.arg_metas_expanded
            - and with the exact same number of elements

        Quadrants doesn't allow defaults, so we don't need to consider default options here,
        but if we did, we'd still have output exactly matching the order and size of
        self.arg_metas_expanded, just with some of the values coming from defaults.

        for kernels, global_context is None. We aren't compiling yet. This is only called once
        per launch, but it is called every launch, even if we already compiled.

        For funcs, this is only called during compilation, once per pass.

        For kernels, the args are NOT expanded at this point, and pruning changes nothing.

        For funcs, the args are expanded at the start of this function
        - first pass, no pruning
        - second pass - with enforcing on - the expanded parameters are pruned
        """
        if is_func and not is_pyfunc:
            assert global_context is not None
            current_kernel = global_context.current_kernel
            assert current_kernel is not None
            pruning = global_context.pruning
            used_by_dataclass_parameters_enforcing = None
            if pruning.enforcing:
                used_by_dataclass_parameters_enforcing = global_context.pruning.used_vars_by_func_id[self.func_id]
            self.arg_metas_expanded = _kernel_impl_dataclass.expand_func_arguments(
                used_by_dataclass_parameters_enforcing,
                self.arg_metas,
            )
        else:
            self.arg_metas_expanded = list(self.arg_metas)

        num_args = len(py_args)
        num_arg_metas = len(self.arg_metas_expanded)
        if num_args > num_arg_metas:
            arg_str = ", ".join(map(str, py_args))
            expected_str = ", ".join(f"{arg.name} : {arg.annotation}" for arg in self.arg_metas_expanded)
            msg_l = []
            msg_l.append(f"Too many arguments. Expected ({expected_str}), got ({arg_str}).")
            for i in range(num_args):
                if i < num_arg_metas:
                    msg_l.append(f" - {i} arg meta: {self.arg_metas_expanded[i].name} arg type: {type(py_args[i])}")
                else:
                    msg_l.append(f" - {i} arg meta: <out of arg metas> arg type: {type(py_args[i])}")
            msg_l.append(f"In function: {self.func}")
            raise QuadrantsSyntaxError("\n".join(msg_l))

        # Early return without further processing if possible for efficiency. This is by far the most common scenario.
        if not (kwargs or num_arg_metas > num_args):
            return py_args

        fused_py_args: list[Any] = [*py_args, *[arg_meta.default for arg_meta in self.arg_metas_expanded[num_args:]]]
        errors_l: list[str] = []
        if kwargs:
            num_invalid_kwargs_args = len(kwargs)
            for i in range(num_args, num_arg_metas):
                arg_meta = self.arg_metas_expanded[i]
                py_arg = kwargs.get(arg_meta.name, _ARG_EMPTY)
                if py_arg is not _ARG_EMPTY:
                    fused_py_args[i] = py_arg
                    num_invalid_kwargs_args -= 1
                elif fused_py_args[i] is _ARG_EMPTY:
                    errors_l.append(f"Missing argument '{arg_meta.name}'.")
                    continue
            if num_invalid_kwargs_args:
                for key, py_arg in kwargs.items():
                    for i, arg_meta in enumerate(self.arg_metas_expanded):
                        if key == arg_meta.name:
                            if i < num_args:
                                errors_l.append(f"Multiple values for argument '{key}'.")
                            break
                    else:
                        errors_l.append(f"Unexpected argument '{key}'.")
        else:
            for i in range(num_args, num_arg_metas):
                if fused_py_args[i] is _ARG_EMPTY:
                    arg_meta = self.arg_metas_expanded[i]
                    errors_l.append(f"Missing argument '{arg_meta.name}'.")
                    continue

        if errors_l:
            if len(errors_l) == 1:
                raise QuadrantsSyntaxError(errors_l[0])
            else:
                primary_, secondaries_ = errors_l[0], errors_l[1:]
                raise QuadrantsSyntaxError(
                    f"Primary exception: {primary_}\n\nAdditional diagnostic/dev info:\n" "\n".join(secondaries_)
                )

        return tuple(fused_py_args)

    def _get_global_vars(self, _func: Callable) -> dict[str, Any]:
        # Discussions: https://github.com/taichi-dev/quadrants/issues/282
        global_vars = _func.__globals__.copy()
        freevar_names = _func.__code__.co_freevars
        closure = _func.__closure__
        if closure:
            freevar_values = list(map(lambda x: x.cell_contents, closure))
            for name, value in zip(freevar_names, freevar_values):
                global_vars[name] = value

        return global_vars

    @staticmethod
    def cast_float(x: float | np.floating | np.integer | int) -> float:
        if not isinstance(x, (int, float, np.integer, np.floating)):
            raise ValueError(f"Invalid argument type '{type(x)}")
        return float(x)

    @staticmethod
    def cast_int(x: int | np.integer) -> int:
        if not isinstance(x, (int, np.integer)):
            raise ValueError(f"Invalid argument type '{type(x)}")
        return int(x)

    @staticmethod
    def _recursive_set_args(
        used_py_dataclass_parameters: set[str],
        py_dataclass_basename: str,
        launch_ctx: KernelLaunchContext,
        launch_ctx_buffer: DefaultDict[KernelBatchedArgType, list[tuple]],
        needed_arg_type: Type,
        provided_arg_type: Type,
        v: Any,
        index: int,
        actual_argument_slot: int,
        callbacks: list[Callable[[], Any]],
    ) -> tuple[int, bool]:
        """
        This function processes all the input python-side arguments of a given kernel so as to add them to the current
        launch context of a given kernel. Apart from a few exceptions, no call is made to the launch context directly,
        but rather accumulated in a buffer to be called all at once in a later stage. This avoid accumulating pybind11
        overhead for every single argument.

        Returns the number of underlying kernel args being set for a given Python arg, and whether the launch context
        buffer can be cached (see 'launch_kernel' for details).

        Note that templates don't set kernel args, and a single scalar, an external array (numpy or torch) or a taichi
        ndarray all set 1 kernel arg. Similarlty, a struct of N ndarrays would set N kernel args.
        """
        if actual_argument_slot >= MAX_ARG_NUM:
            raise QuadrantsRuntimeError(
                f"The number of elements in kernel arguments is too big! Do not exceed {MAX_ARG_NUM} on "
                f"{_qd_core.arch_name(impl.current_cfg().arch)} backend."
            )
        actual_argument_slot += 1

        needed_arg_type_id = id(needed_arg_type)
        needed_arg_basetype = type(needed_arg_type)

        # Note: do not use sth like "needed == f32". That would be slow.
        if needed_arg_type_id in primitive_types.real_type_ids:
            if not isinstance(v, (float, int, np.floating, np.integer)):
                raise QuadrantsRuntimeTypeError.get((index,), needed_arg_type.to_string(), provided_arg_type)
            launch_ctx_buffer[_FLOAT].append((index, float(v)))
            return 1, False
        if needed_arg_type_id in primitive_types.integer_type_ids:
            if not isinstance(v, (int, np.integer)):
                raise QuadrantsRuntimeTypeError.get((index,), needed_arg_type.to_string(), provided_arg_type)
            if is_signed(cook_dtype(needed_arg_type)):
                launch_ctx_buffer[_INT].append((index, int(v)))
            else:
                launch_ctx_buffer[_UINT].append((index, int(v)))
            return 1, False
        needed_arg_fields = getattr(needed_arg_type, _FIELDS, None)
        if needed_arg_fields is not None:
            if provided_arg_type is not needed_arg_type:
                raise QuadrantsRuntimeError("needed", needed_arg_type, "!= provided", provided_arg_type)
            # A dataclass must be frozen to be compatible with caching
            is_launch_ctx_cacheable = needed_arg_type.__hash__ is not None
            idx = 0
            for field in needed_arg_fields.values():
                if field._field_type is not _FIELD:
                    continue
                field_name = field.name
                field_full_name = create_flat_name(py_dataclass_basename, field_name)
                if field_full_name not in used_py_dataclass_parameters:
                    continue
                # Storing attribute in a temporary to avoid repeated attribute lookup (~20ns penalty)
                field_type = field.type
                assert not isinstance(field_type, str)
                field_value = getattr(v, field_name)
                num_args_, is_launch_ctx_cacheable_ = FuncBase._recursive_set_args(
                    used_py_dataclass_parameters,
                    field_full_name,
                    launch_ctx,
                    launch_ctx_buffer,
                    field_type,
                    field_type,
                    field_value,
                    index + idx,
                    actual_argument_slot,
                    callbacks,
                )
                idx += num_args_
                is_launch_ctx_cacheable &= is_launch_ctx_cacheable_
            return idx, is_launch_ctx_cacheable
        if needed_arg_basetype is ndarray_type.NdarrayType and isinstance(v, Ndarray):
            v_primal = v.arr
            v_grad = v.grad.arr if v.grad else None
            if v_grad is None:
                launch_ctx_buffer[_QD_ARRAY].append((index, v_primal))
            else:
                launch_ctx_buffer[_QD_ARRAY_WITH_GRAD].append((index, v_primal, v_grad))
            return 1, True
        if needed_arg_basetype is ndarray_type.NdarrayType:
            # v is things like torch Tensor and numpy array
            # Not adding type for this, since adds additional dependencies
            #
            # Element shapes are already specialized in Quadrants codegen.
            # The shape information for element dims are no longer needed.
            # Therefore we strip the element shapes from the shape vector,
            # so that it only holds "real" array shapes.
            is_soa = needed_arg_type.layout == Layout.SOA
            array_shape = v.shape
            if math.prod(array_shape) > np.iinfo(np.int32).max:
                warnings.warn("Ndarray index might be out of int32 boundary but int64 indexing is not supported yet.")
            needed_arg_dtype = needed_arg_type.dtype
            if needed_arg_dtype is None or id(needed_arg_dtype) in primitive_types.type_ids:
                element_dim = 0
            else:
                element_dim = needed_arg_dtype.ndim
                array_shape = v.shape[element_dim:] if is_soa else v.shape[:-element_dim]
            if isinstance(v, np.ndarray):
                # Check ndarray flags is expensive (~250ns), so it is important to order branches according to hit stats
                if v.flags.c_contiguous:
                    pass
                elif v.flags.f_contiguous:
                    # TODO: A better way that avoids copying is saving strides info.
                    v_contiguous = np.ascontiguousarray(v)
                    v, v_orig_np = v_contiguous, v
                    callbacks.append(partial(np.copyto, v_orig_np, v))
                else:
                    raise ValueError(
                        "Non contiguous numpy arrays are not supported, please call np.ascontiguousarray(arr) "
                        "before passing it into quadrants kernel."
                    )
                launch_ctx.set_arg_external_array_with_shape(index, int(v.ctypes.data), v.nbytes, array_shape, 0)
            elif has_pytorch():
                import torch  # pylint: disable=C0415

                if isinstance(v, torch.Tensor):
                    if not v.is_contiguous():
                        raise ValueError(
                            "Non contiguous tensors are not supported, please call tensor.contiguous() before "
                            "passing it into quadrants kernel."
                        )
                    quadrants_arch = impl.current_cfg().arch

                    # FIXME: only allocate when launching grad kernel
                    if v.requires_grad and v.grad is None:
                        v.grad = torch.zeros_like(v)

                    if v.requires_grad:
                        if not isinstance(v.grad, torch.Tensor):
                            raise ValueError(
                                f"Expecting torch.Tensor for gradient tensor, but getting {v.grad.__class__.__name__} instead"
                            )
                        if not v.grad.is_contiguous():
                            raise ValueError(
                                "Non contiguous gradient tensors are not supported, please call tensor.grad.contiguous() "
                                "before passing it into quadrants kernel."
                            )

                    grad = v.grad
                    if (v.device.type != "cpu") and not (v.device.type == "cuda" and quadrants_arch == _arch_cuda):
                        # For a torch tensor to be passed as as input argument (in and/or out) of a taichi kernel, its
                        # memory must be hosted either on CPU, or on CUDA if and only if Quadrants is using CUDA backend.
                        # We just replace it with a CPU tensor and by the end of kernel execution we'll use the callback
                        # to copy the values back to the original tensor.
                        v_cpu = v.to(device="cpu")
                        v, v_orig_tc = v_cpu, v
                        callbacks.append(partial(v_orig_tc.data.copy_, v))
                        if grad is not None:
                            grad_cpu = grad.to(device="cpu")
                            grad, grad_orig = grad_cpu, grad
                            callbacks.append(partial(grad_orig.data.copy_, grad))

                    launch_ctx.set_arg_external_array_with_shape(
                        index,
                        int(v.data_ptr()),
                        v.element_size() * v.nelement(),
                        array_shape,
                        int(grad.data_ptr()) if grad is not None else 0,
                    )
                else:
                    raise QuadrantsRuntimeTypeError(
                        f"Argument of type {type(v)} cannot be converted into required type {needed_arg_type}"
                    )
            else:
                raise QuadrantsRuntimeTypeError(
                    f"Argument {needed_arg_type} cannot be converted into required type {v}"
                )
            return 1, False
        if issubclass(needed_arg_basetype, MatrixType):
            cast_func: Callable[[Any], int | float] | None = None
            if needed_arg_type.dtype in primitive_types.real_types:
                cast_func = FuncBase.cast_float
            elif needed_arg_type.dtype in primitive_types.integer_types:
                cast_func = FuncBase.cast_int
            else:
                raise ValueError(f"Matrix dtype {needed_arg_type.dtype} is not integer type or real type.")

            try:
                if needed_arg_type.ndim == 2:
                    v = [cast_func(v[i, j]) for i in range(needed_arg_type.n) for j in range(needed_arg_type.m)]
                else:
                    v = [cast_func(v[i]) for i in range(needed_arg_type.n)]
            except ValueError as e:
                raise QuadrantsRuntimeTypeError(
                    f"Argument cannot be converted into required type {needed_arg_type.dtype}"
                ) from e

            v = needed_arg_type(*v)
            needed_arg_type.set_kernel_struct_args(v, launch_ctx, (index,))
            return 1, False
        if needed_arg_basetype is StructType:
            # Unclear how to make the following pass typing checks StructType implements __instancecheck__,
            # which should be a classmethod, but is currently an instance method.
            # TODO: look into this more deeply at some point
            if not isinstance(v, needed_arg_type):  # type: ignore
                raise QuadrantsRuntimeTypeError(
                    f"Argument {provided_arg_type} cannot be converted into required type {needed_arg_type}"
                )
            needed_arg_type.set_kernel_struct_args(v, launch_ctx, (index,))
            return 1, False
        if needed_arg_type is template or needed_arg_basetype is template:
            return 0, True
        if needed_arg_basetype is sparse_matrix_builder:
            # Pass only the base pointer of the qd.types.sparse_matrix_builder() argument
            launch_ctx_buffer[_UINT].append((index, v._get_ndarray_addr()))
            return 1, True
        raise ValueError(f"Argument type mismatch. Expecting {needed_arg_type}, got {type(v)}.")
