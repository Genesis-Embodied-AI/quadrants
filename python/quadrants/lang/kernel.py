import ast
import json
import os
import pathlib
import time
from collections import defaultdict

# Must import 'partial' directly instead of the entire module to avoid attribute lookup overhead.
from functools import partial
from typing import Any, Callable

# Must import 'ReferenceType' directly instead of the entire module to avoid attribute lookup overhead.
from weakref import ReferenceType

from quadrants import _logging
from quadrants._lib.core.quadrants_python import (
    ASTBuilder,
    CompiledKernelData,
    CompileResult,
    KernelCxx,
    KernelLaunchContext,
)
from quadrants.lang import _kernel_impl_dataclass, impl, runtime_ops
from quadrants.lang._fast_caching import src_hasher
from quadrants.lang._wrap_inspect import FunctionSourceInfo, get_source_info_and_src
from quadrants.lang.ast import (
    KernelSimplicityASTChecker,
    transform_tree,
)
from quadrants.lang.ast.ast_transformer_utils import (
    ASTTransformerFuncContext,
    ReturnStatus,
)
from quadrants.lang.exception import (
    QuadrantsRuntimeTypeError,
    QuadrantsSyntaxError,
    handle_exception_from_cpp,
)
from quadrants.lang.impl import Program
from quadrants.lang.shell import _shell_pop_print
from quadrants.lang.util import cook_dtype
from quadrants.types import (
    primitive_types,
    template,
)
from quadrants.types.compound_types import CompoundType
from quadrants.types.enums import AutodiffMode
from quadrants.types.utils import is_signed

from ._func_base import FuncBase
from ._kernel_types import (
    ArgsHash,
    CompiledKernelKeyType,
    FeLlCacheObservations,
    KernelBatchedArgType,
    LaunchObservations,
    LaunchStats,
    SrcLlCacheObservations,
)
from ._pruning import Pruning
from ._quadrants_callable import QuadrantsCallable

# Define proxies for fast lookup
_NONE, _VALIDATION = AutodiffMode.NONE, AutodiffMode.VALIDATION
_FLOAT, _INT, _UINT, _QD_ARRAY, _QD_ARRAY_WITH_GRAD = KernelBatchedArgType


class LaunchContextBufferCache:
    # Here, we are tracking whether a launch context buffer can be cached.
    # The point of caching the launch context buffer is allowing skipping recursive processing of all the input
    # arguments one-by-one, which is adding a significant overhead, without changing anything in regards of the
    # function calls to the launch context that must be made for a given kernel.
    # You can understand this as resolving the static part of the entire control flow of '_recursive_set_args'
    # for a given set of arguments, which is (mostly surely uniquely) characterized by its hash, gathering all
    # the instructions that cannot be evaluated statically and packing them in a buffer without evaluating them at
    # this point. This buffer is then cached once and for all and evaluated every time the exact same set of input
    # argument is passed. This means that, ultimately, it will result in the exact same function calls with or
    # without caching. In this particular case, the function calls corresponds to adding arguments to the current
    # context for this kernel call.
    # A launch context buffer is considered cache-friendly if and only if no direct call to the launch context
    # where made preemptively during the recursive processing of the arguments, all of parameters of the arguments are
    # pointers, the address of these pointers cannot change, and the set of parameters is fixed.
    # The lifetime of a cache entry is bound to the lifetime of any of its input arguments: the first being garbage
    # collected will invalidate the entire entry. Moreover, the entire cache registry is bound to the lifetime of
    # the taichi prog itself, which means that calling `qd.reset()` will automatically clear the cache. Note that
    # the cache stores wear references to pointers, so it does not hold alife any allocated memory.
    def __init__(self) -> None:
        # Keep track of taichi runtime to automatically clear cache if destroyed
        self._prog_weakref: ReferenceType[Program] | None = None

        # The cache key corresponds to the hash of the (packed) python-side input arguments of the kernel.
        # * '_launch_ctx_cache' is storing a backup of the launch context BEFORE ever calling the kernel.
        # * '_launch_ctx_cache_tracker' is used for bounding the lifetime of a cache entry to its corresponding set of
        #   input arguments. Internally, this is done by wrapping all Taichi ndarrays as weak reference.
        # * '_prog_weakref'is used for bounding the lifetime of the entire cache to the Taichi programm managing all
        #   the launch context being stored in cache.
        # See 'launch_kernel' for details regarding the intended use of caching.
        self._launch_ctx_cache: dict["ArgsHash", KernelLaunchContext] = {}
        self._launch_ctx_cache_tracker: dict["ArgsHash", list[ReferenceType | None]] = {}

    @staticmethod
    def _destroy_callback(kernel_ref: ReferenceType["LaunchContextBufferCache"], ref: ReferenceType):
        maybe_kernel = kernel_ref()
        if maybe_kernel is not None:
            maybe_kernel._launch_ctx_cache.clear()
            maybe_kernel._launch_ctx_cache_tracker.clear()
            maybe_kernel._prog_weakref = None

    def cache(
        self,
        t_kernel,
        args_hash: "ArgsHash",
        launch_ctx: KernelLaunchContext,
        launch_ctx_buffer: dict[KernelBatchedArgType, list[tuple]],
    ) -> None:
        # TODO: It some rare occurrences, arguments can be cached yet not hashable. Ignoring for now...
        cached_launch_ctx = t_kernel.make_launch_context()
        cached_launch_ctx.copy(launch_ctx)
        self._launch_ctx_cache[args_hash] = cached_launch_ctx

        # Note that the clearing callback will only be called once despite being registered for each tracked
        # objects, because all the weakrefs get deallocated right away, and their respective callback vanishes
        # with them, without even getting a chance to get called. This means that registering the clearing
        # callback systematically does not incur any cumulative runtime penalty yet ensures full memory safety.
        # Note that it is important to prepend the cache tracker with 'None' to avoid misclassifying no argument
        # with expired cache entry caused by deallocated argument.
        launch_ctx_cache_tracker_: list[ReferenceType | None] = [None]
        clear_callback = lambda ref: launch_ctx_cache_tracker_.clear()
        if launch_ctx_args := launch_ctx_buffer.get(_QD_ARRAY):
            _, arrs = zip(*launch_ctx_args)
            launch_ctx_cache_tracker_ += [ReferenceType(arr, clear_callback) for arr in arrs]
        if launch_ctx_args := launch_ctx_buffer.get(_QD_ARRAY_WITH_GRAD):
            _, arrs, arrs_grad = zip(*launch_ctx_args)
            launch_ctx_cache_tracker_ += [ReferenceType(arr, clear_callback) for arr in arrs]
            launch_ctx_cache_tracker_ += [ReferenceType(arr_grad, clear_callback) for arr_grad in arrs_grad]
        self._launch_ctx_cache_tracker[args_hash] = launch_ctx_cache_tracker_

    def populate_launch_ctx_from_cache(self, args_hash: "ArgsHash", launch_ctx: KernelLaunchContext) -> bool:
        if self._prog_weakref is None:
            prog = impl.get_runtime().prog
            assert prog is not None
            self._prog_weakref = ReferenceType(
                prog, partial(LaunchContextBufferCache._destroy_callback, ReferenceType(self))
            )
        else:
            # Since we already store a weak reference to taichi program, it is much faster to use it rather than
            # paying the overhead of calling pybind11 functions (~200ns vs 5ns).
            prog = self._prog_weakref()
        assert prog is not None

        launch_ctx_cache_tracker: list[ReferenceType | None] | None = None
        try:
            launch_ctx_cache_tracker = self._launch_ctx_cache_tracker[args_hash]
        except KeyError:
            pass
        if not launch_ctx_cache_tracker:  # Empty or none
            return False

        assert args_hash is not None
        launch_ctx.copy(self._launch_ctx_cache[args_hash])
        return True


class ASTGenerator:
    def __init__(
        self,
        ctx: ASTTransformerFuncContext,
        kernel_name: str,
        current_kernel: "Kernel",
        only_parse_function_def: bool,
        tree: ast.Module,
        dump_ast: bool,
    ) -> None:
        self.runtime = impl.get_runtime()
        self.current_kernel = current_kernel
        self.ctx = ctx
        self.kernel_name = kernel_name
        self.tree = tree
        self.only_parse_function_def = only_parse_function_def
        self.dump_ast = dump_ast

    """
    only_parse_function_def will be set when running from fast cache.
    """

    # Do not change the name of 'quadrants_ast_generator'
    # The warning system needs this identifier to remove unnecessary messages
    def __call__(self, kernel_cxx: KernelCxx):
        # nonlocal tree, used_py_dataclass_parameters
        if self.runtime.inside_kernel:
            raise QuadrantsSyntaxError(
                "Kernels cannot call other kernels. I.e., nested kernels are not allowed. "
                "Please check if you have direct/indirect invocation of kernels within kernels. "
                "Note that some methods provided by the Quadrants standard library may invoke kernels, "
                "and please move their invocations to Python-scope."
            )
        self.current_kernel.kernel_cpp = kernel_cxx
        ctx = self.ctx
        pruning = ctx.global_context.pruning
        self.runtime.inside_kernel = True
        assert self.runtime._compiling_callable is None
        self.runtime._compiling_callable = kernel_cxx
        try:
            ctx.ast_builder = kernel_cxx.ast_builder()
            if self.dump_ast:
                self._dump_ast()
            if not pruning.enforcing:
                struct_locals = _kernel_impl_dataclass.extract_struct_locals_from_context(ctx)
            else:
                struct_locals = pruning.used_vars_by_func_id[ctx.func.func_id]
            # struct locals are the expanded py dataclass fields that we will write to
            # local variables, and will then be available to use in build_Call, later.
            tree = _kernel_impl_dataclass.unpack_ast_struct_expressions(self.tree, struct_locals=struct_locals)
            ctx.only_parse_function_def = self.only_parse_function_def
            transform_tree(tree, ctx)
            if not ctx.is_real_function and not ctx.only_parse_function_def:
                if self.current_kernel.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                    raise QuadrantsSyntaxError("Kernel has a return type but does not have a return statement")
        finally:
            self.current_kernel.runtime.inside_kernel = False
            self.runtime._current_global_context = None
            self.current_kernel.runtime._compiling_callable = None

    def _dump_ast(self) -> None:
        target_dir = pathlib.Path("/tmp/ast")
        target_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        ast_str = ast.dump(self.tree, indent=2)
        output_file = target_dir / f"{self.kernel_name}_ast_.txt"
        output_file.write_text(ast_str)
        elapsed_txt = time.time() - start

        start = time.time()
        json_str = json.dumps(self._ast_to_dict(self.tree), indent=2)
        output_file = target_dir / f"{self.kernel_name}_ast.json"
        output_file.write_text(json_str)
        elapsed_json = time.time() - start

        output_file = target_dir / f"{self.kernel_name}_gen_time.json"
        output_file.write_text(json.dumps({"elapsed_txt": elapsed_txt, "elapsed_json": elapsed_json}, indent=2))

    def _ast_to_dict(self, node: ast.AST | list | primitive_types._python_primitive_types):
        if isinstance(node, ast.AST):
            fields = {k: self._ast_to_dict(v) for k, v in ast.iter_fields(node)}
            return {
                "type": node.__class__.__name__,
                "fields": fields,
                "lineno": getattr(node, "lineno", None),
                "col_offset": getattr(node, "col_offset", None),
            }
        if isinstance(node, list):
            return [self._ast_to_dict(x) for x in node]
        return node  # Basic types (str, int, None, etc.)


class Kernel(FuncBase):
    counter = 0

    def __init__(self, _func: Callable, autodiff_mode: AutodiffMode, _is_classkernel=False) -> None:
        super().__init__(
            func=_func,
            is_classfunc=False,
            is_kernel=True,
            is_classkernel=_is_classkernel,
            is_real_function=False,
            func_id=Pruning.KERNEL_FUNC_ID,
        )
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        assert autodiff_mode in (
            AutodiffMode.NONE,
            AutodiffMode.VALIDATION,
            AutodiffMode.FORWARD,
            AutodiffMode.REVERSE,
        )
        self.autodiff_mode = autodiff_mode
        self.grad: "Kernel | None" = None
        impl.get_runtime().kernels.append(self)
        self.reset()
        self.kernel_cpp: None | KernelCxx = None
        # A materialized kernel is a KernelCxx object which may or may not have
        # been compiled. It generally has been converted at least as far as AST
        # and front-end IR, but not necessarily any further.
        self.materialized_kernels: dict[CompiledKernelKeyType, KernelCxx] = {}
        self.has_print = False
        self.quadrants_callable: QuadrantsCallable | None = None
        self.visited_functions: set[FunctionSourceInfo] = set()
        self.kernel_function_info: FunctionSourceInfo | None = None
        self.compiled_kernel_data_by_key: dict[CompiledKernelKeyType, CompiledKernelData] = {}
        self._last_compiled_kernel_data: CompiledKernelData | None = None  # for dev/debug
        self._last_launch_key = None  # for dev/debug

        # next two parameters are ONLY used at kernel launch time,
        # NOT for compilation. (for compilation, global_context.pruning is used).
        # These parameters here are used to filter args passed into the already-compiled kernel.
        # used_py_dataclass_parameters_by_key_enforcing will also be serialized with fast cache.
        self.used_py_dataclass_parameters_by_key_enforcing: dict[CompiledKernelKeyType, set[str]] = {}

        self.src_ll_cache_observations: SrcLlCacheObservations = SrcLlCacheObservations()
        self.fe_ll_cache_observations: FeLlCacheObservations = FeLlCacheObservations()
        self.launch_observations = LaunchObservations()

        self.launch_context_buffer_cache = LaunchContextBufferCache()

    def ast_builder(self) -> ASTBuilder:
        assert self.kernel_cpp is not None
        return self.kernel_cpp.ast_builder()

    def reset(self) -> None:
        self.runtime = impl.get_runtime()
        self.materialized_kernels = {}
        self.compiled_kernel_data_by_key = {}
        self._last_compiled_kernel_data = None
        self.src_ll_cache_observations = SrcLlCacheObservations()
        self.fe_ll_cache_observations = FeLlCacheObservations()

    def _try_load_fastcache(self, args: tuple[Any, ...], key: "CompiledKernelKeyType") -> set[str] | None:
        frontend_cache_key: str | None = None
        if self.runtime.src_ll_cache and self.quadrants_callable and self.quadrants_callable.is_pure:
            kernel_source_info, _src = get_source_info_and_src(self.func)
            self.fast_checksum = src_hasher.create_cache_key(
                self.raise_on_templated_floats, kernel_source_info, args, self.arg_metas
            )
            used_py_dataclass_parameters = None
            if self.fast_checksum:
                self.src_ll_cache_observations.cache_key_generated = True
                used_py_dataclass_parameters, frontend_cache_key = src_hasher.load(self.fast_checksum)
            if used_py_dataclass_parameters is not None and frontend_cache_key is not None:
                self.src_ll_cache_observations.cache_validated = True
                prog = impl.get_runtime().prog
                assert self.fast_checksum is not None
                self.compiled_kernel_data_by_key[key] = prog.load_fast_cache(
                    frontend_cache_key,
                    self.func.__name__,
                    prog.config(),
                    prog.get_device_caps(),
                )
                if self.compiled_kernel_data_by_key[key]:
                    self.src_ll_cache_observations.cache_loaded = True
                    self.used_py_dataclass_parameters_by_key_enforcing[key] = used_py_dataclass_parameters
                    return used_py_dataclass_parameters

        elif self.quadrants_callable and not self.quadrants_callable.is_pure and self.runtime.print_non_pure:
            # The bit in caps should not be modified without updating corresponding test
            # freetext can be freely modified.
            # As for why we are using `print` rather than eg logger.info, it is because
            # this is only printed when qd.init(print_non_pure=..) is True. And it is
            # confusing to set that to True, and see nothing printed.
            print(f"[NOT_PURE] Debug information: not pure: {self.func.__name__}")
        return None

    def materialize(self, key: "CompiledKernelKeyType | None", py_args: tuple[Any, ...], arg_features=None):
        if key is None:
            key = (self.func, 0, self.autodiff_mode)
        self.fast_checksum = None
        if key in self.materialized_kernels:
            return

        with self.runtime.compilation_lock:
            if key in self.materialized_kernels:
                return

            self.runtime.materialize()
            used_py_dataclass_parameters = self._try_load_fastcache(py_args, key)
            kernel_name = f"{self.func.__name__}_c{self.kernel_counter}_{key[1]}"
            _logging.trace(f"Materializing kernel {kernel_name} in {self.autodiff_mode}...")

            pruning = Pruning(kernel_used_parameters=used_py_dataclass_parameters)
            range_begin = 0 if used_py_dataclass_parameters is None else 1
            runtime = impl.get_runtime()
            for _pass in range(range_begin, 2):
                if _pass >= 1:
                    pruning.enforce()
                tree, ctx = self.get_tree_and_ctx(
                    pass_idx=_pass,
                    py_args=py_args,
                    template_slot_locations=self.template_slot_locations,
                    arg_features=arg_features,
                    current_kernel=self,
                    pruning=pruning,
                    currently_compiling_materialize_key=key,
                )
                runtime._current_global_context = ctx.global_context

                if self.autodiff_mode != _NONE:
                    KernelSimplicityASTChecker(self.func).visit(tree)

                quadrants_ast_generator = ASTGenerator(
                    ctx=ctx,
                    kernel_name=kernel_name,
                    current_kernel=self,
                    only_parse_function_def=self.compiled_kernel_data_by_key.get(key) is not None,
                    tree=tree,
                    dump_ast=os.environ.get("QD_DUMP_AST", "") == "1" and _pass == 1,
                )
                quadrants_kernel = impl.get_runtime().prog.create_kernel(
                    quadrants_ast_generator, kernel_name, self.autodiff_mode
                )
                if _pass == 1:
                    assert key not in self.materialized_kernels
                    self.materialized_kernels[key] = quadrants_kernel
                else:
                    for used_parameters in pruning.used_vars_by_func_id.values():
                        new_used_parameters = set()
                        for param in used_parameters:
                            split_param = param.split("__qd_")
                            for i in range(len(split_param), 1, -1):
                                joined = "__qd_".join(split_param[:i])
                                if joined in new_used_parameters:
                                    break
                                new_used_parameters.add(joined)
                        used_parameters.clear()
                        used_parameters.update(new_used_parameters)
                    self.used_py_dataclass_parameters_by_key_enforcing[key] = pruning.used_vars_by_func_id[
                        Pruning.KERNEL_FUNC_ID
                    ]
                runtime._current_global_context = None

    def launch_kernel(self, key, t_kernel: KernelCxx, compiled_kernel_data: CompiledKernelData | None, *args) -> Any:
        assert len(args) == len(self.arg_metas), f"{len(self.arg_metas)} arguments needed but {len(args)} provided"

        callbacks: list[Callable[[], None]] = []
        launch_ctx = t_kernel.make_launch_context()
        # Special treatment for primitive types is unecessary and detrimental. See 'TemplateMapper.lookup' for details.
        args_hash: "ArgsHash" = (id(t_kernel), *[id(arg) for arg in args])
        if not self.launch_context_buffer_cache.populate_launch_ctx_from_cache(args_hash, launch_ctx):
            launch_ctx_buffer: dict[KernelBatchedArgType, list[tuple]] = defaultdict(list)
            actual_argument_slot = 0
            is_launch_ctx_cacheable = True
            template_num = 0
            i_out = 0
            for i_in, val in enumerate(args):
                needed_ = self.arg_metas[i_in].annotation
                if needed_ is template or type(needed_) is template:
                    template_num += 1
                    i_out += 1
                    continue
                num_args_, is_launch_ctx_cacheable_ = self._recursive_set_args(
                    self.used_py_dataclass_parameters_by_key_enforcing[key],
                    self.arg_metas[i_in].name,
                    launch_ctx,
                    launch_ctx_buffer,
                    needed_,
                    type(val),
                    val,
                    i_out - template_num,
                    actual_argument_slot,
                    callbacks,
                )
                i_out += num_args_
                is_launch_ctx_cacheable &= is_launch_ctx_cacheable_

            kernel_args_count_by_type = defaultdict(int)
            kernel_args_count_by_type.update(
                {key: len(launch_ctx_args) for key, launch_ctx_args in launch_ctx_buffer.items()}
            )
            self.launch_stats = LaunchStats(kernel_args_count_by_type=kernel_args_count_by_type)

            # All arguments to context in batches to mitigate overhead of calling Python bindings repeatedly.
            # This is essential because calling any pybind11 function is adding ~180ns penalty no matter what.
            # Note that we are allowed to do this because Quadrants Launch Kernel context is storing the input
            # arguments in an unordered list. The actual runtime (gfx, llvm...) will later query this context
            # in correct order.
            if launch_ctx_args := launch_ctx_buffer.get(_FLOAT):
                launch_ctx.set_args_float(*zip(*launch_ctx_args))  # type: ignore
            if launch_ctx_args := launch_ctx_buffer.get(_INT):
                launch_ctx.set_args_int(*zip(*launch_ctx_args))  # type: ignore
            if launch_ctx_args := launch_ctx_buffer.get(_UINT):
                launch_ctx.set_args_uint(*zip(*launch_ctx_args))  # type: ignore
            if launch_ctx_args := launch_ctx_buffer.get(_QD_ARRAY):
                launch_ctx.set_args_ndarray(*zip(*launch_ctx_args))  # type: ignore
            if launch_ctx_args := launch_ctx_buffer.get(_QD_ARRAY_WITH_GRAD):
                launch_ctx.set_args_ndarray_with_grad(*zip(*launch_ctx_args))  # type: ignore

            if is_launch_ctx_cacheable and args_hash is not None:
                self.launch_context_buffer_cache.cache(t_kernel, args_hash, launch_ctx, launch_ctx_buffer)

        try:
            prog = impl.get_runtime().prog
            if not compiled_kernel_data:
                # Store Taichi program config and device cap for efficiency because they are used at multiple places
                prog_config = prog.config()
                prog_device_cap = prog.get_device_caps()

                compile_result: CompileResult = prog.compile_kernel(prog_config, prog_device_cap, t_kernel)
                compiled_kernel_data = compile_result.compiled_kernel_data
                if compile_result.cache_hit:
                    self.fe_ll_cache_observations.cache_hit = True
                if self.fast_checksum:
                    src_hasher.store(
                        compile_result.cache_key,
                        self.fast_checksum,
                        self.visited_functions,
                        self.used_py_dataclass_parameters_by_key_enforcing[key],
                    )
                    self.src_ll_cache_observations.cache_stored = True
            self._last_compiled_kernel_data = compiled_kernel_data
            prog.launch_kernel(compiled_kernel_data, launch_ctx)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if impl.get_runtime().print_full_traceback:
                raise e
            raise e from None

        for callback in callbacks:
            callback()

        return_type = self.return_type
        if return_type or self.has_print:
            runtime_ops.sync()

        if not return_type:
            return None
        if len(return_type) == 1:
            return self.construct_kernel_ret(launch_ctx, return_type[0], (0,))
        return tuple([self.construct_kernel_ret(launch_ctx, ret_type, (i,)) for i, ret_type in enumerate(return_type)])

    def construct_kernel_ret(self, launch_ctx: KernelLaunchContext, ret_type: Any, indices: tuple[int, ...]):
        if isinstance(ret_type, CompoundType):
            return ret_type.from_kernel_struct_ret(launch_ctx, indices)
        if ret_type in primitive_types.integer_types:
            if is_signed(cook_dtype(ret_type)):
                return launch_ctx.get_struct_ret_int(indices)
            return launch_ctx.get_struct_ret_uint(indices)
        if ret_type in primitive_types.real_types:
            return launch_ctx.get_struct_ret_float(indices)
        raise QuadrantsRuntimeTypeError(f"Invalid return type on index={indices}")

    def ensure_compiled(self, *py_args: tuple[Any, ...]) -> tuple[Callable, int, AutodiffMode]:
        try:
            instance_id, arg_features = self.mapper.lookup(self.raise_on_templated_floats, py_args)
        except Exception as e:
            raise type(e)(f"exception while trying to ensure compiled {self.func}:\n{e}") from e
        key = (self.func, instance_id, self.autodiff_mode)
        self.materialize(key=key, py_args=py_args, arg_features=arg_features)
        return key

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    @_shell_pop_print
    def __call__(self, *py_args, **kwargs) -> Any:
        self.raise_on_templated_floats = impl.current_cfg().raise_on_templated_floats
        py_args = self.fuse_args(is_func=False, is_pyfunc=False, py_args=py_args, kwargs=kwargs, global_context=None)

        # Transform the primal kernel to forward mode grad kernel
        # then recover to primal when exiting the forward mode manager
        if self.runtime.fwd_mode_manager and not self.runtime.grad_replaced:
            # TODO: if we would like to compute 2nd-order derivatives by forward-on-reverse in a nested context manager
            # fashion, i.e., a `Tape` nested in the `FwdMode`, we can transform the kernels with
            # `mode_original == AutodiffMode.REVERSE` only, to avoid duplicate computation for 1st-order derivatives.
            self.runtime.fwd_mode_manager.insert(self)

        # Both the class kernels and the plain-function kernels are unified now.
        # In both cases, |self.grad| is another Kernel instance that computes the
        # gradient. For class kernels, args[0] is always the kernel owner.

        # No need to capture grad kernels because they are already bound with their primal kernels
        if self.autodiff_mode in (_NONE, _VALIDATION) and self.runtime.target_tape and not self.runtime.grad_replaced:
            self.runtime.target_tape.insert(self, py_args)

        if self.autodiff_mode != _NONE and impl.current_cfg().opt_level == 0:
            _logging.warn("""opt_level = 1 is enforced to enable gradient computation.""")
            impl.current_cfg().opt_level = 1
        key = self.ensure_compiled(*py_args)
        self._last_launch_key = key
        kernel_cpp = self.materialized_kernels[key]
        compiled_kernel_data = self.compiled_kernel_data_by_key.get(key, None)
        self.launch_observations.found_kernel_in_materialize_cache = compiled_kernel_data is not None
        ret = self.launch_kernel(key, kernel_cpp, compiled_kernel_data, *py_args)
        if compiled_kernel_data is None:
            assert self._last_compiled_kernel_data is not None
            self.compiled_kernel_data_by_key[key] = self._last_compiled_kernel_data
        return ret
