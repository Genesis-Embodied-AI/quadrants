import ast
import dataclasses
import json
import os
import pathlib
import time
import warnings
from collections import defaultdict
from dataclasses import _FIELDS  # type: ignore[reportAttributeAccessIssue]

# Must import 'partial' directly instead of the entire module to avoid attribute lookup overhead.
from functools import partial
from typing import Any, Callable

# Must import 'ReferenceType' directly instead of the entire module to avoid attribute lookup overhead.
from weakref import ReferenceType

from quadrants import _logging

_GRAPH_ENABLED = os.environ.get("QD_GRAPH", "1") == "1"

from quadrants import _tensor_wrapper


def _kernel_coverage_enabled() -> bool:
    return os.environ.get("QD_KERNEL_COVERAGE") == "1"


from quadrants._lib.core.quadrants_python import (
    Arch,
    ASTBuilder,
    CompiledKernelData,
    CompileResult,
    KernelCxx,
    KernelLaunchContext,
)
from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants.lang import _kernel_impl_dataclass, impl, runtime_ops

# `qd.checkpoint` pause / resume model helpers. See `kernel_checkpoint.py` for the full extracted surface; `Kernel`
# delegates the resume-cookie validation, label translation, per-launch yield_on= arg-id table build, and GraphStatus
# construction to those free functions so this hot file doesn't accrete checkpoint-feature-specific blocks.
from quadrants.lang import kernel_checkpoint as _checkpoint_helpers
from quadrants.lang._fast_caching import src_hasher
from quadrants.lang._template_mapper_hotpath import chain_has_mutable_container
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
from quadrants.lang.util import cook_dtype, is_data_oriented
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
_ARCH_PYTHON = Arch.python


class LaunchContextBufferCache:
    # Here, we are tracking whether a launch context buffer can be cached. The point of caching the launch context
    # buffer is allowing skipping recursive processing of all the input arguments one-by-one, which is adding a
    # significant overhead, without changing anything in regards of the function calls to the launch context that must
    # be made for a given kernel.
    # You can understand this as resolving the static part of the entire control flow of '_recursive_set_args' for a
    # given set of arguments, which is (mostly surely uniquely) characterized by its hash, gathering all the
    # instructions that cannot be evaluated statically and packing them in a buffer without evaluating them at this
    # point. This buffer is then cached once and for all and evaluated every time the exact same set of input argument
    # is passed. This means that, ultimately, it will result in the exact same function calls with or without caching.
    # In this particular case, the function calls corresponds to adding arguments to the current context for this kernel
    # call.
    # A launch context buffer is considered cache-friendly if and only if no direct call to the launch context where
    # made preemptively during the recursive processing of the arguments, all of parameters of the arguments are
    # pointers, the address of these pointers cannot change, and the set of parameters is fixed.
    # The lifetime of a cache entry is bound to the lifetime of any of its input arguments: the first being garbage
    # collected will invalidate the entire entry. Moreover, the entire cache registry is bound to the lifetime of the
    # taichi prog itself, which means that calling `qd.reset()` will automatically clear the cache. Note that the cache
    # stores wear references to pointers, so it does not hold alife any allocated memory.
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

        def _evict_callback(ref, _tracker=launch_ctx_cache_tracker_, _self=self, _hash=args_hash):
            _tracker.clear()
            _self._launch_ctx_cache.pop(_hash, None)
            _self._launch_ctx_cache_tracker.pop(_hash, None)

        if launch_ctx_args := launch_ctx_buffer.get(_QD_ARRAY):
            _, arrs = zip(*launch_ctx_args)
            launch_ctx_cache_tracker_ += [ReferenceType(arr, _evict_callback) for arr in arrs]
        if launch_ctx_args := launch_ctx_buffer.get(_QD_ARRAY_WITH_GRAD):
            _, arrs, arrs_grad = zip(*launch_ctx_args)
            launch_ctx_cache_tracker_ += [ReferenceType(arr, _evict_callback) for arr in arrs]
            launch_ctx_cache_tracker_ += [ReferenceType(arr_grad, _evict_callback) for arr_grad in arrs_grad]
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
            # struct locals are the expanded py dataclass fields that we will write to local variables, and will then
            # be available to use in build_Call, later.
            tree = _kernel_impl_dataclass.unpack_ast_struct_expressions(self.tree, struct_locals=struct_locals)
            ctx.only_parse_function_def = self.only_parse_function_def
            # Rebuild the graph_do_while level table from scratch each compilation pass (build_While appends to it as it
            # walks the AST). Skip when only_parse_function_def: the body is not walked, so build_While never runs to
            # repopulate it -- and on a fast-cache restore the table was already rebuilt from the cached (cond_arg_name,
            # parent_id) pairs in _try_load_fastcache.
            if not ctx.only_parse_function_def:
                self.current_kernel.graph_do_while_levels = []
                self.current_kernel._graph_do_while_level_stack = []
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


@dataclasses.dataclass
class GraphDoWhileLevel:
    """One nested ``qd.graph_do_while`` loop in a ``graph=True`` kernel, indexed by level id (assigned
    outer-before-inner by the AST transformer). Mirrors the C++ ``GraphDoWhileLevel``."""

    cond_arg_name: str
    parent_id: int
    cond_cpp_arg_id: int = -1


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
        impl.get_runtime().kernels.append(self)  # type: ignore[arg-type]
        self.reset()
        self.kernel_cpp: None | KernelCxx = None
        # A materialized kernel is a KernelCxx object which may or may not have been compiled. It generally has been
        # converted at least as far as AST and front-end IR, but not necessarily any further.
        self.materialized_kernels: dict[CompiledKernelKeyType, KernelCxx] = {}
        self.has_print = False
        self.use_graph: bool = False
        # Opt-in flag set by `@qd.kernel(graph=True, checkpoints=True)`. When True, the AST transformer enables
        # `qd.checkpoint(...)` recognition AND auto-wraps every top-level for-loop that isn't already inside a
        # `with qd.checkpoint(...)` block in an implicit no-yield checkpoint. When False, any use of
        # `qd.checkpoint(...)` in the kernel body is rejected at compile time with a fix-it pointing at
        # `checkpoints=True`.
        self.use_checkpoints: bool = False
        # Legacy single-loop arg name, kept for reporting/back-compat; equals the outermost level's condition arg for
        # nested kernels. The authoritative data is `graph_do_while_levels`.
        self.graph_do_while_arg: str | None = None
        # Nested graph_do_while level table, indexed by level id (outer before inner). Rebuilt each compilation pass by
        # the AST transformer; serialized to the launch context at launch.
        self.graph_do_while_levels: list[GraphDoWhileLevel] = []
        # Transient stack of active level ids, used only while transforming the AST.
        self._graph_do_while_level_stack: list[int] = []
        # Per-checkpoint metadata, one entry per `with qd.checkpoint(...)` block (explicit AND auto-injected implicit)
        # in declaration order. List index is the checkpoint's internal `cp_id` (0, 1, 2, ... dense, flat across the
        # kernel). Each entry is the name of the `yield_on=` kernel parameter, or `None` for implicit checkpoints
        # (which never yield). Populated by the AST transformer; empty means the kernel uses no checkpoints.
        self.checkpoint_yield_on_args: list[str | None] = []
        # User-facing labels for explicit checkpoints. Same indexing as `checkpoint_yield_on_args`: entry `i` is the int
        # (or IntEnum value) the user passed as the first positional arg of `qd.checkpoint(cp_id, yield_on)` for the
        # checkpoint whose internal cp_id is `i`. Implicit checkpoints (auto-wrapped) get `None` (they have no
        # user-facing label and can never appear in `GraphStatus.checkpoint`). The label is preserved as-is so an
        # `IntEnum` round-trips: writing `qd.checkpoint(Stage.SIM, ...)` and then reading `status.checkpoint` returns
        # `Stage.SIM` rather than the raw int.
        self.checkpoint_user_labels_by_cp_id: list[int | None] = []
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
        self._struct_ndarray_launch_info_by_key: dict[CompiledKernelKeyType, list] = {}
        self._mutable_nd_cached_key: CompiledKernelKeyType | None = None
        self._mutable_nd_cached_val: list = []
        self._tensor_unwrap_indices: tuple[int, ...] | None = None

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
            cached_graph_do_while_levels: list[tuple[str, int]] | None = None
            if self.fast_checksum:
                self.src_ll_cache_observations.cache_key_generated = True
                used_py_dataclass_parameters, frontend_cache_key, cached_graph_do_while_levels = src_hasher.load(  # type: ignore[reportAssignmentType]
                    self.fast_checksum
                )
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
                    # Fast-cache restore skips AST transformation, so rebuild the gdw level table (and the legacy
                    # outermost-arg alias) from the cached (cond_arg_name, parent_id) pairs.
                    if cached_graph_do_while_levels:
                        self.graph_do_while_levels = [
                            GraphDoWhileLevel(cond_arg_name=name, parent_id=parent)
                            for name, parent in cached_graph_do_while_levels
                        ]
                        self.graph_do_while_arg = self.graph_do_while_levels[0].cond_arg_name
                    return used_py_dataclass_parameters

        elif self.quadrants_callable and not self.quadrants_callable.is_pure and self.runtime.print_non_pure:
            # The bit in caps should not be modified without updating corresponding test
            # freetext can be freely modified.
            # As for why we are using `print` rather than eg logger.info, it is because this is only printed when
            # qd.init(print_non_pure=..) is True. And it is confusing to set that to True, and see nothing printed.
            print(f"[NOT_PURE] Debug information: not pure: {self.func.__name__}")
        return None

    def materialize(self, key: "CompiledKernelKeyType | None", py_args: tuple[Any, ...], arg_features=None):
        if key is None:
            key = (self.func, 0, self.autodiff_mode)
        self.fast_checksum = None
        if key in self.materialized_kernels:
            return

        # Deprecation warning: passing a ``@dataclasses.dataclass`` instance through a ``qd.Template``-annotated kernel
        # parameter was never an intentional Quadrants pattern. It works inadvertently because the template walker
        # happens to handle dataclass-shaped objects, but the supported annotation for a ``@dataclasses.dataclass`` is
        # the dataclass type itself (flat-by-fields path). We fire the warning here, after the ``materialized_kernels``
        # cache-hit early return above, so it only runs on the first compile for each unique spec-key — zero cost on the
        # steady-state launch hot path. Doubly-decorated objects (``@qd.data_oriented`` over ``@dataclasses.dataclass``)
        # are excluded because that combination is a legitimate pattern routed through the data-oriented path.
        for arg_meta, val in zip(self.arg_metas, py_args):
            ann = arg_meta.annotation
            if ann is not template and type(ann) is not template:
                continue
            if dataclasses.is_dataclass(val) and not isinstance(val, type) and not is_data_oriented(val):
                warnings.warn(
                    f"Kernel {self.func.__qualname__!r} parameter {arg_meta.name!r}: passing a "
                    "@dataclasses.dataclass instance into a qd.Template-annotated kernel parameter was "
                    "never intended to be supported, and only works inadvertently. Use the dataclass type "
                    f"itself as the annotation instead (e.g. `def {self.func.__name__}({arg_meta.name}: "
                    f"{type(val).__name__}, ...)`). In a future release this will become an error.",
                    DeprecationWarning,
                    stacklevel=4,
                )

        if _kernel_coverage_enabled():
            from . import _kernel_coverage  # pylint: disable=import-outside-toplevel

            _kernel_coverage.ensure_field_allocated()

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
                    self._struct_ndarray_launch_info_by_key[key] = getattr(
                        ctx.global_context, "struct_ndarray_launch_info", []
                    )
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

    def launch_kernel(
        self,
        key,
        t_kernel: KernelCxx,
        compiled_kernel_data: CompiledKernelData | None,
        *args,
        qd_stream=None,
        _resume_from_checkpoint: int | None = None,
    ) -> Any:
        assert len(args) == len(self.arg_metas), f"{len(self.arg_metas)} arguments needed but {len(args)} provided"

        callbacks: list[Callable[[], None]] = []
        launch_ctx = t_kernel.make_launch_context()
        # Special treatment for primitive types is unecessary and detrimental. See 'TemplateMapper.lookup' for details.
        args_hash: "ArgsHash" = (id(t_kernel), *[id(arg) for arg in args])
        # Stale-cache guard for mutable structs containing ndarrays. Frozen dataclass fields cannot be reassigned, so
        # id(struct) in args_hash is already sufficient. For mutable structs, ndarray attributes can change between
        # calls while the struct id stays the same, so we fold the live ndarray id(s) into the hash.
        #
        # The predicate must catch any "host container in which ndarray member references can be reassigned at
        # runtime" case. Non-frozen dataclasses have ``__hash__ is None`` (Python sets it when ``eq=True,
        # frozen=False``), so they hit the first arm. ``@qd.data_oriented`` classes inherit ``object.__hash__`` so the
        # ``__hash__ is None`` check is False for them — we need a separate arm. Without it, ``state.x = other_nd`` on
        # the same data_oriented instance would not invalidate the launch-context cache and the kernel would re-launch
        # against the stale binding.
        #
        # Mutability must be checked across the *entire* attr-chain, not just the top-level arg. With a frozen outer
        # container wrapping a mutable inner container that holds the ndarray (e.g. frozen dataclass -> data_oriented
        # -> ndarray), id(outer) alone does not capture leaf rebinding because the inner container can still reassign
        # ``.x``. So we OR-fold the mutability check across every parent along ``chain`` from the root down to (but
        # excluding) the leaf attribute.
        if key != self._mutable_nd_cached_key:
            if self._struct_ndarray_launch_info_by_key:
                struct_nd_info = self._struct_ndarray_launch_info_by_key.get(key)
                if struct_nd_info:
                    self._mutable_nd_cached_val = [
                        (idx, chain)
                        for _, idx, chain in struct_nd_info
                        if chain_has_mutable_container(args, idx, chain)
                    ]
                else:
                    self._mutable_nd_cached_val = []
            else:
                self._mutable_nd_cached_val = []
            self._mutable_nd_cached_key = key
        if self._mutable_nd_cached_val:
            args_hash = (
                *args_hash,
                *(id(self._resolve_struct_ndarray(args, idx, chain)) for idx, chain in self._mutable_nd_cached_val),
            )
        if not self.launch_context_buffer_cache.populate_launch_ctx_from_cache(args_hash, launch_ctx):
            launch_ctx_buffer: dict[KernelBatchedArgType, list[tuple]] = defaultdict(list)
            actual_argument_slot = 0
            is_launch_ctx_cacheable = True
            template_num = 0
            i_out = 0
            _checkpoint_helpers.init_yield_on_arg_id_table(self)
            for i_in, val in enumerate(args):
                needed_ = self.arg_metas[i_in].annotation
                if needed_ is template or type(needed_) is template:
                    template_num += 1
                    i_out += 1
                    continue
                # FIXME: This shortcut skips _recursive_set_args() solely when val._qd_all_field is true and the annotation is
                # a dataclass, but _recursive_set_args() is where the strict provided_arg_type-is-needed_arg_type check lives.
                # As a result, once an instance has _qd_all_field=True, passing it to a kernel parameter annotated with a
                # different all-Field dataclass type can be silently accepted instead of raising the previous runtime type error,
                # which weakens API/type safety and can route the wrong struct type through launch.
                if getattr(val, "_qd_all_field", False) and getattr(needed_, _FIELDS, None) is not None:
                    continue
                if self.graph_do_while_levels:
                    for _gdw_level in self.graph_do_while_levels:
                        if self.arg_metas[i_in].name == _gdw_level.cond_arg_name:
                            _gdw_level.cond_cpp_arg_id = i_out - template_num
                _checkpoint_helpers.maybe_record_yield_on_arg(self, self.arg_metas[i_in].name, i_out - template_num)
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

            struct_nd_info = self._struct_ndarray_launch_info_by_key.get(key)
            if struct_nd_info:
                self._set_struct_ndarray_args(struct_nd_info, args, launch_ctx_buffer, is_launch_ctx_cacheable)

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
                        graph_do_while_levels=[  # type: ignore[reportCallIssue]
                            (level.cond_arg_name, level.parent_id) for level in self.graph_do_while_levels
                        ],
                    )
                    self.src_ll_cache_observations.cache_stored = True
            self._last_compiled_kernel_data = compiled_kernel_data
            launch_ctx.use_graph = self.use_graph and _GRAPH_ENABLED
            if self.use_graph and qd_stream is not None:
                raise RuntimeError(
                    "qd_stream is not compatible with graph=True kernels. "
                    "See docs/source/user_guide/streams.md for details."
                )
            for _gdw_level in self.graph_do_while_levels:
                launch_ctx.add_graph_do_while_level(_gdw_level.cond_cpp_arg_id, _gdw_level.parent_id)
            _checkpoint_helpers.forward_yield_on_table_to_ctx(self, launch_ctx)
            # `_resume_from_checkpoint` is `None` for fresh launches (host-side default 0 in `LaunchContextBuilder`,
            # which means "run every checkpoint"). When `Kernel.resume` plumbs an int through, copy it onto the launch
            # context so the GraphManager's `launch_cached_graph` memcpys it into the device-side `resume_point` slot
            # instead of clearing to 0. Slice 2 implementation; pre-CUDA-12.4 / non-CUDA backends ignore the value since
            # they don't have a resume_point slot today (slices 4-6 will add an indirect-dispatch equivalent).
            if _resume_from_checkpoint is not None:
                launch_ctx.resume_from_checkpoint = int(_resume_from_checkpoint)
            stream_handle = qd_stream.handle if qd_stream is not None else 0
            if stream_handle:
                prog.set_current_cuda_stream(stream_handle)
            try:
                prog.launch_kernel(compiled_kernel_data, launch_ctx)
            finally:
                if stream_handle:
                    prog.set_current_cuda_stream(0)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if impl.get_runtime().print_full_traceback:
                raise e
            raise e from None

        for callback in callbacks:
            callback()

        return_type = self.return_type
        if return_type or self.has_print:
            if qd_stream is not None and self.has_print and not return_type:
                qd_stream.synchronize()
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

    @staticmethod
    def _resolve_struct_ndarray(args, template_arg_idx, attr_chain):
        """Walk a struct's attribute chain to find the live ndarray (or Tensor wrapper)."""
        obj = args[template_arg_idx]
        for attr_name in attr_chain:
            obj = getattr(obj, attr_name)
        if type(obj) in _TENSOR_WRAPPER_TYPES:
            obj = obj._unwrap()
        return obj

    @staticmethod
    def _set_struct_ndarray_args(
        launch_info: list,
        args: tuple,
        launch_ctx_buffer: dict,
        is_launch_ctx_cacheable: bool,
    ) -> None:
        """Set ndarray kernel args that were pre-declared from struct template fields during compilation."""
        from quadrants.lang._ndarray import Ndarray  # pylint: disable=C0415

        for arg_id, template_arg_idx, attr_chain in launch_info:
            obj = args[template_arg_idx]
            for attr_name in attr_chain:
                obj = getattr(obj, attr_name)
            if type(obj) in _TENSOR_WRAPPER_TYPES:
                obj = obj._unwrap()
            assert isinstance(obj, Ndarray), f"Expected Ndarray at {attr_chain}, got {type(obj)}"
            v_primal = obj.arr
            v_grad = obj.grad.arr if obj.grad else None
            if v_grad is None:
                launch_ctx_buffer[_QD_ARRAY].append((arg_id, v_primal))
            else:
                launch_ctx_buffer[_QD_ARRAY_WITH_GRAD].append((arg_id, v_primal, v_grad))

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
        qd_stream = kwargs.pop("qd_stream", None)
        # Pop the resume cookie before anything else touches kwargs -- the AST mapper sees user parameter names only, so
        # a stray `from_checkpoint=` would raise "unexpected kwarg". `_resume_from_checkpoint` is the resolved cp_id to
        # copy into the device-side `resume_point` slot before launch; `None` means "fresh start, reset to 0". Plumbed
        # via `Kernel.resume()` only; users do not pass this directly.
        _resume_from_checkpoint = kwargs.pop("_qd_from_checkpoint", None)
        _checkpoint_helpers.validate_resume_cookie(self, _resume_from_checkpoint)
        if qd_stream is not None and self.autodiff_mode != _NONE:
            raise RuntimeError(
                "qd_stream is not compatible with autodiff kernels. Streams cannot be used with "
                "reverse-mode or forward-mode differentiation."
            )
        if qd_stream is not None and self.runtime.target_tape:
            raise RuntimeError(
                "qd_stream is not compatible with autograd Tape. Launch the kernel outside the Tape "
                "context, or omit qd_stream."
            )
        if impl.get_runtime()._arch == _ARCH_PYTHON:
            return self.func(*py_args, **kwargs)
        config = impl.current_cfg()

        self.raise_on_templated_floats = config.raise_on_templated_floats
        py_args = self.fuse_args(is_func=False, is_pyfunc=False, py_args=py_args, kwargs=kwargs, global_context=None)
        # Tensor-wrapper unwrap (stork-17). Substitute each ``qd.Tensor`` instance (including ``VectorTensor`` /
        # ``MatrixTensor`` subclasses) with its underlying ``Ndarray`` / ``ScalarField`` impl *before* anything
        # downstream observes the arg tuple — including the autograd tape (uses identity), the template mapper
        # (cache-keys on ``id(arg)``), ``_extract_arg``, and the AST builder. This guarantees JIT cache stability:
        # ``id(Tensor(impl))`` differs across constructions, but ``id(impl)`` is stable, so wrapper-or-not yields
        # identical cache keys.
        #
        # PERF: On first call, record which arg positions are Tensor wrappers. On subsequent calls, skip entirely
        # (empty indices) or unwrap only the cached positions (no full-arg scan). For a kernel with 30 args and 1
        # Tensor, this reduces per-call type checks from 30 to 1.
        #
        # Safety of caching: kernel parameter annotations are fixed per position (they come from the function
        # signature and are stored in ``self.mapper.arguments``). Whether a given position receives a Tensor wrapper
        # or a bare impl is determined by the caller's annotation pattern, which is stable across calls — a user who
        # passes ``qd.Tensor(impl)`` at position *i* will do so on every call, because the annotation (``qd.Tensor``,
        # ``qd.Template``, ``qd.types.ndarray()``, or a dataclass type) doesn't change. The template mapper
        # enforces a fixed arg count (``len(args) == self.num_args``), so cached indices cannot go out of bounds.
        if _tensor_wrapper._any_tensor_constructed:  # pyright: ignore[reportOptionalMemberAccess]
            _indices = self._tensor_unwrap_indices
            if _indices is None:
                _indices = tuple(i for i, a in enumerate(py_args) if type(a) in _TENSOR_WRAPPER_TYPES)
                self._tensor_unwrap_indices = _indices
                if _indices:
                    py_args_l = list(py_args)
                    for i in _indices:
                        py_args_l[i] = py_args_l[i]._impl  # pyright: ignore[reportAttributeAccessIssue]
                    py_args = tuple(py_args_l)
            elif _indices:
                py_args_l = list(py_args)
                for i in _indices:
                    py_args_l[i] = py_args_l[i]._impl  # pyright: ignore[reportAttributeAccessIssue]
                py_args = tuple(py_args_l)

        # Transform the primal kernel to forward mode grad kernel
        # then recover to primal when exiting the forward mode manager
        if self.runtime.fwd_mode_manager and not self.runtime.grad_replaced:
            # TODO: if we would like to compute 2nd-order derivatives by forward-on-reverse in a nested context manager
            # fashion, i.e., a `Tape` nested in the `FwdMode`, we can transform the kernels with
            # `mode_original == AutodiffMode.REVERSE` only, to avoid duplicate computation for 1st-order derivatives.
            self.runtime.fwd_mode_manager.insert(self)

        # Both the class kernels and the plain-function kernels are unified now. In both cases, |self.grad| is another
        # Kernel instance that computes the gradient. For class kernels, args[0] is always the kernel owner.

        # No need to capture grad kernels because they are already bound with their primal kernels
        if self.autodiff_mode in (_NONE, _VALIDATION) and self.runtime.target_tape and not self.runtime.grad_replaced:
            self.runtime.target_tape.insert(self, py_args)

        if self.autodiff_mode != _NONE and impl.current_cfg().opt_level == 0:
            _logging.warn("""opt_level = 1 is enforced to enable gradient computation.""")
            impl.current_cfg().opt_level = 1
        key = self.ensure_compiled(*py_args)  # type: ignore[arg-type]
        self._last_launch_key = key
        kernel_cpp = self.materialized_kernels[key]
        compiled_kernel_data = self.compiled_kernel_data_by_key.get(key, None)
        self.launch_observations.found_kernel_in_materialize_cache = compiled_kernel_data is not None
        # Translate the user-supplied `from_checkpoint=` label into the dense, source-order internal cp_id the runtime
        # uses. Translation happens here (after `ensure_compiled`) because `checkpoint_user_labels_by_cp_id` is
        # populated during AST processing inside `ensure_compiled`.
        if _resume_from_checkpoint is not None:
            _resume_from_checkpoint = _checkpoint_helpers.translate_user_label_to_internal_cp_id(
                self, _resume_from_checkpoint
            )
        # Only forward `_resume_from_checkpoint` when the caller actually supplied one (i.e. via `Kernel.resume(...)`).
        # Otherwise omit the kwarg entirely so subclasses / monkeypatches of `launch_kernel` that pre-date this kwarg
        # keep working unmodified. The host-side default in `LaunchContextBuilder` is 0 ("run every checkpoint"), which
        # matches the `None` semantics in `launch_kernel`.
        if _resume_from_checkpoint is None:
            ret = self.launch_kernel(
                key,
                kernel_cpp,
                compiled_kernel_data,
                *py_args,
                qd_stream=qd_stream,
            )
        else:
            ret = self.launch_kernel(
                key,
                kernel_cpp,
                compiled_kernel_data,
                *py_args,
                qd_stream=qd_stream,
                _resume_from_checkpoint=_resume_from_checkpoint,
            )
        if compiled_kernel_data is None:
            assert self._last_compiled_kernel_data is not None
            self.compiled_kernel_data_by_key[key] = self._last_compiled_kernel_data
        # Surface a GraphStatus for kernels with `qd.checkpoint(yield_on=...)` so the host can drive the qipc-style
        # re-entrant loop. Kernels without yield-capable checkpoints get `ret` (typically `None`) passed through.
        return _checkpoint_helpers.maybe_build_graph_status(self, ret)
