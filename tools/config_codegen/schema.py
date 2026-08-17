"""Single source of truth for the qd.init / CompileConfig options.

Read this as a declarative table: every attribute of :class:`Options` is one
option. The attribute *name* is the option name (and the qd.init keyword, and
the ``QD_<UPPER>`` environment variable); the *annotation* is its type; the
assigned *value* is the default; and the string literal *directly above it* is
the end-user documentation. Nothing here imports ``quadrants`` or anything
third-party, so it is safe to import both at C++ build time (before the
extension exists) and at docs build time.

Type rules:
  * Annotate with a normal Python type (``bool``, ``int``, ``str``, ``float``).
    The C++ type is derived from it (bool->bool, int->int, float->double,
    str->std::string).
  * Python has a single ``int`` but C++ has several integer widths, so when a
    field needs a specific one override just the C++ side with
    ``Annotated[int, Cpp("std::size_t")]`` -- the Python type stays ``int``.
  * For non-primitive C++ types that the frontend exposes as their own Python
    objects (``Arch``, ``DataType``), annotate with a bare ``Cpp("Arch")``.

Default rules:
  * A compile-time literal (``True``, ``0``, ``"lru"``) becomes an in-class
    member initializer in the generated struct.
  * ``Computed(cpp_expr, doc)`` is a default that is not a literal (e.g.
    ``host_arch()``); the field is assigned ``cpp_expr`` in the generated
    constructor fragment instead, mirroring how compile_config.cpp does it by
    hand today.

Binding rules:
  * Every option is exposed to Python via a nanobind ``def_rw`` by default.
  * A few fields are internal-only (never settable from qd.init); mark them with
    ``Annotated[<type>, NoBind]`` so no binding is generated for them.

This lists every field of the C++ CompileConfig struct. The most commonly-tuned
qd.init options come first; the remainder mirror the historical struct order.
See the B1 design doc, perso_hugh/doc/config_self_documenting_design.md.
"""

from typing import Annotated


class Cpp:
    """Annotation metadata: force a specific C++ type for an option.

    Use inside ``Annotated`` when the default Python->C++ mapping is not the one
    you want (typically to select an exact integer width), or bare as the
    annotation itself for a non-primitive frontend type such as ``Arch``. ``py``
    overrides the Python type name shown in the docs (defaults to ``cpp_type``).
    """

    def __init__(self, cpp_type: str, py: str | None = None):
        self.cpp_type = cpp_type
        self.py = py


class Computed:
    """A default that is not a compile-time literal.

    The field is declared without an in-class initializer and assigned
    ``cpp_expr`` in the generated constructor fragment (mirroring how
    compile_config.cpp sets these by hand today). ``doc`` is the human-readable
    description of the resulting default, shown in the docs table.
    """

    def __init__(self, cpp_expr: str, doc: str):
        self.cpp_expr = cpp_expr
        self.doc = doc


class NoBind:
    """Annotation marker: this field is internal and gets no Python binding.

    Use as ``Annotated[<type>, NoBind]``. The field still exists in the C++
    struct (and is set from its default), but is not exposed as a qd.init
    keyword, so it stays out of the user-facing docs too.
    """


class Options:
    """qd.init / CompileConfig options (see the module docstring)."""

    # --- Commonly-tuned options (shown first in the docs) --------------------

    """Whether compiled-kernel caches persist on disk and are reused across
    separate Python processes."""
    offline_cache: bool = True

    """Directory that holds the on-disk compilation cache."""
    offline_cache_file_path: str = Computed(
        'get_repo_dir() + "qdcache"',
        "<cache dir>/qdcache (e.g. ~/.cache/quadrants/qdcache)",
    )

    """Run the control-flow-graph optimization, an internal compile-time pass
    that simplifies your kernel's branches and loops. Disabling it makes
    compilation much faster at a small runtime cost."""
    cfg_optimization: bool = True

    """Allow IEEE-relaxed floating-point optimizations (e.g. fused
    multiply-add). Faster, but drops strict NaN/inf/signed-zero guarantees."""
    fast_math: bool = True

    """Number of host threads used to compile kernels."""
    num_compile_threads: int = 4

    """Turn on the full suite of correctness checks (field bounds, assertions,
    adstack overflow). Considerably slower; intended for development."""
    debug: bool = False

    """Enable the field out-of-bounds check on tensor indexing without turning
    on the rest of debug mode."""
    check_out_of_bound: bool = False

    """Enable the reverse-mode autodiff pipeline for kernels with
    runtime-bounded loops (the adstack)."""
    ad_stack_experimental_enabled: bool = False

    """Force every autodiff stack to exactly this many slots. 0 lets the
    launch-time sizer choose automatically."""
    ad_stack_size: int = 0

    # Maintainer note: below this conservative-heap threshold a kernel keeps the
    # eager `linear_thread_idx * stride` adstack heap addressing instead of paying
    # the per-launch reducer dispatch + per-task DtoH for the `bound_expr`-driven
    # sparse heap sizing. Above it, the static analyser captures the gating
    # predicate and routes the task through the lazy LCA-block atomic-rmw row
    # claim, sizing the float adstack heap from the runtime-counted
    # gate-passing-thread count rather than `dispatched_threads * stride *
    # sizeof(float)`. Set to 0 to always capture (force the sparse path - useful
    # for tests that pin the reducer-backed sizing) or a very large value to
    # disable it.
    """Byte cutoff below which the sparse adstack sizing path is skipped in favor
    of eager heap allocation."""
    ad_stack_sparse_threshold_bytes: Annotated[int, Cpp("std::size_t")] = 100 * 1024 * 1024

    # Maintainer note: the queue is borrowed (not retained) - the caller must keep
    # it alive for the lifetime of the Quadrants runtime.
    """An MTLCommandQueue pointer (as an integer) to dispatch on instead of
    creating a new Metal queue. 0 means create a new queue."""
    external_metal_command_queue: Annotated[int, Cpp("uint64_t")] = 0

    # Maintainer note: lets Quadrants skip explicit cross-framework sync at interop
    # points (to_torch / from_torch).
    """Set True when external_metal_command_queue is PyTorch MPS's own queue, to
    skip redundant interop synchronization."""
    external_metal_command_queue_is_torch_queue: bool = False

    # --- Full field list (mirrors the historical CompileConfig order) --------

    """Target backend the kernels run on (e.g. qd.cpu, qd.cuda, qd.vulkan,
    qd.metal). Defaults to the best backend available on this machine."""
    arch: Cpp("Arch") = Computed("host_arch()", "the best backend available on this machine")

    # Unused: kept as a struct field for ABI stability but read nowhere.
    """Reserved internal flag; not currently used."""
    validate_autodiff: Annotated[bool, NoBind] = False

    """SIMD lane width used by the CPU backend's vectorizer."""
    simd_width: Annotated[int, NoBind] = Computed("default_simd_width(arch)", "chosen from the target arch")

    """LLVM optimization level applied to generated kernels."""
    opt_level: int = 1

    """Optimization level passed to external backend compilers (e.g. Metal)."""
    external_optimization_level: Annotated[int, NoBind] = 3

    """Maximum vector width, in elements, that the vectorizer will emit."""
    max_vector_width: Annotated[int, NoBind] = 8

    """Raise an error instead of silently specializing a kernel on a Python
    float argument passed by value."""
    raise_on_templated_floats: bool = False

    """Print each kernel's IR right after frontend preprocessing."""
    print_preprocessed_ir: bool = False

    """Print each kernel's IR after compilation (for debugging the compiler)."""
    print_ir: bool = False

    """Print the IR generated for field accessor kernels."""
    print_accessor_ir: bool = False

    """Include source-line debug info when printing IR."""
    print_ir_dbg_info: bool = False

    """Force serial (single-threaded) execution of the compiled schedule."""
    serial_schedule: Annotated[bool, NoBind] = False

    """Run the simplify pass before the lower-access pass."""
    simplify_before_lower_access: bool = True

    """Lower high-level field accesses to low-level pointer arithmetic."""
    lower_access: bool = True

    """Run the simplify pass after the lower-access pass."""
    simplify_after_lower_access: bool = True

    """Hoist loop-invariant computations out of conditional branches."""
    move_loop_invariant_outside_if: bool = False

    # Load-bearing optimization on contact-heavy solves (e.g. duck_in_box). The
    # pass caches a loop-invariant global load into a local, which is only sound
    # when a global's read and write pointers are the same statement. Under
    # per-task CSE that unification is restored by merge_global_ptrs (pre-offload,
    # fields) and cse_offloaded_tasks (post-offload, ndarrays) -- see
    # compile_to_offloads.cpp -- so this pass itself is unchanged from upstream.
    """Cache loop-invariant global loads into locals inside loops."""
    cache_loop_invariant_global_vars: bool = True

    """Lower dense struct-for loops to ordinary range-for loops."""
    demote_dense_struct_fors: bool = True

    """Run the full advanced optimization pass pipeline."""
    advanced_optimization: bool = True

    """Fold constant expressions at compile time."""
    constant_folding: Annotated[bool, NoBind] = True

    """Use the LLVM backend for code generation."""
    use_llvm: bool = True

    """Log a message on every kernel launch."""
    verbose_kernel_launches: bool = False

    """Enable the on-device kernel profiler to collect per-kernel timings."""
    kernel_profiler: bool = False

    """Record a chrome-tracing timeline of compilation and execution."""
    timeline: bool = False

    """Print verbose logging during initialization and compilation."""
    verbose: bool = True

    """Flatten simple if statements into predicated (branchless) form."""
    flatten_if: bool = False

    """Enable the thread-local optimization for reductions."""
    make_thread_local: bool = True

    """Enable the block-local optimization for GPU/mesh accesses."""
    make_block_local: bool = True

    """Detect read-only field accesses to enable further optimization."""
    detect_read_only: bool = True

    """Scalarize matrix/vector operations into per-component scalar ops."""
    real_matrix_scalarize: bool = True

    """Always scalarize matrices, even where vectorized code would be legal."""
    force_scalarize_matrix: bool = False

    """Vectorize pairs of float16 operations into half2 ops (CUDA)."""
    half2_vectorization: bool = False

    """Parallelize outer loops across CPU threads."""
    make_cpu_multithreading_loop: bool = True

    """Default floating-point type for fields and kernels (e.g. qd.f32)."""
    default_fp: Cpp("DataType") = Computed("PrimitiveType::f32", "qd.f32")

    """Default signed-integer type for fields and kernels (e.g. qd.i32)."""
    default_ip: Cpp("DataType") = Computed("PrimitiveType::i32", "qd.i32")

    """Default unsigned-integer type for fields and kernels (e.g. qd.u32)."""
    default_up: Cpp("DataType") = Computed("PrimitiveType::u32", "qd.u32")

    """Number of iterations per CPU parallel-for block."""
    default_cpu_block_dim: int = 32

    """Let the CPU backend choose the parallel-for block size adaptively."""
    cpu_block_dim_adaptive: bool = True

    """Default GPU thread-block size."""
    default_gpu_block_dim: int = 128

    """Maximum registers per GPU thread (0 uses the driver default)."""
    gpu_max_reg: int = 0

    """GPU grid size to launch (0 lets Quadrants pick based on occupancy)."""
    saturating_grid_dim: int = 0

    """Upper bound on GPU block size (0 means no explicit cap)."""
    max_block_dim: int = 0

    """Maximum number of CPU threads Quadrants may use."""
    cpu_max_num_threads: int = Computed("std::thread::hardware_concurrency()", "number of hardware threads")

    """Seed for Quadrants' random-number generation."""
    random_seed: int = 0

    """Print the LLVM IR generated for the data-structure (SNode) module."""
    print_struct_llvm_ir: bool = False

    """Print the LLVM IR generated for each kernel."""
    print_kernel_llvm_ir: bool = False

    """Print each kernel's LLVM IR after LLVM optimization."""
    print_kernel_llvm_ir_optimized: bool = False

    """Print the native assembly generated for each kernel."""
    print_kernel_asm: bool = False

    """Print the AMDGCN assembly generated for each kernel (AMD backend)."""
    print_kernel_amdgcn: bool = False

    """Directory to write IR/asm dumps to when the print_* options are on."""
    debug_dump_path: str = "/tmp/ir/"

    """Amount of GPU memory, in gigabytes, to preallocate."""
    device_memory_GB: float = 1.0

    """Fraction of total GPU memory to preallocate (overrides device_memory_GB
    when greater than 0)."""
    device_memory_fraction: float = 0.0

    """Fuse consecutive stores to quantized (bit-packed) fields."""
    quant_opt_store_fusion: bool = True

    """Demote atomics to plain read-modify-write on quantized fields when
    safe."""
    quant_opt_atomic_demotion: bool = True

    """Enable the block-local optimization for MeshQuadrants attributes."""
    make_mesh_block_local: bool = True

    """Optimize element access through reordered mesh index mappings."""
    optimize_mesh_reordered_mapping: bool = True

    """Cache mesh relation to-end mappings in block-local memory."""
    mesh_localize_to_end_mapping: bool = True

    """Cache mesh relation from-end mappings in block-local memory."""
    mesh_localize_from_end_mapping: bool = False

    """Localize all mesh attribute mappings, not just the ones detected as
    beneficial."""
    mesh_localize_all_attr_mappings: bool = False

    """Demote mesh-for loops that never access mesh attributes to range-fors."""
    demote_no_access_mesh_fors: bool = True

    """Enable the experimental automatic mesh-local optimization."""
    experimental_auto_mesh_local: bool = False

    """Target occupancy used by the automatic mesh-local optimization."""
    auto_mesh_local_default_occupacy: int = 4

    """Eviction policy for the offline cache ("never", "version", "lru", or
    "fifo")."""
    offline_cache_cleaning_policy: str = "lru"

    """Maximum total size, in bytes, of the offline cache before cleaning."""
    offline_cache_max_size_of_files: int = 100 * 1024 * 1024

    """Fraction of the offline cache to evict when it exceeds the size limit."""
    offline_cache_cleaning_factor: float = 0.25

    """Vulkan API version string to request (empty selects the default)."""
    vk_api_version: str = ""

    """Per-thread CUDA stack size limit in bytes (0 uses the driver default)."""
    cuda_stack_limit: Annotated[int, Cpp("std::size_t")] = 0
