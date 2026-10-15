// Bindings for the python frontend

#include <optional>
#include <string>
#include <tuple>
#include "quadrants/ir/snode.h"

#if QD_WITH_LLVM
#include "llvm/Config/llvm-config.h"
#endif

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

#include "quadrants/ir/expression_ops.h"
#include "quadrants/ir/frontend_ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/program/extension.h"
#include "quadrants/program/ndarray.h"
#include "quadrants/rhi/device_capability.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/program/matrix.h"
#include "quadrants/python/export.h"
#include "quadrants/math/svd.h"
#include "quadrants/system/timeline.h"
#include "quadrants/python/snode_registry.h"
#include "quadrants/program/sparse_matrix.h"
#include "quadrants/program/sparse_solver.h"
#include "quadrants/program/conjugate_gradient.h"
#include "quadrants/ir/mesh.h"

#include "quadrants/program/kernel_profiler.h"

#include "quadrants/python/dlpack_funcs.h"

#if defined(QD_WITH_CUDA)
#include "quadrants/rhi/cuda/cuda_context.h"
#endif

namespace quadrants {
bool test_threading();

}  // namespace quadrants

namespace quadrants::lang {

std::string libdevice_path();

}  // namespace quadrants::lang

namespace quadrants {

namespace {
// Tag type used purely to create a Python "InternalOp" namespace-class that exposes each internal op as a
// static property. InternalOp itself is a C++ `enum class` and therefore cannot be bound via nb::class_.
struct InternalOpScope {};
}  // namespace

void export_lang(nb::module_ &m) {
  using namespace quadrants::lang;
  using namespace std::placeholders;

  // nb::exception registers the Python exception type (and its C++->Python translation) as a side
  // effect of construction; the returned handle is intentionally discarded.
  // NOLINTNEXTLINE(bugprone-unused-raii)
  nb::exception<QuadrantsTypeError>(m, "QuadrantsTypeError", PyExc_TypeError);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  nb::exception<QuadrantsSyntaxError>(m, "QuadrantsSyntaxError", PyExc_SyntaxError);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  nb::exception<QuadrantsIndexError>(m, "QuadrantsIndexError", PyExc_IndexError);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  nb::exception<QuadrantsRuntimeError>(m, "QuadrantsRuntimeError", PyExc_RuntimeError);
  // NOLINTNEXTLINE(bugprone-unused-raii)
  nb::exception<QuadrantsAssertionError>(m, "QuadrantsAssertionError", PyExc_AssertionError);
  nb::enum_<Arch>(m, "Arch", nb::is_arithmetic())
#define PER_ARCH(x) .value(#x, Arch::x)
#include "quadrants/inc/archs.inc.h"
#undef PER_ARCH
      .export_values();

  m.def("arch_name", arch_name);
  m.def("arch_from_name", arch_from_name);

  nb::enum_<SNodeType>(m, "SNodeType", nb::is_arithmetic())
#define PER_SNODE(x) .value(#x, SNodeType::x)
#include "quadrants/inc/snodes.inc.h"
#undef PER_SNODE
      .export_values();

  nb::enum_<Extension>(m, "Extension", nb::is_arithmetic())
#define PER_EXTENSION(x) .value(#x, Extension::x)
#include "quadrants/inc/extensions.inc.h"
#undef PER_EXTENSION
      .export_values();

  nb::enum_<DeviceCapability>(m, "DeviceCapability", nb::is_arithmetic())
#define PER_DEVICE_CAPABILITY(x) .value(#x, DeviceCapability::x)
#include "quadrants/inc/rhi_constants.inc.h"
#undef PER_DEVICE_CAPABILITY
      .export_values();

  nb::enum_<ExternalArrayLayout>(m, "Layout", nb::is_arithmetic())
      .value("AOS", ExternalArrayLayout::kAOS)
      .value("SOA", ExternalArrayLayout::kSOA)
      .value("NULL", ExternalArrayLayout::kNull)
      .export_values();

  nb::enum_<AutodiffMode>(m, "AutodiffMode", nb::is_arithmetic())
      .value("NONE", AutodiffMode::kNone)
      .value("VALIDATION", AutodiffMode::kCheckAutodiffValid)
      .value("FORWARD", AutodiffMode::kForward)
      .value("REVERSE", AutodiffMode::kReverse)
      .export_values();

  nb::enum_<SNodeGradType>(m, "SNodeGradType", nb::is_arithmetic())
      .value("PRIMAL", SNodeGradType::kPrimal)
      .value("ADJOINT", SNodeGradType::kAdjoint)
      .value("DUAL", SNodeGradType::kDual)
      .value("ADJOINT_CHECKBIT", SNodeGradType::kAdjointCheckbit)
      .export_values();

  nb::enum_<BoundaryMode>(m, "BoundaryMode", nb::is_arithmetic())
      .value("UNSAFE", BoundaryMode::kUnsafe)
      .value("CLAMP", BoundaryMode::kClamp)
      .export_values();

  // TODO(type): This should be removed
  nb::class_<DataType>(m, "DataTypeCxx")
      .def(nb::init<Type *>())
      .def(nb::self == nb::self)
      .def("__hash__", &DataType::hash)
      .def("to_string", &DataType::to_string)
      .def("__str__", &DataType::to_string)
      .def("shape", &DataType::get_shape)
      .def("element_type", &DataType::get_element_type)
      .def("ptr_removed", &DataType::ptr_removed)
      .def(
          "get_ptr", [](DataType *dtype) -> Type * { return *dtype; }, nb::rv_policy::reference)
      .def("__call__",
           [](DataType *dtype, nb::args args, const nb::kwargs &kwargs) {
             // Defining __call__ here to make DataType callable in Python,
             // which enables us to write `typing.Tuple[qd.i32, qd.i32]`.
             throw QuadrantsSyntaxError(
                 "Quadrants data types cannot be called outside Quadrants "
                 "kernels.");
           })
      .def("__getstate__",
           [](const DataType &dt) {
             // Note: this only works for primitive types, which is fine for now.
             auto primitive = dynamic_cast<const PrimitiveType *>((const Type *)dt);
             QD_ASSERT(primitive);
             return std::make_tuple((std::size_t)primitive->type);
           })
      .def("__setstate__", [](DataType &dt, const std::tuple<std::size_t> &t) {
        new (&dt) DataType(PrimitiveType::get((PrimitiveTypeID)std::get<0>(t)));
      });

  nb::class_<DebugInfo>(m, "DebugInfo")
      .def(nb::init<>())
      .def(nb::init<std::string>())
      .def(nb::init<>())
      .def_rw("tb", &DebugInfo::tb)
      .def_rw("src_loc", &DebugInfo::src_loc);

  nb::class_<CompileConfig>(m, "CompileConfig")
      .def(nb::init<>())
      .def_rw("arch", &CompileConfig::arch,
              "Target backend the kernels run on (e.g. qd.cpu, qd.cuda, qd.vulkan, qd.metal). Defaults to qd.cpu when "
              "arch is not specified. Also settable via the QD_ARCH environment variable; if both are set, this "
              "argument takes precedence and a warning is logged.")
      .def_rw("opt_level", &CompileConfig::opt_level,
              "Quadrants IR optimization level. At 0, IR-level optimizations such as common-subexpression elimination "
              "are disabled; any value above 0 enables them. This is not an LLVM -O level.")
      .def_rw("raise_on_templated_floats", &CompileConfig::raise_on_templated_floats,
              "Raise an error instead of silently specializing a kernel on a Python float it reads as a templated "
              "argument or a captured global variable (each distinct value would otherwise force its own compile).")
      .def_rw("print_ir", &CompileConfig::print_ir,
              "Trace each kernel's IR through compilation by printing it at labeled checkpoints across the pipeline "
              "(for debugging the compiler), rather than as a single dump at the end. Coverage is not exhaustive: some "
              "passes are not instrumented, so a few intermediate stages are not shown.")
      .def_rw("print_preprocessed_ir", &CompileConfig::print_preprocessed_ir,
              "Print each kernel's IR right after frontend preprocessing. Only prints when print_ir is off (print_ir "
              "already shows the initial IR).")
      .def_rw("print_ir_dbg_info", &CompileConfig::print_ir_dbg_info,
              "Include source-line debug info in the print_ir checkpoint output. Has no effect on other IR dumps such "
              "as print_preprocessed_ir.")
      .def_rw("debug", &CompileConfig::debug,
              "Turn on the full suite of correctness checks: field out-of-bounds (implies check_out_of_bound) and "
              "runtime assertions. Considerably slower; intended for development.")
      .def_rw("cfg_optimization", &CompileConfig::cfg_optimization,
              "Run the control-flow-graph optimization pass that simplifies kernel branches and loops. Disabling it "
              "speeds up compilation at a small runtime cost. Only runs when advanced_optimization is on and "
              "opt_level > 0 (both true by default).")
      .def_rw("check_out_of_bound", &CompileConfig::check_out_of_bound,
              "Enable the field out-of-bounds check on tensor indexing without turning on the rest of debug mode. "
              "Reset to off (with a warning) on backends without assertion support, i.e. Vulkan and Metal.")
      .def_rw("print_accessor_ir", &CompileConfig::print_accessor_ir,
              "Also include field accessor kernels in the IR printout. Only has an effect together with print_ir on "
              "the LLVM backends, which otherwise suppress accessor kernels.")
      .def_rw("print_struct_llvm_ir", &CompileConfig::print_struct_llvm_ir,
              "Print the LLVM IR generated for the data-structure (SNode) module.")
      .def_rw("print_kernel_llvm_ir", &CompileConfig::print_kernel_llvm_ir,
              "Print the LLVM IR generated for each kernel.")
      .def_rw("print_kernel_llvm_ir_optimized", &CompileConfig::print_kernel_llvm_ir_optimized,
              "Print each kernel's LLVM IR after LLVM optimization.")
      .def_rw("print_kernel_asm", &CompileConfig::print_kernel_asm,
              "Print the assembly generated for each kernel: native assembly on CPU, PTX (not native SASS) on CUDA.")
      .def_rw("print_kernel_amdgcn", &CompileConfig::print_kernel_amdgcn,
              "Print the AMDGCN assembly generated for each kernel (AMD backend).")
      .def_rw("debug_dump_path", &CompileConfig::debug_dump_path,
              "Directory for IR dumps written and read via the QD_DUMP_IR, QD_DUMP_CFG, and QD_LOAD_IR environment "
              "variables. The print_* options ignore this path: the IR-printing ones (e.g. print_ir) go to stdout, "
              "while the LLVM-IR and assembly ones (e.g. print_kernel_llvm_ir, print_kernel_asm) write files in the "
              "current working directory.")
      .def_rw("move_loop_invariant_outside_if", &CompileConfig::move_loop_invariant_outside_if,
              "Hoist loop-invariant computations out of conditional branches. Runs only within the "
              "advanced_optimization pipeline.")
      .def_rw("cache_loop_invariant_global_vars", &CompileConfig::cache_loop_invariant_global_vars,
              "Cache loop-invariant global loads into locals inside loops.")
      .def_rw("default_cpu_block_dim", &CompileConfig::default_cpu_block_dim,
              "Number of iterations per CPU parallel-for block.")
      .def_rw("default_gpu_block_dim", &CompileConfig::default_gpu_block_dim, "Default GPU thread-block size.")
      .def_rw("saturating_grid_dim", &CompileConfig::saturating_grid_dim,
              "Target GPU grid size (number of blocks) on the CUDA/AMDGPU backends; 0 lets Quadrants pick based on "
              "occupancy. It is an upper bound rather than a guarantee: reverse-mode kernels that carry an autodiff "
              "stack launch with a smaller grid, capped so the concurrent thread count stays within 65536. "
              "Vulkan/Metal compute their dispatch grid automatically and ignore this.")
      .def_rw("max_block_dim", &CompileConfig::max_block_dim,
              "Intended as an upper bound on GPU block size, but currently limits only the block size of internal "
              "list-generation kernels; ordinary range-for/struct-for launches use default_gpu_block_dim instead. 0 "
              "means no cap.")
      .def_rw("cpu_max_num_threads", &CompileConfig::cpu_max_num_threads,
              "Maximum number of CPU threads used to run kernels (the runtime thread pool and CPU parallel-for loops). "
              "Compilation threads are governed separately by num_compile_threads.")
      .def_rw("random_seed", &CompileConfig::random_seed, "Seed for Quadrants' random-number generation.")
      .def_rw("demote_dense_struct_fors", &CompileConfig::demote_dense_struct_fors,
              "Lower dense struct-for loops to ordinary range-for loops. Forced on for the Vulkan/Metal (SPIR-V) "
              "backends, where the value you pass is ignored.")
      .def_rw("kernel_profiler", &CompileConfig::kernel_profiler,
              "Enable the on-device kernel profiler to collect per-kernel timings. Only the CPU, CUDA, and AMDGPU "
              "backends emit the timing hooks; on Vulkan/Metal it produces no per-kernel timings.")
      .def_rw("timeline", &CompileConfig::timeline,
              "Record a chrome-tracing timeline of GPU kernel execution, sourced from the CUDA/AMDGPU kernel profiler "
              "(so kernel_profiler must be enabled). Compilation is not recorded.")
      .def_rw("default_fp", &CompileConfig::default_fp,
              "Default floating-point type for fields and kernels (e.g. qd.f32).")
      .def_rw("default_ip", &CompileConfig::default_ip,
              "Default signed-integer type for fields and kernels (e.g. qd.i32).")
      .def_rw("default_up", &CompileConfig::default_up,
              "Default unsigned-integer type for fields and kernels (e.g. qd.u32).")
      .def_rw("device_memory_GB", &CompileConfig::device_memory_GB,
              "Amount of GPU memory, in gigabytes, to preallocate. Used on the fallback allocator path, and on "
              "pool-supporting CUDA/AMDGPU drivers when the program uses sparse (non-dense) fields; for an all-dense "
              "program on a pool driver it is ignored (allocation is on demand).")
      .def_rw("device_memory_fraction", &CompileConfig::device_memory_fraction,
              "Fraction of total GPU memory to preallocate (overrides device_memory_GB when greater than 0). Same "
              "applicability as device_memory_GB: the fallback allocator path, plus pool-supporting drivers when the "
              "program uses sparse (non-dense) fields.")
      .def_rw("fast_math", &CompileConfig::fast_math,
              "Allow IEEE-relaxed floating-point optimizations (e.g. fused multiply-add). Faster, but drops strict "
              "NaN/inf/signed-zero guarantees.")
      .def_rw("advanced_optimization", &CompileConfig::advanced_optimization,
              "Run the full advanced optimization pass pipeline.")
      .def_rw("flatten_if", &CompileConfig::flatten_if,
              "Flatten simple if statements into predicated (branchless) form.")
      .def_rw("make_thread_local", &CompileConfig::make_thread_local,
              "Enable the thread-local optimization for reductions. Honored on the LLVM backends (CPU, CUDA, AMDGPU) "
              "only; the Vulkan/Metal (SPIR-V) path forces it off.")
      .def_rw("make_block_local", &CompileConfig::make_block_local,
              "Enable the block-local optimization, which stages spatially-local field accesses through GPU shared "
              "memory. CUDA only (the required BLS extension is CUDA-only). Mesh attribute localization is a separate "
              "pass controlled by make_mesh_block_local.")
      .def_rw("detect_read_only", &CompileConfig::detect_read_only,
              "Detect read-only field accesses to enable further optimization.")
      .def_rw("real_matrix_scalarize", &CompileConfig::real_matrix_scalarize,
              "Scalarize matrix/vector operations into per-component scalar ops.")
      .def_rw("force_scalarize_matrix", &CompileConfig::force_scalarize_matrix,
              "Always scalarize matrices, even where vectorized code would be legal.")
      .def_rw("half2_vectorization", &CompileConfig::half2_vectorization,
              "Vectorize pairs of float16 operations into half2 ops. CUDA only, and only when real_matrix_scalarize "
              "is also enabled.")
      .def_rw("make_cpu_multithreading_loop", &CompileConfig::make_cpu_multithreading_loop,
              "Parallelize outer loops across CPU threads.")
      .def_rw("quant_opt_store_fusion", &CompileConfig::quant_opt_store_fusion,
              "Fuse consecutive stores to quantized (bit-packed) fields.")
      .def_rw("quant_opt_atomic_demotion", &CompileConfig::quant_opt_atomic_demotion,
              "Demote atomics to plain read-modify-write on quantized fields when safe.")
      .def_rw("make_mesh_block_local", &CompileConfig::make_mesh_block_local,
              "Enable the block-local optimization for MeshQuadrants attributes. CUDA only (the pass runs only when "
              "arch is CUDA).")
      .def_rw("mesh_localize_to_end_mapping", &CompileConfig::mesh_localize_to_end_mapping,
              "Cache mesh relation to-end mappings in block-local memory. Only used by the CUDA-only "
              "make_mesh_block_local pass.")
      .def_rw("mesh_localize_from_end_mapping", &CompileConfig::mesh_localize_from_end_mapping,
              "Cache mesh relation from-end mappings in block-local memory. Only used by the CUDA-only "
              "make_mesh_block_local pass.")
      .def_rw("optimize_mesh_reordered_mapping", &CompileConfig::optimize_mesh_reordered_mapping,
              "Optimize element access through reordered mesh index mappings.")
      .def_rw("mesh_localize_all_attr_mappings", &CompileConfig::mesh_localize_all_attr_mappings,
              "Localize all mesh attribute mappings, not just the ones detected as beneficial. Used by the CUDA-only "
              "make_mesh_block_local pass, and ignored when experimental_auto_mesh_local is enabled.")
      .def_rw("demote_no_access_mesh_fors", &CompileConfig::demote_no_access_mesh_fors,
              "Demote mesh-for loops that never access mesh attributes to range-fors.")
      .def_rw("experimental_auto_mesh_local", &CompileConfig::experimental_auto_mesh_local,
              "Enable the experimental automatic mesh-local optimization. CUDA only.")
      .def_rw("auto_mesh_local_default_occupacy", &CompileConfig::auto_mesh_local_default_occupacy,
              "Target occupancy used by the automatic mesh-local optimization. Only used on CUDA when "
              "experimental_auto_mesh_local is enabled.")
      .def_rw("offline_cache", &CompileConfig::offline_cache,
              "Whether compiled-kernel caches persist on disk and are reused across separate Python processes.")
      .def_rw("offline_cache_file_path", &CompileConfig::offline_cache_file_path,
              "Directory that holds the on-disk compilation cache.")
      .def_rw("offline_cache_cleaning_policy", &CompileConfig::offline_cache_cleaning_policy,
              "Eviction policy for the compiled-kernel offline cache (\"never\", \"version\", \"lru\", or \"fifo\").")
      .def_rw("offline_cache_max_size_of_files", &CompileConfig::offline_cache_max_size_of_files,
              "Maximum total size, in bytes, of the compiled-kernel offline cache before cleaning. This bounds only "
              "that cache, not the CUDA PTX cache or the source-level fastcache, so the on-disk cache directory can "
              "grow beyond it.")
      .def_rw("offline_cache_cleaning_factor", &CompileConfig::offline_cache_cleaning_factor,
              "Fraction of the compiled-kernel offline cache to evict once it exceeds the size limit. Applied only "
              "under the \"lru\" and \"fifo\" cleaning policies; \"version\" does no size-based eviction.")
      .def_rw("num_compile_threads", &CompileConfig::num_compile_threads,
              "Number of host threads used to compile kernels on the LLVM backends (CPU, CUDA, AMDGPU); other backends "
              "ignore it, and it is forced to 1 when print_ir is set.")
      .def_rw("cuda_stack_limit", &CompileConfig::cuda_stack_limit,
              "Per-thread CUDA stack size limit in bytes (0 uses the driver default).")
      .def_rw("external_metal_command_queue", &CompileConfig::external_metal_command_queue,
              "An MTLCommandQueue pointer (as an integer) to dispatch on instead of creating a new Metal queue. 0 "
              "means create a new queue.")
      .def_rw("external_metal_command_queue_is_torch_queue",
              &CompileConfig::external_metal_command_queue_is_torch_queue,
              "Set True when external_metal_command_queue is PyTorch MPS's own queue, to skip redundant interop "
              "synchronization.");

  m.def("reset_default_compile_config", [&]() { default_compile_config = CompileConfig(); });

  m.def(
      "default_compile_config", [&]() -> CompileConfig & { return default_compile_config; }, nb::rv_policy::reference);

  nb::class_<Program::KernelProfilerQueryResult>(m, "KernelProfilerQueryResult")
      .def_rw("counter", &Program::KernelProfilerQueryResult::counter)
      .def_rw("min", &Program::KernelProfilerQueryResult::min)
      .def_rw("max", &Program::KernelProfilerQueryResult::max)
      .def_rw("avg", &Program::KernelProfilerQueryResult::avg);

  nb::class_<KernelProfileTracedRecord>(m, "KernelProfileTracedRecord")
      .def_rw("register_per_thread", &KernelProfileTracedRecord::register_per_thread)
      .def_rw("shared_mem_per_block", &KernelProfileTracedRecord::shared_mem_per_block)
      .def_rw("grid_size", &KernelProfileTracedRecord::grid_size)
      .def_rw("block_size", &KernelProfileTracedRecord::block_size)
      .def_rw("active_blocks_per_multiprocessor", &KernelProfileTracedRecord::active_blocks_per_multiprocessor)
      .def_rw("kernel_time", &KernelProfileTracedRecord::kernel_elapsed_time_in_ms)
      .def_rw("base_time", &KernelProfileTracedRecord::time_since_base)
      .def_rw("name", &KernelProfileTracedRecord::name)
      .def_rw("metric_values", &KernelProfileTracedRecord::metric_values);

  nb::enum_<SNodeAccessFlag>(m, "SNodeAccessFlag", nb::is_arithmetic())
      .value("block_local", SNodeAccessFlag::block_local)
      .value("read_only", SNodeAccessFlag::read_only)
      .value("mesh_local", SNodeAccessFlag::mesh_local)
      .export_values();

  // Export ASTBuilder
  nb::class_<ASTBuilder>(m, "ASTBuilder")
      .def("make_id_expr", &ASTBuilder::make_id_expr)
      .def("create_kernel_exprgroup_return", &ASTBuilder::create_kernel_exprgroup_return)
      .def("create_print", &ASTBuilder::create_print)
      .def("begin_func", &ASTBuilder::begin_func)
      .def("end_func", &ASTBuilder::end_func)
      .def("stop_grad", &ASTBuilder::stop_gradient)
      .def("begin_frontend_if", &ASTBuilder::begin_frontend_if)
      .def("begin_frontend_if_true", &ASTBuilder::begin_frontend_if_true)
      .def("pop_scope", &ASTBuilder::pop_scope)
      .def("begin_frontend_if_false", &ASTBuilder::begin_frontend_if_false)
      .def("insert_deactivate", &ASTBuilder::insert_snode_deactivate)
      .def("insert_activate", &ASTBuilder::insert_snode_activate)
      .def("expr_snode_get_addr", &ASTBuilder::snode_get_addr)
      .def("expr_snode_append", &ASTBuilder::snode_append)
      .def("expr_snode_is_active", &ASTBuilder::snode_is_active)
      .def("expr_snode_length", &ASTBuilder::snode_length)
      .def("insert_external_func_call", &ASTBuilder::insert_external_func_call)
      .def("make_matrix_expr", &ASTBuilder::make_matrix_expr)
      .def("expr_alloca", &ASTBuilder::expr_alloca)
      .def("expr_alloca_shared_array", &ASTBuilder::expr_alloca_shared_array)
      .def("create_assert_stmt", &ASTBuilder::create_assert_stmt)
      .def("expr_assign", &ASTBuilder::expr_assign)
      .def("set_loop_name", &ASTBuilder::set_loop_name)
      .def("begin_frontend_range_for", &ASTBuilder::begin_frontend_range_for)
      .def("end_frontend_range_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_struct_for_on_snode", &ASTBuilder::begin_frontend_struct_for_on_snode)
      .def("begin_frontend_struct_for_on_external_tensor", &ASTBuilder::begin_frontend_struct_for_on_external_tensor)
      .def("end_frontend_struct_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_mesh_for", &ASTBuilder::begin_frontend_mesh_for)
      .def("end_frontend_mesh_for", &ASTBuilder::pop_scope)
      .def("begin_frontend_while", &ASTBuilder::begin_frontend_while)
      .def("insert_break_stmt", &ASTBuilder::insert_break_stmt)
      .def("insert_continue_stmt", &ASTBuilder::insert_continue_stmt)
      .def("insert_expr_stmt", &ASTBuilder::insert_expr_stmt)
      .def("insert_thread_idx_expr", &ASTBuilder::insert_thread_idx_expr)
      .def("insert_patch_idx_expr", &ASTBuilder::insert_patch_idx_expr)
      .def("expand_exprs", &ASTBuilder::expand_exprs)
      .def("mesh_index_conversion", &ASTBuilder::mesh_index_conversion)
      .def("expr_subscript", &ASTBuilder::expr_subscript)
      .def("insert_func_call", &ASTBuilder::insert_func_call)
      .def("sifakis_svd_f32", sifakis_svd_export<float32, int32>)
      .def("sifakis_svd_f64", sifakis_svd_export<float64, int64>)
      .def("expr_var", &ASTBuilder::make_var)
      .def("bit_vectorize", &ASTBuilder::bit_vectorize)
      .def("parallelize", &ASTBuilder::parallelize)
      .def("strictly_serialize", &ASTBuilder::strictly_serialize)
      .def("block_dim", &ASTBuilder::block_dim)
      .def("insert_snode_access_flag", &ASTBuilder::insert_snode_access_flag)
      .def("reset_snode_access_flag", &ASTBuilder::reset_snode_access_flag)
      .def("begin_stream_parallel", &ASTBuilder::begin_stream_parallel)
      .def("end_stream_parallel", &ASTBuilder::end_stream_parallel)
      .def("begin_graph_parallel_context", &ASTBuilder::begin_graph_parallel_context)
      .def("end_graph_parallel_context", &ASTBuilder::end_graph_parallel_context)
      .def("set_graph_do_while_level_id", &ASTBuilder::set_graph_do_while_level_id)
      .def("begin_checkpoint", &ASTBuilder::begin_checkpoint)
      .def("end_checkpoint", &ASTBuilder::end_checkpoint);

  auto device_capability_config = nb::class_<DeviceCapabilityConfig>(m, "DeviceCapabilityConfig")
                                      .def("get", &DeviceCapabilityConfig::get)
                                      .def("hashed_key", [](const DeviceCapabilityConfig &caps) {
                                        return get_hashed_offline_cache_key_of_device_caps(caps);
                                      });

  auto compiled_kernel_data = nb::class_<CompiledKernelData>(m, "CompiledKernelData")
                                  .def("_debug_dump_to_string", &CompiledKernelData::debug_dump_to_string);

  // nanobind types are not weak-referenceable by default (pybind11 made all bound types so). The Python
  // frontend holds weakrefs to the Program (kernel.py / stream.py), so opt in explicitly.
  auto program_class = nb::class_<Program>(m, "Program", nb::is_weak_referenceable());
  program_class.def(nb::init<>())
      .def(
          "ndarray_to_dlpack",
          [](Program *program, nb::object owner, Ndarray *ndarray, const std::vector<int> &layout, bool versioned) {
            return ndarray_to_dlpack(program, owner, ndarray, layout, versioned);
          },
          nb::arg("owner"), nb::arg("ndarray"), nb::arg("layout") = std::vector<int>{}, nb::arg("versioned") = false)
      .def(
          "field_to_dlpack",
          [](Program *program, SNode *snode, int element_ndim, int n, int m, bool versioned) {
            return field_to_dlpack(program, snode, element_ndim, n, m, versioned);
          },
          nb::arg("snode"), nb::arg("element_ndim"), nb::arg("n"), nb::arg("m"), nb::arg("versioned") = false)
      .def("_get_num_ndarrays", &Program::get_num_ndarrays)
      .def("config", &Program::compile_config, nb::rv_policy::reference)
      .def("sync_kernel_profiler", [](Program *program) { program->profiler->sync(); })
      .def("update_kernel_profiler", [](Program *program) { program->profiler->update(); })
      .def("clear_kernel_profiler", [](Program *program) { program->profiler->clear(); })
      .def("query_kernel_profile_info",
           [](Program *program, const std::string &name) { return program->query_kernel_profile_info(name); })
      .def("get_kernel_profiler_records", [](Program *program) { return program->profiler->get_traced_records(); })
      .def("get_kernel_profiler_device_name", [](Program *program) { return program->profiler->get_device_name(); })
      .def("reinit_kernel_profiler_with_metrics",
           [](Program *program, const std::vector<std::string> metrics) {
             return program->profiler->reinit_with_metrics(metrics);
           })
      .def("kernel_profiler_total_time", [](Program *program) { return program->profiler->get_total_time(); })
      .def("set_kernel_profiler_toolkit",
           [](Program *program, const std::string toolkit_name) {
             return program->profiler->set_profiler_toolkit(toolkit_name);
           })
      .def("timeline_clear", [](Program *) { Timelines::get_instance().clear(); })
      .def("timeline_save", [](Program *, const std::string &fn) { Timelines::get_instance().save(fn); })
      .def("print_memory_profiler_info", &Program::print_memory_profiler_info)
      .def("finalize", &Program::finalize)
      .def("get_total_compilation_time", &Program::get_total_compilation_time)
      .def("get_snode_num_dynamically_allocated", &Program::get_snode_num_dynamically_allocated)
      .def("synchronize", &Program::synchronize_and_assert)
      .def("materialize_runtime", &Program::materialize_runtime)
      .def("get_snode_tree_size", &Program::get_snode_tree_size)
      .def("get_snode_root", &Program::get_snode_root, nb::rv_policy::reference)
      .def("load_fast_cache", &Program::load_fast_cache, nb::rv_policy::reference)
      .def("dump_cache_data_to_disk", &Program::dump_cache_data_to_disk)
      .def(
          "create_kernel",
          [](Program *program, const std::function<void(Kernel *)> &body, const std::string &name,
             AutodiffMode autodiff_mode) -> Kernel * {
            nb::gil_scoped_release release;
            return &program->create_kernel(body, name, autodiff_mode);
          },
          nb::arg("body"), nb::arg("name"), nb::arg("autodiff_mode"), nb::rv_policy::reference)
      .def("create_function", &Program::create_function, nb::arg("func_key"), nb::rv_policy::reference)
      .def("create_sparse_matrix",
           [](Program *program, int n, int m, DataType dtype, std::string storage_format) {
             QD_ERROR_IF(!arch_is_cpu(program->compile_config().arch) && !arch_is_cuda(program->compile_config().arch),
                         "SparseMatrix only supports CPU and CUDA for now.");
             if (arch_is_cpu(program->compile_config().arch))
               return make_sparse_matrix(n, m, dtype, storage_format);
             else
               return make_cu_sparse_matrix(n, m, dtype);
           })
      .def("make_sparse_matrix_from_ndarray",
           [](Program *program, SparseMatrix &sm, const Ndarray &ndarray) {
             QD_ERROR_IF(!arch_is_cpu(program->compile_config().arch) && !arch_is_cuda(program->compile_config().arch),
                         "SparseMatrix only supports CPU and CUDA for now.");
             return make_sparse_matrix_from_ndarray(program, sm, ndarray);
           })
      .def("make_id_expr",
           [](Program *program, const std::string &name) {
             return Expr::make<IdExpression>(program->get_next_global_id(name));
           })
      .def(
          "create_ndarray",
          [&](Program *program, const DataType &dt, const std::vector<int> &shape, ExternalArrayLayout layout,
              bool zero_fill, DebugInfo dbg_info) -> Ndarray * {
            return program->create_ndarray(dt, shape, layout, zero_fill, dbg_info);
          },
          nb::arg("dt"), nb::arg("shape"), nb::arg("layout") = ExternalArrayLayout::kNull, nb::arg("zero_fill") = false,
          nb::arg("dbg_info") = DebugInfo(), nb::rv_policy::reference)
      .def("delete_ndarray", &Program::delete_ndarray)
      .def("get_ndarray_data_ptr_as_int",
           [](Program *program, Ndarray *ndarray) { return program->get_ndarray_data_ptr_as_int(ndarray); })
      .def("fill_float", [](Program *program, Ndarray *ndarray,
                            float val) { program->fill_ndarray_fast_u32(ndarray, reinterpret_cast<uint32_t &>(val)); })
      .def("fill_int", [](Program *program, Ndarray *ndarray,
                          int32_t val) { program->fill_ndarray_fast_u32(ndarray, reinterpret_cast<int32_t &>(val)); })
      .def("fill_uint",
           [](Program *program, Ndarray *ndarray, uint32_t val) { program->fill_ndarray_fast_u32(ndarray, val); })
      .def("get_graphics_device", [](Program *program) { return program->get_graphics_device(); })
      .def("compile_kernel", &Program::compile_kernel, nb::rv_policy::reference)
      .def("launch_kernel", &Program::launch_kernel)
      .def("get_device_caps", &Program::get_device_caps)
      .def("subgroup_size", &Program::subgroup_size)
      .def("get_graph_cache_size", &Program::get_graph_cache_size)
      .def("get_graph_cache_used_on_last_call", &Program::get_graph_cache_used_on_last_call)
      .def("get_num_offloaded_tasks_on_last_call", &Program::get_num_offloaded_tasks_on_last_call)
      .def("get_graph_num_nodes_on_last_call", &Program::get_graph_num_nodes_on_last_call)
      .def("get_graph_num_checkpoints_on_last_call", &Program::get_graph_num_checkpoints_on_last_call)
      .def("get_graph_last_yield_cp_id_on_last_call", &Program::get_graph_last_yield_cp_id_on_last_call)
      .def("get_graph_total_builds", &Program::get_graph_total_builds);
}

}  // namespace quadrants
