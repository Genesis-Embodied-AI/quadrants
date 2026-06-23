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
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/extension.h"
#include "quadrants/program/ndarray.h"
#include "quadrants/rhi/device_capability.h"
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

  nb::exception<QuadrantsTypeError>(m, "QuadrantsTypeError", PyExc_TypeError);
  nb::exception<QuadrantsSyntaxError>(m, "QuadrantsSyntaxError", PyExc_SyntaxError);
  nb::exception<QuadrantsIndexError>(m, "QuadrantsIndexError", PyExc_IndexError);
  nb::exception<QuadrantsRuntimeError>(m, "QuadrantsRuntimeError", PyExc_RuntimeError);
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
             // which enables us to write `typing.Tuple[ti.i32, ti.i32]`.
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
      .def_rw("arch", &CompileConfig::arch)
      .def_rw("opt_level", &CompileConfig::opt_level)
      .def_rw("raise_on_templated_floats", &CompileConfig::raise_on_templated_floats)
      .def_rw("print_ir", &CompileConfig::print_ir)
      .def_rw("print_preprocessed_ir", &CompileConfig::print_preprocessed_ir)
      .def_rw("print_ir_dbg_info", &CompileConfig::print_ir_dbg_info)
      .def_rw("debug", &CompileConfig::debug)
      .def_rw("cfg_optimization", &CompileConfig::cfg_optimization)
      .def_rw("check_out_of_bound", &CompileConfig::check_out_of_bound)
      .def_rw("print_accessor_ir", &CompileConfig::print_accessor_ir)
      .def_rw("use_llvm", &CompileConfig::use_llvm)
      .def_rw("print_struct_llvm_ir", &CompileConfig::print_struct_llvm_ir)
      .def_rw("print_kernel_llvm_ir", &CompileConfig::print_kernel_llvm_ir)
      .def_rw("print_kernel_llvm_ir_optimized", &CompileConfig::print_kernel_llvm_ir_optimized)
      .def_rw("print_kernel_asm", &CompileConfig::print_kernel_asm)
      .def_rw("print_kernel_amdgcn", &CompileConfig::print_kernel_amdgcn)
      .def_rw("debug_dump_path", &CompileConfig::debug_dump_path)
      .def_rw("simplify_before_lower_access", &CompileConfig::simplify_before_lower_access)
      .def_rw("simplify_after_lower_access", &CompileConfig::simplify_after_lower_access)
      .def_rw("lower_access", &CompileConfig::lower_access)
      .def_rw("move_loop_invariant_outside_if", &CompileConfig::move_loop_invariant_outside_if)
      .def_rw("cache_loop_invariant_global_vars", &CompileConfig::cache_loop_invariant_global_vars)
      .def_rw("default_cpu_block_dim", &CompileConfig::default_cpu_block_dim)
      .def_rw("cpu_block_dim_adaptive", &CompileConfig::cpu_block_dim_adaptive)
      .def_rw("default_gpu_block_dim", &CompileConfig::default_gpu_block_dim)
      .def_rw("gpu_max_reg", &CompileConfig::gpu_max_reg)
      .def_rw("saturating_grid_dim", &CompileConfig::saturating_grid_dim)
      .def_rw("max_block_dim", &CompileConfig::max_block_dim)
      .def_rw("cpu_max_num_threads", &CompileConfig::cpu_max_num_threads)
      .def_rw("random_seed", &CompileConfig::random_seed)
      .def_rw("verbose_kernel_launches", &CompileConfig::verbose_kernel_launches)
      .def_rw("verbose", &CompileConfig::verbose)
      .def_rw("demote_dense_struct_fors", &CompileConfig::demote_dense_struct_fors)
      .def_rw("kernel_profiler", &CompileConfig::kernel_profiler)
      .def_rw("timeline", &CompileConfig::timeline)
      .def_rw("default_fp", &CompileConfig::default_fp)
      .def_rw("default_ip", &CompileConfig::default_ip)
      .def_rw("default_up", &CompileConfig::default_up)
      .def_rw("device_memory_GB", &CompileConfig::device_memory_GB)
      .def_rw("device_memory_fraction", &CompileConfig::device_memory_fraction)
      .def_rw("fast_math", &CompileConfig::fast_math)
      .def_rw("advanced_optimization", &CompileConfig::advanced_optimization)
      .def_rw("ad_stack_experimental_enabled", &CompileConfig::ad_stack_experimental_enabled)
      .def_rw("ad_stack_size", &CompileConfig::ad_stack_size)
      .def_rw("ad_stack_sparse_threshold_bytes", &CompileConfig::ad_stack_sparse_threshold_bytes)
      .def_rw("flatten_if", &CompileConfig::flatten_if)
      .def_rw("make_thread_local", &CompileConfig::make_thread_local)
      .def_rw("make_block_local", &CompileConfig::make_block_local)
      .def_rw("detect_read_only", &CompileConfig::detect_read_only)
      .def_rw("real_matrix_scalarize", &CompileConfig::real_matrix_scalarize)
      .def_rw("force_scalarize_matrix", &CompileConfig::force_scalarize_matrix)
      .def_rw("half2_vectorization", &CompileConfig::half2_vectorization)
      .def_rw("make_cpu_multithreading_loop", &CompileConfig::make_cpu_multithreading_loop)
      .def_rw("quant_opt_store_fusion", &CompileConfig::quant_opt_store_fusion)
      .def_rw("quant_opt_atomic_demotion", &CompileConfig::quant_opt_atomic_demotion)
      .def_rw("make_mesh_block_local", &CompileConfig::make_mesh_block_local)
      .def_rw("mesh_localize_to_end_mapping", &CompileConfig::mesh_localize_to_end_mapping)
      .def_rw("mesh_localize_from_end_mapping", &CompileConfig::mesh_localize_from_end_mapping)
      .def_rw("optimize_mesh_reordered_mapping", &CompileConfig::optimize_mesh_reordered_mapping)
      .def_rw("mesh_localize_all_attr_mappings", &CompileConfig::mesh_localize_all_attr_mappings)
      .def_rw("demote_no_access_mesh_fors", &CompileConfig::demote_no_access_mesh_fors)
      .def_rw("experimental_auto_mesh_local", &CompileConfig::experimental_auto_mesh_local)
      .def_rw("auto_mesh_local_default_occupacy", &CompileConfig::auto_mesh_local_default_occupacy)
      .def_rw("offline_cache", &CompileConfig::offline_cache)
      .def_rw("offline_cache_file_path", &CompileConfig::offline_cache_file_path)
      .def_rw("offline_cache_cleaning_policy", &CompileConfig::offline_cache_cleaning_policy)
      .def_rw("offline_cache_max_size_of_files", &CompileConfig::offline_cache_max_size_of_files)
      .def_rw("offline_cache_cleaning_factor", &CompileConfig::offline_cache_cleaning_factor)
      .def_rw("num_compile_threads", &CompileConfig::num_compile_threads)
      .def_rw("vk_api_version", &CompileConfig::vk_api_version)
      .def_rw("cuda_stack_limit", &CompileConfig::cuda_stack_limit)
      .def_rw("external_metal_command_queue", &CompileConfig::external_metal_command_queue)
      .def_rw("external_metal_command_queue_is_torch_queue",
                     &CompileConfig::external_metal_command_queue_is_torch_queue);

  m.def("reset_default_compile_config", [&]() { default_compile_config = CompileConfig(); });

  m.def(
      "default_compile_config", [&]() -> CompileConfig & { return default_compile_config; },
      nb::rv_policy::reference);

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
      .def("set_graph_do_while_level_id", &ASTBuilder::set_graph_do_while_level_id)
      .def("begin_checkpoint", &ASTBuilder::begin_checkpoint)
      .def("end_checkpoint", &ASTBuilder::end_checkpoint);

  auto device_capability_config =
      nb::class_<DeviceCapabilityConfig>(m, "DeviceCapabilityConfig").def("get", &DeviceCapabilityConfig::get);

  auto compiled_kernel_data = nb::class_<CompiledKernelData>(m, "CompiledKernelData")
                                  .def("_debug_dump_to_string", &CompiledKernelData::debug_dump_to_string);

  // nanobind types are not weak-referenceable by default (pybind11 made all bound types so). The Python
  // frontend holds weakrefs to the Program (kernel.py / stream.py), so opt in explicitly.
  auto program_class = nb::class_<Program>(m, "Program", nb::is_weak_referenceable());
  program_class.def(nb::init<>())
      .def(
          "ndarray_to_dlpack",
          [](Program *program, nb::object owner, Ndarray *ndarray, const std::vector<int> &layout,
             bool versioned) { return ndarray_to_dlpack(program, owner, ndarray, layout, versioned); },
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
          nb::rv_policy::reference)
      .def("create_function", &Program::create_function, nb::rv_policy::reference)
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
      .def("get_graph_total_builds", &Program::get_graph_total_builds)
      // Test-only introspection on the max-reducer dispatch counter. Leading underscore signals "internal, not part of
      // the public Python API"; quadrants tests reach these via `impl.get_runtime().prog`. They are intentionally not
      // surfaced on the user-facing `qd.*` namespace and not documented under `docs/`.
      .def("_get_max_reducer_dispatch_count",
           [](Program *program) { return program->adstack_cache().max_reducer_dispatch_count(); })
      .def("_reset_max_reducer_dispatch_count",
           [](Program *program) { program->adstack_cache().reset_max_reducer_dispatch_count(); });
  export_stream(m, program_class);

  nb::class_<CompileResult>(m, "CompileResult")
      .def_prop_ro(
          "compiled_kernel_data",
          [](const CompileResult &self) -> const CompiledKernelData & { return self.compiled_kernel_data; })
      .def_ro("cache_hit", &CompileResult::cache_hit)
      .def_ro("cache_key", &CompileResult::cache_key);

  nb::class_<Axis>(m, "Axis").def(nb::init<int>());
  nb::class_<SNode>(m, "SNodeCxx")
      .def(nb::init<>())
      .def_rw("parent", &SNode::parent)
      .def_ro("type", &SNode::type)
      .def_ro("id", &SNode::id)
      .def("get_snode_tree_id", &SNode::get_snode_tree_id)
      .def_ro("offset", &SNode::index_offsets)
      .def("dense",
           (SNode & (SNode::*)(const std::vector<Axis> &, const std::vector<int> &, const DebugInfo &))(&SNode::dense),
           nb::rv_policy::reference)
      .def(
          "pointer",
          (SNode & (SNode::*)(const std::vector<Axis> &, const std::vector<int> &, const DebugInfo &))(&SNode::pointer),
          nb::rv_policy::reference)
      .def("hash",
           (SNode & (SNode::*)(const std::vector<Axis> &, const std::vector<int> &, const DebugInfo &))(&SNode::hash),
           nb::rv_policy::reference)
      .def("dynamic", &SNode::dynamic, nb::rv_policy::reference)
      .def("bitmasked",
           (SNode &
            (SNode::*)(const std::vector<Axis> &, const std::vector<int> &, const DebugInfo &))(&SNode::bitmasked),
           nb::rv_policy::reference)
      .def("bit_struct", &SNode::bit_struct, nb::rv_policy::reference)
      .def("quant_array", &SNode::quant_array, nb::rv_policy::reference)
      .def("place", &SNode::place)
      .def("data_type", [](SNode *snode) { return snode->dt; })
      .def("name", [](SNode *snode) { return snode->name; })
      .def("get_num_ch", [](SNode *snode) -> int { return (int)snode->ch.size(); })
      .def(
          "get_ch", [](SNode *snode, int i) -> SNode * { return snode->ch[i].get(); },
          nb::rv_policy::reference)
      .def("lazy_grad", &SNode::lazy_grad)
      .def("lazy_dual", &SNode::lazy_dual)
      .def("allocate_adjoint_checkbit", &SNode::allocate_adjoint_checkbit)
      .def("read_int", &SNode::read_int)
      .def("read_uint", &SNode::read_uint)
      .def("read_float", &SNode::read_float)
      .def("has_adjoint", &SNode::has_adjoint)
      .def("has_adjoint_checkbit", &SNode::has_adjoint_checkbit)
      .def("get_snode_grad_type", &SNode::get_snode_grad_type)
      .def("has_dual", &SNode::has_dual)
      .def("is_primal", &SNode::is_primal)
      .def("is_place", &SNode::is_place)
      .def("get_expr", &SNode::get_expr)
      .def("write_int", &SNode::write_int)
      .def("write_uint", &SNode::write_uint)
      .def("write_float", &SNode::write_float)
      .def("get_shape_along_axis", &SNode::shape_along_axis)
      .def("get_physical_index_position",
           [](SNode *snode) {
             return std::vector<int>(snode->physical_index_position,
                                     snode->physical_index_position + quadrants_max_num_indices);
           })
      .def("num_active_indices", [](SNode *snode) { return snode->num_active_indices; })
      .def_ro("cell_size_bytes", &SNode::cell_size_bytes)
      .def_ro("offset_bytes_in_parent_cell", &SNode::offset_bytes_in_parent_cell);

  nb::class_<SNodeTree>(m, "SNodeTreeCxx")
      .def("id", &SNodeTree::id)
      .def("destroy_snode_tree",
           [](SNodeTree *snode_tree, Program *program) { program->destroy_snode_tree(snode_tree); });

  nb::class_<DeviceAllocation>(m, "DeviceAllocation")
      .def(
          "__init__",
          [](DeviceAllocation *self, uint64_t device, uint64_t alloc_id) {
            new (self) DeviceAllocation();
            self->device = (Device *)device;
            self->alloc_id = (DeviceAllocationId)alloc_id;
          },
          nb::arg("device"), nb::arg("alloc_id"))
      .def_ro("device", &DeviceAllocation::device)
      .def_ro("alloc_id", &DeviceAllocation::alloc_id);

  // The frontend holds weakrefs to the underlying Ndarray (kernel launch-context cache eviction in
  // kernel.py); nanobind types are not weak-referenceable by default, so opt in (pybind11 default).
  nb::class_<Ndarray>(m, "NdarrayCxx", nb::is_weak_referenceable())
      .def("device_allocation_ptr", &Ndarray::get_device_allocation_ptr_as_int)
      .def("device_allocation", &Ndarray::get_device_allocation)
      .def("element_size", &Ndarray::get_element_size)
      .def("nelement", &Ndarray::get_nelement)
      .def("read_int", &Ndarray::read_int)
      .def("read_uint", &Ndarray::read_uint)
      .def("read_float", &Ndarray::read_float)
      .def("write_int", &Ndarray::write_int)
      .def("write_float", &Ndarray::write_float)
      .def("total_shape", &Ndarray::total_shape)
      .def("element_shape", &Ndarray::get_element_shape)
      .def("element_data_type", &Ndarray::get_element_data_type)
      .def_ro("dtype", &Ndarray::dtype)
      .def_ro("shape", &Ndarray::shape);

  nb::enum_<BufferFormat>(m, "Format")
#define PER_BUFFER_FORMAT(x) .value(#x, BufferFormat::x)
#include "quadrants/inc/rhi_constants.inc.h"
#undef PER_EXTENSION
      ;

  nb::class_<Kernel>(m, "KernelCxx")
      .def("no_activate",
           [](Kernel *self, SNode *snode) {
             // TODO(#2193): Also apply to @ti.func?
             self->no_activate.push_back(snode);
           })
      .def("to_string", &Kernel::to_string)
      .def("insert_scalar_param", &Kernel::insert_scalar_param)
      .def("insert_arr_param", &Kernel::insert_arr_param)
      .def("insert_ndarray_param", &Kernel::insert_ndarray_param)
      .def("insert_pointer_param", &Kernel::insert_pointer_param)
      .def("insert_ret", &Kernel::insert_ret)
      .def("finalize_rets", &Kernel::finalize_rets)
      .def("finalize_params", &Kernel::finalize_params)
      .def("make_launch_context", &Kernel::make_launch_context)
      .def(
          "ast_builder", [](Kernel *self) -> ASTBuilder * { return &self->context->builder(); },
          nb::rv_policy::reference);

  nb::class_<LaunchContextBuilder>(m, "KernelLaunchContext")
      .def("copy", &LaunchContextBuilder::copy)
      .def("set_arg_int", &LaunchContextBuilder::set_arg_int)
      .def("set_args_int", &LaunchContextBuilder::set_args_int)
      .def("set_arg_uint", &LaunchContextBuilder::set_arg_uint)
      .def("set_args_uint", &LaunchContextBuilder::set_args_uint)
      .def("set_arg_float", &LaunchContextBuilder::set_arg_float)
      .def("set_args_float", &LaunchContextBuilder::set_args_float)
      .def("set_struct_arg_int", &LaunchContextBuilder::set_struct_arg<int64>)
      .def("set_struct_arg_uint", &LaunchContextBuilder::set_struct_arg<uint64>)
      .def("set_struct_arg_float", &LaunchContextBuilder::set_struct_arg<double>)
      .def("set_arg_external_array_with_shape", &LaunchContextBuilder::set_arg_external_array_with_shape)
      .def("set_arg_ndarray", &LaunchContextBuilder::set_arg_ndarray)
      .def("set_args_ndarray", &LaunchContextBuilder::set_args_ndarray)
      .def("set_arg_ndarray_with_grad", &LaunchContextBuilder::set_arg_ndarray_with_grad)
      .def("set_args_ndarray_with_grad", &LaunchContextBuilder::set_args_ndarray_with_grad)
      .def("get_struct_ret_int", &LaunchContextBuilder::get_struct_ret_int)
      .def("get_struct_ret_uint", &LaunchContextBuilder::get_struct_ret_uint)
      .def("get_struct_ret_float", &LaunchContextBuilder::get_struct_ret_float)
      .def_rw("use_graph", &LaunchContextBuilder::use_graph)
      .def("add_graph_do_while_level", &LaunchContextBuilder::add_graph_do_while_level)
      .def_rw("checkpoint_yield_on_arg_ids", &LaunchContextBuilder::checkpoint_yield_on_arg_ids)
      .def_rw("resume_from_checkpoint", &LaunchContextBuilder::resume_from_checkpoint);

  nb::class_<Function>(m, "Function")
      .def("insert_scalar_param", &Function::insert_scalar_param)
      .def("insert_arr_param", &Function::insert_arr_param)
      .def("insert_ndarray_param", &Function::insert_ndarray_param)
      .def("insert_pointer_param", &Function::insert_pointer_param)
      .def("insert_ret", &Function::insert_ret)
      .def("set_function_body",
           static_cast<void (Function::*)(const std::function<void()> &)>(&Function::set_function_body))
      .def("finalize_rets", &Function::finalize_rets)
      .def("finalize_params", &Function::finalize_params)
      .def(
          "ast_builder", [](Function *self) -> ASTBuilder * { return &self->context->builder(); },
          nb::rv_policy::reference);

  nb::class_<Expr> expr(m, "ExprCxx");
  expr.def("snode", &Expr::snode, nb::rv_policy::reference)
      .def("is_external_tensor_expr", [](Expr *expr) { return expr->is<ExternalTensorExpression>(); })
      .def("is_index_expr", [](Expr *expr) { return expr->is<IndexExpression>(); })
      .def("is_primal",
           [](Expr *expr) { return expr->cast<FieldExpression>()->snode_grad_type == SNodeGradType::kPrimal; })
      .def("is_lvalue", [](Expr *expr) { return expr->expr->is_lvalue(); })
      .def("set_dbg_info", &Expr::set_dbg_info)
      .def("get_dbg_info", [](Expr *expr) { return expr->expr->dbg_info; })
      .def("set_name", [&](Expr *expr, std::string na) { expr->cast<FieldExpression>()->name = na; })
      .def("set_grad_type", [&](Expr *expr, SNodeGradType t) { expr->cast<FieldExpression>()->snode_grad_type = t; })
      .def("set_adjoint", &Expr::set_adjoint)
      .def("set_adjoint_checkbit", &Expr::set_adjoint_checkbit)
      .def("set_dual", &Expr::set_dual)
      .def("set_dynamic_index_stride",
           [&](Expr *expr, int dynamic_index_stride) {
             auto matrix_field = expr->cast<MatrixFieldExpression>();
             matrix_field->dynamic_indexable = true;
             matrix_field->dynamic_index_stride = dynamic_index_stride;
           })
      .def("get_dynamic_indexable",
           [&](Expr *expr) -> bool { return expr->cast<MatrixFieldExpression>()->dynamic_indexable; })
      .def("get_dynamic_index_stride",
           [&](Expr *expr) -> int { return expr->cast<MatrixFieldExpression>()->dynamic_index_stride; })
      .def(
          "get_dt", [&](Expr *expr) -> const Type * { return expr->cast<FieldExpression>()->dt; },
          nb::rv_policy::reference)
      .def("get_ret_type", &Expr::get_ret_type)
      .def("get_rvalue_type", [](Expr *expr) { return expr->get_rvalue_type(); })
      .def("is_tensor", [](Expr *expr) { return expr->get_rvalue_type()->is<TensorType>(); })
      .def("is_struct", [](Expr *expr) { return expr->get_rvalue_type()->is<StructType>(); })
      .def("get_shape",
           [](Expr *expr) -> std::optional<std::vector<int>> {
             auto tensor_type = expr->get_rvalue_type()->cast<TensorType>();
             if (tensor_type) {
               return std::optional<std::vector<int>>(tensor_type->get_shape());
             }
             return std::nullopt;
           })
      .def("type_check", &Expr::type_check)
      .def("get_expr_name", [](Expr *expr) { return expr->cast<FieldExpression>()->name; })
      .def("get_raw_address", [](Expr *expr) { return (uint64)expr; })
      .def("get_underlying_ptr_address", [](Expr *e) {
        // The reason that there are both get_raw_address() and
        // get_underlying_ptr_address() is that Expr itself is mostly wrapper
        // around its underlying |expr| (of type Expression). Expr |e| can be
        // temporary, while the underlying |expr| is mostly persistent.
        //
        // Same get_raw_address() implies that get_underlying_ptr_address() are
        // also the same. The reverse is not true.
        return (uint64)e->expr.get();
      });

  nb::class_<ExprGroup>(m, "ExprGroup")
      .def(nb::init<>())
      .def("size", [](ExprGroup *eg) { return eg->exprs.size(); })
      .def("push_back", &ExprGroup::push_back);

  nb::class_<Stmt>(m, "Stmt");  // NOLINT(bugprone-unused-raii)

  m.def("insert_internal_func_call",
        [&](Operation *op, const ExprGroup &args) { return Expr::make<InternalFuncCallExpression>(op, args.exprs); });

  m.def("make_get_element_expr", Expr::make<GetElementExpression, const Expr &, std::vector<int>, const DebugInfo &>);

  m.def("value_cast", static_cast<Expr (*)(const Expr &expr, DataType)>(cast));
  m.def("bits_cast", static_cast<Expr (*)(const Expr &expr, DataType)>(bit_cast));

  m.def("expr_atomic_add",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::add, a, b); });

  m.def("expr_atomic_sub",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::sub, a, b); });

  m.def("expr_atomic_min",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::min, a, b); });

  m.def("expr_atomic_max",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::max, a, b); });

  m.def("expr_atomic_bit_and",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::bit_and, a, b); });

  m.def("expr_atomic_bit_or",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::bit_or, a, b); });

  m.def("expr_atomic_bit_xor",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::bit_xor, a, b); });

  m.def("expr_atomic_mul",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::mul, a, b); });

  m.def("expr_atomic_xchg",
        [&](const Expr &a, const Expr &b) { return Expr::make<AtomicOpExpression>(AtomicOpType::xchg, a, b); });

  m.def("expr_atomic_cas", [&](const Expr &dest, const Expr &expected, const Expr &desired) {
    return Expr::make<AtomicOpExpression>(AtomicOpType::cas, dest, expected, desired);
  });

  m.def("expr_volatile_load", [&](const Expr &target) { return Expr::make<VolatileLoadExpression>(target); });

  m.def("expr_assume_in_range", assume_range);

  m.def("expr_loop_unique", loop_unique);

  m.def("expr_field", expr_field);

  m.def("expr_matrix_field", expr_matrix_field);

#define DEFINE_EXPRESSION_OP(x) m.def("expr_" #x, expr_##x);

  DEFINE_EXPRESSION_OP(neg)
  DEFINE_EXPRESSION_OP(sqrt)
  DEFINE_EXPRESSION_OP(round)
  DEFINE_EXPRESSION_OP(floor)
  DEFINE_EXPRESSION_OP(frexp)
  DEFINE_EXPRESSION_OP(ceil)
  DEFINE_EXPRESSION_OP(abs)
  DEFINE_EXPRESSION_OP(sin)
  DEFINE_EXPRESSION_OP(asin)
  DEFINE_EXPRESSION_OP(cos)
  DEFINE_EXPRESSION_OP(acos)
  DEFINE_EXPRESSION_OP(tan)
  DEFINE_EXPRESSION_OP(tanh)
  DEFINE_EXPRESSION_OP(inv)
  DEFINE_EXPRESSION_OP(rcp)
  DEFINE_EXPRESSION_OP(rsqrt)
  DEFINE_EXPRESSION_OP(exp)
  DEFINE_EXPRESSION_OP(log)
  DEFINE_EXPRESSION_OP(popcnt)
  DEFINE_EXPRESSION_OP(clz)
  DEFINE_EXPRESSION_OP(ffs)

  DEFINE_EXPRESSION_OP(select)
  DEFINE_EXPRESSION_OP(ifte)

  DEFINE_EXPRESSION_OP(cmp_le)
  DEFINE_EXPRESSION_OP(cmp_lt)
  DEFINE_EXPRESSION_OP(cmp_ge)
  DEFINE_EXPRESSION_OP(cmp_gt)
  DEFINE_EXPRESSION_OP(cmp_ne)
  DEFINE_EXPRESSION_OP(cmp_eq)

  DEFINE_EXPRESSION_OP(bit_and)
  DEFINE_EXPRESSION_OP(bit_or)
  DEFINE_EXPRESSION_OP(bit_xor)
  DEFINE_EXPRESSION_OP(bit_shl)
  DEFINE_EXPRESSION_OP(bit_shr)
  DEFINE_EXPRESSION_OP(bit_sar)
  DEFINE_EXPRESSION_OP(bit_not)

  DEFINE_EXPRESSION_OP(logic_not)
  DEFINE_EXPRESSION_OP(logical_and)
  DEFINE_EXPRESSION_OP(logical_or)

  DEFINE_EXPRESSION_OP(add)
  DEFINE_EXPRESSION_OP(sub)
  DEFINE_EXPRESSION_OP(mul)
  DEFINE_EXPRESSION_OP(div)
  DEFINE_EXPRESSION_OP(truediv)
  DEFINE_EXPRESSION_OP(floordiv)
  DEFINE_EXPRESSION_OP(mod)
  DEFINE_EXPRESSION_OP(max)
  DEFINE_EXPRESSION_OP(min)
  DEFINE_EXPRESSION_OP(atan2)
  DEFINE_EXPRESSION_OP(pow)

#undef DEFINE_EXPRESSION_OP

  // nanobind rejects None for pointer arguments by default; pybind11 mapped None -> nullptr. Opt back in with
  // .none() so callers (incl. the binding-surface test) can pass None for these Stmt* operands.
  m.def("make_global_load_stmt", Stmt::make<GlobalLoadStmt, Stmt *>, nb::arg().none());
  m.def("make_global_store_stmt", Stmt::make<GlobalStoreStmt, Stmt *, Stmt *>, nb::arg().none(), nb::arg().none());
  m.def("make_frontend_assign_stmt", Stmt::make<FrontendAssignStmt, const Expr &, const Expr &, const DebugInfo &>);

  m.def("make_arg_load_expr",
        Expr::make<ArgLoadExpression, const std::vector<int> &, const DataType &, bool, bool, const DebugInfo &>,
        "arg_id"_a, "dt"_a, "is_ptr"_a = false, "create_load"_a = true, "dbg_info"_a = DebugInfo());

  m.def("make_reference", Expr::make<ReferenceExpression, const Expr &, const DebugInfo &>);

  m.def("make_external_tensor_expr", Expr::make<ExternalTensorExpression, const DataType &, int,
                                                const std::vector<int> &, bool, const BoundaryMode &>);

  m.def("make_external_tensor_grad_expr", Expr::make<ExternalTensorExpression, Expr *>);

  m.def("make_rand_expr", Expr::make<RandExpression, const DataType &, const DebugInfo &>);

  m.def("make_const_expr_bool", Expr::make<ConstExpression, const DataType &, uint1>);

  m.def("make_const_expr_int", Expr::make<ConstExpression, const DataType &, int64>);

  m.def("make_const_expr_fp", Expr::make<ConstExpression, const DataType &, float64>);

  auto &&bin = nb::enum_<BinaryOpType>(m, "BinaryOpType", nb::is_arithmetic());
  for (int t = 0; t <= (int)BinaryOpType::undefined; t++)
    bin.value(binary_op_type_name(BinaryOpType(t)).c_str(), BinaryOpType(t));
  bin.export_values();
  m.def("make_binary_op_expr", Expr::make<BinaryOpExpression, const BinaryOpType &, const Expr &, const Expr &>);

  auto &&unary = nb::enum_<UnaryOpType>(m, "UnaryOpType", nb::is_arithmetic());
  for (int t = 0; t <= (int)UnaryOpType::undefined; t++)
    unary.value(unary_op_type_name(UnaryOpType(t)).c_str(), UnaryOpType(t));
  unary.export_values();
  m.def("make_unary_op_expr", Expr::make<UnaryOpExpression, const UnaryOpType &, const Expr &>);
#define PER_TYPE(x) m.attr(("DataType_" + data_type_name(PrimitiveType::x)).c_str()) = PrimitiveType::x;
#include "quadrants/inc/data_type.inc.h"
#undef PER_TYPE

  m.def("data_type_size", data_type_size);
  m.def("is_quant", is_quant);
  m.def("is_integral", is_integral);
  m.def("is_signed", is_signed);
  m.def("is_real", is_real);
  m.def("is_unsigned", is_unsigned);
  m.def("is_tensor", is_tensor);

  m.def("data_type_name", data_type_name);

  m.def("subscript_with_multiple_indices", Expr::make<IndexExpression, const Expr &, const std::vector<ExprGroup> &,
                                                      const std::vector<int> &, const DebugInfo &>);

  m.def("get_external_tensor_element_dim", [](const Expr &expr) {
    QD_ASSERT(expr.is<ExternalTensorExpression>());
    // FIXME: no need to make it negative since we don't support SOA
    auto dtype = expr.cast<ExternalTensorExpression>()->dt;
    return dtype->is<TensorType>() ? -dtype->cast<TensorType>()->get_shape().size() : 0;
  });

  m.def("get_external_tensor_needs_grad", [](const Expr &expr) {
    QD_ASSERT(expr.is<ExternalTensorExpression>());
    return expr.cast<ExternalTensorExpression>()->needs_grad;
  });

  m.def("get_external_tensor_element_type", [](const Expr &expr) {
    QD_ASSERT(expr.is<ExternalTensorExpression>());
    auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
    return external_tensor_expr->dt;
  });

  m.def("get_external_tensor_element_shape", [](const Expr &expr) {
    QD_ASSERT(expr.is<ExternalTensorExpression>());
    auto external_tensor_expr = expr.cast<ExternalTensorExpression>();
    return external_tensor_expr->dt.get_shape();
  });

  m.def("get_external_tensor_dim", [](const Expr &expr) {
    if (expr.is<ExternalTensorExpression>()) {
      return expr.cast<ExternalTensorExpression>()->ndim;
    } else {
      QD_ASSERT(false);
      return 0;
    }
  });

  m.def("get_external_tensor_shape_along_axis",
        Expr::make<ExternalTensorShapeAlongAxisExpression, const Expr &, int, const DebugInfo &>);

  m.def("get_external_tensor_real_func_args", [](const Expr &expr, const DebugInfo &dbg_info = DebugInfo()) {
    QD_ASSERT(expr.is<ExternalTensorExpression>());
    auto external_tensor_expr = expr.cast<ExternalTensorExpression>();

    std::vector<Expr> args;
    for (int i = 0; i < external_tensor_expr->ndim; i++) {
      args.push_back(Expr::make<ExternalTensorShapeAlongAxisExpression>(expr, i, expr->dbg_info));
      args.back()->type_check(nullptr);
    }

    args.push_back(Expr::make<ExternalTensorBasePtrExpression>(expr, /*is_grad=*/false, dbg_info));
    args.back()->type_check(nullptr);

    if (external_tensor_expr->needs_grad) {
      args.push_back(Expr::make<ExternalTensorBasePtrExpression>(expr, /*is_grad=*/true, dbg_info));
      args.back()->type_check(nullptr);
    }

    return args;
  });

  // Mesh related.
  m.def("get_relation_size", [](mesh::MeshPtr mesh_ptr, const Expr &mesh_idx, mesh::MeshElementType to_type,
                                const DebugInfo &dbg_info = DebugInfo()) {
    return Expr::make<MeshRelationAccessExpression>(mesh_ptr.ptr.get(), mesh_idx, to_type, dbg_info);
  });

  m.def("get_relation_access", [](mesh::MeshPtr mesh_ptr, const Expr &mesh_idx, mesh::MeshElementType to_type,
                                  const Expr &neighbor_idx, const DebugInfo &dbg_info = DebugInfo()) {
    return Expr::make<MeshRelationAccessExpression>(mesh_ptr.ptr.get(), mesh_idx, to_type, neighbor_idx, dbg_info);
  });

  nb::class_<FunctionKey>(m, "FunctionKey")
      .def(nb::init<const std::string &, int, int>())
      .def_ro("instance_id", &FunctionKey::instance_id);

  m.def("test_throw", [] {
    try {
      throw IRModified();
    } catch (IRModified) {
      QD_INFO("caught");
    }
  });

  m.def("test_throw", [] { throw IRModified(); });

#if QD_WITH_LLVM
  m.def("libdevice_path", libdevice_path);
#endif

  m.def("host_arch", host_arch);
  m.def("arch_uses_llvm", arch_uses_llvm);

  // The Python caller passes a filesystem-encoded path as ``bytes`` (see _lib/utils.py:locale_encode). pybind11's
  // std::string caster accepted bytes directly; nanobind's only accepts str, so take nb::bytes and copy the raw
  // bytes to preserve non-UTF-8 path fidelity.
  m.def("set_lib_dir", [&](nb::bytes dir) { compiled_lib_dir = std::string(dir.c_str(), dir.size()); });
  m.def("set_tmp_dir", [&](const std::string &dir) { runtime_tmp_dir = dir; });

  m.def("get_commit_hash", get_commit_hash);
  m.def("get_version_string", get_version_string);
  m.def("get_version_major", get_version_major);
  m.def("get_version_minor", get_version_minor);
  m.def("get_version_patch", get_version_patch);
  m.def("get_llvm_target_support", [] {
#if defined(QD_WITH_LLVM)
    return LLVM_VERSION_STRING;
#else
    return "targets unsupported";
#endif
  });
  m.def("test_printf", [] { printf("test_printf\n"); });
  m.def("test_logging", [] { QD_INFO("test_logging"); });
  m.def("trigger_crash", [] { *(int *)(1) = 0; });
  m.def("get_max_num_indices", [] { return quadrants_max_num_indices; });
  m.def("test_threading", test_threading);
  m.def("is_extension_supported", is_extension_supported);

  m.def("query_int64", [](const std::string &key) -> int64_t {
    if (key == "cuda_compute_capability") {
#if defined(QD_WITH_CUDA)
      return static_cast<int64_t>(CUDAContext::get_instance().get_compute_capability());
#else
      QD_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_max_shared_memory_bytes") {
#if defined(QD_WITH_CUDA)
      return static_cast<int64_t>(CUDAContext::get_instance().get_max_shared_memory_bytes());
#else
      QD_NOT_IMPLEMENTED
#endif
    } else if (key == "cuda_clock_rate_khz") {
#if defined(QD_WITH_CUDA)
      return CUDAContext::get_instance().get_clock_rate_khz();
#else
      QD_NOT_IMPLEMENTED
#endif
    } else {
      QD_ERROR("Key {} not supported in query_int64", key);
    }
  });

  // Type system

  nb::class_<Type>(m, "Type").def("to_string", &Type::to_string);

  m.def("promoted_type", promoted_type);

  // Note that it is important to specify nb::rv_policy::reference for
  // the factory methods, otherwise pybind11 will delete the Types owned by
  // TypeFactory on Python-scope pointer destruction.
  nb::class_<TypeFactory>(m, "TypeFactory")
      .def("get_quant_int_type", &TypeFactory::get_quant_int_type, nb::arg("num_bits"), nb::arg("is_signed"),
           nb::arg("compute_type"), nb::rv_policy::reference)
      .def("get_quant_fixed_type", &TypeFactory::get_quant_fixed_type, nb::arg("digits_type"), nb::arg("compute_type"),
           nb::arg("scale"), nb::rv_policy::reference)
      .def("get_quant_float_type", &TypeFactory::get_quant_float_type, nb::arg("digits_type"), nb::arg("exponent_type"),
           nb::arg("compute_type"), nb::rv_policy::reference)
      .def(
          "get_tensor_type",
          [&](TypeFactory *factory, std::vector<int> shape, const DataType &element_type) {
            return factory->create_tensor_type(shape, element_type);
          },
          nb::rv_policy::reference)
      .def(
          "get_struct_type",
          [&](TypeFactory *factory, std::vector<std::pair<DataType, std::string>> elements) {
            std::vector<AbstractDictionaryMember> members;
            for (auto &[type, name] : elements) {
              members.push_back({type, name});
            }
            return DataType(factory->get_struct_type(members));
          },
          nb::rv_policy::reference)
      .def("get_ndarray_struct_type", &TypeFactory::get_ndarray_struct_type, nb::arg("dt"), nb::arg("ndim"),
           nb::arg("needs_grad"), nb::rv_policy::reference);

  m.def("get_type_factory_instance", TypeFactory::get_instance, nb::rv_policy::reference);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  nb::class_<BitStructType>(m, "BitStructType");
  nb::class_<BitStructTypeBuilder>(m, "BitStructTypeBuilder")
      .def(nb::init<int>())
      .def("begin_placing_shared_exponent", &BitStructTypeBuilder::begin_placing_shared_exponent)
      .def("end_placing_shared_exponent", &BitStructTypeBuilder::end_placing_shared_exponent)
      .def("add_member", &BitStructTypeBuilder::add_member)
      .def("build", &BitStructTypeBuilder::build, nb::rv_policy::reference);

  nb::class_<SNodeRegistry>(m, "SNodeRegistry")
      .def(nb::init<>())
      .def("create_root", &SNodeRegistry::create_root, nb::rv_policy::reference);

  m.def(
      "finalize_snode_tree",
      [](SNodeRegistry *registry, const SNode *root, Program *program, bool compile_only) -> SNodeTree * {
        return program->add_snode_tree(registry->finalize(root), compile_only);
      },
      nb::rv_policy::reference);

  // Sparse Matrix
  nb::class_<SparseMatrixBuilder>(m, "SparseMatrixBuilder")
      .def(nb::init<int, int, int, DataType, const std::string &>(), nb::arg("rows"), nb::arg("cols"),
           nb::arg("max_num_triplets"), nb::arg("dt") = PrimitiveType::f32, nb::arg("storage_format") = "col_major")
      .def("print_triplets_eigen", &SparseMatrixBuilder::print_triplets_eigen)
      .def("print_triplets_cuda", &SparseMatrixBuilder::print_triplets_cuda)
      .def("create_ndarray", [&](SparseMatrixBuilder *builder, Program *prog) { return builder->create_ndarray(prog); })
      .def("delete_ndarray", [&](SparseMatrixBuilder *builder, Program *prog) { return builder->delete_ndarray(prog); })
      .def("get_ndarray_data_ptr", &SparseMatrixBuilder::get_ndarray_data_ptr)
      .def("build", &SparseMatrixBuilder::build)
      .def("build_cuda", &SparseMatrixBuilder::build_cuda)
      .def("get_addr", [](SparseMatrixBuilder *mat) { return uint64(mat); });

  nb::class_<SparseMatrix>(m, "SparseMatrix")
      .def(nb::init<>())
      .def(nb::init<int, int, DataType>(), nb::arg("rows"), nb::arg("cols"), nb::arg("dt") = PrimitiveType::f32)
      .def(nb::init<SparseMatrix &>())
      .def("to_string", &SparseMatrix::to_string)
      .def("get_element", &SparseMatrix::get_element<float32>)
      .def("set_element", &SparseMatrix::set_element<float32>)
      .def("mmwrite", &SparseMatrix::mmwrite)
      .def("num_rows", &SparseMatrix::num_rows)
      .def("num_cols", &SparseMatrix::num_cols)
      .def("get_data_type", &SparseMatrix::get_data_type);

#define MAKE_SPARSE_MATRIX(TYPE, STORAGE, VTYPE)                                                                   \
  using STORAGE##TYPE##EigenMatrix = Eigen::SparseMatrix<float##TYPE, Eigen::STORAGE>;                             \
  nb::class_<EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>, SparseMatrix>(m, #VTYPE #STORAGE "_EigenSparseMatrix") \
      .def(nb::init<int, int, DataType>())                                                                         \
      .def(nb::init<EigenSparseMatrix<STORAGE##TYPE##EigenMatrix> &>())                                            \
      .def(nb::init<const STORAGE##TYPE##EigenMatrix &>())                                                         \
      .def(nb::self += nb::self)                                                                                   \
      .def(nb::self + nb::self)                                                                                    \
      .def(nb::self -= nb::self)                                                                                   \
      .def(nb::self - nb::self)                                                                                    \
      .def(nb::self *= float##TYPE())                                                                              \
      .def(nb::self *float##TYPE())                                                                                \
      .def(float##TYPE() * nb::self)                                                                               \
      .def(nb::self *nb::self)                                                                                     \
      .def("matmul", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::matmul)                                       \
      .def("spmv", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::spmv)                                           \
      .def("transpose", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::transpose)                                 \
      .def("get_element", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::get_element<float##TYPE>)                \
      .def("set_element", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::set_element<float##TYPE>)                \
      .def("mat_vec_mul", &EigenSparseMatrix<STORAGE##TYPE##EigenMatrix>::mat_vec_mul<Eigen::VectorX##VTYPE>);

  MAKE_SPARSE_MATRIX(32, ColMajor, f);
  MAKE_SPARSE_MATRIX(32, RowMajor, f);
  MAKE_SPARSE_MATRIX(64, ColMajor, d);
  MAKE_SPARSE_MATRIX(64, RowMajor, d);

  nb::class_<CuSparseMatrix, SparseMatrix>(m, "CuSparseMatrix")
      .def(nb::init<int, int, DataType>())
      .def(nb::init<const CuSparseMatrix &>())
      .def("spmv", &CuSparseMatrix::nd_spmv)
      .def(nb::self + nb::self)
      .def(nb::self - nb::self)
      .def(nb::self * float32())
      .def(float32() * nb::self)
      .def("matmul", &CuSparseMatrix::matmul)
      .def("transpose", &CuSparseMatrix::transpose)
      .def("get_element", &CuSparseMatrix::get_element)
      .def("to_string", &CuSparseMatrix::to_string);

  nb::class_<SparseSolver>(m, "SparseSolver")
      .def("compute", &SparseSolver::compute)
      .def("analyze_pattern", &SparseSolver::analyze_pattern)
      .def("factorize", &SparseSolver::factorize)
      .def("info", &SparseSolver::info);

#define REGISTER_EIGEN_SOLVER(dt, type, order, fd)                                                      \
  nb::class_<EigenSparseSolver##dt##type##order, SparseSolver>(m, "EigenSparseSolver" #dt #type #order) \
      .def("compute", &EigenSparseSolver##dt##type##order::compute)                                     \
      .def("analyze_pattern", &EigenSparseSolver##dt##type##order::analyze_pattern)                     \
      .def("factorize", &EigenSparseSolver##dt##type##order::factorize)                                 \
      .def("solve", &EigenSparseSolver##dt##type##order::solve<Eigen::VectorX##fd>)                     \
      .def("solve_rf", &EigenSparseSolver##dt##type##order::solve_rf<Eigen::VectorX##fd, dt>)           \
      .def("info", &EigenSparseSolver##dt##type##order::info);

  REGISTER_EIGEN_SOLVER(float32, LLT, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LLT, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float32, LDLT, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LDLT, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float32, LU, AMD, f)
  REGISTER_EIGEN_SOLVER(float32, LU, COLAMD, f)
  REGISTER_EIGEN_SOLVER(float64, LLT, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LLT, COLAMD, d)
  REGISTER_EIGEN_SOLVER(float64, LDLT, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LDLT, COLAMD, d)
  REGISTER_EIGEN_SOLVER(float64, LU, AMD, d)
  REGISTER_EIGEN_SOLVER(float64, LU, COLAMD, d)

  nb::class_<CuSparseSolver, SparseSolver>(m, "CuSparseSolver")
      .def("compute", &CuSparseSolver::compute)
      .def("analyze_pattern", &CuSparseSolver::analyze_pattern)
      .def("factorize", &CuSparseSolver::factorize)
      .def("solve_rf", &CuSparseSolver::solve_rf)
      .def("info", &CuSparseSolver::info);

  m.def("make_sparse_solver", &make_sparse_solver);
  m.def("make_cusparse_solver", &make_cusparse_solver);

  // Conjugate Gradient solver
  nb::class_<CG<Eigen::VectorXf, float>>(m, "CGf")
      .def(nb::init<SparseMatrix &, int, float, bool>())
      .def("solve", &CG<Eigen::VectorXf, float>::solve)
      .def("set_x", &CG<Eigen::VectorXf, float>::set_x)
      .def("get_x", &CG<Eigen::VectorXf, float>::get_x)
      .def("set_x_ndarray", &CG<Eigen::VectorXf, float>::set_x_ndarray)
      .def("set_b", &CG<Eigen::VectorXf, float>::set_b)
      .def("set_b_ndarray", &CG<Eigen::VectorXf, float>::set_b_ndarray)
      .def("is_success", &CG<Eigen::VectorXf, float>::is_success);
  nb::class_<CG<Eigen::VectorXd, double>>(m, "CGd")
      .def(nb::init<SparseMatrix &, int, double, bool>())
      .def("solve", &CG<Eigen::VectorXd, double>::solve)
      .def("set_x", &CG<Eigen::VectorXd, double>::set_x)
      .def("set_x_ndarray", &CG<Eigen::VectorXd, double>::set_x_ndarray)
      .def("get_x", &CG<Eigen::VectorXd, double>::get_x)
      .def("set_b_ndarray", &CG<Eigen::VectorXd, double>::set_b_ndarray)
      .def("set_b", &CG<Eigen::VectorXd, double>::set_b)
      .def("is_success", &CG<Eigen::VectorXd, double>::is_success);
  m.def("make_float_cg_solver", [](SparseMatrix &A, int max_iters, float tol, bool verbose) {
    return make_cg_solver<Eigen::VectorXf, float>(A, max_iters, tol, verbose);
  });
  m.def("make_double_cg_solver", [](SparseMatrix &A, int max_iters, float tol, bool verbose) {
    return make_cg_solver<Eigen::VectorXd, double>(A, max_iters, tol, verbose);
  });

  nb::class_<CUCG>(m, "CUCG").def("solve", &CUCG::solve);
  m.def("make_cucg_solver", make_cucg_solver);

  // Mesh Class
  // Mesh related.
  nb::enum_<mesh::MeshTopology>(m, "MeshTopology", nb::is_arithmetic())
      .value("Triangle", mesh::MeshTopology::Triangle)
      .value("Tetrahedron", mesh::MeshTopology::Tetrahedron)
      .export_values();

  nb::enum_<mesh::MeshElementType>(m, "MeshElementType", nb::is_arithmetic())
      .value("Vertex", mesh::MeshElementType::Vertex)
      .value("Edge", mesh::MeshElementType::Edge)
      .value("Face", mesh::MeshElementType::Face)
      .value("Cell", mesh::MeshElementType::Cell)
      .export_values();

  nb::enum_<mesh::MeshRelationType>(m, "MeshRelationType", nb::is_arithmetic())
      .value("VV", mesh::MeshRelationType::VV)
      .value("VE", mesh::MeshRelationType::VE)
      .value("VF", mesh::MeshRelationType::VF)
      .value("VC", mesh::MeshRelationType::VC)
      .value("EV", mesh::MeshRelationType::EV)
      .value("EE", mesh::MeshRelationType::EE)
      .value("EF", mesh::MeshRelationType::EF)
      .value("EC", mesh::MeshRelationType::EC)
      .value("FV", mesh::MeshRelationType::FV)
      .value("FE", mesh::MeshRelationType::FE)
      .value("FF", mesh::MeshRelationType::FF)
      .value("FC", mesh::MeshRelationType::FC)
      .value("CV", mesh::MeshRelationType::CV)
      .value("CE", mesh::MeshRelationType::CE)
      .value("CF", mesh::MeshRelationType::CF)
      .value("CC", mesh::MeshRelationType::CC)
      .export_values();

  nb::enum_<mesh::ConvType>(m, "ConvType", nb::is_arithmetic())
      .value("l2g", mesh::ConvType::l2g)
      .value("l2r", mesh::ConvType::l2r)
      .value("g2r", mesh::ConvType::g2r)
      .export_values();

  nb::class_<mesh::Mesh>(m, "Mesh");        // NOLINT(bugprone-unused-raii)
  nb::class_<mesh::MeshPtr>(m, "MeshPtr");  // NOLINT(bugprone-unused-raii)

  m.def("element_order", mesh::element_order);
  m.def("from_end_element_order", mesh::from_end_element_order);
  m.def("to_end_element_order", mesh::to_end_element_order);
  m.def("relation_by_orders", mesh::relation_by_orders);
  m.def("inverse_relation", mesh::inverse_relation);
  m.def("element_type_name", mesh::element_type_name);

  m.def(
      "create_mesh",
      []() {
        auto mesh_shared = std::make_shared<mesh::Mesh>();
        mesh::MeshPtr mesh_ptr = mesh::MeshPtr{mesh_shared};
        return mesh_ptr;
      },
      nb::rv_policy::reference);

  // ad-hoc setters
  m.def("set_owned_offset", [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, SNode *snode) {
    mesh_ptr.ptr->owned_offset.insert(std::pair(type, snode));
  });
  m.def("set_total_offset", [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, SNode *snode) {
    mesh_ptr.ptr->total_offset.insert(std::pair(type, snode));
  });
  m.def("set_num_patches", [](mesh::MeshPtr &mesh_ptr, int num_patches) { mesh_ptr.ptr->num_patches = num_patches; });

  m.def("set_num_elements", [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, int num_elements) {
    mesh_ptr.ptr->num_elements.insert(std::pair(type, num_elements));
  });

  m.def("get_num_elements", [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type) {
    return mesh_ptr.ptr->num_elements.find(type)->second;
  });

  m.def("set_patch_max_element_num", [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType type, int max_element_num) {
    mesh_ptr.ptr->patch_max_element_num.insert(std::pair(type, max_element_num));
  });

  m.def("set_index_mapping",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshElementType element_type, mesh::ConvType conv_type, SNode *snode) {
          mesh_ptr.ptr->index_mapping.insert(std::make_pair(std::make_pair(element_type, conv_type), snode));
        });

  m.def("set_relation_fixed", [](mesh::MeshPtr &mesh_ptr, mesh::MeshRelationType type, SNode *value) {
    mesh_ptr.ptr->relations.insert(std::pair(type, mesh::MeshLocalRelation(value)));
  });

  m.def("set_relation_dynamic",
        [](mesh::MeshPtr &mesh_ptr, mesh::MeshRelationType type, SNode *value, SNode *patch_offset, SNode *offset) {
          mesh_ptr.ptr->relations.insert(std::pair(type, mesh::MeshLocalRelation(value, patch_offset, offset)));
        });

  m.def("wait_for_debugger", []() {
#ifdef WIN32
    while (!::IsDebuggerPresent())
      ::Sleep(100);
#endif
  });

  auto operationClass = nb::class_<Operation>(m, "Operation");
  auto internalOpClass = nb::class_<InternalOpScope>(m, "InternalOp");

#define PER_INTERNAL_OP(x)                      \
  internalOpClass.def_prop_ro_static( \
      #x, [](nb::object) { return Operations::get(InternalOp::x); }, nb::rv_policy::reference);
#include "quadrants/inc/internal_ops.inc.h"
#undef PER_INTERNAL_OP
}

}  // namespace quadrants
