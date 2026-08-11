#include "quadrants/ir/ir.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/pass.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/program/extension.h"
#include "quadrants/program/function.h"
#include "quadrants/program/kernel.h"
#include "quadrants/util/lang_util.h"
#include "quadrants/codegen/ir_dump.h"
#include <fstream>

namespace quadrants::lang {

namespace irpass {

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         const Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast) {
  QD_AUTO_PROF;

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, kernel->get_name(), ir);
  print("Initial IR");

  if (!verbose && config.print_preprocessed_ir && start_from_ast) {
    QD_INFO("[{}] {}:", kernel->get_name(), "Preprocessed IR");
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  }

  if (autodiff_mode == AutodiffMode::kReverse) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  const char *dump_ir_env = std::getenv(DUMP_IR_ENV.data());
  std::filesystem::path ir_dump_dir = config.debug_dump_path;
  bool should_dump = (dump_ir_env != nullptr && std::string(dump_ir_env) == "1");

  auto dump_ir = [&](const std::string &stage_name) {
    if (!should_dump)
      return;
    std::filesystem::create_directories(ir_dump_dir);
    std::filesystem::path filename = ir_dump_dir / (kernel->name + "_" + stage_name + ".ll");
    std::string ir_str;
    irpass::print(ir, &ir_str);
    std::ofstream ofs(filename.string());
    if (ofs.good()) {
      ofs << ir_str;
    }
  };

  dump_ir("from_ast");

  if (start_from_ast) {
    irpass::frontend_type_check(ir);
    irpass::lower_ast(ir);
  }

  dump_ir("quadrants1");
  irpass::compile_quadrants_functions(ir, config, Function::IRStage::BeforeLowerAccess);
  irpass::analysis::gather_func_store_dests(ir);
  irpass::compile_quadrants_functions(ir, config, Function::IRStage::OptimizedIR);
  irpass::analysis::gather_func_store_dests(ir);

  irpass::eliminate_immutable_local_vars(ir);

  irpass::type_check(ir, config);
  irpass::analysis::verify_if_debug(ir, config);

  // TODO: strictly enforce bit vectorization for x86 cpu and CUDA now
  //       create a separate CompileConfig flag for the new pass
  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda || config.arch == Arch::amdgpu) {
    irpass::bit_loop_vectorize(ir);
    irpass::type_check(ir, config);
    irpass::analysis::verify_if_debug(ir, config);
  }

  // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
  irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);
    irpass::die(ir);
  }

  dump_ir("before_simplify_I");
  irpass::full_simplify(
      ir, config,
      {false, /*autodiff_enabled*/ autodiff_mode != AutodiffMode::kNone, kernel->get_name(), verbose, "simplify_I"});
  irpass::analysis::verify_if_debug(ir, config);
  dump_ir("after_simplify_I");

  irpass::handle_external_ptr_boundary(ir, config);

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::analysis::gather_meshfor_relation_types(ir);
  }

  if (config.debug && autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    // Check whether the kernel obeys the autodiff limitation e.g., gloabl data
    // access rule
    // This check should be performed in the forward kernel i.e., autodiff_mode
    // == AutodiffMode::kCheckAutodiffValid
    irpass::demote_atomics(ir, config);
    irpass::differentiation_validation_check(ir, config, kernel->get_name());
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (autodiff_mode == AutodiffMode::kReverse || autodiff_mode == AutodiffMode::kForward) {
    // Remove local atomics here so that we don't have to handle their gradients
    irpass::demote_atomics(ir, config);

    irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ true, kernel->get_name(), verbose, "pre_autodiff"});
    irpass::auto_diff(ir, config, autodiff_mode, ad_use_stack);
    // TODO: Be carefull with the full_simplify when do high-order autodiff
    irpass::full_simplify(ir, config,
                          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "post_autodiff"});
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(ir, config, {kernel->get_name()});
    irpass::analysis::verify_if_debug(ir, config);
  }

  // Merge a global's separate read/write GlobalPtrStmts (same address) into one shared, activate=true pointer BEFORE
  // this first flag_access, so flag_access cannot stamp a read-only (activate=false) copy that the CSE eliminability
  // rule then refuses to re-merge with the in-loop write. Without it, cache_loop_invariant_global_vars sees a split
  // read/write and cannot cache conditional/in-if stores -> the -88% solver break-flag bug + the lost duck_in_box
  // optimization. On main this fell out of whole_kernel_cse running inside every full_simplify fixpoint; per-task CSE
  // does no pre-offload whole-kernel CSE, so we do this one cheap, pointers-only pass here instead (arithmetic is
  // already canonical after simplify_I, so a single call is enough; running it in the fixpoint was a +12-22s
  // compile regression for no extra benefit).
  irpass::merge_global_ptrs(ir);
  irpass::analysis::verify_if_debug(ir, config);

  irpass::flag_access(ir);
  irpass::analysis::verify_if_debug(ir, config);

  irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_II"});
  irpass::analysis::verify_if_debug(ir, config);

  irpass::offload(ir, config);
  irpass::analysis::verify_if_debug(ir, config);

  dump_ir("after_offload");

  // Full per-task CSE now, before flag_access #2 splits a global's read/write pointers by access flag and before
  // simplify_III's LICM hoists the read pointer out of the loop. This restores the pointer-unification that main
  // gets from whole_kernel_cse running inside the post-offload full_simplify (per-task CSE otherwise defers to the
  // codegen workers, which run after cache_loop_invariant_global_vars). Needed for ndarrays, which only become
  // ExternalPtrStmts during offload and so cannot be reached by the pre-offload merge_global_ptrs. See the pass.
  // Gated on opt_level like all other CSE (per_task_cse / upstream whole_kernel_cse): at opt_level 0 there is no CSE
  // to require pointer unification, matching upstream behaviour.
  if (config.opt_level > 0) {
    irpass::cse_offloaded_tasks(ir);
  }

  // NOTE: There was an additional CFG pass here, removed in
  // https://github.com/taichi-dev/taichi/pull/8691
  irpass::flag_access(ir);

  irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_III"});
  irpass::analysis::verify_if_debug(ir, config);

  dump_ir("after_simplify_III");

  // Run the adstack-size pre-pass here, before the per-task split in `KernelCodeGen::compile_kernel_to_module`
  // and before `make_cpu_multithreaded_range_for` in `offload_to_executable` rewrites user ranges into chunk
  // wrappers. The kernel IR still has every `OffloadedStmt` as a sibling in the top-level block, so the pre-
  // pass can resolve a `GlobalLoadStmt(GlobalTemporaryStmt)` source by walking across tasks: prep serial tasks
  // that store a dynamic range bound (e.g. `arr.shape[0]` lowered via `offload::PromoteIntermediateToGlobalTmp`)
  // are still visible alongside the consuming range-for task. Gated on the same reverse+ad_use_stack predicate
  // the per-task call used so compile behaviour is unchanged for forward-only kernels.
  if (autodiff_mode == AutodiffMode::kReverse && ad_use_stack) {
    irpass::determine_ad_stack_size(ir, config);
    print("Autodiff stack size determined");
  }
}

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local) {
  QD_AUTO_PROF;

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, kernel->get_name(), ir);

  // TODO: This is just a proof that we can demote struct-fors after offloading.
  // Eventually we might want the order to be TLS/BLS -> demote struct-for.
  // For now, putting this after TLS will disable TLS, because it can only
  // handle range-fors at this point.

  auto amgr = std::make_unique<AnalysisManager>();

  print("Start offload_to_executable");
  irpass::analysis::verify_if_debug(ir, config);

  if (config.detect_read_only) {
    irpass::detect_read_only(ir);
    print("Detect read-only accesses");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted I");
  irpass::analysis::verify_if_debug(ir, config);

  if (config.cache_loop_invariant_global_vars) {
    irpass::cache_loop_invariant_global_vars(ir, config);
    print("Cache loop-invariant global vars");
  }

  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::type_check(ir, config);
    print("Dense struct-for demoted");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (config.make_cpu_multithreading_loop && arch_is_cpu(config.arch)) {
    irpass::make_cpu_multithreaded_range_for(ir, config);
    irpass::type_check(ir, config);
    print("Make CPU multithreaded range-for");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (is_extension_supported(config.arch, Extension::mesh) && config.demote_no_access_mesh_fors) {
    irpass::demote_no_access_mesh_fors(ir);
    irpass::type_check(ir, config);
    print("No-access mesh-for demoted");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (make_thread_local) {
    irpass::make_thread_local(ir, config);
    print("Make thread local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::make_mesh_thread_local(ir, config, {kernel->get_name()});
    print("Make mesh thread local");
    if (config.make_mesh_block_local && config.arch == Arch::cuda) {
      irpass::make_mesh_block_local(ir, config, {kernel->get_name()});
      print("Make mesh block local");
      irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_X"});
      print("Simplified X");
    }
  }

  if (make_block_local) {
    irpass::make_block_local(ir, config, {kernel->get_name(), verbose});
    print("Make block local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::demote_mesh_statements(ir, config, {kernel->get_name()});
    print("Demote mesh statements");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted II");
  irpass::analysis::verify_if_debug(ir, config);

  if (is_extension_supported(config.arch, Extension::quant) && config.quant_opt_atomic_demotion) {
    irpass::analysis::gather_uniquely_accessed_bit_structs(ir, amgr.get());
  }

  irpass::remove_range_assumption(ir);
  print("Remove range assumption");

  irpass::remove_loop_unique(ir);
  print("Remove loop_unique");
  irpass::analysis::verify_if_debug(ir, config);

  if (lower_global_access) {
    irpass::full_simplify(ir, config,
                          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "before_lower_access"});
    print("Simplified before lower access");
    irpass::lower_access(ir, config, {kernel->no_activate, true});
    print("Access lowered");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify_if_debug(ir, config);
  }

  irpass::demote_operations(ir, config);
  print("Operations demoted");

  irpass::full_simplify(ir, config,
                        {lower_global_access, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_IV"});
  print("Simplified IV");

  // `determine_ad_stack_size` used to run here, but the pre-pass needs the full kernel IR (all offloaded
  // tasks as siblings) so cross-task `GlobalTemporaryStmt` sources can be resolved. It now runs at the end
  // of `compile_to_offloads`, before the per-task split in `KernelCodeGen::compile_kernel_to_module`. The
  // `determine_ad_stack_size` parameter is kept in the signature for API stability but is no longer used.
  (void)determine_ad_stack_size;

  if (is_extension_supported(config.arch, Extension::quant)) {
    irpass::optimize_bit_struct_stores(ir, config, amgr.get());
    print("Bit struct stores optimized");
  }

  bool half2_optimization_enabled =
      (config.arch == Arch::cuda && config.half2_vectorization && !get_custom_cuda_library_path().empty());
  if (config.real_matrix_scalarize) {
    if (irpass::scalarize(ir, half2_optimization_enabled)) {
      irpass::die(ir);
      print("DIE");

      // Remove redundant MatrixInitStmt inserted during scalarization
      irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "scalarize"});
      print("Scalarized");
    }
  }

  // Final field registration correctness & type checking
  irpass::type_check(ir, config);
  irpass::analysis::verify_if_debug(ir, config);
}

void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           AutodiffMode autodiff_mode,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local,
                           bool start_from_ast) {
  QD_AUTO_PROF;

  compile_to_offloads(ir, config, kernel, verbose, autodiff_mode, ad_use_stack, start_from_ast);

  offload_to_executable(ir, config, kernel, verbose,
                        /*determine_ad_stack_size=*/autodiff_mode == AutodiffMode::kReverse && ad_use_stack,
                        lower_global_access, make_thread_local, make_block_local);
}

void compile_function(IRNode *ir,
                      const CompileConfig &config,
                      Function *func,
                      AutodiffMode autodiff_mode,
                      bool verbose,
                      Function::IRStage target_stage) {
  QD_AUTO_PROF;

  auto current_stage = func->ir_stage();
  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, func->get_name(), ir);
  print("Initial IR");

  if (target_stage >= Function::IRStage::BeforeLowerAccess && current_stage < Function::IRStage::BeforeLowerAccess) {
    if (autodiff_mode == AutodiffMode::kReverse) {
      irpass::reverse_segments(ir);
      print("Segment reversed (for autodiff)");
    }

    if (current_stage < Function::IRStage::InitialIR) {
      irpass::frontend_type_check(ir);
      irpass::lower_ast(ir);
      print("Lowered");
    }

    // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
    irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);
    print("Matrix ptr lowered");

    irpass::demote_atomics(ir, config);
    print("Atomics demoted");
    irpass::associate_continue_scope(ir, config);
    print("Associated continue scope");
    func->set_ir_stage(Function::IRStage::BeforeLowerAccess);
  }

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);
  }

  if (target_stage >= Function::IRStage::OptimizedIR && current_stage < Function::IRStage::OptimizedIR) {
    irpass::lower_access(ir, config, {{}, true});
    print("Access lowered");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::type_check(ir, config);
    print("Typechecked");

    irpass::demote_operations(ir, config);
    print("Operations demoted");

    if (config.real_matrix_scalarize) {
      if (irpass::scalarize(ir)) {
        // Remove redundant MatrixInitStmt inserted during scalarization
        irpass::die(ir);
        print("Scalarized");
      }
    }

    irpass::full_simplify(ir, config, {true, autodiff_mode != AutodiffMode::kNone, func->get_name(), verbose, "final"});
    print("Simplified");
    irpass::analysis::verify_if_debug(ir, config);
    func->set_ir_stage(Function::IRStage::OptimizedIR);
  }
}

}  // namespace irpass

}  // namespace quadrants::lang
