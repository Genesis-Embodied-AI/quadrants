#pragma once

#include "quadrants/rhi/arch.h"
#include "quadrants/util/lang_util.h"

namespace quadrants::lang {

struct CompileConfig {
  Arch arch;
  bool validate_autodiff;
  int simd_width;
  int opt_level;
  int external_optimization_level;
  int max_vector_width;
  bool raise_on_templated_floats{false};
  bool print_preprocessed_ir;
  bool print_ir;
  bool print_accessor_ir;
  bool print_ir_dbg_info;
  bool serial_schedule;
  bool simplify_before_lower_access;
  bool lower_access;
  bool simplify_after_lower_access;
  bool move_loop_invariant_outside_if;
  // Load-bearing optimization on contact-heavy solves (e.g. duck_in_box). The pass caches a loop-invariant global
  // load into a local, which is only sound when a global's read and write pointers are the same statement. Under
  // per-task CSE that unification is restored by merge_global_ptrs (pre-offload, fields) and cse_offloaded_tasks
  // (post-offload, ndarrays) -- see compile_to_offloads.cpp -- so this pass itself is unchanged from upstream.
  bool cache_loop_invariant_global_vars{true};
  bool demote_dense_struct_fors;
  bool advanced_optimization;
  bool constant_folding;
  bool use_llvm;
  bool verbose_kernel_launches;
  bool kernel_profiler;
  bool timeline{false};
  bool verbose;
  bool flatten_if;
  bool make_thread_local;
  bool make_block_local;
  bool detect_read_only;
  bool real_matrix_scalarize;
  bool force_scalarize_matrix;
  bool half2_vectorization;
  bool make_cpu_multithreading_loop;
  DataType default_fp;
  DataType default_ip;
  DataType default_up;
  int default_cpu_block_dim;
  bool cpu_block_dim_adaptive;
  int default_gpu_block_dim;
  int gpu_max_reg;

  int saturating_grid_dim;
  int max_block_dim;
  int cpu_max_num_threads;
  int random_seed;

  // Debugging options:
  bool print_struct_llvm_ir;
  bool print_kernel_llvm_ir;
  bool print_kernel_llvm_ir_optimized;
  bool print_kernel_asm;
  bool print_kernel_amdgcn;
  std::string debug_dump_path{"/tmp/ir/"};

  // CUDA/AMDGPU backend options:
  float64 device_memory_GB;
  float64 device_memory_fraction;

  bool quant_opt_store_fusion{true};
  bool quant_opt_atomic_demotion{true};

  // Mesh related.
  // MeshQuadrants options
  bool make_mesh_block_local{true};
  bool optimize_mesh_reordered_mapping{true};
  bool mesh_localize_to_end_mapping{true};
  bool mesh_localize_from_end_mapping{false};
  bool mesh_localize_all_attr_mappings{false};
  bool demote_no_access_mesh_fors{true};
  bool experimental_auto_mesh_local{false};
  int auto_mesh_local_default_occupacy{4};

  // Offline cache options
  std::string offline_cache_cleaning_policy{"lru"};        // "never"|"version"|"lru"|"fifo"
  int offline_cache_max_size_of_files{100 * 1024 * 1024};  // bytes, default: 100MB
  double offline_cache_cleaning_factor{0.25};              // [0.f, 1.f]

  std::string vk_api_version;

  size_t cuda_stack_limit{0};

  // Fields below are generated from tools/config_codegen/schema.py (the single
  // source of truth for their names, types, defaults, and user docs). Run
  // tools/config_codegen/generate.py to regenerate; CMake does this at configure
  // time. DO NOT add hand-written fields here that also exist in the schema.
#include "quadrants/program/compile_config.fields.generated.inc"

  CompileConfig();

  void fit();
};

extern QD_DLL_EXPORT CompileConfig default_compile_config;

}  // namespace quadrants::lang
