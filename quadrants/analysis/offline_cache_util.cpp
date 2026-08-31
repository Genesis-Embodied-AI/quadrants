#include "offline_cache_util.h"

#include "quadrants/common/core.h"
#include "quadrants/common/serialization.h"
#include "quadrants/common/version.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/program/kernel.h"
#include "quadrants/rhi/device_capability.h"

#include "picosha2.h"

#include <map>
#include <vector>

namespace quadrants::lang {

static std::vector<std::uint8_t> get_offline_cache_key_of_parameter_list(
    const std::vector<CallableBase::Parameter> &parameter_list) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(parameter_list);
  serializer.finalize();
  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_rets(const std::vector<CallableBase::Ret> &ret_list) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(ret_list);
  serializer.finalize();
  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_compile_config(const CompileConfig &config) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(config.arch);
  serializer(config.debug);
  serializer(config.cfg_optimization);
  serializer(config.check_out_of_bound);
  serializer(config.opt_level);
  serializer(config.external_optimization_level);
  serializer(config.move_loop_invariant_outside_if);
  serializer(config.demote_dense_struct_fors);
  serializer(config.advanced_optimization);
  serializer(config.constant_folding);
  serializer(config.kernel_profiler);
  serializer(config.fast_math);
  serializer(config.flatten_if);
  serializer(config.make_thread_local);
  serializer(config.make_block_local);
  serializer(config.detect_read_only);
  serializer(config.default_fp->to_string());
  serializer(config.default_ip.to_string());
  if (arch_is_cpu(config.arch)) {
    serializer(config.default_cpu_block_dim);
    serializer(config.cpu_max_num_threads);
  } else if (arch_is_gpu(config.arch)) {
    serializer(config.default_gpu_block_dim);
    serializer(config.saturating_grid_dim);
    serializer(config.cpu_max_num_threads);
    // Mix the per-arch subgroup / warp / wave size into the cache key so cached kernels are invalidated whenever the
    // constant changes (e.g. flipping AMDGPU between wave32 and wave64). Today CUDA is fixed at 32 and AMDGPU is fixed
    // at 64 (see kAmdgpuWaveSize in rhi/arch.h); this serialization path future-proofs the cache against any later
    // toggle. SPIR-V (Vulkan / Metal) returns ``0`` here — the device-probed ``DeviceCapability::spirv_subgroup_size``
    // is part of ``DeviceCapabilityConfig::devcaps`` and gets mixed into the key via
    // ``get_offline_cache_key_of_device_caps`` below, so wave32 vs wave64 SPIR-V devices already get distinct cache
    // entries without needing to plumb the value through here.
    serializer(subgroup_size(config.arch));
  }
  serializer(config.ad_stack_experimental_enabled);
  serializer(config.ad_stack_size);
  serializer(config.ad_stack_sparse_threshold_bytes);
  serializer(config.random_seed);
  serializer(config.make_mesh_block_local);
  serializer(config.optimize_mesh_reordered_mapping);
  serializer(config.mesh_localize_to_end_mapping);
  serializer(config.mesh_localize_from_end_mapping);
  serializer(config.mesh_localize_all_attr_mappings);
  serializer(config.demote_no_access_mesh_fors);
  serializer(config.experimental_auto_mesh_local);
  serializer(config.auto_mesh_local_default_occupacy);
  serializer(config.real_matrix_scalarize);
  serializer(config.force_scalarize_matrix);
  serializer(config.half2_vectorization);
  serializer.finalize();

  return serializer.data;
}

static std::vector<std::uint8_t> get_offline_cache_key_of_device_caps(const DeviceCapabilityConfig &caps) {
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(caps.devcaps);
  serializer.finalize();
  return serializer.data;
}

static void get_offline_cache_key_of_snode_impl(const SNode *snode,
                                                BinaryOutputSerializer &serializer,
                                                std::unordered_set<int> &visited) {
  if (auto iter = visited.find(snode->id); iter != visited.end()) {
    serializer(snode->id);  // Use snode->id as placeholder to identify a snode
    return;
  }

  visited.insert(snode->id);
  for (auto &c : snode->ch) {
    get_offline_cache_key_of_snode_impl(c.get(), serializer, visited);
  }
  for (int i = 0; i < quadrants_max_num_indices; ++i) {
    auto &extractor = snode->extractors[i];
    serializer(extractor.num_elements_from_root);
    serializer(extractor.shape);
    serializer(extractor.acc_shape);
    serializer(extractor.active);
  }
  serializer(snode->index_offsets);
  serializer(snode->num_active_indices);
  serializer(snode->physical_index_position);
  serializer(snode->id);
  serializer(snode->depth);
  serializer(snode->name);
  serializer(snode->num_cells_per_container);
  serializer(snode->chunk_size);
  serializer(snode->cell_size_bytes);
  serializer(snode->offset_bytes_in_parent_cell);
  serializer(snode->dt->to_string());
  serializer(snode->has_ambient);
  if (!snode->ambient_val.dt->is_primitive(PrimitiveTypeID::unknown)) {
    serializer(snode->ambient_val.stringify());
  }
  if (snode->grad_info && !snode->grad_info->is_primal()) {
    if (auto *adjoint_snode = snode->grad_info->adjoint_snode()) {
      get_offline_cache_key_of_snode_impl(adjoint_snode, serializer, visited);
    }
    if (auto *dual_snode = snode->grad_info->dual_snode()) {
      get_offline_cache_key_of_snode_impl(dual_snode, serializer, visited);
    }
  }
  if (snode->physical_type) {
    serializer(snode->physical_type->to_string());
  }
  serializer(snode->id_in_bit_struct);
  serializer(snode->is_bit_level);
  serializer(snode->is_path_all_dense);
  serializer(snode->node_type_name);
  serializer(snode->type);
  serializer(snode->_morton);
  serializer(snode->get_snode_tree_id());
}

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode) {
  QD_ASSERT(snode);

  BinaryOutputSerializer serializer;
  serializer.initialize();
  {
    std::unordered_set<int> visited;
    get_offline_cache_key_of_snode_impl(snode, serializer, visited);
  }
  serializer.finalize();

  picosha2::hash256_one_by_one hasher;
  hasher.process(serializer.data.begin(), serializer.data.end());
  hasher.finish();

  return picosha2::get_hash_hex_string(hasher);
}

std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel) {
  std::vector<std::uint8_t> kernel_params_string, kernel_rets_string;
  std::string kernel_body_string;
  if (kernel) {  // param_list, rets, body
    kernel_params_string = get_offline_cache_key_of_parameter_list(kernel->parameter_list);
    kernel_rets_string = get_offline_cache_key_of_rets(kernel->rets);
    std::ostringstream oss;
    gen_offline_cache_key(kernel->ir.get(), &oss);
    kernel_body_string = oss.str();
  }

  auto compile_config_key = get_offline_cache_key_of_compile_config(config);
  auto device_caps_key = get_offline_cache_key_of_device_caps(caps);
  std::string autodiff_mode = std::to_string(static_cast<std::size_t>(kernel->autodiff_mode));
  picosha2::hash256_one_by_one hasher;
  hasher.process(compile_config_key.begin(), compile_config_key.end());
  hasher.process(device_caps_key.begin(), device_caps_key.end());
  hasher.process(kernel_params_string.begin(), kernel_params_string.end());
  hasher.process(kernel_rets_string.begin(), kernel_rets_string.end());
  hasher.process(kernel_body_string.begin(), kernel_body_string.end());
  hasher.process(autodiff_mode.begin(), autodiff_mode.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(), 'T');  // The key must start with a letter
  return res;
}

std::string get_hashed_offline_cache_key_of_device_caps(const DeviceCapabilityConfig &caps) {
  auto device_caps_key = get_offline_cache_key_of_device_caps(caps);
  picosha2::hash256_one_by_one hasher;
  hasher.process(device_caps_key.begin(), device_caps_key.end());
  hasher.finish();
  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(), 'T');  // The key must start with a letter
  return res;
}

namespace {

// Serialize a task as its printed CHI-IR text (opcodes, SSA-named operands, types, immediates, offload header).
// Stable across processes only because `re_id` just renumbered the task. The printer omits some codegen-relevant
// state -- see the out-of-band fields in get_hashed_per_task_cache_key and the eligibility gate in codegen.cpp.
std::string serialize_task_body(OffloadedStmt *task) {
  std::string s;
  irpass::print(task, &s, /*print_ir_dbg_info=*/false, /*print_kernel_wrapper=*/false);
  return s;
}

// SNode tree roots the task touches, for the key's layout signature. Over-approximating (extra trees) only costs
// dedup precision, not soundness.
std::vector<const SNode *> gather_task_snode_roots(OffloadedStmt *task) {
  std::map<int, const SNode *> roots;  // tree_id -> root (sorted, dedup)
  auto add = [&roots](const SNode *sn) {
    if (sn == nullptr) {
      return;
    }
    const SNode *root = sn->get_root();
    roots[root->get_snode_tree_id()] = root;
  };
  irpass::analysis::gather_statements(task, [&add](Stmt *stmt) {
    if (auto *s = stmt->cast<GlobalPtrStmt>()) {
      add(s->snode);
    } else if (auto *s = stmt->cast<SNodeOpStmt>()) {
      add(s->snode);
    } else if (auto *s = stmt->cast<SNodeLookupStmt>()) {
      add(s->snode);
    } else if (auto *s = stmt->cast<GetChStmt>()) {
      add(s->input_snode);
      add(s->output_snode);
    } else if (auto *s = stmt->cast<ClearListStmt>()) {
      // A clear-list offload has no header SNode; its target lives only on the body's ClearListStmt.
      add(s->snode);
    }
    return false;
  });
  // listgen / gc / struct_for carry their target SNode on the offload header, not in the body.
  add(task->snode);
  std::vector<const SNode *> out;
  out.reserve(roots.size());
  for (const auto &[tree_id, root] : roots) {
    out.push_back(root);
  }
  return out;
}

}  // namespace

std::string get_hashed_per_task_cache_key(const CompileConfig &config,
                                          const DeviceCapabilityConfig &caps,
                                          OffloadedStmt *task,
                                          const Kernel *kernel) {
  QD_ASSERT(task);
  QD_ASSERT(kernel);
  auto compile_config_key = get_offline_cache_key_of_compile_config(config);
  auto device_caps_key = get_offline_cache_key_of_device_caps(caps);
  // Args/returns flow through the kernel's context struct, whose layout is the kernel ABI, not the task body:
  // identical bodies under different ABIs must not share a module.
  auto kernel_params_key = get_offline_cache_key_of_parameter_list(kernel->parameter_list);
  auto kernel_rets_key = get_offline_cache_key_of_rets(kernel->rets);
  std::string task_body_string = serialize_task_body(task);
  std::string autodiff_mode_string = std::to_string(static_cast<std::size_t>(kernel->autodiff_mode));

  picosha2::hash256_one_by_one hasher;
  hasher.process(compile_config_key.begin(), compile_config_key.end());
  hasher.process(device_caps_key.begin(), device_caps_key.end());
  hasher.process(kernel_params_key.begin(), kernel_params_key.end());
  hasher.process(kernel_rets_key.begin(), kernel_rets_key.end());
  // Per touched tree, its layout signature + id: struct-access code is inlined and referenced by tree id, so the
  // printed IR's SNode name alone is not enough -- a cached module is reusable only for the same tree instance.
  for (const SNode *root : gather_task_snode_roots(task)) {
    std::string snode_key = get_hashed_offline_cache_key_of_snode(root);
    hasher.process(snode_key.begin(), snode_key.end());
    std::string tree_id_key = std::to_string(root->get_snode_tree_id());
    hasher.process(tree_id_key.begin(), tree_id_key.end());
  }
  hasher.process(task_body_string.begin(), task_body_string.end());
  hasher.process(autodiff_mode_string.begin(), autodiff_mode_string.end());
  // Graph-region tags (gdw level, stream-parallel group, graph_parallel_region, checkpoint): the printed body carries
  // only gdw_level, but all four steer codegen / launcher metadata (cf. gen_offline_cache_key's emit_graph_region_key).
  std::string region_key = std::to_string(task->graph_do_while_level_id) + ":" +
                           std::to_string(task->stream_parallel_group_id) + ":" +
                           std::to_string(task->graph_parallel_region_id) + ":" + std::to_string(task->checkpoint_id);
  hasher.process(region_key.begin(), region_key.end());
  // bit_vectorized: omitted by the offload printer, but codegen branches on it (create_offload_struct_for) to pick a
  // different traversal, so the same body under each setting must key differently.
  std::string bit_vectorized_key = task->is_bit_vectorized ? "bv1" : "bv0";
  hasher.process(bit_vectorized_key.begin(), bit_vectorized_key.end());
  // Quadrants build version: unlike the whole-kernel `.qdc` (version-gated by its metadata), this tier has no version
  // gate, so without this an upgrade would reuse the old build's PTX under a byte-identical key.
  std::string version_string = std::to_string(QD_VERSION_MAJOR) + "." + std::to_string(QD_VERSION_MINOR) + "." +
                               std::to_string(QD_VERSION_PATCH);
  hasher.process(version_string.begin(), version_string.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(), 'K');  // task-key prefix; a letter, distinct from the 'T' kernel key
  return res;
}

}  // namespace quadrants::lang
