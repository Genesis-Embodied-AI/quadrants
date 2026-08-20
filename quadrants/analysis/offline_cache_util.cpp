#include "offline_cache_util.h"

#include "quadrants/common/core.h"
#include "quadrants/common/serialization.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/offloaded_task_type.h"
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
    serializer(config.gpu_max_reg);
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

// Snode-root gather for a PRE-offload IR node (covers `StructForStmt`, whose `snode` is the loop domain before
// `offload` lowers it into a struct_for task). Over-approx (extra trees) only costs dedup precision, never soundness.
// Deterministic order: keyed by snode-tree id.
static std::vector<const SNode *> gather_ir_snode_roots(IRNode *node) {
  std::map<int, const SNode *> roots;  // tree_id -> root (sorted, dedup)
  auto add = [&roots](const SNode *sn) {
    if (sn == nullptr) {
      return;
    }
    const SNode *root = sn->get_root();
    roots[root->get_snode_tree_id()] = root;
  };
  // `include_containers`: the two branches below that matter most here, `StructForStmt` and `struct_for`
  // `OffloadedStmt`, are container statements, and the predicate is not offered those by default.
  irpass::analysis::gather_statements(
      node,
      [&add](Stmt *stmt) {
        if (auto *s = stmt->cast<GlobalPtrStmt>()) {
          add(s->snode);
        } else if (auto *s = stmt->cast<SNodeOpStmt>()) {
          add(s->snode);
        } else if (auto *s = stmt->cast<SNodeLookupStmt>()) {
          add(s->snode);
        } else if (auto *s = stmt->cast<GetChStmt>()) {
          add(s->input_snode);
          add(s->output_snode);
        } else if (auto *s = stmt->cast<StructForStmt>()) {
          add(s->snode);
        } else if (auto *s = stmt->cast<OffloadedStmt>()) {
          if (s->task_type == OffloadedTaskType::struct_for) {
            add(s->snode);
          }
        }
        return false;
      },
      /*include_containers=*/true);
  std::vector<const SNode *> out;
  out.reserve(roots.size());
  for (const auto &[tree_id, root] : roots) {
    out.push_back(root);
  }
  return out;
}

std::string get_hashed_per_construct_cache_key(const CompileConfig &config,
                                               IRNode *construct,
                                               const Kernel *kernel) {
  QD_ASSERT(construct);
  QD_ASSERT(kernel);
  auto compile_config_key = get_offline_cache_key_of_compile_config(config);
  // The construct's compiled tasks read args / write returns through the kernel's context struct, whose layout is the
  // kernel's parameter/return ABI. Same soundness argument as the per-task key: two byte-identical constructs in
  // kernels with different ABIs must not share a cached frontend result.
  auto kernel_params_key = get_offline_cache_key_of_parameter_list(kernel->parameter_list);
  auto kernel_rets_key = get_offline_cache_key_of_rets(kernel->rets);
  std::string body_string;
  irpass::print(construct, &body_string, /*print_ir_dbg_info=*/false, /*print_kernel_wrapper=*/false);
  std::string autodiff_mode_string = std::to_string(static_cast<std::size_t>(kernel->autodiff_mode));

  picosha2::hash256_one_by_one hasher;
  hasher.process(compile_config_key.begin(), compile_config_key.end());
  hasher.process(kernel_params_key.begin(), kernel_params_key.end());
  hasher.process(kernel_rets_key.begin(), kernel_rets_key.end());
  // Fold only the tree id (a cheap int), NOT the O(tree_size) full-layout hash that
  // `get_hashed_offline_cache_key_of_snode` computes (doing that per construct on a big genesis tree was a >20x
  // cold-compile blowup). Device caps are likewise omitted. Both omissions are cost choices whose soundness rests on
  // the consuming cache's scope; the cross-process manifest that consumes this key is responsible for folding caps
  // and for handling tree-id recycling.
  for (const SNode *root : gather_ir_snode_roots(construct)) {
    std::string tree_id_key = std::to_string(root->get_snode_tree_id());
    hasher.process(tree_id_key.begin(), tree_id_key.end());
  }
  // Fold each statement's graph region tags: graph_do_while_level_id, stream_parallel_group_id,
  // graph_parallel_region_id, checkpoint_id. These are NOT in the printed IR the body hash uses (the printer emits
  // `gdw_level` only for post-offload OffloadedStmts), yet `offload` copies them onto the offloaded tasks, and the
  // runtime rebuilds the graph_do_while level tree (offloads are its leaves) / stream-parallel groups / checkpoint
  // gating purely from these per-task tags. Two constructs with identical bodies but different tags MUST get distinct
  // keys, else a cached clone carries the wrong level and the host graph_do_while loop mis-executes (observed as a
  // non-terminating genesis `_step_kernel`). For statements carry the tags in direct fields; serial statements carry
  // them in the base `region_tag`. Fold both (redundant folds are harmless); traversal order is deterministic.
  std::string tag_string;
  auto fold_tags = [&tag_string](char kind, int lvl, int grp, int reg, int cp) {
    tag_string += kind;
    tag_string += std::to_string(lvl) + "," + std::to_string(grp) + "," + std::to_string(reg) + "," +
                  std::to_string(cp) + ";";
  };
  // `include_containers`: the loops whose tags this is folding are themselves container statements.
  for (Stmt *s : irpass::analysis::gather_statements(
           construct, [](Stmt *) { return true; }, /*include_containers=*/true)) {
    auto [lvl, grp, reg, cp] = s->region_tag.cache_key_members();
    fold_tags('t', lvl, grp, reg, cp);
    if (auto *r = s->cast<RangeForStmt>()) {
      fold_tags('r', r->graph_do_while_level_id, r->stream_parallel_group_id, r->graph_parallel_region_id,
                r->checkpoint_id);
    } else if (auto *sf = s->cast<StructForStmt>()) {
      fold_tags('s', sf->graph_do_while_level_id, sf->stream_parallel_group_id, sf->graph_parallel_region_id,
                sf->checkpoint_id);
    }
  }
  hasher.process(tag_string.begin(), tag_string.end());
  hasher.process(body_string.begin(), body_string.end());
  hasher.process(autodiff_mode_string.begin(), autodiff_mode_string.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(), 'C');  // construct-key prefix; distinct from 'K' (task) and 'T' (kernel)
  return res;
}

}  // namespace quadrants::lang
