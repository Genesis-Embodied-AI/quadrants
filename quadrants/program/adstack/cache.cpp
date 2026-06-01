#include "quadrants/program/adstack/cache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/common/logging.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/adstack/diagnose.h"
#include "quadrants/program/adstack/eval.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {

namespace {

// Decide whether the input that `obs` describes is still consistent with the recorded state. Returns true iff the
// cached `SizeExprCacheEntry` is still valid for this observation. FieldLoadObs / ExternalReadObs use the per-buffer
// gen counter (`snode_write_gen` / `ndarray_data_gen`) as the sole staleness signal: a gen-counter advance forces a
// re-walk regardless of whether the read cells themselves changed. ExternalShapeObs has no gen counter (shapes are
// launch-arg metadata, not buffer content), so it falls back to value comparison against `observed_value`.
bool replay_observation_is_fresh(const AdStackCache::SizeExprReadObservation &obs,
                                 Program *prog,
                                 LaunchContextBuilder *ctx) {
  using Obs = AdStackCache::SizeExprReadObservation;
  switch (obs.kind) {
    case Obs::FieldLoadObs:
      return prog != nullptr && prog->adstack_cache().snode_write_gen(obs.snode_id) == obs.observed_gen;
    case Obs::ExternalReadObs: {
      if (ctx == nullptr || obs.arg_id_path.empty() || prog == nullptr) {
        return false;
      }
      int arg_id = obs.arg_id_path[0];
      ArgArrayPtrKey key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      auto it = ctx->array_ptrs.find(key);
      if (it == ctx->array_ptrs.end()) {
        return false;
      }
      void *data_ptr = it->second;
      return data_ptr == obs.observed_devalloc && prog->adstack_cache().ndarray_data_gen(data_ptr) == obs.observed_gen;
    }
    case Obs::ExternalShapeObs: {
      if (ctx == nullptr) {
        return false;
      }
      std::vector<int> arg_indices(obs.arg_id_path.begin(), obs.arg_id_path.end());
      arg_indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
      arg_indices.push_back(obs.arg_shape_axis);
      return static_cast<int64_t>(ctx->get_struct_arg_host<int32_t>(arg_indices)) == obs.observed_value;
    }
  }
  return false;
}

}  // namespace

void AdStackCache::note_observations(const std::vector<SizeExprReadObservation> &reads) {
  for (const auto &obs : reads) {
    if (obs.kind == SizeExprReadObservation::FieldLoadObs) {
      observed_snode_ids_.insert(obs.snode_id);
    } else if (obs.kind == SizeExprReadObservation::ExternalReadObs) {
      any_external_read_observed_ = true;
      if (obs.observed_devalloc != nullptr) {
        observed_devalloc_ptrs_.insert(obs.observed_devalloc);
      }
    }
  }
}

void AdStackCache::note_per_task_dependencies(const std::vector<std::pair<int, uint64_t>> &snode_gens,
                                              const std::vector<std::tuple<int, void *, uint64_t>> &arg_gens) {
  for (const auto &kv : snode_gens) {
    observed_snode_ids_.insert(kv.first);
  }
  if (!arg_gens.empty()) {
    any_external_read_observed_ = true;
    for (const auto &tup : arg_gens) {
      void *devalloc = std::get<1>(tup);
      if (devalloc != nullptr) {
        observed_devalloc_ptrs_.insert(devalloc);
      }
    }
  }
}

void AdStackCache::note_per_task_dependencies(const std::vector<std::pair<int, uint64_t>> &snode_gens,
                                              const std::vector<ArgGenObservation> &arg_gens) {
  for (const auto &kv : snode_gens) {
    observed_snode_ids_.insert(kv.first);
  }
  if (!arg_gens.empty()) {
    any_external_read_observed_ = true;
    for (const auto &dep : arg_gens) {
      if (dep.devalloc != nullptr) {
        observed_devalloc_ptrs_.insert(dep.devalloc);
      }
    }
  }
}

bool AdStackCache::try_size_expr_cache_hit(Program *prog,
                                           const SerializedSizeExpr *expr_key,
                                           LaunchContextBuilder *ctx,
                                           int64_t &out_result) {
  auto it = size_expr_cache_.find(expr_key);
  if (it == size_expr_cache_.end()) {
    return false;
  }
  const auto &entry = it->second;
  for (const auto &obs : entry.reads) {
    if (!replay_observation_is_fresh(obs, prog, ctx)) {
      size_expr_cache_.erase(it);
      return false;
    }
  }
  out_result = entry.result;
  return true;
}

void AdStackCache::record_size_expr_eval(const SerializedSizeExpr *expr_key,
                                         int64_t result,
                                         std::vector<SizeExprReadObservation> reads) {
  note_observations(reads);
  size_expr_cache_[expr_key] = SizeExprCacheEntry{result, std::move(reads)};
  any_recordings_ = true;
}

namespace {
// Pack a `(registry_id, stack_id, mor_node_idx)` triple into a 64-bit map key. The recognizer caps both `stack_id` and
// `mor_node_idx` at O(10s) per task (per-task adstack count and per-stack node count are both small), well within 16
// bits each, so the packed encoding never collides. `registry_id` uses the full 32 bits since the program-side registry
// can grow to thousands of entries across a long-running session.
inline uint64_t pack_max_reducer_key(uint32_t registry_id, int32_t stack_id, int32_t mor_node_idx) {
  return (static_cast<uint64_t>(registry_id) & 0xFFFFFFFFull) | ((static_cast<uint64_t>(stack_id) & 0xFFFFull) << 32) |
         ((static_cast<uint64_t>(mor_node_idx) & 0xFFFFull) << 48);
}
}  // namespace

bool AdStackCache::try_max_reducer_cache_hit(uint32_t registry_id,
                                             int32_t stack_id,
                                             int32_t mor_node_idx,
                                             LaunchContextBuilder *ctx,
                                             int64_t &out_result) {
  auto it = max_reducer_cache_.find(pack_max_reducer_key(registry_id, stack_id, mor_node_idx));
  if (it == max_reducer_cache_.end()) {
    return false;
  }
  const auto &entry = it->second;
  for (const auto &obs : entry.reads) {
    if (!replay_observation_is_fresh(obs, prog_, ctx)) {
      max_reducer_cache_.erase(it);
      return false;
    }
  }
  out_result = entry.result;
  return true;
}

void populate_max_reducer_body_observations(std::vector<AdStackCache::SizeExprReadObservation> &reads,
                                            LaunchContextBuilder *ctx,
                                            AdStackCache *cache) {
  for (auto &obs : reads) {
    if (obs.kind == AdStackCache::SizeExprReadObservation::FieldLoadObs) {
      // Snapshot the snode write generation so a subsequent launch that has not mutated the SNode replays as a cache
      // hit via the gen-counter check in `replay_observation_is_fresh`. `observed_value` is unused for FieldLoadObs
      // (gen counter is the sole staleness signal) and left at its default-constructed value.
      if (cache != nullptr) {
        obs.observed_gen = cache->snode_write_gen(obs.snode_id);
      }
      continue;
    }
    if (obs.kind != AdStackCache::SizeExprReadObservation::ExternalReadObs || obs.arg_id_path.empty()) {
      continue;
    }
    if (ctx == nullptr) {
      continue;
    }
    int arg_id = obs.arg_id_path[0];
    ArgArrayPtrKey key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
    auto it = ctx->array_ptrs.find(key);
    if (it == ctx->array_ptrs.end()) {
      continue;
    }
    obs.observed_devalloc = it->second;
    // Snapshot the data-pointer + ndarray data generation so a subsequent launch with the same `DeviceAllocation` and
    // an unbumped gen replays as a cache hit. `observed_value` is unused for ExternalReadObs (gen counter is the sole
    // staleness signal) and left at its default-constructed value.
    if (cache != nullptr) {
      obs.observed_gen = cache->ndarray_data_gen(it->second);
    }
  }
}

const std::vector<AdStackCache::SizeExprReadObservation> *
AdStackCache::lookup_max_reducer_reads(uint32_t registry_id, int32_t stack_id, int32_t mor_node_idx) const {
  auto it = max_reducer_cache_.find(pack_max_reducer_key(registry_id, stack_id, mor_node_idx));
  if (it == max_reducer_cache_.end()) {
    return nullptr;
  }
  return &it->second.reads;
}

void AdStackCache::record_max_reducer_eval(uint32_t registry_id,
                                           int32_t stack_id,
                                           int32_t mor_node_idx,
                                           int64_t result,
                                           std::vector<SizeExprReadObservation> reads) {
  note_observations(reads);
  max_reducer_cache_[pack_max_reducer_key(registry_id, stack_id, mor_node_idx)] =
      MaxReducerCacheEntry{result, std::move(reads)};
  ++max_reducer_dispatch_count_;
  any_recordings_ = true;
}

bool AdStackCache::try_max_reducer_launch_cache_hit(
    const void *launch_cache_key,
    LaunchContextBuilder *ctx,
    std::shared_ptr<const std::unordered_map<uint64_t, int64_t>> &out_result) {
  if (launch_cache_key == nullptr || ctx == nullptr) {
    return false;
  }
  auto it = max_reducer_launch_cache_.find(launch_cache_key);
  if (it == max_reducer_launch_cache_.end()) {
    return false;
  }
  const auto &entry = it->second;
  for (const auto &kv : entry.snode_gens) {
    if (snode_write_gen(kv.first) != kv.second) {
      return false;
    }
  }
  for (const auto &dep : entry.arg_gens) {
    ArgArrayPtrKey key{dep.arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
    auto ap_it = ctx->array_ptrs.find(key);
    void *current_devalloc = (ap_it == ctx->array_ptrs.end()) ? nullptr : ap_it->second;
    if (current_devalloc != dep.devalloc || ndarray_data_gen(current_devalloc) != dep.gen) {
      return false;
    }
  }
  // Hand back a `shared_ptr` copy of the cached result. Refcount bump only - no map copy. The cache entry retains its
  // own ownership, so the caller's transient stays valid even after a recursive reentry rewrites the executor's
  // `current_max_reducer_results_` field.
  out_result = entry.result;
  return true;
}

void AdStackCache::record_max_reducer_launch_cache(const void *launch_cache_key,
                                                   const std::vector<const AdStackSizingInfo *> &ad_stacks,
                                                   std::shared_ptr<const std::unordered_map<uint64_t, int64_t>> result,
                                                   LaunchContextBuilder *ctx) {
  if (launch_cache_key == nullptr || ctx == nullptr) {
    return;
  }
  // Aggregate every spec's observation deps into a deduplicated `(snode_id -> gen)` map and `(arg_id -> (devalloc,
  // gen))` map. The fast-path replay walks these maps once per launch; deduplication keeps the replay O(distinct deps)
  // instead of O(specs * obs/spec). `lookup_max_reducer_reads` returns the per-spec observations recorded by either a
  // fresh `record_max_reducer_eval` or a still-warm `populate_max_reducer_body_observations` call earlier in this
  // launch.
  MaxReducerLaunchCacheEntry entry;
  entry.result = std::move(result);
  std::unordered_map<int, uint64_t> snode_gens_map;
  std::unordered_map<int, std::pair<void *, uint64_t>> arg_gens_map;
  for (const auto *ad_stack_ptr : ad_stacks) {
    if (ad_stack_ptr == nullptr) {
      continue;
    }
    const auto &ad_stack = *ad_stack_ptr;
    if (ad_stack.max_reducer_specs.empty() || ad_stack.registry_id == 0) {
      continue;
    }
    for (const auto &spec : ad_stack.max_reducer_specs) {
      const auto *reads = lookup_max_reducer_reads(ad_stack.registry_id, spec.stack_id, spec.mor_node_idx);
      if (reads == nullptr) {
        continue;
      }
      for (const auto &obs : *reads) {
        if (obs.kind == SizeExprReadObservation::FieldLoadObs) {
          if (obs.snode_id >= 0) {
            snode_gens_map[obs.snode_id] = snode_write_gen(obs.snode_id);
          }
        } else if (obs.kind == SizeExprReadObservation::ExternalReadObs) {
          if (!obs.arg_id_path.empty()) {
            const int arg_id = obs.arg_id_path[0];
            ArgArrayPtrKey key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
            auto ap_it = ctx->array_ptrs.find(key);
            void *devalloc = (ap_it == ctx->array_ptrs.end()) ? nullptr : ap_it->second;
            arg_gens_map[arg_id] = {devalloc, ndarray_data_gen(devalloc)};
          }
        }
      }
    }
  }
  entry.snode_gens.reserve(snode_gens_map.size());
  for (const auto &kv : snode_gens_map) {
    entry.snode_gens.emplace_back(kv.first, kv.second);
  }
  entry.arg_gens.reserve(arg_gens_map.size());
  for (const auto &kv : arg_gens_map) {
    entry.arg_gens.push_back({kv.first, kv.second.first, kv.second.second});
  }
  // Mirror the deduplicated dependency footprint into the per-id observed sets; safe to do after `entry` is finalised
  // because `note_per_task_dependencies` reads the snode-gen pair's first element and walks the `arg_gens` list to
  // pull each `(arg_id, devalloc, gen)` triple's devalloc into `observed_devalloc_ptrs_`.
  note_per_task_dependencies(entry.snode_gens, entry.arg_gens);
  max_reducer_launch_cache_[launch_cache_key] = std::move(entry);
  any_recordings_ = true;
}

bool AdStackCache::try_spirv_bytecode_cache_hit(Program *prog,
                                                const void *attribs_key,
                                                LaunchContextBuilder *ctx,
                                                std::vector<uint8_t> &out_bytecode) {
  auto it = spirv_bytecode_cache_.find(attribs_key);
  if (it == spirv_bytecode_cache_.end()) {
    return false;
  }
  const auto &entry = it->second;
  for (const auto &obs : entry.reads) {
    if (!replay_observation_is_fresh(obs, prog, ctx)) {
      spirv_bytecode_cache_.erase(it);
      return false;
    }
  }
  out_bytecode = entry.bytecode;
  return true;
}

void AdStackCache::record_spirv_bytecode_eval(const void *attribs_key,
                                              std::vector<uint8_t> bytecode,
                                              std::vector<SizeExprReadObservation> reads) {
  note_observations(reads);
  spirv_bytecode_cache_[attribs_key] = SpirvBytecodeCacheEntry{std::move(bytecode), std::move(reads)};
  any_recordings_ = true;
}

void AdStackCache::record_per_task_ad_stack(const void *attribs_key,
                                            std::vector<uint32_t> metadata,
                                            uint32_t stride_float,
                                            uint32_t stride_int,
                                            std::vector<std::pair<int, uint64_t>> snode_gens,
                                            std::vector<std::tuple<int, void *, uint64_t>> arg_gens) {
  note_per_task_dependencies(snode_gens, arg_gens);
  per_task_ad_stack_cache_[attribs_key] = PerTaskAdStackCacheEntry{std::move(metadata), stride_float, stride_int,
                                                                   std::move(snode_gens), std::move(arg_gens)};
  any_recordings_ = true;
}

bool AdStackCache::try_per_task_ad_stack_cache_hit(const void *attribs_key,
                                                   LaunchContextBuilder *ctx,
                                                   PerTaskAdStackCacheEntry &out) {
  auto it = per_task_ad_stack_cache_.find(attribs_key);
  if (it == per_task_ad_stack_cache_.end()) {
    return false;
  }
  const auto &entry = it->second;
  for (const auto &snode_pair : entry.snode_gens) {
    if (snode_write_gen(snode_pair.first) != snode_pair.second) {
      per_task_ad_stack_cache_.erase(it);
      return false;
    }
  }
  for (const auto &arg_tuple : entry.arg_gens) {
    int arg_id = std::get<0>(arg_tuple);
    void *recorded_devalloc = std::get<1>(arg_tuple);
    uint64_t recorded_gen = std::get<2>(arg_tuple);
    void *current_devalloc = nullptr;
    if (ctx != nullptr) {
      ArgArrayPtrKey key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      auto ap_it = ctx->array_ptrs.find(key);
      if (ap_it != ctx->array_ptrs.end()) {
        current_devalloc = ap_it->second;
      }
    }
    if (current_devalloc != recorded_devalloc) {
      per_task_ad_stack_cache_.erase(it);
      return false;
    }
    if (ndarray_data_gen(recorded_devalloc) != recorded_gen) {
      per_task_ad_stack_cache_.erase(it);
      return false;
    }
  }
  out = entry;
  return true;
}

void AdStackCache::record_llvm_per_task_ad_stack(const void *attribs_key,
                                                 std::vector<uint64_t> offsets,
                                                 std::vector<uint64_t> max_sizes,
                                                 uint64_t stride_combined,
                                                 uint64_t stride_float,
                                                 uint64_t stride_int,
                                                 std::vector<std::pair<int, uint64_t>> snode_gens,
                                                 std::vector<std::tuple<int, void *, uint64_t>> arg_gens) {
  note_per_task_dependencies(snode_gens, arg_gens);
  llvm_per_task_ad_stack_cache_[attribs_key] =
      LlvmPerTaskAdStackCacheEntry{std::move(offsets), std::move(max_sizes),  stride_combined,    stride_float,
                                   stride_int,         std::move(snode_gens), std::move(arg_gens)};
  any_recordings_ = true;
}

bool AdStackCache::try_llvm_per_task_ad_stack_cache_hit(const void *attribs_key,
                                                        LaunchContextBuilder *ctx,
                                                        LlvmPerTaskAdStackCacheEntry &out) {
  auto it = llvm_per_task_ad_stack_cache_.find(attribs_key);
  if (it == llvm_per_task_ad_stack_cache_.end()) {
    return false;
  }
  const auto &entry = it->second;
  for (const auto &snode_pair : entry.snode_gens) {
    if (snode_write_gen(snode_pair.first) != snode_pair.second) {
      llvm_per_task_ad_stack_cache_.erase(it);
      return false;
    }
  }
  for (const auto &arg_tuple : entry.arg_gens) {
    int arg_id = std::get<0>(arg_tuple);
    void *recorded_devalloc = std::get<1>(arg_tuple);
    uint64_t recorded_gen = std::get<2>(arg_tuple);
    void *current_devalloc = nullptr;
    if (ctx != nullptr) {
      ArgArrayPtrKey key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      auto ap_it = ctx->array_ptrs.find(key);
      if (ap_it != ctx->array_ptrs.end()) {
        current_devalloc = ap_it->second;
      }
    }
    if (current_devalloc != recorded_devalloc) {
      llvm_per_task_ad_stack_cache_.erase(it);
      return false;
    }
    if (ndarray_data_gen(recorded_devalloc) != recorded_gen) {
      llvm_per_task_ad_stack_cache_.erase(it);
      return false;
    }
  }
  out = entry;
  return true;
}

namespace {
// FNV-1a 64-bit, folded to 32-bit; never returns 0 (reserved sentinel). Used to derive a content-stable `registry_id`
// from the (kernel_name, task_id_in_kernel) pair. We need a deterministic hash because the id is baked as an immediate
// into LLVM IR / SPIR-V at codegen time and read back by the host on overflow: `std::hash<std::string>` is
// implementation-defined and not stable across stdlib versions, so we cannot rely on it producing the same id at
// codegen time and at offline-cache-reload-driven runtime registration.
inline uint32_t fnv1a32_for_registry(const std::string &kernel_name, int task_id_in_kernel) {
  constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
  constexpr uint64_t kFnvPrime = 0x100000001b3ULL;
  uint64_t h = kFnvOffsetBasis;
  for (unsigned char c : kernel_name) {
    h ^= c;
    h *= kFnvPrime;
  }
  // Domain separator so `(name + str(N))` and `(name_with_extra_chars)` cannot accidentally collide.
  h ^= ':';
  h *= kFnvPrime;
  uint64_t t = static_cast<uint64_t>(static_cast<uint32_t>(task_id_in_kernel));
  for (int i = 0; i < 8; ++i) {
    h ^= (t >> (i * 8)) & 0xFFu;
    h *= kFnvPrime;
  }
  uint32_t r = static_cast<uint32_t>(h ^ (h >> 32));
  return r == 0 ? 1u : r;
}
}  // namespace

uint32_t AdStackCache::register_adstack_sizing_info(const void *identity_key,
                                                    const std::string &kernel_name,
                                                    int task_id_in_kernel,
                                                    std::vector<int> allocated_max_sizes,
                                                    std::vector<SerializedSizeExpr> size_exprs) {
  std::lock_guard<std::mutex> lk(adstack_sizing_info_registry_mutex_);
  // Idempotent re-registration: same `identity_key` yields the same id across re-compiles and updates the entry's
  // metadata + size_exprs in place. The key is just an opaque dedup token - the registry never dereferences it; all
  // data needed by the diagnose path is copied into the entry below.
  //
  // BUT `identity_key` is the address of an `AdStackSizingAttribs` / `OffloadedTask::ad_stack` held by a launcher's
  // tasks vector or by a transient `current_task` during codegen. When the previous owner is freed and the allocator
  // recycles the address for a brand-new task that belongs to a DIFFERENT logical kernel, the raw-pointer lookup
  // succeeds against the stale entry. Returning the previous id would then cause the new task to inherit the old
  // task's `registry_id`, and the per-spec `max_reducer_cache_` (keyed by `(registry_id, stack_id, mor_node_idx)`)
  // would serve a stale result to the new kernel — observed as `assert dispatch_count > after_first` failing in
  // `test_max_reducer_per_kernel_registry_id_isolation` whenever the allocator happens to recycle the same address
  // across the two `qd.template()` instantiations. Guard the idempotent path on `(kernel_name, task_id_in_kernel)`
  // matching so a recycled address falls through to the content-stable hash path below, which produces (or finds)
  // the correct id for the new kernel.
  if (auto it = adstack_sizing_info_id_by_ptr_.find(identity_key); it != adstack_sizing_info_id_by_ptr_.end()) {
    auto &entry = adstack_sizing_info_registry_[it->second];
    if (entry.kernel_name == kernel_name && entry.task_id_in_kernel == task_id_in_kernel) {
      entry.allocated_max_sizes = std::move(allocated_max_sizes);
      entry.size_exprs = std::move(size_exprs);
      return it->second;
    }
    // Recycled pointer for a different kernel. Drop the stale reverse-lookup entry so the fall-through below treats
    // this as a fresh registration (the registry entry itself stays alive because the previously-registered owner
    // may still resolve its id through `lookup_adstack_sizing_info` on the overflow diagnose path).
    adstack_sizing_info_id_by_ptr_.erase(it);
  }
  // Content-stable hash. Same (kernel_name, task_id_in_kernel) yields the same id across `Program` lifetimes,
  // re-compiles, and offline-cache reloads. The codegen-emitted overflow `cmpxchg(0, registry_id)` writes this same
  // value, so the host lookup of the cmpxchg slot resolves the offending kernel + task without a pre-launch `register`
  // call from the runtime.
  uint32_t id = fnv1a32_for_registry(kernel_name, task_id_in_kernel);
  // Linear-probe past hash collisions (different `(kernel_name, task_id_in_kernel)` pairs that hash to the same id).
  // Vanishingly rare with a 32-bit FNV-1a (~1.2e-4 collision probability for 1000 distinct keys via birthday bound).
  // Same-content / different-identity_key (re-codegen of the same source after a `qd.reset()` produces a fresh
  // `ad_stack` at a new address but the hash inputs are unchanged) MUST keep the existing id; otherwise the codegen-
  // baked immediate in the cached LLVM IR points at one entry while the runtime registration mints another, defeating
  // the content-stable contract. Detect that case by comparing `(kernel_name, task_id_in_kernel)` against the slot's
  // already-stored values; the `identity_key`-equal case cannot land here because the early-out above already returned
  // for any already-registered key. Skip id 0 (reserved sentinel).
  while (true) {
    auto reg_it = adstack_sizing_info_registry_.find(id);
    if (reg_it == adstack_sizing_info_registry_.end()) {
      break;
    }
    if (reg_it->second.kernel_name == kernel_name && reg_it->second.task_id_in_kernel == task_id_in_kernel) {
      // Same content (same hash inputs), different `identity_key`. Update in place and add the new identity_key to the
      // reverse lookup so the new pointer resolves to this same id; the previous identity_key's entry stays valid and
      // continues to resolve here too.
      reg_it->second.allocated_max_sizes = std::move(allocated_max_sizes);
      reg_it->second.size_exprs = std::move(size_exprs);
      adstack_sizing_info_id_by_ptr_.emplace(identity_key, id);
      return id;
    }
    ++id;
    if (id == 0) {
      ++id;  // skip the sentinel
    }
  }
  AdStackSizingInfoEntry entry;
  entry.identity_key = identity_key;
  entry.kernel_name = kernel_name;
  entry.task_id_in_kernel = task_id_in_kernel;
  entry.allocated_max_sizes = std::move(allocated_max_sizes);
  entry.size_exprs = std::move(size_exprs);
  adstack_sizing_info_registry_.emplace(id, std::move(entry));
  adstack_sizing_info_id_by_ptr_.emplace(identity_key, id);
  return id;
}

void AdStackCache::update_adstack_sizing_info_size_exprs(uint32_t id, std::vector<SerializedSizeExpr> size_exprs) {
  std::lock_guard<std::mutex> lk(adstack_sizing_info_registry_mutex_);
  if (id == 0) {
    return;
  }
  auto it = adstack_sizing_info_registry_.find(id);
  if (it == adstack_sizing_info_registry_.end()) {
    return;
  }
  it->second.size_exprs = std::move(size_exprs);
}

void AdStackCache::ensure_runtime_registry_ids_for_max_reducer(std::vector<OffloadedTask> &tasks) {
  for (auto &task : tasks) {
    auto &ad_stack = task.ad_stack;
    if (ad_stack.max_reducer_specs.empty()) {
      continue;
    }
    // Fast-path gate: the `&ad_stack` identity key is stable across launches of the same kernel handle (it lives in
    // `KernelLauncher::contexts_[i].offloaded_tasks`), so once we have populated the registry for it we skip every
    // subsequent launch in O(1). Without this gate, the steady-state hot path would rebuild `allocated_max_sizes` +
    // `size_exprs` and move them into `register_adstack_sizing_info` on every launch even though the entry is already
    // there - which costs a measurable fraction of the recovered FPS on long reverse-mode loops.
    //
    // FIXME: this gate is not content-aware - it only checks `adstack_sizing_info_id_by_ptr_` membership, so a
    // recycled `&ad_stack` address (cache-loaded kernel B reuses the address last held by evicted kernel A) silently
    // short-circuits the registration, and B's `(kernel_name, task_id_in_kernel)` never lands in the `Program`
    // registry. On overflow, B's codegen-baked cmpxchg id then resolves to A's stale entry via the diagnose path.
    // Same recycled-pointer bug class fixed in-process by `register_adstack_sizing_info` (this PR); cache-reload
    // path remains exposed. Fix: introduce a content-validating `is_adstack_sizing_info_registered_with_content(
    // identity_key, kernel_name, task_id_in_kernel)` variant and call it here, so the gate only short-circuits when
    // the live entry's `(kernel_name, task_id_in_kernel)` matches. Cheap (string compare + int compare) and
    // preserves the FPS-sensitive fast path. Not pinned by any existing test - the cache-reload + pointer-recycling
    // combo is rare; `test_max_reducer_registry_seeded_on_offline_cache_reload` covers reload but does not force a
    // recycled address.
    if (is_adstack_sizing_info_registered(static_cast<const void *>(&ad_stack))) {
      continue;
    }
    // After offline-cache load, `ad_stack.registry_id` carries the codegen-baked content-stable hash (now serialised),
    // which already matches the immediate baked into the LLVM IR's overflow `cmpxchg`. The dispatcher and the
    // metadata-publish substitution helper read `registry_id` directly off the ad_stack, so they work without us
    // touching the `Program`-level registry. We still re-populate that registry here on the FIRST launch so the
    // diagnose-on-overflow path can resolve the cmpxchg-recorded id to a kernel + task name; without this seed the
    // `Program` registry would stay empty for cache-loaded kernels and an overflow would print the generic dual-cause
    // fallback. Same `(kernel_name, task_id_in_kernel)` always hashes to the same id, so the registered id matches the
    // one we already have on `ad_stack`.
    std::vector<int> allocated_max_sizes;
    allocated_max_sizes.reserve(ad_stack.allocas.size());
    for (const auto &a : ad_stack.allocas) {
      allocated_max_sizes.push_back(static_cast<int>(a.max_size_compile_time));
    }
    std::vector<SerializedSizeExpr> size_exprs(ad_stack.size_exprs.begin(), ad_stack.size_exprs.end());
    uint32_t id =
        register_adstack_sizing_info(static_cast<const void *>(&ad_stack), ad_stack.kernel_name,
                                     ad_stack.task_id_in_kernel, std::move(allocated_max_sizes), std::move(size_exprs));
    // Defensive: if `registry_id` was never populated (very-stale cache, fresh codegen path that skipped the task-START
    // register call due to a null `Program *`), seed it now from the just-minted id. New kernels already have this set
    // by `codegen_llvm.cpp::finalize_offloaded_task_function`.
    if (ad_stack.registry_id == 0) {
      ad_stack.registry_id = id;
    }
  }
}

bool AdStackCache::is_adstack_sizing_info_registered(const void *identity_key) const {
  std::lock_guard<std::mutex> lk(adstack_sizing_info_registry_mutex_);
  return adstack_sizing_info_id_by_ptr_.find(identity_key) != adstack_sizing_info_id_by_ptr_.end();
}

std::optional<AdStackCache::AdStackSizingInfoEntry> AdStackCache::lookup_adstack_sizing_info(uint32_t id) const {
  std::lock_guard<std::mutex> lk(adstack_sizing_info_registry_mutex_);
  if (id == 0) {
    return std::nullopt;
  }
  auto it = adstack_sizing_info_registry_.find(id);
  if (it == adstack_sizing_info_registry_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::string AdStackCache::diagnose_adstack_overflow_message(uint32_t task_id) const {
  return diagnose_adstack_overflow(task_id).message;
}

AdStackCache::AdStackOverflowDiagnosis AdStackCache::diagnose_adstack_overflow(uint32_t task_id) const {
  // Lazy LLVM capture: if the launcher stashed a pending ctx pointer for this launch (LLVM defers eager
  // capture to avoid the per-launch snapshot cost), capture now before walking size_exprs. SPIR-V already
  // captured eagerly at launch, so `pending_launch_ctx_` is null there.
  if (pending_launch_ctx_ != nullptr) {
    const_cast<AdStackCache *>(this)->capture_diagnose_snapshot(*pending_launch_ctx_);
  }
  std::string identity_block;
  std::string disambiguation_block;
  // Cause classifier: when the synchronous re-run produces required > allocated for ANY stack, the most likely
  // cause is an untracked tensor mutation (DLPack-bypass etc.). When all required <= allocated, the pre-pass
  // undersized the bound (Quadrants bug). When we cannot re-evaluate (e.g. no captured launch snapshot, or a
  // leaf type the diagnose evaluator does not support) we fall through to the static dual-cause body.
  enum class Cause { Unknown, DLPackBypass, QuadrantsBug };
  Cause cause = Cause::Unknown;

  if (task_id != 0) {
    auto entry_opt = lookup_adstack_sizing_info(task_id);
    if (entry_opt.has_value()) {
      const auto &entry = *entry_opt;
      identity_block = "  Offending task: kernel `" + entry.kernel_name + "` offload task #" +
                       std::to_string(entry.task_id_in_kernel) + "; per-stack allocated max_size = [";
      for (size_t i = 0; i < entry.allocated_max_sizes.size(); ++i) {
        if (i != 0) {
          identity_block += ", ";
        }
        identity_block += std::to_string(entry.allocated_max_sizes[i]);
      }
      identity_block += "].\n";

      // Synchronous sizer rerun: walk each stack's `SerializedSizeExpr` and evaluate against the live host /
      // SNode state. Stacks whose tree contains an `ExternalTensorShape` or `ExternalTensorRead` leaf go
      // through the snapshot-based `evaluate_adstack_size_expr_for_diagnose` (see its declaration for the
      // `Device::map` design rationale). Pure host-resolvable trees go through the standard host evaluator.
      // The disambiguation is best-effort: if every stack's tree resolves we get a precise classification;
      // otherwise we report what we have and fall back to the static dual-cause hint.
      if (!entry.size_exprs.empty()) {
        std::vector<int64_t> required_sizes;
        std::vector<bool> required_known;
        size_t any_grew = 0;
        size_t any_unknown = 0;
        size_t total = std::min(entry.size_exprs.size(), entry.allocated_max_sizes.size());
        for (size_t i = 0; i < total; ++i) {
          const auto &expr = entry.size_exprs[i];
          bool host_resolvable = true;
          for (const auto &node : expr.nodes) {
            auto k = static_cast<SizeExpr::Kind>(node.kind);
            if (k == SizeExpr::Kind::ExternalTensorShape || k == SizeExpr::Kind::ExternalTensorRead) {
              host_resolvable = false;
              break;
            }
          }
          int64_t v = -1;
          if (host_resolvable && !expr.nodes.empty()) {
            // Pure host-resolvable: SNode field loads, constants, arithmetic. `ctx == nullptr` is safe because
            // every leaf we kept is host-resolvable; ETS / ETR are the only kinds that touch ctx and we
            // filtered them out.
            SizeExprLaunchScope scope;
            v = evaluate_adstack_size_expr(expr, prog_, nullptr);
          } else if (!expr.nodes.empty()) {
            // Tree contains ETR / ETS leaves. The diagnose evaluator resolves them through the captured launch
            // snapshot (`Device::map`-based ndarray reads). On failure (no snapshot, allocation cannot be
            // mapped, unsupported dtype) the helper returns -1 and we fall through to the `?` placeholder.
            int64_t diag = evaluate_adstack_size_expr_for_diagnose(expr, prog_);
            if (diag >= 0) {
              v = diag;
            }
          }
          required_sizes.push_back(v);
          required_known.push_back(!expr.nodes.empty() && v >= 0);
          if (required_known.back() && static_cast<size_t>(v) > entry.allocated_max_sizes[i]) {
            ++any_grew;
          }
          if (!required_known.back()) {
            ++any_unknown;
          }
        }
        if (any_grew > 0) {
          cause = Cause::DLPackBypass;
        } else if (any_unknown == 0 && total > 0) {
          cause = Cause::QuadrantsBug;
        }
        // Only print the rerun line when at least one stack's bound resolves to a real value. With every leaf
        // unresolved the line would be `required = [?, ?, ...]` which adds zero signal beyond the dual-cause
        // body that follows; the omission keeps the message focused on actionable content.
        if (any_unknown < total) {
          disambiguation_block = "  Synchronous sizer rerun: required max_size = [";
          for (size_t i = 0; i < required_sizes.size(); ++i) {
            if (i != 0) {
              disambiguation_block += ", ";
            }
            if (required_known[i]) {
              disambiguation_block += std::to_string(required_sizes[i]);
            } else {
              disambiguation_block += "?";
            }
          }
          disambiguation_block += "].";
          if (any_unknown > 0) {
            disambiguation_block +=
                " (`?` = sizer rerun could not resolve this stack's bound against the captured "
                "launch state).";
          }
          disambiguation_block += "\n";
        }
      }
    }
  }

  std::string body;
  if (cause == Cause::DLPackBypass) {
    body =
        "Cause (sync sizer rerun): a tensor backing a data-dependent loop bound was mutated outside "
        "Quadrants's tracking - typically a DLPack zero-copy mutation through a torch tensor sharing "
        "storage with a Quadrants ndarray, or a raw pointer write through a non-torch DLPack consumer. "
        "The cached adstack capacity was sized against the value before the mutation. Recovery: route "
        "the mutation through Quadrants APIs (`Ndarray.write` / `fill` / kernel writes) so the cache "
        "invalidates correctly, OR set a generous initial cap if a workload-change milestone genuinely "
        "grew capacity. Restart the iteration / training loop from a clean state.\n";
  } else if (cause == Cause::QuadrantsBug) {
    body =
        "Cause (sync sizer rerun): the freshly-computed required size does not exceed the allocated "
        "size for any stack - this is a Quadrants bug. The pre-pass resolved the alloca to a bound "
        "tighter than the actual runtime push count: either the enclosing loop shape is outside the "
        "current `SizeExpr` grammar, or the Bellman-Ford analyzer undercounted the forward-pass "
        "accumulation. Please file with the kernel IR (`QD_DUMP_IR=1`).\n";
  } else {
    body =
        "Two possible causes (synchronous sizer rerun was not conclusive - some `SizeExpr` trees "
        "depend on ndarray contents that are not host-resolvable without a per-launch context, or the "
        "task-id slot was empty so the registry pointer could not be confirmed live):\n"
        "  1. A tensor backing a data-dependent loop bound was mutated outside Quadrants's tracking "
        "(typically a DLPack zero-copy mutation through a torch tensor sharing storage with a "
        "Quadrants ndarray, or a raw pointer write through a non-torch DLPack consumer). The cached "
        "adstack capacity was sized against the value before the mutation. Recovery: route the "
        "mutation through Quadrants APIs (`Ndarray.write` / `fill` / kernel writes) so the cache "
        "invalidates correctly, OR set a generous initial cap if a workload-change milestone "
        "genuinely grew capacity. Restart the iteration / training loop from a clean state.\n"
        "  2. (Quadrants bug) the pre-pass resolved the alloca to a bound tighter than the actual "
        "runtime push count - the enclosing loop shape is outside the current `SizeExpr` grammar, or "
        "the Bellman-Ford analyzer undercounted the forward-pass accumulation. Please file with the "
        "kernel IR (`QD_DUMP_IR=1`).\n";
  }
  AdStackOverflowDiagnosis result;
  result.message = identity_block + disambiguation_block + body +
                   "Note: kernel state may be inconsistent post-overflow; do not retry the same "
                   "step without addressing the cause and restarting from a clean state.";
  // Flag the cache as confirmed-invalid only when the sync rerun positively identified DLPack-bypass (`required
  // > allocated` for at least one stack with every leaf resolved against the live snapshot). Unknown is a rare
  // fallback now that the snapshot-based evaluator handles ndarray-bound leaves; treating it as
  // confirmed-bypass would silently retry against a possibly-broken cache. Quadrants-bug is excluded for the
  // same reason - the next launch would re-run the same wrong sizer and produce the same wrong bound.
  result.confirmed_invalid_cache = (cause == Cause::DLPackBypass);
  return result;
}

void AdStackCache::capture_diagnose_snapshot(const LaunchContextBuilder &ctx) {
  diagnose_snapshot_.data_ptrs.clear();
  diagnose_snapshot_.dev_alloc_types.clear();
  diagnose_snapshot_.shapes.clear();
  // Pull just the data-pointer slot for each arg; the grad-pointer slot is irrelevant to size_expr leaves.
  for (const auto &kv : ctx.array_ptrs) {
    if (kv.first.ptr_type == TypeFactory::DATA_PTR_POS_IN_NDARRAY) {
      diagnose_snapshot_.data_ptrs[kv.first.arg_id] = kv.second;
    }
  }
  diagnose_snapshot_.dev_alloc_types = ctx.device_allocation_type;
  // Mirror the per-arg shape vectors `LaunchContextBuilder` populated alongside the args-buffer writes. Going
  // through this side map rather than `args_type->get_element_offset` avoids the spurious "Cannot treat as
  // TensorType" diagnostics emitted when an axis lookup overruns the actual rank, and keeps the diagnose path
  // independent of `args_type` lifetime.
  for (const auto &kv : ctx.ndarray_shapes) {
    std::vector<int32_t> shape32(kv.second.begin(), kv.second.end());
    diagnose_snapshot_.shapes[kv.first] = std::move(shape32);
  }
  diagnose_snapshot_.valid = true;
}

const AdStackCache::DiagnoseLaunchSnapshot *AdStackCache::get_diagnose_snapshot() const {
  return diagnose_snapshot_.valid ? &diagnose_snapshot_ : nullptr;
}

}  // namespace quadrants::lang
