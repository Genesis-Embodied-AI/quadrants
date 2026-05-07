#include "quadrants/program/adstack/cache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Read the input that `obs` describes against the live state and `ctx`. Caller compares the result to
// `obs.observed_value` to decide whether the cached `SizeExprCacheEntry` is still valid. Each `obs.kind`
// mirrors the corresponding leaf in `evaluate_field_load` / `evaluate_external_tensor_shape` /
// `evaluate_external_tensor_read`.
int64_t replay_one_observation(const AdStackCache::SizeExprReadObservation &obs,
                               Program *prog,
                               LaunchContextBuilder *ctx) {
  using Obs = AdStackCache::SizeExprReadObservation;
  switch (obs.kind) {
    case Obs::FieldLoadObs: {
      // Gen-counter fast skip: when no kernel has bumped this SNode's write generation since record time, the
      // underlying field value cannot have changed and we can return the recorded `observed_value` without dispatching
      // a reader kernel. The dispatch is the dominant per-launch cost on the hot path for steady-state reverse-mode
      // loops with stable bounds.
      if (prog != nullptr && prog->adstack_cache().snode_write_gen(obs.snode_id) == obs.observed_gen) {
        return obs.observed_value;
      }
      // Max-reducer body FieldLoadObs (bound-var-indexed leaves) records `indices = {}` since the body is evaluated at
      // every cross-product point and there is no single canonical index to re-read. The gen counter is the only valid
      // staleness signal in that mode; a gen mismatch unconditionally invalidates the cache.
      if (obs.indices.empty()) {
        return obs.observed_value + 1;
      }
      int64_t v = read_field_with_launch_cache(obs.snode_id, obs.indices, prog);
      if (v == std::numeric_limits<int64_t>::min()) {
        return obs.observed_value + 1;  // force a mismatch if SNode disappeared
      }
      return v;
    }
    case Obs::ExternalShapeObs: {
      if (ctx == nullptr) {
        return obs.observed_value + 1;
      }
      std::vector<int> arg_indices(obs.arg_id_path.begin(), obs.arg_id_path.end());
      arg_indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
      arg_indices.push_back(obs.arg_shape_axis);
      return static_cast<int64_t>(ctx->get_struct_arg_host<int32_t>(arg_indices));
    }
    case Obs::ExternalReadObs: {
      if (ctx == nullptr || obs.arg_id_path.empty()) {
        return obs.observed_value + 1;
      }
      int arg_id = obs.arg_id_path[0];
      ArgArrayPtrKey key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      auto it = ctx->array_ptrs.find(key);
      if (it == ctx->array_ptrs.end()) {
        return obs.observed_value + 1;
      }
      void *data_ptr = it->second;
      // Gen-counter fast skip: when the data pointer is the same `DeviceAllocation *` we observed at record
      // time AND its data generation has not been bumped since (no kernel write, no host-side `Ndarray.write`
      // / `fill`), the underlying scalar cannot have changed and we can return the recorded value without
      // dereferencing the device pointer (which on GPU would be a DtoH copy, on CPU a host load).
      if (prog != nullptr && data_ptr == obs.observed_devalloc &&
          prog->adstack_cache().ndarray_data_gen(data_ptr) == obs.observed_gen) {
        return obs.observed_value;
      }
      int64_t linear = 0;
      int64_t stride = 1;
      for (std::size_t i = obs.indices.size(); i > 0; --i) {
        linear += static_cast<int64_t>(obs.indices[i - 1]) * stride;
        if (i - 1 > 0) {
          std::vector<int> sh_idx(obs.arg_id_path.begin(), obs.arg_id_path.end());
          sh_idx.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
          sh_idx.push_back(static_cast<int>(i - 1));
          stride *= static_cast<int64_t>(ctx->get_struct_arg_host<int32_t>(sh_idx));
        }
      }
      switch (static_cast<PrimitiveTypeID>(obs.prim_dt)) {
        case PrimitiveTypeID::i32:
          return static_cast<int64_t>(static_cast<int32_t *>(data_ptr)[linear]);
        case PrimitiveTypeID::i64:
          return static_cast<int64_t *>(data_ptr)[linear];
        case PrimitiveTypeID::u32:
          return static_cast<int64_t>(static_cast<uint32_t *>(data_ptr)[linear]);
        case PrimitiveTypeID::u64:
          return static_cast<int64_t>(static_cast<uint64_t *>(data_ptr)[linear]);
        case PrimitiveTypeID::i16:
          return static_cast<int64_t>(static_cast<int16_t *>(data_ptr)[linear]);
        case PrimitiveTypeID::u16:
          return static_cast<int64_t>(static_cast<uint16_t *>(data_ptr)[linear]);
        case PrimitiveTypeID::i8:
          return static_cast<int64_t>(static_cast<int8_t *>(data_ptr)[linear]);
        case PrimitiveTypeID::u8:
          return static_cast<int64_t>(static_cast<uint8_t *>(data_ptr)[linear]);
        default:
          return obs.observed_value + 1;
      }
    }
  }
  return obs.observed_value + 1;
}

}  // namespace

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
    int64_t now = replay_one_observation(obs, prog, ctx);
    if (now != obs.observed_value) {
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
  size_expr_cache_[expr_key] = SizeExprCacheEntry{result, std::move(reads)};
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
    int64_t now = replay_one_observation(obs, prog_, ctx);
    if (now != obs.observed_value) {
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
      // `FieldLoadObs` from a bound-var-indexed body leaf: snapshot the snode write generation so a subsequent launch
      // that has not mutated the SNode replays the cached max via `replay_one_observation`'s gen-fast-skip arm. Same
      // sentinel rationale as `ExternalReadObs` below: the recognizer restricts the leaf dtype so an `INT64_MIN`
      // recorded value cannot equal a freshly-loaded one on cache miss.
      obs.observed_value = std::numeric_limits<int64_t>::min();
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
    // Pick an `observed_value` that no in-range ndarray scalar can equal (`INT64_MIN`). The replay code returns
    // `obs.observed_value` verbatim when `ndarray_data_gen` still matches the recorded snapshot, so an `INT64_MIN`
    // record is a self-equal cache hit. On gen mismatch the replay re-dereferences `data[0]` instead, which (under any
    // sub-i64 prim_dt the recognizer admits) widens to an i64 strictly greater than `INT64_MIN` and forces the cache to
    // invalidate. The dispatched max itself lives in `MaxReducerCacheEntry::result`; this observation only gates
    // whether the cache stays warm.
    obs.observed_value = std::numeric_limits<int64_t>::min();
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
  max_reducer_cache_[pack_max_reducer_key(registry_id, stack_id, mor_node_idx)] =
      MaxReducerCacheEntry{result, std::move(reads)};
  ++max_reducer_dispatch_count_;
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
    int64_t now = replay_one_observation(obs, prog, ctx);
    if (now != obs.observed_value) {
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
  spirv_bytecode_cache_[attribs_key] = SpirvBytecodeCacheEntry{std::move(bytecode), std::move(reads)};
}

void AdStackCache::record_per_task_ad_stack(const void *attribs_key,
                                            std::vector<uint32_t> metadata,
                                            uint32_t stride_float,
                                            uint32_t stride_int,
                                            std::vector<std::pair<int, uint64_t>> snode_gens,
                                            std::vector<std::tuple<int, void *, uint64_t>> arg_gens) {
  per_task_ad_stack_cache_[attribs_key] = PerTaskAdStackCacheEntry{std::move(metadata), stride_float, stride_int,
                                                                   std::move(snode_gens), std::move(arg_gens)};
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
  llvm_per_task_ad_stack_cache_[attribs_key] =
      LlvmPerTaskAdStackCacheEntry{std::move(offsets), std::move(max_sizes),  stride_combined,    stride_float,
                                   stride_int,         std::move(snode_gens), std::move(arg_gens)};
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

uint32_t AdStackCache::register_adstack_sizing_info(const void *identity_key,
                                                    const std::string &kernel_name,
                                                    int task_id_in_kernel,
                                                    std::vector<int> allocated_max_sizes,
                                                    std::vector<SerializedSizeExpr> size_exprs) {
  std::lock_guard<std::mutex> lk(adstack_sizing_info_registry_mutex_);
  // Idempotent re-registration: same `identity_key` yields the same id across re-compiles and updates the
  // entry's metadata + size_exprs in place. The key is just an opaque dedup token - the registry never
  // dereferences it; all data needed by the diagnose path is copied into the entry below.
  auto it = adstack_sizing_info_id_by_ptr_.find(identity_key);
  if (it != adstack_sizing_info_id_by_ptr_.end()) {
    auto &entry = adstack_sizing_info_registry_[it->second];
    entry.kernel_name = kernel_name;
    entry.task_id_in_kernel = task_id_in_kernel;
    entry.allocated_max_sizes = std::move(allocated_max_sizes);
    entry.size_exprs = std::move(size_exprs);
    return it->second;
  }
  uint32_t id = static_cast<uint32_t>(adstack_sizing_info_registry_.size());
  AdStackSizingInfoEntry entry;
  entry.identity_key = identity_key;
  entry.kernel_name = kernel_name;
  entry.task_id_in_kernel = task_id_in_kernel;
  entry.allocated_max_sizes = std::move(allocated_max_sizes);
  entry.size_exprs = std::move(size_exprs);
  adstack_sizing_info_registry_.push_back(std::move(entry));
  adstack_sizing_info_id_by_ptr_.emplace(identity_key, id);
  return id;
}

void AdStackCache::update_adstack_sizing_info_size_exprs(uint32_t id, std::vector<SerializedSizeExpr> size_exprs) {
  std::lock_guard<std::mutex> lk(adstack_sizing_info_registry_mutex_);
  if (id == 0 || id >= adstack_sizing_info_registry_.size()) {
    return;
  }
  adstack_sizing_info_registry_[id].size_exprs = std::move(size_exprs);
}

std::optional<AdStackCache::AdStackSizingInfoEntry> AdStackCache::lookup_adstack_sizing_info(uint32_t id) const {
  std::lock_guard<std::mutex> lk(adstack_sizing_info_registry_mutex_);
  if (id == 0 || id >= adstack_sizing_info_registry_.size()) {
    return std::nullopt;
  }
  return adstack_sizing_info_registry_[id];
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
