#include "quadrants/program/adstack/eval.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "quadrants/common/logging.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/ir/type_utils.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"
#include "quadrants/program/snode_rw_accessors_bank.h"

namespace quadrants::lang {

namespace {

using ReadSink = std::vector<AdStackCache::SizeExprReadObservation>;

// Per-launch cache of `FieldLoad` re-reads, keyed by `(snode_id, indices)`. Within one host-side eval root
// call the SNode field values are pinned (no other kernel runs concurrently), so deduping repeats across
// the size-expr trees evaluated in that window is correctness-safe.
struct LaunchScopedReadCache {
  struct Key {
    int snode_id;
    std::vector<int> indices;
    bool operator==(const Key &o) const noexcept {
      return snode_id == o.snode_id && indices == o.indices;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key &k) const noexcept {
      std::size_t h = std::hash<int>{}(k.snode_id);
      for (int v : k.indices) {
        h ^= std::hash<int>{}(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      }
      return h;
    }
  };
  std::unordered_map<Key, int64_t, KeyHash> map;
};
thread_local LaunchScopedReadCache *t_launch_read_cache = nullptr;

}  // namespace

int64_t evaluate_field_load(const SerializedSizeExprNode &node,
                            std::unordered_map<int32_t, int64_t> &bound_vars,
                            Program *prog,
                            ReadSink *reads) {
  QD_ASSERT_INFO(node.snode_id >= 0, "SerializedSizeExpr FieldLoad with no snode_id");
  SNode *snode = prog->get_snode_by_id(node.snode_id);
  QD_ASSERT_INFO(snode != nullptr,
                 "SerializedSizeExpr FieldLoad snode_id={} not found in the current program's snode trees",
                 node.snode_id);
  std::vector<int> indices;
  indices.reserve(node.indices.size());
  for (int32_t raw : node.indices) {
    if (raw >= 0) {
      indices.push_back(raw);
    } else {
      int32_t var_id = -(raw + 1);
      auto it = bound_vars.find(var_id);
      QD_ASSERT_INFO(it != bound_vars.end(),
                     "SerializedSizeExpr FieldLoad references unbound var_id={} (the enclosing MaxOverRange "
                     "node must have bound it before this read)",
                     var_id);
      indices.push_back(static_cast<int>(it->second));
    }
  }
  int64_t v = read_field_with_launch_cache(node.snode_id, indices, prog);
  if (reads != nullptr) {
    AdStackCache::SizeExprReadObservation obs;
    obs.kind = AdStackCache::SizeExprReadObservation::FieldLoadObs;
    obs.snode_id = node.snode_id;
    obs.indices = std::move(indices);
    obs.arg_shape_axis = 0;
    obs.prim_dt = 0;
    obs.observed_value = v;
    // Snapshot the SNode's write gen so the next replay can fast-skip when no kernel has written this SNode
    // since record time (the dominant case for a steady-state reverse-mode loop with stable bounds).
    obs.observed_gen = prog->adstack_cache().snode_write_gen(node.snode_id);
    reads->push_back(std::move(obs));
  }
  return v;
}

int64_t evaluate_external_tensor_read(const SerializedSizeExprNode &node,
                                      std::unordered_map<int32_t, int64_t> &bound_vars,
                                      Program *prog,
                                      LaunchContextBuilder *ctx,
                                      ReadSink *reads) {
  QD_ASSERT_INFO(ctx != nullptr,
                 "SerializedSizeExpr ExternalTensorRead evaluated with no LaunchContextBuilder; the launcher "
                 "must pass the current launch's context in");
  QD_ASSERT_INFO(!node.arg_id_path.empty(), "SerializedSizeExpr ExternalTensorRead has empty arg_id_path");
  int arg_id = node.arg_id_path[0];
  ArgArrayPtrKey key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
  auto it = ctx->array_ptrs.find(key);
  QD_ASSERT_INFO(it != ctx->array_ptrs.end(),
                 "SerializedSizeExpr ExternalTensorRead: arg {} has no data pointer in launch context", arg_id);
  void *data_ptr = it->second;
  // Resolve each index (possibly via a bound variable) and compose them into the C-order linear offset
  // `sum_i(idx_i * prod_{j>i}(shape_j))`. Multi-dim shapes are read from the launch context through the same
  // `SHAPE_POS_IN_NDARRAY` path `ExternalTensorShape` uses, so an ndarray indexed by two or more loop variables lowers
  // to the correct element rather than the stride-1 sum `arr_flat[i + j + ...]`. Mirrors the per-axis stride that
  // `encode_subtree` precomputes on the SPIR-V path; on CPU the host evaluator is called directly from
  // `publish_adstack_metadata`, so the stride math has to live here too.
  std::vector<int64_t> resolved(node.indices.size());
  for (std::size_t i = 0; i < node.indices.size(); ++i) {
    int32_t raw = node.indices[i];
    if (raw >= 0) {
      resolved[i] = raw;
    } else {
      int32_t var_id = -(raw + 1);
      auto bv = bound_vars.find(var_id);
      QD_ASSERT_INFO(bv != bound_vars.end(), "SerializedSizeExpr ExternalTensorRead references unbound var_id={}",
                     var_id);
      resolved[i] = bv->second;
    }
  }
  int64_t linear = 0;
  int64_t stride = 1;
  for (std::size_t i = node.indices.size(); i > 0; --i) {
    linear += resolved[i - 1] * stride;
    if (i - 1 > 0) {
      std::vector<int> sh_idx(node.arg_id_path.begin(), node.arg_id_path.end());
      sh_idx.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
      sh_idx.push_back(static_cast<int>(i - 1));
      // Ndarray shapes are `int32` in the args struct (same convention `evaluate_external_tensor_shape` relies on);
      // reading as `int64` would sign-extend the adjacent slot into the shape and produce garbage strides.
      stride *= static_cast<int64_t>(ctx->get_struct_arg_host<int32_t>(sh_idx));
    }
  }
  auto prim_dt = static_cast<PrimitiveTypeID>(node.const_value);
  int64_t v;
  switch (prim_dt) {
    case PrimitiveTypeID::i32:
      v = static_cast<int64_t>(static_cast<int32_t *>(data_ptr)[linear]);
      break;
    case PrimitiveTypeID::i64:
      v = static_cast<int64_t *>(data_ptr)[linear];
      break;
    case PrimitiveTypeID::u32:
      v = static_cast<int64_t>(static_cast<uint32_t *>(data_ptr)[linear]);
      break;
    case PrimitiveTypeID::u64:
      v = static_cast<int64_t>(static_cast<uint64_t *>(data_ptr)[linear]);
      break;
    case PrimitiveTypeID::i16:
      v = static_cast<int64_t>(static_cast<int16_t *>(data_ptr)[linear]);
      break;
    case PrimitiveTypeID::u16:
      v = static_cast<int64_t>(static_cast<uint16_t *>(data_ptr)[linear]);
      break;
    case PrimitiveTypeID::i8:
      v = static_cast<int64_t>(static_cast<int8_t *>(data_ptr)[linear]);
      break;
    case PrimitiveTypeID::u8:
      v = static_cast<int64_t>(static_cast<uint8_t *>(data_ptr)[linear]);
      break;
    default:
      QD_ERROR("SerializedSizeExpr ExternalTensorRead: unsupported element type {}", node.const_value);
      v = 0;
  }
  if (reads != nullptr) {
    AdStackCache::SizeExprReadObservation obs;
    obs.kind = AdStackCache::SizeExprReadObservation::ExternalReadObs;
    obs.snode_id = 0;
    obs.indices.reserve(resolved.size());
    for (auto r : resolved)
      obs.indices.push_back(static_cast<int>(r));
    obs.arg_id_path = node.arg_id_path;
    obs.arg_shape_axis = 0;
    obs.prim_dt = static_cast<int>(prim_dt);
    obs.observed_value = v;
    obs.observed_devalloc = data_ptr;
    if (prog != nullptr) {
      // Snapshot the ndarray's data gen so the next replay can fast-skip when no kernel / Ndarray API write
      // has touched the underlying buffer since record time. Mirrors the FieldLoad fast-skip; covers the same
      // steady-state hot path for ndarray-bounded reverse-mode loops.
      obs.observed_gen = prog->adstack_cache().ndarray_data_gen(data_ptr);
    }
    reads->push_back(std::move(obs));
  }
  return v;
}

int64_t evaluate_external_tensor_shape(const SerializedSizeExprNode &node, LaunchContextBuilder *ctx, ReadSink *reads) {
  QD_ASSERT_INFO(ctx != nullptr,
                 "SerializedSizeExpr ExternalTensorShape evaluated with no LaunchContextBuilder; the launcher "
                 "must pass the current launch's context into the evaluator to resolve ndarray shapes");
  std::vector<int> arg_indices(node.arg_id_path.begin(), node.arg_id_path.end());
  arg_indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
  arg_indices.push_back(node.arg_shape_axis);
  // Ndarray shape slots are `int32` in the args struct (same convention `evaluate_external_tensor_read` relies
  // on for its stride multiplies). Using `int64` here reads 8 bytes past the slot and sign-extends the next
  // field into the shape, so a user-visible downstream effect is that any `SizeExpr` node that feeds a
  // shape-derived value into a trip count (e.g. `MaxOverRange(0, ExtShape, ...)`) evaluates its range as
  // garbage - often zero when the adjacent field is zero-initialised - and the containing tree collapses to
  // zero. The adstack max_size is clamped to 1 on a zero tree result, which under-bounds real push counts and
  // trips an overflow assertion at the next `qd.sync()`.
  int64_t v = static_cast<int64_t>(ctx->get_struct_arg_host<int32_t>(arg_indices));
  if (reads != nullptr) {
    AdStackCache::SizeExprReadObservation obs;
    obs.kind = AdStackCache::SizeExprReadObservation::ExternalShapeObs;
    obs.snode_id = 0;
    obs.arg_id_path = node.arg_id_path;
    obs.arg_shape_axis = node.arg_shape_axis;
    obs.prim_dt = 0;
    obs.observed_value = v;
    reads->push_back(std::move(obs));
  }
  return v;
}

int64_t evaluate_node(const SerializedSizeExpr &expr,
                      int32_t node_idx,
                      std::unordered_map<int32_t, int64_t> &bound_vars,
                      Program *prog,
                      LaunchContextBuilder *ctx,
                      ReadSink *reads) {
  QD_ASSERT_INFO(node_idx >= 0 && static_cast<std::size_t>(node_idx) < expr.nodes.size(),
                 "SerializedSizeExpr node_idx {} out of bounds (size={})", node_idx, expr.nodes.size());
  const auto &node = expr.nodes[node_idx];
  switch (static_cast<SizeExpr::Kind>(node.kind)) {
    case SizeExpr::Kind::Const:
      return node.const_value;
    case SizeExpr::Kind::FieldLoad:
      return evaluate_field_load(node, bound_vars, prog, reads);
    case SizeExpr::Kind::Add:
      return evaluate_node(expr, node.operand_a, bound_vars, prog, ctx, reads) +
             evaluate_node(expr, node.operand_b, bound_vars, prog, ctx, reads);
    case SizeExpr::Kind::Sub:
      return std::max<int64_t>(evaluate_node(expr, node.operand_a, bound_vars, prog, ctx, reads) -
                                   evaluate_node(expr, node.operand_b, bound_vars, prog, ctx, reads),
                               0);
    case SizeExpr::Kind::Mul:
      return evaluate_node(expr, node.operand_a, bound_vars, prog, ctx, reads) *
             evaluate_node(expr, node.operand_b, bound_vars, prog, ctx, reads);
    case SizeExpr::Kind::Max:
      return std::max(evaluate_node(expr, node.operand_a, bound_vars, prog, ctx, reads),
                      evaluate_node(expr, node.operand_b, bound_vars, prog, ctx, reads));
    case SizeExpr::Kind::MaxOverRange: {
      int64_t begin = evaluate_node(expr, node.operand_a, bound_vars, prog, ctx, reads);
      int64_t end = evaluate_node(expr, node.operand_b, bound_vars, prog, ctx, reads);
      // Guard against pathological trip counts. The evaluator walks `[begin, end)` linearly and re-evaluates the
      // body at every i; a range of several million would stall the launch hot path for seconds. Real reverse-mode
      // trip counts sit well below this cap (a few hundred to a few thousand in practice); anything above is
      // almost certainly a pre-pass grammar bug the user should file, and a clear QD_ERROR beats a silent hang.
      constexpr int64_t kMaxOverRangeIterations = int64_t{1} << 24;
      QD_ERROR_IF(end > begin && end - begin > kMaxOverRangeIterations,
                  "SerializedSizeExpr MaxOverRange iteration count {} exceeds the {} guard; refusing to enumerate. "
                  "Shrink the enclosing reverse-mode loop or restructure the `SizeExpr` source kernel.",
                  end - begin, kMaxOverRangeIterations);
      int64_t result = 0;
      // Bind `var_id` in `bound_vars` for the duration of the loop and restore the outer-scope value (or erase, if
      // there was none) before returning, so nested `MaxOverRange` bindings of the same `var_id` stay correct without
      // cloning the entire map per iteration.
      auto prev_it = bound_vars.find(node.var_id);
      bool had_prev = prev_it != bound_vars.end();
      int64_t prev_val = had_prev ? prev_it->second : 0;
      for (int64_t i = begin; i < end; ++i) {
        bound_vars[node.var_id] = i;
        int64_t v = evaluate_node(expr, node.body_node_idx, bound_vars, prog, ctx, reads);
        if (v > result) {
          result = v;
        }
      }
      if (had_prev) {
        bound_vars[node.var_id] = prev_val;
      } else {
        bound_vars.erase(node.var_id);
      }
      return result;
    }
    case SizeExpr::Kind::BoundVariable: {
      auto it = bound_vars.find(node.var_id);
      QD_ASSERT_INFO(it != bound_vars.end(),
                     "SerializedSizeExpr BoundVariable var_id={} evaluated outside its MaxOverRange scope",
                     node.var_id);
      return it->second;
    }
    case SizeExpr::Kind::ExternalTensorShape:
      return evaluate_external_tensor_shape(node, ctx, reads);
    case SizeExpr::Kind::ExternalTensorRead:
      return evaluate_external_tensor_read(node, bound_vars, prog, ctx, reads);
  }
  QD_ERROR("unreachable SerializedSizeExpr kind {}", node.kind);
  return 0;
}

int64_t read_field_with_launch_cache(int snode_id, const std::vector<int> &indices, Program *prog) {
  SNode *snode = prog->get_snode_by_id(snode_id);
  if (snode == nullptr) {
    return std::numeric_limits<int64_t>::min();
  }
  if (t_launch_read_cache != nullptr) {
    LaunchScopedReadCache::Key key{snode_id, indices};
    auto it = t_launch_read_cache->map.find(key);
    if (it != t_launch_read_cache->map.end()) {
      return it->second;
    }
    int64_t v = prog->get_snode_rw_accessors_bank().get(snode).read_int(indices);
    t_launch_read_cache->map.emplace(std::move(key), v);
    return v;
  }
  return prog->get_snode_rw_accessors_bank().get(snode).read_int(indices);
}

// Per-thread backing for `SizeExprLaunchScope`. The outer scope on each thread points `t_launch_read_cache` here
// after clearing the map; nested scopes are no-ops.
thread_local LaunchScopedReadCache t_launch_read_cache_storage{};

SizeExprLaunchScope::SizeExprLaunchScope() : owns_(t_launch_read_cache == nullptr) {
  if (owns_) {
    t_launch_read_cache_storage.map.clear();
    t_launch_read_cache = &t_launch_read_cache_storage;
  }
}
SizeExprLaunchScope::~SizeExprLaunchScope() {
  if (owns_) {
    t_launch_read_cache = nullptr;
  }
}

int64_t evaluate_adstack_size_expr_no_cache(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx) {
  if (expr.nodes.empty()) {
    return -1;
  }
  SizeExprLaunchScope local_scope;
  std::unordered_map<int32_t, int64_t> empty_bound_vars;
  std::vector<AdStackCache::SizeExprReadObservation> reads;
  return evaluate_node(expr, static_cast<int32_t>(expr.nodes.size() - 1), empty_bound_vars, prog, ctx, &reads);
}

int64_t evaluate_adstack_size_expr(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx) {
  if (expr.nodes.empty()) {
    return -1;
  }
  // Open a `SizeExprLaunchScope` if no enclosing one is active, so repeated reads within this eval share
  // the launch read cache. Callers that issue several `evaluate_adstack_size_expr` calls back-to-back
  // should open their own scope to span all of them.
  SizeExprLaunchScope local_scope;

  // Cache fast path: replay the recorded reads against the live state and reuse the cached result if
  // every input still matches. The full walk runs only on cache miss.
  if (prog != nullptr) {
    int64_t cached;
    if (prog->adstack_cache().try_size_expr_cache_hit(prog, &expr, ctx, cached)) {
      return cached;
    }
  }
  std::unordered_map<int32_t, int64_t> empty_bound_vars;
  std::vector<AdStackCache::SizeExprReadObservation> reads;
  int64_t result =
      evaluate_node(expr, static_cast<int32_t>(expr.nodes.size() - 1), empty_bound_vars, prog, ctx, &reads);
  if (prog != nullptr) {
    prog->adstack_cache().record_size_expr_eval(&expr, result, std::move(reads));
  }
  return result;
}

int64_t evaluate_adstack_size_expr_at_node(const SerializedSizeExpr &expr,
                                           int32_t node_idx,
                                           Program *prog,
                                           LaunchContextBuilder *ctx) {
  if (node_idx < 0 || static_cast<std::size_t>(node_idx) >= expr.nodes.size()) {
    return -1;
  }
  // The recognizer grammar guarantees the subtree at `node_idx` is closed (no outer-scope `BoundVariable` references),
  // so an empty bound-vars map is sufficient. Read observations are not recorded - the caller (max-reducer launcher)
  // does its own observation tracking via `AdStackCache::record_max_reducer_eval` against the spec key, not the
  // per-`SerializedSizeExpr` key the cache uses for `evaluate_adstack_size_expr`.
  SizeExprLaunchScope local_scope;
  std::unordered_map<int32_t, int64_t> empty_bound_vars;
  return evaluate_node(expr, node_idx, empty_bound_vars, prog, ctx, /*reads=*/nullptr);
}

}  // namespace quadrants::lang
