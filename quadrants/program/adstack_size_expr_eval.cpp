#include "quadrants/program/adstack_size_expr_eval.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/codegen/spirv/adstack_sizer_shader.h"
#include "quadrants/common/logging.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/ir/type_utils.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"
#include "quadrants/program/snode_rw_accessors_bank.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {

namespace {

using ReadSink = std::vector<AdStackCache::SizeExprReadObservation>;

// Forward-declared, defined further down. Reads SNode `snode_id` at `indices` via the per-launch read
// cache (when active) so multiple size-expr trees evaluated within the same outer launch share a single
// reader-kernel dispatch per `(snode_id, indices)` pair.
int64_t read_field_with_launch_cache(int snode_id, const std::vector<int> &indices, Program *prog);

int64_t evaluate_node(const SerializedSizeExpr &expr,
                      int32_t node_idx,
                      std::unordered_map<int32_t, int64_t> &bound_vars,
                      Program *prog,
                      LaunchContextBuilder *ctx,
                      ReadSink *reads);

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

// --------------------------------------------------------------------------------------------------------------
// Device-bytecode encoder helpers
// --------------------------------------------------------------------------------------------------------------

// `contains_device_leaf[i]` is true when subtree rooted at node `i` has at least one leaf that MUST stay on the
// device during encoding (the host fold cannot substitute it with a `Const`). On the LLVM path this is any
// `ExternalTensorRead` leaf - `FieldLoad` can be host-folded via `SNodeRwAccessorsBank::read_int`, which is safe
// on CPU / CUDA / AMDGPU. On the SPIR-V path the caller flips `fieldload_stays_on_device` to true because on
// MoltenVK a nested `read_int` submit crashes inside the descriptor-set bind path; keeping `FieldLoad` on the
// device side (via PSB loads in the sizer shader) avoids that entirely. Computed bottom-up; `SerializedSizeExpr`
// is already in post-order so every operand / body index is < i.
std::vector<bool> compute_contains_device_leaf(const SerializedSizeExpr &expr, bool fieldload_stays_on_device) {
  std::vector<bool> result(expr.nodes.size(), false);
  for (std::size_t i = 0; i < expr.nodes.size(); ++i) {
    const auto &node = expr.nodes[i];
    auto kind = static_cast<SizeExpr::Kind>(node.kind);
    bool hit = (kind == SizeExpr::Kind::ExternalTensorRead) ||
               (fieldload_stays_on_device && kind == SizeExpr::Kind::FieldLoad);
    if (!hit && node.operand_a >= 0)
      hit = result[node.operand_a];
    if (!hit && node.operand_b >= 0)
      hit = result[node.operand_b];
    if (!hit && node.body_node_idx >= 0)
      hit = result[node.body_node_idx];
    result[i] = hit;
  }
  return result;
}

// `free_vars[i]` is the set of `BoundVariable::var_id`s referenced inside subtree(i) but NOT bound by any
// `MaxOverRange` inside that same subtree. An empty set means the subtree is closed and can be evaluated on the
// host without an outer-iteration context. `FieldLoad` / `ExternalTensorRead` index slots use the same
// `-(var_id + 1)` encoding as `BoundVariable` and are accounted for here.
std::vector<std::unordered_set<int32_t>> compute_free_vars(const SerializedSizeExpr &expr) {
  std::vector<std::unordered_set<int32_t>> result(expr.nodes.size());
  for (std::size_t i = 0; i < expr.nodes.size(); ++i) {
    const auto &node = expr.nodes[i];
    auto &fv = result[i];
    auto collect_idx_vars = [&](const std::vector<int32_t> &indices) {
      for (int32_t raw : indices) {
        if (raw < 0)
          fv.insert(-(raw + 1));
      }
    };
    switch (static_cast<SizeExpr::Kind>(node.kind)) {
      case SizeExpr::Kind::Const:
      case SizeExpr::Kind::ExternalTensorShape:
        break;
      case SizeExpr::Kind::BoundVariable:
        fv.insert(node.var_id);
        break;
      case SizeExpr::Kind::FieldLoad:
      case SizeExpr::Kind::ExternalTensorRead:
        collect_idx_vars(node.indices);
        break;
      case SizeExpr::Kind::Add:
      case SizeExpr::Kind::Sub:
      case SizeExpr::Kind::Mul:
      case SizeExpr::Kind::Max:
        fv = result[node.operand_a];
        for (auto v : result[node.operand_b])
          fv.insert(v);
        break;
      case SizeExpr::Kind::MaxOverRange: {
        fv = result[node.operand_a];
        for (auto v : result[node.operand_b])
          fv.insert(v);
        // MaxOverRange binds `var_id` for its body only: body's free vars minus this binding add into the
        // outer set.
        for (auto v : result[node.body_node_idx]) {
          if (v != node.var_id)
            fv.insert(v);
        }
        break;
      }
    }
  }
  return result;
}

// Walks `expr` and builds a dense `original_var_id -> [0, N)` map across every `var_id` the tree references
// (`MaxOverRange` binds, `BoundVariable` leaves, and bound-var entries inside each ETR / FieldLoad index list).
// The walker preserves encounter order so nested `MaxOverRange` binds keep monotonically increasing dense ids,
// which also matches the natural `values[]` indexing the device interpreter does at each bind. Hard-errors if
// the tree references more distinct bound vars than the device interpreter's per-stack scope capacity.
std::unordered_map<int32_t, int32_t> build_dense_var_id_remap(const SerializedSizeExpr &expr) {
  std::unordered_map<int32_t, int32_t> remap;
  auto add = [&](int32_t v) {
    if (v < 0)
      return;
    if (remap.find(v) == remap.end()) {
      int32_t dense = static_cast<int32_t>(remap.size());
      remap.emplace(v, dense);
    }
  };
  for (const auto &node : expr.nodes) {
    const auto kind = static_cast<SizeExpr::Kind>(node.kind);
    if (kind == SizeExpr::Kind::MaxOverRange || kind == SizeExpr::Kind::BoundVariable)
      add(node.var_id);
    for (int32_t raw : node.indices) {
      if (raw < 0)
        add(-(raw + 1));
    }
  }
  QD_ERROR_IF(static_cast<int32_t>(remap.size()) > kAdStackSizeExprDeviceMaxBoundVars,
              "Adstack SizeExpr tree references {} distinct bound variable ids, which exceeds the device "
              "interpreter's per-stack scope capacity ({}). This almost always indicates a deeply nested "
              "reverse-mode loop shape that the pre-pass should have folded earlier; shrink the enclosing "
              "loops or file a bug so the grammar / walker can be tightened.",
              remap.size(), kAdStackSizeExprDeviceMaxBoundVars);
  return remap;
}

// Computes the maximum `MaxOverRange` nesting depth reachable from any root in `expr`, i.e. the deepest
// chain of `MaxOverRange` nodes whose `body_node_idx` recursively references another `MaxOverRange`. The
// sizer shader's per-invocation pending-frame stack is sized to `kAdStackSizerMaxPendingFrames`; the encoder
// hard-errors when a tree's nesting exceeds this so the shader's fixed-size access-chain stays in bounds
// without a runtime guard. Each node's depth is memoised to keep the walk linear in `expr.nodes.size()`.
int32_t compute_max_mor_nesting(const SerializedSizeExpr &expr) {
  std::vector<int32_t> depth(expr.nodes.size(), -1);
  std::function<int32_t(int32_t)> visit = [&](int32_t i) -> int32_t {
    if (i < 0 || static_cast<std::size_t>(i) >= expr.nodes.size())
      return 0;
    if (depth[i] >= 0)
      return depth[i];
    const auto &n = expr.nodes[i];
    int32_t child_max = 0;
    for (int32_t c : {n.operand_a, n.operand_b, n.body_node_idx}) {
      if (c >= 0)
        child_max = std::max(child_max, visit(c));
    }
    int32_t self = static_cast<SizeExpr::Kind>(n.kind) == SizeExpr::Kind::MaxOverRange ? 1 : 0;
    depth[i] = self + child_max;
    return depth[i];
  };
  int32_t max_depth = 0;
  for (std::size_t i = 0; i < expr.nodes.size(); ++i) {
    max_depth = std::max(max_depth, visit(static_cast<int32_t>(i)));
  }
  return max_depth;
}

// Returns the dense id for `original_var_id`, or fires a hard error if the remap lost track of it (which would
// indicate a walker divergence between `build_dense_var_id_remap` and `encode_subtree`).
int32_t remap_var_id(const std::unordered_map<int32_t, int32_t> &remap, int32_t original) {
  auto it = remap.find(original);
  QD_ASSERT_INFO(it != remap.end(),
                 "Adstack SizeExpr encoder saw var_id={} not present in the dense remap; this "
                 "is a walker bug between `build_dense_var_id_remap` and `encode_subtree`.",
                 original);
  return it->second;
}

// Initialises a fresh device node with every unused slot sentinelled so the interpreter can tell them apart from
// legitimate zero-valued slots (e.g. `operand_a == 0` is a valid node index; only `-1` signals "unused").
AdStackSizeExprDeviceNode make_empty_device_node(int32_t kind) {
  AdStackSizeExprDeviceNode dn{};
  dn.kind = kind;
  dn.operand_a = -1;
  dn.operand_b = -1;
  dn.body_node_idx = -1;
  dn.var_id = -1;
  dn.prim_dt = -1;
  dn.arg_buffer_offset = -1;
  dn.indices_offset = 0;
  dn.indices_count = 0;
  dn._pad0 = 0;
  dn.const_value = 0;
  return dn;
}

// Data needed to encode a `FieldLoad` as a `kFieldLoad` device node. Populated by the SPIR-V encoder entry
// point via `GfxRuntime` / `Device` queries; the LLVM encoder passes a default-constructed (empty) emitter,
// which routes every `FieldLoad` through the host-fold path instead (safe on CPU / CUDA / AMDGPU where a
// nested accessor kernel launch is fine).
struct FieldLoadDeviceEmitter {
  // Returns true on success, populating `out_base_psb` with `root_buffer_psb + place_byte_offset_in_root` and
  // `out_elem_strides` with one positive int32 *element* stride per active axis of `snode` (stride in units of
  // the leaf's primitive type, not bytes - the sizer shader reuses `psb_load_scalar` which already multiplies
  // by `sizeof(prim_dt)`). Returns false when the snode layout is not amenable to direct PSB indexing
  // (bitmasked / pointer / hash chain, bit-level place, not-all-dense path), in which case the encoder raises
  // a `QD_ERROR`. The dense-only restriction is deliberate - observed kernels exercise only dense chains in the
  // adstack pre-pass's `SizeExpr::FieldLoad` leaves, and extending this to bitmasked / pointer would require
  // threading the full access codegen through the sizer shader, which is out of scope.
  std::function<bool(SNode *snode, uint64_t *out_base_psb, std::vector<int32_t> *out_elem_strides)> fetch;

  bool empty() const {
    return fetch == nullptr;
  }
};

// Recursive top-down encoder. Each call returns the index of the emitted root in `out_nodes`. Subtrees whose
// leaves are all host-resolvable (no `ExternalTensorRead`, and - on the LLVM path - no `FieldLoad` either) and
// whose bound variables are all locally bound within the subtree get folded to a single `kConst` device node
// by running `evaluate_node` over them. On the SPIR-V path, `FieldLoad` also survives as a `kFieldLoad` device
// node alongside `kExternalTensorRead`, so the shader can resolve the snode read in place via PSB.
int32_t encode_subtree(const SerializedSizeExpr &src,
                       int32_t src_idx,
                       const std::vector<bool> &contains_device_leaf,
                       const std::vector<std::unordered_set<int32_t>> &free_vars,
                       const std::unordered_map<int32_t, int32_t> &var_id_remap,
                       Program *prog,
                       LaunchContextBuilder *ctx,
                       const FieldLoadDeviceEmitter &fl_emitter,
                       std::vector<AdStackSizeExprDeviceNode> &out_nodes,
                       std::vector<int32_t> &out_indices,
                       ReadSink *reads) {
  QD_ASSERT_INFO(src_idx >= 0 && static_cast<std::size_t>(src_idx) < src.nodes.size(),
                 "encode_subtree: src_idx {} out of bounds (size={})", src_idx, src.nodes.size());
  const bool subtree_needs_device = contains_device_leaf[src_idx];
  const bool subtree_closed = free_vars[src_idx].empty();

  if (!subtree_needs_device && subtree_closed) {
    // Whole subtree resolves without any device-resident read and without an outer-iteration context, so fold it
    // to a single `Const` by running the host evaluator over it. This is the only path that can substitute
    // `FieldLoad` / `ExternalTensorShape` leaves - the device interpreter does not know how to walk SNodes or
    // index into `args_type`.
    std::unordered_map<int32_t, int64_t> empty_bound;
    int64_t val = evaluate_node(src, src_idx, empty_bound, prog, ctx, reads);
    AdStackSizeExprDeviceNode dn = make_empty_device_node(static_cast<int32_t>(AdStackSizeExprDeviceKind::kConst));
    dn.const_value = val;
    out_nodes.push_back(dn);
    return static_cast<int32_t>(out_nodes.size() - 1);
  }

  const auto &node = src.nodes[src_idx];
  const auto kind = static_cast<SizeExpr::Kind>(node.kind);
  switch (kind) {
    case SizeExpr::Kind::Const: {
      AdStackSizeExprDeviceNode dn = make_empty_device_node(static_cast<int32_t>(AdStackSizeExprDeviceKind::kConst));
      dn.const_value = node.const_value;
      out_nodes.push_back(dn);
      return static_cast<int32_t>(out_nodes.size() - 1);
    }
    case SizeExpr::Kind::BoundVariable: {
      AdStackSizeExprDeviceNode dn =
          make_empty_device_node(static_cast<int32_t>(AdStackSizeExprDeviceKind::kBoundVariable));
      dn.var_id = remap_var_id(var_id_remap, node.var_id);
      out_nodes.push_back(dn);
      return static_cast<int32_t>(out_nodes.size() - 1);
    }
    case SizeExpr::Kind::Add:
    case SizeExpr::Kind::Sub:
    case SizeExpr::Kind::Mul:
    case SizeExpr::Kind::Max: {
      int32_t a = encode_subtree(src, node.operand_a, contains_device_leaf, free_vars, var_id_remap, prog, ctx,
                                 fl_emitter, out_nodes, out_indices, reads);
      int32_t b = encode_subtree(src, node.operand_b, contains_device_leaf, free_vars, var_id_remap, prog, ctx,
                                 fl_emitter, out_nodes, out_indices, reads);
      AdStackSizeExprDeviceKind dk = AdStackSizeExprDeviceKind::kAdd;
      if (kind == SizeExpr::Kind::Sub)
        dk = AdStackSizeExprDeviceKind::kSub;
      else if (kind == SizeExpr::Kind::Mul)
        dk = AdStackSizeExprDeviceKind::kMul;
      else if (kind == SizeExpr::Kind::Max)
        dk = AdStackSizeExprDeviceKind::kMax;
      AdStackSizeExprDeviceNode dn = make_empty_device_node(static_cast<int32_t>(dk));
      dn.operand_a = a;
      dn.operand_b = b;
      out_nodes.push_back(dn);
      return static_cast<int32_t>(out_nodes.size() - 1);
    }
    case SizeExpr::Kind::MaxOverRange: {
      int32_t a = encode_subtree(src, node.operand_a, contains_device_leaf, free_vars, var_id_remap, prog, ctx,
                                 fl_emitter, out_nodes, out_indices, reads);
      int32_t b = encode_subtree(src, node.operand_b, contains_device_leaf, free_vars, var_id_remap, prog, ctx,
                                 fl_emitter, out_nodes, out_indices, reads);
      int32_t body = encode_subtree(src, node.body_node_idx, contains_device_leaf, free_vars, var_id_remap, prog, ctx,
                                    fl_emitter, out_nodes, out_indices, reads);
      AdStackSizeExprDeviceNode dn =
          make_empty_device_node(static_cast<int32_t>(AdStackSizeExprDeviceKind::kMaxOverRange));
      dn.operand_a = a;
      dn.operand_b = b;
      dn.body_node_idx = body;
      dn.var_id = remap_var_id(var_id_remap, node.var_id);
      out_nodes.push_back(dn);
      return static_cast<int32_t>(out_nodes.size() - 1);
    }
    case SizeExpr::Kind::ExternalTensorRead: {
      QD_ASSERT_INFO(ctx != nullptr && ctx->args_type != nullptr,
                     "encode_subtree: ExternalTensorRead at node {} requires a LaunchContextBuilder with a valid "
                     "args_type to precompute the data_ptr offset",
                     src_idx);
      QD_ASSERT_INFO(!node.arg_id_path.empty(), "ExternalTensorRead at node {} has empty arg_id_path", src_idx);
      std::vector<int> arg_indices(node.arg_id_path.begin(), node.arg_id_path.end());
      arg_indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      const size_t data_ptr_offset = ctx->args_type->get_element_offset(arg_indices);
      AdStackSizeExprDeviceNode dn =
          make_empty_device_node(static_cast<int32_t>(AdStackSizeExprDeviceKind::kExternalTensorRead));
      // Cast to i32 is safe: `arg_buffer` sizes in practice are kilobytes, well under INT32_MAX.
      dn.arg_buffer_offset = static_cast<int32_t>(data_ptr_offset);
      dn.prim_dt = static_cast<int32_t>(node.const_value);  // the pre-pass stashes `PrimitiveTypeID` in const_value
      dn.indices_offset = static_cast<int32_t>(out_indices.size());
      dn.indices_count = static_cast<int32_t>(node.indices.size());
      // Pre-compute per-axis element strides in C order (`stride[k] = prod_{m > k} shape[m]`). Shapes live in
      // the kernel args struct as `int32` slots at the `SHAPE_POS_IN_NDARRAY` path, same source the host
      // evaluator reads; using the live launch context keeps strides consistent with whichever ndarray the
      // user handed to the kernel on this launch. Emit as `[idx_a_raw, elem_stride_a]` pairs per axis,
      // matching the `kFieldLoad` layout so the device interpreter and SPIR-V sizer shader can share one
      // pair-walking offset-computation loop instead of carrying a separate stride-1 path.
      std::vector<int32_t> elem_strides(node.indices.size(), 1);
      if (node.indices.size() > 1) {
        for (std::size_t k = node.indices.size(); k-- > 0;) {
          if (k + 1 < node.indices.size()) {
            std::vector<int> sh_idx(node.arg_id_path.begin(), node.arg_id_path.end());
            sh_idx.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
            sh_idx.push_back(static_cast<int>(k + 1));
            int32_t sh = ctx->get_struct_arg_host<int32_t>(sh_idx);
            elem_strides[k] = elem_strides[k + 1] * sh;
          }
        }
      }
      for (std::size_t k = 0; k < node.indices.size(); ++k) {
        int32_t raw = node.indices[k];
        if (raw < 0) {
          // Remap bound-variable refs so the device interpreter's `scope->values[var]` read lands in the
          // `[0, kAdStackSizeExprDeviceMaxBoundVars)` range regardless of how large the source tree's
          // `var_id_counter` grew across its push-site walks.
          int32_t dense = remap_var_id(var_id_remap, -(raw + 1));
          raw = -(dense + 1);
        }
        out_indices.push_back(raw);
        out_indices.push_back(elem_strides[k]);
      }
      out_nodes.push_back(dn);
      return static_cast<int32_t>(out_nodes.size() - 1);
    }
    case SizeExpr::Kind::FieldLoad: {
      // If we reach here the subtree is not host-substitutable (has free bound vars or sits alongside an
      // `ExternalTensorRead` in the same closed context, or - on the SPIR-V path - `FieldLoad` is deliberately
      // kept on the device via `fl_emitter`). Without an emitter, the LLVM path would have folded it earlier;
      // reaching here without one means the shape is outside what the grammar supports, which is a user-facing
      // bug, not a runtime fallback.
      QD_ASSERT_INFO(
          !fl_emitter.empty(),
          "Adstack SizeExpr FieldLoad at node {} survived the host fold without a FieldLoadDeviceEmitter. The "
          "LLVM encoder should route closed FieldLoads through `evaluate_node` and reject non-closed ones before "
          "the structural pre-pass emits them; if this fires, a SerializedSizeExpr with a bound-var-indexed "
          "FieldLoad leaf reached an LLVM-targeted encoder (which cannot resolve it on-device).",
          src_idx);
      QD_ASSERT_INFO(node.snode_id >= 0, "FieldLoad at node {} has no snode_id", src_idx);
      QD_ASSERT_INFO(prog != nullptr, "encode_subtree: FieldLoad needs a live Program to resolve snode {}",
                     node.snode_id);
      SNode *snode = prog->get_snode_by_id(node.snode_id);
      QD_ASSERT_INFO(snode != nullptr,
                     "FieldLoad at node {} references snode_id={} which is not in the program's snode tree", src_idx,
                     node.snode_id);
      uint64_t base_psb = 0;
      std::vector<int32_t> elem_strides;
      bool fetched = fl_emitter.fetch(snode, &base_psb, &elem_strides);
      QD_ERROR_IF(!fetched,
                  "Adstack SizeExpr FieldLoad at node {} on snode_id={} could not be resolved for device-side "
                  "evaluation: the snode layout is not a pure-dense chain ending in a plain place leaf (bitmasked "
                  "/ pointer / bit-level snodes are not supported by the SPIR-V sizer shader). Rewrite the trip "
                  "count to use a dense field, or extend the shader to walk the non-dense hierarchy.",
                  src_idx, node.snode_id);
      QD_ASSERT_INFO(elem_strides.size() == node.indices.size(),
                     "FieldLoad at node {}: elem_strides.size()={} must match node.indices.size()={} (one stride "
                     "per active axis)",
                     src_idx, elem_strides.size(), node.indices.size());
      AdStackSizeExprDeviceNode dn =
          make_empty_device_node(static_cast<int32_t>(AdStackSizeExprDeviceKind::kFieldLoad));
      dn.const_value = static_cast<int64_t>(base_psb);
      // `PrimitiveTypeID` for the leaf: mirrors ExternalTensorRead's field. The pre-pass emits a `FieldLoad`
      // `SerializedSizeExprNode` with `snode_id` set and the element type implicit in the snode; we look it up
      // here so the shader's existing `emit_psb_load_i64` switch (shared with ETR) can dispatch on it.
      dn.prim_dt = static_cast<int32_t>(snode->dt->cast<PrimitiveType>()->type);
      dn.indices_offset = static_cast<int32_t>(out_indices.size());
      dn.indices_count = static_cast<int32_t>(node.indices.size());
      // Interleaved `[idx_a_raw, elem_stride_a]` pairs per axis. The shader reads 2 i32s per axis and
      // accumulates `idx_a * stride_a` into the element index, then `psb_load_scalar` multiplies by the
      // element size to get the final byte offset. Bound-variable refs (negative entries) are dense-remapped
      // so the device interpreter's fixed-size `scope->values[]` stays in bounds.
      for (std::size_t a = 0; a < node.indices.size(); ++a) {
        int32_t raw = static_cast<int32_t>(node.indices[a]);
        if (raw < 0) {
          int32_t dense = remap_var_id(var_id_remap, -(raw + 1));
          raw = -(dense + 1);
        }
        out_indices.push_back(raw);
        out_indices.push_back(elem_strides[a]);
      }
      out_nodes.push_back(dn);
      return static_cast<int32_t>(out_nodes.size() - 1);
    }
    case SizeExpr::Kind::ExternalTensorShape: {
      // Should have been folded to `Const` by the `subtree_needs_device == false && subtree_closed == true`
      // branch above, since `ExternalTensorShape` has no free vars and cannot be an `ExternalTensorRead`. Hitting
      // this branch is a bug in the encoder, not in the kernel.
      QD_ERROR(
          "Adstack SizeExpr ExternalTensorShape at node {} escaped host folding: this is an encoder invariant"
          " violation - shape nodes are always closed and should have been emitted as Const.",
          src_idx);
      return -1;
    }
  }
  QD_ERROR("encode_subtree: unreachable kind {} at node {}", node.kind, src_idx);
  return -1;
}

}  // namespace

namespace {

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
      // Gen-counter fast skip: when no kernel has bumped this SNode's write generation since record time,
      // the underlying field value cannot have changed and we can return the recorded `observed_value`
      // without dispatching a reader kernel. The dispatch is the dominant per-launch cost on the hot path
      // for steady-state reverse-mode loops with stable bounds.
      if (prog != nullptr && prog->adstack_cache().snode_write_gen(obs.snode_id) == obs.observed_gen) {
        return obs.observed_value;
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
  if (ctx == nullptr) {
    return;
  }
  for (auto &obs : reads) {
    if (obs.kind != AdStackCache::SizeExprReadObservation::ExternalReadObs || obs.arg_id_path.empty()) {
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
    // record is a self-equal cache hit. On gen mismatch the replay re-dereferences `data[0]` instead, which
    // (under any sub-i64 prim_dt the recognizer admits) widens to an i64 strictly greater than `INT64_MIN` and
    // forces the cache to invalidate. The dispatched max itself lives in `MaxReducerCacheEntry::result`; this
    // observation only gates whether the cache stays warm.
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
  // so an empty bound-vars map is sufficient. Read observations are not recorded - the caller (max-reducer
  // launcher) does its own observation tracking via `AdStackCache::record_max_reducer_eval` against the spec key,
  // not the per-`SerializedSizeExpr` key the cache uses for `evaluate_adstack_size_expr`.
  SizeExprLaunchScope local_scope;
  std::unordered_map<int32_t, int64_t> empty_bound_vars;
  return evaluate_node(expr, node_idx, empty_bound_vars, prog, ctx, /*reads=*/nullptr);
}

namespace {

// Diagnose-time leaf reader: resolves an `ExternalTensorRead` against the captured
// `AdStackCache::DiagnoseLaunchSnapshot` and the program's `Device::map` interface. Returns -1 on any failure
// (missing arg in snapshot, unrecognised primitive type, mapping failure) so the caller can substitute the
// `?` placeholder for that stack while keeping the rest of the message intact.
//
// Single-scalar staging-buffer pattern (mirrors `Ndarray::read` in `program/ndarray.cpp`): allocate a tiny
// `host_read=true` staging buffer, `memcpy_internal` the one element from the ndarray's device buffer into
// staging, then map staging to read the value host-side. This works on every backend because every
// `Device` implementation supports `host_read=true` allocations + `map` + `memcpy_internal`. For `kNone`
// numpy passthrough the captured pointer is already host-readable; we read it directly.
int64_t read_diagnose_external_tensor(const SerializedSizeExprNode &node,
                                      const std::vector<int64_t> &resolved_indices,
                                      Program *prog,
                                      const AdStackCache::DiagnoseLaunchSnapshot &snapshot) {
  if (node.arg_id_path.empty()) {
    return -1;
  }
  int arg_id = node.arg_id_path[0];
  auto ptr_it = snapshot.data_ptrs.find(arg_id);
  if (ptr_it == snapshot.data_ptrs.end() || ptr_it->second == nullptr) {
    return -1;
  }
  auto type_it = snapshot.dev_alloc_types.find(arg_id);
  if (type_it == snapshot.dev_alloc_types.end()) {
    return -1;
  }
  auto shape_it = snapshot.shapes.find(arg_id);
  if (shape_it == snapshot.shapes.end()) {
    return -1;
  }
  const std::vector<int32_t> &shape = shape_it->second;
  // Compose C-order linear offset across resolved indices (mirrors `evaluate_external_tensor_read`'s stride
  // math; we cannot share the helper because that one routes through `LaunchContextBuilder::get_struct_arg_host`
  // which is not available here).
  if (resolved_indices.size() > shape.size() && !shape.empty()) {
    // More indices than rank - the size_expr was lowered against a different shape; skip.
    return -1;
  }
  int64_t linear = 0;
  int64_t stride = 1;
  for (std::size_t i = resolved_indices.size(); i > 0; --i) {
    linear += resolved_indices[i - 1] * stride;
    if (i - 1 > 0 && i - 1 < shape.size()) {
      stride *= static_cast<int64_t>(shape[i - 1]);
    }
  }
  auto prim_dt = static_cast<PrimitiveTypeID>(node.const_value);
  std::size_t elem_size = 0;
  switch (prim_dt) {
    case PrimitiveTypeID::i8:
    case PrimitiveTypeID::u8:
      elem_size = 1;
      break;
    case PrimitiveTypeID::i16:
    case PrimitiveTypeID::u16:
      elem_size = 2;
      break;
    case PrimitiveTypeID::i32:
    case PrimitiveTypeID::u32:
      elem_size = 4;
      break;
    case PrimitiveTypeID::i64:
    case PrimitiveTypeID::u64:
      elem_size = 8;
      break;
    default:
      return -1;
  }
  std::size_t byte_offset = static_cast<std::size_t>(linear) * elem_size;
  // Decode the captured pointer to host bytes.
  std::vector<uint8_t> staging_bytes(elem_size);
  if (type_it->second == LaunchContextBuilder::DevAllocType::kNone) {
    // Numpy passthrough: ptr is already a raw host pointer.
    const uint8_t *src = static_cast<const uint8_t *>(ptr_it->second) + byte_offset;
    std::memcpy(staging_bytes.data(), src, elem_size);
  } else if (type_it->second == LaunchContextBuilder::DevAllocType::kNdarray) {
    if (prog == nullptr) {
      return -1;
    }
    auto *alloc = static_cast<DeviceAllocation *>(ptr_it->second);
    if (alloc == nullptr || alloc->device == nullptr) {
      return -1;
    }
    Device::AllocParams params;
    params.host_write = false;
    params.host_read = true;
    params.size = elem_size;
    params.usage = AllocUsage::Storage;
    auto [staging, alloc_res] = alloc->device->allocate_memory_unique(params);
    if (alloc_res != RhiResult::success || !staging) {
      return -1;
    }
    alloc->device->memcpy_internal(staging->get_ptr(), alloc->get_ptr(byte_offset), elem_size);
    void *mapped = nullptr;
    if (alloc->device->map(*staging, &mapped) != RhiResult::success || mapped == nullptr) {
      return -1;
    }
    std::memcpy(staging_bytes.data(), mapped, elem_size);
    alloc->device->unmap(*staging);
  } else {
    return -1;
  }
  // Sign- / zero-extend to int64 according to the captured primitive type.
  switch (prim_dt) {
    case PrimitiveTypeID::i8:
      return static_cast<int64_t>(*reinterpret_cast<const int8_t *>(staging_bytes.data()));
    case PrimitiveTypeID::u8:
      return static_cast<int64_t>(*reinterpret_cast<const uint8_t *>(staging_bytes.data()));
    case PrimitiveTypeID::i16:
      return static_cast<int64_t>(*reinterpret_cast<const int16_t *>(staging_bytes.data()));
    case PrimitiveTypeID::u16:
      return static_cast<int64_t>(*reinterpret_cast<const uint16_t *>(staging_bytes.data()));
    case PrimitiveTypeID::i32:
      return static_cast<int64_t>(*reinterpret_cast<const int32_t *>(staging_bytes.data()));
    case PrimitiveTypeID::u32:
      return static_cast<int64_t>(*reinterpret_cast<const uint32_t *>(staging_bytes.data()));
    case PrimitiveTypeID::i64:
      return *reinterpret_cast<const int64_t *>(staging_bytes.data());
    case PrimitiveTypeID::u64:
      return static_cast<int64_t>(*reinterpret_cast<const uint64_t *>(staging_bytes.data()));
    default:
      return -1;
  }
}

// Mirror of `evaluate_node` for diagnose-time evaluation. Same tree-walk semantics; differs only in the leaf
// case for `ExternalTensorRead` / `ExternalTensorShape`, which route through the snapshot + `Device::map` path
// instead of `LaunchContextBuilder`. Returns -1 on any leaf-resolution failure to short-circuit the rest of
// the walk and let the caller fall back to the static dual-cause body.
int64_t evaluate_node_for_diagnose(const SerializedSizeExpr &expr,
                                   int32_t node_idx,
                                   std::unordered_map<int32_t, int64_t> &bound_vars,
                                   Program *prog,
                                   const AdStackCache::DiagnoseLaunchSnapshot &snapshot) {
  if (node_idx < 0 || static_cast<std::size_t>(node_idx) >= expr.nodes.size()) {
    return -1;
  }
  const auto &node = expr.nodes[node_idx];
  switch (static_cast<SizeExpr::Kind>(node.kind)) {
    case SizeExpr::Kind::Const:
      return node.const_value;
    case SizeExpr::Kind::FieldLoad: {
      // Field reads stay on the existing host path - they do not depend on `LaunchContextBuilder` and the
      // SNode reader-kernel dispatch is host-driven. We pass `nullptr` ReadSink so the recorded observations
      // do not leak into the cache from a diagnose-only walk.
      return evaluate_field_load(node, bound_vars, prog, /*reads=*/nullptr);
    }
    case SizeExpr::Kind::Add: {
      int64_t a = evaluate_node_for_diagnose(expr, node.operand_a, bound_vars, prog, snapshot);
      int64_t b = evaluate_node_for_diagnose(expr, node.operand_b, bound_vars, prog, snapshot);
      if (a < 0 || b < 0) {
        return -1;
      }
      return a + b;
    }
    case SizeExpr::Kind::Sub: {
      int64_t a = evaluate_node_for_diagnose(expr, node.operand_a, bound_vars, prog, snapshot);
      int64_t b = evaluate_node_for_diagnose(expr, node.operand_b, bound_vars, prog, snapshot);
      if (a < 0 || b < 0) {
        return -1;
      }
      return std::max<int64_t>(a - b, 0);
    }
    case SizeExpr::Kind::Mul: {
      int64_t a = evaluate_node_for_diagnose(expr, node.operand_a, bound_vars, prog, snapshot);
      int64_t b = evaluate_node_for_diagnose(expr, node.operand_b, bound_vars, prog, snapshot);
      if (a < 0 || b < 0) {
        return -1;
      }
      return a * b;
    }
    case SizeExpr::Kind::Max: {
      int64_t a = evaluate_node_for_diagnose(expr, node.operand_a, bound_vars, prog, snapshot);
      int64_t b = evaluate_node_for_diagnose(expr, node.operand_b, bound_vars, prog, snapshot);
      if (a < 0 || b < 0) {
        return -1;
      }
      return std::max(a, b);
    }
    case SizeExpr::Kind::MaxOverRange: {
      int64_t begin = evaluate_node_for_diagnose(expr, node.operand_a, bound_vars, prog, snapshot);
      int64_t end = evaluate_node_for_diagnose(expr, node.operand_b, bound_vars, prog, snapshot);
      if (begin < 0 || end < 0) {
        return -1;
      }
      // Same iteration cap as the live evaluator; refusing to enumerate prevents diagnose from stalling
      // the error path on a pathological trip count.
      constexpr int64_t kMaxOverRangeIterations = int64_t{1} << 24;
      if (end > begin && end - begin > kMaxOverRangeIterations) {
        return -1;
      }
      int64_t result = 0;
      auto prev_it = bound_vars.find(node.var_id);
      bool had_prev = prev_it != bound_vars.end();
      int64_t prev_val = had_prev ? prev_it->second : 0;
      for (int64_t i = begin; i < end; ++i) {
        bound_vars[node.var_id] = i;
        int64_t v = evaluate_node_for_diagnose(expr, node.body_node_idx, bound_vars, prog, snapshot);
        if (v < 0) {
          if (had_prev) {
            bound_vars[node.var_id] = prev_val;
          } else {
            bound_vars.erase(node.var_id);
          }
          return -1;
        }
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
      if (it == bound_vars.end()) {
        return -1;
      }
      return it->second;
    }
    case SizeExpr::Kind::ExternalTensorShape: {
      if (node.arg_id_path.empty()) {
        return -1;
      }
      int arg_id = node.arg_id_path[0];
      auto shape_it = snapshot.shapes.find(arg_id);
      if (shape_it == snapshot.shapes.end()) {
        return -1;
      }
      if (node.arg_shape_axis < 0 || static_cast<std::size_t>(node.arg_shape_axis) >= shape_it->second.size()) {
        return -1;
      }
      return static_cast<int64_t>(shape_it->second[node.arg_shape_axis]);
    }
    case SizeExpr::Kind::ExternalTensorRead: {
      // Resolve indices from bound_vars first, then dispatch to the snapshot-aware reader.
      std::vector<int64_t> resolved(node.indices.size());
      for (std::size_t i = 0; i < node.indices.size(); ++i) {
        int32_t raw = node.indices[i];
        if (raw >= 0) {
          resolved[i] = raw;
        } else {
          int32_t var_id = -(raw + 1);
          auto bv = bound_vars.find(var_id);
          if (bv == bound_vars.end()) {
            return -1;
          }
          resolved[i] = bv->second;
        }
      }
      return read_diagnose_external_tensor(node, resolved, prog, snapshot);
    }
  }
  return -1;
}

}  // namespace

int64_t evaluate_adstack_size_expr_for_diagnose(const SerializedSizeExpr &expr, Program *prog) {
  if (expr.nodes.empty() || prog == nullptr) {
    return -1;
  }
  const AdStackCache::DiagnoseLaunchSnapshot *snapshot = prog->adstack_cache().get_diagnose_snapshot();
  if (snapshot == nullptr) {
    return -1;
  }
  std::unordered_map<int32_t, int64_t> bound_vars;
  return evaluate_node_for_diagnose(expr, static_cast<int32_t>(expr.nodes.size() - 1), bound_vars, prog, *snapshot);
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

namespace {

// True iff the body subtree rooted at `node_idx` references only `Const`, `ExternalTensorRead(arg,
// [BoundVariable(expected_var_id)])`, and `Add` / `Sub` / `Mul` / `Max` of those (The recognizer grammar of the
// max-reducer plan). Single-axis ndarray reads only; the index slot must be exactly `-(expected_var_id + 1)`
// per the `SerializedSizeExprNode::indices` encoding (negative entries reference an enclosing
// `MaxOverRange`'s bound variable). Returns false on any out-of-grammar node.
bool max_reducer_body_is_recognizable(const SerializedSizeExpr &expr, int32_t node_idx, int32_t expected_var_id) {
  if (node_idx < 0 || static_cast<std::size_t>(node_idx) >= expr.nodes.size()) {
    return false;
  }
  const auto &n = expr.nodes[node_idx];
  switch (static_cast<SizeExpr::Kind>(n.kind)) {
    case SizeExpr::Kind::Const:
      return true;
    case SizeExpr::Kind::ExternalTensorRead:
      return n.indices.size() == 1 && n.indices[0] == -(expected_var_id + 1);
    case SizeExpr::Kind::Add:
    case SizeExpr::Kind::Sub:
    case SizeExpr::Kind::Mul:
    case SizeExpr::Kind::Max:
      return max_reducer_body_is_recognizable(expr, n.operand_a, expected_var_id) &&
             max_reducer_body_is_recognizable(expr, n.operand_b, expected_var_id);
    default:
      return false;
  }
}

// True iff the bound subtree rooted at `node_idx` evaluates to a closed-form scalar after substituting any
// `MaxOverRange` nodes already captured (`captured_mors`) as `Const`s. Allowed: `Const`, `ExternalTensorShape`,
// `Add` / `Sub` / `Mul` / `Max` of recursively-closed subtrees, and `MaxOverRange` whose node index is in
// `captured_mors`. On success appends every captured-MOR dependency this subtree references to `deps_out`.
bool max_reducer_bound_is_closed(const SerializedSizeExpr &expr,
                                 int32_t node_idx,
                                 const std::unordered_set<int32_t> &captured_mors,
                                 std::vector<int32_t> &deps_out) {
  if (node_idx < 0 || static_cast<std::size_t>(node_idx) >= expr.nodes.size()) {
    return false;
  }
  const auto &n = expr.nodes[node_idx];
  switch (static_cast<SizeExpr::Kind>(n.kind)) {
    case SizeExpr::Kind::Const:
    case SizeExpr::Kind::ExternalTensorShape:
      return true;
    case SizeExpr::Kind::Add:
    case SizeExpr::Kind::Sub:
    case SizeExpr::Kind::Mul:
    case SizeExpr::Kind::Max:
      return max_reducer_bound_is_closed(expr, n.operand_a, captured_mors, deps_out) &&
             max_reducer_bound_is_closed(expr, n.operand_b, captured_mors, deps_out);
    case SizeExpr::Kind::MaxOverRange: {
      if (captured_mors.count(node_idx) == 0) {
        return false;
      }
      deps_out.push_back(node_idx);
      return true;
    }
    default:
      return false;  // FieldLoad, BoundVariable from a non-immediately-enclosing scope, ExternalTensorRead, etc.
  }
}

}  // namespace

std::vector<StaticAdStackMaxReducerSpec> recognize_adstack_max_reducer_specs(
    const std::vector<SerializedSizeExpr> &size_exprs) {
  std::vector<StaticAdStackMaxReducerSpec> specs;
  for (std::size_t stack_id = 0; stack_id < size_exprs.size(); ++stack_id) {
    const auto &expr = size_exprs[stack_id];
    // `SerializedSizeExpr` is built post-order so deeper `MaxOverRange` nodes always have a smaller `n` than
    // the outer `MaxOverRange` that depends on them. Iterating ascending `n` therefore visits dependencies
    // before dependants and `captured_mors` is always populated in the right order for `max_reducer_bound_is_closed`.
    std::unordered_set<int32_t> captured_mors;
    for (std::size_t n = 0; n < expr.nodes.size(); ++n) {
      const auto &node = expr.nodes[n];
      if (static_cast<SizeExpr::Kind>(node.kind) != SizeExpr::Kind::MaxOverRange) {
        continue;
      }
      if (!max_reducer_body_is_recognizable(expr, node.body_node_idx, node.var_id)) {
        continue;
      }
      std::vector<int32_t> deps;
      if (!max_reducer_bound_is_closed(expr, node.operand_a, captured_mors, deps)) {
        continue;
      }
      if (!max_reducer_bound_is_closed(expr, node.operand_b, captured_mors, deps)) {
        continue;
      }
      StaticAdStackMaxReducerSpec spec;
      spec.stack_id = static_cast<int32_t>(stack_id);
      spec.mor_node_idx = static_cast<int32_t>(n);
      spec.begin_node_idx = node.operand_a;
      spec.end_node_idx = node.operand_b;
      spec.body_node_idx = node.body_node_idx;
      spec.var_id = node.var_id;
      spec.dependent_mor_node_idxs = std::move(deps);
      specs.push_back(std::move(spec));
      captured_mors.insert(static_cast<int32_t>(n));
    }
  }
  return specs;
}

EncodedMaxReducerBody encode_max_reducer_body_bytecode(
    const SerializedSizeExpr &expr,
    int32_t body_node_idx,
    int32_t bound_var_id,
    const std::function<int32_t(const std::vector<int32_t> &arg_id_path)> &arg_buffer_offset_resolver) {
  EncodedMaxReducerBody out;
  if (body_node_idx < 0 || static_cast<std::size_t>(body_node_idx) >= expr.nodes.size()) {
    return out;
  }
  // Post-order DFS to collect reachable node indices from `body_node_idx`. The recognizer grammar guarantees no
  // `kMaxOverRange` / `kFieldLoad` in the body subtree, so we only need to follow `operand_a` / `operand_b`
  // (binary ops) and `kExternalTensorRead` (no operands beyond indices). The resulting `post_order` vector is
  // sorted such that any node's operands precede the node itself.
  std::vector<int32_t> post_order;
  std::unordered_map<int32_t, int32_t> old_to_new;  // old idx -> dense [0, body_node_count)
  std::function<void(int32_t)> visit = [&](int32_t idx) {
    if (idx < 0 || old_to_new.count(idx) != 0) {
      return;
    }
    const auto &n = expr.nodes[idx];
    auto kind = static_cast<SizeExpr::Kind>(n.kind);
    if (kind == SizeExpr::Kind::Add || kind == SizeExpr::Kind::Sub || kind == SizeExpr::Kind::Mul ||
        kind == SizeExpr::Kind::Max) {
      visit(n.operand_a);
      visit(n.operand_b);
    }
    // `kConst`, `kBoundVariable`, `kExternalTensorRead` are leaves (`indices` are constants or bound-var refs).
    int32_t new_idx = static_cast<int32_t>(post_order.size());
    old_to_new[idx] = new_idx;
    post_order.push_back(idx);
  };
  visit(body_node_idx);

  out.body_node_count = static_cast<uint32_t>(post_order.size());

  // Build the flat indices table for any `kExternalTensorRead` leaves. Each leaf carries `indices_count` axes;
  // each axis contributes one `(idx_raw, elem_stride)` pair. `idx_raw` mirrors the host SerializedSizeExprNode
  // encoding (`-(var_id + 1)` for bound-var refs, non-negative for constants) - the encoder remaps `bound_var_id`
  // refs to `-(0 + 1) = -1` since the device-side scope has only one bound variable per spec. `elem_stride` is
  // resolved to the per-axis element stride (in elements, not bytes) by the caller via the same convention the
  // adstack sizer encoder uses.
  std::vector<int32_t> indices_table;
  // Build `AdStackSizeExprDeviceNode`s in post-order. We only emit fields the device interpreter reads for the
  // recognized grammar; unused fields stay at their default values.
  std::vector<AdStackSizeExprDeviceNode> device_nodes(post_order.size());
  for (std::size_t i = 0; i < post_order.size(); ++i) {
    const auto &src = expr.nodes[post_order[i]];
    auto &dst = device_nodes[i];
    dst.var_id = -1;
    auto kind = static_cast<SizeExpr::Kind>(src.kind);
    // Map the host `SizeExpr::Kind` enum into the device-side `AdStackSizeExprDeviceKind` enum: the two enums use
    // different integer values (e.g. host `ExternalTensorRead = 9` vs. device `kExternalTensorRead = 7`), so a raw
    // assignment lands every body node in the device interpreter's switch default and returns 0 on every walk.
    // Mirror the explicit translation the per-task adstack-sizer encoder does (search `AdStackSizeExprDeviceKind::`
    // in this TU for the canonical pattern); the max-reducer body grammar narrows to the subset listed below.
    switch (kind) {
      case SizeExpr::Kind::Const:
        dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kConst);
        dst.const_value = src.const_value;
        break;
      case SizeExpr::Kind::BoundVariable:
        // Device-side scope holds the bound variable at slot 0 (only one per spec). The runtime function pre-
        // populates `scope.values[var_id]`; the SPIR-V max-reducer shader substitutes `iter_var` directly.
        dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kBoundVariable);
        dst.var_id = 0;
        break;
      case SizeExpr::Kind::ExternalTensorRead: {
        dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kExternalTensorRead);
        dst.prim_dt = static_cast<int32_t>(src.const_value);
        // Resolve `arg_buffer_offset` from `arg_id_path` via the caller's resolver.
        std::vector<int32_t> path = src.arg_id_path;
        const int32_t arg_buf_off = arg_buffer_offset_resolver(path);
        if (arg_buf_off < 0) {
          return EncodedMaxReducerBody{};  // resolver failed; signal empty result
        }
        dst.arg_buffer_offset = arg_buf_off;
        // Indices table: emit `(idx_raw, elem_stride=1)` per axis. The recognizer restricts to single-axis reads
        // indexed by the bound variable (encoded as `-(this_var_id + 1)` in the host tree); the device-side scope
        // remaps that to slot 0, so we re-encode as `-1` (= `-(0 + 1)`). Element stride is 1 because the body's ndarray
        // is treated as a flat 1D buffer; multi-axis support is future work.
        const int32_t indices_off = static_cast<int32_t>(indices_table.size());
        for (std::size_t a = 0; a < src.indices.size(); ++a) {
          int64_t raw = src.indices[a];
          int32_t emit_raw;
          if (raw >= 0) {
            emit_raw = static_cast<int32_t>(raw);
          } else if (-(raw + 1) == bound_var_id) {
            emit_raw = -1;  // -(0 + 1)
          } else {
            // Body grammar guarantees only the spec's bound var appears; reaching here is an analyzer bug.
            return EncodedMaxReducerBody{};
          }
          indices_table.push_back(emit_raw);
          indices_table.push_back(1);  // elem_stride
        }
        dst.indices_offset = indices_off;
        dst.indices_count = static_cast<int32_t>(src.indices.size());

        // Record a body observation entry so the caller can populate the cache's read list. The caller fills in
        // `observed_value` and `observed_gen` post-eval (we do not have the live ctx here).
        AdStackCache::SizeExprReadObservation obs{};
        obs.kind = AdStackCache::SizeExprReadObservation::ExternalReadObs;
        obs.snode_id = -1;
        obs.arg_id_path = std::vector<int>(src.arg_id_path.begin(), src.arg_id_path.end());
        obs.prim_dt = static_cast<int>(src.const_value);
        out.body_reads.push_back(std::move(obs));
        break;
      }
      case SizeExpr::Kind::Add:
      case SizeExpr::Kind::Sub:
      case SizeExpr::Kind::Mul:
      case SizeExpr::Kind::Max: {
        if (kind == SizeExpr::Kind::Add) {
          dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kAdd);
        } else if (kind == SizeExpr::Kind::Sub) {
          dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kSub);
        } else if (kind == SizeExpr::Kind::Mul) {
          dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kMul);
        } else {
          dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kMax);
        }
        auto map_op = [&](int32_t old) -> int32_t {
          auto it = old_to_new.find(old);
          return it == old_to_new.end() ? -1 : it->second;
        };
        dst.operand_a = map_op(src.operand_a);
        dst.operand_b = map_op(src.operand_b);
        break;
      }
      default:
        // Out-of-grammar kind reached the encoder; the caller should have filtered via
        // `recognize_adstack_max_reducer_specs`. Return empty to signal failure.
        return EncodedMaxReducerBody{};
    }
  }

  out.indices_count = static_cast<uint32_t>(indices_table.size());
  // Concatenate `[device_nodes][indices_table]` into the output bytes buffer.
  const std::size_t nodes_bytes = device_nodes.size() * sizeof(AdStackSizeExprDeviceNode);
  const std::size_t indices_bytes = indices_table.size() * sizeof(int32_t);
  out.bytes.resize(nodes_bytes + indices_bytes);
  if (nodes_bytes > 0) {
    std::memcpy(out.bytes.data(), device_nodes.data(), nodes_bytes);
  }
  if (indices_bytes > 0) {
    std::memcpy(out.bytes.data() + nodes_bytes, indices_table.data(), indices_bytes);
  }
  return out;
}

SerializedSizeExpr substitute_precomputed_max_over_range(const SerializedSizeExpr &expr,
                                                         uint32_t registry_id,
                                                         int32_t stack_id,
                                                         const MaxReducerResultMap &results) {
  if (results.empty()) {
    return expr;
  }
  auto pack_key = [&](std::size_t n) {
    return (static_cast<uint64_t>(registry_id) & 0xFFFFFFFFull) |
           ((static_cast<uint64_t>(stack_id) & 0xFFFFull) << 32) | ((static_cast<uint64_t>(n) & 0xFFFFull) << 48);
  };
  // Cheap precheck: any `MaxOverRange` node in this expr with a key in `results`? If not, return verbatim.
  bool any_match = false;
  for (std::size_t n = 0; n < expr.nodes.size(); ++n) {
    if (static_cast<SizeExpr::Kind>(expr.nodes[n].kind) != SizeExpr::Kind::MaxOverRange) {
      continue;
    }
    if (results.count(pack_key(n)) != 0) {
      any_match = true;
      break;
    }
  }
  if (!any_match) {
    return expr;
  }
  // Build a copy with substitution applied to matching MaxOverRange nodes. Node count is unchanged so operand
  // indices in non-substituted nodes stay valid; substituted nodes become `kConst` leaves whose `const_value` is
  // the dispatched max-reducer result.
  SerializedSizeExpr out = expr;
  for (std::size_t n = 0; n < out.nodes.size(); ++n) {
    auto &node = out.nodes[n];
    if (static_cast<SizeExpr::Kind>(node.kind) != SizeExpr::Kind::MaxOverRange) {
      continue;
    }
    auto it = results.find(pack_key(n));
    if (it == results.end()) {
      continue;
    }
    node.kind = static_cast<int32_t>(SizeExpr::Kind::Const);
    node.const_value = it->second;
    // Defensive cleanup: the host evaluator's `kConst` arm reads only `const_value`. Reset operand / body /
    // var_id slots to -1 so any future reader that does not branch on `kind` produces a deterministic failure
    // rather than reading stale indices.
    node.operand_a = -1;
    node.operand_b = -1;
    node.body_node_idx = -1;
    node.var_id = -1;
  }
  return out;
}

void clip_effective_rows_by_loop_trip_count(std::size_t &effective_rows,
                                            const StaticAdStackBoundExpr &bound_expr,
                                            std::size_t dispatched_threads_ceiling,
                                            Program *prog,
                                            LaunchContextBuilder *ctx) {
  if (bound_expr.loop_iter_static > 0) {
    // Compile-time trip count: integer compare, no per-launch eval cost. Constant `SizeExpr` shapes are
    // already collapsed into this field by the analyzer so they short-circuit the runtime eval below.
    const std::size_t loop_iter_static = static_cast<std::size_t>(bound_expr.loop_iter_static);
    if (loop_iter_static <= dispatched_threads_ceiling) {
      effective_rows = std::min<std::size_t>(effective_rows, loop_iter_static);
    }
    return;
  }
  if (bound_expr.loop_iter_size_expr.nodes.empty() || prog == nullptr || ctx == nullptr) {
    // Runtime tree empty or no resolution context: the analyzer left this field unset for shapes the
    // compile-time path could not cover (or the caller did not supply a `Program` / `LaunchContextBuilder`),
    // so leave `effective_rows` alone and let the caller fall back to the unclipped reducer count.
    return;
  }
  // Runtime-bounded clip: evaluate the captured trip-count `SizeExpr` only when the static field is unset
  // (the analyzer leaves `loop_iter_static == 0` for shapes the compile-time path cannot cover, e.g.
  // `for j in range(field[i])` / `for k in range(arr.shape[axis])`). Cost = one tree walk per launch,
  // dominated by host scalar reads through `SNodeRwAccessorsBank` on `FieldLoad` / `ExternalTensorRead`
  // nodes (CPU: a memory load; CUDA / AMDGPU: a 4-8 byte DtoH). The evaluator returns -1 when the tree
  // references state that is not host-resolvable from `ctx`; in that case we leave `effective_rows`
  // unclipped from this source.
  const int64_t evaluated = evaluate_adstack_size_expr(bound_expr.loop_iter_size_expr, prog, ctx);
  if (evaluated > 0 && static_cast<std::size_t>(evaluated) <= dispatched_threads_ceiling) {
    effective_rows = std::min<std::size_t>(effective_rows, static_cast<std::size_t>(evaluated));
  }
}

namespace {

// Shared back-end for both encoder variants. Takes already-populated stack headers (with
// `entry_size_bytes` / `max_size_compile_time` / `heap_kind` set per stack, `root_node_idx` defaulted to
// `-1`) plus the per-stack source trees, runs the tree-to-bytecode substitution-aware flattening, and
// returns the packed byte buffer ready to upload to a device scratch buffer.
std::vector<uint8_t> encode_bytecode_common(std::vector<AdStackSizeExprDeviceStackHeader> stack_headers,
                                            const std::vector<const SerializedSizeExpr *> &exprs,
                                            Program *prog,
                                            LaunchContextBuilder *ctx,
                                            const FieldLoadDeviceEmitter &fl_emitter,
                                            int max_nodes_per_stack = 0,
                                            ReadSink *reads = nullptr) {
  const std::size_t n_stacks = stack_headers.size();
  QD_ASSERT(exprs.size() == n_stacks);

  std::vector<AdStackSizeExprDeviceNode> nodes;
  std::vector<int32_t> indices;
  nodes.reserve(n_stacks);
  indices.reserve(n_stacks);

  const bool fieldload_stays_on_device = !fl_emitter.empty();
  for (std::size_t i = 0; i < n_stacks; ++i) {
    auto &sh = stack_headers[i];
    const SerializedSizeExpr *expr = exprs[i];
    if (std::getenv("QD_DEBUG_ADSTACK")) {
      fprintf(stderr, "[encode] stack[%zu]: expr=%p nodes=%zu max_size_ct=%u\n", i, (void *)expr,
              expr ? expr->nodes.size() : 0, sh.max_size_compile_time);
      if (expr) {
        for (size_t n = 0; n < expr->nodes.size(); ++n) {
          const auto &node = expr->nodes[n];
          fprintf(stderr,
                  "[encode]   node[%zu]: kind=%d const=%lld snode_id=%d var_id=%d op_a=%d op_b=%d body=%d "
                  "axis=%d",
                  n, node.kind, (long long)node.const_value, node.snode_id, node.var_id, node.operand_a, node.operand_b,
                  node.body_node_idx, node.arg_shape_axis);
          if (!node.arg_id_path.empty()) {
            fprintf(stderr, " arg_id=[");
            for (int32_t v : node.arg_id_path)
              fprintf(stderr, "%d,", v);
            fprintf(stderr, "]");
          }
          if (!node.indices.empty()) {
            fprintf(stderr, " idx=[");
            for (int32_t v : node.indices)
              fprintf(stderr, "%d,", v);
            fprintf(stderr, "]");
          }
          fprintf(stderr, "\n");
        }
        // Host-side ground-truth evaluation: if the shader later writes a different `max_size`, the delta
        // pinpoints a shader-side bug rather than a pre-pass / SerializedSizeExpr bug. Skip when the caller
        // passes `ctx == nullptr` (C++-only test harnesses) and when an `ExternalTensorRead` leaf exists but
        // `ctx->array_ptrs` has not been populated (the CPU launcher populates it via
        // `set_host_accessible_ndarray_ptrs`; SPIR-V launchers use the device-side PSB path instead, so
        // `array_ptrs` is empty and the host-eval would crash on the missing key).
        if (!expr->nodes.empty() && prog != nullptr && ctx != nullptr) {
          bool has_etr = false;
          for (const auto &node : expr->nodes) {
            if (static_cast<SizeExpr::Kind>(node.kind) == SizeExpr::Kind::ExternalTensorRead) {
              has_etr = true;
              break;
            }
          }
          if (has_etr && ctx->array_ptrs.empty()) {
            fprintf(stderr, "[encode] stack[%zu]: host_eval=skipped (ctx->array_ptrs empty)\n", i);
          } else {
            int64_t host_val = evaluate_adstack_size_expr(*expr, prog, ctx);
            fprintf(stderr, "[encode] stack[%zu]: host_eval=%lld\n", i, (long long)host_val);
          }
        }
      }
    }
    if (expr == nullptr || expr->nodes.empty()) {
      // No symbolic bound captured - the device interpreter will route this slot to `max_size_compile_time`.
      sh.root_node_idx = -1;
      continue;
    }
    auto contains_device_leaf = compute_contains_device_leaf(*expr, fieldload_stays_on_device);
    auto free_vars = compute_free_vars(*expr);
    const std::size_t root_src_idx = expr->nodes.size() - 1;
    QD_ASSERT_INFO(free_vars[root_src_idx].empty(),
                   "Adstack SizeExpr tree root for stack {} has {} free bound variable(s); a well-formed tree"
                   " must be closed at the root because no outer MaxOverRange scope exists at publish time.",
                   i, free_vars[root_src_idx].size());
    // Dense-remap the tree's `var_id`s before emitting device nodes: `var_id_counter` on the host is a monotonic
    // per-alloca counter bumped at every chased non-const index / stash, so a complex reverse-mode kernel can
    // exceed the device interpreter's fixed-size scope capacity even with modest nesting. The encoder hard-errors
    // here rather than letting the interpreter silently drop binds and return wrong `max_size` values.
    auto var_id_remap = build_dense_var_id_remap(*expr);
    const int32_t mor_depth = compute_max_mor_nesting(*expr);
    QD_ERROR_IF(mor_depth > spirv::kAdStackSizerMaxPendingFrames,
                "Adstack SizeExpr for stack {} has MaxOverRange nesting depth {}, which exceeds the sizer shader's "
                "`kAdStackSizerMaxPendingFrames` ({}) pending-frame capacity. Past this cap the shader's fixed-size "
                "pending-frame stack would index out of bounds - SPIR-V private-storage OOB is UB. Shrink the "
                "enclosing reverse-mode loop nesting or file a bug so the cap can be raised.",
                i, mor_depth, spirv::kAdStackSizerMaxPendingFrames);
    const std::size_t nodes_before = nodes.size();
    sh.root_node_idx = encode_subtree(*expr, static_cast<int32_t>(root_src_idx), contains_device_leaf, free_vars,
                                      var_id_remap, prog, ctx, fl_emitter, nodes, indices, reads);
    if (max_nodes_per_stack > 0) {
      const std::size_t per_stack = nodes.size() - nodes_before;
      QD_ERROR_IF(per_stack > static_cast<std::size_t>(max_nodes_per_stack),
                  "Adstack SizeExpr for stack {} encodes {} device nodes, which exceeds the sizer shader's per-stack "
                  "`kAdStackSizerMaxNodesPerStack` ({}) scratch capacity. Shrink the reverse-mode loop shape or file a "
                  "bug - past this cap the on-device interpreter would silently truncate its private `values_arr` and "
                  "surface later as a mysterious adstack overflow.",
                  i, per_stack, max_nodes_per_stack);
    }
  }

  // Pack everything into a flat byte buffer: header | stack_headers | nodes | indices.
  AdStackSizeExprDeviceHeader header{};
  header.n_stacks = static_cast<uint32_t>(n_stacks);
  header.total_nodes = static_cast<uint32_t>(nodes.size());
  header.total_indices = static_cast<uint32_t>(indices.size());
  header._pad = 0;

  const std::size_t bytes_header = sizeof(AdStackSizeExprDeviceHeader);
  const std::size_t bytes_stack_headers = sizeof(AdStackSizeExprDeviceStackHeader) * n_stacks;
  const std::size_t bytes_nodes = sizeof(AdStackSizeExprDeviceNode) * nodes.size();
  const std::size_t bytes_indices = sizeof(int32_t) * indices.size();
  const std::size_t total_bytes = bytes_header + bytes_stack_headers + bytes_nodes + bytes_indices;

  std::vector<uint8_t> buffer(total_bytes);
  std::size_t cursor = 0;
  std::memcpy(buffer.data() + cursor, &header, bytes_header);
  cursor += bytes_header;
  if (bytes_stack_headers > 0) {
    std::memcpy(buffer.data() + cursor, stack_headers.data(), bytes_stack_headers);
    cursor += bytes_stack_headers;
  }
  if (bytes_nodes > 0) {
    std::memcpy(buffer.data() + cursor, nodes.data(), bytes_nodes);
    cursor += bytes_nodes;
  }
  if (bytes_indices > 0) {
    std::memcpy(buffer.data() + cursor, indices.data(), bytes_indices);
    cursor += bytes_indices;
  }
  QD_ASSERT(cursor == total_bytes);
  return buffer;
}

}  // namespace

std::vector<uint8_t> encode_adstack_size_expr_device_bytecode(const AdStackSizingInfo &ad_stack,
                                                              Program *prog,
                                                              LaunchContextBuilder *ctx,
                                                              const MaxReducerResultMap &max_reducer_results) {
  const std::size_t n_stacks = ad_stack.allocas.size();
  std::vector<AdStackSizeExprDeviceStackHeader> stack_headers(n_stacks);
  std::vector<const SerializedSizeExpr *> exprs(n_stacks, nullptr);
  // Per-stack substituted trees: if the max-reducer dispatched a value for any
  // captured `MaxOverRange`, swap it in as a `Const` BEFORE the device interpreter walks the tree. Storage owns
  // the substituted copies so `exprs[i]` (a pointer) remains valid through `encode_bytecode_common`.
  std::vector<SerializedSizeExpr> substituted_storage(n_stacks);
  for (std::size_t i = 0; i < n_stacks; ++i) {
    stack_headers[i].entry_size_bytes = static_cast<uint32_t>(ad_stack.allocas[i].entry_size_bytes);
    stack_headers[i].max_size_compile_time = static_cast<uint32_t>(ad_stack.allocas[i].max_size_compile_time);
    // Float allocas land on the lazy float heap, int allocas on the eager int heap. The encoding (`0` = float, `1` =
    // int) matches the SPIR-V `AdStackHeapKind` so the offline-cache bytecode survives a backend swap.
    stack_headers[i].heap_kind = (ad_stack.allocas[i].heap_kind == AdStackAllocaInfo::HeapKind::Float) ? 0u : 1u;
    if (i < ad_stack.size_exprs.size()) {
      if (!max_reducer_results.empty()) {
        substituted_storage[i] = substitute_precomputed_max_over_range(ad_stack.size_exprs[i], ad_stack.registry_id,
                                                                       static_cast<int32_t>(i), max_reducer_results);
        exprs[i] = &substituted_storage[i];
      } else {
        exprs[i] = &ad_stack.size_exprs[i];
      }
    }
  }
  // LLVM path: default-constructed emitter routes every FieldLoad through the host-fold (via `read_int`). That
  // is safe on CPU / CUDA / AMDGPU where a nested accessor kernel launch does not conflict with the enclosing
  // kernel prep.
  FieldLoadDeviceEmitter fl_emitter{};
  return encode_bytecode_common(std::move(stack_headers), exprs, prog, ctx, fl_emitter);
}

// Dense-only element-stride + place-offset computation for a `place` leaf snode. Returns false when the chain
// includes a non-dense snode (bitmasked / pointer / hash / bit-level), a shape with any axis <= 0, or a stride
// that would overflow an `int32`. Success writes `*out_elem_strides` in index order (same order as
// `SerializedSizeExprNode::indices`, each entry is the stride in element units of the leaf's primitive type,
// not bytes) and returns the byte offset of the place within its owning tree via `*out_place_byte_offset_in_root`
// so the caller can fold it into the encoded `base_psb` once and avoid a per-load add.
bool compute_dense_snode_strides(SNode *leaf, std::vector<int32_t> *out_elem_strides) {
  if (leaf == nullptr) {
    return false;
  }
  if (leaf->type != SNodeType::place) {
    return false;
  }
  if (!leaf->is_path_all_dense) {
    // A pointer / bitmasked / hash ancestor requires an on-device activation lookup the sizer shader does not
    // implement. Pushing that into the shader would mean pulling the full SNode codegen subsystem in; refuse here
    // and let the caller raise a user-visible "dense only" error.
    return false;
  }
  if (leaf->is_bit_level) {
    return false;  // quant array / bit-struct leaves need bit-packing logic we do not emit here
  }
  // Refuse multi-child dense parents. The stride computation below assumes the place leaf is the sole
  // occupant of its parent dense cell: `prod(shape[k+1..])` is a valid element-unit stride only when the
  // physical cell size equals `sizeof(leaf_dtype)`. With multiple `.place(...)` siblings under the same
  // dense ancestor (AoS layout), the real per-axis element stride is `cell_size / sizeof(leaf_dtype)`, so
  // this function's output would land on a sibling field at `i >= 1`. Extending to cell-size-aware strides
  // would require walking `SNodeDescriptor` memory-offset metadata the sizer shader does not consume today;
  // refuse and surface a clear "dense-only, single-place parent" error instead.
  for (const SNode *anc = leaf; anc != nullptr && anc->parent != nullptr; anc = anc->parent) {
    const SNode *p = anc->parent;
    if (p->type == SNodeType::dense && p->ch.size() > 1) {
      return false;
    }
  }
  const int n = leaf->num_active_indices;
  if (n < 0) {
    return false;
  }
  // Scalar fields (`qd.field(dt, shape=())`) have `num_active_indices == 0`; the pre-pass emits a `FieldLoad`
  // with an empty `indices` vector and the shader should just load `*base_psb` without any index computation.
  // Return an empty strides vector - `compute_field_load_elem_index`'s loop iterates zero times and produces
  // `elem_idx = 0`, which `psb_load_scalar` resolves to the exact place address.
  std::vector<int> shape(n, 0);
  for (int a = 0; a < n; ++a) {
    int s = leaf->shape_along_axis(a);
    if (s <= 0) {
      return false;
    }
    shape[a] = s;
  }
  out_elem_strides->resize(n);
  for (int a = 0; a < n; ++a) {
    int64_t stride = 1;
    for (int b = a + 1; b < n; ++b) {
      stride *= shape[b];
      if (stride > std::numeric_limits<int32_t>::max()) {
        return false;  // would overflow the i32 slot; refuse rather than encode a truncated stride
      }
    }
    (*out_elem_strides)[a] = static_cast<int32_t>(stride);
  }
  return true;
}

std::vector<uint8_t> encode_adstack_size_expr_device_bytecode_for_spirv(
    const spirv::TaskAttributes::AdStackSizingAttribs &ad_stack,
    Program *prog,
    LaunchContextBuilder *ctx,
    const MaxReducerResultMap &max_reducer_results) {
  const std::size_t n_stacks = ad_stack.allocas.size();
  std::vector<AdStackSizeExprDeviceStackHeader> stack_headers(n_stacks);
  std::vector<const SerializedSizeExpr *> exprs(n_stacks, nullptr);
  // Per-stack substituted trees. when the max-reducer dispatched a value for
  // a captured `MaxOverRange` node, substitute it as a `Const` BEFORE the device sizer encoder walks the tree.
  // Storage owns the substituted copies so `exprs[i]` (a pointer) stays valid through `encode_bytecode_common`.
  std::vector<SerializedSizeExpr> substituted_storage(n_stacks);
  for (std::size_t i = 0; i < n_stacks; ++i) {
    const auto &a = ad_stack.allocas[i];
    // The SPIR-V heaps are element-indexed (f32 / i32), so `entry_size_bytes` in the device header would be
    // misnamed if we set it to the byte count; the SPIR-V sizer shader interprets this field as element count
    // and scales by `2` only for the `Float` heap (to cover primal + adjoint interleaved), matching the
    // `running_offset_float += 2u * max_size` / `running_offset_int += max_size` convention the host path used
    // to perform and the main-kernel code already bakes into its offset arithmetic. Stamp `1` here so the
    // sizer's multiplication by `2` for the float heap lands exactly on `2 * max_size` and the int heap on
    // `1 * max_size`.
    stack_headers[i].entry_size_bytes = 1;
    stack_headers[i].max_size_compile_time = a.max_size_compile_time;
    stack_headers[i].heap_kind = static_cast<uint32_t>(a.heap_kind);  // Float = 0, Int = 1
    if (!max_reducer_results.empty()) {
      substituted_storage[i] = substitute_precomputed_max_over_range(a.size_expr, ad_stack.registry_id,
                                                                     static_cast<int32_t>(i), max_reducer_results);
      exprs[i] = &substituted_storage[i];
    } else {
      exprs[i] = &a.size_expr;
    }
  }
  // SPIR-V path: emit `FieldLoad` as `kFieldLoad` device nodes so the sizer shader can PSB-load the field value
  // in place. This avoids `SNodeRwAccessorsBank::Accessors::read_int`, whose nested accessor-kernel launch
  // deadlocks inside MoltenVK's descriptor-set bind path when the outer launch has already opened its command
  // buffer. The emitter resolves each snode's tree-root PSB via the program's compute device and pre-computes
  // the per-axis byte strides from the dense snode shape.
  FieldLoadDeviceEmitter fl_emitter{};
  fl_emitter.fetch = [prog](SNode *snode, uint64_t *out_base_psb, std::vector<int32_t> *out_elem_strides) -> bool {
    if (snode == nullptr || prog == nullptr) {
      return false;
    }
    if (!compute_dense_snode_strides(snode, out_elem_strides)) {
      return false;
    }
    const int tree_id = snode->get_snode_tree_id();
    DevicePtr tree_root_devptr = prog->get_snode_tree_device_ptr(tree_id);
    Device *dev = prog->get_compute_device();
    if (dev == nullptr) {
      return false;
    }
    // `get_memory_physical_pointer` returns the Vulkan `bufferDeviceAddress` / Metal equivalent for the buffer
    // that backs the snode tree's root. The place's byte offset within the tree comes from the compiled snode
    // descriptor table (`snode_descriptors[id].mem_offset_in_parent_cell` walked up to root), NOT from
    // `SNode::offset_bytes_in_parent_cell` which is a frontend-only field that stays zero on the SPIR-V path.
    // Using the wrong offset silently reads a sibling field (typically the first `qd.field` declared in the
    // program), which looks like a returning-zero bug at runtime.
    uint64_t root_psb = dev->get_memory_physical_pointer(tree_root_devptr);
    if (root_psb == 0) {
      return false;
    }
    size_t place_byte_offset = prog->get_field_in_tree_offset(tree_id, snode);
    if (std::getenv("QD_DEBUG_ADSTACK")) {
      // Pull the live value via the RwAccessors as a ground-truth check on the `PSB + place_off` pair the
      // sizer shader will consume: if the shader later reads a different i32 than `live_val`, the byte offset
      // we encoded is wrong even though `shape_along_axis` lined up and the tree dispatcher emitted a sensible
      // PSB base. `read_int` launches its own accessor kernel plus a `synchronize()`, which is safe here
      // because the encoder runs outside any in-flight main-kernel launch.
      std::vector<int> idx_zero(snode->num_active_indices, 0);
      int64_t live_val = prog->get_snode_rw_accessors_bank().get(snode).read_int(idx_zero);
      fprintf(stderr,
              "[fl.fetch] snode_id=%d type=%d dense=%d n_axes=%d tree_id=%d root_psb=0x%llx place_off=%zu live=%lld\n",
              snode->id, (int)snode->type, (int)snode->is_path_all_dense, snode->num_active_indices, tree_id,
              (unsigned long long)root_psb, place_byte_offset, (long long)live_val);
    }
    *out_base_psb = root_psb + static_cast<uint64_t>(place_byte_offset);
    return true;
  };
  // Bytecode fast path: replay the recorded host-fold reads against the live state and reuse the cached
  // bytecode if every input still matches. The full encode runs only on cache miss.
  if (prog != nullptr) {
    std::vector<uint8_t> cached;
    if (prog->adstack_cache().try_spirv_bytecode_cache_hit(prog, static_cast<const void *>(&ad_stack), ctx, cached)) {
      return cached;
    }
  }
  std::vector<AdStackCache::SizeExprReadObservation> reads;
  std::vector<uint8_t> bytecode = encode_bytecode_common(std::move(stack_headers), exprs, prog, ctx, fl_emitter,
                                                         spirv::kAdStackSizerMaxNodesPerStack, &reads);
  // Thread the max-reducer body's read observations into the bytecode cache entry so a mutation to the gating
  // ndarray invalidates the cached bytecode (the encoder walked the post-substitution tree where each captured
  // `MaxOverRange` has collapsed to a `Const`, so the body's `ExternalTensorRead` leaves are not in `reads`).
  // The observations were populated by the dispatch site via `populate_max_reducer_body_observations` and
  // recorded into the `max_reducer_cache_` alongside the dispatched value. On a subsequent launch the bytecode
  // cache replays them; gen-mismatch paths hit the dereference branch in `replay_one_observation` which returns
  // a value other than the recorded `INT64_MIN` sentinel and forces invalidation.
  if (prog != nullptr) {
    for (const auto &spec : ad_stack.max_reducer_specs) {
      const auto *spec_reads =
          prog->adstack_cache().lookup_max_reducer_reads(ad_stack.registry_id, spec.stack_id, spec.mor_node_idx);
      if (spec_reads != nullptr) {
        reads.insert(reads.end(), spec_reads->begin(), spec_reads->end());
      }
    }
    prog->adstack_cache().record_spirv_bytecode_eval(static_cast<const void *>(&ad_stack), bytecode, std::move(reads));
  }
  return bytecode;
}

void bump_writes_for_kernel_llvm(Program *prog,
                                 LaunchContextBuilder *ctx,
                                 const std::vector<OffloadedTask> &offloaded_tasks) {
  if (prog == nullptr) {
    return;
  }
  auto bump_data_ptr = [&](int arg_id) {
    ArgArrayPtrKey data_key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
    auto it = ctx->array_ptrs.find(data_key);
    if (it != ctx->array_ptrs.end() && it->second != nullptr) {
      prog->adstack_cache().bump_ndarray_data_gen(it->second);
    }
  };
  for (const auto &task : offloaded_tasks) {
    for (int snode_id : task.snode_writes) {
      prog->adstack_cache().bump_snode_write_gen(snode_id);
    }
    for (int arg_id : task.arr_writes) {
      bump_data_ptr(arg_id);
    }
    // Read-only `DevAllocType::kNone` args also need a bump: the user's host array is either H2D-blitted to a
    // temporary device buffer (CUDA / AMDGPU) or read directly (CPU), and in both cases the data pointer used as
    // the cache key is stable across launches, so a content mutation the user performed outside Quadrants's
    // tracking is invisible to the metadata cache without an explicit bump. Mirrors the SPIR-V `kone_h2d_blit`
    // rule in `bump_writes_for_kernel_spirv`.
    for (int arg_id : task.arr_reads) {
      auto type_it = ctx->device_allocation_type.find(arg_id);
      if (type_it == ctx->device_allocation_type.end() ||
          type_it->second != LaunchContextBuilder::DevAllocType::kNone) {
        continue;
      }
      bump_data_ptr(arg_id);
    }
  }
}

void bump_writes_for_kernel_llvm(Program *prog,
                                 LaunchContextBuilder *ctx,
                                 const std::vector<std::vector<int>> &snode_writes_per_task,
                                 const std::vector<std::vector<int>> &arr_writes_per_task,
                                 const std::vector<std::vector<int>> &arr_reads_per_task) {
  if (prog == nullptr) {
    return;
  }
  auto bump_data_ptr = [&](int arg_id) {
    ArgArrayPtrKey data_key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
    auto it = ctx->array_ptrs.find(data_key);
    if (it != ctx->array_ptrs.end() && it->second != nullptr) {
      prog->adstack_cache().bump_ndarray_data_gen(it->second);
    }
  };
  for (const auto &task_snodes : snode_writes_per_task) {
    for (int snode_id : task_snodes) {
      prog->adstack_cache().bump_snode_write_gen(snode_id);
    }
  }
  for (const auto &task_args : arr_writes_per_task) {
    for (int arg_id : task_args) {
      bump_data_ptr(arg_id);
    }
  }
  // Read-only `DevAllocType::kNone` args: see the comment in the CUDA / AMDGPU overload for why CPU LLVM also
  // needs the bump. Empty `arr_reads_per_task` is the legal cache-miss path (offline-cache load that did not
  // capture per-task arr_reads); skip the loop without raising.
  for (const auto &task_args : arr_reads_per_task) {
    for (int arg_id : task_args) {
      auto type_it = ctx->device_allocation_type.find(arg_id);
      if (type_it == ctx->device_allocation_type.end() ||
          type_it->second != LaunchContextBuilder::DevAllocType::kNone) {
        continue;
      }
      bump_data_ptr(arg_id);
    }
  }
}

void bump_writes_for_kernel_spirv(
    Program *prog,
    LaunchContextBuilder *ctx,
    const std::vector<spirv::TaskAttributes> &task_attribs,
    const std::vector<std::pair<std::vector<int>, irpass::ExternalPtrAccess>> &arr_access) {
  if (prog == nullptr) {
    return;
  }
  for (const auto &task : task_attribs) {
    for (int snode_id : task.snode_writes) {
      prog->adstack_cache().bump_snode_write_gen(snode_id);
    }
  }
  for (const auto &kv : arr_access) {
    const std::vector<int> &indices = kv.first;
    uint32_t access = uint32_t(kv.second);
    QD_ASSERT(indices.size() == 1);
    int arg_id = indices[0];
    bool kernel_writes = (access & uint32_t(irpass::ExternalPtrAccess::WRITE)) != 0;
    bool kone_h2d_blit = (access & uint32_t(irpass::ExternalPtrAccess::READ)) != 0 &&
                         ctx->device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone;
    if (!kernel_writes && !kone_h2d_blit) {
      continue;
    }
    ArgArrayPtrKey data_key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
    auto it = ctx->array_ptrs.find(data_key);
    if (it != ctx->array_ptrs.end()) {
      prog->adstack_cache().bump_ndarray_data_gen(it->second);
    }
  }
}

}  // namespace quadrants::lang
