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

int64_t evaluate_node(const SerializedSizeExpr &expr,
                      int32_t node_idx,
                      const std::unordered_map<int32_t, int64_t> &bound_vars,
                      Program *prog,
                      LaunchContextBuilder *ctx);

int64_t evaluate_field_load(const SerializedSizeExprNode &node,
                            const std::unordered_map<int32_t, int64_t> &bound_vars,
                            Program *prog) {
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
  auto accessors = prog->get_snode_rw_accessors_bank().get(snode);
  return accessors.read_int(indices);
}

int64_t evaluate_external_tensor_read(const SerializedSizeExprNode &node,
                                      const std::unordered_map<int32_t, int64_t> &bound_vars,
                                      LaunchContextBuilder *ctx) {
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
  switch (prim_dt) {
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
      QD_ERROR("SerializedSizeExpr ExternalTensorRead: unsupported element type {}", node.const_value);
  }
  return 0;
}

int64_t evaluate_external_tensor_shape(const SerializedSizeExprNode &node, LaunchContextBuilder *ctx) {
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
  return static_cast<int64_t>(ctx->get_struct_arg_host<int32_t>(arg_indices));
}

int64_t evaluate_node(const SerializedSizeExpr &expr,
                      int32_t node_idx,
                      const std::unordered_map<int32_t, int64_t> &bound_vars,
                      Program *prog,
                      LaunchContextBuilder *ctx) {
  QD_ASSERT_INFO(node_idx >= 0 && static_cast<std::size_t>(node_idx) < expr.nodes.size(),
                 "SerializedSizeExpr node_idx {} out of bounds (size={})", node_idx, expr.nodes.size());
  const auto &node = expr.nodes[node_idx];
  switch (static_cast<SizeExpr::Kind>(node.kind)) {
    case SizeExpr::Kind::Const:
      return node.const_value;
    case SizeExpr::Kind::FieldLoad:
      return evaluate_field_load(node, bound_vars, prog);
    case SizeExpr::Kind::Add:
      return evaluate_node(expr, node.operand_a, bound_vars, prog, ctx) +
             evaluate_node(expr, node.operand_b, bound_vars, prog, ctx);
    case SizeExpr::Kind::Sub:
      return std::max<int64_t>(evaluate_node(expr, node.operand_a, bound_vars, prog, ctx) -
                                   evaluate_node(expr, node.operand_b, bound_vars, prog, ctx),
                               0);
    case SizeExpr::Kind::Mul:
      return evaluate_node(expr, node.operand_a, bound_vars, prog, ctx) *
             evaluate_node(expr, node.operand_b, bound_vars, prog, ctx);
    case SizeExpr::Kind::Max:
      return std::max(evaluate_node(expr, node.operand_a, bound_vars, prog, ctx),
                      evaluate_node(expr, node.operand_b, bound_vars, prog, ctx));
    case SizeExpr::Kind::MaxOverRange: {
      int64_t begin = evaluate_node(expr, node.operand_a, bound_vars, prog, ctx);
      int64_t end = evaluate_node(expr, node.operand_b, bound_vars, prog, ctx);
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
      auto extended = bound_vars;
      for (int64_t i = begin; i < end; ++i) {
        extended[node.var_id] = i;
        int64_t v = evaluate_node(expr, node.body_node_idx, extended, prog, ctx);
        if (v > result) {
          result = v;
        }
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
      return evaluate_external_tensor_shape(node, ctx);
    case SizeExpr::Kind::ExternalTensorRead:
      return evaluate_external_tensor_read(node, bound_vars, ctx);
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
                       std::vector<int32_t> &out_indices) {
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
    int64_t val = evaluate_node(src, src_idx, empty_bound, prog, ctx);
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
                                 fl_emitter, out_nodes, out_indices);
      int32_t b = encode_subtree(src, node.operand_b, contains_device_leaf, free_vars, var_id_remap, prog, ctx,
                                 fl_emitter, out_nodes, out_indices);
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
                                 fl_emitter, out_nodes, out_indices);
      int32_t b = encode_subtree(src, node.operand_b, contains_device_leaf, free_vars, var_id_remap, prog, ctx,
                                 fl_emitter, out_nodes, out_indices);
      int32_t body = encode_subtree(src, node.body_node_idx, contains_device_leaf, free_vars, var_id_remap, prog, ctx,
                                    fl_emitter, out_nodes, out_indices);
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

int64_t evaluate_adstack_size_expr(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx) {
  if (expr.nodes.empty()) {
    return -1;
  }
  std::unordered_map<int32_t, int64_t> empty_bound_vars;
  return evaluate_node(expr, static_cast<int32_t>(expr.nodes.size() - 1), empty_bound_vars, prog, ctx);
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
                                            int max_nodes_per_stack = 0) {
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
                                      var_id_remap, prog, ctx, fl_emitter, nodes, indices);
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
                                                              LaunchContextBuilder *ctx) {
  const std::size_t n_stacks = ad_stack.allocas.size();
  std::vector<AdStackSizeExprDeviceStackHeader> stack_headers(n_stacks);
  std::vector<const SerializedSizeExpr *> exprs(n_stacks, nullptr);
  for (std::size_t i = 0; i < n_stacks; ++i) {
    stack_headers[i].entry_size_bytes = static_cast<uint32_t>(ad_stack.allocas[i].entry_size_bytes);
    stack_headers[i].max_size_compile_time = static_cast<uint32_t>(ad_stack.allocas[i].max_size_compile_time);
    stack_headers[i].heap_kind = 0;  // LLVM has a single unified heap; the SPIR-V-specific bit is unused here.
    if (i < ad_stack.size_exprs.size())
      exprs[i] = &ad_stack.size_exprs[i];
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
    LaunchContextBuilder *ctx) {
  const std::size_t n_stacks = ad_stack.allocas.size();
  std::vector<AdStackSizeExprDeviceStackHeader> stack_headers(n_stacks);
  std::vector<const SerializedSizeExpr *> exprs(n_stacks, nullptr);
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
    exprs[i] = &a.size_expr;
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
  return encode_bytecode_common(std::move(stack_headers), exprs, prog, ctx, fl_emitter,
                                spirv::kAdStackSizerMaxNodesPerStack);
}

}  // namespace quadrants::lang
