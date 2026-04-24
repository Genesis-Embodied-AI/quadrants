#include "quadrants/program/adstack_size_expr_eval.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/common/logging.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"
#include "quadrants/program/snode_rw_accessors_bank.h"

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

// `contains_etr[i]` is true when subtree rooted at node `i` has at least one `ExternalTensorRead` leaf. Computed
// bottom-up; `SerializedSizeExpr` is already in post-order so every operand / body index is < i.
std::vector<bool> compute_contains_etr(const SerializedSizeExpr &expr) {
  std::vector<bool> result(expr.nodes.size(), false);
  for (std::size_t i = 0; i < expr.nodes.size(); ++i) {
    const auto &node = expr.nodes[i];
    bool hit = (static_cast<SizeExpr::Kind>(node.kind) == SizeExpr::Kind::ExternalTensorRead);
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

// Recursive top-down encoder. Each call returns the index of the emitted root in `out_nodes`. When the subtree
// at `src_idx` is closed (no free bound vars) and free of `ExternalTensorRead` leaves, the whole subtree is
// evaluated on the host via `evaluate_node` and emitted as a single `Const` device node - this is the only
// lifting mechanism, and it subsumes every `FieldLoad` / `ExternalTensorShape` leaf that the device interpreter
// does not implement. Any `FieldLoad` that survives this lifting (because its indices reference a bound var
// whose `MaxOverRange` has to stay on the device) is rejected with a hard error here, since the device
// interpreter has no SNode-access codegen.
int32_t encode_subtree(const SerializedSizeExpr &src,
                       int32_t src_idx,
                       const std::vector<bool> &contains_etr,
                       const std::vector<std::unordered_set<int32_t>> &free_vars,
                       const std::unordered_map<int32_t, int32_t> &var_id_remap,
                       Program *prog,
                       LaunchContextBuilder *ctx,
                       std::vector<AdStackSizeExprDeviceNode> &out_nodes,
                       std::vector<int32_t> &out_indices) {
  QD_ASSERT_INFO(src_idx >= 0 && static_cast<std::size_t>(src_idx) < src.nodes.size(),
                 "encode_subtree: src_idx {} out of bounds (size={})", src_idx, src.nodes.size());
  const bool subtree_needs_device = contains_etr[src_idx];
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
      int32_t a =
          encode_subtree(src, node.operand_a, contains_etr, free_vars, var_id_remap, prog, ctx, out_nodes, out_indices);
      int32_t b =
          encode_subtree(src, node.operand_b, contains_etr, free_vars, var_id_remap, prog, ctx, out_nodes, out_indices);
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
      int32_t a =
          encode_subtree(src, node.operand_a, contains_etr, free_vars, var_id_remap, prog, ctx, out_nodes, out_indices);
      int32_t b =
          encode_subtree(src, node.operand_b, contains_etr, free_vars, var_id_remap, prog, ctx, out_nodes, out_indices);
      int32_t body = encode_subtree(src, node.body_node_idx, contains_etr, free_vars, var_id_remap, prog, ctx,
                                    out_nodes, out_indices);
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
      // If we reach here the subtree is not host-substitutable (has free bound vars or sits alongside
      // `ExternalTensorRead` in the same closed context). No currently-observed user kernel emits this combination;
      // implementing on-device SNode access would mean threading `snode_rw_accessors_bank` through the device
      // interpreter, which is a separate codegen subsystem. Bail loudly so a future regression is caught rather than
      // silently misreading field memory through a stale DeviceAllocation handle.
      QD_ERROR(
          "Adstack SizeExpr FieldLoad at node {} has free bound variables or is inside an `ExternalTensorRead`"
          " context: on-device SNode access is not yet implemented by the device-side adstack SizeExpr"
          " evaluator. Rewrite the kernel so the FieldLoad-bounded trip count does not depend on a variable"
          " bound by a MaxOverRange that also wraps an ndarray read, or extend the device interpreter to walk"
          " the SNode tree.",
          src_idx);
      return -1;
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

std::vector<uint8_t> encode_adstack_size_expr_device_bytecode(const AdStackSizingInfo &ad_stack,
                                                              Program *prog,
                                                              LaunchContextBuilder *ctx) {
  const std::size_t n_stacks = ad_stack.allocas.size();

  std::vector<AdStackSizeExprDeviceStackHeader> stack_headers(n_stacks);
  std::vector<AdStackSizeExprDeviceNode> nodes;
  std::vector<int32_t> indices;
  nodes.reserve(n_stacks);
  indices.reserve(n_stacks);

  for (std::size_t i = 0; i < n_stacks; ++i) {
    auto &sh = stack_headers[i];
    sh.entry_size_bytes = static_cast<uint32_t>(ad_stack.allocas[i].entry_size_bytes);
    sh.max_size_compile_time = static_cast<uint32_t>(ad_stack.allocas[i].max_size_compile_time);
    sh._pad = 0;
    const SerializedSizeExpr *expr = (i < ad_stack.size_exprs.size()) ? &ad_stack.size_exprs[i] : nullptr;
    if (expr == nullptr || expr->nodes.empty()) {
      // No symbolic bound captured - the device interpreter will route this slot to `max_size_compile_time`.
      sh.root_node_idx = -1;
      continue;
    }
    auto contains_etr = compute_contains_etr(*expr);
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
    sh.root_node_idx = encode_subtree(*expr, static_cast<int32_t>(root_src_idx), contains_etr, free_vars, var_id_remap,
                                      prog, ctx, nodes, indices);
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

}  // namespace quadrants::lang
