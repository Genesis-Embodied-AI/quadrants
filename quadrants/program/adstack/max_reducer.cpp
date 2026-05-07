#include "quadrants/program/adstack/max_reducer.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "quadrants/common/logging.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/adstack/device_bytecode.h"
#include "quadrants/program/adstack/eval.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

namespace {

// True iff the body subtree rooted at `node_idx` references only `Const`, `ExternalTensorRead(arg, [...])` whose every
// index slot is either a non-negative literal constant or `-(v + 1)` for some `v` in `expected_var_ids`, and `Add` /
// `Sub` / `Mul` / `Max` of those. Multi-axis ndarray reads are allowed; multiple distinct bound variables from a
// captured chain of nested `MaxOverRange`s are allowed. The encoder folds the per-axis strides via the live
// `LaunchContextBuilder` shape reads.
bool max_reducer_body_is_recognizable(const SerializedSizeExpr &expr,
                                      int32_t node_idx,
                                      const std::vector<int32_t> &expected_var_ids) {
  if (node_idx < 0 || static_cast<std::size_t>(node_idx) >= expr.nodes.size()) {
    return false;
  }
  const auto &n = expr.nodes[node_idx];
  switch (static_cast<SizeExpr::Kind>(n.kind)) {
    case SizeExpr::Kind::Const:
      return true;
    case SizeExpr::Kind::ExternalTensorRead: {
      if (n.indices.empty()) {
        return false;
      }
      // Reject `i64` / `u64` body leaves. The cache invalidation scheme stores `INT64_MIN` in the recorded observation
      // as a "stale" sentinel and revalidates on launch by re-reading the live ndarray and comparing against the saved
      // value. A 64-bit leaf can legally produce `INT64_MIN` (= 0x80000000_00000000 bit pattern) on host re-read, which
      // would make a mutated cache entry compare equal to the sentinel and false-hit. Restrict to dtypes whose value
      // range cannot overlap the sentinel; the device interpreter's `device_load_element` widens any sub-i64 integer
      // load to i64 via sign- or zero-extension, so this restriction does not lose any reverse-mode trip-count workload
      // (trip counts are uniformly i32 / u32 in practice). The per-task sizer's existing capped path absorbs anything
      // outside this dtype set.
      const auto leaf_dt = static_cast<PrimitiveTypeID>(n.const_value);
      switch (leaf_dt) {
        case PrimitiveTypeID::i8:
        case PrimitiveTypeID::i16:
        case PrimitiveTypeID::i32:
        case PrimitiveTypeID::u8:
        case PrimitiveTypeID::u16:
        case PrimitiveTypeID::u32:
          break;
        default:
          return false;
      }
      for (int32_t raw : n.indices) {
        if (raw >= 0) {
          continue;  // literal constant axis index
        }
        const int32_t var_id = -(raw + 1);
        bool found = false;
        for (int32_t want : expected_var_ids) {
          if (want == var_id) {
            found = true;
            break;
          }
        }
        if (!found) {
          return false;  // foreign bound variable not bound by the captured chain
        }
      }
      return true;
    }
    case SizeExpr::Kind::Add:
    case SizeExpr::Kind::Sub:
    case SizeExpr::Kind::Mul:
    case SizeExpr::Kind::Max:
      return max_reducer_body_is_recognizable(expr, n.operand_a, expected_var_ids) &&
             max_reducer_body_is_recognizable(expr, n.operand_b, expected_var_ids);
    case SizeExpr::Kind::ExternalTensorShape:
      // `ExternalTensorShape` indices are always literal axis numbers (no bound-var references possible), so it is
      // unconditionally closed. The encoder host-folds it to `kConst` at encode time via `evaluate_node`.
      return true;
    case SizeExpr::Kind::FieldLoad:
      // `FieldLoad` accepts both literal indices (host-folded by the encoder via `evaluate_field_load` against an empty
      // bound-var map and emitted as `kConst`) and bound-variable refs from the captured chain. The latter case lowers
      // to a `kFieldLoad` device node whose base pointer is pre-resolved on host (PSB on SPIR-V, `runtime->roots[id] +
      // place_byte_offset` on LLVM) and whose per-axis byte strides come from `compute_dense_snode_strides`. Foreign
      // bound-var refs (var_ids outside the captured chain) are rejected since the device-side scope only carries the
      // chain's axes.
      for (int32_t raw : n.indices) {
        if (raw >= 0) {
          continue;
        }
        const int32_t var_id = -(raw + 1);
        bool found = false;
        for (int32_t want : expected_var_ids) {
          if (want == var_id) {
            found = true;
            break;
          }
        }
        if (!found) {
          return false;
        }
      }
      return true;
    default:
      return false;
  }
}

// True iff the bound subtree rooted at `node_idx` evaluates to a closed-form scalar after substituting any
// `MaxOverRange` nodes already captured (`captured_mors`) as `Const`s. Allowed: `Const`, `ExternalTensorShape`, `Add` /
// `Sub` / `Mul` / `Max` of recursively-closed subtrees, and `MaxOverRange` whose node index is in `captured_mors`. On
// success appends every captured-MOR dependency this subtree references to `deps_out`.
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
    // `SerializedSizeExpr` is built post-order so deeper `MaxOverRange` nodes always have a smaller `n` than the outer
    // `MaxOverRange` that depends on them. Iterating ascending `n` visits dependencies before dependants and
    // `captured_mors` is always populated in the right order for `max_reducer_bound_is_closed`. The walk also tracks
    // which `MaxOverRange` nodes have been absorbed as the inner axis of an outer multi-axis spec; those are not
    // captured separately.
    std::unordered_set<int32_t> captured_mors;
    std::unordered_set<int32_t> absorbed_as_inner_axis;
    for (std::size_t n = 0; n < expr.nodes.size(); ++n) {
      const auto &node = expr.nodes[n];
      if (static_cast<SizeExpr::Kind>(node.kind) != SizeExpr::Kind::MaxOverRange) {
        continue;
      }
      if (absorbed_as_inner_axis.count(static_cast<int32_t>(n)) != 0) {
        continue;  // this node is the inner axis of a multi-axis spec captured at an outer node
      }
      // Greedy chain capture: starting from `n` (the outermost candidate), descend through nested `MaxOverRange` bodies
      // as long as each inner `MaxOverRange`'s `[begin, end)` is closed-form (only depends on `Const` /
      // `ExternalTensorShape` / captured-deeper-MORs). Each layer adds one axis. Stop at the first non-MaxOverRange
      // body or the first inner `MaxOverRange` whose ranges depend on a chain-bound variable (ragged iteration is not
      // supported by the rectangular cross-product dispatch).
      std::vector<int32_t> chain_node_idxs;
      std::vector<int32_t> chain_var_ids;
      std::vector<int32_t> chain_begins;
      std::vector<int32_t> chain_ends;
      std::vector<int32_t> deps;
      int32_t cur = static_cast<int32_t>(n);
      while (true) {
        const auto &cur_node = expr.nodes[cur];
        if (static_cast<SizeExpr::Kind>(cur_node.kind) != SizeExpr::Kind::MaxOverRange) {
          break;
        }
        if (!max_reducer_bound_is_closed(expr, cur_node.operand_a, captured_mors, deps)) {
          break;
        }
        if (!max_reducer_bound_is_closed(expr, cur_node.operand_b, captured_mors, deps)) {
          break;
        }
        chain_node_idxs.push_back(cur);
        chain_var_ids.push_back(cur_node.var_id);
        chain_begins.push_back(cur_node.operand_a);
        chain_ends.push_back(cur_node.operand_b);
        cur = cur_node.body_node_idx;
      }
      if (chain_node_idxs.empty()) {
        continue;  // outermost candidate failed the bound-closed check
      }
      if (!max_reducer_body_is_recognizable(expr, cur, chain_var_ids)) {
        continue;  // body grammar rejects
      }
      StaticAdStackMaxReducerSpec spec;
      spec.stack_id = static_cast<int32_t>(stack_id);
      spec.mor_node_idx = chain_node_idxs.front();
      spec.body_node_idx = cur;
      spec.axis_var_ids = std::move(chain_var_ids);
      spec.axis_begin_node_idxs = std::move(chain_begins);
      spec.axis_end_node_idxs = std::move(chain_ends);
      spec.dependent_mor_node_idxs = std::move(deps);
      specs.push_back(std::move(spec));
      captured_mors.insert(chain_node_idxs.front());
      // Mark the inner axes as absorbed so the outer loop does not re-capture them as standalone specs.
      for (std::size_t i = 1; i < chain_node_idxs.size(); ++i) {
        absorbed_as_inner_axis.insert(chain_node_idxs[i]);
      }
    }
  }
  return specs;
}

EncodedMaxReducerBody encode_max_reducer_body_bytecode(
    const SerializedSizeExpr &expr,
    int32_t body_node_idx,
    const std::vector<int32_t> &bound_var_ids,
    const std::function<int32_t(const std::vector<int32_t> &arg_id_path)> &arg_buffer_offset_resolver,
    LaunchContextBuilder *ctx,
    Program *prog,
    const FieldLoadDeviceEmitter *fl_emitter) {
  EncodedMaxReducerBody out;
  if (body_node_idx < 0 || static_cast<std::size_t>(body_node_idx) >= expr.nodes.size()) {
    return out;
  }
  // Post-order DFS to collect reachable node indices from `body_node_idx`. The recognizer grammar guarantees no
  // `kMaxOverRange` in the body subtree, so we only need to follow `operand_a` / `operand_b` (binary ops);
  // `kExternalTensorRead` / `kExternalTensorShape` / `kFieldLoad` are leaves (their operand fields are unused by the
  // device interpreter). The resulting `post_order` vector is sorted such that any node's operands precede the node
  // itself.
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
    // `kConst`, `kBoundVariable`, `kExternalTensorRead`, `kExternalTensorShape`, `kFieldLoad` are leaves (the latter
    // three are host-folded to `kConst` below; their operand fields hold metadata, not subtree pointers).
    int32_t new_idx = static_cast<int32_t>(post_order.size());
    old_to_new[idx] = new_idx;
    post_order.push_back(idx);
  };
  visit(body_node_idx);

  out.body_node_count = static_cast<uint32_t>(post_order.size());

  // Build the flat indices table for any `kExternalTensorRead` leaves. Each leaf carries `indices_count` axes; each
  // axis contributes one `(idx_raw, elem_stride)` pair. `idx_raw` mirrors the host SerializedSizeExprNode encoding
  // (`-(var_id + 1)` for bound-var refs, non-negative for constants); the encoder remaps every captured chain bound-var
  // ref to a dense device-scope slot in `[0, bound_var_ids.size())` (axis 0 = outermost MaxOverRange = device-scope
  // slot 0, axis 1 = next-inner = slot 1, ...). The dispatch site pre-populates each scope slot per iteration of the
  // cross-product. `elem_stride` is folded against the live ndarray shape, matching the per-task sizer encoder's
  // stride-emission pattern.
  std::vector<int32_t> indices_table;
  // Map each host-side bound-var id in the captured chain to its dense device-scope slot.
  auto remap_chain_var = [&](int32_t host_var_id) -> int32_t {
    for (std::size_t k = 0; k < bound_var_ids.size(); ++k) {
      if (bound_var_ids[k] == host_var_id) {
        return static_cast<int32_t>(k);
      }
    }
    return -1;
  };
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
    // assignment lands every body node in the device interpreter's switch default and returns 0 on every walk. Mirror
    // the explicit translation the per-task adstack-sizer encoder does (search `AdStackSizeExprDeviceKind::` in this TU
    // for the canonical pattern); the max-reducer body grammar narrows to the subset listed below.
    switch (kind) {
      case SizeExpr::Kind::Const:
        dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kConst);
        dst.const_value = src.const_value;
        break;
      case SizeExpr::Kind::BoundVariable: {
        // Device-side scope holds the captured chain bound variables at slots `[0, bound_var_ids.size())`,
        // outermost-first. The runtime function / SPIR-V max-reducer shader pre-populates each slot with the current
        // cross-product index before walking the body bytecode.
        dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kBoundVariable);
        const int32_t slot = remap_chain_var(src.var_id);
        if (slot < 0) {
          // Foreign bound var leaked past the recognizer; signal failure rather than silently aliasing slot 0.
          return EncodedMaxReducerBody{};
        }
        dst.var_id = slot;
        break;
      }
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
        // Indices table: emit `(idx_raw, elem_stride)` per axis. Bound-variable refs (`-(this_var_id + 1)` in the host
        // tree) become `-1` so the device-side scope's single-bound-var slot resolves them. Non-negative entries pass
        // through as compile-time literal indices. Per-axis element strides are folded against the live ndarray shape
        // read from `ctx->args_type` (the same `SHAPE_POS_IN_NDARRAY` path the per-task sizer encoder and the host
        // `evaluate_external_tensor_read` use). `ctx == nullptr` falls back to flat strides; in that mode multi-axis
        // reads are encoded as if they were single-axis, which is correct only for `indices.size() == 1` callers.
        const int32_t indices_off = static_cast<int32_t>(indices_table.size());
        const std::size_t n_axes = src.indices.size();
        std::vector<int32_t> elem_strides(n_axes, 1);
        if (n_axes > 1 && ctx != nullptr) {
          for (std::size_t k = n_axes; k-- > 0;) {
            if (k + 1 < n_axes) {
              std::vector<int> sh_idx(src.arg_id_path.begin(), src.arg_id_path.end());
              sh_idx.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
              sh_idx.push_back(static_cast<int>(k + 1));
              const int32_t sh = ctx->get_struct_arg_host<int32_t>(sh_idx);
              elem_strides[k] = elem_strides[k + 1] * sh;
            }
          }
        }
        for (std::size_t a = 0; a < n_axes; ++a) {
          int64_t raw = src.indices[a];
          int32_t emit_raw;
          if (raw >= 0) {
            emit_raw = static_cast<int32_t>(raw);
          } else {
            const int32_t host_var_id = static_cast<int32_t>(-(raw + 1));
            const int32_t slot = remap_chain_var(host_var_id);
            if (slot < 0) {
              // Foreign bound var leaked past the recognizer; analyzer invariant violation.
              return EncodedMaxReducerBody{};
            }
            // Encode dense device-scope slot as `-(slot + 1)`. The dispatch site / runtime walks the body with
            // `scope.values[slot]` pre-populated for the current cross-product iteration.
            emit_raw = -(slot + 1);
          }
          indices_table.push_back(emit_raw);
          indices_table.push_back(elem_strides[a]);
        }
        dst.indices_offset = indices_off;
        dst.indices_count = static_cast<int32_t>(n_axes);

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
      case SizeExpr::Kind::ExternalTensorShape: {
        // Closed leaf - resolve the shape value host-side at encode time and emit it as a `kConst` so the device
        // interpreter never walks `args_type`. The dispatch site re-runs the encoder per launch, so a subsequent launch
        // binding a different ndarray re-folds against the live shape; the cache invalidation rides on the
        // `ExternalShapeObs` recorded below.
        std::vector<AdStackCache::SizeExprReadObservation> read_sink;
        const int64_t v = evaluate_external_tensor_shape(src, ctx, &read_sink);
        dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kConst);
        dst.const_value = v;
        for (auto &obs : read_sink) {
          out.body_reads.push_back(std::move(obs));
        }
        break;
      }
      case SizeExpr::Kind::FieldLoad: {
        if (prog == nullptr) {
          return EncodedMaxReducerBody{};  // FieldLoad needs a live Program for snode resolution.
        }
        // Closed FieldLoad (every index slot is a literal constant) host-folds via `evaluate_field_load` to a `kConst`
        // leaf at encode time. The recorded `FieldLoadObs` carries the snode write-gen so a subsequent launch that has
        // not bumped the gen replays the cached value, mirroring the `kExternalTensorShape` host-fold path.
        bool has_bound_var_index = false;
        for (int32_t raw : src.indices) {
          if (raw < 0) {
            has_bound_var_index = true;
            break;
          }
        }
        if (!has_bound_var_index) {
          std::unordered_map<int32_t, int64_t> empty_bound;
          std::vector<AdStackCache::SizeExprReadObservation> read_sink;
          const int64_t v = evaluate_field_load(src, empty_bound, prog, &read_sink);
          dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kConst);
          dst.const_value = v;
          for (auto &obs : read_sink) {
            out.body_reads.push_back(std::move(obs));
          }
          break;
        }
        // Bound-var-indexed FieldLoad: emit a `kFieldLoad` device node that the body interpreter resolves per
        // cross-product iteration. Backend-specific base resolution: SPIR-V passes a non-empty `fl_emitter` whose
        // `fetch` returns `root_psb + place_byte_offset` (pre-baked PSB address); LLVM passes a null emitter and we
        // resolve `(snode_root_id, place_byte_offset)` directly via `prog`, which the LLVM device interpreter then
        // resolves at runtime via `runtime->roots[snode_root_id] + place_byte_offset`. Per-axis byte strides come from
        // `compute_dense_snode_strides` (units = leaf primitive type, not bytes), shared with the per-task sizer's
        // `kFieldLoad` arm.
        SNode *snode = prog->get_snode_by_id(src.snode_id);
        if (snode == nullptr) {
          return EncodedMaxReducerBody{};
        }
        auto *prim_ty = snode->dt->cast<PrimitiveType>();
        if (prim_ty == nullptr) {
          return EncodedMaxReducerBody{};
        }
        // Same dtype restriction as `kExternalTensorRead`: the cache-revalidation sentinel `INT64_MIN` must be
        // unreachable from a freshly-loaded leaf value, so reject `i64 / u64` leaves where a mutated cell could legally
        // hold the sentinel and false-hit on revalidation.
        const auto leaf_dt = prim_ty->type;
        switch (leaf_dt) {
          case PrimitiveTypeID::i8:
          case PrimitiveTypeID::i16:
          case PrimitiveTypeID::i32:
          case PrimitiveTypeID::u8:
          case PrimitiveTypeID::u16:
          case PrimitiveTypeID::u32:
            break;
          default:
            return EncodedMaxReducerBody{};
        }
        std::vector<int32_t> elem_strides;
        if (!compute_dense_snode_strides(snode, &elem_strides)) {
          return EncodedMaxReducerBody{};
        }
        if (elem_strides.size() != src.indices.size()) {
          return EncodedMaxReducerBody{};
        }
        int32_t snode_root_id = -1;
        int64_t base_or_place_off = 0;
        if (fl_emitter != nullptr && !fl_emitter->empty()) {
          uint64_t base_psb = 0;
          std::vector<int32_t> emitter_strides;
          if (!fl_emitter->fetch(snode, &base_psb, &emitter_strides)) {
            return EncodedMaxReducerBody{};
          }
          base_or_place_off = static_cast<int64_t>(base_psb);
        } else {
          // LLVM path: store `snode_root_id` in `arg_buffer_offset` (unused by FieldLoad on SPIR-V) and
          // `place_byte_offset` in `const_value`. The LLVM device interpreter reads `runtime->roots[snode_root_id] +
          // place_byte_offset` and adds the per-axis-stride-weighted element offset.
          snode_root_id = snode->get_snode_tree_id();
          base_or_place_off = static_cast<int64_t>(prog->get_field_in_tree_offset(snode_root_id, snode));
        }
        dst.kind = static_cast<int32_t>(AdStackSizeExprDeviceKind::kFieldLoad);
        dst.prim_dt = static_cast<int32_t>(leaf_dt);
        dst.arg_buffer_offset = snode_root_id;
        dst.const_value = base_or_place_off;
        const int32_t indices_off = static_cast<int32_t>(indices_table.size());
        for (std::size_t a = 0; a < src.indices.size(); ++a) {
          int32_t emit_raw;
          int64_t raw = src.indices[a];
          if (raw >= 0) {
            emit_raw = static_cast<int32_t>(raw);
          } else {
            const int32_t host_var_id = static_cast<int32_t>(-(raw + 1));
            const int32_t slot = remap_chain_var(host_var_id);
            if (slot < 0) {
              return EncodedMaxReducerBody{};
            }
            emit_raw = -(slot + 1);
          }
          indices_table.push_back(emit_raw);
          indices_table.push_back(elem_strides[a]);
        }
        dst.indices_offset = indices_off;
        dst.indices_count = static_cast<int32_t>(src.indices.size());
        // Push a `FieldLoadObs` skeleton: snode_id is the staleness key; `indices = {}` signals to
        // `replay_one_observation`'s FieldLoadObs arm that the gen counter is the sole staleness signal (the body is
        // evaluated at every cross-product point so there is no canonical scalar to re-read on a gen mismatch).
        // `populate_max_reducer_body_observations` fills in `observed_value` (sentinel) and `observed_gen` at dispatch
        // time once a live `AdStackCache` is in scope.
        AdStackCache::SizeExprReadObservation obs{};
        obs.kind = AdStackCache::SizeExprReadObservation::FieldLoadObs;
        obs.snode_id = src.snode_id;
        obs.prim_dt = static_cast<int>(leaf_dt);
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

}  // namespace quadrants::lang
