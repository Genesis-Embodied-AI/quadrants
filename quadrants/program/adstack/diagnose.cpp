#include "quadrants/program/adstack/diagnose.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include "quadrants/common/logging.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/adstack/cache.h"
#include "quadrants/program/adstack/eval.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {

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

}  // namespace quadrants::lang
