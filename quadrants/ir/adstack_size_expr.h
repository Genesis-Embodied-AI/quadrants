#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "quadrants/common/serialization.h"

namespace quadrants::lang {

class SNode;
class Stmt;

// Flat, offline-cache-serialisable representation of a `SizeExpr` tree. Each node references at most two operands
// by index into the same vector; the tree root is always the last element so the evaluator can walk the vector
// linearly in post-order. `snode_id` is the owning `SNode`'s global id (the same handle `offline_cache_util` uses
// for all snode serialisation) so the host-side evaluator can resolve it back to the live `SNode *` via
// `Program::get_snode_by_id(...)` after a cache hit. `indices` are stored as `int32`; non-negative entries are
// fixed constant indices, negative entries `-(var_id + 1)` reference a currently-bound loop variable captured by
// an enclosing `MaxOverRange` node.
struct SerializedSizeExprNode {
  // Mirror of `SizeExpr::Kind`. Encoded as `int32` for stable serialization across compiler versions.
  int32_t kind{0};
  int64_t const_value{0};
  int32_t snode_id{-1};
  std::vector<int32_t> indices;
  // Indices into the owning `std::vector<SerializedSizeExprNode>`. `-1` when the operand slot is unused. For
  // `MaxOverRange`, `operand_a` and `operand_b` hold the inclusive-begin and exclusive-end nodes of the iterated
  // range and `body_node_idx` points at the inner expression to maximise over the range.
  int32_t operand_a{-1};
  int32_t operand_b{-1};
  // Only used by `MaxOverRange`: the loop-variable identifier that is bound while iterating the range, and the
  // index of the body node. Unused (left at -1) for every other kind.
  int32_t var_id{-1};
  int32_t body_node_idx{-1};
  // Only used by `ExternalTensorShape`: argument path into the kernel's launch context plus the shape axis.
  // `LaunchContextBuilder::get_struct_arg<i64>` reads the live shape at dispatch time.
  std::vector<int32_t> arg_id_path;
  int32_t arg_shape_axis{-1};
  QD_IO_DEF(kind,
            const_value,
            snode_id,
            indices,
            operand_a,
            operand_b,
            var_id,
            body_node_idx,
            arg_id_path,
            arg_shape_axis);
};

// One flat tree per `AdStackAllocaStmt`; the tree root is `nodes.back()`. An empty vector encodes a "no
// SizeExpr captured" marker (Bellman-Ford-only resolution of a truly constant stack depth, where the compile-time
// `AdStackAllocaInfo::max_size_compile_time` suffices on its own).
struct SerializedSizeExpr {
  std::vector<SerializedSizeExprNode> nodes;
  QD_IO_DEF(nodes);
};

// Compile-time captured symbolic expression for an `AdStackAllocaStmt`'s max_size. The
// determine_ad_stack_size pre-pass builds one of these per bounded adstack by walking the enclosing
// loop structure; the host evaluates it against the live field state right before each kernel
// launch to size the per-thread adstack heap stride. The allowed shapes are intentionally minimal:
// integer constants, scalar i32/i64 field loads at constant indices, and add / sub / max binary
// ops. Anything outside this shape triggers a compile error in the pre-pass - there is no silent
// compile-time size fallback here anymore.
class SizeExpr {
 public:
  enum class Kind {
    Const,                // `const_value`
    FieldLoad,            // scalar load of `snode[indices...]`; indices can be const or `-(var_id + 1)` loop var
    Add,                  // `operands[0] + operands[1]`
    Sub,                  // `operands[0] - operands[1]`
    Mul,                  // `operands[0] * operands[1]` (non-const * non-const trip-count product)
    Max,                  // `max(operands[0], operands[1])`
    MaxOverRange,         // `max_{var = operands[0] .. operands[1]-1} operands[2]`; `var_id` names the iterated
    BoundVariable,        // resolves to the current value of loop variable `var_id` in an enclosing MaxOverRange
    ExternalTensorShape,  // ndarray-argument shape along `arg_shape_axis`, resolved from the launch context
    ExternalTensorRead    // ndarray-argument scalar read: `arg[indices...]`; indices share `FieldLoad` encoding
  };

  Kind kind{Kind::Const};
  int64_t const_value{0};
  SNode *snode{nullptr};
  std::vector<int64_t> indices;
  int32_t var_id{-1};
  std::vector<int32_t> arg_id_path;
  int32_t arg_shape_axis{-1};
  std::vector<std::unique_ptr<SizeExpr>> operands;
  // Only used by `MaxOverRange`, and only while `determine_ad_stack_size` is still building trees: the source IR
  // loop whose index the range wrapper iterates, or `nullptr` for the conservative whole-axis fallback wraps that
  // bound an index shape the pre-pass could not chase to a `LoopIndexStmt`. Two wrappers pair the same iteration
  // only when they come from the same source loop, so `expr_sub`'s fusion keys on this pointer; alpha-equal ranges
  // alone also match two independent fallback wraps, whose fused body would pair unrelated indices. Intentionally
  // absent from `SerializedSizeExprNode`: the pointer is dead weight once the pass returns (fusion never runs on
  // deserialized trees) and would dangle across the offline-cache round trip anyway.
  Stmt *source_loop{nullptr};

  SizeExpr() = default;
  SizeExpr(const SizeExpr &) = delete;
  SizeExpr &operator=(const SizeExpr &) = delete;
  SizeExpr(SizeExpr &&) = default;
  SizeExpr &operator=(SizeExpr &&) = default;

  static std::unique_ptr<SizeExpr> make_const(int64_t v) {
    auto e = std::make_unique<SizeExpr>();
    e->kind = Kind::Const;
    e->const_value = v;
    return e;
  }

  static std::unique_ptr<SizeExpr> make_field_load(SNode *snode, std::vector<int64_t> indices) {
    auto e = std::make_unique<SizeExpr>();
    e->kind = Kind::FieldLoad;
    e->snode = snode;
    e->indices = std::move(indices);
    return e;
  }

  static std::unique_ptr<SizeExpr> make_binary(Kind kind,
                                               std::unique_ptr<SizeExpr> lhs,
                                               std::unique_ptr<SizeExpr> rhs) {
    auto e = std::make_unique<SizeExpr>();
    e->kind = kind;
    e->operands.reserve(2);
    e->operands.push_back(std::move(lhs));
    e->operands.push_back(std::move(rhs));
    return e;
  }

  static std::unique_ptr<SizeExpr> make_bound_variable(int32_t var_id) {
    auto e = std::make_unique<SizeExpr>();
    e->kind = Kind::BoundVariable;
    e->var_id = var_id;
    return e;
  }

  static std::unique_ptr<SizeExpr> make_max_over_range(int32_t var_id,
                                                       std::unique_ptr<SizeExpr> begin,
                                                       std::unique_ptr<SizeExpr> end,
                                                       std::unique_ptr<SizeExpr> body,
                                                       Stmt *source_loop = nullptr) {
    auto e = std::make_unique<SizeExpr>();
    e->kind = Kind::MaxOverRange;
    e->var_id = var_id;
    e->operands.reserve(3);
    e->operands.push_back(std::move(begin));
    e->operands.push_back(std::move(end));
    e->operands.push_back(std::move(body));
    e->source_loop = source_loop;
    return e;
  }

  static std::unique_ptr<SizeExpr> make_external_tensor_shape(std::vector<int32_t> arg_id_path, int32_t axis) {
    auto e = std::make_unique<SizeExpr>();
    e->kind = Kind::ExternalTensorShape;
    e->arg_id_path = std::move(arg_id_path);
    e->arg_shape_axis = axis;
    return e;
  }

  // `arg_element_dt` is serialised as `const_value` and carries the element-type enum (PrimitiveTypeID cast to
  // int) so the evaluator can decode the raw ndarray bytes at the computed offset.
  static std::unique_ptr<SizeExpr> make_external_tensor_read(std::vector<int32_t> arg_id_path,
                                                             std::vector<int64_t> indices,
                                                             int64_t element_prim_dt) {
    auto e = std::make_unique<SizeExpr>();
    e->kind = Kind::ExternalTensorRead;
    e->arg_id_path = std::move(arg_id_path);
    e->indices = std::move(indices);
    e->const_value = element_prim_dt;
    return e;
  }

  std::unique_ptr<SizeExpr> clone() const {
    auto e = std::make_unique<SizeExpr>();
    e->kind = kind;
    e->const_value = const_value;
    e->snode = snode;
    e->indices = indices;
    e->var_id = var_id;
    e->arg_id_path = arg_id_path;
    e->arg_shape_axis = arg_shape_axis;
    e->source_loop = source_loop;
    e->operands.reserve(operands.size());
    for (const auto &child : operands) {
      e->operands.push_back(child ? child->clone() : nullptr);
    }
    return e;
  }

  // Flatten this tree into the post-order `SerializedSizeExpr` representation that survives the offline cache.
  // The root lands at `nodes.back()`; callers that want to walk it evaluate left-to-right because each node's
  // operand indices point to strictly earlier slots. `SNode *` is serialised as `SNode::id`.
  SerializedSizeExpr serialize() const;
};

}  // namespace quadrants::lang
