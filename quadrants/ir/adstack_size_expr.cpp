#include "quadrants/ir/adstack_size_expr.h"

#include "quadrants/ir/snode.h"

namespace quadrants::lang {

namespace {

// Post-order flatten: emits every subtree before its parent so operand indices in a node always point at
// already-emitted slots. Returns the index of the node just emitted so the parent can refer to it.
int32_t flatten_recursive(const SizeExpr &expr, std::vector<SerializedSizeExprNode> &out) {
  SerializedSizeExprNode node;
  node.kind = static_cast<int32_t>(expr.kind);
  node.const_value = expr.const_value;
  node.snode_id = (expr.snode != nullptr) ? expr.snode->id : -1;
  node.indices.reserve(expr.indices.size());
  for (int64_t v : expr.indices) {
    node.indices.push_back(static_cast<int32_t>(v));
  }
  node.var_id = expr.var_id;
  node.arg_id_path = expr.arg_id_path;
  node.arg_shape_axis = expr.arg_shape_axis;
  node.operand_a = -1;
  node.operand_b = -1;
  node.body_node_idx = -1;
  if (!expr.operands.empty() && expr.operands[0]) {
    node.operand_a = flatten_recursive(*expr.operands[0], out);
  }
  if (expr.operands.size() > 1 && expr.operands[1]) {
    node.operand_b = flatten_recursive(*expr.operands[1], out);
  }
  if (expr.operands.size() > 2 && expr.operands[2]) {
    node.body_node_idx = flatten_recursive(*expr.operands[2], out);
  }
  out.push_back(std::move(node));
  return static_cast<int32_t>(out.size() - 1);
}

}  // namespace

SerializedSizeExpr SizeExpr::serialize() const {
  SerializedSizeExpr result;
  flatten_recursive(*this, result.nodes);
  return result;
}

}  // namespace quadrants::lang
