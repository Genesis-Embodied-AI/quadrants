#include "quadrants/ir/precise.h"

#include "quadrants/ir/frontend_ir.h"

namespace quadrants::lang {

Expr precise(const Expr &input) {
  // Walk the subtree via a worklist (bounded stack depth for deep AST chains).
  // Tag every BinaryOpExpression and UnaryOpExpression as `precise`. Recurse through
  // TernaryOpExpression (e.g. select) without tagging the ternary itself (it's a
  // conditional move, not FP arithmetic). Stop at any other expression kind.
  std::vector<Expr> stack{input};
  while (!stack.empty()) {
    Expr cur = std::move(stack.back());
    stack.pop_back();
    if (auto bin = cur.cast<BinaryOpExpression>()) {
      bin->precise = true;
      stack.push_back(bin->lhs);
      stack.push_back(bin->rhs);
    } else if (auto un = cur.cast<UnaryOpExpression>()) {
      un->precise = true;
      stack.push_back(un->operand);
    } else if (auto tri = cur.cast<TernaryOpExpression>()) {
      stack.push_back(tri->op1);
      stack.push_back(tri->op2);
      stack.push_back(tri->op3);
    }
  }
  return input;
}

}  // namespace quadrants::lang
