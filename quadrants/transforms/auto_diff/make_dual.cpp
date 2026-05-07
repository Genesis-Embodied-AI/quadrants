#include "quadrants/transforms/auto_diff/auto_diff_common.h"
#include "quadrants/transforms/auto_diff/make_dual.h"

namespace quadrants::lang {

namespace {

// Forward mode autodiff
class MakeDual : public ADTransform {
 public:
  using ADTransform::visit;
  Stmt *current_stmt;
  Block *current_block;
  Block *alloca_block;
  std::map<Stmt *, Stmt *> dual_stmt;

  explicit MakeDual(Block *block) {
    current_stmt = nullptr;
    alloca_block = block;
    current_block = block;
  }

  static void run(Block *block) {
    auto p = MakeDual(block);
    block->accept(&p);
  }

  Stmt *insert_grad_stmt(std::unique_ptr<Stmt> &&stmt) override {
    auto ptr = stmt.get();
    current_stmt = current_stmt->insert_after_me(std::move(stmt));
    return ptr;
  }

  void visit(Block *block) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : block->statements) {
      statements.push_back(stmt.get());
    }
    for (auto stmt : statements) {
      current_stmt = stmt;
      stmt->accept(this);
    }
  }

  // Accumulate [value] to the dual of [primal]
  void accumulate(Stmt *primal, Stmt *value) {
    auto alloca_ = dual(primal);
    if (!alloca_ || alloca_->is<ConstStmt>())
      return;  // primal may be int variable

    QD_ASSERT(alloca_->is<AllocaStmt>());
    auto alloca = alloca_->as<AllocaStmt>();
    auto local_load = insert<LocalLoadStmt>(alloca);
    insert<LocalStoreStmt>(alloca, add(local_load, value));
  }

  Stmt *dual(Stmt *stmt) {
    auto dual_type = stmt->ret_type.ptr_removed();
    if (!is_real(dual_type.get_element_type()) || stmt->is<ConstStmt>()) {
      return constant(0);
    }
    if (dual_stmt.find(stmt) == dual_stmt.end()) {
      // normal SSA cases

      // Create the alloca. Using the statement's own `ret_type` tends to fit better than the kernel-wide
      // `get_current_program().config.gradient_dt` default.
      auto alloca = Stmt::make<AllocaStmt>(dual_type);
      dual_stmt[stmt] = alloca.get();

      // TODO: check whether there are any edge cases for the alloca_block
      alloca_block->insert(std::move(alloca), 0);
    }
    return dual_stmt[stmt];
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->op_type == UnaryOpType::neg) {
      accumulate(stmt, negate(dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::abs) {
      accumulate(stmt, mul(sgn(stmt->operand), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::sin) {
      accumulate(stmt, mul(cos(stmt->operand), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::cos) {
      accumulate(stmt, negate(mul(sin(stmt->operand), dual(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::tan) {
      // d/dx tan(x) = 1 + tan(x)^2. Forward mode executes in primal order, so `stmt` is the current-iteration tan value
      // - no BackupSSA stale-value concern; reusing it is per-iteration correct.
      accumulate(stmt, mul(add(constant(1), sqr(stmt)), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::tanh) {
      accumulate(stmt, mul(sub(constant(1), sqr(stmt)), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::asin) {
      accumulate(stmt, mul(div(constant(1), sqrt(sub(constant(1), sqr(stmt->operand)))), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::acos) {
      accumulate(stmt, mul(negate(div(constant(1), sqrt(sub(constant(1), sqr(stmt->operand))))), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::exp) {
      accumulate(stmt, mul(stmt, dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::log) {
      accumulate(stmt, div(dual(stmt->operand), stmt->operand));
    } else if (stmt->op_type == UnaryOpType::sqrt) {
      accumulate(stmt, mul(div(constant(0.5f), sqrt(stmt->operand)), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::rsqrt) {
      accumulate(stmt, mul(mul(constant(-0.5f), pow(rsqrt(stmt->operand), constant(3))), dual(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      if (is_real(stmt->cast_type.get_element_type()) && is_real(stmt->operand->ret_type.get_element_type())) {
        accumulate(stmt, dual(stmt->operand));
      }
    } else if (stmt->op_type == UnaryOpType::logic_not) {
      // do nothing
    } else {
      QD_P(unary_op_type_name(stmt->op_type));
      QD_NOT_IMPLEMENTED
    }
  }

  void visit(BinaryOpStmt *bin) override {
    if (bin->op_type == BinaryOpType::add) {
      accumulate(bin, dual(bin->lhs));
      accumulate(bin, dual(bin->rhs));
    } else if (bin->op_type == BinaryOpType::sub) {
      accumulate(bin, dual(bin->lhs));
      accumulate(bin, negate(dual(bin->rhs)));
    } else if (bin->op_type == BinaryOpType::mul) {
      // d (x * y) = y * dx + x * dy
      accumulate(bin, mul(bin->lhs, dual(bin->rhs)));
      accumulate(bin, mul(bin->rhs, dual(bin->lhs)));
    } else if (bin->op_type == BinaryOpType::mod) {
      // Do nothing
    } else if (bin->op_type == BinaryOpType::div) {
      accumulate(bin, div(dual(bin->lhs), bin->rhs));
      accumulate(bin, negate(div(mul(dual(bin->rhs), bin->lhs), mul(bin->rhs, bin->rhs))));
    } else if (bin->op_type == BinaryOpType::atan2) {
      auto numerator = add(sqr(bin->lhs), sqr(bin->rhs));
      accumulate(bin, div(mul(bin->rhs, dual(bin->lhs)), numerator));
      accumulate(bin, negate(div(mul(bin->lhs, dual(bin->rhs)), numerator)));
    } else if (bin->op_type == BinaryOpType::pow) {
      // d (x ^ y) = x ^ (y-1) * (y * dx + log(x) * x * dy)
      auto common_coeff = pow(bin->lhs, sub(bin->rhs, constant(1)));  // x ^ (y-1)
      accumulate(bin, mul(dual(bin->lhs), mul(bin->rhs, common_coeff)));
      accumulate(bin, mul(dual(bin->rhs), mul(log(bin->lhs), mul(bin->lhs, common_coeff))));
    } else if (bin->op_type == BinaryOpType::min || bin->op_type == BinaryOpType::max) {
      auto cmp = bin->op_type == BinaryOpType::min ? cmp_lt(bin->lhs, bin->rhs) : cmp_lt(bin->rhs, bin->lhs);
      auto zero = insert_const_for_grad(bin->ret_type, bin, 0);
      accumulate(bin, sel(cmp, dual(bin->lhs), zero));
      accumulate(bin, sel(cmp, zero, dual(bin->rhs)));
    } else if (bin->op_type == BinaryOpType::floordiv) {
      // do nothing
    } else if (is_comparison(bin->op_type) || is_bit_op(bin->op_type)) {
      // do nothing
    } else {
      QD_WARN("gradient of binary op {}\n{}", binary_op_type_name(bin->op_type), bin->get_tb());
      QD_NOT_IMPLEMENTED
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    QD_ASSERT(stmt->op_type == TernaryOpType::select);
    auto zero = insert_const_for_grad(stmt->ret_type, stmt, 0);
    accumulate(stmt, insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1, load(dual(stmt->op2)), zero));
    accumulate(stmt, insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1, zero, load(dual(stmt->op3))));
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements) {
      std::vector<Stmt *> true_statements;
      for (auto &stmt : if_stmt->true_statements->statements) {
        true_statements.push_back(stmt.get());
      }

      for (auto stmt : true_statements) {
        current_stmt = stmt;
        stmt->accept(this);
      }
    }
    if (if_stmt->false_statements) {
      std::vector<Stmt *> false_statements;
      for (auto &stmt : if_stmt->false_statements->statements) {
        false_statements.push_back(stmt.get());
      }

      for (auto stmt : false_statements) {
        current_stmt = stmt;
        stmt->accept(this);
      }
    }
  }

  void visit(RangeForStmt *for_stmt) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : for_stmt->body->statements) {
      statements.push_back(stmt.get());
    }
    auto previous_alloca_block = alloca_block;
    alloca_block = for_stmt->body.get();
    for (auto stmt : statements) {
      current_stmt = stmt;
      stmt->accept(this);
    }
    alloca_block = previous_alloca_block;
  }

  void visit(StructForStmt *for_stmt) override {
    // Save/restore mirrors visit(RangeForStmt) above and MakeAdjoint::visit(StructForStmt). An enclosing compound
    // visitor that resumes iterating its body after this StructForStmt needs alloca_block to still point at its own
    // block, not the sparse-for body, so new dual allocas land where the enclosing reverse code can reach them.
    auto previous_alloca_block = alloca_block;
    alloca_block = for_stmt->body.get();
    for_stmt->body->accept(this);
    alloca_block = previous_alloca_block;
  }

  void visit(LocalLoadStmt *stmt) override {
    // QD_ASSERT(!needs_grad(stmt->ret_type));
    accumulate(stmt, dual(stmt->src));
  }

  void visit(LocalStoreStmt *stmt) override {
    // Clear the dual of the dest before local store, because LocalStoreStmt overwrites the dest. If the alloca serves
    // as the dest of multiple LocalStoreStmts, only the last one counts and the prior accumulated dual must be
    // discarded.
    auto dtype = stmt->dest->ret_type.ptr_removed();
    if (is_real(dtype.get_element_type())) {
      auto zero = insert_const_for_grad(dtype, stmt, 0);
      insert<LocalStoreStmt>(dual(stmt->dest), zero);
    }

    accumulate(stmt->dest, dual(stmt->val));
  }

  void visit(GlobalLoadStmt *stmt) override {
    // issue global store to dual
    GlobalPtrStmt *src = nullptr;
    bool is_ptr_offset = false;
    if (stmt->src->is<MatrixPtrStmt>()) {
      is_ptr_offset = true;
      src = stmt->src->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    } else {
      src = stmt->src->as<GlobalPtrStmt>();
    }
    auto snode = src->snode;
    if (!snode->has_dual()) {
      // No dual SNode. Do nothing
      return;
    }
    if (gradients_stopped(stmt, snode)) {
      // gradients stopped, do nothing.
      return;
    }
    QD_ASSERT(snode->get_dual() != nullptr);
    snode = snode->get_dual();
    auto dual_ptr = insert<GlobalPtrStmt>(snode, src->indices);
    dual_ptr->ret_type = src->ret_type;
    if (is_ptr_offset) {
      dual_ptr = insert<MatrixPtrStmt>(dual_ptr, stmt->src->as<MatrixPtrStmt>()->offset);
    }
    accumulate(stmt, insert<GlobalLoadStmt>(dual_ptr));
  }

  void visit(GlobalStoreStmt *stmt) override {
    GlobalPtrStmt *dest = nullptr;
    bool is_ptr_offset = false;
    if (stmt->dest->is<MatrixPtrStmt>()) {
      is_ptr_offset = true;
      dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    } else {
      dest = stmt->dest->as<GlobalPtrStmt>();
    }
    auto snode = dest->snode;
    if (!snode->has_dual()) {
      // no gradient (likely integer types)
      return;
    }
    QD_ASSERT(snode->get_dual() != nullptr);
    snode = snode->get_dual();
    auto dual_ptr = insert<GlobalPtrStmt>(snode, dest->indices);
    dual_ptr->ret_type = dest->ret_type;
    if (is_ptr_offset) {
      dual_ptr = insert<MatrixPtrStmt>(dual_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);
    }
    insert<AtomicOpStmt>(AtomicOpType::add, dual_ptr, load(dual(stmt->val)));
  }

  void visit(AtomicOpStmt *stmt) override {
    GlobalPtrStmt *dest = nullptr;
    bool is_ptr_offset = false;
    if (stmt->dest->is<MatrixPtrStmt>()) {
      is_ptr_offset = true;
      dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    } else {
      dest = stmt->dest->as<GlobalPtrStmt>();
    }
    auto snode = dest->snode;
    if (!snode->has_dual()) {
      // no gradient (likely integer types)
      return;
    }
    QD_ASSERT(snode->get_dual() != nullptr);
    snode = snode->get_dual();
    auto dual_ptr = insert<GlobalPtrStmt>(snode, dest->indices);
    dual_ptr->ret_type = dest->ret_type;
    if (is_ptr_offset) {
      dual_ptr = insert<MatrixPtrStmt>(dual_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);
    }
    insert<AtomicOpStmt>(AtomicOpType::add, dual_ptr, load(dual(stmt->val)));
  }

  void visit(MatrixInitStmt *stmt) override {
    std::vector<Stmt *> duals;
    for (auto &s : stmt->values) {
      duals.push_back(dual(s));
    }
    auto dual_stmt = insert<MatrixInitStmt>(duals);
    dual_stmt->ret_type = stmt->ret_type;

    accumulate(stmt, dual_stmt);
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<GlobalPtrStmt>()) {
      // Handled in GlobalLoadStmt and GlobalStoreStmt
      return;
    }

    auto origin_dual = dual(stmt->origin);
    auto origin_dual_ptr = insert<MatrixPtrStmt>(origin_dual, stmt->offset);
    origin_dual_ptr->ret_type = stmt->ret_type;

    accumulate(stmt, origin_dual_ptr);
  }
};

}  // namespace

void make_dual(Block *block) {
  MakeDual::run(block);
}

}  // namespace quadrants::lang
