#include "quadrants/transforms/auto_diff/auto_diff_common.h"
#include "quadrants/transforms/auto_diff/validation.h"

namespace quadrants::lang {

namespace {

// Detect cross-iteration read-after-write through a `needs_grad` global field at the body of an offload-level
// range-for. `MakeAdjoint` treats each iteration of an offload-level loop as independent and only chases loop-carried
// dependencies through `AdStackAllocaJudger`, which inspects local `AllocaStmt`s - cross-iteration dependencies through
// `GlobalStoreStmt` / `GlobalLoadStmt` on a `has_adjoint()` SNode are silently dropped. A kernel like `out[i] =
// out[i-1] + x[None]` therefore returns wrong gradients with no diagnostic, even with `ad_stack_experimental_enabled`.
// Pair every same-SNode (store, load) and raise when the index Stmt*s differ. Same-Stmt indices (`out[i] = out[i] +
// x[None]`) are in-place accumulation and the per-iteration backward pass handles those correctly; pointer equality on
// the SSA index Stmts after the `pre_autodiff` simplify (`compile_to_offloads.cpp:121`) is enough to distinguish the
// two cases. Walks the body of every direct-child `RangeForStmt` of `root`; nested for-loops' bodies are skipped via
// `inside_nested_` because their cross-iter behavior is governed by the existing IB / adstack machinery, not by this
// offload-level guard.
class OffloadLevelGlobalCrossIterRAWChecker : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(GlobalStoreStmt *stmt) override {
    if (inside_nested_)
      return;
    if (auto *gp = extract_global_ptr(stmt->dest)) {
      if (gp->snode->has_adjoint())
        stores_[gp->snode].push_back(gp);
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (inside_nested_)
      return;
    if (auto *gp = extract_global_ptr(stmt->src)) {
      if (gp->snode->has_adjoint())
        loads_[gp->snode].push_back(gp);
    }
  }

  void visit(RangeForStmt *stmt) override {
    walk_nested_loop_body(stmt->body.get());
  }

  void visit(StructForStmt *stmt) override {
    walk_nested_loop_body(stmt->body.get());
  }

  void visit(MeshForStmt *stmt) override {
    walk_nested_loop_body(stmt->body.get());
  }

  void visit(WhileStmt *stmt) override {
    walk_nested_loop_body(stmt->body.get());
  }

  static void run(Block *offload_body) {
    OffloadLevelGlobalCrossIterRAWChecker checker;
    offload_body->accept(&checker);
    for (const auto &kv : checker.stores_) {
      auto *snode = kv.first;
      auto load_it = checker.loads_.find(snode);
      if (load_it == checker.loads_.end())
        continue;
      for (auto *store : kv.second) {
        // The store's index must reference the offload's `LoopIndexStmt` on at least one axis - otherwise it is
        // iteration-independent (writes the same slot every iteration) and there is no cross-iter chain to drop.
        if (!any_axis_references_loop_index(store))
          continue;
        for (auto *load : load_it->second) {
          // Skip loads whose index is iteration-independent on every axis (constants, captures from outer scopes, ...).
          // Such loads read from a slice of the SNode that the loop never writes to in this iteration, so they are not
          // a cross-iter RAW hazard.
          if (!any_axis_references_loop_index(load))
            continue;
          if (!indices_have_no_cross_iter_dependency(store, load)) {
            QD_ERROR(
                "Cross-iteration read-after-write on the `needs_grad` global field `{}` at the offload-level "
                "loop body is not supported in reverse-mode autodiff: the cross-iteration adjoint chain is "
                "not generated and the backward pass returns wrong gradients with no diagnostic. Wrap the "
                "loop in an outer for-loop in the kernel so the inner range becomes adstack-eligible, or "
                "restructure the kernel so the offload-level loop has no global field cross-iteration "
                "read-after-write. Sibling-component access on a disjoint axis like `out[i, 0] = out[i, 1]` "
                "is allowed; in-place accumulation `out[i] = out[i] + x` (same index Stmt on every axis) is "
                "also fine.",
                snode->get_node_type_name_hinted());
          }
        }
      }
    }
  }

 private:
  void walk_nested_loop_body(Block *body) {
    bool prev = inside_nested_;
    inside_nested_ = true;
    body->accept(this);
    inside_nested_ = prev;
  }

  static GlobalPtrStmt *extract_global_ptr(Stmt *s) {
    if (auto *gp = s->cast<GlobalPtrStmt>())
      return gp;
    if (auto *mp = s->cast<MatrixPtrStmt>()) {
      if (auto *gp = mp->origin->cast<GlobalPtrStmt>())
        return gp;
    }
    return nullptr;
  }

  // Decide whether two `GlobalPtrStmt`s on the same SNode index iter-independent slices of that SNode (no
  // cross-iter dependency). The rules are per-axis:
  //   - axis where neither index references `LoopIndexStmt`: constants or outer-scope captures, may differ
  //     (sibling-component access on a disjoint axis like `state[i, 0]` vs `state[i, 1]`); never a cross-iter
  //     dependency.
  //   - axis where both indices reference `LoopIndexStmt`: must be the same SSA `Stmt*` (e.g. both bare
  //     `LoopIndexStmt(i)`). If one is `LoopIndex` and the other is `LoopIndex - 1` or similar, the load reads
  //     a slot the store wrote in a different iteration: this is cross-iter.
  //   - axis where exactly one references `LoopIndexStmt`: load and store have iter-dependent vs iter-
  //     independent indexing on the same axis; conservatively flag as cross-iter, since whether the constant
  //     index aliases an iteration of the loop is not provable here.
  static bool indices_have_no_cross_iter_dependency(GlobalPtrStmt *a, GlobalPtrStmt *b) {
    if (a->indices.size() != b->indices.size())
      return false;
    for (std::size_t i = 0; i < a->indices.size(); ++i) {
      bool a_refs = references_loop_index(a->indices[i]);
      bool b_refs = references_loop_index(b->indices[i]);
      if (a_refs != b_refs)
        return false;
      if (a_refs && b_refs && a->indices[i] != b->indices[i])
        return false;
    }
    return true;
  }

  // Walk an index `Stmt*` and decide whether it (transitively) references a `LoopIndexStmt` of any enclosing loop.
  // Recurses through linear arithmetic ops because an index like `i - 1` lowers to `BinaryOpStmt(LoopIndexStmt(i), Sub,
  // ConstStmt(1))` and we want to recognise the `LoopIndexStmt` underneath. Conservatively returns false for anything
  // else (constants, ndarray loads from outer scope, ...) - those express iteration-independent indexing that the
  // cross-iter guard intentionally exempts.
  static bool references_loop_index(Stmt *s) {
    if (s == nullptr)
      return false;
    if (s->is<LoopIndexStmt>())
      return true;
    if (auto *bo = s->cast<BinaryOpStmt>())
      return references_loop_index(bo->lhs) || references_loop_index(bo->rhs);
    if (auto *uo = s->cast<UnaryOpStmt>())
      return references_loop_index(uo->operand);
    if (auto *to = s->cast<TernaryOpStmt>())
      return references_loop_index(to->op1) || references_loop_index(to->op2) || references_loop_index(to->op3);
    return false;
  }

  static bool any_axis_references_loop_index(GlobalPtrStmt *gp) {
    for (auto *idx : gp->indices) {
      if (references_loop_index(idx))
        return true;
    }
    return false;
  }

  std::unordered_map<SNode *, std::vector<GlobalPtrStmt *>> stores_;
  std::unordered_map<SNode *, std::vector<GlobalPtrStmt *>> loads_;
  bool inside_nested_ = false;
};

class GloablDataAccessRuleChecker : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(GlobalLoadStmt *stmt) override {
    GlobalPtrStmt *src = nullptr;
    if (stmt->src->is<GlobalPtrStmt>()) {
      src = stmt->src->as<GlobalPtrStmt>();
    } else {
      QD_ASSERT(stmt->src->is<MatrixPtrStmt>());
      src = stmt->src->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }
    auto snode = src->snode;
    if (!snode->has_adjoint_checkbit()) {
      return;
    }
    QD_ASSERT(snode->get_adjoint_checkbit() != nullptr);
    snode = snode->get_adjoint_checkbit();
    auto global_ptr = stmt->insert_after_me(Stmt::make<GlobalPtrStmt>(snode, src->indices));
    auto dtype = global_ptr->ret_type.ptr_removed();

    auto one = insert_const(dtype, global_ptr, 1, false /*insert_before_me*/);
    one->insert_after_me(Stmt::make<GlobalStoreStmt>(global_ptr, one));
  }

  void visit_gloabl_store_stmt_and_atomic_add(Stmt *stmt, GlobalPtrStmt *dest) {
    auto snode = dest->snode;
    if (!snode->has_adjoint_checkbit()) {
      return;
    }
    QD_ASSERT(snode->get_adjoint_checkbit() != nullptr);
    snode = snode->get_adjoint_checkbit();
    auto global_ptr = stmt->insert_before_me(Stmt::make<GlobalPtrStmt>(snode, dest->indices));
    auto global_load = stmt->insert_before_me(Stmt::make<GlobalLoadStmt>(global_ptr));
    auto dtype = global_ptr->ret_type.ptr_removed();
    auto zero = insert_const(dtype, stmt, 0, /*insert_before_me=*/true);
    auto check_equal = stmt->insert_before_me(Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_eq, global_load, zero));
    std::string msg = fmt::format(
        "(kernel={}) Breaks the global data access rule. Snode {} is "
        "overwritten unexpectedly.",
        kernel_name_, dest->snode->get_node_type_name());
    msg += "\n" + stmt->get_tb();

    stmt->insert_before_me(Stmt::make<AssertStmt>(check_equal, msg, std::vector<Stmt *>()));
  }

  void visit(GlobalStoreStmt *stmt) override {
    GlobalPtrStmt *dest = nullptr;
    if (stmt->dest->is<GlobalPtrStmt>()) {
      dest = stmt->dest->as<GlobalPtrStmt>();
    } else {
      QD_ASSERT(stmt->dest->is<MatrixPtrStmt>());
      dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }
    visit_gloabl_store_stmt_and_atomic_add(stmt, dest);
  }

  void visit(AtomicOpStmt *stmt) override {
    GlobalPtrStmt *dest = nullptr;
    if (stmt->dest->is<GlobalPtrStmt>()) {
      dest = stmt->dest->as<GlobalPtrStmt>();
    } else {
      QD_ASSERT(stmt->dest->is<MatrixPtrStmt>());
      dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }
    visit_gloabl_store_stmt_and_atomic_add(stmt, dest);
  }

  static void run(IRNode *root, const std::string &kernel_name) {
    GloablDataAccessRuleChecker checker;
    checker.kernel_name_ = kernel_name;
    root->accept(&checker);
  }

 private:
  std::string kernel_name_;
};

}  // namespace

void offload_level_global_cross_iter_raw_check(Block *offload_body) {
  OffloadLevelGlobalCrossIterRAWChecker::run(offload_body);
}

void global_data_access_rule_check(IRNode *root, const std::string &kernel_name) {
  GloablDataAccessRuleChecker::run(root, kernel_name);
}

namespace irpass {

void differentiation_validation_check(IRNode *root, const CompileConfig &config, const std::string &kernel_name) {
  global_data_access_rule_check(root, kernel_name);
}

}  // namespace irpass

}  // namespace quadrants::lang
