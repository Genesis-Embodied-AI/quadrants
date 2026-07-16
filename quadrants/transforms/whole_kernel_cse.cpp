#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/system/profiler.h"

#include <typeindex>

namespace quadrants::lang {

// Walks the same scope as `replace_all_usages_with(nullptr, old_stmt, ...)` - `old_stmt`'s parent block subtree
// plus the top-level statements of every ancestor block, which SSA dominance guarantees contains every user of
// `old_stmt`. Per visited statement: if it has `old_stmt` as an operand, erase it from `visited_` (so the next
// outer iteration of the CSE fixpoint loop re-evaluates it under the new operand pointer) and replace the operand
// with `new_stmt`. Combining the previous separate `MarkUndone::run` and `replace_all_usages_with` walks halves the
// IR-walking cost per CSE elimination.
class ReplaceAndMarkUndone : public BasicStmtVisitor {
 private:
  std::unordered_set<int> *const visited_;
  Stmt *const old_stmt_;
  Stmt *const new_stmt_;

  void mark_and_replace(Stmt *stmt) {
    if (stmt->has_operand(old_stmt_)) {
      visited_->erase(stmt->instance_id);
      stmt->replace_operand_with(old_stmt_, new_stmt_);
    }
  }

 public:
  using BasicStmtVisitor::visit;

  ReplaceAndMarkUndone(std::unordered_set<int> *visited, Stmt *old_stmt, Stmt *new_stmt)
      : visited_(visited), old_stmt_(old_stmt), new_stmt_(new_stmt) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    mark_and_replace(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    mark_and_replace(stmt);
  }

  static void run(std::unordered_set<int> *visited, Stmt *old_stmt, Stmt *new_stmt) {
    ReplaceAndMarkUndone walker(visited, old_stmt, new_stmt);
    if (old_stmt->parent == nullptr) {
      old_stmt->get_ir_root()->accept(&walker);
      return;
    }
    old_stmt->parent->accept(&walker);
    auto current_block = old_stmt->parent->parent_block();
    while (current_block != nullptr) {
      for (auto &stmt : current_block->statements) {
        if (stmt->has_operand(old_stmt)) {
          visited->erase(stmt->instance_id);
          stmt->replace_operand_with(old_stmt, new_stmt);
        }
      }
      current_block = current_block->parent_block();
    }
  }
};

// Whole Kernel Common Subexpression Elimination
class WholeKernelCSE : public BasicStmtVisitor {
 private:
  std::unordered_set<int> visited_;
  // Single hash-bucketed visibility table covering every active scope, keyed by `operand_hash`. Each `visit(Block*)`
  // pushes a fresh entry on `scope_inserts_` recording what it added so the corresponding entries can be removed
  // from `visible_stmts_` on scope exit. Equivalent in semantics to the prior per-scope `unordered_map` stack but
  // collapses the per-stmt visibility lookup from O(nesting depth) to O(1) hash + O(bucket size) bucket walk - the
  // scope-chain walk was the bottleneck on deeply-nested autodiff IR.
  std::unordered_map<std::size_t, std::vector<Stmt *>> visible_stmts_;
  std::vector<std::vector<std::pair<std::size_t, Stmt *>>> scope_inserts_;
  DelayedIRModifier modifier_;

  // When true, only address-computation statements (Global/External/MatrixPtr) are eliminated; all other statements
  // are left untouched. Used pre-offload to merge same-address read/write pointers (the cheap, load-bearing part of
  // whole-kernel CSE) without doing the expensive whole-kernel compute dedup, which is deferred to per-task CSE.
  bool ptrs_only_ = false;

 public:
  using BasicStmtVisitor::visit;

  explicit WholeKernelCSE(bool ptrs_only = false) : ptrs_only_(ptrs_only) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  static bool is_ptr_stmt(Stmt *stmt) {
    return stmt->is<GlobalPtrStmt>() || stmt->is<ExternalPtrStmt>() || stmt->is<MatrixPtrStmt>();
  }

  bool is_done(Stmt *stmt) {
    return visited_.find(stmt->instance_id) != visited_.end();
  }

  void set_done(Stmt *stmt) {
    visited_.insert(stmt->instance_id);
  }

  static std::size_t operand_hash(const Stmt *stmt) {
    std::size_t hash_code{0};
    // Use the dynamic type via `typeid(*stmt)` - `typeid(stmt)` operates on the pointer expression and returns the
    // `Stmt*` static type for every input, collapsing every statement class into the same hash component.
    auto hash_type = std::hash<std::type_index>{}(std::type_index(typeid(*stmt)));
    if (stmt->is<GlobalPtrStmt>() || stmt->is<LoopUniqueStmt>()) {
      // special cases in common_statement_eliminable()
      return hash_type;
    }
    auto op = stmt->get_operands();
    for (auto &x : op) {
      if (x == nullptr)
        continue;
      // Hash the addresses of the operand pointers.
      hash_code = (hash_code * 33) ^ (std::hash<unsigned long>{}(reinterpret_cast<unsigned long>(x)));
    }
    return hash_type ^ hash_code;
  }

  static bool common_statement_eliminable(Stmt *this_stmt, Stmt *prev_stmt) {
    // Is this_stmt eliminable given that prev_stmt appears before it and has the same type with it? `operand_hash`
    // mixes `typeid(*stmt)` into the bucket key, so any prev_stmt reaching here almost always already matches the
    // type and this check is effectively a no-op for correctly-bucketed candidates. It is still load-bearing
    // because the `as<...>` casts below are unchecked static-cast convenience wrappers - if a hash collision ever
    // lets a different-type prev_stmt through, those casts reinterpret memory and the downstream
    // `definitely_same_address` / `same_value` calls hit UB. The previous implementation called `Stmt::type()`
    // (which constructs a `StatementTypeNameVisitor`, dispatches `accept()`, and allocates a `std::string` per
    // call) inside this same hot path, which dominated the pass on large autodiff kernels; RTTI here is cheap.
    if (typeid(*this_stmt) != typeid(*prev_stmt))
      return false;
    if (this_stmt->is<GlobalPtrStmt>()) {
      auto this_ptr = this_stmt->as<GlobalPtrStmt>();
      auto prev_ptr = prev_stmt->as<GlobalPtrStmt>();
      return irpass::analysis::definitely_same_address(this_ptr, prev_ptr) &&
             (this_ptr->activate == prev_ptr->activate || prev_ptr->activate);
    }
    if (this_stmt->is<ExternalPtrStmt>()) {
      auto this_ptr = this_stmt->as<ExternalPtrStmt>();
      auto prev_ptr = prev_stmt->as<ExternalPtrStmt>();
      return irpass::analysis::definitely_same_address(this_ptr, prev_ptr);
    }
    if (this_stmt->is<LoopUniqueStmt>()) {
      auto this_loop_unique = this_stmt->as<LoopUniqueStmt>();
      auto prev_loop_unique = prev_stmt->as<LoopUniqueStmt>();
      if (irpass::analysis::same_value(this_loop_unique->input, prev_loop_unique->input)) {
        // Merge the "covers" information into prev_loop_unique.
        // Notice that this_loop_unique->covers is corrupted here.
        prev_loop_unique->covers.insert(this_loop_unique->covers.begin(), this_loop_unique->covers.end());
        return true;
      }
      return false;
    }
    return irpass::analysis::same_statements(this_stmt, prev_stmt);
  }

  void register_visible(std::size_t hash_value, Stmt *stmt) {
    visible_stmts_[hash_value].push_back(stmt);
    scope_inserts_.back().emplace_back(hash_value, stmt);
  }

  void visit(Stmt *stmt) override {
    if (!stmt->common_statement_eliminable())
      return;
    // container_statement does not need to be CSE-ed
    if (stmt->is_container_statement())
      return;
    // Pointers-only mode: skip every non-address statement (leave compute CSE to per-task CSE).
    if (ptrs_only_ && !is_ptr_stmt(stmt))
      return;
    // Generic visitor for all CSE-able statements.
    std::size_t hash_value = operand_hash(stmt);
    if (is_done(stmt)) {
      register_visible(hash_value, stmt);
      return;
    }
    auto it = visible_stmts_.find(hash_value);
    if (it != visible_stmts_.end()) {
      for (auto *prev_stmt : it->second) {
        if (common_statement_eliminable(stmt, prev_stmt)) {
          ReplaceAndMarkUndone::run(&visited_, stmt, prev_stmt);
          modifier_.erase(stmt);
          return;
        }
      }
    }
    register_visible(hash_value, stmt);
    set_done(stmt);
  }

  void visit(Block *stmt_list) override {
    scope_inserts_.emplace_back();
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    // On scope exit drop every entry inserted at this depth from the global visibility table; entries from outer
    // scopes (recorded in earlier `scope_inserts_` frames) survive so they remain visible to sibling subtrees.
    for (auto &[hash_value, stmt] : scope_inserts_.back()) {
      auto it = visible_stmts_.find(hash_value);
      if (it == visible_stmts_.end()) {
        continue;
      }
      auto &bucket = it->second;
      auto pos = std::find(bucket.begin(), bucket.end(), stmt);
      if (pos != bucket.end()) {
        bucket.erase(pos);
      }
      if (bucket.empty()) {
        visible_stmts_.erase(it);
      }
    }
    scope_inserts_.pop_back();
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements) {
      if (if_stmt->true_statements->statements.empty()) {
        if_stmt->set_true_statements(nullptr);
      }
    }

    if (if_stmt->false_statements) {
      if (if_stmt->false_statements->statements.empty()) {
        if_stmt->set_false_statements(nullptr);
      }
    }

    // Move common statements at the beginning or the end of both branches
    // outside. Skipped in pointers-only mode: we do not want to relocate arbitrary compute.
    if (!ptrs_only_ && if_stmt->true_statements && if_stmt->false_statements) {
      auto &true_clause = if_stmt->true_statements;
      auto &false_clause = if_stmt->false_statements;
      if (irpass::analysis::same_statements(true_clause->statements[0].get(), false_clause->statements[0].get())) {
        // Directly modify this because it won't invalidate any iterators.
        auto common_stmt = true_clause->extract(0);
        irpass::replace_all_usages_with(false_clause.get(), false_clause->statements[0].get(), common_stmt.get());
        modifier_.insert_before(if_stmt, std::move(common_stmt));
        false_clause->erase(0);
      }
      if (!true_clause->statements.empty() && !false_clause->statements.empty() &&
          irpass::analysis::same_statements(true_clause->statements.back().get(),
                                            false_clause->statements.back().get())) {
        // Directly modify this because it won't invalidate any iterators.
        auto common_stmt = true_clause->extract((int)true_clause->size() - 1);
        irpass::replace_all_usages_with(false_clause.get(), false_clause->statements.back().get(), common_stmt.get());
        modifier_.insert_after(if_stmt, std::move(common_stmt));
        false_clause->erase((int)false_clause->size() - 1);
      }
    }

    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements)
      if_stmt->false_statements->accept(this);
  }

  static bool run(IRNode *node, bool ptrs_only = false) {
    WholeKernelCSE eliminator(ptrs_only);
    bool modified = false;
    while (true) {
      node->accept(&eliminator);
      if (eliminator.modifier_.modify_ir())
        modified = true;
      else
        break;
    }
    return modified;
  }
};

namespace {

// Collect the top-level offloaded tasks of |root| iff |root| is an already-offloaded kernel body (a Block whose
// statements are all OffloadedStmt). Empty otherwise.
std::vector<OffloadedStmt *> collect_offloaded_tasks(IRNode *root) {
  std::vector<OffloadedStmt *> tasks;
  auto *block = root->cast<Block>();
  if (block == nullptr || block->statements.empty()) {
    return tasks;
  }
  for (auto &stmt : block->statements) {
    if (!stmt->is<OffloadedStmt>()) {
      return {};
    }
  }
  for (auto &stmt : block->statements) {
    tasks.push_back(stmt->as<OffloadedStmt>());
  }
  return tasks;
}

// Run CSE on a single offloaded task, scoped to that task alone via a throwaway wrapper block.
bool cse_one_task(Block *parent, OffloadedStmt *off) {
  const int location = parent->locate(off);
  QD_ASSERT(location != -1);
  Block wrapper;
  wrapper.insert(parent->extract(off));
  const bool modified = WholeKernelCSE::run(&wrapper);
  parent->insert(wrapper.extract(off), location);
  return modified;
}

}  // namespace

namespace irpass {
bool whole_kernel_cse(IRNode *root) {
  QD_AUTO_PROF;
  return WholeKernelCSE::run(root);
}

// Cheap whole-kernel merge of same-address pointer statements only (Global/External/MatrixPtr), leaving all compute
// alone. Run pre-offload (before the first flag_access) so a global's read and write pointers become one shared,
// activate=true pointer -- the precondition cache_loop_invariant_global_vars relies on to cache conditional/in-if
// stores. This is the load-bearing part of whole-kernel CSE; the expensive compute dedup stays per-task.
bool merge_global_ptrs(IRNode *root) {
  QD_AUTO_PROF;
  return WholeKernelCSE::run(root, /*ptrs_only=*/true);
}

// Per-offloaded-task CSE, parallelized across the codegen worker pool. The post-offload full_simplify passes
// (offload_to_executable: before_lower_access / simplify_IV / scalarize) run inside compile_task, which is enqueued
// per task to the codegen worker pool. At that point each worker's block holds exactly ONE offloaded task, so CSE
// runs here -> per task, in parallel, inside the simplify fixpoint (full quality). On the pre-split monolith
// (full_simplify in compile_to_offloads, where the block still holds every task) CSE is skipped: it is deferred to
// the parallel per-task pass above. Tasks are optimized independently, so doing all CSE in the workers is
// value-identical to doing it on the monolith -- just parallel. Before offload there are no offloaded tasks, so CSE
// is deferred likewise.
bool per_task_cse(IRNode *root) {
  QD_AUTO_PROF;
  auto tasks = collect_offloaded_tasks(root);
  if (tasks.size() != 1) {
    return false;
  }
  auto *block = root->as<Block>();
  return cse_one_task(block, tasks[0]);
}
}  // namespace irpass

}  // namespace quadrants::lang
