#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/type_utils.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/system/profiler.h"

#include <typeindex>

namespace quadrants::lang {

// Records users from statement operands, including the operands of container statements.
class CollectUsers : public BasicStmtVisitor {
 private:
  std::unordered_map<Stmt *, std::vector<Stmt *>> *const users_;

  void collect_users(Stmt *stmt) {
    for (auto *op : stmt->get_operands()) {
      if (op != nullptr) {
        (*users_)[op].push_back(stmt);
      }
    }
  }

 public:
  using BasicStmtVisitor::visit;

  explicit CollectUsers(std::unordered_map<Stmt *, std::vector<Stmt *>> *users) : users_(users) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    collect_users(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    collect_users(stmt);
  }
};

// Whole Kernel Common Subexpression Elimination
class WholeKernelCSE : public BasicStmtVisitor {
 private:
  std::unordered_set<int> visited_;
  // Index visible statements across active scopes by operand_hash, avoiding a lookup per enclosing scope.
  // scope_inserts_ records each block's additions for removal on scope exit.
  std::unordered_map<std::size_t, std::vector<Stmt *>> visible_stmts_;
  std::vector<std::vector<std::pair<std::size_t, Stmt *>>> scope_inserts_;
  DelayedIRModifier modifier_;

  // Root of the IR tree being optimized, saved at round start so CollectUsers can traverse the whole tree.
  IRNode *root_ = nullptr;
  // Indexes users under `root_` to avoid a tree walk per replacement. Built before the first elimination or hoist.
  // Updated as operands change; queued insertions and erased statements stay alive until the round ends.
  std::unordered_map<Stmt *, std::vector<Stmt *>> users_;
  bool users_built_ = false;

  // Statements eliminated in the current round. Erased together at the end of the round, one pass per block: the
  // per-statement DelayedIRModifier::erase does a linear find + vector erase each, quadratic in the block size when
  // many root-block statements are eliminated.
  std::vector<Stmt *> to_erase_;

  // Limit pre-offload CSE to address computations and eligible integer expressions; defer other work to per-task CSE.
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

  // Allow address computations and integer expressions so equivalent pointer indices can merge. The caller also
  // requires common_statement_eliminable(), which excludes global loads and stores.
  static bool eligible_in_ptrs_only(Stmt *stmt) {
    if (is_ptr_stmt(stmt)) {
      return true;
    }
    if ((Type *)stmt->ret_type == nullptr) {
      return false;
    }
    return is_integral(stmt->ret_type.get_element_type());
  }

  bool is_done(Stmt *stmt) {
    return visited_.find(stmt->instance_id) != visited_.end();
  }

  void set_done(Stmt *stmt) {
    visited_.insert(stmt->instance_id);
  }

  static std::size_t operand_hash(const Stmt *stmt) {
    std::size_t hash_code{0};
    // Hash the statement's dynamic type; typeid(stmt) would give Stmt* for every statement.
    auto hash_type = std::hash<std::type_index>{}(std::type_index(typeid(*stmt)));
    if (stmt->is<GlobalPtrStmt>() || stmt->is<ExternalPtrStmt>() || stmt->is<LoopUniqueStmt>()) {
      // Bucket by type alone for value-based comparison. Equal addresses can have distinct index operands; hashing
      // those operands would prevent read/write pointer merging.
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
    // Reject type mismatches from hash collisions before using unchecked as<...> casts. typeid avoids the visitor and
    // string allocation used by Stmt::type().
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
        // Merge the duplicate's coverage into the retained statement.
        prev_loop_unique->covers.insert(this_loop_unique->covers.begin(), this_loop_unique->covers.end());
        return true;
      }
      return false;
    }
    return irpass::analysis::same_statements(this_stmt, prev_stmt);
  }

  void begin_round(IRNode *root) {
    root_ = root;
    users_.clear();
    users_built_ = false;
  }

  // Build before extracting hoisted statements so their operands remain indexed while insertion is pending.
  void replace_uses_and_mark_users_dirty(Stmt *duplicate_stmt, Stmt *replacement_stmt) {
    if (!users_built_) {
      CollectUsers collect(&users_);
      root_->accept(&collect);
      users_built_ = true;
    }
    auto it = users_.find(duplicate_stmt);
    if (it == users_.end()) {
      return;
    }
    std::vector<Stmt *> duplicate_stmt_users = std::move(it->second);
    users_.erase(it);
    auto &replacement_stmt_users = users_[replacement_stmt];
    for (auto *user : duplicate_stmt_users) {
      // Erased users remain alive in their block, but no longer need updating. Repeated operands can repeat users.
      if (!user->erased && user->has_operand(duplicate_stmt)) {
        // Mark the user dirty: remove its ID from the already-checked set so CSE can check its changed operands again.
        visited_.erase(user->instance_id);
        user->replace_operand_with(duplicate_stmt, replacement_stmt);
        replacement_stmt_users.push_back(user);
      }
    }
  }

  bool erase_eliminated() {
    if (to_erase_.empty()) {
      return false;
    }
    std::unordered_map<Block *, std::unordered_set<Stmt *>> by_block;
    for (auto *stmt : to_erase_) {
      by_block[stmt->parent].insert(stmt);
    }
    for (auto &[block, stmts] : by_block) {
      block->erase(std::move(stmts));
    }
    to_erase_.clear();
    return true;
  }

  void register_visible(std::size_t hash_value, Stmt *stmt) {
    visible_stmts_[hash_value].push_back(stmt);
    scope_inserts_.back().emplace_back(hash_value, stmt);
  }

  void visit(Stmt *stmt) override {
    if (!stmt->common_statement_eliminable())
      return;
    // Container statements are not CSE candidates.
    if (stmt->is_container_statement())
      return;
    // Defer non-address, non-integer computation to full per-task CSE.
    if (ptrs_only_ && !eligible_in_ptrs_only(stmt))
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
          replace_uses_and_mark_users_dirty(stmt, prev_stmt);
          to_erase_.push_back(stmt);
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
    // Remove this block's entries; outer-scope entries remain visible to sibling blocks.
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

    // Hoist matching branch prefixes and suffixes. Skip in pointers-only mode to avoid relocating other computation.
    if (!ptrs_only_ && if_stmt->true_statements && if_stmt->false_statements) {
      auto &true_clause = if_stmt->true_statements;
      auto &false_clause = if_stmt->false_statements;
      if (irpass::analysis::same_statements(true_clause->statements[0].get(), false_clause->statements[0].get())) {
        // These branch edits do not invalidate the enclosing traversal.
        replace_uses_and_mark_users_dirty(false_clause->statements[0].get(), true_clause->statements[0].get());
        auto common_stmt = true_clause->extract(0);
        modifier_.insert_before(if_stmt, std::move(common_stmt));
        false_clause->erase(0);
      }
      if (!true_clause->statements.empty() && !false_clause->statements.empty() &&
          irpass::analysis::same_statements(true_clause->statements.back().get(),
                                            false_clause->statements.back().get())) {
        // These branch edits do not invalidate the enclosing traversal.
        replace_uses_and_mark_users_dirty(false_clause->statements.back().get(), true_clause->statements.back().get());
        auto common_stmt = true_clause->extract((int)true_clause->size() - 1);
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
      eliminator.begin_round(node);
      node->accept(&eliminator);
      // Inserts (if-branch hoisting) first, then erasures, as in DelayedIRModifier::modify_ir.
      bool round_modified = eliminator.modifier_.modify_ir();
      round_modified |= eliminator.erase_eliminated();
      if (round_modified)
        modified = true;
      else
        break;
    }
    return modified;
  }
};

namespace {

// Return the tasks if root is a nonempty block containing only OffloadedStmts; otherwise return an empty list.
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
bool cse_one_task(Block *parent, OffloadedStmt *off, bool ptrs_only = false) {
  const int location = parent->locate(off);
  QD_ASSERT(location != -1);
  Block wrapper;
  wrapper.insert(parent->extract(off));
  const bool modified = WholeKernelCSE::run(&wrapper, ptrs_only);
  parent->insert(wrapper.extract(off), location);
  return modified;
}

}  // namespace

namespace irpass {
bool whole_kernel_cse(IRNode *root) {
  QD_AUTO_PROF;
  return WholeKernelCSE::run(root);
}

// Merge address computations and eligible integer expressions before offload and the first flag_access. Shared
// read/write pointers let cache_loop_invariant_global_vars cache conditional stores safely.
bool merge_global_ptrs(IRNode *root) {
  QD_AUTO_PROF;
  // Skip already-offloaded kernels; pointers are merged within individual tasks after offload.
  if (auto *block = root->cast<Block>()) {
    for (auto &stmt : block->statements) {
      if (stmt->is<OffloadedStmt>()) {
        return false;
      }
    }
  }
  return WholeKernelCSE::run(root, /*ptrs_only=*/true);
}

// Run full CSE independently on each task after offload, before flag_access and loop-invariant caching. Ndarray
// ExternalPtrStmts appear during offload; their indices must also merge to unify read/write pointers. Worker CSE runs
// too late for caching: split pointers can cause stale cached values or prevent caching entirely.
bool cse_offloaded_tasks(IRNode *root) {
  QD_AUTO_PROF;
  auto tasks = collect_offloaded_tasks(root);
  if (tasks.empty()) {
    return false;
  }
  auto *block = root->as<Block>();
  bool modified = false;
  for (auto *off : tasks) {
    if (cse_one_task(block, off)) {
      modified = true;
    }
  }
  return modified;
}

// Run full CSE when the worker block contains one offloaded task. Multi-task blocks are deferred to codegen workers,
// which optimize tasks independently and in parallel within the simplify fixpoint. Before offload, defer likewise.
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
