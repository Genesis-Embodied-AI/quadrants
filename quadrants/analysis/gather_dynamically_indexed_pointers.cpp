#include "quadrants/analysis/gather_uniquely_accessed_pointers.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/util/hash.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace quadrants::lang {

bool is_leaf_nodes_on_same_branch(SNode *snode0, SNode *snode1) {
  // Verify: place snode
  if (!snode0->is_place() || !snode1->is_place()) {
    return false;
  }

  // Check parent snode
  if (snode0->parent != snode1->parent) {
    return false;
  }

  return true;
}

class DynamicIndexingAnalyzer : public BasicStmtVisitor {
  void record_dynamic_indexed_ptr(ExternalPtrStmt *extern_ptr) {
    dynamically_indexed_ptrs_.insert(extern_ptr);
    // Find aliased ExternPtrStmt
    for (auto *other_extern_ptr : extern_ptrs_) {
      if (other_extern_ptr != extern_ptr && other_extern_ptr->base_ptr == extern_ptr->base_ptr) {
        // Aliased ExternalPtrStmt, with same base_ptr and outter index
        dynamically_indexed_ptrs_.insert(other_extern_ptr);
      }
    }
  }

  void record_dynamic_indexed_ptr(GlobalPtrStmt *global_ptr) {
    dynamically_indexed_ptrs_.insert(global_ptr);
    // Find aliased GlobalPtrStmt
    for (auto *other_global_ptr : global_ptrs_) {
      if (other_global_ptr != global_ptr && is_leaf_nodes_on_same_branch(other_global_ptr->snode, global_ptr->snode)) {
        dynamically_indexed_ptrs_.insert(other_global_ptr);
      }
    }
  }

  // A pointer that may alias (alias_analysis == uncertain) another access to the same buffer within
  // this offload must not be cached: the loop-invariant caching pass can buffer one access into a
  // local slot and defer its write-back past the loop, while the aliasing access reads/writes global
  // memory directly, so when the two addresses coincide at runtime the direct access observes a stale
  // value (issue #810). A serialized loop bypasses the parallel-loop uniqueness analysis that would
  // otherwise reject such a pair, so both accesses are flagged here to keep them out of the caching
  // pass. Accesses that are all definitely-same or all definitely-different are left cacheable, so
  // read-modify-write accumulators (same address) and disjoint accesses (different address, e.g. a
  // literal row a[0, i] and a[1, i]) remain cacheable.
  void record_may_alias_ptr(ExternalPtrStmt *extern_ptr) {
    if (!extern_ptr->base_ptr->is<ArgLoadStmt>()) {
      return;
    }
    // Only external pointers sharing the same arg_id can alias -- alias_analysis() returns 'different'
    // for distinct arg_ids -- so consult just that bucket instead of scanning every external pointer.
    // This keeps the guard identical while avoiding O(n^2) compile-time work on kernels with many
    // disjoint accesses (e.g. statically unrolled kernels).
    auto it = extern_ptrs_by_arg_.find(extern_ptr->base_ptr->as<ArgLoadStmt>()->arg_id);
    if (it == extern_ptrs_by_arg_.end()) {
      return;
    }
    for (auto *other_extern_ptr : it->second) {
      if (other_extern_ptr == extern_ptr) {
        continue;
      }
      if (irpass::analysis::alias_analysis(extern_ptr, other_extern_ptr) == AliasResult::uncertain) {
        record_dynamic_indexed_ptr(extern_ptr);
        return;
      }
    }
  }

  void record_may_alias_ptr(GlobalPtrStmt *global_ptr) {
    // Only global pointers on the same SNode can alias -- alias_analysis() returns 'different' for
    // distinct SNodes -- so consult just that bucket instead of scanning every global pointer. This
    // keeps the guard identical while avoiding O(n^2) compile-time work on kernels with many disjoint
    // accesses (e.g. statically unrolled kernels).
    auto it = global_ptrs_by_snode_.find(global_ptr->snode);
    if (it == global_ptrs_by_snode_.end()) {
      return;
    }
    for (auto *other_global_ptr : it->second) {
      if (other_global_ptr == global_ptr) {
        continue;
      }
      if (irpass::analysis::alias_analysis(global_ptr, other_global_ptr) == AliasResult::uncertain) {
        record_dynamic_indexed_ptr(global_ptr);
        return;
      }
    }
  }

 public:
  explicit DynamicIndexingAnalyzer(IRNode *node) {
  }

  void visit(GlobalPtrStmt *stmt) override {
    for (auto *index_stmt : stmt->indices) {
      if (!index_stmt->is<ConstStmt>() && !index_stmt->is<LoopIndexStmt>()) {
        record_dynamic_indexed_ptr(stmt);
      }
    }
    record_may_alias_ptr(stmt);

    if (global_ptrs_.insert(stmt).second) {
      global_ptrs_by_snode_[stmt->snode].push_back(stmt);
    }
  }

  void visit(ExternalPtrStmt *stmt) override {
    for (auto *index_stmt : stmt->indices) {
      if (!index_stmt->is<ConstStmt>() && !index_stmt->is<LoopIndexStmt>()) {
        record_dynamic_indexed_ptr(stmt);
      }
    }
    record_may_alias_ptr(stmt);

    if (extern_ptrs_.insert(stmt).second && stmt->base_ptr->is<ArgLoadStmt>()) {
      extern_ptrs_by_arg_[stmt->base_ptr->as<ArgLoadStmt>()->arg_id].push_back(stmt);
    }
  }

  void visit(MatrixPtrStmt *stmt) override {
    GlobalPtrStmt *global_ptr = nullptr;
    ExternalPtrStmt *extern_ptr = nullptr;

    if (stmt->origin->is<GlobalPtrStmt>()) {
      global_ptr = stmt->origin->as<GlobalPtrStmt>();
    } else if (stmt->origin->is<ExternalPtrStmt>()) {
      extern_ptr = stmt->origin->as<ExternalPtrStmt>();
    } else {
      return;
    }

    // Is dynamic index
    if (stmt->offset->is<ConstStmt>()) {
      return;
    }

    if (global_ptr) {
      record_dynamic_indexed_ptr(global_ptr);
    }

    if (extern_ptr) {
      record_dynamic_indexed_ptr(extern_ptr);
    }
  }

  std::unordered_set<Stmt *> get_dynamically_indexed_ptrs() {
    return dynamically_indexed_ptrs_;
  }

 private:
  using BasicStmtVisitor::visit;
  std::unordered_set<Stmt *> dynamically_indexed_ptrs_;
  std::unordered_set<GlobalPtrStmt *> global_ptrs_;
  std::unordered_set<ExternalPtrStmt *> extern_ptrs_;
  // Buckets keyed by buffer for the may-alias comparison: only pointers to the same buffer (same
  // external arg_id / same global SNode) can be 'uncertain' aliases, so record_may_alias_ptr()
  // consults just its own bucket instead of every previously seen pointer.
  std::unordered_map<std::vector<int>, std::vector<ExternalPtrStmt *>, hashing::Hasher<std::vector<int>>>
      extern_ptrs_by_arg_;
  std::unordered_map<SNode *, std::vector<GlobalPtrStmt *>> global_ptrs_by_snode_;
};

namespace irpass::analysis {

std::unordered_set<Stmt *> gather_dynamically_indexed_pointers(IRNode *root) {
  DynamicIndexingAnalyzer pass(root);

  // This pass is intended to run twice
  root->accept(&pass);
  root->accept(&pass);

  auto dynamically_indexed_ptrs = pass.get_dynamically_indexed_ptrs();
  return dynamically_indexed_ptrs;
}

}  // namespace irpass::analysis
}  // namespace quadrants::lang
