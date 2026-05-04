#include "quadrants/transforms/auto_diff/auto_diff_common.h"
#include "quadrants/transforms/auto_diff/post_adjoint_cleanup.h"

namespace quadrants::lang {

namespace {

// ============================================================================
// BackupSSA: spill cross-block SSA operands the reverse-mode clones reference.
//
// MakeAdjoint clones forward stmts into the reverse scope but shares operand
// pointers with the forward graph; when those operands live inside the
// forward body (e.g. an inner-for's `begin`/`end`), the reverse clone's
// operand no longer dominates its use. This pass rewrites such operands:
// AdStackLoadTopStmt / ArgLoadStmt get cloned in place, AdStackAllocaStmt
// gets re-rooted at the IB, generic SSA values get a per-IB backup
// AllocaStmt + LocalStore + LocalLoad chain.
// ============================================================================
class BackupSSA : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  Block *independent_block;
  std::map<Stmt *, Stmt *> backup_alloca;

  explicit BackupSSA(Block *independent_block) : independent_block(independent_block) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  Stmt *load(Stmt *stmt) {
    if (backup_alloca.find(stmt) == backup_alloca.end()) {
      auto alloca = Stmt::make<AllocaStmt>(stmt->ret_type.ptr_removed());
      auto alloca_ptr = alloca.get();
      independent_block->insert(std::move(alloca), 0);
      auto local_store = Stmt::make<LocalStoreStmt>(alloca_ptr, stmt);
      stmt->insert_after_me(std::move(local_store));
      backup_alloca[stmt] = alloca_ptr;
    }
    return backup_alloca[stmt];
  }

  void generic_visit(Stmt *stmt) {
    std::vector<Block *> leaf_to_root;
    auto t = stmt->parent;
    while (t != nullptr) {
      leaf_to_root.push_back(t);
      t = t->parent_block();
    }
    int num_operands = stmt->get_operands().size();
    for (int i = 0; i < num_operands; i++) {
      auto op = stmt->operand(i);
      if (op == nullptr) {
        continue;
      }
      if (std::find(leaf_to_root.begin(), leaf_to_root.end(), op->parent) == leaf_to_root.end() &&
          !op->is<AllocaStmt>()) {
        if (op->is<AdStackLoadTopStmt>()) {
          // Just create another AdStackLoadTopStmt
          stmt->set_operand(i, stmt->insert_before_me(op->clone()));
        } else if (op->is<AdStackAllocaStmt>()) {
          // Backup AdStackAllocaStmt because it should not be local stored and
          // local loaded
          auto stack_alloca = op->as<AdStackAllocaStmt>();
          if (backup_alloca.find(op) == backup_alloca.end()) {
            auto backup_stack_alloca = Stmt::make<AdStackAllocaStmt>(stack_alloca->dt, stack_alloca->max_size);
            auto backup_stack_alloca_ptr = backup_stack_alloca.get();
            independent_block->insert(std::move(backup_stack_alloca), 0);
            backup_alloca[op] = backup_stack_alloca_ptr;
            // Replace usages of all blocks i.e., the entry point for the
            // replace is the top level block
            irpass::replace_all_usages_with(leaf_to_root.back(), op, backup_stack_alloca_ptr);
            // Erase the outdated AdStackAllocaStmt
            op->parent->erase(op);
          }
        } else if (op->is<ArgLoadStmt>()) {
          stmt->set_operand(i, stmt->insert_before_me(op->clone()));
        } else {
          // Recomputable-chain fallback before the last-iter `load(op)` spill: when the cross-block SSA op
          // is rooted in a DAG of side-effect-free arithmetic over already-stack-backed allocas, kernel-args,
          // constants, and loop indices, clone the chain into the reverse scope at the consumer site. The
          // cloned chain reads stack tops live (matching `MakeAdjoint`'s pop ordering, which fires pops
          // AFTER all uses of a stack within one reverse iter) and reads kernel-args / constants / loop
          // indices via direct clones - exactly the path that was already correct for AdStackLoadTopStmt and
          // ArgLoadStmt operands.
          //
          // The pre-existing `load(op)` fallback below remains correct for genuinely non-recomputable
          // cross-block ops (a forward `GlobalLoadStmt` of a needs_grad SNode whose value the reverse must
          // read at last-write rather than recompute, for instance) and for shapes outside one of our
          // independent blocks. Adding the recomputable path above the fallback strictly subsets the
          // previous behaviour: a chain that fails the predicate falls through to the old `load(op)` line.
          if (RecomputableChainAnalyzer::is_recomputable(op, recomputable_cache_)) {
            std::unordered_map<Stmt *, Stmt *> clone_cache;
            Stmt *cloned = RecomputableChainCloner::clone_at(op, stmt, clone_cache);
            stmt->set_operand(i, cloned);
          } else {
            auto alloca = load(op);
            stmt->set_operand(i, stmt->insert_before_me(Stmt::make<LocalLoadStmt>(alloca)));
          }
        }
      }
    }
  }

  // Memoization cache for `RecomputableChainAnalyzer::is_recomputable` queries within one BackupSSA run.
  // Re-used across all generic_visit calls; invariant during the visit because forward IR is read-only here.
  std::unordered_map<Stmt *, bool> recomputable_cache_;

  void visit(Stmt *stmt) override {
    generic_visit(stmt);
  }

  void visit(IfStmt *stmt) override {
    generic_visit(stmt);
    BasicStmtVisitor::visit(stmt);
  }

  // generic_visit spills cross-block operands (the for-loop's `begin` and `end`) the same way it does for an
  // IfStmt's cond. MakeAdjoint clones a forward for-loop into the reverse scope and shares the clone's
  // `begin`/`end` pointers with the forward stmt; when those operands live inside the forward for's body (e.g.
  // inner `for k in range(j)` where `j` is an enclosing loop's index promoted to a per-iter adstack), the reverse
  // clone's operand no longer dominates its use. generic_visit's AdStackLoadTopStmt branch handles this by
  // inserting a fresh AdStackLoadTop in the reverse scope, which reads the correct per-iteration value.
  void visit(RangeForStmt *stmt) override {
    generic_visit(stmt);
    stmt->body->accept(this);
  }

  void visit(StructForStmt *stmt) override {
    generic_visit(stmt);
    stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) override {
    QD_ERROR("WhileStmt not supported in AutoDiff for now.");
  }

  void visit(Block *block) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : block->statements) {
      statements.push_back(stmt.get());
    }
    for (auto stmt : statements) {
      QD_ASSERT(!stmt->erased);
      stmt->accept(this);
    }
  }

  static void run(Block *block) {
    BackupSSA pass(block);
    block->accept(&pass);
  }
};

// ============================================================================
// CoalesceAdStackLoads: dedup redundant AdStackLoadTop / AdStackLoadTopAdj
// reads emitted by the reverse pass.
//
// Within a single straight-line block, multiple `AdStackLoadTopStmt` reads of the same stack with no
// intervening `AdStackPushStmt` / `AdStackPopStmt` for that stack return the same value, and multiple
// `AdStackLoadTopAdjStmt` reads are equivalent under the same conditions plus no intervening
// `AdStackAccAdjointStmt`. After Python-side static unrolling collapses an inner loop into straight-line IR,
// every iteration's read of an outer-loop-invariant adstack value emits a separate LoadTop in the same block;
// each individual load is cheap (read u64 count + GEP) but unrolled-loop counts of hundreds to thousands
// inflate PTX size and ptxas register-allocator cost. This pass walks each block, caches the most recent
// LoadTop / LoadTopAdj per stack, replaces subsequent same-stack reads with the cached SSA value until a
// Push / Pop / AccAdjoint invalidates the cache for that stack, and conservatively clears the cache when
// crossing into nested control flow (IfStmt / RangeForStmt / StructForStmt / WhileStmt) where unseen
// push/pop/AccAdjoint may appear.
// ============================================================================

// clang-format off
/*
Support for TensorType: How to handle MatrixPtrStmt & MatrixInitStmt

[Original Quadrants Code]

@ti.kernel
def test(...):
    b = ti.Vector([0, 1, 2, 3])
    b[2] = 100
    y = b[3] * b[2] * x


[Forward]                          [Forward-Replaced]              [Backward]
$b = alloca Tensor<4 x i32>   -->  $b = adstack alloca <4 x i32>
$1 = matrix init [0, 1, 2, 3] -->  $1 = matrix init [0, 1, 2, 3]
                                                                       adstack pop
$2:  local store $b, $1       -->  adstack push $1                 --> acc($1_adj, adstack top adj())

$3 = matrix ptr $b, 2         -->  $2 = adstack top(is_ptr=True)   --> adstack acc adj($2_adj)

                                                                       acc($2_adj, $14)
                                   $3 = matrix ptr $2, 0           --> $14 = matrix_init({$3_adj, 0, 0, 0})

                                                                       acc($2_adj, $13)
                                   $4 = matrix ptr $2, 2           --> $13 = matrix_init({0, 0, $4_adj, 0})

                                                                       acc($2_adj, $12)
                                   $5 = matrix ptr $2, 3           --> $12 = matrix_init({0, 0, 0, $5_adj})

                                   $6 = load($3)                   --> acc($3_adj, $6_adj)
                                   $7 = load($4)                   --> acc($4_adj, $7_adj)
                                   $8 = load($5)                   --> acc($5_adj, $8_adj)

                                                                       acc($8_adj, matrix ptr($9_adj, 3))
                                                                       acc($7_adj, matrix ptr($9_adj, 2))
                                                                       acc(100_adj, matrix ptr($9_adj, 1))
                                   $9 = matrix_init($6,100,$7,$8)  --> acc($6_adj, matrix ptr($9_adj, 0))

                                                                       adstack pop
$4 = local store $3, 100      -->  adstack push $9                 --> acc($9_adj, adstack top adj())

                                   $10 = adstack top(is_ptr=True)  --> adstack acc adj($10_adj)

                                                                       acc($10_adj, $18)
$5 = matrix ptr $b, 3         -->  $11 = matrix ptr $10, 3         --> $18 = matrix_init({0, 0, 0, $11_adj})
$b3 = local load $5           -->  $b3 = local load $11            --> acc($11_adj, $b3_adj)

                                   $12 = adstack top(is_ptr=True)  --> adstack acc adj($12_adj)

                                                                       acc($12_adj, $17)
$6 = matrix ptr $b, 2         -->  $13 = matrix ptr $12, 2         --> $17 = matrix_init({0, 0, $13_adj, 0})
$b2 = local load $6           -->  $b2 = local load $13            --> acc($13_adj, $b2_adj)

                                                                       acc($b3_adj, $15)
                                                                       acc($b2_adj, $16)
                                                                       $16 = mul($tmp_adj, $b3)
$tmp = mul b3, b2             -->  $tmp = mul $b3, $b2             --> $15 = mul($tmp_adj, $b2)

                                                                       acc($tmp_adj, $14)
$y = mul $tmp, $x             -->  $y = mul $tmp, $x               --> $14 = mul($y_adj, $x)
*/
// clang-format on
class CoalesceAdStackLoads : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  CoalesceAdStackLoads() {
    invoke_default_visitor = true;
    allow_undefined_visitor = true;
  }

  void visit(Block *block) override {
    std::unordered_map<Stmt *, Stmt *> primal_cache;
    std::unordered_map<Stmt *, Stmt *> adjoint_cache;
    // Iterate over a snapshot so erase()s during the walk do not invalidate the iteration. The
    // DelayedIRModifier still applies erases at the end via `modify_ir()` so SSA users get rewritten in one
    // pass; the snapshot only protects the visitor's own walk.
    std::vector<Stmt *> stmts;
    stmts.reserve(block->statements.size());
    for (auto &s : block->statements) {
      stmts.push_back(s.get());
    }
    for (Stmt *s : stmts) {
      if (auto *lt = s->cast<AdStackLoadTopStmt>()) {
        // `return_ptr=true` returns a pointer into the stack slot rather than loading a value; coalescing
        // those is unsound because subsequent stores via the pointer would alias with each other through
        // the cached SSA value. The non-pointer variant is the common case driven by reverse-pass formulas.
        if (lt->return_ptr) {
          continue;
        }
        auto it = primal_cache.find(lt->stack);
        if (it != primal_cache.end()) {
          irpass::replace_all_usages_with(lt->parent, lt, it->second);
          modifier_.erase(lt);
        } else {
          primal_cache[lt->stack] = lt;
        }
      } else if (auto *la = s->cast<AdStackLoadTopAdjStmt>()) {
        auto it = adjoint_cache.find(la->stack);
        if (it != adjoint_cache.end()) {
          irpass::replace_all_usages_with(la->parent, la, it->second);
          modifier_.erase(la);
        } else {
          adjoint_cache[la->stack] = la;
        }
      } else if (auto *p = s->cast<AdStackPushStmt>()) {
        primal_cache.erase(p->stack);
        adjoint_cache.erase(p->stack);
      } else if (auto *po = s->cast<AdStackPopStmt>()) {
        primal_cache.erase(po->stack);
        adjoint_cache.erase(po->stack);
      } else if (auto *aa = s->cast<AdStackAccAdjointStmt>()) {
        // Accumulating into the top adjoint slot does not alter the primal half, so the primal cache for
        // the same stack stays valid; only the adjoint cache must be invalidated.
        adjoint_cache.erase(aa->stack);
      } else if (s->is<IfStmt>() || s->is<RangeForStmt>() || s->is<StructForStmt>() || s->is<WhileStmt>()) {
        // The pass is intentionally intra-block: any push / pop hidden inside a nested block would
        // invalidate caches asymmetrically across branches and make a sound merge complex. Conservatively
        // drop both caches before descending so reads after the nested block see no stale entries.
        primal_cache.clear();
        adjoint_cache.clear();
        s->accept(this);
      } else {
        // Any other statement is assumed inert with respect to the AdStack count headers; recurse into
        // potential nested blocks (defensively) but leave the caches intact.
        s->accept(this);
      }
    }
  }

  static bool run(IRNode *root) {
    CoalesceAdStackLoads pass;
    root->accept(&pass);
    return pass.modifier_.modify_ir();
  }

 private:
  DelayedIRModifier modifier_;
};

}  // namespace

void backup_ssa(Block *block) {
  BackupSSA::run(block);
}

bool coalesce_ad_stack_loads(IRNode *root) {
  return CoalesceAdStackLoads::run(root);
}

}  // namespace quadrants::lang
