#include "quadrants/transforms/auto_diff/auto_diff_common.h"
#include "quadrants/transforms/auto_diff/forward_state_spill.h"

namespace quadrants::lang {

namespace {

// ============================================================================
// PromoteSSA2LocalVar: hoist forward-pass SSA defs that the reverse pass will re-read into AllocaStmt + LocalStore +
// LocalLoad. Demand-driven.
//
// Note that SSA does not mean the instruction will be executed at most once. For instructions that may be executed
// multiple times, we treat them as a mutable local variables.
// ============================================================================
class PromoteSSA2LocalVar : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit PromoteSSA2LocalVar(Block *block) {
    alloca_block_ = block;
    invoke_default_visitor = true;
    execute_once_ = true;
  }

  // Demand-driven `required_defs_` set: the SSA defining stmts that downstream consumers actually require to be
  // available at every iteration. A consumer requires its operand iff its adjoint formula reads it - the precise set is
  // the operands of any non-linear unary / binary / ternary op (per `NonLinearOps::*_collections`), the indices of any
  // `GlobalPtrStmt` / `ExternalPtrStmt` (the reverse pass replays the load), and the `cond` of any `IfStmt` / the
  // `begin`/`end` of any `RangeForStmt` (the reverse pass clones the control flow). Stmts outside this set are left in
  // pure SSA form: their values stay register-resident inside the forward pass, and `MakeAdjoint`'s reverse-pass
  // formulas (which never read those values directly) generate correct adjoint accumulations against the adstack-backed
  // defs that ARE in the set. Skipping promotion for the rest of the body collapses the alloca + LocalStore + LocalLoad
  // triple-multiplier on unrolled IR that emitted a spill + reload pair per non-required arithmetic op with no
  // reverse-pass consumer for the spilled value.
  //
  // Operands of `LocalStoreStmt` are not added because `MakeAdjoint::visit(LocalStoreStmt)` reads only
  // `adjoint(stmt->dest)`, not the forward `stmt->val`. Linear ops (add / sub / mod / cmp / neg / floor / ceil / cast /
  // logic_not / bit-ops) are likewise excluded because their adjoint formulas read only `adjoint(stmt)`.
  static void compute_required_defs(Block *block, std::unordered_set<Stmt *> &out) {
    std::function<void(Block *)> walk = [&](Block *b) {
      for (auto &owned : b->statements) {
        Stmt *stmt = owned.get();
        if (auto *u = stmt->cast<UnaryOpStmt>()) {
          if (NonLinearOps::unary_collections.find(u->op_type) != NonLinearOps::unary_collections.end()) {
            out.insert(u->operand);
          }
        } else if (auto *bin = stmt->cast<BinaryOpStmt>()) {
          if (NonLinearOps::binary_collections.find(bin->op_type) != NonLinearOps::binary_collections.end()) {
            out.insert(bin->lhs);
            out.insert(bin->rhs);
          }
        } else if (auto *tern = stmt->cast<TernaryOpStmt>()) {
          if (NonLinearOps::ternary_collections.find(tern->op_type) != NonLinearOps::ternary_collections.end()) {
            out.insert(tern->op1);
            out.insert(tern->op2);
            out.insert(tern->op3);
          }
        } else if (auto *gp = stmt->cast<GlobalPtrStmt>()) {
          for (auto *idx : gp->indices) {
            out.insert(idx);
          }
        } else if (auto *ep = stmt->cast<ExternalPtrStmt>()) {
          for (auto *idx : ep->indices) {
            out.insert(idx);
          }
        } else if (auto *mp = stmt->cast<MatrixPtrStmt>()) {
          // Reverse-mode `MakeAdjoint::visit(MatrixPtrStmt)` reads `stmt->offset` for dynamic-index adjoint routing, so
          // a per-iteration-varying offset producer left in pure SSA would be backed by BackupSSA's single
          // overwrite-each-iteration alloca and the reverse pass would read the last forward offset for every
          // iteration. Promote it to alloca so `AdStackAllocaJudger::visit(MatrixPtrStmt)` can adstack-promote
          // loop-varying offsets. The cost of skipping this promotion is silent gradient corruption on `tensor[i +
          // j]`-style local-tensor indexing inside a serial range-for.
          out.insert(mp->offset);
        } else if (auto *if_s = stmt->cast<IfStmt>()) {
          out.insert(if_s->cond);
          if (if_s->true_statements) {
            walk(if_s->true_statements.get());
          }
          if (if_s->false_statements) {
            walk(if_s->false_statements.get());
          }
        } else if (auto *rf = stmt->cast<RangeForStmt>()) {
          out.insert(rf->begin);
          out.insert(rf->end);
          walk(rf->body.get());
        } else if (auto *sf = stmt->cast<StructForStmt>()) {
          if (sf->body) {
            walk(sf->body.get());
          }
        } else if (auto *while_s = stmt->cast<WhileStmt>()) {
          if (while_s->body) {
            walk(while_s->body.get());
          }
        }
      }
    };
    walk(block);
  }

  void visit(Stmt *stmt) override {
    if (execute_once_)
      return;
    if (!(stmt->is<UnaryOpStmt>() || stmt->is<BinaryOpStmt>() || stmt->is<TernaryOpStmt>() ||
          stmt->is<GlobalLoadStmt>() || stmt->is<LoopIndexStmt>() || stmt->is<AllocaStmt>())) {
      // TODO: this list may be incomplete
      return;
    }

    // `AllocaStmt`s always need to be hoisted to the top of the IB regardless of consumer analysis: a user-level `var =
    // ...` construct inside a loop body must own a fixed slot at the IB's entry so every iteration shares it
    // (cross-iteration accumulators are exactly the shape that drives the hoist). The demand-driven gate only applies
    // to value-producing stmts (UnaryOp / BinaryOp / TernaryOp / GlobalLoad / LoopIndex) where the alloca + LocalStore
    // + LocalLoad triple is purely a reverse-pass-readable spill - those skip when no consumer requires the value. The
    // `LocalStoreStmt`s emitted here are placeholders that `ReplaceLocalVarWithStacks` rewrites into `AdStackPushStmt`s
    // downstream.
    if (stmt->is<AllocaStmt>()) {
      auto dtype = stmt->ret_type.ptr_removed();
      auto alloc = Stmt::make<AllocaStmt>(dtype);
      auto alloc_ptr = alloc.get();
      QD_ASSERT(alloca_block_);
      alloca_block_->insert(std::move(alloc), 0);
      immediate_modifier_->replace_usages_with(stmt, alloc_ptr);

      auto zero = insert_const(dtype, stmt, 0);
      zero->insert_after_me(Stmt::make<LocalStoreStmt>(alloc_ptr, zero));
      stmt->parent->erase(stmt);
      return;
    }

    if (required_defs_.find(stmt) == required_defs_.end()) {
      return;
    }

    auto alloc = Stmt::make<AllocaStmt>(stmt->ret_type.ptr_removed());
    auto alloc_ptr = alloc.get();
    QD_ASSERT(alloca_block_);
    alloca_block_->insert(std::move(alloc), 0);
    auto load = stmt->insert_after_me(Stmt::make<LocalLoadStmt>(alloc_ptr));
    immediate_modifier_->replace_usages_with(stmt, load);
    // Create the load first so that the operand of the store does not get rewritten to point at the load (the SSA value
    // `stmt` is still the right thing to spill; only the downstream consumers see the load).
    stmt->insert_after_me(Stmt::make<LocalStoreStmt>(alloc_ptr, stmt));
  }

  void visit(RangeForStmt *stmt) override {
    auto old_execute_once = execute_once_;
    execute_once_ = false;  // loop body may be executed many times
    stmt->body->accept(this);
    execute_once_ = old_execute_once;
  }

  static void run(Block *block) {
    PromoteSSA2LocalVar pass(block);
    compute_required_defs(block, pass.required_defs_);
    pass.immediate_modifier_ = std::make_unique<ImmediateIRModifier>(block);
    block->accept(&pass);
  }

 private:
  Block *alloca_block_{nullptr};
  bool execute_once_;
  std::unordered_set<Stmt *> required_defs_;
  // ImmediateIRModifier collapses each `replace_usages_with` from a whole-tree walk (O(N)) to a constant-time
  // operand-pointer rewrite: the modifier gathers every (consumer, operand_index) pair feeding any existing stmt once
  // at construction (one O(N) pass), and per-replacement just looks up the table for `old_stmt` and rewrites each
  // consumer's operand pointer. Without it, `PromoteSSA2LocalVar` would run K (= number of promoted defs) full IR walks
  // - O(K*N), which dominates the Quadrants-IR-side compile cost on unrolled reverse-mode bodies.
  std::unique_ptr<ImmediateIRModifier> immediate_modifier_;
};

// ============================================================================
// AdStackAllocaJudger: per-AllocaStmt analysis. Decides whether an alloca's runtime sequence of values must be
// preserved on an AdStack so the reverse pass can re-read every forward-time value, or whether a single
// overwrite-each-iteration backing slot suffices.
// ============================================================================
class AdStackAllocaJudger : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  // Find the usage of the stmt recursively along the LocalLoadStmt
  void visit(LocalLoadStmt *stmt) override {
    if (stmt->src == target_alloca_) {
      local_loaded_ = true;
      target_alloca_ = stmt;
    }
  }

  // Track whether the alloca has any store at all so a load-only alloca (no adstack-relevant data flow either
  // direction) can short-circuit `run()` regardless of what the per-op visitors below find. The decision of whether the
  // alloca needs adstack promotion is made entirely by the precise visitors: non-linear unary / binary / ternary,
  // GlobalPtr / ExternalPtr index, and IfStmt / RangeForStmt bound. Plus an alloca that is both loaded and stored
  // anywhere in the IB is treated as loop-carried, which is needed for kernels like `for j: p, q = q, p + q` where the
  // reverse pass routes the gradient through the cross-iteration recurrence and BackupSSA's single
  // overwrite-each-iteration alloca cannot back the read-after-write across iterations. The visit-order-dependent
  // load+store evidence here is conservative: any alloca with both a load and a store inside the IB triggers it,
  // including pure accumulators whose adjoint formulas don't actually need per-iteration values - the slight
  // over-promotion cost is the price of correctness on Fibonacci-style recurrences (silent gradient corruption
  // otherwise).
  void visit(LocalStoreStmt *stmt) override {
    if (stmt->dest == target_alloca_backup_) {
      load_only_ = false;
      // Gate the load+store-implies-stack-needed rule on actually being inside a dynamic RangeForStmt at the point this
      // evidence accumulates. The rule's purpose is to preserve cross-iteration RAW dependencies (`for j: p, q = q, p +
      // q` Fibonacci-style) that BackupSSA's single overwrite-each-iteration alloca cannot back. With no enclosing
      // dynamic for-loop the IB body executes once: there is no cross-iteration RAW to preserve, and the "load+store"
      // pattern is just an in-block accumulator that the reverse pass handles via plain SSA cloning. Promoting such
      // allocas under a static-unrolled loop body wastes one AdStack per accumulator (one push per unrolled-iter store
      // + one load_top per unrolled-iter load) without any reverse-pass consumer needing per-iter replay.
      //
      // `dynamic_for_depth_` is incremented in `visit(RangeForStmt)` and decremented on exit. The judger walks the IB
      // tree from the alloca's enclosing block, so depth here reflects exactly the nesting of *dynamic* for-loops
      // between the alloca and the current load/store. StructFor / WhileStmt do not increment because their bodies
      // still execute per-iter and need the same RAW protection (StructFor is the kernel-level offload-loop in some
      // cases, but its body is a per-thread independent block; load+store there is the same shape as for a top-level
      // alloca).
      if (local_loaded_ && dynamic_for_depth_ > 0) {
        is_stack_needed_ = true;
      }
    }
  }

  // Check if the alloca is load only
  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest == target_alloca_backup_)
      load_only_ = false;
  }

  // The stack is needed if the alloca serves as the index of any global variables. Same cursor-vs-backup pattern as
  // visit(IfStmt) / visit(RangeForStmt) below: `index` is always a value-producing stmt (typically a `LocalLoadStmt`
  // reading the alloca, or a `ConstStmt`), never the alloca itself. The raw `index == target_alloca_` comparison only
  // matches the first load's instance the `visit(LocalLoadStmt)` cursor advanced to - any subsequent load of the same
  // alloca used as a different GlobalPtr index slips through. Resolve the LocalLoad chain and compare `ll->src` against
  // `target_alloca_backup_` to catch every load.
  void visit(GlobalPtrStmt *stmt) override {
    if (is_stack_needed_)
      return;
    for (const auto &index : stmt->indices) {
      auto *index_ll = index->cast<LocalLoadStmt>();
      if (index_ll && index_ll->src == target_alloca_backup_)
        is_stack_needed_ = true;
    }
  }

  void visit(ExternalPtrStmt *stmt) override {
    if (is_stack_needed_)
      return;
    for (const auto &index : stmt->indices) {
      auto *index_ll = index->cast<LocalLoadStmt>();
      if (index_ll && index_ll->src == target_alloca_backup_)
        is_stack_needed_ = true;
    }
  }

  // Reverse-mode `MakeAdjoint::visit(MatrixPtrStmt)` reads `stmt->offset` for dynamic-index adjoint routing, so a
  // runtime-varying offset whose value comes from this alloca needs adstack promotion - otherwise BackupSSA backs the
  // offset with a single overwrite-each-iteration slot and the reverse pass routes every iteration's adjoint into the
  // last forward offset's slot. Same cursor-vs-backup pattern as the index visitors above.
  void visit(MatrixPtrStmt *stmt) override {
    if (is_stack_needed_)
      return;
    auto *offset_ll = stmt->offset->cast<LocalLoadStmt>();
    if (offset_ll && offset_ll->src == target_alloca_backup_)
      is_stack_needed_ = true;
  }

  // Check whether the target alloca is fed into a non-linear unary op. Same cursor-vs-backup pattern as
  // visit(GlobalPtrStmt) above: `stmt->operand` is a value-producing stmt (typically LocalLoad), never the alloca
  // itself, so resolve the LocalLoad chain and compare against the backup.
  void visit(UnaryOpStmt *stmt) override {
    if (is_stack_needed_)
      return;
    if (NonLinearOps::unary_collections.find(stmt->op_type) != NonLinearOps::unary_collections.end()) {
      auto *operand_ll = stmt->operand->cast<LocalLoadStmt>();
      if (operand_ll && operand_ll->src == target_alloca_backup_)
        is_stack_needed_ = true;
    }
  }

  // Check whether the target alloca is fed into a non-linear binary op. Same cursor-vs-backup pattern.
  void visit(BinaryOpStmt *stmt) override {
    if (is_stack_needed_)
      return;
    if (NonLinearOps::binary_collections.find(stmt->op_type) != NonLinearOps::binary_collections.end()) {
      auto *lhs_ll = stmt->lhs->cast<LocalLoadStmt>();
      auto *rhs_ll = stmt->rhs->cast<LocalLoadStmt>();
      if ((lhs_ll && lhs_ll->src == target_alloca_backup_) || (rhs_ll && rhs_ll->src == target_alloca_backup_))
        is_stack_needed_ = true;
    }
  }

  // Check whether the target alloca is fed into a non-linear ternary op. Same cursor-vs-backup pattern.
  void visit(TernaryOpStmt *stmt) override {
    if (is_stack_needed_)
      return;
    if (NonLinearOps::ternary_collections.find(stmt->op_type) != NonLinearOps::ternary_collections.end()) {
      auto *op1_ll = stmt->op1->cast<LocalLoadStmt>();
      auto *op2_ll = stmt->op2->cast<LocalLoadStmt>();
      auto *op3_ll = stmt->op3->cast<LocalLoadStmt>();
      if ((op1_ll && op1_ll->src == target_alloca_backup_) || (op2_ll && op2_ll->src == target_alloca_backup_) ||
          (op3_ll && op3_ll->src == target_alloca_backup_))
        is_stack_needed_ = true;
    }
  }

  // Check whether the target alloca feeds the condition of an if stmt. `stmt->cond` is always a value-producing stmt -
  // typically a direct `LocalLoadStmt` reading the alloca, but also commonly a `BinaryOpStmt` wrapping such a load
  // (e.g. `j < i+1`). Walk the expression chain via `feeds_target_alloca` to catch every load of the target alloca,
  // including loads nested under comparison or other linear ops (which `visit(BinaryOpStmt)` does not flag because
  // comparison ops are not in `NonLinearOps`). The walk is defensive: IR simplification currently collapses most
  // BinaryOp-wrapped conds before the judger sees them, so most input IR uses the bare-LocalLoad shape; the walker
  // stays sound when the wrapping is preserved.
  void visit(IfStmt *stmt) override {
    if (is_stack_needed_)
      return;

    if (feeds_target_alloca(stmt->cond)) {
      is_stack_needed_ = true;
      return;
    }

    if (stmt->true_statements)
      stmt->true_statements->accept(this);
    if (stmt->false_statements)
      stmt->false_statements->accept(this);
  }

  // Check whether the target alloca feeds the begin or end of a range-for bound. Under reverse-mode AD, if an inner
  // for-loop's bound is an enclosing loop-carried counter (the canonical triangular-nested `for k in range(j)` shape,
  // or the `range(j+1)` / `range(n-i)` shapes where the bound is a linear arithmetic expression of a loop-carried
  // alloca), its reverse clone must read the bound from the per-iteration forward value; without an adstack the reverse
  // pass sees only the last forward value and the inner loop over- or under-runs, silently corrupting gradients for the
  // earliest inner indices (those visited most often across outer iterations). This check is the only thing that
  // promotes such a loop-counter alloca - `visit(LocalStoreStmt)`'s `local_loaded_` short-circuit does not fire because
  // the counter is only LOAD-ed inside the inner-loop bound, not LOAD-then-STORE-ed. Walk the expression chain through
  // `feeds_target_alloca` so both direct LocalLoads (`range(j)`) and LocalLoads nested under linear ops (`range(j+1)`,
  // `range(n-i)`, ...) trigger promotion. The BinaryOp-wrapped case is covered defensively: IR simplification currently
  // collapses most such bounds before the judger sees them, so most input IR uses the bare-LocalLoad shape; the walker
  // stays sound when the wrapping is preserved. The bare-LocalLoad shape takes the first branch of the walker
  // trivially.
  void visit(RangeForStmt *stmt) override {
    if (is_stack_needed_)
      return;

    if (feeds_target_alloca(stmt->begin) || feeds_target_alloca(stmt->end)) {
      is_stack_needed_ = true;
      return;
    }

    dynamic_for_depth_++;
    stmt->body->accept(this);
    dynamic_for_depth_--;
  }

  static bool run(AllocaStmt *target_alloca) {
    AdStackAllocaJudger judger;
    judger.target_alloca_ = target_alloca;
    judger.target_alloca_backup_ = target_alloca;
    judger.dynamic_for_depth_ = 0;
    target_alloca->parent->accept(&judger);
    return (!judger.load_only_) && judger.is_stack_needed_;
  }

 private:
  // Recursively walk a value expression to decide whether it transitively reads `target_alloca_backup_` via a
  // `LocalLoadStmt`. Used by `visit(IfStmt)` and `visit(RangeForStmt)` to detect the target alloca feeding a bound or
  // condition even when wrapped by linear ops (e.g. `range(j+1)`, `j < i+1`). Linear binary/unary ops are traversed
  // because `visit(BinaryOpStmt)`/`visit(UnaryOpStmt)` only flag *non-linear* ops - their linear-op path does not
  // otherwise promote the alloca. `ConstStmt`s and unrelated values return false and terminate the recursion; the
  // walker is always finite because SSA IR guarantees acyclic operand graphs.
  bool feeds_target_alloca(Stmt *expr) const {
    if (auto *ll = expr->cast<LocalLoadStmt>()) {
      return ll->src == target_alloca_backup_;
    }
    if (auto *bop = expr->cast<BinaryOpStmt>()) {
      return feeds_target_alloca(bop->lhs) || feeds_target_alloca(bop->rhs);
    }
    if (auto *uop = expr->cast<UnaryOpStmt>()) {
      return feeds_target_alloca(uop->operand);
    }
    if (auto *top = expr->cast<TernaryOpStmt>()) {
      return feeds_target_alloca(top->op1) || feeds_target_alloca(top->op2) || feeds_target_alloca(top->op3);
    }
    return false;
  }

  Stmt *target_alloca_;
  Stmt *target_alloca_backup_;
  bool is_stack_needed_ = false;
  bool local_loaded_ = false;
  bool load_only_ = true;
  // Nesting depth of dynamic `RangeForStmt` containers between the alloca's enclosing block and the current visit
  // cursor. Static-unrolled `qd.static(range(...))` loops are removed by the AST transformer before the judger sees the
  // IR, so they do not contribute to depth. The load+store-implies-stack-needed rule fires only when this depth is
  // positive; see the rationale in `visit(LocalStoreStmt)`.
  int dynamic_for_depth_ = 0;
};

// ============================================================================
// ReplaceLocalVarWithStacks: rewrite each AllocaStmt that AdStackAllocaJudger approved into AdStackAllocaStmt +
// AdStackPush + AdStackLoadTop. Tensor-typed allocas need extra care because per-slot stack writes are not a reaching
// def for the store-to-load forwarding walker.
// ============================================================================
class ReplaceLocalVarWithStacks : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  int ad_stack_size;
  DelayedIRModifier delayed_modifier_;

  explicit ReplaceLocalVarWithStacks(int ad_stack_size) : ad_stack_size(ad_stack_size) {
  }

  void visit(AllocaStmt *alloc) override {
    bool is_stack_needed = AdStackAllocaJudger::run(alloc);
    if (is_stack_needed) {
      auto dtype = alloc->ret_type.ptr_removed();
      auto stack_alloca = Stmt::make<AdStackAllocaStmt>(dtype, ad_stack_size);
      auto stack_alloca_ptr = stack_alloca.get();

      alloc->replace_with(VecStatement(std::move(stack_alloca)));

      // Note that unlike AllocaStmt, AdStackAllocaStmt does NOT have a 0 as initial value, so we push an initial 0
      // here.
      auto zero = insert_const(dtype, stack_alloca_ptr, 0);
      zero->insert_after_me(Stmt::make<AdStackPushStmt>(stack_alloca_ptr, zero));
    }
  }

  void visit(LocalLoadStmt *stmt) override {
    if (stmt->src->is<AdStackAllocaStmt>()) {
      auto stack_load = Stmt::make<AdStackLoadTopStmt>(stmt->src);
      stack_load->ret_type = stmt->ret_type;

      stmt->replace_with(std::move(stack_load));
      return;
    }

    // Slot load from a stack-backed tensor. After `visit(MatrixPtrStmt)`, `stmt->src` is of the form
    // `MatrixPtrStmt(AdStackLoadTopStmt(stack, return_ptr=true), offset)`. A direct load through that pointer leaves
    // the store-to-load forwarding walker in `ir/control_flow_graph.cpp` with no reaching definition, because the only
    // producer for the stack's top slots is an `AdStackPushStmt` (tagged `ir_traits::Load`, invisible to
    // `get_store_destination`). Replace the load with a full-tensor `AdStackLoadTopStmt` materialized into a fresh
    // regular `AllocaStmt`, then re-subscript it - a plain alloca + LocalStore sequence is a shape the reach-in walker
    // can trace end-to-end.
    if (stmt->src->is<MatrixPtrStmt>()) {
      auto matrix_ptr = stmt->src->as<MatrixPtrStmt>();
      if (matrix_ptr->origin->is<AdStackLoadTopStmt>() && matrix_ptr->origin->as<AdStackLoadTopStmt>()->return_ptr) {
        auto stack = matrix_ptr->origin->as<AdStackLoadTopStmt>()->stack;
        QD_ASSERT(stack->is<AdStackAllocaStmt>());
        auto tensor_type = stack->ret_type.ptr_removed();

        auto full_load = Stmt::make<AdStackLoadTopStmt>(stack);
        full_load->ret_type = tensor_type;
        auto full_load_ptr = full_load.get();

        auto fresh_alloca = Stmt::make<AllocaStmt>(tensor_type);
        auto fresh_alloca_ptr = fresh_alloca.get();
        fresh_alloca->ret_type = tensor_type;
        fresh_alloca->ret_type.set_is_pointer(true);

        auto fresh_store = Stmt::make<LocalStoreStmt>(fresh_alloca_ptr, full_load_ptr);

        auto new_matrix_ptr = Stmt::make<MatrixPtrStmt>(fresh_alloca_ptr, matrix_ptr->offset);
        new_matrix_ptr->ret_type = stmt->ret_type;

        auto new_load = Stmt::make<LocalLoadStmt>(new_matrix_ptr.get());
        new_load->ret_type = stmt->ret_type;

        stmt->insert_before_me(std::move(full_load));
        stmt->insert_before_me(std::move(fresh_alloca));
        stmt->insert_before_me(std::move(fresh_store));
        stmt->insert_before_me(std::move(new_matrix_ptr));
        stmt->replace_with(std::move(new_load));
      }
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    if (stmt->dest->is<MatrixPtrStmt>()) {
      auto matrix_ptr_stmt = stmt->dest->as<MatrixPtrStmt>();
      if (matrix_ptr_stmt->origin->is<AdStackLoadTopStmt>()) {
        auto stack_top_stmt = matrix_ptr_stmt->origin->as<AdStackLoadTopStmt>();
        QD_ASSERT(stack_top_stmt->return_ptr == true);

        if (!stack_top_stmt->ret_type.ptr_removed()->is<TensorType>()) {
          return;
        }

        auto tensor_type = stack_top_stmt->ret_type.ptr_removed()->as<TensorType>();
        auto num_elements = tensor_type->get_num_elements();

        if (matrix_ptr_stmt->offset->is<ConstStmt>()) {
          /*
            [Static index]
            Load the full current top as a tensor via `AdStackLoadTopStmt` and merge the new value at `offset`
            using a boolean mask + `select`. Mirrors the dynamic-index lowering below so that every slot of the
            new pushed tensor derives from either `stmt->val` or the loaded top tensor and the IR contains no
            per-slot `LocalLoadStmt` on a stack-backed `MatrixPtrStmt`.

            Why that invariant matters: the store-to-load forwarding walker in `ir/control_flow_graph.cpp` does
            not treat `AdStackPushStmt` as a reaching definition (it is tagged `ir_traits::Load`, so
            `get_store_destination` returns nothing for it), so a `LocalLoadStmt(MatrixPtrStmt(stack_top_ptr,
            i))` inserted here has no reaching def and ends up reading an uninitialized adjoint slot in the
            reverse kernel. Keep the `AdStackLoadTopStmt(stack)` + mask-select shape when touching this path.

            Fwd:
            $1 = alloca <4 x i32>
            $2 = matrix ptr $1, 2 // offset = 2
            $3 : local store $2, $val

            Replaced:
            $1 = alloca <4 x i32>

            $2 = matrix init [$val, $val, $val, $val]
            $3 = matrix init [false, false, true, false]   // mask with `offset == i`

            $4 = ad stack load top (full tensor) $1
            $5 = select $3, $2, $4

            $6 : stack push $1, $5
          */
          int offset = matrix_ptr_stmt->offset->as<ConstStmt>()->val.val_int32();

          QD_ASSERT(offset < num_elements);

          auto tensor_shape = tensor_type->get_shape();
          auto cmp_tensor_type = TypeFactory::get_instance().get_tensor_type(tensor_shape, PrimitiveType::u1);

          std::vector<Stmt *> val_values(num_elements, stmt->val);
          std::vector<Stmt *> mask_values(num_elements);
          for (int i = 0; i < num_elements; i++) {
            mask_values[i] = insert_const(PrimitiveType::u1, stmt, i == offset ? 1 : 0, true);
          }

          auto matrix_val = Stmt::make<MatrixInitStmt>(val_values);
          matrix_val->ret_type = tensor_type;

          auto matrix_mask = Stmt::make<MatrixInitStmt>(mask_values);
          matrix_mask->ret_type = cmp_tensor_type;

          auto matrix_alloca_value = Stmt::make<AdStackLoadTopStmt>(stack_top_stmt->stack);
          matrix_alloca_value->ret_type = tensor_type;

          auto matrix_select = Stmt::make<TernaryOpStmt>(TernaryOpType::select, matrix_mask.get(), matrix_val.get(),
                                                         matrix_alloca_value.get());
          matrix_select->ret_type = tensor_type;

          auto stack_push = Stmt::make<AdStackPushStmt>(stack_top_stmt->stack, matrix_select.get());

          stmt->insert_before_me(std::move(matrix_val));
          stmt->insert_before_me(std::move(matrix_mask));
          stmt->insert_before_me(std::move(matrix_alloca_value));
          stmt->insert_before_me(std::move(matrix_select));
          stmt->replace_with(std::move(stack_push));

          return;

        } else {
          /*
            [Dynamic index]
            Fwd:
            $1 = alloca <4 x i32>
            $2 = matrix ptr $1, $offset // offset = 2
            $3 : local store $2, $val

            Replaced:
            $1 = alloca <4 x i32>

            $2 = matrix init [$val, $val, $val, $val]

            $3 = matrix init [$offset, $offset, $offset, $offset]
            $4 = matrix init [0, 1, 2, 3]

            $5 = bin_eq $3, $4
            $6 = select $5, $2, $1

            $7 : store $1, $6
          */
          auto tensor_type = stack_top_stmt->ret_type.ptr_removed()->as<TensorType>();
          auto num_elements = tensor_type->get_num_elements();

          auto tensor_shape = tensor_type->get_shape();
          auto index_tensor_type = TypeFactory::get_instance().get_tensor_type(tensor_shape, PrimitiveType::i32);

          std::vector<Stmt *> val_values(num_elements, stmt->val);
          std::vector<Stmt *> offset_values(num_elements, matrix_ptr_stmt->offset);
          std::vector<Stmt *> index_values(num_elements);
          for (int i = 0; i < num_elements; i++) {
            index_values[i] = insert_const(PrimitiveType::i32, stmt, i, true);
          }

          auto matrix_val = Stmt::make<MatrixInitStmt>(val_values);
          matrix_val->ret_type = tensor_type;

          auto matrix_offset = Stmt::make<MatrixInitStmt>(offset_values);
          matrix_offset->ret_type = index_tensor_type;

          auto matrix_index = Stmt::make<MatrixInitStmt>(index_values);
          matrix_index->ret_type = index_tensor_type;

          auto cmp_tensor_type = TypeFactory::get_instance().get_tensor_type(tensor_shape, PrimitiveType::u1);
          auto matrix_eq = Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_eq, matrix_offset.get(), matrix_index.get());
          matrix_eq->ret_type = cmp_tensor_type;

          auto matrix_alloca_value = Stmt::make<AdStackLoadTopStmt>(stack_top_stmt->stack);
          matrix_alloca_value->ret_type = tensor_type;

          auto matrix_select = Stmt::make<TernaryOpStmt>(TernaryOpType::select, matrix_eq.get(), matrix_val.get(),
                                                         matrix_alloca_value.get());
          matrix_select->ret_type = tensor_type;

          auto stack_push = Stmt::make<AdStackPushStmt>(stack_top_stmt->stack, matrix_select.get());

          stmt->insert_before_me(std::move(matrix_val));
          stmt->insert_before_me(std::move(matrix_offset));
          stmt->insert_before_me(std::move(matrix_index));
          stmt->insert_before_me(std::move(matrix_eq));
          stmt->insert_before_me(std::move(matrix_alloca_value));
          stmt->insert_before_me(std::move(matrix_select));
          stmt->replace_with(std::move(stack_push));

          return;
        }
      }
    }

    // Non Tensor-type
    if (stmt->dest->is<AdStackAllocaStmt>())
      stmt->replace_with(Stmt::make<AdStackPushStmt>(stmt->dest, stmt->val));
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<AdStackAllocaStmt>()) {
      auto stack_top = Stmt::make<AdStackLoadTopStmt>(stmt->origin, true /*is_ptr*/);
      stack_top->ret_type = stmt->origin->ret_type;
      stack_top->ret_type.set_is_pointer(true);

      Stmt *stack_top_stmt = stack_top.get();
      stmt->insert_before_me(std::move(stack_top));

      auto new_matrix_ptr_stmt = Stmt::make<MatrixPtrStmt>(stack_top_stmt, stmt->offset);
      new_matrix_ptr_stmt->ret_type = stmt->ret_type;
      stmt->replace_with(std::move(new_matrix_ptr_stmt));
    }
  }
};

}  // namespace

void promote_ssa_to_local_var(Block *block) {
  PromoteSSA2LocalVar::run(block);
}

void replace_local_var_with_stacks(Block *block, int ad_stack_size) {
  ReplaceLocalVarWithStacks pass(ad_stack_size);
  block->accept(&pass);
}

}  // namespace quadrants::lang
