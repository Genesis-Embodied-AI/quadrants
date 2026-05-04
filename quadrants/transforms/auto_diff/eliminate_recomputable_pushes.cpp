#include "quadrants/transforms/auto_diff/auto_diff_common.h"
#include "quadrants/transforms/auto_diff/forward_state_spill.h"

namespace quadrants::lang {

namespace {

// Eliminate AdStackAllocaStmts whose pushed value is recomputable from already-stack-backed allocas, kernel
// args, constants, and loop indices. Runs between `ReplaceLocalVarWithStacks` and `MakeAdjoint`, so the reverse
// pass is generated against the cleaned IR (no spurious AdStackPushStmt / AdStackLoadTopStmt scaffolding for
// values the reverse can reconstruct on the fly via cloned forward DAGs).
//
// Eligibility per AdStackAllocaStmt S in an independent block:
//   1. S is written by exactly one AdStackPushStmt (single-push pattern). Multi-push allocas hold loop-carried
//      state where each iteration's push depends on the previous - the reverse pass cannot reconstruct
//      iteration k's value from iteration (k-1)'s value, so the stack is genuinely needed.
//   2. The pushed value's transitive operand DAG is recomputable per RecomputableChainAnalyzer (leaves at
//      AdStackLoadTop / AdStackAlloca / ArgLoad / Const / LoopIndex; interior side-effect-free ops only).
//   3. S has no AdStackLoadTopStmt with `return_ptr=true` consumers (those return a pointer aliasing the slot
//      and a follow-up store would not be modeled by an SSA replacement).
//
// Action on eligibility: replace every `AdStackLoadTopStmt(S)` with the original pushed SSA value (the
// `AdStackPushStmt::v`), then erase the push and the alloca. The pushed SSA value's chain stays in the forward
// IR; downstream consumers in the forward pass now reference it directly as SSA, and `MakeAdjoint` plus
// `BackupSSA` reconstruct the chain on demand in the reverse pass via the same DAG-clone path that
// `BackupSSA::generic_visit` exercises for cross-block SSA references.
//
// Iterates to fixed point: eliminating one stack can newly expose another stack's chain as recomputable
// (S2's pushed value chained through `AdStackLoadTopStmt(S1)`; once S1 is gone, the chain may collapse into
// pure SSA chained through S1's pushed-value chain instead). Each pass eliminates at least one stack or
// terminates.
//
// Cost model: this pass trades forward-pass adstack pushes (memory write + top-pointer bump per push) for
// extra arithmetic in the reverse pass (the cloned DAG re-executes the forward chain). For pure-arithmetic
// chains rooted at a single loop-carried alloca - the dominant shape on Genesis-style rigid-step kernels -
// the win is typically 5-10x: each push is a memory op crossing the L1 boundary on every iteration, while
// the recomputed arithmetic stays in registers and reuses warmed-up sin/cos/exp pipeline state.
class EliminateRecomputableAdStackPushes {
 public:
  static void run(Block *ib) {
    // Iterate to fixed point. Each pass either eliminates at least one stack or terminates. The hard cap
    // protects against analysis bugs (e.g. an eligibility check that returns true on the same stmt twice
    // in a row); under a correct implementation each pass strictly reduces the AdStackAllocaStmt count, so
    // the actual iteration bound is the number of stacks at entry. 1024 is well above any realistic kernel.
    for (int i = 0; i < 1024; i++) {
      if (!run_one_pass(ib))
        return;
    }
  }

 private:
  // True iff `s` is a literal-zero ConstStmt or a MatrixInitStmt whose every element is a literal-zero
  // ConstStmt. Matches the init-zero push pattern that ReplaceLocalVarWithStacks emits right after each
  // AdStackAllocaStmt: scalar allocas get a ConstStmt(0), tensor-typed allocas get a MatrixInitStmt of
  // ConstStmt(0)s. Both are erased along with the alloca during elimination - they only matter if a load
  // could observe them before the body push fires, which the standard PromoteSSA2LocalVar +
  // ReplaceLocalVarWithStacks codegen never produces (loads always follow the body push within the loop
  // iter that emits them).
  static bool is_zero_init_value(Stmt *s) {
    if (auto *c = s->cast<ConstStmt>()) {
      return c->val.equal_value(0);
    }
    if (auto *mi = s->cast<MatrixInitStmt>()) {
      for (auto *elem : mi->values) {
        auto *c = elem->cast<ConstStmt>();
        if (c == nullptr || !c->val.equal_value(0))
          return false;
      }
      return true;
    }
    return false;
  }

  // Returns true if any stack was eliminated this pass.
  static bool run_one_pass(Block *ib) {
    // Collect every AdStackAllocaStmt anywhere within the IB. The IB is the root of a contiguous AD scope, so
    // a single walk is sufficient.
    std::vector<AdStackAllocaStmt *> stacks;
    auto collected = irpass::analysis::gather_statements(ib, [&](Stmt *s) { return s->is<AdStackAllocaStmt>(); });
    for (auto *s : collected) {
      stacks.push_back(s->as<AdStackAllocaStmt>());
    }

    bool modified = false;
    std::unordered_map<Stmt *, bool> recomputable_cache;

    for (auto *stack : stacks) {
      // Re-classify users of `stack` each iteration: a previous elimination on a downstream stack may have
      // removed users that previously disqualified `stack`.
      std::vector<AdStackPushStmt *> pushes;
      std::vector<AdStackLoadTopStmt *> load_tops;
      bool disqualified = false;

      auto users = irpass::analysis::gather_statements(ib, [&](Stmt *s) {
        if (auto *p = s->cast<AdStackPushStmt>())
          return p->stack == stack;
        if (auto *lt = s->cast<AdStackLoadTopStmt>())
          return lt->stack == stack;
        if (auto *po = s->cast<AdStackPopStmt>())
          return po->stack == stack;
        if (auto *aa = s->cast<AdStackAccAdjointStmt>())
          return aa->stack == stack;
        if (auto *la = s->cast<AdStackLoadTopAdjStmt>())
          return la->stack == stack;
        return false;
      });
      for (auto *user : users) {
        if (auto *p = user->cast<AdStackPushStmt>()) {
          pushes.push_back(p);
        } else if (auto *lt = user->cast<AdStackLoadTopStmt>()) {
          if (lt->return_ptr) {
            // return_ptr=true returns a pointer into the slot; replacing with an SSA value loses the
            // pointer-identity contract. Keep the stack.
            disqualified = true;
            break;
          }
          load_tops.push_back(lt);
        } else if (user->is<AdStackPopStmt>() || user->is<AdStackAccAdjointStmt>() ||
                   user->is<AdStackLoadTopAdjStmt>()) {
          // Pre-MakeAdjoint, pop / adj-acc / load-top-adj should not appear yet for this stack. If they do,
          // some upstream pass has already touched it - keep as-is to avoid double-rewrites.
          disqualified = true;
          break;
        }
      }
      if (disqualified || pushes.empty()) {
        continue;
      }

      // ReplaceLocalVarWithStacks emits a const-zero "init" AdStackPushStmt immediately after the
      // AdStackAllocaStmt as the stack's initial value (auto_diff.cpp:867-872 in pre-fix tree). Real
      // user-level allocas in a loop body have at most ONE additional "body" push per iteration. Treat
      // const-zero pushes whose value is a `ConstStmt` with all-zero `TypedConstant` as inits and require at
      // most one non-init "body" push for elimination eligibility. Multi-body-push allocas hold loop-carried
      // state (each iteration's push depends on the previous), so the reverse pass cannot reconstruct
      // iteration k from iteration k-1 and the stack must stay.
      AdStackPushStmt *body_push = nullptr;
      std::vector<AdStackPushStmt *> init_pushes;
      for (auto *p : pushes) {
        if (is_zero_init_value(p->v)) {
          init_pushes.push_back(p);
        } else {
          if (body_push != nullptr) {
            disqualified = true;
            break;
          }
          body_push = p;
        }
      }
      if (disqualified || body_push == nullptr) {
        continue;
      }

      Stmt *pushed_val = body_push->v;
      if (!RecomputableChainAnalyzer::is_recomputable(pushed_val, recomputable_cache)) {
        continue;
      }
      // Loop-carried self-reference: when the pushed value's transitive operand DAG includes an
      // AdStackLoadTopStmt of the SAME stack we are about to eliminate, the value is `read prev iter from
      // stack -> compute -> push next iter`. Rewriting load_top($S) -> pushed_value_SSA leaves a self-cycle
      // in the SSA graph (`acc_new = acc_new + sin(a)`) which both corrupts the IR and loses the iteration
      // recurrence the stack was carrying. The recomputable-chain analyzer happily accepts such chains
      // because AdStackLoadTopStmt is a recomputable leaf in general; the additional check below specifically
      // disqualifies self-loaded stacks.
      //
      // The classic shape this protects: `acc = 0.0; for j in range(N): acc += sin(...)`. After
      // PromoteSSA2LocalVar + ReplaceLocalVarWithStacks the IR has `$acc = stack_alloc; init push;
      // for j: $tmp = stack_load_top $acc; $new = $tmp + sin(...); push $acc, val=$new`. The chain `$tmp +
      // sin(...)` is recomputable per the leaf rules (LoadTop is a leaf), but rewriting load_top($acc) to
      // ($tmp + sin(...)) makes the SSA graph self-reference $new and the reverse pass loses every
      // iteration's accumulator state.
      // Read-before-write protection: substituting `AdStackLoadTopStmt(S)` with the body push's value SSA
      // chain only preserves semantics when the body push DOMINATES every load_top in the forward IR -
      // that is, every load_top reads what the body push just wrote in the same iteration. The
      // PromoteSSA2LocalVar + ReplaceLocalVarWithStacks pipeline emits exactly this shape for required-def
      // spills: `alloca; push val=def; load_top` in document order, with the push immediately following
      // the def and loads occurring later in the same iteration.
      //
      // Loop-carried recurrences violate the dominance rule. Take the canonical Fibonacci shape
      // `p, q = q, p + q` lowered to IR:
      //
      //     $tmp_p = load_top($p)         # reads PREVIOUS iter's push into $p
      //     $tmp_q = load_top($p) + load_top($q)
      //     push $p, val=$tmp_p           # writes NEXT iter's value into $p
      //     push $q, val=$tmp_q
      //
      // Here load_top($p) reads BEFORE the body push of $p within the same iter, so it observes iter
      // (k-1)'s value, not iter k's. Substituting `load_top($p) -> $tmp_p`'s SSA chain (which reads
      // load_top($q) at iter k) gives iter k's q-value instead of iter (k-1)'s p-value, off by one
      // iteration and producing zero gradients on the Fibonacci-style regression test pinned by
      // `test_ad_fibonacci_index`.
      //
      // The check below asserts that for every load_top in `load_tops`, the body push is its predecessor
      // within the same containing block (or, equivalently, load_top comes AFTER the body push in
      // document order at the body push's block level). Loads that live inside nested control flow
      // (`IfStmt`, `RangeForStmt`) under the body push's block are also fine because the body push
      // dominates them. We approximate dominance with the lexical "push's block contains load_top's
      // ancestor block AND push position < load_top's ancestor's position in that block" check.
      auto block_position = [](Stmt *s) -> int {
        Block *b = s->parent;
        if (b == nullptr)
          return -1;
        for (size_t i = 0; i < b->statements.size(); i++) {
          if (b->statements[i].get() == s)
            return static_cast<int>(i);
        }
        return -1;
      };
      Block *push_block = body_push->parent;
      int push_pos = block_position(body_push);
      bool dominates_all_loads = true;
      for (auto *lt : load_tops) {
        Stmt *cursor = lt;
        Block *cursor_block = cursor->parent;
        while (cursor_block != nullptr && cursor_block != push_block) {
          cursor = cursor_block->parent_stmt();
          if (cursor == nullptr)
            break;
          cursor_block = cursor->parent;
        }
        if (cursor_block != push_block) {
          // load_top is outside the push's block scope: cannot establish dominance.
          dominates_all_loads = false;
          break;
        }
        int cursor_pos = block_position(cursor);
        if (cursor_pos <= push_pos) {
          // load_top precedes the body push (or its enclosing container does): it reads the previous
          // iteration's value, not iter k's. Substitution would shift iterations by one.
          dominates_all_loads = false;
          break;
        }
      }
      if (!dominates_all_loads) {
        continue;
      }

      // Reverse-position correctness for chain leaves pushed in the same block as S.
      //
      // After elimination, every reverse stmt that consumed `load_top(S)` instead consumes the chain
      // (cloned by `BackupSSA::generic_visit` at the consumer's reverse position). The cloned chain reads
      // `load_top(T)` for each chain leaf T, where the read happens at the consumer's reverse cursor.
      //
      // `MakeAdjoint` visits forward stmts in REVERSE order, emitting at a cursor that advances per visit.
      // For a forward stmt at position P, its reverse emissions land "later" in reverse block iff P is
      // smaller. T's pop is emitted when `MakeAdjoint` visits T's body push at position P_T. The cloned
      // chain at the consumer's reverse position reads `load_top(T)` POST-pop iff T's pop has fired by then,
      // i.e. iff visit(P_T) precedes visit(C_pos), i.e. iff P_T > C_pos for every consumer C of S's
      // load_tops.
      //
      // Forward chain evaluates at S's push position P_S, before any of T's same-block body pushes (since
      // we already required P_T > P_S via the dominance check above for S's loads). So forward chain reads
      // T's pre-iter-k-push value. Reverse clone post-pop also returns T's pre-iter-k-push value (= iter k
      // INPUT for loop-carried T). Match.
      //
      // Without this check, the canonical Fibonacci-style shape (`p, q = q, p + q; b[q] += a[q]`) where S
      // stores `p + q` and chain leaves p_stack / q_stack are pushed AFTER S but BEFORE the GlobalPtr
      // index consumer would silently corrupt gradients: the reverse clone at the GlobalPtr's adjoint
      // emission reads p_stack and q_stack PRE-pop (iter k POST-push values), summing to a different value
      // than the forward chain at P_S used.
      //
      // Chain leaves T whose stacks have NO body push in S's containing block are stable within the iter
      // (their tops do not change during the iter), so neither forward nor reverse evaluation order shifts
      // their values. Excluded from this check.
      auto find_consumers_of_load_tops = [&](std::vector<int> &out_positions) {
        std::function<void(Block *, int)> walk = [&](Block *b, int container_pos_in_push_block) {
          for (size_t i = 0; i < b->statements.size(); i++) {
            Stmt *s = b->statements[i].get();
            int effective_pos = (b == push_block) ? static_cast<int>(i) : container_pos_in_push_block;
            for (auto *op : s->get_operands()) {
              if (op == nullptr)
                continue;
              for (auto *lt : load_tops) {
                if (op == lt) {
                  out_positions.push_back(effective_pos);
                  break;
                }
              }
            }
            if (auto *if_s = s->cast<IfStmt>()) {
              if (if_s->true_statements)
                walk(if_s->true_statements.get(), effective_pos);
              if (if_s->false_statements)
                walk(if_s->false_statements.get(), effective_pos);
            } else if (auto *rf = s->cast<RangeForStmt>()) {
              if (rf->body)
                walk(rf->body.get(), effective_pos);
            } else if (auto *sf = s->cast<StructForStmt>()) {
              if (sf->body)
                walk(sf->body.get(), effective_pos);
            }
          }
        };
        walk(push_block, -1);
      };
      std::vector<int> consumer_positions;
      find_consumers_of_load_tops(consumer_positions);
      int max_consumer_pos = -1;
      for (int p : consumer_positions) {
        if (p > max_consumer_pos)
          max_consumer_pos = p;
      }

      // Collect all distinct AdStackAllocaStmt referenced via AdStackLoadTopStmt leaves in pushed_val's chain.
      std::unordered_set<AdStackAllocaStmt *> chain_leaf_stacks;
      std::unordered_set<Stmt *> walked_for_leaves;
      std::function<void(Stmt *)> collect_leaves = [&](Stmt *s) {
        if (walked_for_leaves.count(s))
          return;
        walked_for_leaves.insert(s);
        if (auto *lt = s->cast<AdStackLoadTopStmt>()) {
          if (auto *as = lt->stack->cast<AdStackAllocaStmt>())
            chain_leaf_stacks.insert(as);
          return;
        }
        if (s->is<AdStackAllocaStmt>() || s->is<ArgLoadStmt>() || s->is<ConstStmt>())
          return;
        for (auto *op : s->get_operands()) {
          if (op != nullptr)
            collect_leaves(op);
        }
      };
      collect_leaves(pushed_val);

      bool reverse_safe = true;
      for (auto *T : chain_leaf_stacks) {
        // Find T's body pushes (non-init) in push_block. Only same-block pushes can interfere: pushes in
        // outer blocks happen once per outer iteration, so their tops are stable across the inner iter.
        for (size_t i = 0; i < push_block->statements.size(); i++) {
          Stmt *s = push_block->statements[i].get();
          if (auto *p = s->cast<AdStackPushStmt>()) {
            if (p->stack == T && !is_zero_init_value(p->v)) {
              if (static_cast<int>(i) <= max_consumer_pos) {
                reverse_safe = false;
                break;
              }
            }
          }
        }
        if (!reverse_safe)
          break;
      }
      if (!reverse_safe) {
        continue;
      }

      bool is_self_loaded = false;
      std::unordered_set<Stmt *> visited;
      std::function<void(Stmt *)> walk = [&](Stmt *s) {
        if (is_self_loaded || visited.count(s))
          return;
        visited.insert(s);
        if (auto *lt = s->cast<AdStackLoadTopStmt>()) {
          if (lt->stack == stack) {
            is_self_loaded = true;
            return;
          }
          // load_top of a different stack: the operand chain stops here at this leaf for the self-check.
          return;
        }
        if (s->is<AdStackAllocaStmt>() || s->is<ArgLoadStmt>() || s->is<ConstStmt>()) {
          return;  // leaf, not a self-load
        }
        for (auto *op : s->get_operands()) {
          if (op != nullptr)
            walk(op);
        }
      };
      walk(pushed_val);
      if (is_self_loaded) {
        continue;
      }

      // Control-flow-cond consumers: by the time the IR reaches this pass, every IfStmt cond and RangeFor begin/end
      // is either a bare `AdStackLoadTopStmt` (loop-carried local promoted via `PromoteSSA2LocalVar` ->
      // `AdStackAllocaJudger::visit(IfStmt|RangeForStmt)` -> `ReplaceLocalVarWithStacks`) or a stmt outside any
      // adstack-bearing alloca (no load_top in the chain at all). `MakeAdjoint::visit(IfStmt)` (auto_diff.cpp around
      // line 2452) caps the bare-load_top case with a 1-push-per-execution snap-stack when the if body itself pushes
      // to the cond's backing stack so the reverse cond reads the forward-time value, not a stack top mutated by the
      // body.
      //
      // If we eliminate a stack whose load_top IS the bare cond of an IfStmt / inner RangeFor, the rewrite turns the
      // cond / loop-bound into an inlined recomputed SSA chain. The snap-stack guard at the `AdStackLoadTopStmt`-cond
      // check in `MakeAdjoint::visit(IfStmt)` stops firing (the cond is no longer bare), and `BackupSSA`'s clone path
      // positions a fresh `AdStackLoadTopStmt` of an enclosing loop-carried stack at the consumer's IR location -
      // which sits BEFORE the per-iter pops, returning the post-iteration value instead of the iteration-k value the
      // cond was originally computed against. Net effect: silent gradient corruption in any if/loop nested inside a
      // dynamic for-loop with a loop-carried alloca.
      //
      // Keep stacks whose load_tops have such consumers. The consumer check below trips only on a load_top of THIS
      // stack feeding DIRECTLY into a control-flow-shaping operand of another stmt; that is also the exhaustive set
      // of unsafe-elim shapes here, because compound-cond cases (arithmetic over a load_top feeding a cond) reach
      // this pass with the cond rewritten into a separate adstack-promoted value via the alloca-promotion pipeline
      // above and so look like the bare case from this guard's POV.
      // `irpass::analysis::gather_statements` walks via `BasicStmtVisitor`, whose visit overrides for
      // IfStmt / RangeForStmt / StructForStmt do not invoke the per-stmt test predicate on the container
      // itself - only on stmts inside their bodies. Walk container stmts manually here.
      bool has_control_flow_consumer = false;
      std::function<void(Block *)> walk_block = [&](Block *b) {
        if (has_control_flow_consumer)
          return;
        for (auto &owned : b->statements) {
          Stmt *s = owned.get();
          if (auto *if_s = s->cast<IfStmt>()) {
            for (auto *lt : load_tops) {
              if (if_s->cond == lt) {
                has_control_flow_consumer = true;
                return;
              }
            }
            if (if_s->true_statements)
              walk_block(if_s->true_statements.get());
            if (has_control_flow_consumer)
              return;
            if (if_s->false_statements)
              walk_block(if_s->false_statements.get());
          } else if (auto *rf = s->cast<RangeForStmt>()) {
            for (auto *lt : load_tops) {
              if (rf->begin == lt || rf->end == lt) {
                has_control_flow_consumer = true;
                return;
              }
            }
            if (rf->body)
              walk_block(rf->body.get());
          } else if (auto *sf = s->cast<StructForStmt>()) {
            if (sf->body)
              walk_block(sf->body.get());
          } else if (auto *off = s->cast<OffloadedStmt>()) {
            if (off->body)
              walk_block(off->body.get());
            if (off->tls_prologue)
              walk_block(off->tls_prologue.get());
            if (off->bls_prologue)
              walk_block(off->bls_prologue.get());
            if (off->bls_epilogue)
              walk_block(off->bls_epilogue.get());
            if (off->tls_epilogue)
              walk_block(off->tls_epilogue.get());
          }
        }
      };
      walk_block(ib);
      if (has_control_flow_consumer) {
        continue;
      }

      // Eligible: rewrite each load_top to use the pushed value directly. The pushed value lives in the
      // forward IR and dominates each load_top by SSA construction (the load_top reads what the push wrote;
      // both are inside the same loop body in the dynamic-loop case, with the push preceding the load_top).
      for (auto *lt : load_tops) {
        irpass::replace_all_usages_with(ib, lt, pushed_val);
        lt->parent->erase(lt);
      }
      // Erase init-zero pushes (they only matter if a load could observe them, but the rewriting above just
      // routed every load to the body-pushed SSA value), the body push, and the alloca itself.
      for (auto *p : init_pushes) {
        p->parent->erase(p);
      }
      body_push->parent->erase(body_push);
      stack->parent->erase(stack);

      // Invalidate the cache: the eliminated stack might have been referenced by other recomputable chains
      // we evaluated earlier in this pass, but those evaluations only matter for stacks we eliminate THIS
      // pass; re-running from scratch on the next iteration recomputes them.
      recomputable_cache.clear();
      modified = true;
    }
    return modified;
  }
};

}  // namespace

void eliminate_recomputable_ad_stack_pushes(Block *ib) {
  EliminateRecomputableAdStackPushes::run(ib);
}

}  // namespace quadrants::lang
