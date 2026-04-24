#include "quadrants/ir/analysis.h"
#include "quadrants/ir/control_flow_graph.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"

#include <limits>
#include <queue>
#include <unordered_map>

namespace quadrants::lang {

namespace irpass {

namespace {

// Structural pre-pass that resolves many AdStackAllocaStmts that would otherwise fall back to
// `default_ad_stack_size` in the CFG Bellman-Ford analyzer. The Bellman-Ford implementation treats
// any range-for whose body net-pushes an adstack as a positive loop and gives up, even when the
// loop's trip count is statically bounded (e.g. `range(N)` with a `ConstStmt` N). For reverse-mode
// kernels with many local variables promoted to adstacks inside such inner bounded loops, that
// pessimism bloats the per-thread heap stride by orders of magnitude; on wide-ndrange reverse
// kernels with hundreds of adstacks it forces a multi-GB heap per SPIR-V dispatch even when the
// real depth is bounded by the inner-loop trip count alone.
//
// For each adaptive AdStackAllocaStmt A we compute an upper bound on the stack depth by walking up
// the IR parent chain from every AdStackPushStmt that targets A. Each enclosing `RangeForStmt`
// whose begin and end are both `ConstStmt`s contributes its static trip count as a multiplier;
// non-loop container statements (if, while, ...) are passed through without affecting the
// multiplier. If every push for A is reachable only through bounded range-fors, the sum of the
// per-push multipliers is a safe upper bound on the concurrent stack depth and is assigned to
// `A->max_size` before Bellman-Ford runs. If any push reaches A through a non-constant range-for
// or a while-loop, A is left adaptive and handed to the existing CFG analyzer (which may still
// resolve it or fall back to `default_ad_stack_size`).
//
// The bound is pessimistic with respect to mutually exclusive if-branches (summing all pushes
// across both arms rather than taking the max), which is safe for heap sizing; mutually exclusive
// pushes waste slots but never under-allocate.
struct AdStackBoundResult {
  bool bounded{true};
  std::size_t max_size{0};
};

// Recursively evaluate a Stmt as an integer constant, folding through `BinaryOpStmt`
// add / sub / mul / div on const leaves. Returns `true` and writes `out` (widened to int64) if
// the stmt folds to an integer constant. Handles both signed and unsigned `ConstStmt` leaves
// (i8 / i16 / i32 / i64 / u1 / u8 / u16 / u32 / u64) rather than assuming `val.val_int32()`; a
// kernel compiled with `default_ip=i64` produces `RangeForStmt` bounds whose `ConstStmt`s are
// stored as i64 and would trip the i32-only accessor's internal assert before Bellman-Ford gets
// a chance to take over.
//
// The LLVM-backend pipeline sometimes leaves the inner `range(N)` loop bounds as an unfolded
// `BinaryOpStmt(add, ConstStmt(0), ConstStmt(N))` (e.g. when the frontend emits
// `range(begin, begin + n)` with `begin = 0`); full_simplify's constant-fold normally collapses
// these but is not guaranteed under every option toggle. This evaluator covers the small
// constant-arithmetic shapes the bounded-loop pre-pass cares about without depending on
// full_simplify's state.
bool try_eval_const_int(Stmt *stmt, int64_t *out) {
  if (stmt == nullptr) {
    return false;
  }
  if (auto *c = stmt->cast<ConstStmt>()) {
    const auto &dt = c->val.dt;
    if (is_signed(dt)) {
      *out = c->val.val_int();
      return true;
    }
    if (is_unsigned(dt)) {
      uint64_t u = c->val.val_uint();
      // Reject values that do not fit into an int64 round-trip; the downstream caller uses
      // `int64_t` arithmetic for trip-count computation and passing a too-large unsigned bound
      // would alias negative. In practice loop bounds are always well within i64 range; this
      // check just avoids silent wraparound.
      if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
      }
      *out = static_cast<int64_t>(u);
      return true;
    }
    return false;  // float / quant / other non-integer const leaf
  }
  if (auto *b = stmt->cast<BinaryOpStmt>()) {
    int64_t lhs = 0;
    int64_t rhs = 0;
    if (!try_eval_const_int(b->lhs, &lhs) || !try_eval_const_int(b->rhs, &rhs)) {
      return false;
    }
    switch (b->op_type) {
      case BinaryOpType::add:
        *out = lhs + rhs;
        return true;
      case BinaryOpType::sub:
        *out = lhs - rhs;
        return true;
      case BinaryOpType::mul:
        *out = lhs * rhs;
        return true;
      case BinaryOpType::div:
        if (rhs == 0) {
          return false;
        }
        *out = lhs / rhs;
        return true;
      default:
        return false;
    }
  }
  return false;
}

AdStackBoundResult compute_bounded_adstack_size(AdStackAllocaStmt *alloca, IRNode *root) {
  AdStackBoundResult result;
  auto push_stmts = irpass::analysis::gather_statements(root, [&](Stmt *s) {
    auto *push = s->cast<AdStackPushStmt>();
    return push != nullptr && push->stack == alloca;
  });
  for (Stmt *s : push_stmts) {
    auto *push = s->as<AdStackPushStmt>();
    std::size_t multiplier = 1;
    Block *blk = push->parent;
    while (blk != nullptr) {
      Stmt *parent = blk->parent_stmt();
      if (parent == nullptr) {
        break;
      }
      if (auto *range_for = parent->cast<RangeForStmt>()) {
        int64_t begin_v = 0;
        int64_t end_v = 0;
        if (!try_eval_const_int(range_for->begin, &begin_v) || !try_eval_const_int(range_for->end, &end_v)) {
          result.bounded = false;
          return result;
        }
        int64_t trip = end_v - begin_v;
        if (trip <= 0) {
          trip = 0;
        }
        // Guard against overflow before multiplying into `multiplier`.
        if (trip != 0 && multiplier > std::numeric_limits<std::size_t>::max() / std::size_t(trip)) {
          result.bounded = false;
          return result;
        }
        multiplier *= std::size_t(trip);
      } else if (parent->is<StructForStmt>() || parent->is<WhileStmt>()) {
        // Struct-for siblings inside an offloaded body or any while-loop are unbounded at
        // compile-time; the caller falls back to Bellman-Ford (and then the default) for this
        // adstack.
        result.bounded = false;
        return result;
      }
      // IfStmt, MeshForStmt-body, and other containers pass through without affecting the
      // multiplier - they do not iterate, so one enclosing execution of them equals one push.
      blk = parent->parent;
    }
    // Guard against overflow before accumulating.
    if (multiplier > std::numeric_limits<std::size_t>::max() - result.max_size) {
      result.bounded = false;
      return result;
    }
    result.max_size += multiplier;
  }
  if (result.bounded && result.max_size == 0) {
    // No pushes reach this alloca but it still exists in IR - match Bellman-Ford's behavior of
    // leaving such stacks alone (the verifier/DCE should have removed them).
    result.max_size = 1;
  }
  return result;
}

}  // namespace

bool determine_ad_stack_size(IRNode *root, const CompileConfig &config) {
  auto adaptive_allocas = irpass::analysis::gather_statements(root, [&](Stmt *s) {
    auto *ad_stack = s->cast<AdStackAllocaStmt>();
    return ad_stack != nullptr && ad_stack->max_size == 0;
  });
  if (adaptive_allocas.empty()) {
    return false;
  }
  // Phase 1: Bellman-Ford on the CFG. This pass tracks per-basic-block push / pop dynamics and takes the max
  // across branches, so it resolves tight cases the structural pre-pass would over-approximate: balanced push /
  // pop pairs (e.g. the `Basic` regression test pushes seven times but only reaches depth 4 because of
  // intervening pops) and max-across-if-branches (pessimistically summed by the structural pre-pass since it
  // sees both arms' pushes through an `IfStmt` pass-through). `apply_fallback = false` keeps positive-loop
  // stacks at `max_size = 0` so phase 2 can take over - that is exactly the shape the structural pre-pass is
  // designed for (statically-bounded inner `range(N)` kernels where Bellman-Ford gives up).
  auto cfg = analysis::build_cfg(root);
  cfg->simplify_graph();
  cfg->determine_ad_stack_size(config.default_ad_stack_size, /*apply_fallback=*/false);
  // Phase 2: structural pre-pass on the residual adaptive stacks (those Bellman-Ford deferred). Stacks already
  // resolved by phase 1 have `max_size != 0` and are skipped, so the tight Bellman-Ford bound survives. Any
  // alloca still at `max_size = 0` after this loop is handed back to the CFG analyzer via a second run with
  // `apply_fallback = true` so it gets the generic `default_ad_stack_size` cap.
  for (Stmt *s : adaptive_allocas) {
    auto *alloca = s->as<AdStackAllocaStmt>();
    if (alloca->max_size != 0) {
      continue;
    }
    auto bound = compute_bounded_adstack_size(alloca, root);
    if (bound.bounded) {
      alloca->max_size = bound.max_size;
    }
  }
  // Re-run Bellman-Ford so any alloca still unresolved picks up the `default_ad_stack_size` cap. The CFG
  // analyzer skips stacks whose `max_size` is already non-zero, so phase 1 / phase 2 results survive.
  auto cfg_final = analysis::build_cfg(root);
  cfg_final->simplify_graph();
  cfg_final->determine_ad_stack_size(config.default_ad_stack_size, /*apply_fallback=*/true);
  return true;
}

}  // namespace irpass

}  // namespace quadrants::lang
