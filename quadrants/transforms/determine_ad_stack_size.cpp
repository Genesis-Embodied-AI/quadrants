#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/control_flow_graph.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/type_utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

namespace quadrants::lang {

namespace irpass {

namespace {

// Structural / symbolic pre-pass that resolves each `AdStackAllocaStmt`'s max_size as a `SizeExpr`
// rooted at constants and scalar integer field loads. The Bellman-Ford CFG analyzer (phase 1)
// already handles balanced push/pop pairs and max-across-if-branches exactly; this phase takes over
// for the "positive-loop" shapes Bellman-Ford gives up on. For each remaining adaptive alloca we
// walk up the parent chain from every `AdStackPushStmt` targeting it and build a symbolic
// upper-bound expression:
//   - Every enclosing `RangeForStmt` contributes `end - begin` as a multiplicative factor. Both
//     bounds must themselves be expressible in the strict-minimum grammar (`Const`, scalar i32/i64
//     `GlobalLoadStmt`, `add` / `sub` / `max` of the above; `min` is conservatively upper-bounded
//     by `max`).
//   - Non-iterating containers (`IfStmt`, ...) pass through without affecting the multiplier - one
//     enclosing execution equals one push, and max-across-branches is handled in phase 1.
//   - `StructForStmt` and `WhileStmt` are unbounded at compile time, as are range-fors whose bounds use shapes
//     outside the grammar. There is no size-fallback: any alloca still unresolved after phases 1 and 2 is a
//     hard compile error so overflows surface at compile time rather than silently over-allocating at runtime.
// The per-push multipliers are summed (pessimistic across mutually-exclusive `if` branches, which is safe for
// sizing). The host evaluator reads the resulting expression against the live field state right before every
// kernel launch so field-load-dependent bounds size the adstack heap precisely per-launch instead of tripping
// an overflow at runtime.

struct AdStackBoundResult {
  std::unique_ptr<SizeExpr> expr;
  bool bounded{true};
  // Set when the structural walk encountered a data-flow cycle (`build_value_expr` hit a stmt already on its
  // recursion path) and the idempotency probe could not discharge it. Consumers use this to produce a specific
  // "stash data-flow cycle" error rather than the generic "unresolved shape" one, since the symptom and the
  // remedy differ from a plain grammar-gap case.
  bool cycle_detected{false};
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

std::unique_ptr<SizeExpr> expr_max(std::unique_ptr<SizeExpr> a, std::unique_ptr<SizeExpr> b);

std::unique_ptr<SizeExpr> expr_add(std::unique_ptr<SizeExpr> a, std::unique_ptr<SizeExpr> b) {
  if (a->kind == SizeExpr::Kind::Const && b->kind == SizeExpr::Kind::Const) {
    return SizeExpr::make_const(a->const_value + b->const_value);
  }
  if (a->kind == SizeExpr::Kind::Const && a->const_value == 0) {
    return b;
  }
  if (b->kind == SizeExpr::Kind::Const && b->const_value == 0) {
    return a;
  }
  return SizeExpr::make_binary(SizeExpr::Kind::Add, std::move(a), std::move(b));
}

// Alpha-equivalence over `SizeExpr` trees: two trees are equal when they match structurally AND every
// `BoundVariable` reference on the `a` side binds to the same `MaxOverRange` wrapper as the corresponding
// reference on the `b` side. `var_map` tracks the current mapping `a.var_id -> b.var_id` for each bound
// variable we've walked into. This matters for `expr_sub`'s MaxOverRange-fusion: two independently-built
// range wrappers over the same user loop end up with DIFFERENT fresh `var_id`s in any inner MaxOverRange
// emitted while resolving the enclosing loop's end (e.g. a stashed loop index whose load-top chase is
// wrapped), so naive structural equality on `var_id` rejects pairs that are semantically identical.
bool expr_equal_alpha(const SizeExpr *a, const SizeExpr *b, std::unordered_map<int32_t, int32_t> *var_map) {
  if (a == nullptr || b == nullptr) {
    return a == b;
  }
  if (a->kind != b->kind)
    return false;
  if (a->const_value != b->const_value)
    return false;
  if (a->snode != b->snode)
    return false;
  if (a->arg_shape_axis != b->arg_shape_axis)
    return false;
  if (a->arg_id_path != b->arg_id_path)
    return false;
  if (a->operands.size() != b->operands.size())
    return false;
  if (a->kind == SizeExpr::Kind::BoundVariable) {
    auto it = var_map->find(a->var_id);
    int32_t mapped = (it != var_map->end()) ? it->second : a->var_id;
    return mapped == b->var_id;
  }
  if (a->kind == SizeExpr::Kind::FieldLoad || a->kind == SizeExpr::Kind::ExternalTensorRead) {
    if (a->indices.size() != b->indices.size())
      return false;
    for (std::size_t i = 0; i < a->indices.size(); ++i) {
      int64_t ai = a->indices[i];
      int64_t bi = b->indices[i];
      // Constant-index entries use non-negative encoding and must match literally. `BoundVariable`
      // references use `-(var_id + 1)` encoding and must match under the current var rename map.
      if (ai >= 0 || bi >= 0) {
        if (ai != bi)
          return false;
      } else {
        int32_t av = static_cast<int32_t>(-(ai + 1));
        int32_t bv = static_cast<int32_t>(-(bi + 1));
        auto it = var_map->find(av);
        int32_t mapped = (it != var_map->end()) ? it->second : av;
        if (mapped != bv)
          return false;
      }
    }
  } else {
    if (a->indices != b->indices)
      return false;
  }
  if (a->kind == SizeExpr::Kind::MaxOverRange && a->operands.size() == 3) {
    // Compare begin / end in the enclosing scope.
    if (!expr_equal_alpha(a->operands[0].get(), b->operands[0].get(), var_map))
      return false;
    if (!expr_equal_alpha(a->operands[1].get(), b->operands[1].get(), var_map))
      return false;
    // Extend the rename map with the newly-bound variable, compare bodies, then restore.
    auto prev_it = var_map->find(a->var_id);
    bool had = (prev_it != var_map->end());
    int32_t prev = had ? prev_it->second : 0;
    (*var_map)[a->var_id] = b->var_id;
    bool eq = expr_equal_alpha(a->operands[2].get(), b->operands[2].get(), var_map);
    if (had) {
      (*var_map)[a->var_id] = prev;
    } else {
      var_map->erase(a->var_id);
    }
    return eq;
  }
  // Anything else (Const, FieldLoad, Add, Sub, Mul, Max, ExternalTensor*): compare every operand
  // recursively. `var_id` is already accounted for by the `BoundVariable` / `MaxOverRange` cases above;
  // for the remaining kinds, leave `var_id` at its default (-1) and do not gate equality on it.
  for (std::size_t i = 0; i < a->operands.size(); ++i) {
    if (!expr_equal_alpha(a->operands[i].get(), b->operands[i].get(), var_map))
      return false;
  }
  return true;
}

bool expr_equal(const SizeExpr *a, const SizeExpr *b) {
  std::unordered_map<int32_t, int32_t> var_map;
  return expr_equal_alpha(a, b, &var_map);
}

// Rename every free reference to bound variable `from` inside `e` to `to`, in place. Respects shadowing by
// inner `MaxOverRange` nodes that rebind `from` (i.e. a nested `MaxOverRange(var=from, ...)` shadows `from`
// in its body; we only rewrite the nested node's begin/end, not its body). `FieldLoad` / `ExternalTensorRead`
// nodes encode bound-variable references inside their `indices` as `-(var_id + 1)`, so substitution has to
// walk that encoding too.
void substitute_var_in_place(SizeExpr *e, int32_t from, int32_t to) {
  if (e == nullptr)
    return;
  if (e->kind == SizeExpr::Kind::BoundVariable) {
    if (e->var_id == from)
      e->var_id = to;
    return;
  }
  if (e->kind == SizeExpr::Kind::MaxOverRange && e->var_id == from) {
    // Inner rebind shadows the outer `from`. Walk begin/end (they're evaluated in the outer scope) but skip
    // the body (evaluated in the inner scope where `from` means something different).
    if (e->operands.size() >= 2) {
      substitute_var_in_place(e->operands[0].get(), from, to);
      substitute_var_in_place(e->operands[1].get(), from, to);
    }
    return;
  }
  if (e->kind == SizeExpr::Kind::FieldLoad || e->kind == SizeExpr::Kind::ExternalTensorRead) {
    for (auto &idx : e->indices) {
      if (idx < 0 && -(idx + 1) == static_cast<int64_t>(from)) {
        idx = -static_cast<int64_t>(to + 1);
      }
    }
  }
  for (auto &child : e->operands) {
    substitute_var_in_place(child.get(), from, to);
  }
}

// True if `e` or any descendant node has `kind`. Used by `expr_sub` to refuse the `MaxOverRange` fusion
// when the synthesised body would pair a `FieldLoad` with an `ExternalTensorRead`.
bool contains_kind(const SizeExpr *e, SizeExpr::Kind kind) {
  if (e == nullptr) {
    return false;
  }
  if (e->kind == kind) {
    return true;
  }
  for (const auto &child : e->operands) {
    if (contains_kind(child.get(), kind)) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<SizeExpr> expr_sub(std::unique_ptr<SizeExpr> a, std::unique_ptr<SizeExpr> b) {
  if (a->kind == SizeExpr::Kind::Const && b->kind == SizeExpr::Kind::Const) {
    return SizeExpr::make_const(a->const_value - b->const_value);
  }
  if (b->kind == SizeExpr::Kind::Const && b->const_value == 0) {
    return a;
  }
  if (a->kind == SizeExpr::Kind::Const && a->const_value == 0) {
    // Every `SizeExpr` in this pass represents a non-negative integer (a count, an index, or a shape), so
    // `b >= 0` at runtime, and `Sub` itself clamps its result to `max(a - b, 0)` in the host evaluator. Hence
    // `Sub(0, *)` always evaluates to `0`, which lets the idempotency probe recognise cross-stack cyclic
    // patterns like `sub(load_top(self), load_top(other))` as safely idempotent-at-zero even when the other
    // stack's non-cyclic bound is not itself a constant.
    return SizeExpr::make_const(0);
  }
  // Fuse `Sub(MaxOverRange(X, B, E, body_a), MaxOverRange(Y, B, E, body_b))` into
  // `MaxOverRange(X, B, E, Sub(body_a, body_b[Y->X]))` when the two wrappers are over the SAME user loop.
  // `expr_equal` on the ends gates this via alpha-equivalence (see `expr_equal_alpha`'s comment): it only
  // returns true when the two independently-built range wrappers bottom out at the same source IR loop, so
  // fusing is semantically identical to the user's per-iteration paired subtraction. Without fusion the
  // unfused form evaluates to `max_i body_a(i) - max_j body_b(j)` at runtime, which under-counts
  // `max_i (body_a(i) - body_b(i))` whenever the per-operand maxima are attained at different indices.
  // Cross-user-loop fusion (two MaxOverRange wrappers from two independent enclosing loops that happen to
  // iterate over same-axis ndarray shapes) is NOT safe: the user's two loops use independent indices, and
  // fusing under a single `v` fabricates an index pairing that does not exist in the source - the bound
  // collapses to `max_v (arr_a[v] - arr_b[v])` over `min(shape_a, shape_b)`, which silently misses any peak
  // in `arr_a` past `shape_b` and drives the adstack toward overflow at `qd.sync()`. Those cases fall to
  // the `MaxOverRange_a` over-approximation below.
  if (a->kind == SizeExpr::Kind::MaxOverRange && b->kind == SizeExpr::Kind::MaxOverRange && a->operands.size() == 3 &&
      b->operands.size() == 3 && expr_equal(a->operands[0].get(), b->operands[0].get()) &&
      expr_equal(a->operands[1].get(), b->operands[1].get())) {
    // Decline the fusion when the resulting body would mix a `FieldLoad` and an `ExternalTensorRead`. Each
    // side is independently host-foldable (its MaxOverRange wraps a closed subtree), but the fused body
    // would carry a `FieldLoad` indexed by the bound variable `v` alongside an `ExternalTensorRead[v]` - a
    // combination that `encode_adstack_size_expr_device_bytecode` rejects on CUDA / AMDGPU because the LLVM
    // device interpreter has no on-device SNode access. Falling through to the `MaxOverRange_a` branch
    // below keeps the adstack-size bound sound (`max(0, a - b) <= a` when b is a non-negative `SizeExpr`)
    // and avoids synthesising the mixed body for the encoder to choke on.
    const bool a_has_fieldload = contains_kind(a->operands[2].get(), SizeExpr::Kind::FieldLoad);
    const bool b_has_fieldload = contains_kind(b->operands[2].get(), SizeExpr::Kind::FieldLoad);
    const bool a_has_extread = contains_kind(a->operands[2].get(), SizeExpr::Kind::ExternalTensorRead);
    const bool b_has_extread = contains_kind(b->operands[2].get(), SizeExpr::Kind::ExternalTensorRead);
    const bool would_mix = (a_has_fieldload && b_has_extread) || (a_has_extread && b_has_fieldload);
    if (!would_mix) {
      int32_t var_x = a->var_id;
      int32_t var_y = b->var_id;
      auto body_b_renamed = b->operands[2] ? b->operands[2]->clone() : nullptr;
      if (body_b_renamed != nullptr && var_x != var_y) {
        substitute_var_in_place(body_b_renamed.get(), var_y, var_x);
      }
      auto new_body = expr_sub(std::move(a->operands[2]), std::move(body_b_renamed));
      return SizeExpr::make_max_over_range(var_x, std::move(a->operands[0]), std::move(a->operands[1]),
                                           std::move(new_body));
    }
  }
  // Sound upper bound on `Sub(a, b)` when both operands are `MaxOverRange` and we could not fuse them (cross-
  // user-loop, or same loop but mixed FieldLoad / ExternalTensorRead body). `SizeExpr` trees always evaluate
  // to non-negative integers (counts / indices / shapes), so `max(0, eval(a) - eval(b)) <= eval(a)` - taking
  // `a` alone is a sound over-approximation. The host evaluator's grammar cannot express the tight
  // `max_i a(i) - min_j b(j)` form without a `Min` operator; dropping back to `a` trades a small amount of
  // adstack memory for correctness and keeps the encoder free of mixed-leaf bodies.
  if (a->kind == SizeExpr::Kind::MaxOverRange && b->kind == SizeExpr::Kind::MaxOverRange) {
    return a;
  }
  return SizeExpr::make_binary(SizeExpr::Kind::Sub, std::move(a), std::move(b));
}

std::unique_ptr<SizeExpr> expr_max(std::unique_ptr<SizeExpr> a, std::unique_ptr<SizeExpr> b) {
  if (a->kind == SizeExpr::Kind::Const && b->kind == SizeExpr::Kind::Const) {
    return SizeExpr::make_const(std::max(a->const_value, b->const_value));
  }
  return SizeExpr::make_binary(SizeExpr::Kind::Max, std::move(a), std::move(b));
}

// No `Mul` in the strict-minimum `SizeExpr` grammar. For the common nested-loop shape
// `const_inner * field_load_outer` we materialize `field_load + field_load + ...` via `Add` - a
// functionally identical upper bound that stays inside the grammar. Two non-const factors is a
// compile error (returns nullptr): nested field-load-bounded loops are out of scope for this pass.
// The const-factor cap keeps the resulting tree small; a factor larger than `kMaxReplicate` is
// rejected rather than blowing up the `SizeExpr` allocation.
constexpr int64_t kMaxReplicate = 1024;

std::unique_ptr<SizeExpr> expr_mul(std::unique_ptr<SizeExpr> a, std::unique_ptr<SizeExpr> b) {
  if (a->kind == SizeExpr::Kind::Const && b->kind == SizeExpr::Kind::Const) {
    return SizeExpr::make_const(a->const_value * b->const_value);
  }
  if (a->kind != SizeExpr::Kind::Const && b->kind != SizeExpr::Kind::Const) {
    // Neither side is a compile-time constant: emit a dynamic `Mul` node the evaluator multiplies at launch
    // time. This handles nested-loop trip-count products where each loop's bound is itself dynamic
    // (e.g. `for i in range(arr.shape[0]): for j in range(arr[i]): ...` - both trips resolve to non-const
    // `SizeExpr` trees whose product is the total push count the adstack must accommodate).
    return SizeExpr::make_binary(SizeExpr::Kind::Mul, std::move(a), std::move(b));
  }
  if (b->kind == SizeExpr::Kind::Const) {
    std::swap(a, b);  // put the const on `a`
  }
  int64_t factor = a->const_value;
  if (factor <= 0) {
    return SizeExpr::make_const(0);
  }
  if (factor == 1) {
    return b;
  }
  if (factor > kMaxReplicate) {
    // Large const factor with a non-const sibling: replicating via `Add` would blow up the tree, so fall
    // back to a dynamic `Mul` instead of refusing the bound outright.
    return SizeExpr::make_binary(SizeExpr::Kind::Mul, std::move(a), std::move(b));
  }
  auto result = b->clone();
  for (int64_t i = 1; i < factor; ++i) {
    result = expr_add(std::move(result), b->clone());
  }
  return result;
}

std::unique_ptr<SizeExpr> build_value_expr(Stmt *stmt, IRNode *root, int32_t *var_id_counter);
std::unique_ptr<SizeExpr> resolve_range_for_begin(RangeForStmt *range_for, IRNode *root, int32_t *var_id_counter);
std::unique_ptr<SizeExpr> resolve_global_tmp_value(std::size_t offset, IRNode *root, int32_t *var_id_counter);

// Resolve the value stored into the kernel's global-temporary buffer at `offset` by walking the whole kernel
// IR for a matching `GlobalStoreStmt`. `offload::PromoteIntermediateToGlobalTmp` emits exactly one such store
// per offset (the prep serial task's body); a dynamic range-for bound (`for i in range(arr.shape[0])`) and
// the corresponding read-back from the consuming `range_for` task both route through the same offset. This
// pre-pass runs on the full kernel IR *before* `KernelCodeGen::compile_kernel_to_module` splits offload
// tasks into separate blocks, so the prep task's store is visible alongside the consuming task's load.
// Returns nullptr if the store is missing or ambiguous (either indicates running this pre-pass after the
// per-task split, which is a misconfiguration).
std::unique_ptr<SizeExpr> resolve_global_tmp_value(std::size_t offset, IRNode *root, int32_t *var_id_counter) {
  auto stores = irpass::analysis::gather_statements(root, [&](Stmt *s) {
    auto *st = s->cast<GlobalStoreStmt>();
    if (st == nullptr) {
      return false;
    }
    auto *dest_gt = st->dest ? st->dest->cast<GlobalTemporaryStmt>() : nullptr;
    return dest_gt != nullptr && dest_gt->offset == offset;
  });
  if (stores.size() != 1) {
    return nullptr;
  }
  return build_value_expr(stores[0]->as<GlobalStoreStmt>()->val, root, var_id_counter);
}

// Resolve a `RangeForStmt`'s `end` as a `SizeExpr`, applying the CPU-chunk-wrapper unwrap when the range-for is
// the body-level child of an `OffloadedStmt`. `make_cpu_multithreaded_range_for` rewrites the original user
// body into such a wrapper whose end is `min(end_stmt, block_begin + block_range)`; the right-hand `min`
// operand uses `bit_shl`/`add` arithmetic that is deliberately outside the `SizeExpr` grammar, so the generic
// `min->max` widening in `build_value_expr` falls out of the grammar. The logical bound of a chunk-wrapper
// loop index is nonetheless just `end_stmt` (across every thread's chunk the union of iterations covers
// `[begin, end_stmt)`), which is what the user-written enclosing range-for expressed; returning `end_stmt`
// here keeps every call site (`LoopIndexStmt` handler, `SNodeLookupStmt` / `ExternalPtrStmt` pending-wrap
// resolvers, and `compute_bounded_adstack_size`'s outer trip-count multiplier) consistent.
// Thin wrapper over `build_value_expr(range_for->end, ...)` kept as a named helper so every consumer
// (`LoopIndexStmt` handler, `SNodeLookupStmt` / `ExternalPtrStmt` pending-wrap resolvers,
// `compute_bounded_adstack_size`'s outer trip-count multiplier, `resolve_loop_end`) shares one call shape
// with `resolve_range_for_begin`. `determine_ad_stack_size` runs in `compile_to_offloads` before
// `make_cpu_multithreaded_range_for`, so the user's original `RangeForStmt` bounds are still visible here
// and the chunk wrapper's `min(end_stmt, bit_shl-add)` shape never appears in this pre-pass's input IR.
std::unique_ptr<SizeExpr> resolve_range_for_end(RangeForStmt *range_for, IRNode *root, int32_t *var_id_counter) {
  return build_value_expr(range_for->end, root, var_id_counter);
}

// Resolve the begin/end of an arbitrary loop statement (`RangeForStmt` user range-for or `OffloadedStmt`
// parallel-for) as a `SizeExpr`. For offloads with a dynamic end the value lives in the kernel's global
// temporary buffer at `offload->end_offset`, populated by the prep serial task via
// `offload::PromoteIntermediateToGlobalTmp`; this pre-pass runs on the whole kernel IR before the per-task
// split so that store is visible from any consuming task. Callers use these helpers wherever they need to
// wrap a `BoundVariable` for a `LoopIndexStmt` whose parent may be either kind of loop (e.g. the
// `ExternalPtrStmt` index chase that backs the `ExternalTensorRead` SizeExpr kind).
// Resolve a LOWER bound for a loop's index domain, suitable as the `begin` operand of a `MaxOverRange` wrapper
// enumerating every possible runtime value of a stashed loop index. This differs from `resolve_loop_begin`,
// which returns an UPPER bound (the `SizeExpr` of `range_for->begin`, which for a stashed begin is a load-top
// whose value is the max of push values). Using the upper bound there collapses ranges like
// `range(i_outer, N)` to empty when the stash's upper bound equals `N`. Only a literal constant begin can be
// tightened safely; any other shape widens to `0`, which is the implicit lower bound of every non-negative
// `SizeExpr` leaf. Callers that need the exact value of the loop's begin (not a bound) still use
// `resolve_loop_begin`.
std::unique_ptr<SizeExpr> resolve_loop_begin_lower_bound(Stmt *loop) {
  if (auto *rf = loop->cast<RangeForStmt>()) {
    if (rf->begin != nullptr) {
      if (auto *cs = rf->begin->cast<ConstStmt>()) {
        const auto &dt = cs->ret_type;
        if (dt->is_primitive(PrimitiveTypeID::i32)) {
          return SizeExpr::make_const((int64_t)cs->val.val_i32);
        }
        if (dt->is_primitive(PrimitiveTypeID::i64)) {
          return SizeExpr::make_const(cs->val.val_i64);
        }
      }
    }
    return SizeExpr::make_const(0);
  }
  if (auto *off = loop->cast<OffloadedStmt>()) {
    if (off->task_type == OffloadedTaskType::struct_for) {
      return SizeExpr::make_const(0);
    }
    if (off->const_begin) {
      return SizeExpr::make_const(off->begin_value);
    }
    return SizeExpr::make_const(0);
  }
  return SizeExpr::make_const(0);
}

std::unique_ptr<SizeExpr> resolve_loop_end(Stmt *loop, IRNode *root, int32_t *var_id_counter) {
  if (auto *rf = loop->cast<RangeForStmt>()) {
    return resolve_range_for_end(rf, root, var_id_counter);
  }
  if (auto *off = loop->cast<OffloadedStmt>()) {
    if (off->task_type == OffloadedTaskType::struct_for && off->snode != nullptr) {
      // `for i in some_field` lowers to `OffloadedStmt(task_type=struct_for, snode=<leaf>)` whose LoopIndex
      // ranges over the snode's flat cell count. `offload->const_{begin,end}` / `end_stmt` are not populated
      // for struct_for (the iteration space is the snode tree, not a `[begin, end)` interval), so the
      // range-for path above is not reachable; use the leaf's flat size instead. For a dense-only chain the
      // total is `product over axis in [0, num_active_indices) of shape_along_axis(axis)`. Any axis with
      // `dim <= 0` (non-dense or unknown) disqualifies the path; in that case we fall through and leave the
      // alloca unresolved so the pre-pass can report the gap precisely rather than emit a bogus bound.
      int64_t total = 1;
      bool ok = off->snode->num_active_indices > 0;
      for (int axis = 0; axis < off->snode->num_active_indices; ++axis) {
        int dim = off->snode->shape_along_axis(axis);
        if (dim <= 0) {
          ok = false;
          break;
        }
        total *= dim;
      }
      if (ok && total > 0) {
        return SizeExpr::make_const(total);
      }
    }
    if (off->const_end) {
      return SizeExpr::make_const(off->end_value);
    }
    if (off->end_stmt != nullptr) {
      return build_value_expr(off->end_stmt, root, var_id_counter);
    }
    return resolve_global_tmp_value(off->end_offset, root, var_id_counter);
  }
  return nullptr;
}

// Companion to `resolve_range_for_end`; same rationale.
std::unique_ptr<SizeExpr> resolve_range_for_begin(RangeForStmt *range_for, IRNode *root, int32_t *var_id_counter) {
  return build_value_expr(range_for->begin, root, var_id_counter);
}

// Attempt to build a `SizeExpr` representing the runtime value of `stmt` using only the strict-minimum grammar
// plus `LoopIndexStmt` / `AdStackLoadTopStmt` extensions. Returns nullptr on unsupported shape. `root` is the
// kernel-IR root handed to the structural pre-pass; it is needed so the handlers for `AdStackLoadTopStmt` can
// gather all push sites into the same stack and take the upper-bound max over their captured values.
// `var_id_counter` hands out unique ids for `BoundVariable` / `MaxOverRange` pairs introduced when a field-load
// index turns out to reference an enclosing loop index (a shape widely used by hierarchical-array kernels).
//
// Some kernels have IR shapes where the flow of `LocalLoadStmt` stores and `AdStackPushStmt` push-values forms a
// cycle (e.g. stack A holds a flat loop index and a later push pushes `sub(load_top(A), load_top(A))` - i.e.
// the mod pattern `flat - (flat // N) * N` that `qd.ndrange(M, N)` lowers into). The Bellman-Ford phase is
// CFG-local so it would not cross these references; this structural pass walks stores and pushes symbolically,
// which means without a guard it recurses unboundedly through the data-flow cycle and overflows the host
// stack. `t_on_path` is the set of `Stmt*`s currently on the recursion stack of `build_value_expr`; revisiting
// a stmt that is already on the path sets the thread-local `t_cycle_detected` flag and returns `nullptr`.
//
// The `AdStackLoadTopStmt` handler then distinguishes two outcomes: (a) the cycle is on an already-bounded
// stash whose cyclic push, when evaluated with its own `load_top` substituted by a constant zero, reduces to
// a constant zero - the idempotent-at-zero pattern, which by induction is dominated by the non-cyclic pushes'
// max (the mod case: `sub(0, ...) = 0`, Sub's `max(a-b, 0)` clamp pushes cyclic contributions to 0 at the
// base case), so we soundly return the non-cyclic max as the bound; (b) the cyclic push's idempotency cannot
// be proven, in which case we propagate `nullptr` and the driver raises a "stash data-flow cycle" error.
// `t_loadtop_subst_zero` is the set of stacks for which the current walk should short-circuit
// `load_top(stack)` to `const(0)`, used to implement the idempotency probe without mutating the IR.
thread_local std::unordered_set<Stmt *> t_on_path;
thread_local bool t_cycle_detected = false;
thread_local std::unordered_set<AdStackAllocaStmt *> t_loadtop_subst_zero;
struct OnPathGuard {
  Stmt *s;
  bool inserted;
  explicit OnPathGuard(Stmt *stmt) : s(stmt), inserted(t_on_path.insert(stmt).second) {
  }
  ~OnPathGuard() {
    if (inserted) {
      t_on_path.erase(s);
    }
  }
};

std::unique_ptr<SizeExpr> build_value_expr(Stmt *stmt, IRNode *root, int32_t *var_id_counter) {
  if (stmt == nullptr) {
    return nullptr;
  }
  // Honour the idempotency-probe substitution FIRST, before the visited-set cycle guard. During phase 2 of a
  // cyclic `AdStackLoadTopStmt` resolution we're re-entering the same sub-expression that triggered the cycle
  // in phase 1; the prior `OnPathGuard` frames for the BinaryOp / operand chain are still live on `t_on_path`
  // because the outer handler hasn't unwound. Treating the substitution as "cycle-break at the leaf" rather
  // than a cycle lets the probe evaluate `Sub(load_top(self), load_top(self))` to `Sub(0, 0) = 0` instead of
  // bailing with `nullptr` the second we re-enter the cyclic BinaryOp.
  if (auto *load_top_early = stmt->cast<AdStackLoadTopStmt>()) {
    if (load_top_early->stack != nullptr) {
      if (auto *stack_alloca_early = load_top_early->stack->cast<AdStackAllocaStmt>()) {
        if (t_loadtop_subst_zero.count(stack_alloca_early) != 0) {
          return SizeExpr::make_const(0);
        }
      }
    }
  }
  if (t_on_path.count(stmt) != 0) {
    t_cycle_detected = true;
    return nullptr;
  }
  OnPathGuard path_guard(stmt);
  int64_t const_val = 0;
  if (try_eval_const_int(stmt, &const_val)) {
    return SizeExpr::make_const(const_val);
  }
  if (auto *load = stmt->cast<GlobalLoadStmt>()) {
    if (load->src == nullptr) {
      return nullptr;
    }
    const auto &dt = load->ret_type;
    if (!(dt->is_primitive(PrimitiveTypeID::i32) || dt->is_primitive(PrimitiveTypeID::i64))) {
      return nullptr;
    }
    // Two `GlobalLoadStmt` source shapes matter: before `lower_access` the source is a `GlobalPtrStmt` that carries
    // the snode and the raw index stmts; after `lower_access` it becomes a `GetChStmt` rooted in a chain of
    // `SNodeLookupStmt` / `GetChStmt` / `GetRootStmt`. `determine_ad_stack_size` runs after `lower_access` in the
    // normal pipeline (the structural pre-pass used to accept only the pre-lowering shape and therefore gave up on
    // every scalar field load), so we walk both forms here.
    if (auto *ptr = load->src->cast<GlobalPtrStmt>()) {
      if (ptr->snode == nullptr) {
        return nullptr;
      }
      // Walk each index through the same chase used by the `ExternalPtrStmt` branch: constant-fold first, then
      // peel casts and single-store local-alloca spills to find the underlying `LoopIndexStmt`, and finally chase
      // through an `AdStackLoadTopStmt` whose sole non-const push value is a `LoopIndexStmt` (the autodiff-
      // stashed loop variable pattern). Indices resolved to a loop index are recorded as `BoundVariable`
      // placeholders; opaque shapes that can't be chased but still index into this field are conservatively
      // bounded by the snode's shape along that axis (every valid index is in `[0, shape_along_axis)`).
      std::vector<int64_t> idx_values;
      idx_values.reserve(ptr->indices.size());
      struct PendingWrap {
        int32_t var_id;
        Stmt *loop;  // non-null: iterate [loop.begin, loop.end); null: use snode shape_along_axis.
        int32_t snode_axis;
      };
      std::vector<PendingWrap> pending_wraps;
      int32_t axis_pos = 0;
      for (Stmt *idx_stmt : ptr->indices) {
        int64_t v = 0;
        if (try_eval_const_int(idx_stmt, &v)) {
          idx_values.push_back(v);
          ++axis_pos;
          continue;
        }
        Stmt *idx_expr = idx_stmt;
        while (idx_expr != nullptr) {
          if (auto *u = idx_expr->cast<UnaryOpStmt>()) {
            if (unary_op_is_cast(u->op_type) && u->operand != nullptr && is_integral(u->ret_type) &&
                is_integral(u->operand->ret_type)) {
              idx_expr = u->operand;
              continue;
            }
            break;
          }
          if (auto *ll = idx_expr->cast<LocalLoadStmt>()) {
            auto *alloca_dst = ll->src ? ll->src->cast<AllocaStmt>() : nullptr;
            if (alloca_dst == nullptr) {
              break;
            }
            auto stores = irpass::analysis::gather_statements(root, [&](Stmt *s) {
              auto *store = s->cast<LocalStoreStmt>();
              return store != nullptr && store->dest == alloca_dst;
            });
            if (stores.size() != 1) {
              break;
            }
            idx_expr = stores[0]->as<LocalStoreStmt>()->val;
            continue;
          }
          if (auto *lt = idx_expr->cast<AdStackLoadTopStmt>()) {
            // Stash chase: mirrors the `ExternalPtrStmt` branch. The stash's push values must resolve (modulo
            // const-zero initialisers) to a single `LoopIndexStmt` for us to pick up a tight range; otherwise
            // we fall through to the shape-along-axis guard below.
            auto *stack = lt->stack ? lt->stack->cast<AdStackAllocaStmt>() : nullptr;
            if (stack == nullptr) {
              break;
            }
            auto pushes = irpass::analysis::gather_statements(root, [&](Stmt *s) {
              auto *push = s->cast<AdStackPushStmt>();
              return push != nullptr && push->stack == stack;
            });
            Stmt *loop_idx_push_val = nullptr;
            for (Stmt *p : pushes) {
              Stmt *pv = p->as<AdStackPushStmt>()->v;
              int64_t c = 0;
              if (try_eval_const_int(pv, &c)) {
                continue;
              }
              while (pv != nullptr) {
                if (auto *u2 = pv->cast<UnaryOpStmt>()) {
                  if (unary_op_is_cast(u2->op_type) && u2->operand != nullptr && is_integral(u2->ret_type) &&
                      is_integral(u2->operand->ret_type)) {
                    pv = u2->operand;
                    continue;
                  }
                  break;
                }
                if (auto *ll2 = pv->cast<LocalLoadStmt>()) {
                  auto *dst = ll2->src ? ll2->src->cast<AllocaStmt>() : nullptr;
                  if (dst == nullptr) {
                    break;
                  }
                  auto sts = irpass::analysis::gather_statements(root, [&](Stmt *s) {
                    auto *st = s->cast<LocalStoreStmt>();
                    return st != nullptr && st->dest == dst;
                  });
                  if (sts.size() != 1) {
                    break;
                  }
                  pv = sts[0]->as<LocalStoreStmt>()->val;
                  continue;
                }
                break;
              }
              if (pv != nullptr && pv->is<LoopIndexStmt>()) {
                if (loop_idx_push_val != nullptr && loop_idx_push_val != pv) {
                  loop_idx_push_val = nullptr;
                  break;
                }
                loop_idx_push_val = pv;
              }
            }
            if (loop_idx_push_val == nullptr) {
              break;
            }
            idx_expr = loop_idx_push_val;
            continue;
          }
          break;
        }
        int32_t var_id = (*var_id_counter)++;
        idx_values.push_back(-static_cast<int64_t>(var_id + 1));
        auto *loop_idx = idx_expr ? idx_expr->cast<LoopIndexStmt>() : nullptr;
        if (loop_idx != nullptr && loop_idx->loop != nullptr) {
          pending_wraps.push_back({var_id, loop_idx->loop, axis_pos});
        } else {
          // Opaque index shape. A valid field read requires `0 <= idx < snode.shape_along_axis(axis)`, so the
          // snode's shape is a safe upper bound for the index range. Any axis that reports a non-positive shape
          // (non-dense or unknown) falls out here by returning nullptr below, because we refuse to emit a bogus
          // range that would skip iterations at launch time.
          pending_wraps.push_back({var_id, nullptr, axis_pos});
        }
        ++axis_pos;
      }
      std::unique_ptr<SizeExpr> result = SizeExpr::make_field_load(ptr->snode, std::move(idx_values));
      for (auto &wrap : pending_wraps) {
        std::unique_ptr<SizeExpr> begin_e;
        std::unique_ptr<SizeExpr> end_e;
        if (wrap.loop != nullptr) {
          begin_e = resolve_loop_begin_lower_bound(wrap.loop);
          end_e = resolve_loop_end(wrap.loop, root, var_id_counter);
        } else {
          int dim = ptr->snode->shape_along_axis(wrap.snode_axis);
          if (dim <= 0) {
            return nullptr;
          }
          begin_e = SizeExpr::make_const(0);
          end_e = SizeExpr::make_const(dim);
        }
        if (!begin_e || !end_e) {
          return nullptr;
        }
        result = SizeExpr::make_max_over_range(wrap.var_id, std::move(begin_e), std::move(end_e), std::move(result));
      }
      return result;
    }
    if (auto *leaf_child = load->src->cast<GetChStmt>()) {
      // Walk the `GetChStmt(SNodeLookupStmt(..., index)(GetChStmt(...(GetRootStmt))))` chain back to the root,
      // collecting an index value per `SNodeLookupStmt` level. Each index is either a constant int (stored as-is)
      // or a `LoopIndexStmt` of an enclosing `RangeForStmt` (recorded as a `BoundVariable` reference and wrapped
      // in a `MaxOverRange` at return time, so the host evaluator can take the max over the outer loop's range).
      // Any other shape on the chain is unsupported and the pre-pass treats the bound as unresolved.
      SNode *leaf_snode = leaf_child->output_snode;
      if (leaf_snode == nullptr) {
        return nullptr;
      }
      std::vector<int64_t> collected;
      // Pending range-max wrappers, outer-first so they get applied in reverse (innermost wrap first). Each entry
      // is (var_id, range_for_stmt); `var_id` is the `BoundVariable` placeholder baked into `collected` above.
      std::vector<std::pair<int32_t, RangeForStmt *>> pending_wraps;
      Stmt *cur = leaf_child->input_ptr;
      while (cur != nullptr) {
        if (auto *lookup = cur->cast<SNodeLookupStmt>()) {
          int64_t v = 0;
          if (try_eval_const_int(lookup->input_index, &v)) {
            collected.push_back(v);
            cur = lookup->input_snode;
            continue;
          }
          // Chase through `Cast` and `LocalLoad` to find the defining `LoopIndexStmt`. Frontends that wrap each
          // loop index with an explicit integer cast and a local-alloca-spill bury the raw `LoopIndexStmt`
          // several hops behind the `SNodeLookupStmt::input_index`.
          Stmt *idx_expr = lookup->input_index;
          while (idx_expr != nullptr) {
            if (auto *u = idx_expr->cast<UnaryOpStmt>()) {
              if (unary_op_is_cast(u->op_type) && u->operand != nullptr && is_integral(u->ret_type) &&
                  is_integral(u->operand->ret_type)) {
                idx_expr = u->operand;
                continue;
              }
              break;
            }
            if (auto *ll = idx_expr->cast<LocalLoadStmt>()) {
              auto *alloca_dst = ll->src ? ll->src->cast<AllocaStmt>() : nullptr;
              if (alloca_dst == nullptr) {
                break;
              }
              auto stores = irpass::analysis::gather_statements(root, [&](Stmt *s) {
                auto *store = s->cast<LocalStoreStmt>();
                return store != nullptr && store->dest == alloca_dst;
              });
              if (stores.size() != 1) {
                break;  // Multi-store locals are not a loop-index alias; give up.
              }
              idx_expr = stores[0]->as<LocalStoreStmt>()->val;
              continue;
            }
            break;
          }
          if (auto *loop_idx = idx_expr ? idx_expr->cast<LoopIndexStmt>() : nullptr) {
            if (auto *range_for = loop_idx->loop ? loop_idx->loop->cast<RangeForStmt>() : nullptr) {
              int32_t var_id = (*var_id_counter)++;
              collected.push_back(-static_cast<int64_t>(var_id + 1));
              pending_wraps.emplace_back(var_id, range_for);
              cur = lookup->input_snode;
              continue;
            }
          }
          return nullptr;
        }
        if (auto *next_ch = cur->cast<GetChStmt>()) {
          cur = next_ch->input_ptr;
          continue;
        }
        if (cur->is<GetRootStmt>()) {
          break;
        }
        return nullptr;
      }
      // `SNodeRwAccessorsBank::Accessors::read_int` expects exactly `leaf_snode->num_active_indices` integer
      // indices in tree-walk order. The collected list above contains one entry per `SNodeLookupStmt` from the
      // leaf upwards (root lookup last); trim to the active-indices count and reverse to hand the evaluator a
      // shape that matches the reader-kernel contract.
      std::reverse(collected.begin(), collected.end());
      int n_active = leaf_snode->num_active_indices;
      if (static_cast<int>(collected.size()) < n_active) {
        return nullptr;
      }
      std::vector<int64_t> idx_values(collected.end() - n_active, collected.end());
      // Only wrap with `MaxOverRange` for loop variables that actually appear in the retained `idx_values` - the
      // trimmed-away indices (root / intermediate lookups that are always 0 for a scalar shape) do not need their
      // outer loop enumerated.
      std::vector<bool> used_var_id;
      for (int64_t v : idx_values) {
        if (v < 0) {
          int32_t var_id = static_cast<int32_t>(-(v + 1));
          if (static_cast<std::size_t>(var_id) >= used_var_id.size()) {
            used_var_id.resize(var_id + 1, false);
          }
          used_var_id[var_id] = true;
        }
      }
      std::unique_ptr<SizeExpr> result = SizeExpr::make_field_load(leaf_snode, std::move(idx_values));
      // Wrap innermost-first so a nested field-load inside another expression still composes correctly: the
      // outer-most wrap ends up at the root of the returned tree, so each `BoundVariable` is bound by the
      // enclosing iteration during evaluation.
      for (auto &[var_id, range_for] : pending_wraps) {
        if (static_cast<std::size_t>(var_id) >= used_var_id.size() || !used_var_id[var_id]) {
          continue;
        }
        auto begin_e = resolve_range_for_begin(range_for, root, var_id_counter);
        auto end_e = resolve_range_for_end(range_for, root, var_id_counter);
        if (!begin_e || !end_e) {
          return nullptr;
        }
        result = SizeExpr::make_max_over_range(var_id, std::move(begin_e), std::move(end_e), std::move(result));
      }
      return result;
    }
    if (auto *ext_ptr = load->src->cast<ExternalPtrStmt>()) {
      // Ndarray-argument element read: `ndarray[indices...]`. `base_ptr` points at the `ArgLoadStmt` that carries
      // the argument id. Indices are integer `Stmt *`s that may be `ConstStmt`s or (after chase-through of
      // `Cast` / `LocalLoad`) `LoopIndexStmt`s of enclosing range-fors; the latter get encoded as
      // `BoundVariable` references and wrapped in `MaxOverRange` so the launch-time evaluator can take the max
      // across all iterations.
      auto *arg_load = ext_ptr->base_ptr ? ext_ptr->base_ptr->cast<ArgLoadStmt>() : nullptr;
      if (arg_load == nullptr) {
        return nullptr;
      }
      std::vector<int32_t> arg_id_path(arg_load->arg_id.begin(), arg_load->arg_id.end());
      std::vector<int64_t> idx_values;
      // Pending `MaxOverRange` wraps for each non-const index. `loop != nullptr` means the chase below resolved the
      // index to a `LoopIndexStmt` and we use that loop's `[begin, end)` as the tight iteration range. `loop ==
      // nullptr` falls back to `[0, shape_along_axis(ndarray_axis))`, which is a safe upper bound for any valid
      // ndarray index regardless of how it was computed (compound arithmetic, nested stash loads, whatever the
      // frontend emitted).
      struct PendingWrap {
        int32_t var_id;
        Stmt *loop;
        int32_t ndarray_axis;
      };
      std::vector<PendingWrap> pending_wraps;
      int32_t axis_pos = 0;
      for (Stmt *idx_stmt : ext_ptr->indices) {
        int64_t v = 0;
        if (try_eval_const_int(idx_stmt, &v)) {
          idx_values.push_back(v);
          ++axis_pos;
          continue;
        }
        Stmt *idx_expr = idx_stmt;
        while (idx_expr != nullptr) {
          if (auto *u = idx_expr->cast<UnaryOpStmt>()) {
            if (unary_op_is_cast(u->op_type) && u->operand != nullptr && is_integral(u->ret_type) &&
                is_integral(u->operand->ret_type)) {
              idx_expr = u->operand;
              continue;
            }
            break;
          }
          if (auto *ll = idx_expr->cast<LocalLoadStmt>()) {
            auto *alloca_dst = ll->src ? ll->src->cast<AllocaStmt>() : nullptr;
            if (alloca_dst == nullptr) {
              break;
            }
            auto stores = irpass::analysis::gather_statements(root, [&](Stmt *s) {
              auto *store = s->cast<LocalStoreStmt>();
              return store != nullptr && store->dest == alloca_dst;
            });
            if (stores.size() != 1) {
              break;
            }
            idx_expr = stores[0]->as<LocalStoreStmt>()->val;
            continue;
          }
          if (auto *lt = idx_expr->cast<AdStackLoadTopStmt>()) {
            // Stash-and-reload pattern: the autodiff pipeline pushes a loop variable (typically the cast of an
            // enclosing RangeFor's loop index) onto a dedicated adstack so the reverse pass can reconstruct it.
            // Every downstream ndarray read `arr[i_l]` then lowers to `ExternalPtrStmt(arr, [load_top(stash)])`.
            // Chase through the stash to the backing LoopIndex by finding the only push whose value resolves
            // (through cast / local-load of a single-store) to a LoopIndexStmt; pushes whose value folds to a
            // constant (e.g. the codegen-emitted zero-initialiser) are ignored because they contribute a
            // strictly smaller upper bound. A stack with no loop-index-backed push, or with multiple such
            // pushes sourced from different loops, is intentionally rejected - handling those cleanly means
            // emitting a `Max` across per-push sub-expressions and is left for a future grammar extension.
            auto *stack = lt->stack ? lt->stack->cast<AdStackAllocaStmt>() : nullptr;
            if (stack == nullptr) {
              break;
            }
            auto pushes = irpass::analysis::gather_statements(root, [&](Stmt *s) {
              auto *push = s->cast<AdStackPushStmt>();
              return push != nullptr && push->stack == stack;
            });
            Stmt *loop_idx_push_val = nullptr;
            for (Stmt *p : pushes) {
              Stmt *pv = p->as<AdStackPushStmt>()->v;
              int64_t c = 0;
              if (try_eval_const_int(pv, &c)) {
                continue;
              }
              while (pv != nullptr) {
                if (auto *u2 = pv->cast<UnaryOpStmt>()) {
                  if (unary_op_is_cast(u2->op_type) && u2->operand != nullptr && is_integral(u2->ret_type) &&
                      is_integral(u2->operand->ret_type)) {
                    pv = u2->operand;
                    continue;
                  }
                  break;
                }
                if (auto *ll2 = pv->cast<LocalLoadStmt>()) {
                  auto *dst = ll2->src ? ll2->src->cast<AllocaStmt>() : nullptr;
                  if (dst == nullptr) {
                    break;
                  }
                  auto sts = irpass::analysis::gather_statements(root, [&](Stmt *s) {
                    auto *st = s->cast<LocalStoreStmt>();
                    return st != nullptr && st->dest == dst;
                  });
                  if (sts.size() != 1) {
                    break;
                  }
                  pv = sts[0]->as<LocalStoreStmt>()->val;
                  continue;
                }
                break;
              }
              if (pv != nullptr && pv->is<LoopIndexStmt>()) {
                if (loop_idx_push_val != nullptr && loop_idx_push_val != pv) {
                  loop_idx_push_val = nullptr;
                  break;
                }
                loop_idx_push_val = pv;
              }
            }
            if (loop_idx_push_val == nullptr) {
              break;
            }
            idx_expr = loop_idx_push_val;
            continue;
          }
          break;
        }
        int32_t var_id = (*var_id_counter)++;
        idx_values.push_back(-static_cast<int64_t>(var_id + 1));
        auto *loop_idx = idx_expr ? idx_expr->cast<LoopIndexStmt>() : nullptr;
        if (loop_idx != nullptr && loop_idx->loop != nullptr) {
          pending_wraps.push_back({var_id, loop_idx->loop, axis_pos});
        } else {
          // Opaque index shape (e.g. stash `load_top` whose push value is a compound arithmetic expression). A
          // valid ndarray read requires `0 <= idx < arr.shape[axis]`, so using that shape as the iteration range
          // is a safe conservative upper bound that keeps the rest of the enclosing loop chain bounded without
          // refusing the whole alloca.
          pending_wraps.push_back({var_id, nullptr, axis_pos});
        }
        ++axis_pos;
      }
      int64_t prim_dt = static_cast<int64_t>(load->ret_type->cast<PrimitiveType>()->type);
      std::unique_ptr<SizeExpr> result = SizeExpr::make_external_tensor_read(arg_id_path, idx_values, prim_dt);
      for (auto &wrap : pending_wraps) {
        std::unique_ptr<SizeExpr> begin_e;
        std::unique_ptr<SizeExpr> end_e;
        if (wrap.loop != nullptr) {
          begin_e = resolve_loop_begin_lower_bound(wrap.loop);
          end_e = resolve_loop_end(wrap.loop, root, var_id_counter);
        } else {
          begin_e = SizeExpr::make_const(0);
          end_e = SizeExpr::make_external_tensor_shape(arg_id_path, wrap.ndarray_axis);
        }
        if (!begin_e || !end_e) {
          return nullptr;
        }
        result = SizeExpr::make_max_over_range(wrap.var_id, std::move(begin_e), std::move(end_e), std::move(result));
      }
      return result;
    }
    if (auto *gt = load->src->cast<GlobalTemporaryStmt>()) {
      return resolve_global_tmp_value(gt->offset, root, var_id_counter);
    }
    return nullptr;
  }
  if (auto *shape = stmt->cast<ExternalTensorShapeAlongAxisStmt>()) {
    // Ndarray-argument shape resolved per-launch from `LaunchContextBuilder::get_struct_arg<i64>(arg_id + [
    // SHAPE_POS_IN_NDARRAY, axis])`. A range-for bounded by `arr.shape[axis]` of an ndarray kernel argument
    // lowers to this stmt; without it the pre-pass would fall out of the grammar for every ndarray-driven
    // parallel-for.
    std::vector<int32_t> path;
    path.reserve(shape->arg_id.size());
    for (int v : shape->arg_id) {
      path.push_back(v);
    }
    return SizeExpr::make_external_tensor_shape(std::move(path), static_cast<int32_t>(shape->axis));
  }
  if (auto *unary = stmt->cast<UnaryOpStmt>()) {
    // Integer-to-integer casts (frontends routinely wrap loop indices with an explicit integer cast) are
    // value-preserving for the small integer ranges that bound adstack depth, so we can pass the operand's
    // `SizeExpr` through unchanged. Non-int casts or non-cast unary ops are rejected.
    if (unary_op_is_cast(unary->op_type) && is_integral(unary->ret_type) && unary->operand != nullptr &&
        is_integral(unary->operand->ret_type)) {
      return build_value_expr(unary->operand, root, var_id_counter);
    }
    return nullptr;
  }
  if (auto *local_load = stmt->cast<LocalLoadStmt>()) {
    // Widely-used frontend pattern: a user writes `i = int(i_raw)` (or any loop-index alias), which lowers
    // into an `AllocaStmt` with one or more `LocalStoreStmt`s and every later reference is a `LocalLoadStmt`
    // off that alloca. The load's value at any point in the kernel is upper-bounded by the max of every stored
    // value; build a `SizeExpr` for each store's `val` and fold them with `Max`. Alloca with no stores is
    // treated as zero (a safe upper bound on max-over-an-empty-set).
    auto *alloca_dst = local_load->src ? local_load->src->cast<AllocaStmt>() : nullptr;
    if (alloca_dst == nullptr) {
      return nullptr;
    }
    auto stores = irpass::analysis::gather_statements(root, [&](Stmt *s) {
      auto *store = s->cast<LocalStoreStmt>();
      return store != nullptr && store->dest == alloca_dst;
    });
    std::unique_ptr<SizeExpr> max_expr;
    for (Stmt *s : stores) {
      auto *store = s->as<LocalStoreStmt>();
      auto v = build_value_expr(store->val, root, var_id_counter);
      if (!v) {
        return nullptr;
      }
      if (!max_expr) {
        max_expr = std::move(v);
      } else {
        max_expr = expr_max(std::move(max_expr), std::move(v));
      }
    }
    if (!max_expr) {
      return SizeExpr::make_const(0);
    }
    return max_expr;
  }
  if (auto *idx = stmt->cast<LoopIndexStmt>()) {
    // A loop index at any point in its lifetime is strictly less than the loop's `end`, so `end` is a safe (if
    // usually off-by-one-loose) upper bound. This lets reverse-mode kernels whose inner `range(j)` picks up `j`
    // from an enclosing `for j in range(N)` resolve to `N` rather than falling out of the grammar. Handles both
    // inner `RangeForStmt`s and the outer `OffloadedStmt`'s parallel-for variant (which is where the outer
    // parallel-for index of a root kernel comes from).
    if (auto *range_for = idx->loop->cast<RangeForStmt>()) {
      return resolve_range_for_end(range_for, root, var_id_counter);
    }
    if (auto *offload = idx->loop->cast<OffloadedStmt>()) {
      return resolve_loop_end(offload, root, var_id_counter);
    }
    return nullptr;
  }
  if (auto *load_top = stmt->cast<AdStackLoadTopStmt>()) {
    (void)0;  // no-op
    // The top of a stack at any moment is at most `max` of every value ever pushed onto it. Gather the pushes
    // into `load_top->stack`, build a `SizeExpr` for each push's `val`, and fold them with `Max`. If any push's
    // value is outside the grammar, the whole bound is considered unresolved.
    auto *stack_alloca = load_top->stack ? load_top->stack->cast<AdStackAllocaStmt>() : nullptr;
    if (stack_alloca == nullptr) {
      return nullptr;
    }
    // Idempotency probe: outer caller has substituted `load_top(this stack) = 0` while checking that a cyclic
    // push reduces to const zero at the base case. Short-circuit so the recursion terminates without re-
    // entering the cycle.
    if (t_loadtop_subst_zero.count(stack_alloca) != 0) {
      return SizeExpr::make_const(0);
    }
    auto pushes = irpass::analysis::gather_statements(root, [&](Stmt *s) {
      auto *push = s->cast<AdStackPushStmt>();
      return push != nullptr && push->stack == stack_alloca;
    });
    std::unique_ptr<SizeExpr> max_expr;
    std::vector<AdStackPushStmt *> cyclic_pushes;
    for (Stmt *p : pushes) {
      auto *push = p->as<AdStackPushStmt>();
      bool saved_cycle = t_cycle_detected;
      t_cycle_detected = false;
      auto v = build_value_expr(push->v, root, var_id_counter);
      bool this_push_cyclic = t_cycle_detected;
      // Preserve the outer cycle flag so unrelated cycles propagate, but clear the per-push flag once we've
      // decided how to treat this push. Cyclic pushes are held for the phase-2 idempotency probe below.
      t_cycle_detected = saved_cycle;
      if (this_push_cyclic) {
        cyclic_pushes.push_back(push);
        continue;
      }
      if (!v) {
        return nullptr;  // genuine grammar gap (non-cyclic): whole bound is unresolved.
      }
      if (!max_expr) {
        max_expr = std::move(v);
      } else {
        max_expr = expr_max(std::move(max_expr), std::move(v));
      }
    }
    if (cyclic_pushes.empty()) {
      return max_expr;
    }
    if (!max_expr) {
      // Every push referenced `load_top` of this same stack - there is no base case to anchor the recursion
      // so the stack is genuinely unbounded. Surface this to the caller as a cycle for the driver's specific
      // error message.
      t_cycle_detected = true;
      return nullptr;
    }
    // Phase 2 - idempotency probe. For each cyclic push, substitute `load_top(this stack) = 0` and verify the
    // push value reduces to `const(0)`. If it does, the push cannot lift the stack above the non-cyclic max:
    // starting from `top = const(0)` every subsequent cyclic re-push stays at 0, and merging with the non-
    // cyclic branches can only pull `top` up to their max. If any cyclic push fails the probe (e.g. a counter-
    // like `load_top + 1` pattern) we fall through to the cycle error rather than silently under-bound.
    t_loadtop_subst_zero.insert(stack_alloca);
    bool all_idempotent = true;
    for (AdStackPushStmt *push : cyclic_pushes) {
      auto v = build_value_expr(push->v, root, var_id_counter);
      if (v == nullptr || v->kind != SizeExpr::Kind::Const || v->const_value != 0) {
        all_idempotent = false;
        break;
      }
    }
    t_loadtop_subst_zero.erase(stack_alloca);
    if (!all_idempotent) {
      t_cycle_detected = true;
      return nullptr;
    }
    return max_expr;
  }
  if (auto *b = stmt->cast<BinaryOpStmt>()) {
    auto lhs = build_value_expr(b->lhs, root, var_id_counter);
    auto rhs = build_value_expr(b->rhs, root, var_id_counter);
    if (!lhs || !rhs) {
      return nullptr;
    }
    switch (b->op_type) {
      case BinaryOpType::add:
        return expr_add(std::move(lhs), std::move(rhs));
      case BinaryOpType::sub:
        return expr_sub(std::move(lhs), std::move(rhs));
      case BinaryOpType::max:
        return expr_max(std::move(lhs), std::move(rhs));
      case BinaryOpType::min:
        // `min(a, b) <= max(a, b)` - a conservative upper bound that stays inside the grammar.
        return expr_max(std::move(lhs), std::move(rhs));
      case BinaryOpType::mul:
        return expr_mul(std::move(lhs), std::move(rhs));
      case BinaryOpType::bit_shl:
        // `a << k` with a non-negative `a` and a const shift `k >= 0` equals `a * (1 << k)`. `full_simplify` rewrites
        // `a * 2^k` into this shape for small k, so a trip-count like `range(c[i] * 2)` ends up as
        // `bit_shl(load(c[i]), 1)` by the time this pre-pass runs; without this branch the shape falls out of the
        // grammar even though the dynamic `Mul` representation is exactly what we would have emitted for the
        // pre-folded `c[i] * 2` tree.
        if (rhs->kind == SizeExpr::Kind::Const && rhs->const_value >= 0 && rhs->const_value < 63) {
          int64_t factor = int64_t{1} << rhs->const_value;
          return expr_mul(std::move(lhs), SizeExpr::make_const(factor));
        }
        return nullptr;
      case BinaryOpType::bit_shr:
      case BinaryOpType::bit_sar:
        // `a >> k` with a non-negative `a` and `k >= 0` is at most `a`, so it is a safe upper bound inside this
        // non-negative `SizeExpr` grammar. This covers the `ndrange`-lowered mod pattern `(flat // N) * N` whose
        // inner `flat // N` shape reaches here after simplification.
        if (rhs->kind == SizeExpr::Kind::Const && rhs->const_value >= 0) {
          return lhs;
        }
        return nullptr;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

AdStackBoundResult compute_bounded_adstack_size(AdStackAllocaStmt *alloca, IRNode *root) {
  AdStackBoundResult result;
  result.expr = SizeExpr::make_const(0);
  auto push_stmts = irpass::analysis::gather_statements(root, [&](Stmt *s) {
    auto *push = s->cast<AdStackPushStmt>();
    return push != nullptr && push->stack == alloca;
  });
  // Pre-compute the block chain that contains the `AdStackAllocaStmt` itself: walking up from a push site, we must
  // stop *at* the alloca's block before factoring in any further enclosing loops. `AdStackAllocaStmt` lowers into a
  // `stack_init` at visit time that resets the stack's count header to zero; re-entry of a loop that surrounds the
  // alloca therefore sees a fresh stack, not compounded depth from prior iterations. The canonical offender is
  // the CPU multithreaded outer `range_for(thread_idx << k, min(N, thread_idx << k + chunk_size))` that
  // `make_cpu_multithreaded_range_for` emits around the user body: its bounds are unparseable in the `SizeExpr`
  // grammar (`bit_shl` of a `LoopIndexStmt`), and without this stop condition the walker would treat the outer
  // chunk loop as a multiplier the pre-pass cannot prove constant, which would have forced every user-written
  // adstack kernel on CPU to hard-error at compile time purely because of the chunk-loop wrapping.
  std::unordered_set<Block *> alloca_scope_chain;
  for (Block *b = alloca->parent; b != nullptr; b = (b->parent_stmt() ? b->parent_stmt()->parent : nullptr)) {
    alloca_scope_chain.insert(b);
  }
  // Counter for `BoundVariable` / `MaxOverRange` ids; shared across every push-site walk for this alloca so the
  // ids are unique over the whole expression tree this function returns.
  int32_t var_id_counter = 0;
  for (Stmt *s : push_stmts) {
    auto *push = s->as<AdStackPushStmt>();
    auto multiplier = SizeExpr::make_const(1);
    Block *blk = push->parent;
    while (blk != nullptr) {
      // Stop as soon as the walker reaches a block at or above the alloca's scope: the `stack_init` emitted by
      // the alloca resets the count on each entry, so outer loops do not compound per-iteration depth.
      if (alloca_scope_chain.count(blk) != 0) {
        break;
      }
      Stmt *parent = blk->parent_stmt();
      if (parent == nullptr) {
        break;
      }
      if (auto *range_for = parent->cast<RangeForStmt>()) {
        auto end_e = resolve_range_for_end(range_for, root, &var_id_counter);
        if (!end_e) {
          result.bounded = false;
          result.expr = nullptr;
          result.cycle_detected = t_cycle_detected;
          return result;
        }
        // Upper-bound trip = end_upper - begin_lower. Using `resolve_range_for_begin` here returns begin's
        // UPPER bound (the max across stashed push values for a `load_top` begin), which collapses triangular
        // scans like `range(i_outer, N)` to zero when the stash's max push equals `N`. The lower-bound helper
        // tightens only for literal-const begins and falls back to `0` (the implicit lower bound of every
        // non-negative `SizeExpr` leaf) otherwise.
        auto begin_lower = resolve_loop_begin_lower_bound(range_for);
        std::unique_ptr<SizeExpr> trip = expr_sub(std::move(end_e), std::move(begin_lower));
        if (trip->kind == SizeExpr::Kind::Const && trip->const_value < 0) {
          trip = SizeExpr::make_const(0);
        }
        auto new_multiplier = expr_mul(std::move(multiplier), std::move(trip));
        if (!new_multiplier) {
          result.bounded = false;
          result.expr = nullptr;
          return result;
        }
        multiplier = std::move(new_multiplier);
      } else if (parent->is<StructForStmt>() || parent->is<WhileStmt>()) {
        // Struct-for siblings inside an offloaded body or any while-loop are unbounded at
        // compile-time.
        result.bounded = false;
        result.expr = nullptr;
        return result;
      }
      // IfStmt, MeshForStmt-body, and other containers pass through without affecting the
      // multiplier - they do not iterate, so one enclosing execution of them equals one push.
      blk = parent->parent;
    }
    result.expr = expr_add(std::move(result.expr), std::move(multiplier));
  }
  if (result.bounded && push_stmts.empty()) {
    // No pushes reach this alloca but it still exists in IR - match Bellman-Ford's behavior of
    // leaving such stacks alone (the verifier/DCE should have removed them).
    result.expr = SizeExpr::make_const(1);
  }
  return result;
}

// True if the tree references any inner-domain enumeration node (`MaxOverRange` or its `BoundVariable`
// references). Used by the per-task trip-count capture to refuse trees the host evaluator would have to
// walk index-by-index at launch: the trip count of an `OffloadedStmt`'s `[begin, end)` is a scalar bound
// by construction, so a `MaxOverRange` here means `build_value_expr` walked into a nested loop (e.g. an
// `end_stmt` referencing a `LoopIndexStmt` of an outer offload, whose range is the dispatched-thread
// count) and the resulting tree would trip the 1<<24 enumeration guard in `evaluate_adstack_size_expr`.
bool size_expr_contains_inner_domain_enumeration(const SizeExpr *e) {
  if (e == nullptr) {
    return false;
  }
  if (e->kind == SizeExpr::Kind::MaxOverRange || e->kind == SizeExpr::Kind::BoundVariable) {
    return true;
  }
  for (const auto &child : e->operands) {
    if (size_expr_contains_inner_domain_enumeration(child.get())) {
      return true;
    }
  }
  return false;
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
  // Phase 1: Bellman-Ford on the CFG. Precise pass: tracks push/pop dynamics per basic block and takes max across
  // branches, so it resolves the common shapes (balanced push/pop pairs inside a range-for, max-across-if-branches)
  // exactly. Stacks that hit a positive loop are left with `max_size = 0` for the structural pre-pass below to try
  // to bound; there is no size-fallback anymore - any stack still at `max_size = 0` with a shape outside the
  // `SizeExpr` grammar is a hard compile error (the `QD_ERROR` branch below).
  auto cfg = analysis::build_cfg(root);
  cfg->simplify_graph();
  cfg->determine_ad_stack_size();
  // Phase 2: structural / symbolic pre-pass on residual adaptive stacks. Walks up from every push site and builds a
  // `SizeExpr` upper bound from the enclosing loop bounds. Constant shapes collapse to a `Const` leaf and are also
  // written into `max_size` so downstream consumers that still read compile-time values (e.g. heap-stride
  // pre-sizing in `init_offloaded_task_function`) see a sensible number. Field-load-bounded shapes stay at
  // `max_size = 0` and flow through `AdStackSizingInfo::size_exprs` to the host launcher for per-dispatch
  // evaluation against the live field state.
  // Track per-alloca whether the structural walk bailed out because `build_value_expr` detected a data-flow
  // cycle. The hard-error pass below reads this to emit a specific "stash data-flow cycle" message rather than
  // the generic "unresolved shape" message when the idempotency probe in the `AdStackLoadTopStmt` handler
  // could not discharge the cycle.
  std::unordered_map<AdStackAllocaStmt *, bool> alloca_cycle_detected;
  for (Stmt *s : adaptive_allocas) {
    auto *alloca = s->as<AdStackAllocaStmt>();
    if (alloca->max_size != 0) {
      continue;  // Already resolved by Bellman-Ford in phase 1.
    }
    t_cycle_detected = false;
    t_on_path.clear();
    t_loadtop_subst_zero.clear();
    auto bound = compute_bounded_adstack_size(alloca, root);
    if (!bound.bounded) {
      if (bound.cycle_detected) {
        alloca_cycle_detected[alloca] = true;
      }
      continue;  // Left at `max_size = 0`; the hard-error pass below catches it.
    }
    if (bound.expr->kind == SizeExpr::Kind::Const) {
      alloca->max_size = static_cast<std::size_t>(std::max<int64_t>(bound.expr->const_value, 1));
    } else {
      // Non-const symbolic bound: the LLVM launcher evaluates the tree per-launch via
      // `publish_adstack_metadata` and the SPIR-V launcher via the `AdStackMetadata` buffer. `stmt->max_size`
      // is still read by legacy codegen assertions that have not been updated to tolerate zero, so seed it
      // with `1` as the minimum legal value - the runtime always replaces this with the per-dispatch
      // evaluated bound before any kernel runs.
      alloca->max_size = 1;
    }
    alloca->size_expr = std::move(bound.expr);
  }
  // Mirror a `Const` SizeExpr onto every Bellman-Ford-resolved stack too, so downstream host evaluators read one
  // uniform representation regardless of which phase resolved the bound.
  for (Stmt *s : adaptive_allocas) {
    auto *alloca = s->as<AdStackAllocaStmt>();
    if (!alloca->size_expr && alloca->max_size != 0) {
      alloca->size_expr = SizeExpr::make_const(static_cast<int64_t>(alloca->max_size));
    }
  }
  // Any adstack still unresolved here uses a bound shape neither the Bellman-Ford phase nor the structural
  // `SizeExpr` grammar could parse. There is deliberately no compile-time size fallback: silently picking a
  // default capacity would mask a grammar gap, and a wrong bound only surfaces much later as a runtime
  // overflow with no indication of which alloca tripped it. Emit a hard compile error naming the source
  // location so the user can either restructure the reverse-mode loop to match the grammar or extend the
  // grammar. When `QD_DUMP_IR=1` is set, the full kernel IR is also dumped to
  // `<tmp>/ir_adstack_unresolved/unresolved_alloca_<id>.ll` for offline inspection.
  for (Stmt *s : adaptive_allocas) {
    auto *alloca = s->as<AdStackAllocaStmt>();
    if (alloca->max_size != 0 || alloca->size_expr) {
      continue;
    }
    std::string dump_hint;
    const char *dump_env = std::getenv("QD_DUMP_IR");
    if (dump_env != nullptr && std::string(dump_env) == "1") {
      irpass::re_id(root);
      std::filesystem::path dump_dir = std::filesystem::temp_directory_path() / "ir_adstack_unresolved";
      std::error_code ec;
      std::filesystem::create_directories(dump_dir, ec);
      std::filesystem::path dump_path = dump_dir / fmt::format("unresolved_alloca_{}.ll", alloca->id);
      std::ofstream ofs(dump_path.string());
      auto *old = std::cout.rdbuf(ofs.rdbuf());
      irpass::print(root, /*output=*/nullptr, /*print_ir_dbg_info=*/false, /*print_kernel_wrapper=*/true);
      std::cout.flush();
      std::cout.rdbuf(old);
      dump_hint = fmt::format(" Full kernel IR dumped to {}.", dump_path.string());
    }
    if (alloca_cycle_detected.count(alloca) != 0) {
      QD_ERROR(
          "adstack bound at {} could not be resolved: the structural pre-pass detected a stash data-flow cycle"
          " while walking the push / local-load / ad-stack-load-top chain (stack A's push value reads through a"
          " chain that re-enters stack A, and the idempotency-at-zero probe could not discharge it).{} Either"
          " restructure the reverse-mode loop to break the cycle, or extend the `SizeExpr` grammar / pre-pass"
          " with a sound fixed-point solver for this shape.",
          alloca->get_last_tb(), dump_hint);
    }
    QD_ERROR(
        "adstack bound at {} is unresolved after Bellman-Ford + structural pre-pass.{} Either restructure the"
        " reverse-mode loop so every enclosing range is bounded by an integer constant, a scalar field load,"
        " or a shape the pre-pass already recognises, or extend the `SizeExpr` grammar to cover this shape.",
        alloca->get_last_tb(), dump_hint);
  }
  // Build a per-task `SizeExpr` for each adstack-bearing parallel range-for offload's loop trip count, captured
  // here BEFORE `make_cpu_multithreaded_range_for` rewrites the user loop into chunks. The same `SizeExpr`
  // grammar `compute_bounded_adstack_size` uses for per-thread stack sizing applies: integer constants, scalar
  // i32 / i64 field reads, ndarray shape axes, ndarray element reads at constant or loop indices, plus the
  // arithmetic / max-over-range combinators. The runtime float-heap clip evaluates this expression at launch
  // and uses it as an upper bound on row claims (each loop iteration claims at most one row at the LCA-block,
  // so the heap needs at most `evaluate(loop_trip_count_expr)` rows regardless of how many cells of an
  // oversized gating SNode the reducer counted). Skipped for non-range-for offloads (struct-for / mesh-for /
  // serial / listgen / gc) because their iteration model does not fit the simple `[begin, end)` walk the
  // expression captures; those tasks fall back to the unclipped reducer count.
  auto offloaded_tasks = irpass::analysis::gather_statements(root, [&](Stmt *s) { return s->is<OffloadedStmt>(); });
  for (Stmt *s : offloaded_tasks) {
    auto *task = s->as<OffloadedStmt>();
    if (task->task_type != OffloadedTaskType::range_for) {
      continue;
    }
    bool has_adstack = false;
    irpass::analysis::gather_statements(task, [&](Stmt *inner) {
      if (inner->is<AdStackAllocaStmt>()) {
        has_adstack = true;
      }
      return false;
    });
    if (!has_adstack) {
      continue;
    }
    int32_t var_id_counter = 0;
    auto end_e = resolve_loop_end(task, root, &var_id_counter);
    if (!end_e) {
      continue;
    }
    auto begin_e = resolve_loop_begin_lower_bound(task);
    auto trip = expr_sub(std::move(end_e), std::move(begin_e));
    if (!trip) {
      continue;
    }
    if (trip->kind == SizeExpr::Kind::Const && trip->const_value < 0) {
      trip = SizeExpr::make_const(0);
    }
    // Refuse trees that would force the host evaluator to enumerate an inner index domain at launch. The
    // trip count of `[begin, end)` is a scalar bound by construction; a `MaxOverRange` / `BoundVariable`
    // node here means `resolve_loop_end` walked through a `LoopIndexStmt` whose enclosing loop end ends up
    // as the `MaxOverRange` end (typically the dispatched-thread count of an outer offload), and the
    // resulting tree would trip `evaluate_node`'s 1<<24 guard at every launch. Skip the clip for these
    // tasks; the runtime falls back to the unclipped reducer count, same as the runtime-bound path.
    if (size_expr_contains_inner_domain_enumeration(trip.get())) {
      continue;
    }
    task->pre_chunk_loop_trip_count_expr = std::shared_ptr<SizeExpr>(std::move(trip));
  }
  return true;
}

}  // namespace irpass

}  // namespace quadrants::lang
