#pragma once

#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace quadrants::lang {
namespace irpass {

// Content-keyed ordering for cross-offload global-temp slot assignment.
//
// A cross-offload value's global-temp offset is part of the IR of every task that reads or writes it, so if the
// offset moved whenever an unrelated edit reordered offload traversal, those otherwise-unchanged tasks would
// re-key and miss the per-task compile cache. GlobalTmpOrdering removes that coupling: given the collected
// cross-offload values (in first-encounter / traversal order) and, per alloca, the local-store / atomic
// statements that write into it, sort() reorders the values so a value's position is a function of WHAT it is
// (its operand-def DAG content) rather than of traversal order.
//
// Ownership split: the caller (IdentifyValuesUsedInOtherOffloads) collects the values and the alloca-store map
// during its normal traversal and does the actual offset allocation; this helper only computes the stable order.
class GlobalTmpOrdering {
 public:
  // Per-alloca list of the local-store / atomic statements that write into it, in traversal order.
  using AllocaStores = std::unordered_map<Stmt *, std::vector<Stmt *>>;

  // Stable-sort `values` in place by content key. stable_sort keeps traversal order among equal keys, so a slot's
  // offset is a function of its content rather than of traversal order.
  void sort(std::vector<Stmt *> &values, const AllocaStores &alloca_stores) {
    std::stable_sort(values.begin(), values.end(), [this, &alloca_stores](Stmt *a, Stmt *b) {
      return sort_key(a, alloca_stores) < sort_key(b, alloca_stores);
    });
  }

 private:
  // FNV-1a string hash / mixing step, shared by local_key() and stable_key().
  static std::uint64_t hstr(const std::string &str) {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : str)
      h = (h ^ c) * 1099511628211ULL;
    return h;
  }
  static std::uint64_t mix(std::uint64_t h, std::uint64_t x) {
    return (h ^ x) * 1099511628211ULL;
  }

  // Offset-free content hash of a single statement, EXCLUDING its operands: statement kind, type, and
  // constant/arg identity. Operand keys are folded in by stable_key(); factoring this out lets stable_key()
  // walk the operand-def DAG iteratively.
  std::uint64_t local_key(Stmt *s) {
    std::uint64_t h = hstr(typeid(*s).name());
    h = mix(h, hstr(s->ret_type.to_string()));
    if (auto *c = s->cast<ConstStmt>())
      h = mix(h, hstr(c->val.stringify()));
    if (auto *a = s->cast<ArgLoadStmt>()) {
      for (int id : a->arg_id)
        h = mix(h, (std::uint64_t)(unsigned)id);
      h = mix(h, (std::uint64_t)a->is_ptr);
    }
    // Semantic discriminant fields: statements of the same class, return type, and operands can still differ in
    // an op_type (or cast target / bit-vectorization flag). Without these, e.g. a + b and a - b hash equal and
    // stable_sort falls back to traversal order for them, reintroducing exactly the offset instability this
    // content-keyed ordering exists to remove (PR #864 review r3776549281).
    if (auto *u = s->cast<UnaryOpStmt>()) {
      h = mix(h, (std::uint64_t)u->op_type);
      if (u->is_cast())
        h = mix(h, hstr(u->cast_type.to_string()));
    }
    if (auto *b = s->cast<BinaryOpStmt>()) {
      h = mix(h, (std::uint64_t)b->op_type);
      h = mix(h, (std::uint64_t)b->is_bit_vectorized);
    }
    if (auto *t = s->cast<TernaryOpStmt>())
      h = mix(h, (std::uint64_t)t->op_type);
    if (auto *a = s->cast<AtomicOpStmt>())
      h = mix(h, (std::uint64_t)a->op_type);
    if (auto *sn = s->cast<SNodeOpStmt>())
      h = mix(h, (std::uint64_t)sn->op_type);
    return h;
  }

  // Offset-free content hash over the value's operand-def DAG: a value's key depends on WHAT it is (local_key)
  // plus the keys of its operands, not on offload traversal order. Memoized (also dedups shared subexpressions),
  // and used to order global-temp slot assignment stably (S2a').
  //
  // Computed with an explicit worklist rather than recursion so that native stack depth does NOT grow with the
  // length of the SSA dependency chain: a deeply unrolled / generated straight-line expression could otherwise
  // overflow the compiler's stack here (PR #864 review r3796280238). Post-order: a statement is hashed only once
  // all its operands are memoized. `on_stack_` breaks any back-edge defensively (the SSA def DAG is acyclic, so
  // this never fires in practice); a back-edge operand contributes key 0.
  std::uint64_t stable_key(Stmt *s) {
    if (s == nullptr)
      return 0;
    if (auto it = key_memo_.find(s); it != key_memo_.end())
      return it->second;
    std::vector<Stmt *> stack{s};
    while (!stack.empty()) {
      Stmt *cur = stack.back();
      if (key_memo_.count(cur)) {
        stack.pop_back();
        continue;
      }
      bool ready = true;
      for (auto *op : cur->get_operands()) {
        if (op != nullptr && !key_memo_.count(op) && !on_stack_.count(op)) {
          stack.push_back(op);
          ready = false;
        }
      }
      if (ready) {
        std::uint64_t h = local_key(cur);
        for (auto *op : cur->get_operands())
          h = mix(h, op != nullptr && key_memo_.count(op) ? key_memo_[op] : 0);
        key_memo_[cur] = h;
        on_stack_.erase(cur);
        stack.pop_back();
      } else {
        on_stack_.insert(cur);
      }
    }
    return key_memo_[s];
  }

  // The value written by a local store / atomic (the RHS of the update), or null for anything we don't model.
  static Stmt *store_value(Stmt *store) {
    if (auto *s = store->cast<LocalStoreStmt>())
      return s->val;
    if (auto *a = store->cast<AtomicOpStmt>())
      return a->val;
    return nullptr;
  }

  // Ordering key for global-temp slot assignment: stable_key(), plus -- for allocas only -- a content signature of
  // the values written into the local. An alloca has no operands, so stable_key() keys it on statement kind + type
  // alone; two same-typed cross-offload locals would then share a key and fall back to traversal order, retaining
  // the offset instability this ordering removes (PR #864 review r3797057477). Fold in local_key(store) (which
  // carries the store-vs-atomic kind and, for atomics, the op) and the content key of each stored value. Those
  // stored values reference the alloca only through loads, and a load reaches the alloca via the operand DAG where
  // the alloca is a childless leaf, so stable_key() stays acyclic and this signature is independent of evaluation
  // order. Best-effort: two locals with identical types AND identical update chains still collide (harmless -- they
  // are genuinely interchangeable), and store order (stable for a given local) is significant.
  std::uint64_t sort_key(Stmt *s, const AllocaStores &alloca_stores) {
    std::uint64_t h = stable_key(s);
    if (s->is<AllocaStmt>()) {
      if (auto it = alloca_stores.find(s); it != alloca_stores.end())
        for (auto *store : it->second)
          h = mix(mix(h, local_key(store)), stable_key(store_value(store)));
    }
    return h;
  }

  std::unordered_map<Stmt *, std::uint64_t> key_memo_;
  // Statements currently on stable_key()'s worklist, used to defensively break back-edges.
  std::unordered_set<Stmt *> on_stack_;
};

}  // namespace irpass
}  // namespace quadrants::lang
