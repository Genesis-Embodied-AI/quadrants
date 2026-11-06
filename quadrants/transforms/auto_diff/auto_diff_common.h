#pragma once

#include "quadrants/ir/analysis.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/transforms/utils.h"

#include <typeinfo>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace quadrants::lang {

// ----------------------------------------------------------------------------
// Shared helpers used across multiple autodiff translation units.
// ----------------------------------------------------------------------------

template <typename T>
Stmt *insert_const(const DataType &dtype, Stmt *stmt, const T &value, bool insert_before_me = false) {
  auto type = dtype.ptr_removed();
  Stmt *zero = nullptr;
  if (insert_before_me)
    zero = stmt->insert_before_me(Stmt::make<ConstStmt>(TypedConstant(type.get_element_type(), value)));
  else
    zero = stmt->insert_after_me(Stmt::make<ConstStmt>(TypedConstant(type.get_element_type(), value)));

  if (type->is<TensorType>()) {
    auto t_dtype = type->as<TensorType>();
    std::vector<Stmt *> values(t_dtype->get_num_elements(), zero);
    if (insert_before_me) {
      zero = zero->insert_before_me(Stmt::make<MatrixInitStmt>(values));
    } else {
      zero = zero->insert_after_me(Stmt::make<MatrixInitStmt>(values));
    }
    zero->ret_type = type;
  }
  return zero;
}

class IndependentBlockMetaData {
 public:
  bool is_ib = true;
  bool is_smallest_ib = true;
};

class NonLinearOps {
 public:
  inline static const std::set<TernaryOpType> ternary_collections{TernaryOpType::select};
  inline static const std::set<UnaryOpType> unary_collections{
      UnaryOpType::abs,  UnaryOpType::sin, UnaryOpType::cos, UnaryOpType::tan,  UnaryOpType::tanh, UnaryOpType::asin,
      UnaryOpType::acos, UnaryOpType::exp, UnaryOpType::log, UnaryOpType::sqrt, UnaryOpType::rsqrt};
  inline static const std::set<BinaryOpType> binary_collections{BinaryOpType::mul,   BinaryOpType::div,
                                                                BinaryOpType::atan2, BinaryOpType::pow,
                                                                BinaryOpType::min,   BinaryOpType::max};
};

// Recognize the stack-materialization shape `ReplaceLocalVarWithStacks` emits to subscript a stack-backed tensor's
// current top: a `LocalLoadStmt(MatrixPtrStmt(alloca, const_offset))` where `alloca` is a fresh `AllocaStmt` written by
// exactly one full-tensor `LocalStoreStmt` (no partial or atomic writes), placed before the load in the same block.
// That single store's value is the loaded tensor top; the alloca exists only so the forward store-to-load walker can
// trace the read (`AdStackPushStmt` is not a reaching def). Returns the store's value, or nullptr when the load does
// not match. The reverse pass uses this to reconstruct the per-iteration component from the recomputable top instead
// of spilling the last iteration's value through a single overwrite-each-iteration alloca.
inline Stmt *ad_stack_materialization_source(LocalLoadStmt *ll) {
  auto *mp = ll->src != nullptr ? ll->src->cast<MatrixPtrStmt>() : nullptr;
  if (mp == nullptr || mp->offset == nullptr || !mp->offset->is<ConstStmt>())
    return nullptr;
  auto *alloca = mp->origin != nullptr ? mp->origin->cast<AllocaStmt>() : nullptr;
  Block *block = ll->parent;
  if (alloca == nullptr || block == nullptr || alloca->parent != block)
    return nullptr;
  Stmt *store_val = nullptr;
  int store_pos = -1;
  int load_pos = -1;
  int write_count = 0;
  for (int i = 0; i < (int)block->statements.size(); i++) {
    Stmt *s = block->statements[i].get();
    if (s == ll) {
      load_pos = i;
    } else if (auto *st = s->cast<LocalStoreStmt>()) {
      if (st->dest == alloca) {
        store_val = st->val;
        store_pos = i;
        write_count++;
      } else if (auto *dst_mp = st->dest->cast<MatrixPtrStmt>()) {
        if (dst_mp->origin == alloca)
          write_count++;
      }
    } else if (auto *ao = s->cast<AtomicOpStmt>()) {
      if (ao->dest == alloca)
        write_count++;
    }
  }
  if (store_val == nullptr || write_count != 1 || store_pos < 0 || load_pos < 0 || store_pos >= load_pos)
    return nullptr;
  return store_val;
}

// ----------------------------------------------------------------------------
// Recomputable chain analyzer + cloner: decide whether a forward SSA value can be reconstructed in the reverse-pass
// scope from already-stack-backed allocas, kernel args, constants, and loop indices, and clone the DAG at a target
// insertion point. Cross-stage shared infrastructure: used by `EliminateRecomputableAdStackPushes`
// (forward_state_spill stage) to drop pushes whose values are recomputable, and by `BackupSSA::generic_visit`
// (post_adjoint_cleanup stage) to clone such chains in place of cross-block SSA reads.
// ----------------------------------------------------------------------------

// Returns true iff `stmt`'s transitive operand DAG terminates at recomputable leaves via side-effect-free interior ops
// only. Used by `EliminateRecomputableAdStackPushes` and `BackupSSA::generic_visit` to decide whether a forward SSA
// value can be reconstructed in the reverse-pass scope from already-stack-backed allocas + kernel-args + constants +
// loop indices, instead of being spilled to a per-iteration adstack or to `BackupSSA::load`'s last-iteration plain
// alloca.
//
// Recomputable leaves: AdStackLoadTopStmt (re-readable via cloned load), AdStackAllocaStmt (the stack itself, shared
// not cloned), ArgLoadStmt (kernel-arg, immutable within the launch), ConstStmt, LoopIndexStmt (clonable to read the
// reverse-direction loop's index, which matches the forward iteration the reverse is currently processing).
// Side-effect-free interior ops: UnaryOp, BinaryOp, TernaryOp, MatrixPtr, GlobalPtr, ExternalPtr.
//
// `GlobalLoadStmt` is admitted as a recomputable interior node only when its source SNode is not mutated anywhere in
// the surrounding IB (per `mutated_snodes`); a re-clone at the reverse cursor of a load whose SNode is also written
// observes post-write state instead of the iter-k value the forward chain consumed. Loads whose source is not a
// `GlobalPtrStmt` (e.g. `ExternalPtrStmt`, `GlobalTemporaryStmt`) are rejected outright since their write set is not
// collected here. `LocalLoadStmt` aliases mutable allocas; the reverse pass reads its forward value through the
// dedicated spill path.
//
// Caller passes a `cache` to share memoization across multiple queries on the same DAG (diamond shapes) and the
// `mutated_snodes` set produced by `collect_mutated_snodes(ib)`.
class RecomputableChainAnalyzer {
 public:
  // Set of SNodes written somewhere in the IB. `GlobalStoreStmt`, `AtomicOpStmt`, and `SNodeOpStmt` destinations are
  // tracked; `MatrixPtrStmt` is chased one level to its origin so post-`lower_matrix_ptr` shapes are captured.
  static std::unordered_set<SNode *> collect_mutated_snodes(Block *ib) {
    std::unordered_set<SNode *> result;
    auto add_dest = [&](Stmt *dest) {
      if (dest == nullptr)
        return;
      if (auto *mp = dest->cast<MatrixPtrStmt>())
        dest = mp->origin;
      if (dest == nullptr)
        return;
      if (auto *gp = dest->cast<GlobalPtrStmt>()) {
        if (gp->snode != nullptr)
          result.insert(gp->snode);
      } else if (auto *mgp = dest->cast<MatrixOfGlobalPtrStmt>()) {
        for (auto *s : mgp->snodes)
          if (s != nullptr)
            result.insert(s);
      }
    };
    auto stmts = irpass::analysis::gather_statements(ib, [](Stmt *) { return true; });
    for (auto *s : stmts) {
      if (auto *gs = s->cast<GlobalStoreStmt>()) {
        add_dest(gs->dest);
      } else if (auto *ao = s->cast<AtomicOpStmt>()) {
        add_dest(ao->dest);
      } else if (auto *so = s->cast<SNodeOpStmt>()) {
        if (so->snode != nullptr)
          result.insert(so->snode);
      }
    }
    return result;
  }

  static bool is_recomputable(Stmt *stmt,
                              std::unordered_map<Stmt *, bool> &cache,
                              const std::unordered_set<SNode *> &mutated_snodes) {
    auto it = cache.find(stmt);
    if (it != cache.end())
      return it->second;
    // Tentatively false to break cycles in pathological IR (real SSA DAGs are acyclic, but the cache also serves as a
    // visited set during recursion).
    cache[stmt] = false;
    bool result = check(stmt, cache, mutated_snodes);
    cache[stmt] = result;
    return result;
  }

 private:
  static bool check(Stmt *stmt,
                    std::unordered_map<Stmt *, bool> &cache,
                    const std::unordered_set<SNode *> &mutated_snodes) {
    // Recomputable leaves: ConstStmt, ArgLoadStmt, AdStackLoadTopStmt, AdStackAllocaStmt. LoopIndexStmt is
    // intentionally excluded - cloning a LoopIndexStmt copies the reference to the forward RangeForStmt, but the cloned
    // consumer lives inside the reverse RangeForStmt (a separate stmt with its own loop index), so the cloned read
    // points at undefined state and silently double-accumulates gradients.
    //
    // AdStackLoadTopStmt as a leaf is correct under the dominance + control-flow-consumer + self-load guards in
    // `EliminateRecomputableAdStackPushes::run_one_pass`. The reasoning:
    //
    //   - In FORWARD: the eliminated stack S's body push dominates each `AdStackLoadTopStmt(S)` (the
    //     dominance guard), and the chain leaves' pushes dominate the chain's evaluation point. Each
    //     stack referenced by a chain leaf has a stable "top" value within one forward iteration body
    //     (no pops in forward), so substituting load_top(S) with the chain re-evaluates to the same
    //     value (no iteration shift).
    //
    //   - In REVERSE: `MakeAdjoint` visits forward stmts in reverse order, emitting reverse code at a
    //     cursor that advances one position per visit. For a forward stmt F at position P_F, the
    //     emitted reverse code lands at cursor position roughly inversely correlated with P_F. The
    //     dominance guard ensures every chain-leaf stack T has its body push at a position P_T with
    //     P_T < P (where P is load_top(S)'s forward position). Therefore T's pop in reverse (emitted
    //     when MakeAdjoint visits T's body push) lands AFTER the chain consumer's reverse emission
    //     (the consumer's forward position is P > P_T). At the consumer's reverse cursor, T has not
    //     been popped yet, so load_top(T) returns T's iter-k-push value - matching what the original
    //     load_top(S) returned in forward iter k.
    //
    // The control-flow-consumer guard in `run_one_pass` covers a separate issue: stacks whose load_tops are direct
    // operands of IfStmt cond / RangeFor begin/end. `MakeAdjoint::visit(IfStmt)` runs a dedicated snap-stack fixup that
    // ONLY triggers when the cond is a bare `AdStackLoadTopStmt` (line 2168-2202). Eliminating the cond stack converts
    // the cond into a compound stmt, the snap-stack does not trigger, and the reverse cond falls back to BackupSSA's
    // load(op) which is single-slot last-iter only - silent gradient corruption on multi-iter loops.
    if (stmt->is<AdStackLoadTopStmt>() || stmt->is<AdStackAllocaStmt>() || stmt->is<ArgLoadStmt>() ||
        stmt->is<ConstStmt>()) {
      return true;
    }
    // A `GlobalLoadStmt` chain leaf is recomputable only when its `GlobalPtrStmt` source resolves to an SNode the IB
    // never writes; a reverse-side re-clone of a write-aliased SNode read returns post-write state. Non-`GlobalPtrStmt`
    // sources (e.g. `ExternalPtrStmt`, `GlobalTemporaryStmt`) are rejected because their write set is not collected.
    if (auto *gl = stmt->cast<GlobalLoadStmt>()) {
      auto *src = gl->src;
      auto *gp = src ? src->cast<GlobalPtrStmt>() : nullptr;
      if (gp == nullptr || gp->snode == nullptr || mutated_snodes.count(gp->snode)) {
        return false;
      }
    }
    bool is_interior = stmt->is<UnaryOpStmt>() || stmt->is<BinaryOpStmt>() || stmt->is<TernaryOpStmt>() ||
                       stmt->is<MatrixPtrStmt>() || stmt->is<GlobalPtrStmt>() || stmt->is<ExternalPtrStmt>() ||
                       stmt->is<GlobalLoadStmt>();
    if (!is_interior) {
      return false;
    }
    auto operands = stmt->get_operands();
    for (auto *op : operands) {
      if (op == nullptr)
        continue;
      if (!is_recomputable(op, cache, mutated_snodes))
        return false;
    }
    return true;
  }
};

// Clones the SSA chain rooted at `src` into the IR, inserting cloned stmts before `insert_point`. Returns the cloned
// root. Per-stmt cache shared across one resolution materializes each SSA value at most once: diamond DAGs see two
// consumers but get one shared clone. `AdStackAllocaStmt` is treated as a leaf and shared (not cloned) - the stack
// itself is a unique storage handle that must not be duplicated.
//
// Pop-ordering safety: cloned `AdStackLoadTopStmt`s read the live top at the cloned position. `MakeAdjoint` emits
// `AdStackPopStmt` for each surviving `AdStackPushStmt`, and the existing reverse-pass scheme places the pop AFTER all
// uses of that stack within the reverse iteration (uses include both the original `AdStackLoadTopStmt`s emitted by
// `ReplaceLocalVarWithStacks` and the consumers' clones). For loop-carried allocas the pop fires early to expose the
// iteration's INPUT primal as the new top, which is exactly the value the recomputed chain needs - the existing
// per-consumer clone path at `BackupSSA::generic_visit` line ~2697 relies on this same property and has been correct in
// production.
class RecomputableChainCloner {
 public:
  static Stmt *clone_at(Stmt *src, Stmt *insert_point, std::unordered_map<Stmt *, Stmt *> &cache) {
    auto it = cache.find(src);
    if (it != cache.end())
      return it->second;
    Stmt *cloned = nullptr;
    if (src->is<AdStackAllocaStmt>()) {
      // The alloca is shared, not cloned: every load reads the same physical stack.
      cloned = src;
    } else if (src->is<AdStackLoadTopStmt>() || src->is<ArgLoadStmt>() || src->is<ConstStmt>()) {
      auto cloned_unique = src->clone();
      cloned = insert_point->insert_before_me(std::move(cloned_unique));
      // For AdStackLoadTopStmt clones, the cloned stmt's `stack` operand still points at the original AdStackAllocaStmt
      // - that's the desired sharing.
    } else {
      // Compound op: clone first, then walk operands and rewire each to a recursive clone.
      auto cloned_unique = src->clone();
      cloned = insert_point->insert_before_me(std::move(cloned_unique));
      int n = src->num_operands();
      for (int i = 0; i < n; i++) {
        auto *op = src->operand(i);
        if (op != nullptr) {
          Stmt *new_op = clone_at(op, cloned, cache);
          cloned->set_operand(i, new_op);
        }
      }
    }
    cache[src] = cloned;
    return cloned;
  }
};

// ----------------------------------------------------------------------------
// ADTransform: shared base for reverse-mode (MakeAdjoint) and forward-mode (MakeDual) IR builders. Methods are inline
// so derived classes in separate translation units can use them without ODR concerns.
// ----------------------------------------------------------------------------
class ADTransform : public IRVisitor {
 protected:
  Stmt *constant(float32 x, DataType dtype = PrimitiveType::unknown) {
    dtype.set_is_pointer(false);
    if (!dtype->is<TensorType>())
      return insert<ConstStmt>(TypedConstant(x));

    auto tensor_type = dtype->as<TensorType>();
    auto num_elements = tensor_type->get_num_elements();
    std::vector<Stmt *> values;
    for (int i = 0; i < num_elements; i++) {
      values.push_back(insert<ConstStmt>(TypedConstant(x)));
    }
    auto matrix_init_stmt = insert<MatrixInitStmt>(values);
    matrix_init_stmt->ret_type = tensor_type;
    return matrix_init_stmt;
  }

  // utils
  Stmt *sgn(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::sgn, load(inp));
  }

  // utils
  Stmt *negate(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::neg, load(inp));
  }

  Stmt *sqrt(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::sqrt, load(inp));
  }

  Stmt *rsqrt(Stmt *inp) {
    return insert<UnaryOpStmt>(UnaryOpType::rsqrt, load(inp));
  }

  Stmt *mul(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::mul, load(op1), load(op2));
  }

  Stmt *sqr(Stmt *op1) {
    return mul(op1, op1);
  }

  Stmt *add(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::add, load(op1), load(op2));
  }

  Stmt *cmp_lt(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::cmp_lt, load(op1), load(op2));
  }

  Stmt *sub(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::sub, load(op1), load(op2));
  }

  Stmt *div(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::div, load(op1), load(op2));
  }

  Stmt *sel(Stmt *op1, Stmt *op2, Stmt *op3) {
    return insert<TernaryOpStmt>(TernaryOpType::select, load(op1), load(op2), load(op3));
  }

  Stmt *cos(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::cos, load(op1));
  }

  Stmt *sin(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::sin, load(op1));
  }

  Stmt *log(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::log, load(op1));
  }

  Stmt *tan(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::tan, load(op1));
  }

  Stmt *tanh(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::tanh, load(op1));
  }

  Stmt *exp(Stmt *op1) {
    return insert<UnaryOpStmt>(UnaryOpType::exp, load(op1));
  }

  Stmt *pow(Stmt *op1, Stmt *op2) {
    return insert<BinaryOpStmt>(BinaryOpType::pow, load(op1), load(op2));
  }

 public:
  virtual Stmt *insert_grad_stmt(std::unique_ptr<Stmt> &&stmt) = 0;

  template <typename T, typename... Args>
  Stmt *insert(Args &&...args) {
    return insert_grad_stmt(Stmt::make<T>(args...));
  }

  template <typename T>
  Stmt *insert_const_for_grad(const DataType &dtype, Stmt *stmt, const T &val) {
    auto zero = insert<ConstStmt>(TypedConstant(dtype.ptr_removed().get_element_type(), val));
    if (dtype.ptr_removed()->is<TensorType>()) {
      auto t_dtype = dtype.ptr_removed()->as<TensorType>();
      std::vector<Stmt *> values(t_dtype->get_num_elements(), zero);
      zero = insert<MatrixInitStmt>(values);
      zero->ret_type = dtype.ptr_removed();
    }
    return zero;
  }

  void visit(AllocaStmt *alloca) override {
    // do nothing.
  }

  void visit(AdStackAllocaStmt *alloca) override {
    // do nothing.
  }

  void visit(ArgLoadStmt *stmt) override {
    // do nothing.
  }

  void visit(GetElementStmt *stmt) override {
    // do nothing
  }

  void visit(LoopIndexStmt *stmt) override {
    // do nothing.
  }

  void visit(MatrixPtrStmt *stmt) override {
    // do nothing.
  }

  void visit(PrintStmt *print_stmt) override {
    // do nothing
  }

  void visit(ConstStmt *const_stmt) override {
    // do nothing
  }

  void visit(ReturnStmt *stmt) override {
    // do nothing
  }

  void visit(WhileControlStmt *stmt) override {
    QD_NOT_IMPLEMENTED
  }

  void visit(ContinueStmt *stmt) override {
    QD_NOT_IMPLEMENTED;
  }

  void visit(WhileStmt *stmt) override {
    QD_NOT_IMPLEMENTED
  }

  void visit(GlobalPtrStmt *stmt) override {
    // do nothing
  }

  void visit(ExternalPtrStmt *stmt) override {
    // do nothing
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    // do nothing
  }

  void visit(DecorationStmt *stmt) override {
    // do nothing
  }

  Stmt *load(Stmt *alloc) {
    QD_ASSERT(alloc != nullptr);
    if (alloc->is<AllocaStmt>() || alloc->is<MatrixPtrStmt>()) {
      return insert<LocalLoadStmt>(alloc);
    } else {
      // non alloca
      return alloc;
    }
  }

  bool gradients_stopped(GlobalLoadStmt *stmt, SNode *snode) {
    for (auto block = stmt->parent; block; block = block->parent_block()) {
      for (auto s : block->stop_gradients) {
        if (s == snode) {
          return true;
        }
      }
    }
    return false;
  }

  void visit(AssertStmt *stmt) override {
    // do nothing
  }

  void visit(RangeAssumptionStmt *stmt) override {
    // do nothing
  }

  void visit(LinearizeStmt *stmt) override {
    // do nothing
  }

  void visit(IntegerOffsetStmt *stmt) override {
    // do nothing
  }

  void visit(RandStmt *stmt) override {
    QD_ERROR("RandStmt not supported in AutoDiff for now.");
  }
};

}  // namespace quadrants::lang
