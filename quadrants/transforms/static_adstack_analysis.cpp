// Implementation of the static-IR-bound sparse-adstack-heap analysis. Walks the OffloadedStmt body once to compute
// per-thread strides, the LCA of float push/load-top sites, the autodiff-bootstrap push set, and (if a recognized
// gate sits on the LCA-to-root chain) a captured `StaticAdStackBoundExpr`. The analysis is shared between SPIR-V
// and LLVM codegens so the gate-recognition grammar stays single-source; backend-specific SNode descriptor lookup
// is parameterized via the resolver callback in the header.
#include "quadrants/transforms/static_adstack_analysis.h"

#include <functional>
#include <unordered_map>

#include "quadrants/ir/snode.h"
#include "quadrants/ir/statements.h"

namespace quadrants::lang {

namespace {

// True iff the push is an autodiff-bootstrap shape: parent block belongs to an `OffloadedStmt`, the previous
// sibling is the matching `AdStackAllocaStmt`, and the pushed value is a `ConstStmt`. The autodiff transform emits
// these immediately after the alloca so the matching reverse pop has a value to consume on every dispatched thread
// regardless of any later gating.
bool is_autodiff_bootstrap_push(AdStackPushStmt *p) {
  if (p->v == nullptr || !p->v->is<ConstStmt>()) {
    return false;
  }
  Block *parent = p->parent;
  if (parent == nullptr || parent->parent_stmt() == nullptr || !parent->parent_stmt()->is<OffloadedStmt>()) {
    return false;
  }
  AdStackAllocaStmt *target = p->stack ? p->stack->cast<AdStackAllocaStmt>() : nullptr;
  if (target == nullptr) {
    return false;
  }
  int idx = -1;
  for (int i = 0; i < (int)parent->statements.size(); ++i) {
    if (parent->statements[i].get() == p) {
      idx = i;
      break;
    }
  }
  if (idx <= 0) {
    return false;
  }
  return parent->statements[idx - 1].get() == target;
}

// The float-stack predicate folded into the LCA computation: push/load-top/load-top-adj sites where the underlying
// alloca's `ret_type` is `f32`. Pop sites are deliberately NOT included - they only mutate `count_var` and impose
// no dominance requirement on the row claim.
bool stack_is_float(Stmt *push_or_load) {
  AdStackAllocaStmt *alloca = nullptr;
  if (auto *p = push_or_load->cast<AdStackPushStmt>()) {
    alloca = p->stack ? p->stack->cast<AdStackAllocaStmt>() : nullptr;
  } else if (auto *l = push_or_load->cast<AdStackLoadTopStmt>()) {
    alloca = l->stack ? l->stack->cast<AdStackAllocaStmt>() : nullptr;
  } else if (auto *l = push_or_load->cast<AdStackLoadTopAdjStmt>()) {
    alloca = l->stack ? l->stack->cast<AdStackAllocaStmt>() : nullptr;
  }
  return alloca != nullptr && alloca->ret_type == PrimitiveType::f32;
}

// Generic IR walker that descends into block / control-flow children. The analysis uses this for the alloca + push
// scan; the gate matcher uses a similar shape to collect per-stack push values.
template <class Fn>
void walk_ir(IRNode *node, Fn &&visit) {
  if (auto *blk = dynamic_cast<Block *>(node)) {
    for (auto &s : blk->statements) {
      visit(s.get());
      walk_ir(s.get(), visit);
    }
    return;
  }
  if (auto *if_stmt = dynamic_cast<IfStmt *>(node)) {
    if (if_stmt->true_statements) {
      walk_ir(if_stmt->true_statements.get(), visit);
    }
    if (if_stmt->false_statements) {
      walk_ir(if_stmt->false_statements.get(), visit);
    }
    return;
  }
  if (auto *range_for = dynamic_cast<RangeForStmt *>(node)) {
    walk_ir(range_for->body.get(), visit);
    return;
  }
  if (auto *struct_for = dynamic_cast<StructForStmt *>(node)) {
    walk_ir(struct_for->body.get(), visit);
    return;
  }
  if (auto *mesh_for = dynamic_cast<MeshForStmt *>(node)) {
    walk_ir(mesh_for->body.get(), visit);
    return;
  }
  if (auto *while_stmt = dynamic_cast<WhileStmt *>(node)) {
    walk_ir(while_stmt->body.get(), visit);
    return;
  }
}

}  // namespace

StaticAdStackAnalysisResult analyze_adstack_static_bounds(OffloadedStmt *task_ir,
                                                          const SNodeDescriptorResolver &snode_descriptor_resolver) {
  StaticAdStackAnalysisResult result;
  if (task_ir == nullptr || task_ir->body == nullptr) {
    return result;
  }

  // First scan: collect alloca strides, classify each push as bootstrap or not, gather f32 push/load-top blocks for
  // the LCA reduce.
  std::vector<Block *> push_side_blocks;
  walk_ir(task_ir->body.get(), [&](Stmt *s) {
    if (auto *alloca = s->cast<AdStackAllocaStmt>()) {
      if (alloca->ret_type == PrimitiveType::f32) {
        result.per_thread_stride_float += 2u * uint32_t(alloca->max_size);
        result.num_ad_stacks++;
      } else if (alloca->ret_type == PrimitiveType::i32 || alloca->ret_type == PrimitiveType::u1) {
        // i32 / u1 adstacks have no adjoint; auto_diff.cpp only emits AdStackAccAdjoint / LoadTopAdj on real-typed
        // stacks. An int adjoint would also be meaningless: the docs document gradients silently reading as zero
        // through integer casts.
        result.per_thread_stride_int += uint32_t(alloca->max_size);
        result.num_ad_stacks++;
      }
      return;
    }
    if (s->is<AdStackPushStmt>() || s->is<AdStackLoadTopStmt>() || s->is<AdStackLoadTopAdjStmt>()) {
      if (!stack_is_float(s)) {
        return;
      }
      if (auto *p = s->cast<AdStackPushStmt>(); p && is_autodiff_bootstrap_push(p)) {
        result.bootstrap_pushes.insert(p);
      } else {
        push_side_blocks.push_back(s->parent);
      }
    }
  });

  // Pairwise LCA reduce. Empty `push_side_blocks` means the task has no f32 adstack push sites and the LCA stays
  // null (the float heap is unbound and no row claim is emitted by the codegen). A single block is its own LCA.
  if (!push_side_blocks.empty()) {
    auto lca_of = [](Block *a, Block *b) -> Block * {
      if (a == b) {
        return a;
      }
      std::unordered_set<Block *> a_ancestors;
      for (Block *cur = a; cur != nullptr; cur = cur->parent_block()) {
        a_ancestors.insert(cur);
      }
      for (Block *cur = b; cur != nullptr; cur = cur->parent_block()) {
        if (a_ancestors.count(cur)) {
          return cur;
        }
      }
      // Both blocks live under the same task-body root, so their ancestor chains converge at that root at the
      // latest. Falling through to nullptr would degrade to the eager (root-block) claim path which is still
      // correct, just non-optimal.
      return nullptr;
    };
    Block *lca = push_side_blocks[0];
    for (size_t i = 1; i < push_side_blocks.size() && lca != nullptr; ++i) {
      lca = lca_of(lca, push_side_blocks[i]);
    }
    result.lca_block_float = lca;
  }

  if (result.lca_block_float == nullptr) {
    return result;
  }

  // Second scan: per-stack pushed values, used by the gate matcher to resolve autodiff-spilled gate predicates of
  // shape `IfStmt(cond = AdStackLoadTopStmt(stack=S))` (the gate predicate's bool is spilled onto a u1 adstack in
  // the forward direction and replayed via load_top in the reverse direction).
  std::unordered_map<AdStackAllocaStmt *, std::vector<Stmt *>> per_stack_pushed_values;
  walk_ir(task_ir->body.get(), [&](Stmt *s) {
    if (auto *push = s->cast<AdStackPushStmt>()) {
      if (auto *alloca = push->stack ? push->stack->cast<AdStackAllocaStmt>() : nullptr) {
        per_stack_pushed_values[alloca].push_back(push->v);
      }
    }
  });

  // Resolve a `GlobalLoadStmt::src` chain to a captured field source. Returns true on a recognized shape (ndarray
  // ext-ptr or SNode root->dense->place(scalar)); on success populates the source-kind-specific fields of `out`.
  auto match_field_source = [&](Stmt *load_src, StaticAdStackBoundExpr &out) -> bool {
    if (auto *ext = load_src->cast<ExternalPtrStmt>()) {
      if (auto *base_arg = ext->base_ptr->cast<ArgLoadStmt>()) {
        out.field_source_kind = StaticAdStackBoundExpr::FieldSourceKind::NdArray;
        out.ndarray_arg_id = base_arg->arg_id;
        return true;
      }
      return false;
    }
    if (auto *getch = load_src->cast<GetChStmt>()) {
      const SNode *leaf = getch->output_snode;
      if (leaf == nullptr) {
        return false;
      }
      const SNode *dense = leaf->parent;
      if (dense == nullptr || dense->type != SNodeType::dense) {
        return false;
      }
      const SNode *root_snode = dense->parent;
      if (root_snode == nullptr || root_snode->type != SNodeType::root) {
        return false;
      }
      if (!snode_descriptor_resolver) {
        return false;
      }
      auto desc_opt = snode_descriptor_resolver(leaf, dense);
      if (!desc_opt.has_value()) {
        return false;
      }
      out.field_source_kind = StaticAdStackBoundExpr::FieldSourceKind::SNode;
      out.snode_id = leaf->id;
      out.snode_root_id = desc_opt->root_id;
      out.snode_byte_base_offset = desc_opt->byte_base_offset;
      out.snode_byte_cell_stride = desc_opt->byte_cell_stride;
      out.snode_iter_count = desc_opt->iter_count;
      return true;
    }
    return false;
  };

  // Recursive gate matcher. Accepts both the direct-comparison shape `BinaryOp(cmp, GlobalLoad, Const)` and the
  // autodiff-spilled shape `AdStackLoadTopStmt(S)` (resolved by walking back to the unique non-const push onto S).
  std::function<bool(Stmt *, bool, StaticAdStackBoundExpr &)> try_match_gate_cond;
  try_match_gate_cond = [&](Stmt *cond, bool polarity, StaticAdStackBoundExpr &out) -> bool {
    if (auto *load_top = cond->cast<AdStackLoadTopStmt>()) {
      auto *target_stack = load_top->stack ? load_top->stack->cast<AdStackAllocaStmt>() : nullptr;
      if (target_stack == nullptr) {
        return false;
      }
      auto pushes_it = per_stack_pushed_values.find(target_stack);
      if (pushes_it == per_stack_pushed_values.end()) {
        return false;
      }
      Stmt *real_pushed_value = nullptr;
      for (Stmt *pushed : pushes_it->second) {
        if (pushed->is<ConstStmt>()) {
          continue;
        }
        if (real_pushed_value != nullptr) {
          // More than one non-const push - the gate's logical value depends on which path executed, and the
          // reducer cannot mirror that without re-emitting the full forward IR. Fall through to worst-case sizing.
          return false;
        }
        real_pushed_value = pushed;
      }
      if (real_pushed_value == nullptr) {
        return false;
      }
      return try_match_gate_cond(real_pushed_value, polarity, out);
    }
    auto *bin = cond->cast<BinaryOpStmt>();
    if (bin == nullptr) {
      return false;
    }
    const auto op = bin->op_type;
    const bool is_cmp = (op == BinaryOpType::cmp_lt || op == BinaryOpType::cmp_le || op == BinaryOpType::cmp_gt ||
                         op == BinaryOpType::cmp_ge || op == BinaryOpType::cmp_eq || op == BinaryOpType::cmp_ne);
    if (!is_cmp) {
      return false;
    }
    // Accept either `field cmp literal` (the typical `if field[i] > literal`) or the symmetric `literal cmp field`
    // (e.g. `if literal < field[i]`). The symmetric form gets the comparison op flipped so the runtime reducer
    // always evaluates `field cmp literal` against the captured `literal_*`.
    Stmt *lhs = bin->lhs;
    Stmt *rhs = bin->rhs;
    auto *lhs_load = lhs->cast<GlobalLoadStmt>();
    auto *rhs_const = rhs->cast<ConstStmt>();
    auto *rhs_load = rhs->cast<GlobalLoadStmt>();
    auto *lhs_const = lhs->cast<ConstStmt>();
    GlobalLoadStmt *load = nullptr;
    ConstStmt *cst = nullptr;
    BinaryOpType captured_op = op;
    if (lhs_load != nullptr && rhs_const != nullptr) {
      load = lhs_load;
      cst = rhs_const;
    } else if (rhs_load != nullptr && lhs_const != nullptr) {
      load = rhs_load;
      cst = lhs_const;
      switch (op) {
        case BinaryOpType::cmp_lt:
          captured_op = BinaryOpType::cmp_gt;
          break;
        case BinaryOpType::cmp_le:
          captured_op = BinaryOpType::cmp_ge;
          break;
        case BinaryOpType::cmp_gt:
          captured_op = BinaryOpType::cmp_lt;
          break;
        case BinaryOpType::cmp_ge:
          captured_op = BinaryOpType::cmp_le;
          break;
        case BinaryOpType::cmp_eq:
        case BinaryOpType::cmp_ne:
          // Symmetric, keep the captured op as-is.
          break;
        default:
          return false;
      }
    } else {
      return false;
    }
    if (!match_field_source(load->src, out)) {
      return false;
    }
    out.cmp_op = static_cast<int>(captured_op);
    out.polarity = polarity;
    if (cst->val.dt->is_primitive(PrimitiveTypeID::f32)) {
      out.field_dtype_is_float = true;
      out.literal_f32 = cst->val.val_f32;
      return true;
    }
    if (cst->val.dt->is_primitive(PrimitiveTypeID::i32)) {
      out.field_dtype_is_float = false;
      out.literal_i32 = cst->val.val_i32;
      return true;
    }
    // Other types (f64 / i64 / etc.) fall through; the reducer kernel never has to dispatch on heterogeneous
    // literal kinds.
    return false;
  };

  // Walk the chain from LCA up to the task body root, collecting IfStmt gates. RangeForStmt / StructForStmt /
  // MeshForStmt / WhileStmt / OffloadedStmt parents are skipped (iterators sweep threads rather than gating
  // them; the offload boundary is the kernel entry). Anything else aborts the chain - unfamiliar control-flow
  // structures might gate threads in ways the reducer cannot mirror.
  int gate_count = 0;
  bool chain_ok = true;
  StaticAdStackBoundExpr captured;
  for (Block *cur = result.lca_block_float; cur != nullptr; cur = cur->parent_block()) {
    Stmt *parent = cur->parent_stmt();
    if (parent == nullptr) {
      break;  // task body root reached
    }
    if (auto *if_stmt = parent->cast<IfStmt>()) {
      const bool polarity = (cur == if_stmt->true_statements.get());
      ++gate_count;
      if (gate_count > 1) {
        chain_ok = false;
        break;  // compound predicate; fall back.
      }
      if (!try_match_gate_cond(if_stmt->cond, polarity, captured)) {
        chain_ok = false;
        break;
      }
    } else if (parent->is<RangeForStmt>() || parent->is<StructForStmt>() || parent->is<MeshForStmt>() ||
               parent->is<WhileStmt>() || parent->is<OffloadedStmt>()) {
      continue;
    } else {
      chain_ok = false;
      break;
    }
  }
  if (chain_ok && gate_count == 1) {
    result.bound_expr = captured;
  }
  return result;
}

}  // namespace quadrants::lang
