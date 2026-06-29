// Implementation of the static-IR-bound sparse-adstack-heap analysis. Walks the OffloadedStmt body once to compute
// per-thread strides, the LCA of float push/load-top sites, the autodiff-bootstrap push set, and (if a recognized gate
// sits on the LCA-to-root chain) a captured `StaticAdStackBoundExpr`. The analysis is shared between SPIR-V and LLVM
// codegens so the gate-recognition grammar stays single-source; backend-specific SNode descriptor lookup is
// parameterized via the resolver callback in the header.
#include "quadrants/transforms/static_adstack_analysis.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>

#include "quadrants/ir/analysis.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/statements.h"

namespace quadrants::lang {

namespace {

// True iff the push is an autodiff-bootstrap shape: parent block belongs to an `OffloadedStmt`, the pushed value is a
// `ConstStmt`, and the matching `AdStackAllocaStmt` lies just before the push - either as the immediately previous
// sibling (SPIR-V IR shape, the const literal is folded into the push's `v` field as a `ConstStmt` that is itself the
// previous sibling), or with the const's `ConstStmt` sitting between them (LLVM IR shape, the const is materialised as
// its own statement between the alloca and the push). The autodiff transform emits these pushes immediately after the
// alloca so the matching reverse pop has a value to consume on every dispatched thread regardless of any later gating.
bool is_autodiff_bootstrap_push(AdStackPushStmt *p) {
  if (p->v == nullptr || !p->v->is<ConstStmt>()) {
    return false;
  }
  Block *parent = p->parent;
  if (parent == nullptr) {
    return false;
  }
  // Accept a parent block whose owning statement is either the `OffloadedStmt` directly (the SPIR-V codegen IR shape)
  // or a `RangeForStmt` / `StructForStmt` / `MeshForStmt` that is itself a direct child of an `OffloadedStmt` (the LLVM
  // codegen IR shape, where the offload's body contains a single for-stmt that wraps the user's loop body). In both
  // shapes the push runs unconditionally on every dispatched thread - the inner for body iterates once per logical loop
  // iteration, but each iteration's bootstrap push is balanced by its matching pop, so the "always executes" property
  // `is_autodiff_bootstrap_push` is checking still holds.
  Stmt *parent_stmt = parent->parent_stmt();
  if (parent_stmt == nullptr) {
    return false;
  }
  bool unconditional_in_offload = parent_stmt->is<OffloadedStmt>();
  if (!unconditional_in_offload &&
      (parent_stmt->is<RangeForStmt>() || parent_stmt->is<StructForStmt>() || parent_stmt->is<MeshForStmt>())) {
    Block *grand = parent_stmt->parent;
    if (grand != nullptr && grand->parent_stmt() != nullptr && grand->parent_stmt()->is<OffloadedStmt>()) {
      unconditional_in_offload = true;
    }
  }
  if (!unconditional_in_offload) {
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
  Stmt *prev = parent->statements[idx - 1].get();
  if (prev == target) {
    return true;
  }
  // Allow a single intermediary `ConstStmt` between the alloca and the push - this is the LLVM IR shape, where the
  // const value the push consumes is materialised as its own statement (`ConstStmt` -> `AdStackPushStmt(v = const)`)
  // rather than being inlined as the push's `v` operand from the alloca's previous sibling. The const sitting between
  // them is by construction the same `ConstStmt` `p->v` points to (no other statement is emitted between an
  // autodiff-emitted alloca and its bootstrap push in either pipeline), so we identity-check it to keep the predicate
  // as tight as the SPIR-V-shape variant above.
  if (prev == p->v && idx >= 2 && parent->statements[idx - 2].get() == target) {
    return true;
  }
  return false;
}

// The float-stack predicate folded into the LCA computation: push/load-top/load-top-adj sites where the underlying
// alloca's `ret_type` is real (f32 or f64). Pop sites are deliberately NOT included - they only mutate `count_var` and
// impose no dominance requirement on the row claim.
bool stack_is_float(Stmt *push_or_load) {
  AdStackAllocaStmt *alloca = nullptr;
  if (auto *p = push_or_load->cast<AdStackPushStmt>()) {
    alloca = p->stack ? p->stack->cast<AdStackAllocaStmt>() : nullptr;
  } else if (auto *l = push_or_load->cast<AdStackLoadTopStmt>()) {
    alloca = l->stack ? l->stack->cast<AdStackAllocaStmt>() : nullptr;
  } else if (auto *l = push_or_load->cast<AdStackLoadTopAdjStmt>()) {
    alloca = l->stack ? l->stack->cast<AdStackAllocaStmt>() : nullptr;
  }
  return alloca != nullptr && (alloca->ret_type == PrimitiveType::f32 || alloca->ret_type == PrimitiveType::f64);
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
                                                          const SNodeDescriptorResolver &snode_descriptor_resolver,
                                                          std::size_t sparse_heap_threshold_bytes,
                                                          bool task_range_is_original_loop) {
  StaticAdStackAnalysisResult result;
  if (task_ir == nullptr || task_ir->body == nullptr) {
    return result;
  }

  // First scan: collect alloca strides, classify each push as bootstrap or not, gather f32 push/load-top blocks for the
  // LCA reduce.
  std::vector<Block *> push_side_blocks;
  walk_ir(task_ir->body.get(), [&](Stmt *s) {
    if (auto *alloca = s->cast<AdStackAllocaStmt>()) {
      if (alloca->ret_type == PrimitiveType::f32 || alloca->ret_type == PrimitiveType::f64) {
        // Both f32 and f64 reverse-mode adstacks share the float heap on LLVM. The analyser tracks stride in
        // entry-count units (each entry = primal + adjoint = 2 elements) so the heap footprint scales naturally with
        // `entry_size_bytes` at sizing time. f64 carries 4 bytes/element more than f32; the launcher's
        // `align_up_8(sizeof(int64_t) + entry_size_bytes * max_size)` step in `publish_adstack_metadata` picks up the
        // larger element size automatically. The per-kind byte stride is tracked alongside so the sparse-heap
        // threshold check below stays accurate on f64 allocas (where the entries-unit-times-`sizeof(float)` estimate
        // would underestimate the real heap by 2x).
        result.per_thread_stride_float += 2u * uint32_t(alloca->max_size);
        result.per_thread_stride_float_bytes +=
            2ull * static_cast<uint64_t>(data_type_size(alloca->ret_type)) * static_cast<uint64_t>(alloca->max_size);
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

  // Pairwise LCA reduce. Empty `push_side_blocks` means the task has no f32 adstack push sites and the LCA stays null
  // (the float heap is unbound and no row claim is emitted by the codegen). A single block is its own LCA.
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
      // Both blocks live under the same task-body root, so their ancestor chains converge at that root at the latest.
      // Falling through to nullptr would degrade to the eager (root-block) claim path which is still correct, just
      // non-optimal.
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

  // Second scan: per-stack pushed values, used by the gate matcher to resolve autodiff-spilled gate predicates of shape
  // `IfStmt(cond = AdStackLoadTopStmt(stack=S))` (the gate predicate's bool is spilled onto a u1 adstack in the forward
  // direction and replayed via load_top in the reverse direction).
  std::unordered_map<AdStackAllocaStmt *, std::vector<Stmt *>> per_stack_pushed_values;
  walk_ir(task_ir->body.get(), [&](Stmt *s) {
    if (auto *push = s->cast<AdStackPushStmt>()) {
      if (auto *alloca = push->stack ? push->stack->cast<AdStackAllocaStmt>() : nullptr) {
        per_stack_pushed_values[alloca].push_back(push->v);
      }
    }
  });

  // Compile-time loop trip count of the analyzed task, used by the runtime to clip the reducer's gate-passing-cell
  // count down to the actual maximum row claim count: each iteration claims at most one row at the LCA-block, so
  // `loop_iter` is a sound upper bound on heap rows regardless of how oversized the gating SNode / ndarray is.
  // Zero when:
  //   * the task's loop is runtime-bounded (`task_ir->end_stmt != nullptr` or non-const begin / end - the analyzer
  //     cannot resolve the trip count from IR alone), or
  //   * `task_range_is_original_loop` is false (the caller signals that the task's `[begin_value, end_value)` is a
  //     post-chunking subrange, e.g. CPU LLVM after `make_cpu_multithreaded_range_for` rewrote a 256-iteration
  //     range-for into 16 parallel chunks of 16; the chunked subrange would massively undersize the heap clip
  //     because the atomic row counter is shared across all chunks of the same task).
  // In both cases the runtime falls back to the unclipped reducer count.
  const bool task_static_bound = task_ir->const_begin && task_ir->const_end && task_ir->end_stmt == nullptr;
  const int64_t task_loop_iter =
      task_static_bound ? (static_cast<int64_t>(task_ir->end_value) - static_cast<int64_t>(task_ir->begin_value)) : 0;
  const uint32_t loop_iter_static_for_bound_expr =
      (task_range_is_original_loop && task_loop_iter > 0 &&
       static_cast<uint64_t>(task_loop_iter) <= std::numeric_limits<uint32_t>::max())
          ? static_cast<uint32_t>(task_loop_iter)
          : 0;
  // Snapshot the task's pre-chunking loop trip-count `SizeExpr` (populated by `determine_ad_stack_size` in
  // `compile_to_offloads`, before `make_cpu_multithreaded_range_for` rewrites the loop on CPU). Empty when
  // the task is not a range-for, the SizeExpr grammar could not bound the trip count, or the trip count is
  // already a `Const` (in which case `loop_iter_static` carries the same value at zero per-launch cost).
  // Runtime-bounded shapes like `for j in range(field[i])` are the actual reason this exists; encoding
  // them here moves the trip-count resolution off the compile-time path and onto the same per-launch
  // `evaluate_adstack_size_expr` walk the per-thread stack-sizer already runs. The runtime evaluates this
  // ONLY when `loop_iter_static == 0`, so compile-time-known loops never pay the per-launch eval cost.
  SerializedSizeExpr loop_iter_size_expr_for_bound_expr;
  if (task_ir->pre_chunk_loop_trip_count_expr &&
      task_ir->pre_chunk_loop_trip_count_expr->kind != SizeExpr::Kind::Const) {
    loop_iter_size_expr_for_bound_expr = task_ir->pre_chunk_loop_trip_count_expr->serialize();
  }

  // Resolve a `GlobalLoadStmt::src` chain to a captured field source. Returns true on a recognized shape (ndarray
  // ext-ptr or SNode root->dense->place(scalar)); on success populates the source-kind-specific fields of `out`.
  auto match_field_source = [&](Stmt *load_src, StaticAdStackBoundExpr &out) -> bool {
    if (auto *ext = load_src->cast<ExternalPtrStmt>()) {
      if (auto *base_arg = ext->base_ptr->cast<ArgLoadStmt>()) {
        // Validate the gate's index expression: every axis must be a `LoopIndexStmt`. Anything more complex
        // (`selector[i % 5]`, `selector[42]`, `selector[2 * i]`, `selector[i + 1]`, `selector[other_field[i]]`) would
        // have the reducer walk `selector[0..length)` and count gate-passing cells on a different index basis than the
        // main pass's LCA-block atomic-rmw, causing the reducer count to diverge from the actual claim count and either
        // undersize the heap (silent gradient corruption on LLVM, hard overflow on SPIR-V) or oversize it. Plain
        // `selector[i]` (one axis = one `LoopIndexStmt`) is the only shape the reducer's flat-walk semantics matches.
        for (Stmt *idx : ext->indices) {
          if (idx == nullptr || !idx->is<LoopIndexStmt>()) {
            return false;
          }
        }
        out.field_source_kind = StaticAdStackBoundExpr::FieldSourceKind::NdArray;
        out.loop_iter_static = loop_iter_static_for_bound_expr;
        out.loop_iter_size_expr = loop_iter_size_expr_for_bound_expr;
        out.ndarray_arg_id = base_arg->arg_id;
        // Capture the gating ndarray's ndim so the host launcher can walk shape[0..ndim) at dispatch time and product
        // them into the reducer's flat-element walk bound. Without this the launcher would have to fall back to
        // `ctx.array_runtime_sizes[arg_id]`, which carries different units depending on whether the caller used
        // `set_arg_external_array_with_shape` (bytes) or `set_args_ndarray` (element count) - the latter would
        // undercount by `sizeof(elem)` for `qd.ndarray` arguments and silently corrupt gradients on every kernel that
        // goes through the gating path with a `qd.ndarray` selector.
        out.ndarray_ndim = static_cast<int>(ext->indices.size());
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
      // Iteration-count check (statically-bounded loops only): reject when the task's loop iterates more times
      // than the SNode has cells. The reducer walks `desc_opt->iter_count` cells and counts gate-passing ones,
      // the main pass claims a heap row per gated iteration, so a mismatch under-allocates the heap and aliases
      // excess iterations onto a shared row. Structural signature of `for i in range(n): if field[i % K] > eps:
      // <push f32>` (K < n, field on a K-cell SNode): `loop_iter = n > K = snode_iter_count`. Legitimate
      // multi-axis kernels of the shape `for ii, jj, kk, ib in qd.ndrange(...): if grid[f, ii, jj, kk, ib] >
      // eps:` keep `loop_iter <= snode_iter_count` (slice axes like the kernel-arg `f` make the reducer
      // over-count by the slice factor, which is benign over-allocation).
      const bool static_bound = task_ir->const_begin && task_ir->const_end && task_ir->end_stmt == nullptr;
      const int64_t loop_iter =
          static_bound ? (static_cast<int64_t>(task_ir->end_value) - static_cast<int64_t>(task_ir->begin_value)) : 0;
      if (static_bound) {
        if (loop_iter <= 0 || static_cast<uint64_t>(loop_iter) > static_cast<uint64_t>(desc_opt->iter_count)) {
          return false;
        }
      }
      // Per-axis classification on the gate's `LinearizeStmt::inputs`. After `lower_access` the input_index is
      // either a `LinearizeStmt` (StructFor path preserves it) or an `add`/`mul` arithmetic tree (ndrange path
      // expanded form); a recursive collector recovers per-axis components from either shape. For each axis,
      // walk through `BinaryOpStmt` / `UnaryOpStmt` looking for a `LoopIndexStmt` to classify as iterating; pure
      // `ConstStmt` / `ArgLoadStmt` axes are slices. Reject when:
      //   * `n_iterating == 0`: every axis is loop-invariant (`field[42]`, `field[arg]`, `field[other_field[i]]`).
      //   * `n_iterating == 1 && n_bare_iterating == 0`: single non-bare iterating axis (`field[i / 2]`,
      //     `field[i % K]`, `field[i + 5]`). Conservatively rejects bijective shifts too; the worst-case heap
      //     stays correct, just larger.
      // Multi-axis cases with `n_iterating >= 2` are accepted: the canonical `qd.ndrange(*shape)` decomposes a
      // single linear loop index into multiple axes via floordiv / mod chains whose joint mapping is bijective.
      auto *lookup = getch->input_ptr ? getch->input_ptr->cast<SNodeLookupStmt>() : nullptr;
      if (lookup == nullptr || lookup->input_index == nullptr) {
        return false;
      }
      std::vector<Stmt *> axes;
      if (auto *lin = lookup->input_index->cast<LinearizeStmt>()) {
        axes.assign(lin->inputs.begin(), lin->inputs.end());
      } else {
        std::function<void(Stmt *)> collect_axes = [&](Stmt *s) {
          if (auto *bin = s->cast<BinaryOpStmt>()) {
            if (bin->op_type == BinaryOpType::add) {
              collect_axes(bin->lhs);
              collect_axes(bin->rhs);
              return;
            }
            if (bin->op_type == BinaryOpType::mul) {
              if (bin->rhs && bin->rhs->is<ConstStmt>()) {
                axes.push_back(bin->lhs);
                return;
              }
              if (bin->lhs && bin->lhs->is<ConstStmt>()) {
                axes.push_back(bin->rhs);
                return;
              }
            }
          }
          axes.push_back(s);
        };
        collect_axes(lookup->input_index);
      }
      std::function<bool(Stmt *, int)> contains_loop_index = [&](Stmt *s, int depth) -> bool {
        if (s == nullptr || depth > 8) {
          return false;
        }
        if (s->is<LoopIndexStmt>()) {
          return true;
        }
        if (auto *bin = s->cast<BinaryOpStmt>()) {
          return contains_loop_index(bin->lhs, depth + 1) || contains_loop_index(bin->rhs, depth + 1);
        }
        if (auto *un = s->cast<UnaryOpStmt>()) {
          return contains_loop_index(un->operand, depth + 1);
        }
        // The reverse task replays multi-axis loop indices by spilling them onto per-axis adstacks during the forward
        // pass and loading via `AdStackLoadTopStmt` in the reverse pass. The push (in the forward `OffloadedStmt`) and
        // the load (in the reverse `OffloadedStmt`) sit in different tasks, so the per-task `per_stack_pushed_values`
        // map cannot trace from this load back to the `LoopIndexStmt` source. Treat the load as iterating directly:
        // autodiff only spills runtime-varying values onto adstacks, and per-axis adstacks - the only ones reaching a
        // `LinearizeStmt::input` - carry replayed loop indices by construction.
        if (s->is<AdStackLoadTopStmt>()) {
          return true;
        }
        return false;
      };
      // Reject fold-attack shapes like `field[i % 2, i % 2]` where two iterating axes evaluate to the same
      // value, causing multiple pushes per outer-loop iteration to alias onto the same SNode cell and
      // undersize the heap. Use `same_value` rather than pointer-identity so an attacker cannot bypass the
      // check by inserting a no-op like `i % 2 + 0 - 0` that defeats CSE: the value-equivalence walk
      // collapses arithmetic identities the upstream simplifier missed. The canonical `qd.ndrange(*shape)`
      // decomposition produces axes with structurally different values (`i // K0`, `(i % K0) // K1`,
      // `i % K1`) even though every axis is rooted at the same `LoopIndexStmt`, so this admits the
      // joint-bijective ndrange shape uniformly across LLVM and SPIR-V backends.
      int n_iterating = 0;
      int n_bare_iterating = 0;
      std::vector<Stmt *> distinct_iterating_axes;
      for (Stmt *axis : axes) {
        if (contains_loop_index(axis, 0)) {
          n_iterating++;
          if (axis->is<LoopIndexStmt>()) {
            n_bare_iterating++;
          }
          bool already_seen = false;
          for (Stmt *prev : distinct_iterating_axes) {
            if (prev == axis || irpass::analysis::same_value(prev, axis)) {
              already_seen = true;
              break;
            }
          }
          if (!already_seen) {
            distinct_iterating_axes.push_back(axis);
          }
        }
      }
      if (n_iterating == 0) {
        return false;
      }
      if (n_iterating == 1 && n_bare_iterating == 0) {
        return false;
      }
      if (static_cast<int>(distinct_iterating_axes.size()) < n_iterating) {
        return false;
      }
      // Joint-axis-space check: when no iterating axis is the task loop's bare `LoopIndexStmt` (which would
      // make the joint mapping bijective by itself), bound the product of axis value ranges from above and
      // require it to cover the loop trip count. Without this an oversized SNode lets a fold attack like
      // `field[i % 8, (i // 8) % 8]` with `loop_iter > 64` slip through: every iterating axis is value-
      // distinct (so the same_value dedup admits the shape) and `loop_iter <= snode_iter_count` because the
      // SNode is large, yet the joint mapping wraps and aliases iterations onto a 64-cell subspace,
      // undersizing the float heap. Each axis range is recovered by walking the lowered arithmetic for
      // `_ % K`, `_ // K`, and the `sub(_, mul/bit_shl(floordiv(_, K), K))` post-`lower_access` shape; an
      // unrecognised shape contributes the parent's range conservatively. The check is gated on
      // `static_bound` because the trip count needs to be a known constant for the comparison to be sound.
      std::function<int64_t(Stmt *, int)> axis_max_range = [&](Stmt *s, int depth) -> int64_t {
        if (s == nullptr || depth > 12) {
          return loop_iter;
        }
        if (s->is<LoopIndexStmt>()) {
          return loop_iter;
        }
        if (auto *bin = s->cast<BinaryOpStmt>()) {
          if (bin->op_type == BinaryOpType::mod) {
            if (auto *c = bin->rhs ? bin->rhs->cast<ConstStmt>() : nullptr) {
              const int64_t k = c->val.val_as_int64();
              if (k > 0) {
                return std::min<int64_t>(k, axis_max_range(bin->lhs, depth + 1));
              }
            }
          }
          if (bin->op_type == BinaryOpType::floordiv) {
            if (auto *c = bin->rhs ? bin->rhs->cast<ConstStmt>() : nullptr) {
              const int64_t k = c->val.val_as_int64();
              if (k > 0) {
                const int64_t parent = axis_max_range(bin->lhs, depth + 1);
                return (parent + k - 1) / k;
              }
            }
          }
          if (bin->op_type == BinaryOpType::sub) {
            // Post-`lower_access` `_ % K` is `sub(L, mul(floordiv(L, K), K))` (or `bit_shl` for power-of-two K).
            if (auto *factor = bin->rhs ? bin->rhs->cast<BinaryOpStmt>() : nullptr) {
              int64_t k_factor = -1;
              Stmt *div_inner = nullptr;
              if (factor->op_type == BinaryOpType::mul) {
                if (auto *c = factor->rhs ? factor->rhs->cast<ConstStmt>() : nullptr) {
                  k_factor = c->val.val_as_int64();
                  div_inner = factor->lhs;
                }
              } else if (factor->op_type == BinaryOpType::bit_shl) {
                if (auto *c = factor->rhs ? factor->rhs->cast<ConstStmt>() : nullptr) {
                  const int64_t shift = c->val.val_as_int64();
                  if (shift >= 0 && shift < 62) {
                    k_factor = (int64_t)1 << shift;
                    div_inner = factor->lhs;
                  }
                }
              }
              if (k_factor > 0 && div_inner != nullptr) {
                if (auto *div = div_inner->cast<BinaryOpStmt>(); div && div->op_type == BinaryOpType::floordiv) {
                  if (auto *div_c = div->rhs ? div->rhs->cast<ConstStmt>() : nullptr) {
                    if (div_c->val.val_as_int64() == k_factor && div->lhs == bin->lhs) {
                      return std::min<int64_t>(k_factor, axis_max_range(bin->lhs, depth + 1));
                    }
                  }
                }
              }
            }
            return axis_max_range(bin->lhs, depth + 1);
          }
        }
        return loop_iter;
      };
      const bool any_task_loop_bare_index = std::any_of(axes.begin(), axes.end(), [&](Stmt *axis) {
        auto *li = axis->cast<LoopIndexStmt>();
        return li != nullptr && li->loop == task_ir;
      });
      if (static_bound && !any_task_loop_bare_index) {
        constexpr int64_t kRangeCap = std::numeric_limits<int64_t>::max() / 2;
        int64_t joint_product = 1;
        for (Stmt *axis : axes) {
          if (!contains_loop_index(axis, 0)) {
            continue;
          }
          int64_t r = axis_max_range(axis, 0);
          if (r <= 0) {
            r = loop_iter;
          }
          if (r > kRangeCap || joint_product > kRangeCap / std::max<int64_t>(r, 1)) {
            joint_product = kRangeCap;
            break;
          }
          joint_product *= r;
          if (joint_product >= loop_iter) {
            break;
          }
        }
        if (joint_product < loop_iter) {
          return false;
        }
      }
      out.field_source_kind = StaticAdStackBoundExpr::FieldSourceKind::SNode;
      out.snode_root_id = desc_opt->root_id;
      out.snode_byte_base_offset = desc_opt->byte_base_offset;
      out.snode_byte_cell_stride = desc_opt->byte_cell_stride;
      out.snode_iter_count = desc_opt->iter_count;
      out.loop_iter_static = loop_iter_static_for_bound_expr;
      out.loop_iter_size_expr = loop_iter_size_expr_for_bound_expr;
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
          // More than one non-const push - the gate's logical value depends on which path executed, and the reducer
          // cannot mirror that without re-emitting the full forward IR. Fall through to worst-case sizing.
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
    // (e.g. `if literal < field[i]`). The symmetric form gets the comparison op flipped so the runtime reducer always
    // evaluates `field cmp literal` against the captured `literal_*`.
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
      out.field_dtype_is_double = false;
      out.literal_f32 = cst->val.val_f32;
      return true;
    }
    if (cst->val.dt->is_primitive(PrimitiveTypeID::f64)) {
      out.field_dtype_is_float = true;
      out.field_dtype_is_double = true;
      out.literal_f64 = cst->val.val_f64;
      return true;
    }
    if (cst->val.dt->is_primitive(PrimitiveTypeID::i32)) {
      out.field_dtype_is_float = false;
      out.field_dtype_is_double = false;
      out.literal_i32 = cst->val.val_i32;
      return true;
    }
    // Other types (i64 / etc.) fall through; the reducer kernel never has to dispatch on heterogeneous literal kinds.
    return false;
  };

  // Walk the chain from LCA up to the task body root, collecting IfStmt gates. RangeForStmt / StructForStmt /
  // MeshForStmt / WhileStmt / OffloadedStmt parents are skipped (iterators sweep threads rather than gating them; the
  // offload boundary is the kernel entry). Anything else aborts the chain - unfamiliar control-flow structures might
  // gate threads in ways the reducer cannot mirror.
  int gate_count = 0;
  bool chain_ok = true;
  StaticAdStackBoundExpr captured;
  Stmt *gate_index_owning_loop = nullptr;
  Stmt *first_iter_loop_above_lca = nullptr;
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
      // Find the gate index's owning loop. The gate condition has the shape `field[i] cmp lit` (or the symmetric form
      // `lit cmp field[i]`) where `i` is a `LoopIndexStmt` (validated by `match_field_source` and the SNode arm). Pull
      // the first index off the matched source so the chain check below can verify the gate is sweeping the FIRST
      // iter-loop above the LCA, not a nested-deeper one.
      if (auto *bin = if_stmt->cond->cast<BinaryOpStmt>()) {
        // Probe both operands: the matcher above accepts both `load cmp const` and `const cmp load`, so the load can
        // sit on either side. Picking only `bin->lhs` would bypass the validation on the symmetric form
        // (`gate_index_owning_loop` stays null, the inequality check below short-circuits, and a nested-loop gate slips
        // through).
        GlobalLoadStmt *gl = bin->lhs->cast<GlobalLoadStmt>();
        if (gl == nullptr) {
          gl = bin->rhs->cast<GlobalLoadStmt>();
        }
        if (gl != nullptr) {
          if (auto *ext = gl->src->cast<ExternalPtrStmt>()) {
            if (!ext->indices.empty()) {
              if (auto *li = ext->indices[0]->cast<LoopIndexStmt>()) {
                gate_index_owning_loop = li->loop;
              }
            }
          } else if (auto *getch = gl->src->cast<GetChStmt>()) {
            // SNode-backed gates use `for i in field` where `i` is a `LoopIndexStmt` of the enclosing for-loop, and the
            // access lowers to a `GetChStmt` chained off the loop index. Walk up to the original `LoopIndexStmt`
            // operand so the validation below has the same gate-index-owning-loop signal as the ndarray arm. The
            // `getch->input_snode` field would name the parent SNode but does not carry the loop binding; the load
            // chain's input statement does.
            for (Stmt *cur = getch->input_ptr; cur != nullptr;) {
              if (auto *li = cur->cast<LoopIndexStmt>()) {
                gate_index_owning_loop = li->loop;
                break;
              }
              if (auto *child = cur->cast<GetChStmt>()) {
                cur = child->input_ptr;
                continue;
              }
              if (auto *lookup = cur->cast<SNodeLookupStmt>()) {
                cur = lookup->input_index;
                continue;
              }
              if (auto *lin = cur->cast<LinearizeStmt>()) {
                if (!lin->inputs.empty()) {
                  cur = lin->inputs[0];
                  continue;
                }
              }
              break;
            }
          }
        }
      }
    } else if (parent->is<RangeForStmt>() || parent->is<StructForStmt>() || parent->is<MeshForStmt>() ||
               parent->is<WhileStmt>() || parent->is<OffloadedStmt>()) {
      if (first_iter_loop_above_lca == nullptr) {
        first_iter_loop_above_lca = parent;
      }
      continue;
    } else {
      chain_ok = false;
      break;
    }
  }
  // Defensive validation: when a gate is captured, the gate-index `LoopIndexStmt`'s owning loop must be the FIRST
  // iter-loop encountered when walking from the LCA toward the root. Nested-loop patterns of the form `for t in
  // range(M): for i in range(N): if active[i] > 0:` would otherwise have the reducer count gate-passing cells in
  // `active` once (= K), but the LCA-block atomic-rmw fires `M * K` times across the outer-iter dispatched threads;
  // rows past K alias onto row K-1 and reverse-mode gradients silently diverge. Reject and fall through to the
  // dispatched-threads worst case rather than silently mis-sizing.
  //
  // Reachability: on every Python kernel pattern observed today, this branch is unreachable - the autodiff transform
  // emits the forward-pass float pushes inside the forward IfStmt's `true_statements` block and the reverse-pass float
  // load_top / load_top_adj / pop sites inside a SEPARATE reverse IfStmt's `true_statements` block, so the LCA reduce
  // collapses up to the offload body (the common ancestor of two distinct `if_true` blocks) for any kernel where the
  // gate sits inside an inner for-loop that is NOT the offload itself. With the LCA at the offload body, the chain walk
  // above terminates at the OffloadedStmt without ever incrementing `gate_count`, so `bound_expr` is not captured and
  // this validation does not run. Single-loop kernels where the offload IS the gating for-loop combine forward and
  // reverse under a single shared IfStmt instead, so the LCA stays inside the gate and the capture succeeds; in that
  // shape `gate_index_owning_loop` equals the offload's RangeForStmt which is also `first_iter_loop_above_lca`, so the
  // inequality below is false and the validation again does not reject. The branch is therefore live only on a
  // hypothetical autodiff refactor that combines fwd / rev under one IfStmt for nested-loop kernels too, plus it
  // documents the required invariant for that future shape.
  if (chain_ok && gate_count == 1) {
    if (gate_index_owning_loop != nullptr && first_iter_loop_above_lca != nullptr &&
        gate_index_owning_loop != first_iter_loop_above_lca) {
      chain_ok = false;
    }
  }
  if (chain_ok && gate_count == 1) {
    // Latency-vs-memory threshold: capturing `bound_expr` routes the task through the lazy LCA-block atomic-rmw row
    // claim, which costs a runtime reducer compute-shader dispatch + per-task device-to-host capacity readback at
    // every kernel launch. The savings are proportional to the `dispatched_threads * stride_float * sizeof(float)`
    // worst-case heap allocation the lazy path replaces; below the configured threshold the conservative eager
    // allocation is cheap enough that the reducer's per-launch overhead dominates and the backward pass slows down.
    // Skip the capture in that regime so the codegen falls back to the eager `linear_thread_idx * stride` mapping
    // (no LCA-block atomic, no reducer dispatch, no host-side per-task DtoH per launch). Threads bound at the
    // SPIR-V grid-stride advisory cap (`kMaxNumThreadsGridStrideLoop = 131072`) - the larger of the two backend
    // ceilings (LLVM CUDA / AMDGPU floor at 65536 via `kAdStackMaxConcurrentThreads` in the launchers); using the
    // SPIR-V ceiling keeps the test tight on both. `per_thread_stride_float_bytes` is the real per-thread byte cost
    // (`2 * sizeof(dtype) * max_size` per alloca, summed across every f32 / f64 alloca in the task) - tracking
    // bytes directly rather than scaling the entries-unit `per_thread_stride_float` by `sizeof(float)` keeps the
    // threshold check accurate on f64 allocas, where the entries-unit estimate would undersize by 2x.
    // Threshold default lives in `CompileConfig::ad_stack_sparse_threshold_bytes` (100 MiB); set to 0 to always
    // capture (tests that pin the reducer-backed sizing path) or to a very large value to always disable.
    constexpr size_t kAdvisoryThreadsCeiling = 131072;
    const size_t conservative_heap_bytes_upper =
        static_cast<size_t>(result.per_thread_stride_float_bytes) * kAdvisoryThreadsCeiling;
    if (conservative_heap_bytes_upper >= sparse_heap_threshold_bytes) {
      result.bound_expr = captured;
    }
  }
  return result;
}

}  // namespace quadrants::lang
