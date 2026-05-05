#include "quadrants/transforms/auto_diff/forward_state_spill.h"
#include "quadrants/transforms/auto_diff/ir_shaping.h"
#include "quadrants/transforms/auto_diff/make_adjoint.h"
#include "quadrants/transforms/auto_diff/make_dual.h"
#include "quadrants/transforms/auto_diff/post_adjoint_cleanup.h"
#include "quadrants/transforms/auto_diff/validation.h"

#include "quadrants/ir/analysis.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/transforms.h"

namespace quadrants::lang {

namespace irpass {

void auto_diff(IRNode *root, const CompileConfig &config, AutodiffMode autodiff_mode, bool use_stack) {
  QD_AUTO_PROF;
  if (autodiff_mode == AutodiffMode::kReverse) {
    if (auto *root_block = root->cast<Block>()) {
      // Top-level range-fors at the kernel root become the offload-level loop. Walk each direct child once and refuse
      // cross-iteration global RAW on a needs_grad SNode before the AD transformation runs - any diagnostic from inside
      // `MakeAdjoint` would already point at the partially-rewritten reverse IR and be much harder to read. StructFor
      // offloads are out of scope: their loop variable is an SNode index produced by the runtime, not a user-controlled
      // integer arithmeticed against itself, so the `out[i-1]` shape this guard targets does not arise.
      for (auto &s : root_block->statements) {
        if (auto *rf = s->cast<RangeForStmt>())
          offload_level_global_cross_iter_raw_check(rf->body.get());
      }
    }
    regulate_tensor_typed_statements(root);
    if (use_stack) {
      auto IB = identify_independent_blocks(root);
      reverse_outer_loops(root, IB);

      for (auto ib : IB) {
        promote_ssa_to_local_var(ib);
        replace_local_var_with_stacks(ib, config.ad_stack_size);
        type_check(root, config);

        // Disabled: `RecomputableChainAnalyzer` admits `GlobalLoadStmt` as a recomputable interior node without
        // verifying SNode read-only-ness. When the chain reaches a `GlobalLoadStmt` of an SNode written elsewhere in
        // the same kernel execution, `BackupSSA`'s reverse-side re-clone reads post-write state instead of the iter-k
        // value the original `AdStackLoadTopStmt` would have returned, silently corrupting gradients on MPM-style
        // kernels that mix per-particle reads with grid writes. Re-enable once the chain analyzer rejects
        // mutated-SNode loads (or the cost model gates on read-only-verified leaves).
        // eliminate_recomputable_ad_stack_pushes(ib);
        type_check(root, config);

        make_adjoint(ib);
        type_check(root, config);
        backup_ssa(ib);
        // After MakeAdjoint emits the reverse-pass body, an outer-loop-invariant value pulled into the reverse
        // direction by `accumulate_unary_operand_checked` becomes a fresh `AdStackLoadTopStmt` per user-side use; in
        // straight-line unrolled IR those reads coalesce to one load per stack per block, which the dedicated pass
        // handles before the IR reaches `irpass::analysis::verify_if_debug`.
        coalesce_ad_stack_loads(ib);
        irpass::analysis::verify_if_debug(root, config);
      }
    } else {
      auto IB = identify_independent_blocks(root);
      reverse_outer_loops(root, IB);
      type_check(root, config);
      for (auto ib : IB) {
        make_adjoint(ib);
        type_check(root, config);
        backup_ssa(ib);
        coalesce_ad_stack_loads(ib);
        irpass::analysis::verify_if_debug(root, config);
      }
    }
  } else if (autodiff_mode == AutodiffMode::kForward) {
    // Forward mode autodiff
    Block *block = root->as<Block>();
    make_dual(block);
  }
  type_check(root, config);
  irpass::analysis::verify_if_debug(root, config);
}

}  // namespace irpass

}  // namespace quadrants::lang
