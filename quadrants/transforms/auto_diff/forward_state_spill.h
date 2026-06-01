#pragma once

namespace quadrants::lang {

class Block;

// Stage: Forward-state spill. Pre-MakeAdjoint passes that prepare the forward IR so the reverse pass can re-read the
// per-iteration values it needs.

// Hoist forward-pass SSA defs that the reverse pass will re-read (operands of non-linear ops, indices of GlobalPtr /
// ExternalPtr, range-for bounds, if conds) into AllocaStmt + LocalStore + LocalLoad. Demand-driven: defs no reverse
// formula reads stay in pure SSA form.
void promote_ssa_to_local_var(Block *block);

// Replace each AllocaStmt the reverse pass needs to read across iterations with an AdStackAllocaStmt + AdStackPushStmt
// / AdStackLoadTopStmt. The AllocaJudger (file-private) decides which allocas need the promotion.
void replace_local_var_with_stacks(Block *block, int ad_stack_size);

// Drop AdStackAllocaStmts whose pushed value is recomputable from already-stack-backed allocas + kernel args +
// constants + loop indices. Trades forward-pass adstack memory traffic for cloned arithmetic in the reverse scope,
// which `BackupSSA::generic_visit` synthesizes on demand via the shared RecomputableChainCloner. Must run after
// `replace_local_var_with_stacks` (so the analyzer sees the AdStack shape) and before `make_adjoint` (so the reverse
// pass is generated against the cleaned forward IR).
void eliminate_recomputable_ad_stack_pushes(Block *ib);

}  // namespace quadrants::lang
