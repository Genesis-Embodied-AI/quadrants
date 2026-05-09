#pragma once

namespace quadrants::lang {

class IRNode;
class Block;

// Stage: Post-MakeAdjoint cleanup.

// Spill cross-block SSA operands the reverse-mode clones reference, so the reverse IR is verifier-clean: AdStackLoadTop
// / ArgLoad get cloned in place, AdStackAlloca gets re-rooted, generic SSA values get a per-IB backup AllocaStmt +
// LocalStore + LocalLoad chain.
void backup_ssa(Block *block);

// Within a single straight-line block, dedup repeated AdStackLoadTop / AdStackLoadTopAdj reads of the same stack with
// no intervening Push / Pop / AccAdjoint. Returns whether any IR mutation took place.
bool coalesce_ad_stack_loads(IRNode *root);

}  // namespace quadrants::lang
