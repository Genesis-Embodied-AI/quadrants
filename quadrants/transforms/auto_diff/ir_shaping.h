#pragma once

#include <set>

namespace quadrants::lang {

class IRNode;
class Block;

// Stage: IR shaping. Pre-MakeAdjoint passes that mutate or analyze the forward IR so the downstream reverse-mode
// codegen sees the expected shape.

// Rewrite tensor-typed LocalStore / GlobalStore through MatrixPtr into an explicit gather + matrix-init + scalar store
// - downstream passes assume stores never touch a sub-tensor in place.
void regulate_tensor_typed_statements(IRNode *root);

// Discover every Independent Block (loop body whose iterations carry no dependency on previous iterations or outer
// scopes) under `root`. Returns the set of blocks where MakeAdjoint will emit the reverse pass.
std::set<Block *> identify_independent_blocks(IRNode *root);

// Flip iteration direction of every outer (non-IB) RangeForStmt so the reverse pass walks the iteration trace backward;
// reorders sibling for-loops inside non-IB container blocks accordingly.
void reverse_outer_loops(IRNode *root, const std::set<Block *> &IB);

}  // namespace quadrants::lang
