#pragma once

namespace quadrants::lang {

class Block;

// Stage: Reverse-mode codegen. Emit the reverse pass IR in `ib` (an Independent Block produced by
// identify_independent_blocks). Forward IR must already have been shaped by ir_shaping + forward_state_spill.
void make_adjoint(Block *ib);

}  // namespace quadrants::lang
