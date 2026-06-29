#pragma once

namespace quadrants::lang {

class Block;

// Stage: Forward-mode codegen. Emit a tangent-propagating dual pass for `block` (the kernel root for forward-mode
// autodiff).
void make_dual(Block *block);

}  // namespace quadrants::lang
