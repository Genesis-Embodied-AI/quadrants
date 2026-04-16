#pragma once

#include "quadrants/ir/ir.h"

namespace quadrants::lang {
namespace irpass {

// Replace precise-tagged UnaryOpStmt/BinaryOpStmt with opaque InternalFuncStmt wrappers
// so that subsequent optimization passes (alg_simp, binary_op_simplify, demote_operations,
// constant_fold, CSE, etc.) cannot see or rewrite them. Must be called before optimization.
void fence_precise_ops(IRNode *root);

// Lower the opaque wrappers back to real UnaryOpStmt/BinaryOpStmt, preserving the
// kDisableFastMath codegen hint so backends emit them without fast-math. Must be called
// after optimization, before codegen.
void unfence_precise_ops(IRNode *root);

}  // namespace irpass
}  // namespace quadrants::lang
