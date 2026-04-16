#pragma once

#include "quadrants/ir/expr.h"

namespace quadrants::lang {

// Recursively tag every BinaryOp and UnaryOp expression in `input`'s subtree as `precise`:
// IEEE-strict evaluation in source order, with no reassociation, FMA contraction,
// approximate-transcendental substitution, or algebraic simplification, regardless of the
// module-level `fast_math` setting. Mirrors MSL/HLSL `precise`.
//
// Recursion descends through BinaryOp / UnaryOp / TernaryOp wrappers and stops at any other
// expression kind (loads, constants, qd.func calls, ndarray accesses, ...). The tag is
// propagated from Expression to Stmt by each class's `flatten()`.
Expr precise(const Expr &input);

}  // namespace quadrants::lang
