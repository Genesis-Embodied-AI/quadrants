#pragma once

#include <unordered_map>

#include "quadrants/ir/statements.h"

namespace quadrants::lang {
namespace spirv {

// Pre-scan the IR block tree to find shared float AllocaStmts targeted by
// atomic operations. These arrays may need uint-backing so that CAS-based
// atomic emulation can use integer atomics (Metal/MoltenVK lack threadgroup
// float atomics).
//
// out[alloca] = true  → has non-add atomic ops, CAS needed unconditionally
// out[alloca] = false → only add ops, native shared float atomics can be used
//                       if the device supports them
void scan_shared_atomic_allocs(Block *ir_block,
                               std::unordered_map<const Stmt *, bool> &out);

}  // namespace spirv
}  // namespace quadrants::lang
