#pragma once

#include <unordered_map>

#include "quadrants/ir/statements.h"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"

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

// Like get_quadrants_uint_type but returns at least u32 so that the result
// is usable with atomic ops (Metal/Vulkan lack 16-bit atomics).
DataType get_atomic_uint_dtype(IRBuilder &ir, const DataType &dt);

// Convert a SPIR-V value from shared-memory uint backing to float dt.
// Handles the width mismatch when the backing type is wider (e.g. u32
// for f16): narrows to same-width uint first, then bitcasts to float.
Value shared_uint_to_float(IRBuilder &ir, Value val, const DataType &dt);

// Convert a SPIR-V value from float dt to shared-memory uint backing.
// Bitcasts to same-width uint, then widens if the backing type is wider.
Value float_to_shared_uint(IRBuilder &ir, Value val, const DataType &dt);

// CAS-based float atomic for shared (workgroup) arrays. Unlike
// IRBuilder::float_atomic, this handles width-mismatched uint backing
// (e.g. u32 backing for f16 arrays, since Metal/Vulkan lack 16-bit atomics).
Value shared_float_atomic(IRBuilder &ir,
                          AtomicOpType op_type,
                          Value addr_ptr,
                          Value data,
                          const DataType &dt);

}  // namespace spirv
}  // namespace quadrants::lang
