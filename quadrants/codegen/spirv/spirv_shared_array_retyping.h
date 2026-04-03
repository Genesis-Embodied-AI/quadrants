#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "quadrants/ir/statements.h"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {
namespace spirv {

// Pre-scan the IR block tree to find shared float AllocaStmts targeted by
// atomic operations. These arrays may need uint-backing so that CAS-based
// atomic emulation can use integer atomics (Metal/MoltenVK lack threadgroup
// float atomics).
//
// out[alloca] = true  -> has non-add atomic ops, CAS needed unconditionally
// out[alloca] = false -> only add ops, native shared float atomics can be used
//                        if the device supports them
void scan_shared_atomic_allocs(Block *ir_block,
                               std::unordered_map<const Stmt *, bool> &out);

// Flatten nested tensor types and optionally retype to uint for shared float
// atomics. Returns (elem_count, element_stype) ready for array construction.
std::pair<uint32_t, SType> prepare_shared_alloca_type(
    IRBuilder &ir,
    const DeviceCapabilityConfig &caps,
    const AllocaStmt *alloca,
    const TensorType *tensor_type,
    const std::unordered_map<const Stmt *, bool> &alloc_map,
    std::unordered_set<const Stmt *> &retyped_stmts);

// Flatten nested tensor types in dt, then if origin is in retyped_stmts,
// propagate retyping to stmt and return the uint-backed element SType.
// Otherwise return the default SType for dt.
SType maybe_retype_derived_ptr(IRBuilder &ir,
                               const Stmt *origin,
                               const Stmt *stmt,
                               DataType &dt,
                               std::unordered_set<const Stmt *> &retyped_stmts);

// Load from a uint-backed shared float pointer: loads as uint, bitcasts to
// float. Only call when ptr is known to be in retyped_stmts.
Value load_uint_backed_shared_float(IRBuilder &ir,
                                    Value ptr_val,
                                    const DataType &element_type);

// Convert a float value to uint for storing into a uint-backed shared array.
// Only call when dest is known to be in retyped_stmts.
Value float_to_shared_uint(IRBuilder &ir, Value val, const DataType &dt);

// CAS-based float atomic for shared (workgroup) arrays. Unlike
// IRBuilder::float_atomic, this handles width-mismatched uint backing
// (e.g. u32 backing for f16 arrays, since Metal/Vulkan lack 16-bit atomics).
Value shared_float_atomic(IRBuilder &ir,
                          AtomicOpType op_type,
                          Value addr_ptr,
                          Value data,
                          const DataType &dt);

// Check whether the device has native float atomic add for dt.
// When is_shared=true, checks shared/workgroup capabilities;
// when is_shared=false, checks buffer capabilities.
bool has_native_float_atomic_add(const DeviceCapabilityConfig &caps,
                                 const DataType &dt,
                                 bool is_shared);

}  // namespace spirv
}  // namespace quadrants::lang
