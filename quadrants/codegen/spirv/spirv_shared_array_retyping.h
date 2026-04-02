#pragma once

#include <unordered_map>
#include <unordered_set>

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

// If alloca needs uint-backed retyping (based on device caps and atomic ops),
// return the retyped SType and insert into retyped_stmts. Otherwise return
// the original SType for scalar_dtype.
SType maybe_retype_shared_float_alloca(
    IRBuilder &ir,
    const DeviceCapabilityConfig &caps,
    const AllocaStmt *alloca,
    const DataType &scalar_dtype,
    const std::unordered_map<const Stmt *, bool> &alloc_map,
    std::unordered_set<const Stmt *> &retyped_stmts);

// If origin is in retyped_stmts, propagate retyping to stmt and return the
// uint-backed element SType. Otherwise return the default SType for dt.
SType maybe_retype_derived_ptr(IRBuilder &ir,
                               const Stmt *origin,
                               const Stmt *stmt,
                               const DataType &dt,
                               std::unordered_set<const Stmt *> &retyped_stmts);

// Load from a shared float pointer, bitcasting from uint if retyped.
Value load_shared_float(IRBuilder &ir,
                        Value ptr_val,
                        const Stmt *ptr,
                        const DataType &element_type,
                        const std::unordered_set<const Stmt *> &retyped_stmts);

// Store to a shared float pointer, bitcasting to uint if retyped.
Value store_shared_float(IRBuilder &ir,
                         Value val,
                         const Stmt *dest,
                         const DataType &val_type,
                         const std::unordered_set<const Stmt *> &retyped_stmts);

// CAS-based float atomic for shared (workgroup) arrays. Unlike
// IRBuilder::float_atomic, this handles width-mismatched uint backing
// (e.g. u32 backing for f16 arrays, since Metal/Vulkan lack 16-bit atomics).
Value shared_float_atomic(IRBuilder &ir,
                          AtomicOpType op_type,
                          Value addr_ptr,
                          Value data,
                          const DataType &dt);

// Check whether the device has native shared float atomic add for dt.
// For device buffers (dest_is_ptr=false), checks buffer capabilities instead.
bool has_native_float_atomic_add(const DeviceCapabilityConfig &caps,
                                 const DataType &dt,
                                 bool is_shared);

}  // namespace spirv
}  // namespace quadrants::lang
