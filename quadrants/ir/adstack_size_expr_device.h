#pragma once

#include <cstdint>

// POD node layout and bytecode header for the host-to-device serialisation of a `SerializedSizeExpr` tree. The
// struct is shared between `quadrants/ir/adstack_size_expr.cpp` (host encoder) and
// `quadrants/runtime/llvm/runtime_module/runtime.cpp` (device interpreter), so every field is plain-old-data with a
// fixed layout - no `std::vector`, no signed-unsigned mixing inside a single 8-byte slot, natural alignment all the
// way through. The interpreter in runtime.cpp is compiled to LLVM bitcode and linked into each kernel module, so
// this header MUST be includable from both the host C++ build and the runtime bitcode build (which uses a
// restricted subset of headers).
//
// The bytecode is produced per-task on the host, shortly before the sizer runtime function runs, and uploaded to a
// device-resident scratch buffer owned by the `LlvmRuntimeExecutor`. It is regenerated on every launch because the
// `arg_buffer_offset` values (precomputed from the launcher context's `args_type`) and the `Const`-substituted
// `FieldLoad` / `ExternalTensorShape` values change per dispatch; caching across launches would re-introduce the
// stale-read bug we explicitly solved by moving the resolution on-device.
//
// Layout in the bytecode buffer:
//   [AdStackSizeExprDeviceHeader]
//   [AdStackSizeExprDeviceStackHeader x header.n_stacks]
//   [AdStackSizeExprDeviceNode       x header.total_nodes]
//   [int32                           x header.total_indices]
// All offsets are in units of the corresponding element (not bytes) so the interpreter can index directly.

namespace quadrants::lang {

// Only the node kinds that survive the host pre-substitution. `FieldLoad` is replaced with `Const(field_value)` and
// `ExternalTensorShape` with `Const(shape_value)` before encoding, so the device interpreter never sees them. Keep
// the numeric ids stable across versions so the interpreter can dispatch via a switch without re-remapping.
enum class AdStackSizeExprDeviceKind : int32_t {
  kConst = 0,
  kAdd = 1,
  kSub = 2,
  kMul = 3,
  kMax = 4,
  kMaxOverRange = 5,
  kBoundVariable = 6,
  kExternalTensorRead = 7,
};

struct AdStackSizeExprDeviceHeader {
  uint32_t n_stacks{0};
  uint32_t total_nodes{0};
  uint32_t total_indices{0};
  uint32_t _pad{0};
};

// One per alloca in the task. `root_node_idx` points into the global node array; `-1` signals "no SizeExpr
// captured" (empty tree) and routes this slot to `max_size_compile_time` without running the interpreter.
struct AdStackSizeExprDeviceStackHeader {
  int32_t root_node_idx{-1};
  // Entry size and compile-time fallback, inlined here so the interpreter can compute this stack's offset /
  // stride contribution without needing a second parallel array. `entry_size_bytes` matches
  // `AdStackAllocaInfo::entry_size_bytes` (= 2 * element size); `max_size_compile_time` seeds the fallback when
  // `root_node_idx < 0`.
  uint32_t entry_size_bytes{0};
  uint32_t max_size_compile_time{0};
  uint32_t _pad{0};
};

struct AdStackSizeExprDeviceNode {
  int32_t kind{0};            // cast from `AdStackSizeExprDeviceKind`
  int32_t operand_a{-1};      // node index, or -1 if unused
  int32_t operand_b{-1};      // node index, or -1 if unused
  int32_t body_node_idx{-1};  // `MaxOverRange` only
  int32_t var_id{-1};         // `BoundVariable` / `MaxOverRange`
  int32_t prim_dt{-1};        // `ExternalTensorRead`: `PrimitiveTypeID` of the element (i32/i64/u32/u64/i16/u16/i8/u8)
  // Byte offset into `RuntimeContext::arg_buffer` where the data pointer for the referenced ndarray lives.
  // The interpreter does `void *data_ptr = *(void **)(arg_buffer + arg_buffer_offset);` and then derefs.
  // Precomputed host-side from the kernel's `args_type` struct via
  // `StructType::get_element_offset({arg_id, DATA_PTR_POS_IN_NDARRAY})`.
  int32_t arg_buffer_offset{-1};
  // Offset into the global indices array + count. Indices follow the same `SerializedSizeExprNode::indices`
  // encoding: non-negative = constant index; `-(var_id + 1)` = currently-bound loop variable captured by the
  // nearest enclosing `MaxOverRange`.
  int32_t indices_offset{0};
  int32_t indices_count{0};
  int32_t _pad0{0};
  int64_t const_value{0};  // `kConst` literal
};

}  // namespace quadrants::lang
