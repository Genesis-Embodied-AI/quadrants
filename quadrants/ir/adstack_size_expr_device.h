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

// Maximum number of distinct `BoundVariable` ids a single adstack's device-side tree may reference. The device
// interpreter in `runtime.cpp` and the SPIR-V sizer shader both keep per-invocation scope arrays sized by this
// constant; the host encoder dense-remaps each tree's `var_id`s into `[0, kAdStackSizeExprDeviceMaxBoundVars)`
// before emitting device nodes, and hard-errors host-side when a tree references more distinct bound vars than
// this cap. Keeping the constant here (rather than in the interpreter's private .cpp) lets both the encoder and
// the interpreter agree on the bound without a cross-TU include.
constexpr int32_t kAdStackSizeExprDeviceMaxBoundVars = 32;

// Node kinds the device interpreter understands. Every kind listed here has a corresponding case in both the LLVM
// device-side interpreter (`quadrants/runtime/llvm/runtime_module/runtime.cpp`) and the SPIR-V sizer shader
// (`quadrants/codegen/spirv/adstack_sizer_shader.cpp`). `ExternalTensorShape` never reaches device code - it is
// always closed, so the host encoder folds it into a `Const` leaf. `FieldLoad` is host-folded on the LLVM path
// (via `SNodeRwAccessorsBank::read_int`) because CPU/CUDA/AMDGPU can safely run a nested accessor kernel, but the
// SPIR-V encoder emits `kFieldLoad` device nodes so the sizer shader reads the snode data in-place via PSB
// instead - nested accessor launches on the MoltenVK queue deadlock in the descriptor-set bind path, so moving
// the read on-device eliminates the host round-trip entirely. Keep numeric ids stable across versions so the
// interpreters can dispatch via a switch without re-remapping.
enum class AdStackSizeExprDeviceKind : int32_t {
  kConst = 0,
  kAdd = 1,
  kSub = 2,
  kMul = 3,
  kMax = 4,
  kMaxOverRange = 5,
  kBoundVariable = 6,
  kExternalTensorRead = 7,
  kFieldLoad = 8,
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
  // Heap kind for the SPIR-V dual-heap layout, matching `TaskAttributes::AdStackAllocaAttribs::HeapKind`:
  // `0` = float heap (f32), `1` = int heap (i32 / u1). Ignored by the LLVM runtime which has a single
  // unified heap. The SPIR-V sizer uses this bit to route each stack's `max_size` contribution to the
  // matching `stride_float` / `stride_int` running sum that drives `BufferType::AdStackHeapFloat` /
  // `AdStackHeapInt` allocation on the host. The encoder populates it from
  // `AdStackAllocaAttribs::HeapKind` on the SPIR-V path; LLVM encoder leaves it at `0`.
  uint32_t heap_kind{0};
};

struct AdStackSizeExprDeviceNode {
  int32_t kind{0};            // cast from `AdStackSizeExprDeviceKind`
  int32_t operand_a{-1};      // node index, or -1 if unused
  int32_t operand_b{-1};      // node index, or -1 if unused
  int32_t body_node_idx{-1};  // `MaxOverRange` only
  int32_t var_id{-1};         // `BoundVariable` / `MaxOverRange`
  int32_t prim_dt{-1};        // `kExternalTensorRead` / `kFieldLoad`: `PrimitiveTypeID` of the loaded element
  // `kExternalTensorRead`: byte offset into `RuntimeContext::arg_buffer` where the data pointer for the referenced
  // ndarray lives. The interpreter does `void *data_ptr = *(void **)(arg_buffer + arg_buffer_offset);` and then
  // derefs. Precomputed host-side from the kernel's `args_type` struct via
  // `StructType::get_element_offset({arg_id, DATA_PTR_POS_IN_NDARRAY})`. Unused (-1) for other kinds.
  int32_t arg_buffer_offset{-1};
  // Offset into the global indices array + `indices_count` = number of axes. Layout inside the global indices
  // array depends on the node kind:
  //   * `kExternalTensorRead`: `indices_count` consecutive int32 index values, stride-1 summed (legacy, fine for
  //     1-D ndarrays). Indices use the same encoding as `SerializedSizeExprNode::indices` - non-negative =
  //     constant index, `-(var_id + 1)` = currently-bound loop variable.
  //   * `kFieldLoad`: `2 * indices_count` int32 entries as pairs `(idx_a_raw, elem_stride_a)` per axis. The
  //     element stride is positive and pre-computed on host from the snode's dense shape chain (in *elements*
  //     of the leaf's primitive type, matching how `psb_load_scalar` in the sizer shader multiplies the final
  //     element index by `sizeof(prim_dt)`). `elem_stride_a` sits in the slot immediately after its matching
  //     `idx_raw` so a single sequential scan of `indices[off .. off + 2N)` computes the element index.
  int32_t indices_offset{0};
  int32_t indices_count{0};
  int32_t _pad0{0};
  // `kConst`: the literal value. `kFieldLoad`: pre-computed PSB base address = `snode_root_psb +
  // place_byte_offset_in_root`, so the interpreter only has to add the per-index byte offset and deref. Unused
  // for other kinds.
  int64_t const_value{0};
};

}  // namespace quadrants::lang
