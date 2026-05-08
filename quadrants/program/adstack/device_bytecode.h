#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/codegen/spirv/kernel_utils.h"
#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/program/adstack/max_reducer.h"

namespace quadrants::lang {

class LaunchContextBuilder;
class Program;
class SNode;

// Data needed to encode a `FieldLoad` as a `kFieldLoad` device node on the SPIR-V backend. Populated by the SPIR-V
// dispatch site (per-task sizer or max-reducer) via `GfxRuntime` / `Device` queries; the LLVM encoder paths pass a
// default-constructed (`empty()`) emitter and resolve `(snode_root_id, place_byte_offset)` directly via `prog`. The
// fetch closure returns `out_base_psb = root_buffer_psb + place_byte_offset_in_root` and per-active-axis element
// strides (in units of the leaf primitive type, not bytes - the shader multiplies by `sizeof(prim_dt)` separately via
// `psb_load_scalar`). Returns false when the snode is not amenable to direct PSB indexing (bitmasked / pointer / hash
// chain, bit-level place, not-all-dense path); the encoder treats that as a hard error on the per-task sizer path or
// drops the spec on the max-reducer path.
struct FieldLoadDeviceEmitter {
  std::function<bool(SNode *snode, uint64_t *out_base_psb, std::vector<int32_t> *out_elem_strides)> fetch;

  bool empty() const {
    return fetch == nullptr;
  }
};

// Compute per-active-axis element strides for a dense `place`-leaf SNode (units = leaf primitive type, not bytes).
// Matches the SPIR-V FieldLoad emitter's stride convention; the max-reducer encoder reuses this to lay out the
// `[idx_a_raw, elem_stride_a]` indices-table pairs that the body interpreter walks. Returns false on non-dense /
// bit-level / multi-child-dense layouts (same restriction as the per-task sizer's `FieldLoadDeviceEmitter::fetch`).
bool compute_dense_snode_strides(SNode *leaf, std::vector<int32_t> *out_elem_strides);

// Flattens every alloca's `SerializedSizeExpr` tree into the device-readable bytecode defined in
// `quadrants/ir/adstack_size_expr_device.h` and returns the raw bytes ready to upload to a device scratch buffer.
// Two transforms happen at encoding time:
//
//   1. Pre-substitution of host-resolvable subtrees. Any subtree whose leaves consist only of `Const`,
//      `BoundVariable`, `FieldLoad`, and `ExternalTensorShape` nodes - i.e. nothing that requires an
//      on-device pointer dereference - is collapsed to a single `Const` node by running the existing host
//      evaluator over it. This routes `FieldLoad` through `SNodeRwAccessorsBank::read_int` (which itself
//      handles device-to-host via a tiny reader kernel on GPU) and `ExternalTensorShape` through the kernel
//      arg buffer that the host just wrote, so the device interpreter in `runtime.cpp` never has to walk
//      an SNode tree or index into `args_type` - it only has to handle arithmetic plus
//      `ExternalTensorRead`, which is the one leaf kind that actually needs device-resident memory.
//   2. `arg_buffer_offset` precomputation. Every surviving `ExternalTensorRead` carries the byte offset into
//      `RuntimeContext::arg_buffer` where the referenced ndarray's data pointer lives, resolved here against
//      `ctx->args_type->get_element_offset({arg_id, DATA_PTR_POS_IN_NDARRAY})`. The device interpreter does
//      a direct `*(void **)(arg_buffer + offset)` to fetch the ndarray pointer at launch time - no map
//      lookup, no `LaunchContextBuilder` touches from device code.

// Mixed subtrees that contain both an `ExternalTensorRead` and a `FieldLoad` are rejected with a hard error:
// the device interpreter does not support on-device SNode access, so a `FieldLoad` that cannot be lifted out
// to a host-resolvable `Const` has nowhere to run. The grammar today does not emit this combination and no
// user kernel has been observed to do so; the hard error pins the assumption so a future regression cannot
// slip past.
std::vector<uint8_t> encode_adstack_size_expr_device_bytecode(
    const AdStackSizingInfo &ad_stack,
    Program *prog,
    LaunchContextBuilder *ctx,
    const MaxReducerResultMap &max_reducer_results = MaxReducerResultMap{});

// SPIR-V-flavour encoder. Same transforms as the LLVM variant, but sources per-stack metadata from
// `TaskAttributes::AdStackSizingAttribs::allocas` (each entry has a `HeapKind` - `Float = 0`, `Int = 1` -
// that routes the stack onto the `AdStackHeapFloat` or `AdStackHeapInt` backing buffer on the host). The
// `heap_kind` field of each `AdStackSizeExprDeviceStackHeader` carries that selector into the shader; the
// shader splits the running-offset / stride computation into a float accumulator and an int accumulator so
// the output metadata buffer matches the layout the main kernel already reads today:
// `[stride_float, stride_int, (offset_i, max_size_i)*]`. The `entry_size_bytes` field is set to 1 on the
// SPIR-V path because the backing buffers are element-indexed (f32 / i32) rather than byte-indexed and the
// shader multiplies by `2` only for the `Float` heap (primal + adjoint interleaved) - see the running-offset
// arithmetic in `GfxRuntime::launch_kernel` for the convention this matches.
std::vector<uint8_t> encode_adstack_size_expr_device_bytecode_for_spirv(
    const spirv::TaskAttributes::AdStackSizingAttribs &ad_stack,
    Program *prog,
    LaunchContextBuilder *ctx,
    const MaxReducerResultMap &max_reducer_results = MaxReducerResultMap{});

}  // namespace quadrants::lang
