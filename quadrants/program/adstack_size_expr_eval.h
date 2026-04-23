#pragma once

#include <cstdint>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/codegen/spirv/kernel_utils.h"
#include "quadrants/ir/adstack_size_expr.h"

namespace quadrants::lang {

class Program;
class LaunchContextBuilder;

// Evaluates a compile-time captured `SerializedSizeExpr` against the current field state of `prog` and the
// per-launch argument values in `ctx`, returning the concrete adstack capacity for this launch. Scalar i32/i64
// field loads are serviced by `SNodeRwAccessorsBank` (one reader-kernel dispatch each); ndarray-argument shapes
// are read from `ctx->get_struct_arg<int64>`; constants and arithmetic are folded in plain C++; `MaxOverRange`
// enumerates its range and takes the max of the body expression across the bound variable. Returns -1 when the
// expression is empty (no symbolic bound captured), signalling to the caller to use the compile-time fallback.
int64_t evaluate_adstack_size_expr(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx);

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
//
// Mixed subtrees that contain both an `ExternalTensorRead` and a `FieldLoad` are rejected with a hard error:
// the device interpreter does not support on-device SNode access, so a `FieldLoad` that cannot be lifted out
// to a host-resolvable `Const` has nowhere to run. The grammar today does not emit this combination and no
// user kernel has been observed to do so; the hard error pins the assumption so a future regression cannot
// slip past.
std::vector<uint8_t> encode_adstack_size_expr_device_bytecode(const AdStackSizingInfo &ad_stack,
                                                              Program *prog,
                                                              LaunchContextBuilder *ctx);

// SPIR-V-flavour encoder. Same transforms as the LLVM variant, but sources per-stack metadata from
// `TaskAttributes::AdStackSizingAttribs::allocas` (each entry has a `HeapKind` — `Float = 0`, `Int = 1` —
// that routes the stack onto the `AdStackHeapFloat` or `AdStackHeapInt` backing buffer on the host). The
// `heap_kind` field of each `AdStackSizeExprDeviceStackHeader` carries that selector into the shader; the
// shader splits the running-offset / stride computation into a float accumulator and an int accumulator so
// the output metadata buffer matches the layout the main kernel already reads today:
// `[stride_float, stride_int, (offset_i, max_size_i)*]`. The `entry_size_bytes` field is set to 1 on the
// SPIR-V path because the backing buffers are element-indexed (f32 / i32) rather than byte-indexed and the
// shader multiplies by `2` only for the `Float` heap (primal + adjoint interleaved) — see the running-offset
// arithmetic in `GfxRuntime::launch_kernel` for the convention this matches.
std::vector<uint8_t> encode_adstack_size_expr_device_bytecode_for_spirv(
    const spirv::TaskAttributes::AdStackSizingAttribs &ad_stack,
    Program *prog,
    LaunchContextBuilder *ctx);

}  // namespace quadrants::lang
