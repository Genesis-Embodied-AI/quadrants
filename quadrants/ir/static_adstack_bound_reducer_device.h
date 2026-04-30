// Device-side parameter blob for the LLVM static-adstack bound reducer. The host (LlvmRuntimeExecutor)
// fills this struct on each launch with the captured `StaticAdStackBoundExpr` and an iteration `length`,
// memcpys it into a small device buffer, and calls `runtime_eval_static_bound_count(runtime, ctx, blob_ptr)`
// as a single-thread serial function via the LLVM runtime JIT module. The runtime function (defined in
// `runtime.cpp`) walks `[0, length)`, evaluates the captured comparison + polarity against the gating
// ndarray's elements (read through `ctx->arg_buffer` at `arg_word_offset`), counts the matches, and writes
// the count into `runtime->adstack_bound_row_capacities[task_index]`. The codegen-emitted clamp at the
// float LCA-block claim site reads that slot back as the per-task capacity.
//
// SNode-backed gates are not captured on the LLVM analysis path today; this struct only encodes the
// ndarray-backed shape. SNode capture (and a matching device read path) is a future extension.
#pragma once

#include <cstdint>

namespace quadrants::lang {

// Comparison-op encoding shared between the host launcher (encode_cmp_op_for_llvm_reducer) and the device
// reducer's switch statement. Mirrors the SPIR-V reducer's `kAdStackBoundReducerOp*` values so the same
// `cmp_op` numeric value is meaningful across both backends. Values stay 0-5 even if `BinaryOpType`'s int
// representation drifts.
constexpr uint32_t kLlvmReducerCmpLt = 0;
constexpr uint32_t kLlvmReducerCmpLe = 1;
constexpr uint32_t kLlvmReducerCmpGt = 2;
constexpr uint32_t kLlvmReducerCmpGe = 3;
constexpr uint32_t kLlvmReducerCmpEq = 4;
constexpr uint32_t kLlvmReducerCmpNe = 5;

struct LlvmAdStackBoundReducerDeviceParams {
  // Slot index in `runtime->adstack_bound_row_capacities` that the count is written into. Matches the
  // `task_codegen_id` the codegen burned into the LCA-block claim's bounds-clamp GEP.
  uint32_t task_index;
  // Number of iterations to walk - the iteration bound of the gating predicate (same value the SPIR-V
  // reducer dispatches over). The reducer runs single-threaded on whatever arch it's JIT'd to (CPU is the
  // host evaluator path; CUDA / AMDGPU is a single-thread GPU kernel via `runtime_jit->call`), so no
  // workgroup rounding-up is needed.
  uint32_t length;
  // Encoded comparison op: one of `kLlvmReducerCmp*` above (0-5).
  uint32_t cmp_op;
  // 1 when the gating field's element type is f32; 0 when i32. The reducer uses this to pick the right
  // load width (4 bytes either way, but the comparison semantics differ between signed-int and float).
  uint32_t field_dtype_is_float;
  // 1 when the gate enters on the predicate holding; 0 when it sits inside the `else` branch and the
  // predicate must be inverted. Mirrors the SPIR-V reducer's `polarity` field.
  uint32_t polarity;
  // Bit-pattern of the captured threshold literal. Reinterpreted as f32 when `field_dtype_is_float`, as
  // i32 otherwise.
  uint32_t threshold_bits;
  // u32 word offset into `ctx->arg_buffer` where the ndarray data pointer (u64, two adjacent u32 words)
  // lives. The reducer reads `arg_buffer[arg_word_offset]` + `arg_buffer[arg_word_offset+1]` to
  // reconstruct the device pointer, then strides through the field by element index.
  uint32_t arg_word_offset;
  // Padding to keep the struct 8-byte aligned for h2d memcpy alignment.
  uint32_t padding;
};

}  // namespace quadrants::lang
