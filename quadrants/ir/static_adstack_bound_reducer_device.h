// Device-side parameter blob for the LLVM static-adstack bound reducer. The host (LlvmRuntimeExecutor) fills this
// struct on each launch with the captured `StaticAdStackBoundExpr` and an iteration `length`, memcpys it into a small
// device buffer, and calls `runtime_eval_static_bound_count(runtime, ctx, blob_ptr)` as a single-thread serial function
// via the LLVM runtime JIT module. The runtime function (defined in `runtime.cpp`) walks `[0, length)`, evaluates the
// captured comparison + polarity against the gating field's elements (read through `ctx->arg_buffer` at
// `arg_word_offset` for ndarray sources, or through `runtime->roots[snode_root_id]` at
// `snode_byte_base_offset + gid * snode_byte_cell_stride` for SNode-backed sources), counts the matches, and writes
// the count into `runtime->adstack_bound_row_capacities[task_index]`. The codegen-emitted clamp at the float LCA-block
// claim site reads that slot back as the per-task capacity.
//
// `field_source_is_snode` selects between the two source shapes per dispatch; the ndarray and SNode trailing fields
// below are mutually exclusive (only the matching set is read by the reducer).
#pragma once

#include <cstdint>

namespace quadrants::lang {

// Comparison-op encoding shared between the host launcher (encode_cmp_op_for_llvm_reducer) and the device reducer's
// switch statement. Mirrors the SPIR-V reducer's `kAdStackBoundReducerOp*` values so the same `cmp_op` numeric value is
// meaningful across both backends. Values stay 0-5 even if `BinaryOpType`'s int representation drifts.
constexpr uint32_t kLlvmReducerCmpLt = 0;
constexpr uint32_t kLlvmReducerCmpLe = 1;
constexpr uint32_t kLlvmReducerCmpGt = 2;
constexpr uint32_t kLlvmReducerCmpGe = 3;
constexpr uint32_t kLlvmReducerCmpEq = 4;
constexpr uint32_t kLlvmReducerCmpNe = 5;

struct LlvmAdStackBoundReducerDeviceParams {
  // Slot index in `runtime->adstack_bound_row_capacities` that the count is written into. Matches the `task_codegen_id`
  // the codegen burned into the LCA-block claim's bounds-clamp GEP.
  uint32_t task_index;
  // Number of iterations to walk - the iteration bound of the gating predicate (same value the SPIR-V reducer
  // dispatches over). The reducer runs single-threaded on whatever arch it's JIT'd to (CPU is the host evaluator path;
  // CUDA / AMDGPU is a single-thread GPU kernel via `runtime_jit->call`), so no workgroup rounding-up is needed.
  uint32_t length;
  // Encoded comparison op: one of `kLlvmReducerCmp*` above (0-5).
  uint32_t cmp_op;
  // 1 when the gating field's element type is f32 / f64; 0 when i32. The reducer combines this with
  // `field_dtype_is_double` to select element width (4 vs 8 bytes) and load-as-int-vs-float arm.
  uint32_t field_dtype_is_float;
  // 1 when the gating field's element type is f64 (and the source ndarray's stride is 8 bytes per cell). Read only when
  // `field_dtype_is_float == 1`.
  uint32_t field_dtype_is_double;
  // 1 when the gate enters on the predicate holding; 0 when it sits inside the `else` branch and the predicate must be
  // inverted. Mirrors the SPIR-V reducer's `polarity` field.
  uint32_t polarity;
  // Bit-pattern of the captured threshold literal. Reinterpreted as f32 when `field_dtype_is_float == 1` and
  // `field_dtype_is_double == 0`, as i32 when `field_dtype_is_float == 0`. f64 thresholds use the
  // `(threshold_bits_high, threshold_bits)` 64-bit pair below.
  uint32_t threshold_bits;
  // High 32 bits of an f64 threshold, valid only when `field_dtype_is_double == 1`. The reducer reassembles the 64-bit
  // bit pattern from `(threshold_bits_high << 32) | threshold_bits` and bitcasts to `double`.
  uint32_t threshold_bits_high;
  // 0 when the gating field comes from a kernel ndarray argument (resolved via the kernel arg buffer); 1 when it comes
  // from a SNode-backed `qd.field(...)` placed under `qd.root.dense(...)` (resolved via a direct word load from
  // `runtime->roots[snode_root_id]` at byte offset `snode_byte_base_offset + gid * snode_byte_cell_stride`). The two
  // paths are mutually exclusive per dispatch and pick which trailing fields the reducer reads.
  uint32_t field_source_is_snode;
  // ndarray path: u32 word offset into `ctx->arg_buffer` where the ndarray data pointer (u64, two adjacent u32 words)
  // lives. Read only when `field_source_is_snode == 0`.
  uint32_t arg_word_offset;
  // SNode path: index into `runtime->roots[]` selecting the root buffer the gating field lives under. Read only when
  // `field_source_is_snode == 1`.
  uint32_t snode_root_id;
  // SNode path: byte offset of the gating field's first cell within the bound root buffer (precomputed by the IR
  // pattern matcher from the snode descriptor's prefix sums). Read only when `field_source_is_snode == 1`.
  uint32_t snode_byte_base_offset;
  // SNode path: stride per `gid` step in bytes (the dense parent's `cell_stride`). The reducer walks the gating field
  // via `byte_offset = snode_byte_base_offset + gid * snode_byte_cell_stride` and loads one u32 / u64 word from there.
  // Read only when `field_source_is_snode == 1`.
  uint32_t snode_byte_cell_stride;
};

}  // namespace quadrants::lang
