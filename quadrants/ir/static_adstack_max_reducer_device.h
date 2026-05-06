// Device-side parameter blob for the LLVM static-adstack max reducer. The host
// (`LlvmRuntimeExecutor`) fills this struct on each launch with one captured `StaticAdStackMaxReducerSpec`'s
// dispatch parameters, memcpys it (plus the body bytecode trailing blob) into a small device buffer, and calls
// `runtime_eval_adstack_max_reduce(runtime, ctx, params_blob, body_bytecode)` as a single-thread serial function
// via the LLVM runtime JIT module - mirrors how `runtime_eval_static_bound_count` is invoked for the bound reducer.
//
// The body bytecode is a separate pointer because it varies in size per spec (nodes + indices arrays) while the
// params blob has a fixed POD layout. The runtime function walks `[0, length)`, evaluates the body subtree against
// the bound variable `iter_var = begin + i`, tracks the per-launch running max, and writes the result into
// `runtime->adstack_max_reducer_outputs[output_slot]`. The caller substitutes the dispatched value as a `Const` into
// the per-stack `SizeExpr` tree before any of the LLVM eval paths walks it.
//
// Shared with the SPIR-V variant (`AdStackMaxReducerParams` in `quadrants/codegen/spirv/adstack_max_reducer_shader.h`)
// at the field-semantics level: `output_slot`, `length`, `begin` carry the same meaning. The LLVM variant differs in
// `body_root_node_idx` / `var_id` (vs. the SPIR-V `body_bytecode_offset_words` / `body_node_count` which address into
// a shared bytecode buffer at descriptor-aligned offsets); the LLVM caller passes one body bytecode blob per call.
#pragma once

#include <cstdint>

namespace quadrants::lang {

struct LlvmAdStackMaxReducerDeviceParams {
  // Slot index in `runtime->adstack_max_reducer_outputs` that this dispatch's running max is written into. Allocated
  // by the host launcher per `StaticAdStackMaxReducerSpec` from the same `MaxReducerCacheKey -> output_slot` table the
  // SPIR-V launcher uses, so the same numeric slot is consistent across backends.
  uint32_t output_slot;
  // Number of iterations to walk (`end - begin` of the captured `MaxOverRange`, host-evaluated against the live ctx
  // before the dispatch). Single-thread serial walk: no workgroup rounding-up needed.
  uint32_t length;
  // Base of the iteration variable, host-evaluated against the live ctx. Per-iteration the body sees
  // `iter_var = begin + i` (i64 to cover the worst-case big ndarray-axis case where `length` itself can exceed 1<<24
  // and `begin + i` may wrap an i32). The body bytecode references this variable through any node whose
  // `idx_raw == -(var_id + 1)`.
  int64_t begin;
  // Number of `AdStackSizeExprDeviceNode`s in the body bytecode trailing blob. Bytecode layout:
  // `[AdStackSizeExprDeviceNode x body_node_count][int32 x indices_count]`. `indices_count` is implicit in the
  // node-side `indices_offset` / `indices_count` fields - the bytecode buffer simply contains the contiguous indices
  // table after the nodes.
  uint32_t body_node_count;
  // Index of the body subtree's root within the body bytecode (post-order encoding places the root last, so
  // `body_root_node_idx == body_node_count - 1`). Cached here so the runtime function does not need to subtract.
  int32_t body_root_node_idx;
  // Bound-variable id this spec's `MaxOverRange` introduced. The runtime function pre-populates
  // `scope.values[var_id] = begin + i` before each iteration; any body leaf encoded as `-(var_id + 1)` resolves to that
  // value via the existing `device_eval_node` path. The recognizer guarantees a single bound variable per spec, so a
  // single scope slot suffices.
  int32_t var_id;
  // Padding to keep the struct aligned to 8 bytes. Future stages may use this for additional flags (e.g. body dtype
  // when the encoder ever needs to widen beyond i64-as-internal).
  uint32_t _pad0;
};

}  // namespace quadrants::lang
