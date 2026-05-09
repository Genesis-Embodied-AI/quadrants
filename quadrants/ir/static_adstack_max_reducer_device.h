// Device-side parameter blob for the LLVM static-adstack max reducer. The host (`LlvmRuntimeExecutor`) fills this
// struct on each launch with one captured `StaticAdStackMaxReducerSpec`'s dispatch parameters, memcpys it (plus the
// body bytecode trailing blob) into a small device buffer, and calls `runtime_eval_adstack_max_reduce(runtime, ctx,
// params_blob, body_bytecode)` as a single-thread serial function via the LLVM runtime JIT module - mirrors how
// `runtime_eval_static_bound_count` is invoked for the bound reducer.
//
// The body bytecode is a separate pointer because it varies in size per spec (nodes + indices arrays) while the params
// blob has a fixed POD layout. The runtime function walks the cross-product of every axis range, evaluates the body
// subtree against each axis's pre-populated bound variable, tracks the per-launch running max, and writes the result
// into `runtime->adstack_max_reducer_outputs[output_slot]`. The caller substitutes the dispatched value as a `Const`
// into the per-stack `SizeExpr` tree before any of the LLVM eval paths walks it.
//
// Shared with the SPIR-V variant (`AdStackMaxReducerParams` in `quadrants/codegen/spirv/adstack_max_reducer_shader.h`)
// at the field-semantics level: `output_slot`, per-axis lengths / begins carry the same meaning. The LLVM variant
// passes one body bytecode blob per call (vs. the SPIR-V `body_bytecode_offset_words` / `body_node_count` which address
// into a shared bytecode buffer at descriptor-aligned offsets).
#pragma once

#include <cstdint>

namespace quadrants::lang {

// Maximum number of nested `MaxOverRange` axes the recognizer may absorb into a single max-reducer dispatch. The
// recognizer's greedy chain capture (in `recognize_adstack_max_reducer_specs`) walks down nested `MaxOverRange` bodies
// and accumulates one axis per layer; specs whose chain exceeds this cap fall back to the per-thread sizer. Bumping the
// constant raises the per-spec params blob size by 16 bytes/axis on LLVM and 4 words/axis on SPIR-V. Practical
// workloads (Genesis rigid-body kernels) capture 1-3 axes; keep the cap modest so the SPIR-V params blob stays well
// below the descriptor-set min push-constant budget.
constexpr int32_t kAdStackMaxReducerMaxAxes = 8;

struct LlvmAdStackMaxReducerDeviceParams {
  // Slot index in `runtime->adstack_max_reducer_outputs` that this dispatch's running max is written into. Allocated by
  // the host launcher per `StaticAdStackMaxReducerSpec` from the same `MaxReducerCacheKey -> output_slot` table the
  // SPIR-V launcher uses, so the same numeric slot is consistent across backends.
  uint32_t output_slot;
  // Number of captured chain axes (1..kAdStackMaxReducerMaxAxes). Axis 0 is the outermost `MaxOverRange`, axis
  // `num_axes - 1` is the innermost.
  uint32_t num_axes;
  // Number of `AdStackSizeExprDeviceNode`s in the body bytecode trailing blob. Bytecode layout:
  // `[AdStackSizeExprDeviceNode x body_node_count][int32 x indices_count]`. `indices_count` is implicit in the
  // node-side `indices_offset` / `indices_count` fields - the bytecode buffer simply contains the contiguous indices
  // table after the nodes.
  uint32_t body_node_count;
  // Index of the body subtree's root within the body bytecode (post-order encoding places the root last, so
  // `body_root_node_idx == body_node_count - 1`). Cached here so the runtime function does not need to subtract.
  int32_t body_root_node_idx;
  // Per-axis iteration length (`end - begin`), ordered outermost-first. Axes beyond `num_axes` are zero-padded.
  uint32_t per_axis_length[kAdStackMaxReducerMaxAxes];
  // Per-axis iteration base (`begin`), ordered outermost-first. The runtime pre-populates
  // `scope.values[per_axis_var_id[k]] = per_axis_begin[k] + i_k` for each axis before walking the body.
  int64_t per_axis_begin[kAdStackMaxReducerMaxAxes];
  // Per-axis device-scope slot id for the bound variable. Encoded by the host as a dense remap of the captured chain
  // bound-var ids into `[0, num_axes)`; the body bytecode encodes references as `-(slot + 1)` (matching the existing
  // device-side `device_eval_node` convention).
  int32_t per_axis_var_id[kAdStackMaxReducerMaxAxes];
};

}  // namespace quadrants::lang
