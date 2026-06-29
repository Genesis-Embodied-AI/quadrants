// Stage C of the LLVM sparse-adstack-heap lazy-claim pipeline: heap-allocation lifecycle. Sizes the float /
// int adstack heaps from the per-task counts Stage A published and the per-kind strides Stage B resolved, so
// each heap holds exactly `count * stride` bytes per dispatch instead of the dispatched-threads worst case.
// `ensure_adstack_heap_float` and `ensure_adstack_heap_int` are the amortised-doubling growers; on first
// grow they cache the four split-heap field-of-LLVMRuntime addresses through
// `runtime_get_adstack_split_heap_field_ptrs`. `ensure_per_task_float_heap_post_reducer` reads the
// reducer-published per-task count and is the bridge from Stage A's gate output to the lazy float-heap
// sizing. `check_adstack_overflow` is the sync-time consumer that raises when a kernel pushed past the
// captured capacity; it polls the pinned-host overflow flag with a relaxed atomic exchange, no DtoH and no
// JIT call.
//
// All entry points are member methods of `LlvmRuntimeExecutor` and stay declared in
// `quadrants/runtime/llvm/llvm_runtime_executor.h`. This header is reserved for future cross-stage helpers;
// the four methods presently share no helpers other than the runtime field-pointer cache they all consult.

#pragma once

namespace quadrants::lang {

// Reserved for future cross-stage helper declarations.

}  // namespace quadrants::lang
