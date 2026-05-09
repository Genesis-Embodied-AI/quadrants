// Public surface of the adstack runtime helpers. The implementation lives in adstack_runtime.cpp, which is
// `#include`d once from runtime.cpp so the bitcode build stays a single translation unit (matching the existing
// `node_*.h` / `internal_functions.h` / `locked_task.h` pattern in this directory). Both files share the
// `LLVMRuntime` struct memory layout via llvm_runtime.h. A real two-TU build via `llvm-link` would collapse
// structurally identical named struct types (`RootMeta` / `DenseMeta` / ..., `BitmaskedMeta` / `PointerMeta`)
// under a single name, which then breaks the host `QuadrantsLLVMContext::get_runtime_type` lookup by name.
//
// All `runtime_*adstack*` functions are entry points called via the LLVM JIT (`runtime_jit->call` on CPU and CUDA /
// AMDGPU - the `runtime_` prefix is what `init_runtime_module` keys off to mark the function as a `.entry` kernel on
// GPU backends). The internal helper machinery (the SizeExpr device interpreter, etc.) lives in adstack_runtime.cpp
// and is not part of this header.
//
// The `STRUCT_FIELD` getters for the adstack-prefixed `LLVMRuntime` fields live in adstack_runtime.cpp; the zero-init
// block in `materialize_runtime` (runtime.cpp) calls the `adstack_runtime_zero_init` helper declared below.

#pragma once

#include "llvm_runtime.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/static_adstack_bound_reducer_device.h"
#include "quadrants/ir/static_adstack_max_reducer_device.h"

extern "C" {

// Stashes the addresses of the per-launch adstack heap / metadata / lazy-claim / max-reducer fields on the
// `LLVMRuntime` into the result buffer so the host-side executor can cache them once and grow / publish via
// straight `memcpy_host_to_device` writes without per-grow kernel launches. Each getter writes a fixed slot order
// (documented in adstack_runtime.cpp); host callers in `llvm_runtime_executor.cpp` know the slot layout.
void runtime_get_adstack_heap_field_ptrs(LLVMRuntime *runtime);
void runtime_get_adstack_split_heap_field_ptrs(LLVMRuntime *runtime);
void runtime_get_adstack_metadata_field_ptrs(LLVMRuntime *runtime);
void runtime_get_adstack_lazy_claim_field_ptrs(LLVMRuntime *runtime);
void runtime_get_adstack_max_reducer_field_ptr(LLVMRuntime *runtime);

// Per-launch device-resident reducers / interpreters consumed by the host launcher right before each adstack-bearing
// kernel dispatch. `runtime_eval_static_bound_count` walks a captured gating ndarray / SNode field and writes the
// gate-passing count into `runtime->adstack_bound_row_capacities[task_index]`.
// `runtime_eval_adstack_max_reduce` walks a captured `StaticAdStackMaxReducerSpec` body over its multi-axis
// cross-product and reduce-maxes into `runtime->adstack_max_reducer_outputs[output_slot]` via a grid-strided launch
// with an `atomic_max_i64` reduction. CPU is not supported: the recognizer is skipped on CPU (see
// `codegen/llvm/codegen_llvm.cpp::finalize_offloaded_task_function`) so this entry point is only reached on CUDA /
// AMDGPU. `runtime_eval_adstack_size_expr` walks every alloca's SizeExpr tree and publishes per-stack offsets /
// max_sizes plus the per-thread strides into `LLVMRuntime`. The blob layouts are defined in the
// `quadrants/ir/...adstack...device.h` headers.
void runtime_eval_static_bound_count(LLVMRuntime *runtime, RuntimeContext *ctx, Ptr params_blob);
void runtime_eval_adstack_max_reduce(LLVMRuntime *runtime, RuntimeContext *ctx, Ptr params_blob, Ptr body_bytecode);
void runtime_eval_adstack_size_expr(LLVMRuntime *runtime, RuntimeContext *ctx, Ptr bytecode);

// Publish the device-mapped addresses of the pinned-host overflow flag / task-id slots the host allocated at
// `materialize_runtime` time. The codegen-emitted `stack_push` reads through these pointers when overflow fires.
void runtime_set_adstack_overflow_flag_dev_ptr(LLVMRuntime *runtime, void *dev_ptr);
void runtime_set_adstack_overflow_task_id_dev_ptr(LLVMRuntime *runtime, void *dev_ptr);

// Zero-init the adstack metadata fields on `LLVMRuntime`. `LLVMRuntime` is allocated from a raw memory pool rather
// than constructed via `new`, so the C++ default-member-initializers on the adstack fields never run. The host
// launcher writes real values into them via `publish_adstack_metadata` before dispatching any adstack-bearing
// kernel, but we still zero them here so an assert-driven read (or a stale cached kernel that runs before any
// publish) sees well-defined zeros instead of garbage. Called once from `materialize_runtime` in runtime.cpp.
void adstack_runtime_zero_init(LLVMRuntime *runtime);

// Adstack push / pop / top primitives. Called by codegen-emitted reverse-mode IR (not via `runtime_jit->call`), so
// the names lack the `runtime_` prefix that would otherwise mark them as `.entry` kernels on GPU backends. The
// overflow path on `stack_push` writes through the pinned-host slots published by the two setters above.
Ptr stack_top_primal(Ptr stack, std::size_t element_size);
Ptr stack_top_adjoint(Ptr stack, std::size_t element_size);
void stack_init(Ptr stack);
void stack_pop(Ptr stack);
void stack_push(LLVMRuntime *runtime,
                Ptr stack,
                std::size_t max_num_elements,
                std::size_t element_size,
                i64 task_registry_id);

}  // extern "C"
