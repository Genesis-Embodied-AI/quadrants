// Stage B of the LLVM sparse-adstack-heap lazy-claim pipeline: per-launch metadata publish. Encodes the
// per-task SizeExpr trees into device bytecode (or runs the host evaluator when every leaf is
// host-resolvable), launches the sizer runtime function on CUDA / AMDGPU when needed, packs and publishes
// `[stride_combined, stride_float, stride_int, offsets[n_stacks], max_sizes[n_stacks]]` through pinned-host
// async H2Ds on GPU or synchronous `memcpy` on CPU. The per-task metadata cache fast path republishes a hit's
// payload through the same pinned-async pipeline. Closes by invoking `ensure_adstack_heap_int` /
// `ensure_adstack_heap_float` (Stage C) for the stripes the codegen will read at row-claim time.
//
// All entry points are member methods of `LlvmRuntimeExecutor` and stay declared in
// `quadrants/runtime/llvm/llvm_runtime_executor.h`. This header is the place to add file-local helpers that
// future cross-stage callers might need; the stage's lambdas (`align_up_8`, `grow_to`, `copy_h2d`,
// `copy_d2h`, `publish_metadata_pinned_async`) all stay scoped inside `publish_adstack_metadata`.

#pragma once

namespace quadrants::lang {

// Reserved for future cross-stage helper declarations.

}  // namespace quadrants::lang
