// Gate kernel for qd.checkpoint() IF conditional nodes (CUDA 12.4+ native path).
//
// Each `qd.checkpoint(...)` block inserts:
//   1. one of these gate kernel nodes immediately before
//   2. a CUDA graph IF conditional node wrapping the checkpoint body
//
// The gate kernel reads the framework-internal `resume_point` scalar (a single int32
// device allocation owned by the CachedGraph) and writes the conditional handle to
// enable or disable the IF body. The handle's per-call default is `0`, so an unset
// handle skips the body; we explicitly set it here every launch.
//
// Semantics:
//   handle := (cp_id >= *resume_point) ? 1 : 0
//
// Slice 1c only ever launches with `*resume_point == 0`, so every checkpoint runs every
// launch (no behaviour change vs. a non-checkpointed kernel). Slice 1d's host-side
// `step.resume(from_checkpoint=cp_id)` will write `cp_id` into the resume_point scalar
// before relaunching the same cached graph, causing checkpoints with cp_id < resume_point
// to skip while the resuming checkpoint and all subsequent ones run.
//
// After editing, regenerate the pre-built fatbin:
//   python scripts/build_checkpoint_gate_fatbin.py

#include <cstdint>
#include <cuda_runtime.h>

extern "C" __global__ void _qd_checkpoint_if_gate(cudaGraphConditionalHandle handle,
                                                  int32_t cp_id,
                                                  int32_t *resume_point) {
  unsigned int enabled = (cp_id >= *resume_point) ? 1u : 0u;
  cudaGraphSetConditional(handle, enabled);
}
