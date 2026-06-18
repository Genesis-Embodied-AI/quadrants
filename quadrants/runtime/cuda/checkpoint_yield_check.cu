// Yield-check kernel for `qd.checkpoint(yield_on=...)` blocks.
//
// Inserted by the GraphManager at the end of each `with qd.checkpoint(yield_on=foo):` body. On
// SM 9.0+ the node lives inside the same IF conditional subgraph as the body kernels, so the
// conditional gate already prevents it from running when the checkpoint is skipped. On
// pre-Hopper (CUDA without conditional graph nodes) the node lives in the top-level flat
// graph and runs unconditionally; the explicit self-gate at the top of this kernel matches
// the SM 9.0+ semantics by reading `*resume_point` / `*yield_signal` and early-returning when
// this checkpoint should be skipped, identical to the codegen-emitted body-kernel prologue.
//
// Semantics on launch (when not skipped):
//   1. Read `*yield_on` (the user-side ndarray pointed to via the indirection slot).
//   2. If non-zero, the checkpoint body asked the host to handle something this iteration:
//        a. atomicCAS `yield_signal` from -1 to this cp_id -- the first checkpoint in
//           declaration order to yield wins (slice 1d "atomic_cas race" requirement).
//        b. Bump `*resume_point` to INT_MAX so every subsequent checkpoint's gate kernel sees
//           `cp_id >= INT_MAX == false` and skips its body.
//   3. If zero, no-op.
//
// The framework does NOT reset `*yield_on` here. The user owns that buffer (it's their ndarray)
// and is responsible for clearing it before the resume launch -- otherwise the resumed body's
// yield-check will see the same non-zero value and yield again on the same checkpoint, looping
// the host loop forever. The canonical host loop in `docs/source/user_guide/graph.md` shows the
// explicit clear-before-resume.
//
// The indirection on `yield_on` matches `graph_do_while`'s `flag_slot`: the slot's device
// address is baked into the graph at build time, but the pointer it contains is host-updated
// each launch via memcpy, so a user can pass a different ndarray each call.
//
// After editing, regenerate the pre-built fatbin:
//
//   python scripts/build_checkpoint_yield_check_fatbin.py

#include <climits>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" __global__ void _qd_checkpoint_yield_check(int32_t **yield_on_ptr_slot,
                                                      int32_t cp_id,
                                                      int32_t *yield_signal,
                                                      int32_t *resume_point) {
  // Self-gate: identical predicate to the codegen-emitted body-kernel prologue. On SM 9.0+
  // the surrounding IF conditional makes this dead code in the common path; on pre-Hopper
  // (flat-graph path) it skips the yield-read when this checkpoint should not have run.
  if (*resume_point > cp_id) {
    return;
  }
  if (*yield_signal != -1) {
    return;
  }
  int32_t *yield_on = *yield_on_ptr_slot;
  if (*yield_on != 0) {
    atomicCAS(yield_signal, -1, cp_id);
    *resume_point = INT_MAX;
  }
}
