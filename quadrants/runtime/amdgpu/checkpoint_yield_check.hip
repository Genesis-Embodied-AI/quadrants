// Yield-check kernel for `qd.checkpoint(yield_on=...)` blocks on AMDGPU.
//
// HIP 7.2 has no conditional graph nodes and no indirect-dispatch primitive, so the AMDGPU
// `qd.checkpoint` lowering mirrors the pre-Hopper CUDA design (see
// `runtime/cuda/checkpoint_yield_check.cu`):
//
//   * Every body kernel launches as a regular HIP graph node.
//   * Each cp_id >= 0 body kernel carries a codegen-emitted LLVM-IR prologue that reads
//     `RuntimeContext::checkpoint_resume_point_ptr` / `checkpoint_yield_signal_ptr` and
//     early-returns when the checkpoint should be skipped.
//   * This yield-check kernel is inserted inline in the same flat HIP graph after each
//     yielding checkpoint's last body kernel; it self-gates with the same predicate.
//
// Semantics on launch (when not skipped):
//   1. Read `*yield_on` (the user-side ndarray pointed to via the indirection slot).
//   2. If non-zero, the checkpoint body asked the host to handle something this iteration:
//        a. atomicCAS `yield_signal` from -1 to this cp_id -- the first checkpoint in
//           declaration order to yield wins (matches CUDA semantics).
//        b. Bump `*resume_point` to INT_MAX so every subsequent checkpoint's prologue sees
//           `*resume_point > cp_id` and skips its body.
//   3. If zero, no-op.
//
// The framework does NOT reset `*yield_on` here. The user owns that buffer (their ndarray) and
// is responsible for clearing it before the resume launch; the canonical host loop in
// `docs/source/user_guide/graph.md` shows the explicit clear-before-resume.
//
// The indirection on `yield_on` (a pointer to a pointer) mirrors the CUDA version's
// `flag_slot`: the slot's device address is baked into the graph at build time, but the
// pointer it contains is host-updated each launch via memcpy, so a user can pass a different
// ndarray each call.
//
// After editing, regenerate the pre-built HSACO:
//
//   python scripts/build_checkpoint_yield_check_hsaco.py
//
// We do NOT pull in `<climits>` here: `INT_MAX` is open-coded as the literal `2147483647`.
// `<climits>` triggers a `__hip_atomic_compare_exchange_strong` requirement on RDNA3 in some
// HIP versions; the literal sidesteps that.

#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void _qd_checkpoint_yield_check(int32_t **yield_on_ptr_slot,
                                                      int32_t cp_id,
                                                      int32_t *yield_signal,
                                                      int32_t *resume_point) {
  // Self-gate: identical predicate to the codegen-emitted body-kernel prologue. On AMDGPU's flat-graph path it skips
  // the yield-read when this checkpoint should not have run. Pre- Hopper CUDA uses the exact same pattern.
  if (*resume_point > cp_id) {
    return;
  }
  if (*yield_signal != -1) {
    return;
  }
  int32_t *yield_on = *yield_on_ptr_slot;
  if (*yield_on != 0) {
    atomicCAS(yield_signal, -1, cp_id);
    *resume_point = 2147483647;  // INT_MAX
  }
}
