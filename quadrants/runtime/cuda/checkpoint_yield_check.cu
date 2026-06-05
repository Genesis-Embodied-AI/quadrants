// Yield-check kernel for `qd.checkpoint(yield_on=...)` blocks (CUDA 12.4+ native path).
//
// Inserted by the GraphManager at the end of each `with qd.checkpoint(yield_on=foo):` body
// (inside the same IF conditional subgraph as the body kernels, so it only runs when the
// checkpoint actually executes).
//
// Semantics on launch:
//   1. Read `*yield_on` (the user-side ndarray pointed to via the indirection slot).
//   2. If non-zero, the checkpoint body asked the host to handle something this iteration:
//        a. atomicCAS `yield_signal` from -1 to this cp_id -- the first checkpoint in
//           declaration order to yield wins (slice 1d "atomic_cas race" requirement).
//        b. Bump `*resume_point` to INT_MAX so every subsequent checkpoint's gate kernel sees
//           `cp_id >= INT_MAX == false` and skips its body.
//        c. Reset `*yield_on` to 0 so the host doesn't have to clear it before re-launching
//           (matches the convenience semantics laid out in `perso_hugh/doc/qipc/reentrant.md`
//           section 5.2).
//   3. If zero, no-op.
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
  int32_t *yield_on = *yield_on_ptr_slot;
  if (*yield_on != 0) {
    atomicCAS(yield_signal, -1, cp_id);
    *resume_point = INT_MAX;
    *yield_on = 0;
  }
}
