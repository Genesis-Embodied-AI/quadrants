// Lazy loaders for the three CUDA fatbin-backed kernels that implement `qd.checkpoint(...)` on the CUDA-graph path.
// Extracted out of `graph_manager.cpp` so the main `GraphManager::try_launch` / `launch_cached_graph` orchestration
// stays focused on the standard CUDA-graph build flow; every function in this file is checkpoint-specific and only
// matters when a kernel has at least one `qd.checkpoint(...)` block.
//
// Three loaders, all lazy, all idempotent across launches:
//
//   ensure_cond_with_yield_kernel_loaded:
//     `_qd_graph_do_while_cond_with_yield` -- the cond-with-yield variant of the WHILE condition kernel. Required only
//     when the kernel has BOTH `qd.graph_do_while` and at least one `qd.checkpoint(yield_on=...)` inside the WHILE
//     body. Lives in the same fatbin as the regular cond kernel (loaded by `ensure_condition_kernel_loaded`), so this
//     loader just resolves the function symbol from the already-loaded module; it silently no-ops when the module isn't
//     loaded yet so callers can invoke it unconditionally.
//
//   ensure_checkpoint_gate_kernel_loaded:
//     `_qd_checkpoint_if_gate` -- the gate kernel that writes `(cp_id >= *resume_point)` into a conditional handle
//     before each checkpoint's IF body. SM 9.0+ only (CUDA 12.4+ conditional graph nodes are required); on pre-Hopper
//     the gate is intentionally not loaded and `try_launch` falls back to the flat-graph path where the codegen-emitted
//     body-kernel prologue (`codegen_cuda.cpp::emit_checkpoint_gate_prologue`) does the gating from inside each body
//     kernel.
//
//   ensure_checkpoint_yield_check_kernel_loaded:
//     `_qd_checkpoint_yield_check` -- the yield-check kernel that atomic-CASes a cp_id into `yield_signal` and bumps
//     `resume_point` to INT_MAX. Needed on BOTH SM 9.0+ (inside the IF body) AND pre-Hopper (in the flat-graph chain),
//     so the fatbin covers the broader SM range than the gate kernel's fatbin and no SM gate is necessary here.

#include "quadrants/runtime/cuda/graph_manager.h"

#include "quadrants/common/logging.h"
#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"

// Pre-built fatbins -- see `scripts/build_checkpoint_gate_fatbin.py` and
// `scripts/build_checkpoint_yield_check_fatbin.py` for the regeneration commands. Both are bundled to cover sm_90 /
// sm_100 / sm_120 (the SM 9.0+ tiers that support conditional-graph nodes).
#include "quadrants/runtime/cuda/checkpoint_gate_fatbin.h"
#include "quadrants/runtime/cuda/checkpoint_yield_check_fatbin.h"

namespace quadrants::lang {
namespace cuda {

// Lazy load of the cond-with-yield kernel from the same fatbin as the regular cond kernel. Returns silently when the
// base module isn't loaded yet -- callers are expected to call `ensure_condition_kernel_loaded()` immediately before,
// mirroring the existing pattern.
void GraphManager::ensure_cond_with_yield_kernel_loaded() {
  if (cond_with_yield_kernel_func_)
    return;
  if (!cond_kernel_module_)
    return;
  CUDADriver::get_instance().module_get_function(&cond_with_yield_kernel_func_, cond_kernel_module_,
                                                 "_qd_graph_do_while_cond_with_yield");
  QD_TRACE("Loaded graph_do_while cond-with-yield kernel from pre-built fatbin");
}

// Loads the qd.checkpoint() IF-gate kernel from the pre-built fatbin (same shape and SM-coverage policy as
// `ensure_condition_kernel_loaded`). The CUDA 12.4+ IF conditional node mechanism is gated on SM 9.0+; on older devices
// we early-return and the caller falls back to flattening checkpoints into unconditional top-level kernels (slice 1c
// keeps the same behaviour-equivalent fallback as today's `graph_do_while` plus a clear log so users know why they
// aren't seeing the IF path).
void GraphManager::ensure_checkpoint_gate_kernel_loaded() {
  if (gate_kernel_func_)
    return;

  int cc = CUDAContext::get_instance().get_compute_capability();
  if (cc < 90) {
    // Pre-Hopper CUDA has no conditional-graph-node API (12.4+ only), so the IF-gate kernel is never used there. The
    // pre-Hopper path instead relies on the codegen-emitted body-kernel prologue (see
    // `codegen_cuda.cpp::emit_checkpoint_gate_prologue`) for GPU-side gating: every body kernel still launches as a
    // graph node, but reads `RuntimeContext::checkpoint_*_ptr` and self-early-returns when its cp_id is below
    // `*resume_point` or `*yield_signal != -1`. We early-return silently here; the gate-kernel `func` stays nullptr and
    // the caller's pre-Hopper graph-build path branches on that.
    return;
  }

  auto &driver = CUDADriver::get_instance();

  static_assert(kCheckpointGateKernelFatbinSize > 0,
                "Checkpoint gate kernel fatbin is empty -- regenerate with "
                "scripts/build_checkpoint_gate_fatbin.py");

  uint32_t ret = driver.module_load_data.call(&gate_kernel_module_, kCheckpointGateKernelFatbin);
  QD_ERROR_IF(ret != CUDA_SUCCESS,
              "Failed to load qd.checkpoint gate kernel fatbin (CUDA error {}). This SM ({}) "
              "may not be included in the fatbin -- regenerate with "
              "scripts/build_checkpoint_gate_fatbin.py",
              ret, cc);

  driver.module_get_function(&gate_kernel_func_, gate_kernel_module_, "_qd_checkpoint_if_gate");
  QD_TRACE("Loaded qd.checkpoint IF-gate kernel from pre-built fatbin");
}

// Loads the yield-check kernel from its pre-built fatbin. Mirrors the gate-kernel loader: same SM 9.0+ requirement (it
// lives inside the IF body, which already needs conditional nodes), same lazy-load pattern, separate fatbin so kernels
// without `yield_on=` don't pay for it.
void GraphManager::ensure_checkpoint_yield_check_kernel_loaded() {
  if (yield_check_kernel_func_)
    return;

  // The yield-check kernel is needed on BOTH SM 9.0+ (where it lives inside an IF conditional body) AND pre-Hopper
  // (where it sits inline after each yielding checkpoint's last body kernel in the flat graph). The pre-built fatbin
  // built by `scripts/build_checkpoint_yield_check_fatbin.py` targets a broad SM range; check at load time which SMs it
  // actually covers.

  auto &driver = CUDADriver::get_instance();
  int cc = CUDAContext::get_instance().get_compute_capability();

  static_assert(kCheckpointYieldCheckKernelFatbinSize > 0,
                "Checkpoint yield-check kernel fatbin is empty -- regenerate with "
                "scripts/build_checkpoint_yield_check_fatbin.py");

  uint32_t ret = driver.module_load_data.call(&yield_check_kernel_module_, kCheckpointYieldCheckKernelFatbin);
  QD_ERROR_IF(ret != CUDA_SUCCESS,
              "Failed to load qd.checkpoint yield-check kernel fatbin (CUDA error {}). This SM "
              "({}) may not be included in the fatbin -- regenerate with "
              "scripts/build_checkpoint_yield_check_fatbin.py",
              ret, cc);

  driver.module_get_function(&yield_check_kernel_func_, yield_check_kernel_module_, "_qd_checkpoint_yield_check");
  QD_TRACE("Loaded qd.checkpoint yield-check kernel from pre-built fatbin");
}

}  // namespace cuda
}  // namespace quadrants::lang
