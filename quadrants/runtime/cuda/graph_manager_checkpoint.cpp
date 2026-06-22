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

#include <cstdlib>
#include <unordered_map>

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

CheckpointBuildPlan GraphManager::compute_checkpoint_plan_for_build(const std::vector<OffloadedTask> &offloaded_tasks,
                                                                    const LaunchContextBuilder &ctx,
                                                                    bool use_graph_do_while) {
  CheckpointBuildPlan plan;

  // Scan for qd.checkpoint() metadata once: any task with `checkpoint_id >= 0` opts the kernel into the IF-conditional
  // path. We need this before constructing CachedGraph so the latter can allocate the `resume_point` and `yield_signal`
  // scalars exactly when (and only when) they will be referenced.
  {
    int prev_cp = -1;
    for (const auto &task : offloaded_tasks) {
      if (task.checkpoint_id >= 0) {
        plan.has_checkpoints = true;
        if (task.checkpoint_id != prev_cp) {
          ++plan.num_distinct_checkpoints;
          prev_cp = task.checkpoint_id;
        }
        if (task.checkpoint_id > plan.max_cp_id) {
          plan.max_cp_id = task.checkpoint_id;
        }
      } else {
        prev_cp = -1;
      }
    }
  }

  // Determine which checkpoints actually have a resolved `yield_on=` ndarray this launch. Only those need a per-cp
  // persistent slot + a yield-check kernel inside the IF body. If every cp has a null dev_ptr, treat this graph as
  // yield-free even if the kernel declared yield_on= for some cp (covers the case where the user passes the same
  // kernel without a ready ndarray -- though `set_args` should reject that earlier; this is defence in depth).
  for (void *p : ctx.checkpoint_yield_on_dev_ptrs) {
    if (p) {
      plan.has_yield = true;
      break;
    }
  }

  // Unsupported combined case: a `qd.checkpoint()` block whose body contains a nested `qd.graph_do_while` (one cp_id
  // spanning more than one loop level). build_level's per-level IF grouping assumes a checkpoint's tasks are flat
  // within a single level, so fall back to the non-graph launch path (correct results, just no on-device gating)
  // rather than build a wrong graph.
  if (plan.has_checkpoints && use_graph_do_while) {
    std::unordered_map<int, int> cp_first_level;
    for (const auto &task : offloaded_tasks) {
      if (task.checkpoint_id >= 0) {
        auto [iter, inserted] = cp_first_level.emplace(task.checkpoint_id, task.graph_do_while_level_id);
        if (!inserted && iter->second != task.graph_do_while_level_id) {
          QD_INFO(
              "graph=True: a qd.checkpoint() block containing a nested qd.graph_do_while is not yet "
              "supported on the CUDA graph path; falling back to the non-graph launch.");
          plan.reject_graph_build = true;
          return plan;
        }
      }
    }
  }

  // On SM 9.0+ the checkpoint gating uses CUDA 12.4+ conditional-graph-node IF bodies driven by the gate kernel; on
  // pre-Hopper (SM < 90) the gate kernel is not available (conditional graph nodes need SM 9.0+) and we instead build
  // a flat graph where every body kernel still launches as a node but reads the codegen-emitted prologue and
  // self-early-returns when its cp_id is skipped or another checkpoint has yielded. Both paths still need the
  // yield-check kernel when `has_yield` is true -- the yield-check kernel uses only atomicCAS and direct pointer
  // writes (no device-runtime calls), so its fatbin targets every SM we support.
  if (plan.has_checkpoints) {
    ensure_checkpoint_gate_kernel_loaded();
    if (!gate_kernel_func_) {
      // Pre-Hopper CUDA: gate kernel intentionally not loaded. Switch to the flat-graph path (codegen prologue gates
      // each body kernel from inside). This path requires the yield-check kernel too when `has_yield` is set, and
      // that fatbin does cover pre-Hopper.
      plan.use_pre_hopper_flat_graph = true;
    } else if (std::getenv("QD_CUDA_FORCE_FLAT_CHECKPOINT_GRAPH") != nullptr) {
      // Debug / test knob: force the pre-Hopper flat-graph path even on SM 9.0+. Lets the checkpoint-prologue +
      // flat-graph path be exercised on Hopper+ hardware (the only kind most CI runners and dev boxes have) without
      // waiting for a pre-Hopper machine. The gate kernel is loaded but ignored; build_level's flat-graph branch
      // chains every body kernel into the level's graph and relies on the codegen-emitted prologue for gating,
      // identical to the pre-Hopper code path.
      plan.use_pre_hopper_flat_graph = true;
      QD_TRACE("QD_CUDA_FORCE_FLAT_CHECKPOINT_GRAPH=1 set; using pre-Hopper flat-graph path on SM 9.0+");
    }
    if (plan.has_yield) {
      ensure_checkpoint_yield_check_kernel_loaded();
      if (!yield_check_kernel_func_) {
        plan.reject_graph_build = true;
        return plan;
      }
    }
  }

  return plan;
}

void GraphManager::allocate_checkpoint_yield_on_slots(CachedGraph &cached,
                                                      const LaunchContextBuilder &ctx,
                                                      const CheckpointBuildPlan &plan) {
  // Pre-size the per-checkpoint ptr-slot table to (max_cp_id + 1) so we can index by cp_id directly. Slots for
  // checkpoints without `yield_on=` stay `nullptr` and are skipped at launch time. Only allocate when this graph
  // actually has yield-bearing checkpoints -- otherwise the table stays empty and `launch_cached_graph`'s memcpy loop
  // iterates zero times.
  if (!(plan.has_yield && plan.max_cp_id >= 0)) {
    return;
  }
  cached.checkpoint_yield_on_ptr_slots.assign((std::size_t)plan.max_cp_id + 1, nullptr);
  for (std::size_t cp = 0; cp < ctx.checkpoint_yield_on_dev_ptrs.size() && (int)cp <= plan.max_cp_id; ++cp) {
    if (ctx.checkpoint_yield_on_dev_ptrs[cp]) {
      void *slot = nullptr;
      CUDADriver::get_instance().malloc(&slot, sizeof(void *));
      // Initial pointer write -- subsequent launches re-memcpy via launch_cached_graph.
      void *user_ptr = ctx.checkpoint_yield_on_dev_ptrs[cp];
      CUDADriver::get_instance().memcpy_host_to_device(slot, &user_ptr, sizeof(void *));
      cached.checkpoint_yield_on_ptr_slots[cp] = slot;
    }
  }
}

}  // namespace cuda
}  // namespace quadrants::lang
